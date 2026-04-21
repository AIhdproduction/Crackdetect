"""
CrackDetect - Automatische Riss-Erkennung in Bildern und Orthofotos
====================================================================
Powered by U-Net (trainiertes Modell -> Pixel-Masken)

Pipeline:
  1. Bild laden (JPG/PNG/TIFF - mit rasterio falls GeoTIFF)
  2. Automatisches Tiling fuer grosse Bilder (>= Kachelgroesse)
  3. U-Net erkennt Risse (direkte Pixel-Segmentierung)
  4. Morphologisches Closing verbindet unterbrochene Risse
  5. Masken werden zu Shapely-Polygonen vektorisiert
  6. NMS entfernt Duplikate aus ueberlappenden Kacheln
  7. Kontur-Extraktion: Umrandung jeder Riss-Region
  8. Distance Transform: Rissbreite (avg + max) pro Risspixel
  9. Pixel-Koordinaten -> Weltkoordinaten (falls GeoTIFF mit CRS)
 10. Export als GeoJSON und DXF
"""

import os
import sys
import json
import time
import datetime as _dt
import traceback
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import queue
import threading
from tkinter import filedialog

import numpy as np
import cv2
from PIL import Image
import torch
import customtkinter as ctk

# --- Optionale Geo-Imports ---
try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio nicht gefunden - GeoTIFF-Koordinaten nicht verfuegbar")

try:
    from shapely.geometry import Polygon, LineString, mapping
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("[WARN] shapely nicht gefunden - nur Bild-Ausgabe moeglich")

try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False
    print("[WARN] ezdxf nicht gefunden - DXF-Export deaktiviert")

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("[WARN] onnxruntime nicht gefunden - U-Net Inferenz nicht verfuegbar")

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False




# ==============================================================================
#  KONFIGURATION
# ==============================================================================

BASE_DIR = Path(__file__).parent

TILE_SIZE        = 512    # Muss mit Modell-Eingang uebereinstimmen (training: image_size=512)
TILE_OVERLAP_PCT = 0.30   # Ueberlappung zwischen Kacheln (30%)


DEFAULT_CONFIDENCE = 0.70  # U-Net Schwellwert (0.01 - 0.99)
DEFAULT_MIN_AREA   = 200   # Minimale Rissflaeche in px^2

MAX_POLY_AREA    = 500_000  # Maximale Polygon-Flaeche px^2

# Morphologisches Closing (verbindet unterbrochene Risse)
CLOSING_KERNEL_SIZE = 5

# Contour simplification factor (adjustable via settings dialog)
# Lower = more detailed contours, higher = smoother/simpler
_CONTOUR_EPSILON = 0.001

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

WORLD_FILE_EXT = {
    ".jpg":  [".jgw", ".jpw"],
    ".jpeg": [".jgw", ".jpw"],
    ".png":  [".pgw"],
    ".tif":  [".tfw"],
    ".tiff": [".tfw"],
    ".bmp":  [".bpw"],
    ".webp": [".wew"],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_unet_session = None   # onnxruntime InferenceSession fuer das trainierte U-Net

# Pfad zum trainierten U-Net ONNX (liegt im model/ Ordner neben crackdetect.py)
UNET_ONNX_PATH = BASE_DIR / "model" / "crack_unet.onnx"


# ==============================================================================
#  MODELL LADEN
# ==============================================================================

def load_models() -> None:
    """Laedt das U-Net ONNX Modell. Wird beim App-Start aufgerufen."""
    print(f"[INFO] Verwende Device: {DEVICE.upper()}")
    ok = load_unet()
    if not ok:
        raise RuntimeError(
            "Kein Modell gefunden. Lege crack_unet.onnx in den model/ Ordner."
        )


def load_unet() -> bool:
    """
    Laedt das trainierte U-Net als ONNX-Session.
    Gibt True zurueck wenn erfolgreich, False wenn kein Modell vorhanden.
    """
    global _unet_session

    if not HAS_ONNX:
        print("[WARN] onnxruntime fehlt - U-Net nicht verfuegbar (pip install onnxruntime)")
        return False

    onnx_path = UNET_ONNX_PATH
    if not onnx_path.exists():
        print(f"[INFO] Kein U-Net ONNX gefunden ({onnx_path})")
        print("       Lege dein trainiertes Modell hier ab: model/crack_unet.onnx")
        return False

    print(f"[INFO] Lade U-Net ONNX: {onnx_path}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _unet_session = ort.InferenceSession(str(onnx_path), providers=providers)
    used = _unet_session.get_providers()[0]
    print(f"[OK]   U-Net bereit ({used}).")
    return True



# ==============================================================================
#  GEO-BILD-WRAPPER
# ==============================================================================

class GeoImage:
    """
    Bild mit optionalen Geo-Metadaten aus GeoTIFF.
    Falls kein GeoTIFF: Koordinaten bleiben in Pixel.
    """

    def __init__(
        self,
        image: np.ndarray,
        transform=None,
        crs=None,
        source_path: str = "",
    ):
        self.image       = image
        self.transform   = transform
        self.crs         = crs
        self.source_path = source_path
        self.has_geo     = (transform is not None and crs is not None)

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    def pixel_size_m(self) -> Optional[float]:
        """Returns pixel size in meters if geo-referenced, else None."""
        if not self.has_geo or self.transform is None:
            return None
        a = self.transform.a  # pixel width (x)
        e = self.transform.e  # pixel height (y), negative
        return abs(a + e) / 2.0

    def px_to_world(self, px: float, py: float) -> Tuple[float, float]:
        if not self.has_geo:
            return px, py
        wx, wy = self.transform * (px, py)
        return wx, wy

    def coords_to_world(self, pixel_coords: List[Tuple]) -> List[Tuple]:
        if not self.has_geo:
            return pixel_coords
        return [self.px_to_world(x, y) for x, y in pixel_coords]


# --- World-File / PRJ Hilfsfunktionen ---

def _find_world_file(image_path: Path) -> Optional[Path]:
    ext = image_path.suffix.lower()
    candidates = WORLD_FILE_EXT.get(ext, [])
    for wext in candidates:
        wf = image_path.with_suffix(wext)
        if wf.exists():
            return wf
    return None


def _parse_world_file(wf_path: Path):
    if not HAS_RASTERIO:
        return None
    try:
        lines = wf_path.read_text().strip().splitlines()
        if len(lines) < 6:
            return None
        vals = [float(line.strip()) for line in lines[:6]]
        return Affine(vals[0], vals[2], vals[4], vals[1], vals[3], vals[5])
    except Exception as e:
        print(f"[WARN] World-File Lesefehler ({wf_path.name}): {e}")
        return None


def _read_prj_crs(image_path: Path):
    prj_path = image_path.with_suffix(".prj")
    if not prj_path.exists() or not HAS_RASTERIO:
        return None
    try:
        wkt = prj_path.read_text().strip()
        from rasterio.crs import CRS
        return CRS.from_wkt(wkt)
    except Exception as e:
        print(f"[WARN] PRJ-Datei Lesefehler ({prj_path.name}): {e}")
        return None


def load_geo_image(path: str) -> GeoImage:
    """
    Laedt Bild mit optionalen Geo-Metadaten.

    Prioritaet:
      1. Eingebettetes GeoTIFF (CRS + Transform im TIFF-Header)
      2. World-File (.jgw/.tfw/etc.) + .prj (via rasterio/GDAL)
      3. PIL (Pixel-Koordinaten)
    """
    p = Path(path)

    if HAS_RASTERIO:
        try:
            with rasterio.open(str(p)) as src:
                bands = min(src.count, 3)
                channels = [src.read(i + 1) for i in range(bands)]
                if bands == 1:
                    channels = channels * 3
                image = np.dstack(channels)

                if image.dtype != np.uint8:
                    lo, hi = image.min(), image.max()
                    image = ((image.astype(np.float32) - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

                transform = src.transform
                crs = src.crs

                has_embedded_geo = (crs is not None and transform is not None)

                if has_embedded_geo:
                    crs_label = crs.to_epsg() or crs.to_string()
                    print(f"[INFO] Georeferenziert: {src.width}x{src.height} px | CRS: {crs_label}")
                    return GeoImage(image, transform, crs, str(p))

                wf = _find_world_file(p)
                if wf:
                    wf_transform = _parse_world_file(wf)
                    wf_crs = _read_prj_crs(p)
                    if wf_transform:
                        crs_str = ""
                        if wf_crs:
                            crs_str = f" | CRS: {wf_crs.to_epsg() or wf_crs.to_string()}"
                        print(f"[INFO] World-File: {src.width}x{src.height} px{crs_str}")
                        return GeoImage(image, wf_transform, wf_crs, str(p))

                print(f"[INFO] Bild geladen (ohne Geo): {src.width}x{src.height} px")
                return GeoImage(image, source_path=str(p))

        except Exception as e:
            print(f"[WARN] rasterio-Fehler, Fallback auf PIL: {e}")

    pil = Image.open(str(p)).convert("RGB")
    image = np.array(pil)

    wf = _find_world_file(p)
    if wf:
        wf_transform = _parse_world_file(wf)
        wf_crs = _read_prj_crs(p)
        if wf_transform:
            crs_str = ""
            if wf_crs:
                crs_str = f" | CRS: {wf_crs.to_epsg() or wf_crs.to_string()}"
            print(f"[INFO] PIL + World-File: {pil.width}x{pil.height} px{crs_str}")
            return GeoImage(image, wf_transform, wf_crs, str(p))

    return GeoImage(image, source_path=str(p))


# ==============================================================================
#  TILING
# ==============================================================================

def compute_tiles(
    w: int,
    h: int,
    tile_size: int = TILE_SIZE,
    overlap_pct: float = TILE_OVERLAP_PCT,
) -> List[Tuple[int, int, int, int]]:
    """
    Gibt Liste von (x0, y0, x1, y1) Kacheln zurueck.

    Der Overlap ist adaptiv: als Minimum gilt overlap_pct (Standard 30%),
    aber die Anzahl Kacheln N wird so gewaehlt, dass alle Tiles gleichmaessig
    verteilt sind und exakt tile_size gross bleiben. Kein Rand-Sliver.

    Algorithmus:
      1. Berechne minimale Kachelanzahl N fuer jede Achse (Overlap >= min 30%)
      2. Berechne den gleichmaessigen Stride fuer N Tiles
      3. Platziere Tiles mit exaktem Stride; letztes Tile wird rechtsbuendig gesetzt
    """
    def _axis_starts(dim: int) -> List[int]:
        if dim <= tile_size:
            return [0]
        min_overlap = int(tile_size * overlap_pct)
        max_stride  = tile_size - min_overlap
        # Mindestanzahl Kacheln damit alles abgedeckt ist
        n_tiles = int(np.ceil((dim - tile_size) / max_stride)) + 1
        if n_tiles < 2:
            return [0]
        # Gleichmaessiger Stride fuer genau n_tiles Kacheln
        stride = (dim - tile_size) / (n_tiles - 1)
        starts = [int(round(i * stride)) for i in range(n_tiles - 1)]
        # Letztes Tile immer rechtsbuendig, damit kein Pixel fehlt
        starts.append(dim - tile_size)
        return starts

    xs = _axis_starts(w)
    ys = _axis_starts(h)

    tiles: List[Tuple[int, int, int, int]] = []
    seen: set = set()
    for y0 in ys:
        y1 = min(h, y0 + tile_size)
        for x0 in xs:
            x1 = min(w, x0 + tile_size)
            entry = (x0, y0, x1, y1)
            if entry not in seen:
                seen.add(entry)
                tiles.append(entry)

    return tiles



def compute_refine_tiles(
    polygons: List,
    img_w: int,
    img_h: int,
    tile_size: int,
    pad_factor: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """
    Berechnet zentrierte Refine-Tiles fuer erkannte Riss-Polygone (Pass 2).

    Fuer jedes Polygon wird die Bounding-Box berechnet und ein Tile mittig
    darueber gelegt. Falls die BB groesser als tile_size ist, werden mehrere
    ueberlappende Tiles entlang der BB erzeugt (30% Ueberlappung).
    Duplikate werden dedupliziert.

    Args:
        polygons:   Liste der Shapely-Polygone aus Pass 1.
        img_w:      Bildbreite in Pixeln.
        img_h:      Bildhöhe in Pixeln.
        tile_size:  Tile-Groesse in Pixeln.
        pad_factor: Padding = pad_factor * tile_size (wird um BB herum addiert).

    Returns:
        Sortierte, deduplizierte Liste von (x0, y0, x1, y1) Refine-Tiles.
    """
    if not polygons:
        return []

    pad = int(tile_size * pad_factor)
    overlap = int(tile_size * TILE_OVERLAP_PCT)
    stride  = tile_size - overlap
    seen: set = set()
    result: List[Tuple[int, int, int, int]] = []

    for poly in polygons:
        try:
            minx, miny, maxx, maxy = poly.bounds
        except Exception:
            continue

        # Bounding-Box mit Padding erweitern
        bx0 = max(0, int(minx) - pad)
        by0 = max(0, int(miny) - pad)
        bx1 = min(img_w, int(maxx) + pad)
        by1 = min(img_h, int(maxy) + pad)

        bb_w = bx1 - bx0
        bb_h = by1 - by0

        # Bounding-Box passt in ein Tile: zentriertes Tile erzeugen
        if bb_w <= tile_size and bb_h <= tile_size:
            cx = (bx0 + bx1) // 2
            cy = (by0 + by1) // 2
            tx0 = max(0, cx - tile_size // 2)
            ty0 = max(0, cy - tile_size // 2)
            tx1 = tx0 + tile_size
            ty1 = ty0 + tile_size
            # Randkorrektur: Tile darf nicht ueber das Bild hinausgehen
            if tx1 > img_w:
                tx0 = max(0, img_w - tile_size)
                tx1 = img_w
            if ty1 > img_h:
                ty0 = max(0, img_h - tile_size)
                ty1 = img_h
            entry = (tx0, ty0, tx1, ty1)
            if entry not in seen:
                seen.add(entry)
                result.append(entry)

        else:
            # Grosser Riss: Tiles entlang der BB mit 30% Overlap
            sub_tiles = compute_tiles(bb_w, bb_h, tile_size=tile_size,
                                      overlap_pct=TILE_OVERLAP_PCT)
            for sx0, sy0, sx1, sy1 in sub_tiles:
                tx0 = min(img_w, bx0 + sx0)
                ty0 = min(img_h, by0 + sy0)
                tx1 = min(img_w, bx0 + sx1)
                ty1 = min(img_h, by0 + sy1)
                entry = (tx0, ty0, tx1, ty1)
                if entry not in seen:
                    seen.add(entry)
                    result.append(entry)

    return result



def _apply_closing(mask: np.ndarray, kernel_size: int = CLOSING_KERNEL_SIZE) -> np.ndarray:
    """Morphologisches Closing: verbindet unterbrochene Riss-Segmente."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)



# ==============================================================================

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def detect_unet_tile(
    tile_np: np.ndarray,
    threshold: float = 0.50,
    log_fn=None,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Erkennt Risse via trainiertes U-Net ONNX Modell.
    Eingabe: RGB-Bild als numpy array (H x W x 3).
    Ausgabe: (masks, scores) - analog zu detect_in_tile.
    """
    global _unet_session

    if _unet_session is None:
        if log_fn:
            log_fn("   [FEHLER] U-Net nicht geladen. Export zunaechst benoetigt.")
        return [], []

    h, w = tile_np.shape[:2]
    target_size = 512   # must match training config (config.yaml: image_size)

    # Preprocessing: resize + normalize (ImageNet stats)
    resized = cv2.resize(tile_np, (target_size, target_size),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    normalized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = normalized.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    try:
        input_name = _unet_session.get_inputs()[0].name
        logits = _unet_session.run(None, {input_name: tensor})[0]  # (1, 1, H, W)
    except Exception as exc:
        if log_fn:
            log_fn(f"   [FEHLER] U-Net Inferenz: {exc}")
        return [], []

    # Sigmoid + threshold -> binary mask
    prob = 1.0 / (1.0 + np.exp(-logits[0, 0].astype(np.float64)))

    # Upscale probability map with bilinear interpolation for smooth edges,
    # then threshold. Much smoother than upscaling a binary mask with NEAREST.
    prob_resized_full = cv2.resize(
        prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
    )
    binary_resized = (prob_resized_full > threshold).astype(bool)

    if not np.any(binary_resized):
        return [], []

    # Confidence = mittlere Wahrscheinlichkeit im Rissbereich
    prob_resized = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    crack_probs = prob_resized[binary_resized]
    score = float(np.mean(crack_probs))

    # Morphologisches Closing um unterbrochene Linien zu verbinden
    binary_resized = _apply_closing(binary_resized)

    return [binary_resized], [score]




# ==============================================================================
#  MASKE -> POLYGON
# ==============================================================================

def mask_to_polygon(
    mask: np.ndarray,
    offset_x: int = 0,
    offset_y: int = 0,
    min_area: int = 100,
) -> Optional[object]:
    """Konvertiert bool-Maske -> Shapely-Polygon mit Kachel-Offset."""
    if not HAS_SHAPELY:
        return None

    mask_u8 = (mask.astype(np.uint8)) * 255

    # Smooth the mask before contour extraction to remove jagged staircase edges
    mask_u8 = cv2.GaussianBlur(mask_u8, (5, 5), sigmaX=1.5)
    mask_u8 = (mask_u8 > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    # Gentle simplification to keep the contour smooth
    eps    = _CONTOUR_EPSILON * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True)

    if len(approx) < 3:
        return None

    coords = [(int(pt[0][0]) + offset_x, int(pt[0][1]) + offset_y) for pt in approx]

    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if (poly.is_valid and poly.area >= min_area) else None
    except Exception:
        return None


def nms_polygons(
    polygons: List,
    scores: List[float],
    iou_threshold: float = 0.5,
) -> Tuple[List, List[float]]:
    """IoU-basierte NMS mit STRtree-Spatial-Index."""
    if not polygons:
        return [], []

    from shapely.strtree import STRtree

    order = sorted(range(len(polygons)), key=lambda i: polygons[i].area, reverse=True)
    polys = [polygons[i] for i in order]
    scrs  = [scores[i]   for i in order]

    suppressed = [False] * len(polys)
    tree       = STRtree(polys)

    for i, poly in enumerate(polys):
        if suppressed[i]:
            continue
        candidates = tree.query(poly)
        for j in candidates:
            if j <= i or suppressed[j]:
                continue
            try:
                inter = poly.intersection(polys[j]).area
                if inter == 0:
                    continue
                union = poly.area + polys[j].area - inter
                if union > 0 and inter / union > iou_threshold:
                    suppressed[j] = True
            except Exception:
                pass

    keep_polys  = [p for p, s in zip(polys, suppressed) if not s]
    keep_scores = [sc for sc, s in zip(scrs, suppressed) if not s]
    return keep_polys, keep_scores


# ==============================================================================
#  SKELETT-HILFSFUNKTIONEN
# ==============================================================================

def _thin_mask(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a boolean mask. Returns uint8 1-pixel-wide skeleton."""
    if HAS_SKIMAGE:
        return _skimage_skeletonize(mask).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros_like(mask, dtype=np.uint8)
    m = mask.astype(np.uint8).copy()
    for _ in range(80):
        eroded = cv2.erode(m, kernel)
        temp   = cv2.dilate(eroded, kernel)
        temp   = cv2.subtract(m, temp)
        skel   = cv2.bitwise_or(skel, temp)
        m      = eroded
        if cv2.countNonZero(m) == 0:
            break
    return skel


def merge_overlapping_polygons(polygons: List) -> List:
    """Merge all overlapping/touching Shapely polygons into disjoint regions."""
    if not HAS_SHAPELY or not polygons:
        return polygons

    from shapely.ops import unary_union
    from shapely.geometry import MultiPolygon

    buffered = [p.buffer(5) for p in polygons]
    union    = unary_union(buffered)

    if union.is_empty:
        return []

    geoms  = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
    shrunk = [g.buffer(-5) for g in geoms]
    result = []
    for g in shrunk:
        if g is None or g.is_empty or g.area <= 0:
            continue
        if g.geom_type == "MultiPolygon":
            result.extend([p for p in g.geoms if not p.is_empty and p.area > 0])
        else:
            result.append(g)
    return result


def polygon_to_crack_line(poly, canvas_h: int, canvas_w: int,
                          min_length_px: int = 15) -> Optional[object]:
    """
    Convert a filled Shapely Polygon (crack region) into a skeleton LineString.

    1. Rasterise polygon into local binary mask
    2. Skeletonise -> 1-px-wide centre line
    3. Sort skeleton pixels along principal axis (PCA)
    4. Douglas-Peucker simplification -> compact LineString
    """
    if not HAS_SHAPELY:
        return None

    minx, miny, maxx, maxy = poly.bounds
    minx = max(0, int(minx) - 2)
    miny = max(0, int(miny) - 2)
    maxx = min(canvas_w, int(maxx) + 3)
    maxy = min(canvas_h, int(maxy) + 3)
    bw, bh = maxx - minx, maxy - miny
    if bw < 2 or bh < 2:
        return None

    mask = np.zeros((bh, bw), dtype=np.uint8)
    ext  = np.array(
        [(int(x) - minx, int(y) - miny) for x, y in poly.exterior.coords],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [ext], 1)

    skel = _thin_mask(mask.astype(bool))
    ys, xs = np.where(skel)
    if len(xs) < 3:
        return None

    pts = np.column_stack([xs + minx, ys + miny]).astype(float)

    center   = pts.mean(axis=0)
    centered = pts - center
    sample   = centered[:min(len(centered), 3000)]
    _, _, Vt = np.linalg.svd(sample, full_matrices=False)
    axis     = Vt[0]
    order    = np.argsort(centered @ axis)
    ordered  = pts[order]

    if len(ordered) > 1000:
        idx     = np.round(np.linspace(0, len(ordered) - 1, 1000)).astype(int)
        ordered = ordered[idx]

    cv_pts     = ordered.reshape(-1, 1, 2).astype(np.int32)
    simplified = cv2.approxPolyDP(cv_pts, 2.0, closed=False)
    if simplified is None or len(simplified) < 2:
        return None
    simplified = simplified.reshape(-1, 2)

    coords = [(int(p[0]), int(p[1])) for p in simplified]
    line   = LineString(coords)
    return line if line.length >= min_length_px else None


def compute_crack_width(poly, canvas_h: int, canvas_w: int) -> Tuple[float, float]:
    """
    Berechnet durchschnittliche und maximale Rissbreite via Distance Transform.

    Gibt (width_avg_px, width_max_px) zurueck.
    Die Breite entspricht dem 2-fachen der mittleren Distanz zum Maskenrand
    an jedem Skelett-Pixel.
    """
    minx, miny, maxx, maxy = poly.bounds
    minx = max(0, int(minx) - 2)
    miny = max(0, int(miny) - 2)
    maxx = min(canvas_w, int(maxx) + 3)
    maxy = min(canvas_h, int(maxy) + 3)
    bw, bh = maxx - minx, maxy - miny
    if bw < 2 or bh < 2:
        return 1.0, 1.0

    mask = np.zeros((bh, bw), dtype=np.uint8)
    ext  = np.array(
        [(int(x) - minx, int(y) - miny) for x, y in poly.exterior.coords],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [ext], 255)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    skel = _thin_mask(mask.astype(bool))
    skel_ys, skel_xs = np.where(skel)

    if len(skel_xs) == 0:
        return float(dist.max() * 2), float(dist.max() * 2)

    skel_dists = dist[skel_ys, skel_xs]
    width_avg  = float(np.mean(skel_dists) * 2)
    width_max  = float(np.max(skel_dists) * 2)
    return width_avg, width_max


# ==============================================================================
#  RISS-DATEN CONTAINER
# ==============================================================================

class CrackFeature:
    """
    Haelt alle Attribute eines erkannten Risses fuer GeoJSON-Export.
    """
    def __init__(
        self,
        geometry,         # Shapely LineString oder Polygon
        score: float,
        width_avg_px: float,
        width_max_px: float,
        source: str,
    ):
        self.geometry     = geometry
        self.score        = score
        self.width_avg_px = width_avg_px
        self.width_max_px = width_max_px
        self.source       = source
        self.detected_at  = _dt.datetime.now().isoformat(timespec="seconds")


# ==============================================================================
#  HAUPT-PIPELINE
# ==============================================================================

def _run_scale(
    img: np.ndarray,
    tile_size: int,
    confidence: float,
    min_area: int,
    log_fn=None,
    progress_cb=None,
    scale_label: str = "",
    pct_start: float = 0.0,
    pct_end: float = 0.8,
) -> Tuple[List, List[float]]:
    """Fuhrt einen vollstaendigen Tiling-Durchlauf via U-Net durch."""
    h, w = img.shape[:2]
    needs_tiling = max(w, h) >= tile_size
    all_polygons: List = []
    all_scores: List[float] = []

    if needs_tiling:
        tiles = compute_tiles(w, h, tile_size=tile_size)
        prefix = f"[{scale_label}] " if scale_label else ""
        if log_fn:
            log_fn(f"   {prefix}Tiling: {len(tiles)} Kacheln fuer {w}x{h} px")
        raw_total = 0
        for i, (x0, y0, x1, y1) in enumerate(tiles):
            pct = pct_start + (pct_end - pct_start) * i / len(tiles)
            if progress_cb:
                progress_cb(pct, f"{prefix}Kachel {i+1}/{len(tiles)}")
            tile = img[y0:y1, x0:x1]
            masks, scores = detect_unet_tile(tile, threshold=confidence, log_fn=log_fn)
            raw_total += len(masks)
            for mask, score in zip(masks, scores):
                poly = mask_to_polygon(mask, offset_x=x0, offset_y=y0, min_area=min_area)
                if poly is not None:
                    all_polygons.append(poly)
                    all_scores.append(score)
        if log_fn:
            if raw_total == 0:
                log_fn(f"   {prefix}0 Masken - Confidence ({confidence:.2f}) weiter senken")
            else:
                log_fn(f"   {prefix}{raw_total} Roh-Masken -> {len(all_polygons)} Polygone")
    else:
        if log_fn and scale_label:
            log_fn(f"   [{scale_label}] Direkt: {w}x{h} px")
        if progress_cb:
            progress_cb(pct_start + (pct_end - pct_start) * 0.2, "Erkenne Risse ...")
        masks, scores = detect_unet_tile(img, threshold=confidence, log_fn=log_fn)
        if log_fn and len(masks) == 0:
            log_fn(f"   0 Masken erkannt - Confidence ({confidence:.2f}) weiter senken")
        for mask, score in zip(masks, scores):
            poly = mask_to_polygon(mask, min_area=min_area)
            if poly is not None:
                all_polygons.append(poly)
                all_scores.append(score)
    return all_polygons, all_scores


def process_geo_image(
    geo_image: GeoImage,
    confidence: float    = 0.50,
    min_area: int        = 100,
    tile_size: int       = TILE_SIZE,
    multi_scale: bool    = False,
    progress_cb=None,
    log_fn=None,
) -> Tuple[np.ndarray, List[CrackFeature]]:
    """
    Verarbeitet ein GeoImage komplett via U-Net.

    Returns:
        annotated_np   - Bild mit farbigem Overlay (H x W x 3 uint8)
        crack_features - Liste von CrackFeature-Objekten
    """
    img  = geo_image.image
    h, w = img.shape[:2]
    pixel_size_m = geo_image.pixel_size_m()


    if log_fn:
        log_fn("   [U-Net] Trainiertes Modell aktiv (ONNX)")

    # ------------------------------------------------------------------
    # Pass 1: Globales Tiling mit 30% Ueberlappung
    # ------------------------------------------------------------------
    if log_fn:
        log_fn("   [Pass 1] Globales Tiling (30% Overlap) ...")
    polys1, scores1 = _run_scale(
        img, tile_size, confidence, min_area,
        log_fn=log_fn, progress_cb=progress_cb,
        scale_label="Pass 1", pct_start=0.05, pct_end=0.50,
    )

    # ------------------------------------------------------------------
    # Pass 2: Zentrierte Refine-Tiles ueber erkannte Risse
    # ------------------------------------------------------------------
    polys2:  List = []
    scores2: List[float] = []

    if HAS_SHAPELY and polys1:
        refine_tiles = compute_refine_tiles(
            polys1, img_w=w, img_h=h, tile_size=tile_size
        )
        if log_fn:
            log_fn(f"   [Pass 2] {len(refine_tiles)} Refine-Tiles ueber erkannte Risse ...")
        if progress_cb:
            progress_cb(0.52, f"Pass 2: {len(refine_tiles)} Refine-Tiles ...")

        for i, (x0, y0, x1, y1) in enumerate(refine_tiles):
            pct = 0.52 + 0.26 * i / max(1, len(refine_tiles))
            if progress_cb:
                progress_cb(pct, f"Refine {i+1}/{len(refine_tiles)}")
            tile = img[y0:y1, x0:x1]
            masks, scores = detect_unet_tile(tile, threshold=confidence, log_fn=log_fn)
            for mask, score in zip(masks, scores):
                poly = mask_to_polygon(mask, offset_x=x0, offset_y=y0, min_area=min_area)
                if poly is not None:
                    polys2.append(poly)
                    scores2.append(score)

        if log_fn:
            log_fn(f"   [Pass 2] {len(polys2)} zusaetzliche Polygone aus Refine-Tiles")
    elif log_fn:
        log_fn("   [Pass 2] Uebersprungen (keine Risse in Pass 1 gefunden)")

    # ------------------------------------------------------------------
    # Optional: Multi-Scale (halbe Kachelgroesse) - Legacy-Fallback
    # ------------------------------------------------------------------
    polys_ms:  List = []
    scores_ms: List[float] = []
    if multi_scale:
        small_tile = max(256, tile_size // 2)
        if log_fn:
            log_fn(f"   [Multi-Scale] Feiner Durchlauf mit Kachelgroesse {small_tile} px ...")
        polys_ms, scores_ms = _run_scale(
            img, small_tile, confidence, min_area,
            log_fn=log_fn, progress_cb=progress_cb,
            scale_label="Multi-Scale", pct_start=0.78, pct_end=0.80,
        )

    all_polygons = polys1 + polys2 + polys_ms
    all_scores   = scores1 + scores2 + scores_ms


    # --- NMS ---
    if HAS_SHAPELY and len(all_polygons) > 1:
        if progress_cb:
            progress_cb(0.81, f"NMS: {len(all_polygons)} Masken ...")
        before = len(all_polygons)
        all_polygons, all_scores = nms_polygons(all_polygons, all_scores, iou_threshold=0.3)
        removed = before - len(all_polygons)
        if removed > 0 and log_fn:
            log_fn(f"   NMS: {removed} Duplikate entfernt -> {len(all_polygons)} Risse")

    # --- Regionen zusammenfuehren ---
    if HAS_SHAPELY and len(all_polygons) > 1:
        if progress_cb:
            progress_cb(0.85, f"Regionen zusammenfuehren ...")
        before = len(all_polygons)
        all_polygons = merge_overlapping_polygons(all_polygons)
        all_scores   = [1.0] * len(all_polygons)
        if before != len(all_polygons) and log_fn:
            log_fn(f"   Merge: {before} -> {len(all_polygons)} Riss-Regionen")

    # --- Polygone -> Kontur-Linien (Umrandung) + Breite ---
    n_polys = len(all_polygons)
    if progress_cb:
        progress_cb(0.87, f"Konturen: {n_polys} Regionen ...")

    crack_features: List[CrackFeature] = []
    _report_every = max(1, n_polys // 10)

    for skel_i, (poly, score) in enumerate(zip(all_polygons, all_scores)):
        if hasattr(poly, 'area') and poly.area > MAX_POLY_AREA:
            if log_fn:
                log_fn(f"   [Skip] Polygon {skel_i+1} zu gross ({int(poly.area):,} px^2) - uebersprungen")
            continue

        # Use the polygon boundary (outline) as the exported geometry
        try:
            outline = LineString(poly.exterior.coords)
        except Exception:
            outline = poly

        # Rissbreite berechnen
        try:
            w_avg, w_max = compute_crack_width(poly, h, w)
        except Exception:
            w_avg, w_max = 1.0, 1.0

        crack_features.append(CrackFeature(
            geometry     = outline,
            score        = score,
            width_avg_px = w_avg,
            width_max_px = w_max,
            source       = Path(geo_image.source_path).name,
        ))

        if progress_cb and n_polys > 0 and skel_i % _report_every == 0:
            pct = 0.87 + 0.08 * skel_i / n_polys
            progress_cb(pct, f"Konturen: {skel_i+1}/{n_polys} ...")

    print(f"[INFO] {len(crack_features)} Riss-Konturen erstellt")

    # --- Render: Farbiges Overlay ---
    if progress_cb:
        progress_cb(0.96, "Bild mit Riss-Konturen rendern ...")

    MAX_PREVIEW = 2000
    if max(h, w) > MAX_PREVIEW:
        coord_scale = MAX_PREVIEW / max(h, w)
        render = cv2.resize(img, (int(w * coord_scale), int(h * coord_scale)),
                            interpolation=cv2.INTER_AREA)
    else:
        coord_scale = 1.0
        render = img.copy()

    rh, rw = render.shape[:2]

    LINE_COLOR  = (60, 120, 255)   # Hellblau (RGB)
    NUM_COLOR   = (255, 255, 255)  # Weiss
    LINE_THICK  = 1

    def _scaled_pts(coords):
        arr = np.array(coords, dtype=np.float32) * coord_scale
        return arr.astype(np.int32).reshape(-1, 1, 2)

    # Kontur-Linien zeichnen
    for idx, feat in enumerate(crack_features):
        obj = feat.geometry
        if obj is None:
            continue
        try:
            if hasattr(obj, "geoms"):
                for part in obj.geoms:
                    cv2.polylines(render, [_scaled_pts(list(part.coords))], False, LINE_COLOR, LINE_THICK)
            elif hasattr(obj, "coords"):    # LineString
                pts = _scaled_pts(list(obj.coords))
                cv2.polylines(render, [pts], False, LINE_COLOR, LINE_THICK)
                # Riss-Nummer an Mittelpunkt
                mid_idx = len(pts) // 2
                mx, my = int(pts[mid_idx][0][0]), int(pts[mid_idx][0][1])
                cv2.putText(render, str(idx + 1), (mx + 3, my - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, NUM_COLOR, 1, cv2.LINE_AA)
            elif hasattr(obj, "exterior"):  # Polygon-Fallback
                pts = _scaled_pts(list(obj.exterior.coords))
                cv2.polylines(render, [pts], True, LINE_COLOR, LINE_THICK)
                cx = int(np.mean([p[0][0] for p in pts]))
                cy = int(np.mean([p[0][1] for p in pts]))
                cv2.putText(render, str(idx + 1), (cx + 3, cy - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, NUM_COLOR, 1, cv2.LINE_AA)
        except Exception:
            pass

    # Legende einzeichnen
    legend_y = rh - 10
    cv2.rectangle(render, (8, legend_y - 20), (160, legend_y + 2), (20, 20, 20), -1)
    cv2.line(render, (12, legend_y - 8), (24, legend_y - 8), LINE_COLOR, 2)
    cv2.putText(render, "Risskontur", (28, legend_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)

    return render, crack_features


# ==============================================================================
#  EXPORT
# ==============================================================================

def export_geojson(
    entries: List[Tuple[GeoImage, List[CrackFeature]]],
    output_path: str,
) -> str:
    """
    Exportiert Crack-Linien/-Polygone als GeoJSON mit erweiterten Attributen:
    - id, source_file, type
    - length_px, length_m (falls GeoTIFF)
    - width_avg_px, width_max_px, width_avg_m, width_max_m
    - confidence, detection_date, has_geo, crs
    """
    if not HAS_SHAPELY:
        raise RuntimeError("shapely nicht installiert")

    features = []
    for geo_image, crack_features in entries:
        pixel_m = geo_image.pixel_size_m()

        for feat in crack_features:
            geom = feat.geometry
            if geom is None:
                continue

            # Koordinaten in Weltkoordinaten umrechnen
            if geo_image.has_geo:
                if hasattr(geom, "coords"):
                    wc = geo_image.coords_to_world(list(geom.coords))
                    export_geom = LineString(wc)
                elif hasattr(geom, "exterior"):
                    wc = geo_image.coords_to_world(list(geom.exterior.coords))
                    from shapely.geometry import Polygon as _Poly
                    export_geom = _Poly(wc)
                else:
                    export_geom = geom
            else:
                export_geom = geom

            length_px = float(geom.length) if hasattr(geom, "length") else 0.0

            props: Dict[str, Any] = {
                "id":           len(features) + 1,
                "source_file":  feat.source,
                "type":         "crack",
                "length_px":    round(length_px, 2),
                "width_avg_px": round(feat.width_avg_px, 2),
                "width_max_px": round(feat.width_max_px, 2),
                "confidence":   round(feat.score, 4),
                "detection_date": feat.detected_at,
                "has_geo":      geo_image.has_geo,
            }

            if pixel_m is not None and pixel_m > 0:
                props["length_m"]    = round(length_px * pixel_m, 4)
                props["width_avg_m"] = round(feat.width_avg_px * pixel_m, 4)
                props["width_max_m"] = round(feat.width_max_px * pixel_m, 4)

            if geo_image.has_geo and geo_image.crs:
                props["crs"] = geo_image.crs.to_string()

            features.append({
                "type":       "Feature",
                "geometry":   mapping(export_geom),
                "properties": props,
            })

    doc: Dict[str, Any] = {"type": "FeatureCollection", "features": features}

    for geo_image, _ in entries:
        if geo_image.has_geo and geo_image.crs:
            doc["crs"] = {"type": "name", "properties": {"name": geo_image.crs.to_string()}}
            break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    return output_path


def export_dxf(
    entries: List[Tuple[GeoImage, List[CrackFeature]]],
    output_path: str,
) -> str:
    """Exportiert Polygone/Linien als DXF R2010."""
    if not HAS_EZDXF:
        raise RuntimeError("ezdxf nicht installiert")
    if not HAS_SHAPELY:
        raise RuntimeError("shapely nicht installiert")

    doc = ezdxf.new(dxfversion="R2010")
    has_any_geo = any(gi.has_geo for gi, _ in entries)
    if has_any_geo:
        doc.header["$INSUNITS"] = 6

    doc.layers.add("CRACKS",       color=1)
    doc.layers.add("CRACK_LABELS", color=3)

    msp      = doc.modelspace()
    crack_id = 1

    for geo_image, crack_features in entries:
        has_geo = geo_image.has_geo

        for feat in crack_features:
            geom = feat.geometry
            if geom is None:
                continue

            if hasattr(geom, "coords"):
                raw_coords = list(geom.coords)
                is_closed  = False
                mid_px     = raw_coords[len(raw_coords) // 2]
            elif hasattr(geom, "exterior"):
                raw_coords = list(geom.exterior.coords)
                is_closed  = True
                mid_px     = (geom.centroid.x, geom.centroid.y)
            else:
                continue

            world = geo_image.coords_to_world(raw_coords) if has_geo else raw_coords
            pts_2d = [(float(x), float(y)) for x, y in world]

            msp.add_lwpolyline(
                pts_2d,
                dxfattribs={"layer": "CRACKS", "closed": is_closed, "color": 1},
            )

            cx_px, cy_px = float(mid_px[0]), float(mid_px[1])
            if has_geo:
                cx_px, cy_px = geo_image.px_to_world(cx_px, cy_px)

            label_height = 0.1 if has_geo else 20
            msp.add_text(
                f"Riss {crack_id}",
                dxfattribs={"layer": "CRACK_LABELS", "height": label_height,
                            "insert": (cx_px, cy_px), "color": 3},
            )
            crack_id += 1

    doc.saveas(output_path)
    return output_path





# ==============================================================================
#  APPEARANCE
# ==============================================================================

APP_COLORS = {
    "bg":        "#0d0d1a",
    "panel":     "#16213e",
    "accent":    "#0f3460",
    "highlight": "#e94560",
    "hover":     "#c73652",
    "text":      "#eaeaea",
    "text_dim":  "#a0a0b0",
    "success":   "#4ade80",
    "warning":   "#fbbf24",
    "error":     "#f87171",
}


# ==============================================================================
#  PROCESSING WORKER
# ==============================================================================

class Worker:
    """Runs crack detection in a background thread, sends messages via queue."""

    def __init__(self, msg_q: queue.Queue):
        self._q          = msg_q
        self._stop_event = threading.Event()

    def _log(self, text: str):
        self._q.put(("log", text))

    def _status(self, text: str):
        self._q.put(("status", text))

    def _progress(self, pct: float):
        self._q.put(("progress", max(0.0, min(1.0, pct))))

    def _preview(self, img_np: np.ndarray):
        self._q.put(("preview", img_np.copy()))

    def _done(self, success: bool = True):
        self._q.put(("done", success))

    def stop(self):
        self._stop_event.set()

    def run(
        self,
        image_paths: List[Path],
        confidence: float,
        min_area: int,
        tile_size: int,
        multi_scale: bool,
        do_geojson: bool = True,
        do_dxf: bool = False,
        do_mask: bool = False,
    ):
        self._stop_event.clear()

        input_dir  = image_paths[0].parent
        output_dir = input_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Output-Ordner: {output_dir}")

        total     = len(image_paths)
        all_entries: List[Tuple[GeoImage, List[CrackFeature]]] = []
        t_start      = time.time()
        geojson_ok   = False
        dxf_ok       = False
        last_geojson = ""
        last_dxf     = ""

        for idx, img_path in enumerate(image_paths):
            if self._stop_event.is_set():
                self._log("Abgebrochen.")
                self._done(False)
                return

            pct_base = idx / total
            self._status(f"Lade {img_path.name} ...")
            self._progress(pct_base * 0.05)
            self._log(f"\n>> {img_path.name}")

            try:
                geo_image = load_geo_image(str(img_path))
                geo_flag  = ""
                if geo_image.has_geo:
                    epsg     = geo_image.crs.to_epsg() if geo_image.crs else "?"
                    px_m     = geo_image.pixel_size_m()
                    px_str   = f" | {px_m*100:.1f} cm/px" if px_m else ""
                    geo_flag = f" | EPSG:{epsg}{px_str}"
                self._log(f"   {geo_image.width}x{geo_image.height} px{geo_flag}")

                _tile_start  = time.time()
                _crack_count = [0]

                def _tile_cb(tile_pct: float, tile_msg: str):
                    if self._stop_event.is_set():
                        return
                    elapsed = time.time() - _tile_start
                    if tile_pct > 0.02 and elapsed > 2:
                        eta_s = (elapsed / tile_pct) * (1.0 - tile_pct)
                        m, s  = divmod(int(eta_s), 60)
                        h_, m = divmod(m, 60)
                        if h_ > 0:   eta = f"~{h_}h {m:02d}m"
                        elif m > 0:  eta = f"~{m}m {s:02d}s"
                        else:        eta = f"~{s}s"
                    else:
                        eta = ""

                    overall = pct_base * 0.97 + tile_pct / total * 0.90
                    msg = f"{img_path.name}: {tile_msg} | {_crack_count[0]} Risse"
                    if eta:
                        msg += f" | {eta} verbleibend"
                    self._status(msg)
                    self._progress(overall)

                annotated, crack_features = process_geo_image(
                    geo_image,
                    confidence   = confidence,
                    min_area     = min_area,
                    tile_size    = tile_size,
                    multi_scale  = multi_scale,
                    progress_cb  = _tile_cb,
                    log_fn       = self._log,
                )

                _crack_count[0] = len(crack_features)
                elapsed_img     = time.time() - _tile_start

                self._progress((pct_base + 1.0 / total) * 0.97)
                self._status(f"{img_path.name}: {len(crack_features)} Risse in {elapsed_img:.1f}s")
                self._log(f"   {len(crack_features)} Risse erkannt in {elapsed_img:.1f}s")

                # Vorschau
                self._preview(annotated)

                # Bild speichern
                stem    = img_path.stem
                out_img = output_dir / f"{stem}_cracks.png"
                self._status(f"Speichere {out_img.name} ...")
                Image.fromarray(annotated).save(str(out_img))
                self._q.put(("last_output", str(out_img)))
                self._log(f"   Gespeichert: {out_img.name} ({annotated.shape[1]}x{annotated.shape[0]} px)")

                # --- Mask export (binary PNG, full resolution) ---
                if do_mask and crack_features:
                    img_h, img_w = geo_image.height, geo_image.width
                    mask_canvas = np.zeros((img_h, img_w), dtype=np.uint8)
                    for feat in crack_features:
                        geom = feat.geometry
                        if geom is None:
                            continue
                        try:
                            if hasattr(geom, "exterior"):
                                pts = np.array(list(geom.exterior.coords), dtype=np.int32)
                            elif hasattr(geom, "coords"):
                                pts = np.array(list(geom.coords), dtype=np.int32)
                            else:
                                continue
                            cv2.fillPoly(mask_canvas, [pts], 255)
                        except Exception:
                            pass
                    mask_path = output_dir / f"{stem}_mask.png"
                    cv2.imwrite(str(mask_path), mask_canvas)
                    self._log(f"   Maske:   {mask_path.name} ({img_w}x{img_h} px)")

                # --- GeoJSON Export (per image) ---
                geojson_path = str(output_dir / f"{stem}_cracks.geojson")
                if do_geojson and crack_features:
                    try:
                        if HAS_SHAPELY:
                            export_geojson([(geo_image, crack_features)], geojson_path)
                            self._log(f"   GeoJSON: {stem}_cracks.geojson ({len(crack_features)} Features)")
                            geojson_ok = True
                            last_geojson = geojson_path
                        else:
                            self._log("   GeoJSON uebersprungen (shapely fehlt)")
                    except Exception as exc:
                        self._log(f"   GeoJSON-Fehler: {exc}")

                # --- DXF Export (per image) ---
                dxf_path = str(output_dir / f"{stem}_cracks.dxf")
                if do_dxf and crack_features:
                    try:
                        if HAS_EZDXF and HAS_SHAPELY:
                            export_dxf([(geo_image, crack_features)], dxf_path)
                            self._log(f"   DXF:     {stem}_cracks.dxf")
                            dxf_ok = True
                            last_dxf = dxf_path
                        elif not HAS_EZDXF:
                            self._log("   DXF uebersprungen (ezdxf fehlt - pip install ezdxf)")
                    except Exception as exc:
                        self._log(f"   DXF-Fehler: {exc}")

                all_entries.append((geo_image, crack_features))

            except Exception as exc:
                self._log(f"   Fehler: {exc}")
                traceback.print_exc()

        total_cracks  = sum(len(g) for _, g in all_entries)
        total_elapsed = time.time() - t_start

        # Build summary line for enabled exports
        export_parts = []
        if do_geojson:
            export_parts.append(f"GeoJSON: {'OK' if geojson_ok else 'FEHLER'}")
        if do_dxf:
            export_parts.append(f"DXF: {'OK' if dxf_ok else 'FEHLER'}")
        if do_mask:
            export_parts.append("Maske: OK")
        export_summary = "  |  ".join(export_parts) if export_parts else "Keine Exporte"

        self._log(
            f"\n{'─' * 54}\n"
            f"  {total_cracks} Risse | {total} Bild(er) | {total_elapsed:.1f}s\n"
            f"  {export_summary}\n"
            f"  Output: {output_dir}\n"
            f"{'─' * 54}"
        )
        self._progress(1.0)
        self._status(f"Fertig - {total_cracks} Risse erkannt")
        # Exportpfade fuer UI-Buttons bekanntgeben
        self._q.put(("export_paths", {
            "geojson": last_geojson if geojson_ok else None,
            "dxf":     last_dxf     if dxf_ok     else None,
            "output_dir": str(output_dir),
        }))
        self._done(True)


# ==============================================================================
#  DESKTOP APP
# ==============================================================================

class ZoomWindow(ctk.CTkToplevel):
    """Scrollbares Vollbild-Zoom-Fenster fuer das Ergebnis-Bild."""

    def __init__(self, parent, img_np: np.ndarray):
        super().__init__(parent)
        self.title("Ergebnis - Zoom")
        self.geometry("1200x800")
        self.configure(fg_color=APP_COLORS["bg"])

        # Canvas mit Scrollbars
        self._canvas = ctk.CTkCanvas(self, bg=APP_COLORS["bg"],
                                     highlightthickness=0)
        sb_x = ctk.CTkScrollbar(self, orientation="horizontal",
                                 command=self._canvas.xview)
        sb_y = ctk.CTkScrollbar(self, orientation="vertical",
                                 command=self._canvas.yview)
        self._canvas.configure(xscrollcommand=sb_x.set,
                                yscrollcommand=sb_y.set)

        sb_x.pack(side="bottom", fill="x")
        sb_y.pack(side="right",  fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        pil_img = Image.fromarray(img_np)
        from PIL import ImageTk
        self._tk_img  = ImageTk.PhotoImage(pil_img)
        self._canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

        # Mouse-Wheel-Scrolling
        self._canvas.bind("<MouseWheel>",
                          lambda e: self._canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self._canvas.bind("<Shift-MouseWheel>",
                          lambda e: self._canvas.xview_scroll(-1 * (e.delta // 120), "units"))


class CrackDetectApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("CrackDetect - Automatische Riss-Erkennung")
        self.geometry("1180x970")
        self.minsize(900, 640)
        self.configure(fg_color=APP_COLORS["bg"])

        self._input_paths:     List[Path] = []
        self._worker:          Optional[Worker]           = None
        self._thread:          Optional[threading.Thread] = None
        self._msg_q:           queue.Queue                = queue.Queue()
        self._preview_image:   Optional[ctk.CTkImage]    = None
        self._last_output_img: Optional[str]              = None
        self._last_annotated:  Optional[np.ndarray]      = None
        self._zoom_win:        Optional[ZoomWindow]       = None
        self._last_export:     dict                       = {}

        # Settings (adjustable via settings dialog)
        self._export_geojson = ctk.BooleanVar(value=True)
        self._export_dxf     = ctk.BooleanVar(value=False)
        self._export_mask    = ctk.BooleanVar(value=False)
        self._epsilon_var    = ctk.DoubleVar(value=0.001)
        self._tile_var       = ctk.StringVar(value="512")
        self._multiscale_var = ctk.BooleanVar(value=False)

        self._build_ui()
        self._poll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- UI ---

    def _build_ui(self):
        # Header
        hdr = ctk.CTkFrame(self, fg_color=APP_COLORS["accent"], corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(
            hdr, text="CrackDetect",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=APP_COLORS["highlight"],
        ).pack(side="left", padx=22, pady=14)
        unet_available = UNET_ONNX_PATH.exists()
        model_hint = "U-Net" if unet_available else "Kein Modell"
        ctk.CTkLabel(
            hdr, text=f"Automatische Riss-Erkennung  |  {model_hint}  |  GeoJSON & DXF",
            font=ctk.CTkFont(size=12),
            text_color=APP_COLORS["text_dim"],
        ).pack(side="left", padx=0, pady=14)

        # Main
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=18, pady=14)

        # --- Left panel ---
        left = ctk.CTkFrame(main, fg_color=APP_COLORS["panel"], corner_radius=12, width=290)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        def section(text):
            ctk.CTkLabel(
                left, text=text,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=APP_COLORS["text_dim"],
            ).pack(anchor="w", padx=16, pady=(14, 2))
            ctk.CTkFrame(left, height=1, fg_color=APP_COLORS["accent"]).pack(fill="x", padx=16, pady=(0, 6))

        def mkbtn(text, cmd):
            ctk.CTkButton(
                left, text=text,
                font=ctk.CTkFont(size=13),
                fg_color=APP_COLORS["accent"],
                hover_color="#1a4a7a",
                corner_radius=8, height=36, anchor="w",
                command=cmd, width=258,
            ).pack(padx=16, pady=(0, 6))

        # Input
        section("EINGABE")
        mkbtn("  Bild(er) auswaehlen", self._browse_files)
        mkbtn("  Ordner auswaehlen",   self._browse_folder)
        self._lbl_input = ctk.CTkLabel(
            left, text="Keine Auswahl",
            font=ctk.CTkFont(size=11), text_color=APP_COLORS["text_dim"],
            wraplength=250, justify="left",
        )
        self._lbl_input.pack(anchor="w", padx=16, pady=(0, 6))

        # Settings
        section("EINSTELLUNGEN")

        ctk.CTkLabel(left, text="Confidence (empfindlicher unten)",
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=16, pady=(0, 2))
        self._conf_var = ctk.DoubleVar(value=DEFAULT_CONFIDENCE)
        self._conf_label = ctk.CTkLabel(left, text=f"{DEFAULT_CONFIDENCE:.2f}",
                                        font=ctk.CTkFont(size=11),
                                        text_color=APP_COLORS["text_dim"])
        self._conf_label.pack(anchor="e", padx=16)
        ctk.CTkSlider(
            left, from_=0.01, to=0.99, variable=self._conf_var,
            width=258, progress_color=APP_COLORS["highlight"],
            command=lambda v: self._conf_label.configure(text=f"{v:.2f}"),
        ).pack(padx=16, pady=(0, 10))

        ctk.CTkLabel(left, text="Min. Flaeche (px^2)",
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=16, pady=(0, 2))
        self._area_var = ctk.IntVar(value=DEFAULT_MIN_AREA)
        self._area_label = ctk.CTkLabel(left, text=f"{DEFAULT_MIN_AREA}",
                                         font=ctk.CTkFont(size=11),
                                         text_color=APP_COLORS["text_dim"])
        self._area_label.pack(anchor="e", padx=16)
        ctk.CTkSlider(
            left, from_=0, to=400, variable=self._area_var,
            width=258, progress_color=APP_COLORS["highlight"],
            command=lambda v: self._area_label.configure(text=f"{int(v)}"),
        ).pack(padx=16, pady=(0, 10))

        # Modell-Status
        section("MODELL")
        unet_ok = UNET_ONNX_PATH.exists()
        model_color = APP_COLORS["success"] if unet_ok else APP_COLORS["warning"]
        model_text  = f"U-Net geladen: {UNET_ONNX_PATH.name}" if unet_ok else "Kein Modell gefunden.\nLege crack_unet.onnx in den model/ Ordner."
        ctk.CTkLabel(
            left,
            text=model_text,
            font=ctk.CTkFont(size=11),
            text_color=model_color,
            wraplength=240, justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 10))

        # --- Right panel ---
        right = ctk.CTkFrame(main, fg_color=APP_COLORS["panel"], corner_radius=12)
        right.pack(side="left", fill="both", expand=True)

        ctk.CTkLabel(right, text="ERGEBNIS",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=APP_COLORS["text_dim"]).pack(anchor="w", padx=16, pady=(14, 2))
        ctk.CTkFrame(right, height=1, fg_color=APP_COLORS["accent"]).pack(fill="x", padx=16, pady=(0, 6))

        # Preview image
        self._preview_label = ctk.CTkLabel(
            right, text="Kein Bild",
            fg_color="#0d0d1a",
            font=ctk.CTkFont(size=13), text_color=APP_COLORS["text_dim"],
            corner_radius=8,
        )
        self._preview_label.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        self._preview_label.bind("<Button-1>", self._open_zoom_window)

        hint = ctk.CTkLabel(right, text="Klick auf Bild oeffnet Zoom-Ansicht",
                            font=ctk.CTkFont(size=10),
                            text_color=APP_COLORS["text_dim"])
        hint.pack(anchor="e", padx=14, pady=(0, 4))

        # Log
        ctk.CTkLabel(right, text="PROTOKOLL",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=APP_COLORS["text_dim"]).pack(anchor="w", padx=16, pady=(6, 2))
        self._log_box = ctk.CTkTextbox(
            right, fg_color="#0d0d1a", text_color="#c0c0d0",
            font=ctk.CTkFont(family="Consolas", size=11),
            corner_radius=8, height=160,
        )
        self._log_box.pack(fill="x", padx=12, pady=(0, 8))
        self._log_box.configure(state="disabled")

        # Progress
        prog = ctk.CTkFrame(right, fg_color="transparent")
        prog.pack(fill="x", padx=12, pady=(0, 8))
        self._lbl_status = ctk.CTkLabel(prog, text="Bereit",
                                         font=ctk.CTkFont(size=12),
                                         text_color=APP_COLORS["text_dim"])
        self._lbl_status.pack(anchor="w")
        self._progress_bar = ctk.CTkProgressBar(
            prog, fg_color=APP_COLORS["accent"],
            progress_color=APP_COLORS["highlight"],
            corner_radius=4, height=10,
        )
        self._progress_bar.set(0)
        self._progress_bar.pack(fill="x", pady=(4, 0))

        # Bottom bar
        bar = ctk.CTkFrame(self, fg_color=APP_COLORS["accent"], corner_radius=0)
        bar.pack(fill="x", side="bottom")

        # Settings button (left-most)
        ctk.CTkButton(
            bar, text="Einstellungen",
            font=ctk.CTkFont(size=13),
            fg_color="#2d3748", hover_color="#4a5568",
            corner_radius=8, height=42, width=130,
            command=self._open_settings,
        ).pack(side="left", padx=(16, 8), pady=10)

        self._btn_start = ctk.CTkButton(
            bar, text="  Risse erkennen",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=APP_COLORS["highlight"], hover_color=APP_COLORS["hover"],
            corner_radius=8, height=42, command=self._start,
        )
        self._btn_start.pack(side="left", padx=(0, 8), pady=10)

        self._btn_stop = ctk.CTkButton(
            bar, text="  Abbrechen",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#374151", hover_color="#4b5563",
            corner_radius=8, height=42, command=self._stop, state="disabled",
        )
        self._btn_stop.pack(side="left", padx=(0, 8), pady=10)

        # Export-Buttons (werden nach der Analyse aktiv)
        self._btn_geojson = ctk.CTkButton(
            bar, text="  GeoJSON",
            font=ctk.CTkFont(size=13),
            fg_color="#1a4a3a", hover_color="#256b52",
            corner_radius=8, height=42, width=118,
            command=self._open_geojson, state="disabled",
        )
        self._btn_geojson.pack(side="left", padx=(0, 6), pady=10)

        self._btn_dxf = ctk.CTkButton(
            bar, text="  DXF",
            font=ctk.CTkFont(size=13),
            fg_color="#1a3a4a", hover_color="#255a6b",
            corner_radius=8, height=42, width=100,
            command=self._open_dxf,
            state="disabled" if HAS_EZDXF else "disabled",
        )
        self._btn_dxf.pack(side="left", padx=(0, 8), pady=10)

        ctk.CTkButton(
            bar, text="  Output",
            font=ctk.CTkFont(size=13),
            fg_color="#374151", hover_color="#4b5563",
            corner_radius=8, height=42, width=110,
            command=self._open_output,
        ).pack(side="right", padx=16, pady=10)

    # --- Settings Dialog ---

    def _open_settings(self):
        """Opens a settings dialog for contour tolerance and export formats."""
        win = ctk.CTkToplevel(self)
        win.title("Einstellungen")
        win.geometry("380x520")
        win.resizable(False, False)
        win.configure(fg_color=APP_COLORS["panel"])
        win.transient(self)
        win.grab_set()

        def _section(parent, text, top_pad=18):
            ctk.CTkLabel(
                parent, text=text,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=APP_COLORS["text_dim"],
            ).pack(anchor="w", padx=20, pady=(top_pad, 2))
            ctk.CTkFrame(parent, height=1, fg_color=APP_COLORS["accent"]).pack(fill="x", padx=20, pady=(0, 8))

        # --- Contour tolerance ---
        _section(win, "KONTUR-GLAETTUNG")

        ctk.CTkLabel(
            win, text="Niedrig = detailliert, hoch = glatt",
            font=ctk.CTkFont(size=11), text_color=APP_COLORS["text_dim"],
        ).pack(anchor="w", padx=20)

        eps_label = ctk.CTkLabel(
            win, text=f"{self._epsilon_var.get():.4f}",
            font=ctk.CTkFont(size=11), text_color=APP_COLORS["text_dim"],
        )
        eps_label.pack(anchor="e", padx=20)

        ctk.CTkSlider(
            win, from_=0.0, to=0.002, variable=self._epsilon_var,
            width=340, progress_color=APP_COLORS["highlight"],
            command=lambda v: eps_label.configure(text=f"{v:.4f}"),
        ).pack(padx=20, pady=(0, 12))

        # --- Tile size ---
        _section(win, "KACHELGROESSE", top_pad=4)

        ctk.CTkLabel(
            win, text="Kachelgroesse",
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w", padx=20, pady=(0, 4))
        ctk.CTkSegmentedButton(
            win, values=["512", "768", "1024"],
            variable=self._tile_var,
            font=ctk.CTkFont(size=12), width=340,
            selected_color=APP_COLORS["highlight"],
            selected_hover_color=APP_COLORS["hover"],
        ).pack(padx=20, pady=(0, 4))

        ctk.CTkLabel(
            win,
            text="Empfehlung: 512 (= Modell-Trainingsgroesse, bestes Ergebnis)",
            font=ctk.CTkFont(size=10), text_color=APP_COLORS["text_dim"],
            wraplength=340, justify="left",
        ).pack(anchor="w", padx=20, pady=(0, 4))

        ctk.CTkCheckBox(
            win,
            text="Feine Risse (Multi-Scale)",
            variable=self._multiscale_var,
            font=ctk.CTkFont(size=12),
            checkbox_width=18, checkbox_height=18,
            checkmark_color=APP_COLORS["highlight"],
            border_color=APP_COLORS["accent"],
        ).pack(anchor="w", padx=20, pady=(4, 8))

        # --- Export formats ---
        _section(win, "EXPORT-FORMATE", top_pad=4)

        for label, var in [
            ("GeoJSON (Vektordaten)", self._export_geojson),
            ("DXF (CAD-Export)",      self._export_dxf),
            ("Maske (PNG, Vollaufloesung)", self._export_mask),
        ]:
            ctk.CTkCheckBox(
                win, text=label, variable=var,
                font=ctk.CTkFont(size=12),
                checkbox_width=18, checkbox_height=18,
                checkmark_color=APP_COLORS["highlight"],
                border_color=APP_COLORS["accent"],
            ).pack(anchor="w", padx=20, pady=(0, 8))

        # --- Close button ---
        ctk.CTkButton(
            win, text="Schliessen",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=APP_COLORS["highlight"], hover_color=APP_COLORS["hover"],
            corner_radius=8, height=38, width=160,
            command=win.destroy,
        ).pack(pady=(14, 16))

    # --- Helpers ---


    def _browse_files(self):
        paths = filedialog.askopenfilenames(
            title="Bilder auswaehlen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"), ("Alle", "*.*")],
        )
        if paths:
            self._input_paths = [Path(p) for p in paths if Path(p).suffix.lower() in SUPPORTED_EXT]
            n = len(self._input_paths)
            label = f"{n} Bild(er)" if n > 1 else self._input_paths[0].name
            self._lbl_input.configure(text=label)

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Ordner mit Bildern auswaehlen")
        if folder:
            f = Path(folder)
            found: List[Path] = []
            for ext in SUPPORTED_EXT:
                found.extend(f.glob(f"*{ext}"))
                found.extend(f.glob(f"*{ext.upper()}"))
            self._input_paths = sorted(set(found))
            n = len(self._input_paths)
            self._lbl_input.configure(text=f"{n} Bild(er) in {f.name}/")

    def _open_output(self):
        out_dir = self._last_export.get("output_dir")
        if out_dir and Path(out_dir).exists():
            os.startfile(str(out_dir))
        elif self._input_paths:
            out = self._input_paths[0].parent / "output"
            out.mkdir(exist_ok=True)
            os.startfile(str(out))

    def _open_geojson(self):
        path = self._last_export.get("geojson")
        if path and Path(path).exists():
            os.startfile(str(path))
        else:
            self._append_log("GeoJSON nicht gefunden - Analyse zuerst ausfuehren.")

    def _open_dxf(self):
        path = self._last_export.get("dxf")
        if path and Path(path).exists():
            os.startfile(str(path))
        else:
            self._append_log("DXF nicht gefunden - Analyse zuerst ausfuehren.")

    def _open_zoom_window(self, _event=None):
        """Klick auf Vorschau oeffnet das gespeicherte Ergebnisbild im System-Viewer."""
        if self._last_output_img and Path(self._last_output_img).exists():
            os.startfile(str(self._last_output_img))
        elif self._last_annotated is not None:
            # Fallback: Bild temporaer speichern und oeffnen
            import tempfile
            tmp = Path(tempfile.mktemp(suffix="_crackdetect_preview.png"))
            Image.fromarray(self._last_annotated).save(str(tmp))
            os.startfile(str(tmp))

    def _append_log(self, text: str):
        self._log_box.configure(state="normal")
        self._log_box.insert("end", text + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _update_preview(self, img_np: np.ndarray):
        self._last_annotated = img_np
        h, w = img_np.shape[:2]
        max_w, max_h = 680, 420
        scale = min(max_w / w, max_h / h, 1.0)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        resized = cv2.resize(img_np, (nw, nh), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        self._preview_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(nw, nh))
        self._preview_label.configure(image=self._preview_image, text="")

    # --- Queue polling ---

    def _poll(self):
        try:
            while True:
                kind, data = self._msg_q.get_nowait()
                if kind == "log":
                    self._append_log(str(data))
                elif kind == "status":
                    self._lbl_status.configure(text=str(data))
                elif kind == "progress":
                    self._progress_bar.set(float(data))
                elif kind == "preview":
                    self._update_preview(data)
                elif kind == "last_output":
                    self._last_output_img = data
                elif kind == "export_paths":
                    self._last_export = data
                    # GeoJSON-Button aktivieren
                    if data.get("geojson"):
                        self._btn_geojson.configure(
                            state="normal",
                            text=f"  GeoJSON",
                            fg_color="#22c55e", hover_color="#16a34a",
                            text_color="#000000",
                        )
                    # DXF-Button aktivieren
                    if data.get("dxf") and HAS_EZDXF:
                        self._btn_dxf.configure(
                            state="normal",
                            text=f"  DXF",
                            fg_color="#3b82f6", hover_color="#2563eb",
                            text_color="#ffffff",
                        )
                elif kind == "done":
                    self._btn_start.configure(state="normal")
                    self._btn_stop.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(80, self._poll)

    # --- Start / Stop ---

    def _start(self):
        if not self._input_paths:
            self._append_log("Bitte zuerst Bilder oder Ordner auswaehlen.")
            return
        if self._thread and self._thread.is_alive():
            return

        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._lbl_status.configure(text="Starte Analyse ...")
        self._progress_bar.set(0)
        self._append_log("=" * 54)
        multi = self._multiscale_var.get()
        tile  = int(self._tile_var.get())
        self._append_log(
            f"Starte: {len(self._input_paths)} Bild(er) | "
            f"Kachel: {tile}px | Multi-Scale: {'ja' if multi else 'nein'} | "
            f"Modell: U-Net"
        )

        # Export-Buttons zuruecksetzen
        self._btn_geojson.configure(state="disabled", fg_color="#1a4a3a",
                                    text="  GeoJSON", text_color="white")
        self._btn_dxf.configure(state="disabled", fg_color="#1a3a4a",
                                text="  DXF", text_color="white")
        self._last_export = {}

        # Apply contour epsilon setting
        global _CONTOUR_EPSILON
        _CONTOUR_EPSILON = self._epsilon_var.get()

        self._worker = Worker(self._msg_q)
        self._thread = threading.Thread(
            target=self._worker.run,
            args=(
                self._input_paths,
                self._conf_var.get(),
                int(self._area_var.get()),
                tile,
                multi,
                self._export_geojson.get(),
                self._export_dxf.get(),
                self._export_mask.get(),
            ),
            daemon=True,
        )
        self._thread.start()

    def _stop(self):
        if self._worker:
            self._worker.stop()
        self._btn_stop.configure(state="disabled")

    def _on_close(self):
        self._stop()
        self.destroy()


# ==============================================================================
#  EINSTIEGSPUNKT
# ==============================================================================

if __name__ == "__main__":
    print()
    print(" ================================================================")
    print("   CrackDetect - Automatische Riss-Erkennung")
    print("   Powered by U-Net")
    print(" ================================================================")
    print()

    try:
        load_models()
    except Exception as e:
        print(f"  [ERROR] {e}")
        print("  Lege dein trainiertes Modell hier ab: model/crack_unet.onnx")
        input("  Enter druecken zum Beenden ...")
        sys.exit(1)

    print("[INFO] Starte Desktop-App ...")
    app = CrackDetectApp()
    app.mainloop()
