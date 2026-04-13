"""
CrackDetect – Automatische Riss-Erkennung in Bildern und Orthofotos
====================================================================
Powered by SAM3 (Text-Prompt → Pixel-Masken)

Pipeline:
  1. Bild laden (JPG/PNG/TIFF – mit rasterio falls GeoTIFF)
  2. Automatisches Tiling für grosse Bilder (>2500 px)
  3. SAM3 erkennt Risse via Text-Prompt "crack" (direkte Segmentierung)
  4. Masken werden zu Shapely-Polygonen vektorisiert
  5. NMS entfernt Duplikate aus überlappenden Kacheln
  6. Pixel-Koordinaten → Weltkoordinaten (falls GeoTIFF mit CRS)
  7. Export als GeoJSON + DXF

Für Details: siehe projekt.md
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import queue
import threading
from tkinter import filedialog

import numpy as np
import cv2
from PIL import Image
import torch
import customtkinter as ctk

# ─── Optionale Geo-Imports ─────────────────────────────────────────────────────
try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio nicht gefunden – GeoTIFF-Koordinaten nicht verfügbar")

try:
    from shapely.geometry import Polygon, mapping
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("[WARN] shapely nicht gefunden – nur Bild-Ausgabe möglich")

try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False
    print("[WARN] ezdxf nicht gefunden – DXF-Export deaktiviert")

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ─── SAM3 Imports ─────────────────────────────────────────────────────────────
HAS_SAM3 = False
try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    HAS_SAM3 = True
except ImportError:
    print("[ERROR] SAM3 nicht installiert! Bitte start.bat erneut ausführen.")

# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR         = Path(__file__).parent
CHECKPOINTS_DIR  = BASE_DIR / "checkpoints"
SAM3_CKPT_DIR    = CHECKPOINTS_DIR / "sam3"
OUTPUT_DIR       = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TILE_SIZE        = 1024   # Kachelgrösse in Pixeln
TILE_OVERLAP     = 256    # Überlappung zwischen Kacheln
MAX_DIRECT_SIZE  = 2500   # Über diesem Wert wird automatisch getiled

SAM3_CONFIDENCE       = 0.15   # Standard-Confidence für SAM3 Text-Suche
SAM3_MIN_MASK_PX      = 100    # Mindestgrösse einer Maske in Pixeln
SAM3_MAX_MASKS_PER_TILE = 30   # Maximale Masken pro Kachel (Top-N nach Score)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# World-File Endungen pro Bildformat (für Sidecar-Georeferenzierung)
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

# Globale SAM3-Instanz
_sam3_proc = None


# ══════════════════════════════════════════════════════════════════════════════
#  MODELLE LADEN
# ══════════════════════════════════════════════════════════════════════════════

def load_models() -> None:
    """Lädt SAM3 einmalig. Wird beim App-Start aufgerufen."""
    global _sam3_proc

    if not HAS_SAM3:
        raise RuntimeError("SAM3 nicht installiert – start.bat erneut ausführen.")

    print(f"[INFO] Verwende Device: {DEVICE.upper()}")

    # ── Monkey-patch: fused addmm_act → standard float32 ────────────────────
    # Das Original erzwingt bfloat16 für einen fused CUDA-Kernel der
    # möglicherweise nicht verfügbar ist. Ersatz durch Standard-Linear.
    try:
        import sam3.perflib.fused as _fused_mod
        import sam3.model.vitdet as _vitdet_mod

        def _addmm_act_f32(activation, linear, mat1):
            x = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
            return activation()(x)

        _fused_mod.addmm_act = _addmm_act_f32
        _vitdet_mod.addmm_act = _addmm_act_f32
    except Exception as e:
        print(f"[WARN] Monkey-patch für addmm_act fehlgeschlagen: {e}")

    # ── BPE-Vokabular-Pfad für SAM3 Text-Encoder ────────────────────────────
    import sam3 as _sam3_pkg
    bpe_path = Path(_sam3_pkg.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.exists():
        bpe_path = Path(_sam3_pkg.__file__).parent.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    # ── Checkpoint-Pfad auflösen ─────────────────────────────────────────────
    ckpt_path = None
    if SAM3_CKPT_DIR.exists():
        for candidate in ("sam3.pt", "model.safetensors"):
            if (SAM3_CKPT_DIR / candidate).exists():
                ckpt_path = str(SAM3_CKPT_DIR / candidate)
                break

    if ckpt_path is None:
        print("[INFO] Checkpoint nicht lokal – lade von HuggingFace ...")

    print("[INFO] Lade SAM3 ...")
    model = build_sam3_image_model(
        checkpoint_path=ckpt_path,
        bpe_path=str(bpe_path) if bpe_path.exists() else None,
        device=DEVICE,
        load_from_HF=(ckpt_path is None),
    )
    # Sicherstellen: float32 (Checkpoint kann bfloat16 sein)
    model = model.float()

    _sam3_proc = Sam3Processor(
        model,
        confidence_threshold=SAM3_CONFIDENCE,
        device=DEVICE,
    )

    print(f"[OK]   SAM3 bereit ({DEVICE.upper()}).")


# ══════════════════════════════════════════════════════════════════════════════
#  GEO-BILD-WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

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
        self.image       = image          # HxWx3 uint8 RGB
        self.transform   = transform      # rasterio Affin-Transform
        self.crs         = crs            # rasterio CRS
        self.source_path = source_path
        self.has_geo     = (transform is not None and crs is not None)

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    def px_to_world(self, px: float, py: float) -> Tuple[float, float]:
        if not self.has_geo:
            return px, py
        wx, wy = self.transform * (px, py)
        return wx, wy

    def coords_to_world(self, pixel_coords: List[Tuple]) -> List[Tuple]:
        if not self.has_geo:
            return pixel_coords
        return [self.px_to_world(x, y) for x, y in pixel_coords]


# ── World-File / PRJ Hilfsfunktionen ──────────────────────────────────────────

def _find_world_file(image_path: Path) -> Optional[Path]:
    """Sucht ein World-File (.jgw, .tfw, etc.) neben dem Bild."""
    ext = image_path.suffix.lower()
    candidates = WORLD_FILE_EXT.get(ext, [])
    for wext in candidates:
        wf = image_path.with_suffix(wext)
        if wf.exists():
            return wf
    return None


def _parse_world_file(wf_path: Path):
    """
    Liest ein World-File und gibt ein rasterio Affine-Transform zurück.

    World-File Format (6 Zeilen):
      A  = Pixel-Größe X (z.B. 0.002 m)
      D  = Rotation Y (normalerweise 0)
      B  = Rotation X (normalerweise 0)
      E  = Pixel-Größe Y (negativ, z.B. -0.002 m)
      C  = X-Koordinate obere linke Ecke
      F  = Y-Koordinate obere linke Ecke
    """
    if not HAS_RASTERIO:
        return None
    try:
        lines = wf_path.read_text().strip().splitlines()
        if len(lines) < 6:
            return None
        vals = [float(line.strip()) for line in lines[:6]]
        # Affine(a, b, c, d, e, f) = Affine(pixel_x, rot_x, origin_x, rot_y, pixel_y, origin_y)
        return Affine(vals[0], vals[2], vals[4], vals[1], vals[3], vals[5])
    except Exception as e:
        print(f"[WARN] World-File Lesefehler ({wf_path.name}): {e}")
        return None


def _read_prj_crs(image_path: Path):
    """Liest CRS aus einer .prj Sidecar-Datei (WKT-Format)."""
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
    Lädt Bild mit optionalen Geo-Metadaten.

    Geo-Quellen (in Prioritätsreihenfolge):
      1. Eingebettetes GeoTIFF (CRS + Transform im TIFF-Header)
      2. GDAL-Sidecar: World-File (.jgw/.tfw/etc.) + .prj (automatisch via rasterio)
      3. Manuelle Sidecar-Analyse als Fallback
      4. Kein Geo → Pixel-Koordinaten
    """
    p = Path(path)

    # ── Versuch 1: rasterio für ALLE Formate (GDAL liest GeoTIFF, World-Files, etc.) ──
    if HAS_RASTERIO:
        try:
            with rasterio.open(str(p)) as src:
                bands = min(src.count, 3)
                channels = [src.read(i + 1) for i in range(bands)]
                if bands == 1:
                    channels = channels * 3
                image = np.dstack(channels)

                # Normalisiere auf uint8 (GeoTIFF kann 16-bit, 32-bit float sein)
                if image.dtype != np.uint8:
                    lo, hi = image.min(), image.max()
                    image = ((image.astype(np.float32) - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

                transform = src.transform
                crs = src.crs

                # Prüfe ob rasterio/GDAL echte Geo-Daten erkannt hat
                has_embedded_geo = (crs is not None and transform is not None)

                if has_embedded_geo:
                    crs_label = crs.to_epsg() or crs.to_string()
                    print(f"[INFO] Georeferenziert geladen: {src.width}×{src.height} px "
                          f"| CRS: {crs_label}")
                    return GeoImage(image, transform, crs, str(p))

                # rasterio hat kein CRS → versuche Sidecar World-File + PRJ manuell
                wf = _find_world_file(p)
                if wf:
                    wf_transform = _parse_world_file(wf)
                    wf_crs = _read_prj_crs(p)
                    if wf_transform:
                        crs_str = ""
                        if wf_crs:
                            crs_str = f" | CRS: {wf_crs.to_epsg() or wf_crs.to_string()}"
                        print(f"[INFO] World-File geladen: {src.width}×{src.height} px"
                              f"{crs_str} | {wf.name}")
                        return GeoImage(image, wf_transform, wf_crs, str(p))

                print(f"[INFO] Bild geladen (ohne Geo): {src.width}×{src.height} px")
                return GeoImage(image, source_path=str(p))

        except Exception as e:
            print(f"[WARN] rasterio-Fehler, Fallback auf PIL: {e}")

    # ── Versuch 2: PIL + manuelle World-File-Analyse ──────────────────────────
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
            print(f"[INFO] PIL + World-File: {pil.width}×{pil.height} px{crs_str}")
            return GeoImage(image, wf_transform, wf_crs, str(p))

    return GeoImage(image, source_path=str(p))


# ══════════════════════════════════════════════════════════════════════════════
#  TILING
# ══════════════════════════════════════════════════════════════════════════════

def compute_tiles(
    w: int,
    h: int,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> List[Tuple[int, int, int, int]]:
    """Gibt Liste von (x0, y0, x1, y1) Kacheln zurück."""
    stride = tile_size - overlap
    tiles  = []

    y0 = 0
    while y0 < h:
        x0 = 0
        while x0 < w:
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            tiles.append((x0, y0, x1, y1))
            if x1 >= w:
                break
            x0 += stride
        if y0 + stride >= h:
            break
        y0 += stride

    return tiles


# ══════════════════════════════════════════════════════════════════════════════
#  ERKENNUNG (SAM3 TEXT-PROMPT)
# ══════════════════════════════════════════════════════════════════════════════

def detect_in_tile(
    tile_np: np.ndarray,
    text_prompts: List[str],
    confidence: float,
    log_fn=None,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Erkennt Risse in einem einzelnen Tile via SAM3 Text-Prompt.

    Args:
        tile_np       – RGB uint8 Array (H×W×3)
        text_prompts  – Liste von Text-Prompts (z.B. ["crack", "fracture"])
        confidence    – Confidence-Schwellwert für SAM3
        log_fn        – Optionale Log-Funktion für Fehler/Debug-Ausgaben

    Returns:
        masks  – Liste von bool-Masken (H×W)
        scores – geschätzte Konfidenz-Scores
    """
    global _sam3_proc

    h, w = tile_np.shape[:2]
    pil_tile = Image.fromarray(tile_np)

    # ── SAM3: Bild encodieren ─────────────────────────────────────────────
    try:
        state = _sam3_proc.set_image(pil_tile)
    except Exception as exc:
        msg = f"[FEHLER] SAM3 set_image: {exc}"
        print(msg)
        if log_fn:
            log_fn(f"   ❌ {msg}")
        return [], []

    # Temporär Confidence anpassen
    original_conf = None
    if hasattr(_sam3_proc, "confidence_threshold"):
        original_conf = _sam3_proc.confidence_threshold
        _sam3_proc.confidence_threshold = confidence

    masks_out  = []
    scores_out = []

    for prompt in text_prompts:
        try:
            _sam3_proc.reset_all_prompts(state)
            result_state = _sam3_proc.set_text_prompt(prompt, state)
        except Exception as exc:
            msg = f"[FEHLER] SAM3 Prompt '{prompt}': {exc}"
            print(msg)
            if log_fn:
                log_fn(f"   ❌ {msg}")
            continue

        raw_masks = result_state.get("masks")
        raw_scores = result_state.get("scores")

        if raw_masks is None or (hasattr(raw_masks, "__len__") and len(raw_masks) == 0):
            if log_fn:
                log_fn(f"   [SAM3] '{prompt}': 0 Masken (alle unter Schwellwert {confidence:.2f})")
            continue

        n_raw = raw_masks.shape[0] if hasattr(raw_masks, 'shape') else len(raw_masks)

        if hasattr(raw_masks, "cpu"):
            raw_masks = raw_masks.cpu().numpy()
        if raw_scores is not None and hasattr(raw_scores, "cpu"):
            raw_scores = raw_scores.cpu().numpy()

        # Masken auf Top-N nach Score begrenzen
        if n_raw > SAM3_MAX_MASKS_PER_TILE:
            top_idx    = np.argsort(raw_scores)[::-1][:SAM3_MAX_MASKS_PER_TILE]
            raw_masks  = raw_masks[top_idx]
            raw_scores = raw_scores[top_idx]
            n_kept = SAM3_MAX_MASKS_PER_TILE
        else:
            n_kept = n_raw
        if log_fn:
            log_fn(f"   [SAM3] '{prompt}': {n_raw} Masken über Schwellwert {confidence:.2f} (behalte Top-{n_kept})")
        for i, m in enumerate(raw_masks):
            # Dimensionen normalisieren
            if m.ndim == 4:     # (1, 1, H, W)
                m = m[0, 0]
            elif m.ndim == 3:   # (1, H, W)
                m = m[0]
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_LINEAR)

            binary = (m > 0.5).astype(bool)
            px = int(np.count_nonzero(binary))

            if px < SAM3_MIN_MASK_PX:
                continue

            score = float(raw_scores[i]) if raw_scores is not None and i < len(raw_scores) else confidence
            masks_out.append(binary)
            scores_out.append(score)

    # SAM3 State freigeben
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Confidence zurücksetzen
    if original_conf is not None:
        _sam3_proc.confidence_threshold = original_conf

    return masks_out, scores_out


# ══════════════════════════════════════════════════════════════════════════════
#  MASKE → POLYGON
# ══════════════════════════════════════════════════════════════════════════════

def mask_to_polygon(
    mask: np.ndarray,
    offset_x: int = 0,
    offset_y: int = 0,
    min_area: int = 100,
) -> Optional[object]:
    """Konvertiert bool-Maske → Shapely-Polygon mit Kachel-Offset."""
    if not HAS_SHAPELY:
        return None

    mask_u8   = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    # Kontur vereinfachen
    eps   = 0.002 * cv2.arcLength(largest, True)
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
    """IoU-basierte NMS mit STRtree-Spatial-Index – O(n log n) statt O(n²).

    Returns:
        keep_polys  – gefilterte Polygone
        keep_scores – zugehörige Scores (gleiche Reihenfolge)
    """
    if not polygons:
        return [], []

    from shapely.strtree import STRtree

    # Größte zuerst (höchste Fläche = bevorzugt beibehalten)
    order  = sorted(range(len(polygons)), key=lambda i: polygons[i].area, reverse=True)
    polys  = [polygons[i] for i in order]
    scrs   = [scores[i]   for i in order]

    suppressed = [False] * len(polys)
    tree       = STRtree(polys)

    for i, poly in enumerate(polys):
        if suppressed[i]:
            continue
        # Kandidaten via Bounding-Box vorfiltern
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


# ══════════════════════════════════════════════════════════════════════════════
#  SKELETT-HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def _thin_mask(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a boolean mask. Returns uint8 1-pixel-wide skeleton."""
    if HAS_SKIMAGE:
        return _skimage_skeletonize(mask).astype(np.uint8)
    # Morphological thinning fallback (Zhang-Suen-style via erosion chain)
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
    """
    Merge all overlapping / touching Shapely polygons into disjoint regions.
    Uses a small buffer so adjacent cracks also fuse.
    """
    if not HAS_SHAPELY or not polygons:
        return polygons

    from shapely.ops import unary_union
    from shapely.geometry import MultiPolygon

    buffered = [p.buffer(4) for p in polygons]        # grow 4 px
    union    = unary_union(buffered)

    if union.is_empty:
        return []

    geoms = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
    shrunk = [g.buffer(-4) for g in geoms]            # shrink back
    return [g for g in shrunk if g is not None and not g.is_empty and g.area > 0]


def polygon_to_crack_line(poly, canvas_h: int, canvas_w: int,
                          min_length_px: int = 20) -> Optional[object]:
    """
    Convert a filled Shapely Polygon (crack region) into a skeleton LineString.

    Steps:
      1. Rasterise the polygon into a local binary mask
      2. Skeletonise → 1-px-wide centre line
      3. Sort skeleton pixels along their principal axis (PCA)
      4. Douglas-Peucker simplification → compact LineString
    """
    if not HAS_SHAPELY:
        return None

    from shapely.geometry import LineString

    minx, miny, maxx, maxy = poly.bounds
    minx = max(0, int(minx) - 2)
    miny = max(0, int(miny) - 2)
    maxx = min(canvas_w, int(maxx) + 3)
    maxy = min(canvas_h, int(maxy) + 3)
    bw, bh = maxx - minx, maxy - miny
    if bw < 2 or bh < 2:
        return None

    # Rasterise polygon into local mask
    mask = np.zeros((bh, bw), dtype=np.uint8)
    ext  = np.array(
        [(int(x) - minx, int(y) - miny) for x, y in poly.exterior.coords],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [ext], 1)

    # Skeletonise
    skel = _thin_mask(mask.astype(bool))
    ys, xs = np.where(skel)
    if len(xs) < 3:
        return None

    # Global pixel coords
    pts = np.column_stack([xs + minx, ys + miny]).astype(float)

    # PCA-based ordering so points run along the crack, not random
    center   = pts.mean(axis=0)
    centered = pts - center
    sample   = centered[: min(len(centered), 3000)]
    _, _, Vt = np.linalg.svd(sample, full_matrices=False)
    axis     = Vt[0]
    order    = np.argsort(centered @ axis)
    ordered  = pts[order]

    # Downsample to max 800 points before simplification
    if len(ordered) > 800:
        idx     = np.round(np.linspace(0, len(ordered) - 1, 800)).astype(int)
        ordered = ordered[idx]

    # Douglas-Peucker
    cv_pts     = ordered.reshape(-1, 1, 2).astype(np.int32)
    simplified = cv2.approxPolyDP(cv_pts, 3.0, closed=False)
    if simplified is None or len(simplified) < 2:
        return None
    simplified = simplified.reshape(-1, 2)

    coords = [(int(p[0]), int(p[1])) for p in simplified]
    line   = LineString(coords)
    return line if line.length >= min_length_px else None


def _geom_coords_and_closed(geom) -> Tuple[List[Tuple], bool]:
    """Return (coords_list, is_closed) for Polygon or LineString."""
    if hasattr(geom, "exterior"):           # Polygon
        return list(geom.exterior.coords), True
    if hasattr(geom, "coords"):             # LineString
        return list(geom.coords), False
    if hasattr(geom, "geoms"):              # Multi* – take longest part
        longest = max(geom.geoms, key=lambda g: g.length)
        return _geom_coords_and_closed(longest)
    return [], False


# ══════════════════════════════════════════════════════════════════════════════
#  HAUPT-PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _parse_prompts(text_prompt: str) -> List[str]:
    """Parst den Text-Prompt in eine Liste einzelner Prompts.
    
    Eingabe: "crack . fracture . fissure"
    Ausgabe: ["crack", "fracture", "fissure"]
    """
    parts = [p.strip() for p in text_prompt.split(".")]
    return [p for p in parts if p]


def process_geo_image(
    geo_image: GeoImage,
    text_prompt: str     = "crack",
    confidence: float    = 0.05,
    min_area: int        = 100,
    progress_cb=None,
    log_fn=None,
) -> Tuple[np.ndarray, List, List[float]]:
    """
    Verarbeitet ein GeoImage komplett (mit automatischem Tiling).

    Returns:
        annotated_np  – Bild mit blauem Overlay (H×W×3 uint8)
        polygons      – Liste von Shapely-Polygonen (Pixel-Koordinaten)
        scores        – Konfidenz-Scores
    """
    img  = geo_image.image
    h, w = img.shape[:2]

    text_prompts = _parse_prompts(text_prompt)
    if not text_prompts:
        text_prompts = ["crack"]

    all_polygons: List = []
    all_scores:  List[float] = []

    needs_tiling = max(w, h) > MAX_DIRECT_SIZE

    if needs_tiling:
        tiles = compute_tiles(w, h)
        print(f"[INFO] Tiling: {len(tiles)} Kacheln für {w}×{h} px")

        _tile_masks_total = 0
        for i, (x0, y0, x1, y1) in enumerate(tiles):
            if progress_cb:
                progress_cb(i / len(tiles), f"Kachel {i+1}/{len(tiles)}")
            tile = img[y0:y1, x0:x1]
            # log_fn nur für erste Kachel (Fehler sichtbar, kein Log-Flut)
            tile_log = log_fn if i == 0 else None
            masks, scores = detect_in_tile(tile, text_prompts, confidence, log_fn=tile_log)
            _tile_masks_total += len(masks)
            for mask, score in zip(masks, scores):
                poly = mask_to_polygon(mask, offset_x=x0, offset_y=y0, min_area=min_area)
                if poly is not None:
                    all_polygons.append(poly)
                    all_scores.append(score)
        if log_fn:
            if _tile_masks_total == 0:
                log_fn(f"   ⚠ Alle {len(tiles)} Kacheln: 0 Masken — Confidence ({confidence:.2f}) weiter senken (min. 0.01)")
            else:
                log_fn(f"   [Tiling] {len(tiles)} Kacheln, {_tile_masks_total} Roh-Masken gefunden")
    else:
        print(f"[INFO] Direkt: {w}×{h} px")
        if progress_cb:
            progress_cb(0.2, "Erkenne Risse ...")
        masks, scores = detect_in_tile(img, text_prompts, confidence, log_fn=log_fn)
        if log_fn and len(masks) == 0:
            log_fn(f"   ⚠ 0 Masken erkannt — Confidence ({confidence:.2f}) weiter senken (Slider bis 0.01)")
        for mask, score in zip(masks, scores):
            poly = mask_to_polygon(mask, min_area=min_area)
            if poly is not None:
                all_polygons.append(poly)
                all_scores.append(score)

    # NMS nach Tiling (aggressiv: iou_threshold 0.3)
    if HAS_SHAPELY and len(all_polygons) > 1:
        if progress_cb:
            progress_cb(0.82, f"NMS: {len(all_polygons)} Masken deduplizieren …")
        before = len(all_polygons)
        all_polygons, all_scores = nms_polygons(all_polygons, all_scores, iou_threshold=0.3)
        removed = before - len(all_polygons)
        if removed > 0:
            print(f"[INFO] NMS: {removed} Duplikate entfernt → {len(all_polygons)} Risse")

    # ── Überlappende Regionen zusammenführen ──────────────────────────────────
    if HAS_SHAPELY and len(all_polygons) > 1:
        if progress_cb:
            progress_cb(0.87, f"Regionen zusammenführen: {len(all_polygons)} …")
        before = len(all_polygons)
        all_polygons = merge_overlapping_polygons(all_polygons)
        all_scores   = [1.0] * len(all_polygons)
        if before != len(all_polygons):
            print(f"[INFO] Merge: {before} → {len(all_polygons)} Riss-Regionen")

    # ── Polygone → Skelett-Linien ─────────────────────────────────────────────
    n_polys = len(all_polygons)
    if progress_cb:
        progress_cb(0.92, f"Skelettierung: 0/{n_polys} …")

    crack_lines: List = []   # LineString oder Polygon als Fallback
    _report_every = max(1, n_polys // 20)   # max. 20 Status-Updates
    for skel_i, poly in enumerate(all_polygons):
        line = polygon_to_crack_line(poly, h, w)
        crack_lines.append(line if line is not None else poly)
        if progress_cb and n_polys > 0 and skel_i % _report_every == 0:
            pct = 0.92 + 0.05 * skel_i / n_polys
            progress_cb(pct, f"Skelettierung: {skel_i + 1}/{n_polys} …")

    print(f"[INFO] {len(crack_lines)} Risslinien nach Skelettierung")

    # ── Bild skalieren, dann Risslinien einzeichnen ───────────────────────────
    if progress_cb:
        progress_cb(0.97, "Bild mit Risslinien rendern …")

    MAX_PREVIEW = 2000
    if max(h, w) > MAX_PREVIEW:
        coord_scale = MAX_PREVIEW / max(h, w)
        render = cv2.resize(img, (int(w * coord_scale), int(h * coord_scale)),
                            interpolation=cv2.INTER_AREA)
    else:
        coord_scale = 1.0
        render = img.copy()

    CRACK_COLOR = (50, 50, 255)   # Blau (RGB)
    LINE_THICK  = 3

    def _scaled_pts(coords):
        arr = np.array(coords, dtype=np.float32) * coord_scale
        return arr.astype(np.int32).reshape(-1, 1, 2)

    for obj in crack_lines:
        if obj is None:
            continue
        try:
            if hasattr(obj, "geoms"):
                for part in obj.geoms:
                    cv2.polylines(render, [_scaled_pts(list(part.coords))], False, CRACK_COLOR, LINE_THICK)
            elif hasattr(obj, "coords"):
                cv2.polylines(render, [_scaled_pts(list(obj.coords))], False, CRACK_COLOR, LINE_THICK)
            else:
                cv2.polylines(render, [_scaled_pts(list(obj.exterior.coords))], True, CRACK_COLOR, LINE_THICK)
        except Exception:
            pass

    return render, crack_lines, all_scores


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_geojson(
    entries: List[Tuple[GeoImage, List]],
    output_path: str,
) -> str:
    """
    Exportiert Crack-Linien/-Polygone aller Bilder als GeoJSON.
    Geometrien: LineString (Skelett) oder Polygon-Fallback.
    Koordinaten: Weltkoordinaten wenn GeoTIFF, sonst Pixel-Koordinaten.
    """
    if not HAS_SHAPELY:
        raise RuntimeError("shapely nicht installiert")

    from shapely.geometry import LineString as _LS

    features = []
    for geo_image, geoms in entries:
        for geom in geoms:
            if geom is None:
                continue

            # Koordinaten in Weltkoordinaten umrechnen (falls GeoTIFF)
            if geo_image.has_geo:
                if hasattr(geom, "coords"):            # LineString
                    wc = geo_image.coords_to_world(list(geom.coords))
                    export_geom = _LS(wc)
                elif hasattr(geom, "exterior"):        # Polygon
                    wc = geo_image.coords_to_world(list(geom.exterior.coords))
                    from shapely.geometry import Polygon as _Poly
                    export_geom = _Poly(wc)
                else:
                    export_geom = geom
            else:
                export_geom = geom

            length_px = geom.length if hasattr(geom, "length") else 0.0

            feature = {
                "type": "Feature",
                "geometry": mapping(export_geom),
                "properties": {
                    "id":          len(features) + 1,
                    "source_file": Path(geo_image.source_path).name,
                    "length_px":   round(length_px, 2),
                    "has_geo":     geo_image.has_geo,
                },
            }
            if geo_image.has_geo and geo_image.crs:
                feature["properties"]["crs"] = geo_image.crs.to_string()

            features.append(feature)

    doc = {"type": "FeatureCollection", "features": features}

    # CRS auf FeatureCollection-Ebene (vom ersten GeoTIFF)
    for geo_image, _ in entries:
        if geo_image.has_geo and geo_image.crs:
            doc["crs"] = {"type": "name", "properties": {"name": geo_image.crs.to_string()}}
            break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    return output_path


def export_dxf(
    entries: List[Tuple[GeoImage, List]],
    output_path: str,
) -> str:
    """
    Exportiert Polygone als DXF R2010.
    Layer CRACKS: Polygone als LWPOLYLINE (rot)
    Layer CRACK_LABELS: Beschriftungen (grün)
    """
    if not HAS_EZDXF:
        raise RuntimeError("ezdxf nicht installiert")
    if not HAS_SHAPELY:
        raise RuntimeError("shapely nicht installiert")

    doc = ezdxf.new(dxfversion="R2010")
    # Einheiten nur auf Meter setzen wenn mindestens ein Bild Geo-Daten hat
    has_any_geo = any(gi.has_geo for gi, _ in entries)
    if has_any_geo:
        doc.header["$INSUNITS"] = 6  # Meter

    doc.layers.add("CRACKS",       color=1)   # ACI rot
    doc.layers.add("CRACK_LABELS", color=3)   # ACI grün

    msp      = doc.modelspace()
    crack_id = 1

    for geo_image, geoms in entries:
        has_geo = geo_image.has_geo

        for geom in geoms:
            if geom is None:
                continue

            # Koordinaten ermitteln (LineString oder Polygon)
            if hasattr(geom, "coords"):             # LineString
                raw_coords = list(geom.coords)
                is_closed   = False
                mid_px      = raw_coords[len(raw_coords) // 2]
            elif hasattr(geom, "exterior"):         # Polygon
                raw_coords = list(geom.exterior.coords)
                is_closed   = True
                cx_px, cy_px = geom.centroid.x, geom.centroid.y
                mid_px       = (cx_px, cy_px)
            else:
                continue

            if has_geo:
                world = geo_image.coords_to_world(raw_coords)
            else:
                world = raw_coords

            pts_3d = [(float(x), float(y), 0.0) for x, y in world]

            # LWPOLYLINE (open for lines, closed for polygon fallback)
            msp.add_lwpolyline(
                pts_3d,
                dxfattribs={
                    "layer":  "CRACKS",
                    "closed": is_closed,
                    "color":  1,
                },
            )

            # Label-Position
            cx_px, cy_px = float(mid_px[0]), float(mid_px[1])
            if has_geo:
                cx_px, cy_px = geo_image.px_to_world(cx_px, cy_px)

            label_height = 0.1 if has_geo else 20

            msp.add_text(
                f"Riss {crack_id}",
                dxfattribs={
                    "layer":  "CRACK_LABELS",
                    "height": label_height,
                    "insert": (cx_px, cy_px),
                    "color":  3,
                },
            )
            crack_id += 1

    doc.saveas(output_path)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT-VORLAGEN
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_PRESETS = {
    "Strassenrisse":    "crack . pavement crack . fissure",
    "Betonrisse":       "crack . concrete crack . spalling",
    "Mauerwerk":        "crack . fracture . masonry crack",
    "Fassade":          "crack . facade crack . plaster crack",
    "Bruecke":          "crack . structural crack . fracture . spalling",
    "Benutzerdefiniert": "",
}

PROMPT_PRESET_NAMES = list(PROMPT_PRESETS.keys())
DEFAULT_PRESET = PROMPT_PRESET_NAMES[0]


# ══════════════════════════════════════════════════════════════════════════════
#  DESKTOP APP (customtkinter)
# ══════════════════════════════════════════════════════════════════════════════


import datetime as _dt


# ─── Appearance ───────────────────────────────────────────────────────────────
APP_COLORS = {
    "bg":         "#0d0d1a",
    "panel":      "#16213e",
    "accent":     "#0f3460",
    "highlight":  "#e94560",
    "hover":      "#c73652",
    "text":       "#eaeaea",
    "text_dim":   "#a0a0b0",
    "success":    "#4ade80",
    "warning":    "#fbbf24",
    "error":      "#f87171",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PROCESSING WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class Worker:
    """Runs crack detection in a background thread, sends messages via queue."""

    def __init__(self, msg_q: queue.Queue):
        self._q          = msg_q
        self._stop_event = threading.Event()

    # ── helpers ───────────────────────────────────────────────────────────────
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

    # ── main entry point ──────────────────────────────────────────────────────
    def run(
        self,
        image_paths: List[Path],
        preset: str,
        custom_prompt: str,
        confidence: float,
        min_area: int,
    ):
        self._stop_event.clear()

        # Build prompt
        if preset == "Benutzerdefiniert":
            text_prompt = custom_prompt.strip() or "crack"
        else:
            base = PROMPT_PRESETS.get(preset, "crack")
            text_prompt = f"{base} . {custom_prompt.strip()}" if custom_prompt.strip() else base

        # Output folder next to first input file
        input_dir  = image_paths[0].parent
        output_dir = input_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Output-Ordner: {output_dir}")

        total = len(image_paths)
        all_entries: List[Tuple[GeoImage, List]] = []
        t_start = time.time()

        for idx, img_path in enumerate(image_paths):
            if self._stop_event.is_set():
                self._log("Abgebrochen.")
                self._done(False)
                return

            pct_base = idx / total
            self._status(f"Lade {img_path.name} …")
            self._progress(pct_base * 0.05)
            self._log(f"\n>> {img_path.name}")

            try:
                geo_image = load_geo_image(str(img_path))
                geo_flag = ""
                if geo_image.has_geo:
                    epsg     = geo_image.crs.to_epsg() if geo_image.crs else "?"
                    geo_flag = f" | EPSG:{epsg}"
                self._log(f"   {geo_image.width}×{geo_image.height} px{geo_flag}")

                _tile_start   = time.time()
                _crack_count  = [0]

                def _tile_cb(tile_pct: float, tile_msg: str):
                    if self._stop_event.is_set():
                        return
                    if tile_pct >= 0.90:
                        overall = pct_base * 0.97 + (0.90 + (tile_pct - 0.90) * 0.7) * (0.97 / total)
                        self._status(f"{img_path.name}: {tile_msg}")
                        self._progress(overall)
                        return
                    elapsed = time.time() - _tile_start
                    if tile_pct > 0.02 and elapsed > 2:
                        eta_s  = (elapsed / tile_pct) * (1.0 - tile_pct)
                        m, s   = divmod(int(eta_s), 60)
                        h_, m  = divmod(m, 60)
                        if h_ > 0:   eta = f"~{h_}h {m:02d}m verbleibend"
                        elif m > 0:  eta = f"~{m}m {s:02d}s verbleibend"
                        else:        eta = f"~{s}s verbleibend"
                    else:
                        eta = ""
                    overall = pct_base * 0.97 + tile_pct / total * 0.90
                    msg = f"{img_path.name}: {tile_msg} | {_crack_count[0]} Risse"
                    if eta:
                        msg += f" | {eta}"
                    self._status(msg)
                    self._progress(overall)

                annotated, crack_lines, scores = process_geo_image(
                    geo_image,
                    text_prompt  = text_prompt,
                    confidence   = confidence,
                    min_area     = min_area,
                    progress_cb  = _tile_cb,
                    log_fn       = self._log,
                )

                _crack_count[0] = len(crack_lines)
                elapsed_img     = time.time() - _tile_start

                self._progress((pct_base + 1.0 / total) * 0.97)
                self._status(f"✅ {img_path.name}: {len(crack_lines)} Risse in {elapsed_img:.1f}s")
                self._log(f"   ✅ {len(crack_lines)} Risse erkannt in {elapsed_img:.1f}s")

                # ── Vorschau anzeigen ─────────────────────────────────────────
                self._preview(annotated)

                # ── Bild mit Markierungen speichern ───────────────────────────
                # `annotated` ist bereits skaliert (max 2000 px) und hat blaue
                # Risslinien eingezeichnet → direkt speichern, kein Re-Render.
                stem    = img_path.stem
                out_img = output_dir / f"{stem}_cracks.png"
                self._status(f"Speichere {out_img.name} …")
                Image.fromarray(annotated).save(str(out_img))
                self._log(f"   Bild gespeichert: {out_img.name} ({annotated.shape[1]}×{annotated.shape[0]} px)")

                all_entries.append((geo_image, crack_lines))

            except Exception as exc:
                self._log(f"   ❌ Fehler: {exc}")
                traceback.print_exc()

        # ── Export GeoJSON + DXF ─────────────────────────────────────────────
        ts           = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        geojson_path = str(output_dir / f"cracks_{ts}.geojson")
        dxf_path     = str(output_dir / f"cracks_{ts}.dxf")

        self._status("Exportiere GeoJSON …")
        self._progress(0.97)
        try:
            export_geojson(all_entries, geojson_path)
            self._log(f"   GeoJSON: cracks_{ts}.geojson")
        except Exception as exc:
            self._log(f"   ⚠ GeoJSON-Fehler: {exc}")

        self._status("Exportiere DXF …")
        self._progress(0.99)
        try:
            if HAS_EZDXF:
                export_dxf(all_entries, dxf_path)
                self._log(f"   DXF:     cracks_{ts}.dxf")
            else:
                self._log("   ⚠ DXF übersprungen (ezdxf fehlt)")
        except Exception as exc:
            self._log(f"   ⚠ DXF-Fehler: {exc}")

        total_cracks  = sum(len(g) for _, g in all_entries)
        total_elapsed = time.time() - t_start
        self._log(
            f"\n{'─'*54}\n"
            f"  {total_cracks} Risse | {total} Bild(er) | {total_elapsed:.1f}s\n"
            f"  Output: {output_dir}\n"
            f"{'─'*54}"
        )
        self._progress(1.0)
        self._status(f"✅ Fertig – {total_cracks} Risse erkannt")
        self._done(True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DESKTOP APP
# ═══════════════════════════════════════════════════════════════════════════════

class CrackDetectApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("CrackDetect – Automatische Riss-Erkennung")
        self.geometry("1050x720")
        self.minsize(860, 600)
        self.configure(fg_color=APP_COLORS["bg"])

        self._input_paths: List[Path] = []
        self._worker: Optional[Worker]          = None
        self._thread: Optional[threading.Thread] = None
        self._msg_q: queue.Queue                = queue.Queue()
        self._preview_image: Optional[ctk.CTkImage] = None

        self._build_ui()
        self._poll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = ctk.CTkFrame(self, fg_color=APP_COLORS["accent"], corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(
            hdr, text="🔍  CrackDetect",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=APP_COLORS["highlight"],
        ).pack(side="left", padx=22, pady=14)
        ctk.CTkLabel(
            hdr, text="Automatische Riss-Erkennung · SAM3 · GeoJSON & DXF",
            font=ctk.CTkFont(size=12),
            text_color=APP_COLORS["text_dim"],
        ).pack(side="left", padx=0, pady=14)

        # Main
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=18, pady=14)

        # ── Left panel ────────────────────────────────────────────────────────
        left = ctk.CTkFrame(main, fg_color=APP_COLORS["panel"], corner_radius=12, width=280)
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
                command=cmd, width=248,
            ).pack(padx=16, pady=(0, 6))

        # Input
        section("EINGABE")
        mkbtn("📂  Bild(er) auswählen", self._browse_files)
        mkbtn("📁  Ordner auswählen",   self._browse_folder)
        self._lbl_input = ctk.CTkLabel(
            left, text="Keine Auswahl",
            font=ctk.CTkFont(size=11), text_color=APP_COLORS["text_dim"],
            wraplength=240, justify="left",
        )
        self._lbl_input.pack(anchor="w", padx=16, pady=(0, 6))

        # Settings
        section("EINSTELLUNGEN")

        ctk.CTkLabel(left, text="Erkennungs-Typ",
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=16, pady=(0, 2))
        self._preset_var = ctk.StringVar(value=DEFAULT_PRESET)
        ctk.CTkOptionMenu(
            left, values=PROMPT_PRESET_NAMES,
            variable=self._preset_var,
            fg_color=APP_COLORS["accent"],
            button_color=APP_COLORS["highlight"],
            button_hover_color=APP_COLORS["hover"],
            dropdown_fg_color="#1a1a2e",
            font=ctk.CTkFont(size=13), width=248,
            command=self._on_preset_change,
        ).pack(padx=16, pady=(0, 10))

        ctk.CTkLabel(left, text="Zusätzliche Suchbegriffe",
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=16, pady=(0, 2))
        self._custom_entry = ctk.CTkEntry(
            left, placeholder_text="z.B. spalling . delamination",
            font=ctk.CTkFont(size=12), width=248,
        )
        self._custom_entry.pack(padx=16, pady=(0, 10))

        ctk.CTkLabel(left, text="Confidence (empfindlicher ↓)",
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=16, pady=(0, 2))
        self._conf_var = ctk.DoubleVar(value=SAM3_CONFIDENCE)
        self._conf_label = ctk.CTkLabel(left, text=f"{SAM3_CONFIDENCE:.2f}",
                                        font=ctk.CTkFont(size=11),
                                        text_color=APP_COLORS["text_dim"])
        self._conf_label.pack(anchor="e", padx=16)
        ctk.CTkSlider(
            left, from_=0.01, to=0.50, variable=self._conf_var,
            width=248, progress_color=APP_COLORS["highlight"],
            command=lambda v: self._conf_label.configure(text=f"{v:.2f}"),
        ).pack(padx=16, pady=(0, 10))

        ctk.CTkLabel(left, text="Min. Fläche (px²)",
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=16, pady=(0, 2))
        self._area_var = ctk.IntVar(value=SAM3_MIN_MASK_PX)
        self._area_label = ctk.CTkLabel(left, text=f"{SAM3_MIN_MASK_PX}",
                                         font=ctk.CTkFont(size=11),
                                         text_color=APP_COLORS["text_dim"])
        self._area_label.pack(anchor="e", padx=16)
        ctk.CTkSlider(
            left, from_=10, to=2000, variable=self._area_var,
            width=248, progress_color=APP_COLORS["highlight"],
            command=lambda v: self._area_label.configure(text=f"{int(v)}"),
        ).pack(padx=16, pady=(0, 14))

        # ── Right panel ───────────────────────────────────────────────────────
        right = ctk.CTkFrame(main, fg_color=APP_COLORS["panel"], corner_radius=12)
        right.pack(side="left", fill="both", expand=True)

        ctk.CTkLabel(right, text="ERGEBNIS",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=APP_COLORS["text_dim"]).pack(anchor="w", padx=16, pady=(14, 2))
        ctk.CTkFrame(right, height=1, fg_color=APP_COLORS["accent"]).pack(fill="x", padx=16, pady=(0, 6))

        # Preview image
        self._preview_label = ctk.CTkLabel(
            right, text="Kein Bild", fg_color="#0d0d1a",
            font=ctk.CTkFont(size=13), text_color=APP_COLORS["text_dim"],
            corner_radius=8,
        )
        self._preview_label.pack(fill="both", expand=True, padx=12, pady=(0, 6))

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

        self._btn_start = ctk.CTkButton(
            bar, text="▶  Risse erkennen",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=APP_COLORS["highlight"], hover_color=APP_COLORS["hover"],
            corner_radius=8, height=42, command=self._start,
        )
        self._btn_start.pack(side="left", padx=16, pady=10)

        self._btn_stop = ctk.CTkButton(
            bar, text="⏹  Abbrechen",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#374151", hover_color="#4b5563",
            corner_radius=8, height=42, command=self._stop, state="disabled",
        )
        self._btn_stop.pack(side="left", padx=(0, 16), pady=10)

        ctk.CTkButton(
            bar, text="📁  Output öffnen",
            font=ctk.CTkFont(size=13),
            fg_color="#374151", hover_color="#4b5563",
            corner_radius=8, height=42, width=150,
            command=self._open_output,
        ).pack(side="right", padx=16, pady=10)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _on_preset_change(self, _value):
        pass  # could enable/disable custom entry

    def _browse_files(self):
        paths = filedialog.askopenfilenames(
            title="Bilder auswählen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"), ("Alle", "*.*")],
        )
        if paths:
            self._input_paths = [Path(p) for p in paths if Path(p).suffix.lower() in SUPPORTED_EXT]
            n = len(self._input_paths)
            label = f"{n} Bild(er)" if n > 1 else self._input_paths[0].name
            self._lbl_input.configure(text=label)

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Ordner mit Bildern auswählen")
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
        if self._input_paths:
            out = self._input_paths[0].parent / "output"
            out.mkdir(exist_ok=True)
            os.startfile(str(out))

    def _append_log(self, text: str):
        self._log_box.configure(state="normal")
        self._log_box.insert("end", text + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _update_preview(self, img_np: np.ndarray):
        h, w = img_np.shape[:2]
        # Fit into preview label (max 600 wide, 400 tall)
        max_w, max_h = 600, 400
        scale = min(max_w / w, max_h / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img_np, (nw, nh), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        self._preview_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(nw, nh))
        self._preview_label.configure(image=self._preview_image, text="")

    # ── Queue polling ─────────────────────────────────────────────────────────
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
                elif kind == "done":
                    self._btn_start.configure(state="normal")
                    self._btn_stop.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(80, self._poll)

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def _start(self):
        if not self._input_paths:
            self._append_log("❌ Bitte zuerst Bilder oder Ordner auswählen.")
            return
        if self._thread and self._thread.is_alive():
            return

        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._lbl_status.configure(text="⏳ Starte Analyse …")
        self._progress_bar.set(0)
        self._append_log("=" * 54)
        self._append_log(f"Starte: {len(self._input_paths)} Bild(er)")

        self._worker = Worker(self._msg_q)
        self._thread = threading.Thread(
            target=self._worker.run,
            args=(
                self._input_paths,
                self._preset_var.get(),
                self._custom_entry.get(),
                self._conf_var.get(),
                int(self._area_var.get()),
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


# ═══════════════════════════════════════════════════════════════════════════════
#  EINSTIEGSPUNKT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print(" ================================================================")
    print("   CrackDetect – Automatische Riss-Erkennung")
    print("   Powered by SAM3")
    print(" ================================================================")
    print()
    print("[INFO] Lade KI-Modelle …")
    try:
        load_models()
    except Exception as e:
        print(f"  [FEHLER] {e}")
        print("  Bitte start.bat erneut ausführen.")
        input("  Enter drücken zum Beenden …")
        sys.exit(1)

    print("[INFO] Starte Desktop-App …")
    app = CrackDetectApp()
    app.mainloop()
