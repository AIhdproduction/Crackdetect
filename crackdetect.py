"""
CrackDetect – Automatische Riss-Erkennung in Bildern und Orthofotos
====================================================================
Powered by Grounded-SAM2 (Grounding DINO + SAM2.1)

Pipeline:
  1. Bild laden (JPG/PNG/TIFF – mit rasterio falls GeoTIFF)
  2. Automatisches Tiling für grosse Bilder (>2500 px)
  3. Grounding DINO findet Bounding-Boxes via Text-Prompt "crack"
  4. SAM2 segmentiert präzise Masken aus den Boxes
  5. Masken werden zu Shapely-Polygonen vektorisiert
  6. NMS entfernt Duplikate aus überlappenden Kacheln
  7. Pixel-Koordinaten → Weltkoordinaten (falls GeoTIFF mit CRS)
  8. Export als GeoJSON + DXF

Für Details: siehe projekt.md
"""

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import cv2
from PIL import Image
import torch
import gradio as gr

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

# ─── Modell-Imports ────────────────────────────────────────────────────────────
try:
    from groundingdino.util.inference import load_model as _gdino_load, predict as _gdino_predict
    import torchvision.transforms as T
    HAS_GDINO = True
except ImportError:
    HAS_GDINO = False
    print("[ERROR] GroundingDINO nicht installiert! Bitte start.bat erneut ausführen.")

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False
    print("[ERROR] SAM2 nicht installiert! Bitte start.bat erneut ausführen.")

# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR         = Path(__file__).parent
CHECKPOINTS_DIR  = BASE_DIR / "checkpoints"
OUTPUT_DIR       = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

GDINO_CONFIG  = CHECKPOINTS_DIR / "GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = CHECKPOINTS_DIR / "groundingdino_swint_ogc.pth"
SAM2_WEIGHTS  = CHECKPOINTS_DIR / "sam2.1_hiera_large.pt"
SAM2_CONFIG   = "configs/sam2.1/sam2.1_hiera_l.yaml"   # im sam2-Paket enthalten

TILE_SIZE        = 1024   # Kachelgrösse in Pixeln
TILE_OVERLAP     = 256    # Überlappung zwischen Kacheln
MAX_DIRECT_SIZE  = 2500   # Über diesem Wert wird automatisch getiled

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

# Globale Modell-Instanzen (einmalig beim Start geladen)
_gdino_model    = None
_sam2_predictor = None

# Grounding DINO Bild-Transform
_gdino_transform = None


# ══════════════════════════════════════════════════════════════════════════════
#  MODELLE LADEN
# ══════════════════════════════════════════════════════════════════════════════

def load_models() -> None:
    """Lädt Grounding DINO und SAM2 einmalig. Wird beim App-Start aufgerufen."""
    global _gdino_model, _sam2_predictor, _gdino_transform

    if not HAS_GDINO:
        raise RuntimeError("GroundingDINO nicht installiert – start.bat erneut ausführen.")
    if not HAS_SAM2:
        raise RuntimeError("SAM2 nicht installiert – start.bat erneut ausführen.")
    if not GDINO_WEIGHTS.exists():
        raise FileNotFoundError(f"Checkpoint fehlt: {GDINO_WEIGHTS}\nstart.bat erneut ausführen.")
    if not SAM2_WEIGHTS.exists():
        raise FileNotFoundError(f"Checkpoint fehlt: {SAM2_WEIGHTS}\nstart.bat erneut ausführen.")

    print(f"[INFO] Verwende Device: {DEVICE.upper()}")
    print("[INFO] Lade Grounding DINO ...")
    _gdino_model = _gdino_load(str(GDINO_CONFIG), str(GDINO_WEIGHTS))
    _gdino_model = _gdino_model.to(DEVICE)
    _gdino_model.eval()
    print("[OK]   Grounding DINO bereit.")

    print("[INFO] Lade SAM2.1 Hiera Large ...")
    _sam2 = build_sam2(SAM2_CONFIG, str(SAM2_WEIGHTS), device=DEVICE)
    _sam2_predictor = SAM2ImagePredictor(_sam2)
    print("[OK]   SAM2 bereit.")

    _gdino_transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"[OK]   Alle Modelle geladen ({DEVICE.upper()}).")


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
#  ERKENNUNG (GROUNDING DINO + SAM2)
# ══════════════════════════════════════════════════════════════════════════════

def detect_in_tile(
    tile_np: np.ndarray,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Erkennt Risse in einem einzelnen Tile.

    Returns:
        masks  – Liste von bool-Masken (H×W)
        scores – Konfidenz-Scores (von Grounding DINO)
    """
    global _gdino_model, _sam2_predictor, _gdino_transform

    h, w = tile_np.shape[:2]
    pil_tile = Image.fromarray(tile_np)

    # ── Grounding DINO: Bounding-Boxes finden ─────────────────────────────────
    # predict() erwartet 3D-Tensor (C,H,W), NICHT 4D-Batch
    img_tensor = _gdino_transform(pil_tile).to(DEVICE)

    with torch.no_grad():
        boxes, logits, _ = _gdino_predict(
            model=_gdino_model,
            image=img_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=DEVICE,
        )

    if boxes is None or len(boxes) == 0:
        return [], []

    # Boxes sind in [0,1] normalisiert (cx, cy, bw, bh) → absolute xyxy
    scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
    boxes_abs = boxes * scale

    x1 = boxes_abs[:, 0] - boxes_abs[:, 2] / 2
    y1 = boxes_abs[:, 1] - boxes_abs[:, 3] / 2
    x2 = boxes_abs[:, 0] + boxes_abs[:, 2] / 2
    y2 = boxes_abs[:, 1] + boxes_abs[:, 3] / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1).cpu().numpy()
    scores     = logits.cpu().numpy().tolist()

    # ── SAM2: Segmentierung aus Boxes ─────────────────────────────────────────
    _sam2_predictor.set_image(tile_np)

    masks_out  = []
    scores_out = []

    for box, score in zip(boxes_xyxy, scores):
        sam_masks, sam_scores, _ = _sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],   # SAM2 erwartet (1,4)
            multimask_output=False,
        )
        if sam_masks is not None and len(sam_masks) > 0:
            masks_out.append(sam_masks[0].astype(bool))
            scores_out.append(float(score))

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
    """IoU-basierte NMS – entfernt stark überlappende Duplikate aus Kachel-Overlap.
    
    Returns:
        keep_polys  – gefilterte Polygone
        keep_scores – zugehörige Scores (gleiche Reihenfolge)
    """
    if not polygons:
        return [], []

    # Größte zuerst, Scores mitsortieren
    paired = sorted(zip(polygons, scores), key=lambda ps: ps[0].area, reverse=True)
    keep_polys:  List = []
    keep_scores: List[float] = []

    for poly, score in paired:
        dominated = False
        for kept in keep_polys:
            try:
                inter = poly.intersection(kept).area
                union = poly.union(kept).area
                if union > 0 and inter / union > iou_threshold:
                    dominated = True
                    break
            except Exception:
                pass
        if not dominated:
            keep_polys.append(poly)
            keep_scores.append(score)

    return keep_polys, keep_scores


# ══════════════════════════════════════════════════════════════════════════════
#  HAUPT-PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_geo_image(
    geo_image: GeoImage,
    text_prompt: str     = "crack",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    min_area: int        = 100,
    progress_cb=None,
) -> Tuple[np.ndarray, List, List[float]]:
    """
    Verarbeitet ein GeoImage komplett (mit automatischem Tiling).

    Returns:
        annotated_np  – Bild mit rotem Overlay (H×W×3 uint8)
        polygons      – Liste von Shapely-Polygonen (Pixel-Koordinaten)
        scores        – Konfidenz-Scores
    """
    img  = geo_image.image
    h, w = img.shape[:2]

    all_polygons: List = []
    all_scores:  List[float] = []

    needs_tiling = max(w, h) > MAX_DIRECT_SIZE

    if needs_tiling:
        tiles = compute_tiles(w, h)
        print(f"[INFO] Tiling: {len(tiles)} Kacheln für {w}×{h} px")

        for i, (x0, y0, x1, y1) in enumerate(tiles):
            if progress_cb:
                progress_cb(i / len(tiles), f"Kachel {i+1}/{len(tiles)}")
            tile = img[y0:y1, x0:x1]
            masks, scores = detect_in_tile(tile, text_prompt, box_threshold, text_threshold)
            for mask, score in zip(masks, scores):
                poly = mask_to_polygon(mask, offset_x=x0, offset_y=y0, min_area=min_area)
                if poly is not None:
                    all_polygons.append(poly)
                    all_scores.append(score)
    else:
        print(f"[INFO] Direkt: {w}×{h} px")
        if progress_cb:
            progress_cb(0.2, "Erkenne Risse ...")
        masks, scores = detect_in_tile(img, text_prompt, box_threshold, text_threshold)
        for mask, score in zip(masks, scores):
            poly = mask_to_polygon(mask, min_area=min_area)
            if poly is not None:
                all_polygons.append(poly)
                all_scores.append(score)

    # NMS nach Tiling
    if HAS_SHAPELY and len(all_polygons) > 1:
        before = len(all_polygons)
        all_polygons, all_scores = nms_polygons(all_polygons, all_scores)
        removed = before - len(all_polygons)
        if removed > 0:
            print(f"[INFO] NMS: {removed} Duplikate entfernt → {len(all_polygons)} Risse")

    # Annotiertes Vorschau-Bild erstellen
    annotated = img.copy()
    overlay   = img.copy()

    for poly in all_polygons:
        if HAS_SHAPELY:
            try:
                coords = np.array(list(poly.exterior.coords), dtype=np.int32)
                cv2.fillPoly(overlay, [coords], color=(220, 50, 50))
                cv2.polylines(annotated, [coords], True, (255, 0, 0), 2)
            except Exception:
                pass

    # Alpha-Blend: 35% Overlay-Farbe, 65% Original
    annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

    return annotated, all_polygons, all_scores


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_geojson(
    entries: List[Tuple[GeoImage, List]],
    output_path: str,
) -> str:
    """
    Exportiert Polygone aller verarbeiteten Bilder als GeoJSON.
    Koordinaten: Weltkoordinaten wenn GeoTIFF, sonst Pixel-Koordinaten.
    """
    if not HAS_SHAPELY:
        raise RuntimeError("shapely nicht installiert")

    features = []
    for geo_image, polygons in entries:
        for i, poly in enumerate(polygons):
            if geo_image.has_geo:
                wc = geo_image.coords_to_world(list(poly.exterior.coords))
                export_poly = Polygon(wc)
            else:
                export_poly = poly

            feature = {
                "type": "Feature",
                "geometry": mapping(export_poly),
                "properties": {
                    "id":          len(features) + 1,
                    "source_file": Path(geo_image.source_path).name,
                    "area_px":     round(poly.area, 2),
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

    for geo_image, polygons in entries:
        has_geo = geo_image.has_geo

        for poly in polygons:
            # Koordinaten
            if has_geo:
                world = geo_image.coords_to_world(list(poly.exterior.coords))
            else:
                world = list(poly.exterior.coords)

            pts_3d = [(float(x), float(y), 0.0) for x, y in world]

            # LWPOLYLINE
            msp.add_lwpolyline(
                pts_3d,
                dxfattribs={
                    "layer":  "CRACKS",
                    "closed": True,
                    "color":  1,
                },
            )

            # Label am Schwerpunkt
            cx, cy = poly.centroid.x, poly.centroid.y
            if has_geo:
                cx, cy = geo_image.px_to_world(cx, cy)

            label_height = 0.1 if has_geo else 20

            msp.add_text(
                f"Riss {crack_id}",
                dxfattribs={
                    "layer":  "CRACK_LABELS",
                    "height": label_height,
                    "insert": (float(cx), float(cy)),
                    "color":  3,
                },
            )
            crack_id += 1

    doc.saveas(output_path)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

def run_detection(
    uploaded_files,
    folder_path_text: str,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    min_area: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Gradio-Callback: verarbeitet Eingaben, gibt Ergebnis zurück."""

    # ── Dateipfade sammeln ────────────────────────────────────────────────────
    image_paths: List[Path] = []

    if uploaded_files:
        for f in (uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]):
            p = Path(f) if isinstance(f, str) else Path(f.name)
            if p.suffix.lower() in SUPPORTED_EXT:
                image_paths.append(p)
    
    if not image_paths and folder_path_text and folder_path_text.strip():
        folder = Path(folder_path_text.strip())
        if folder.is_file() and folder.suffix.lower() in SUPPORTED_EXT:
            image_paths = [folder]
        elif folder.is_dir():
            for ext in SUPPORTED_EXT:
                image_paths.extend(folder.glob(f"*{ext}"))
                image_paths.extend(folder.glob(f"*{ext.upper()}"))
            image_paths = sorted(set(image_paths))

    if not image_paths:
        return None, "❌ Keine Bilder gefunden. Bitte Datei hochladen oder Pfad eingeben.", None, None

    # ── Verarbeitung ──────────────────────────────────────────────────────────
    entries: List[Tuple[GeoImage, List]] = []
    status_lines = []
    last_annotated = None

    for idx, img_path in enumerate(image_paths):
        try:
            progress((idx / len(image_paths)) * 0.9, f"Verarbeite {img_path.name} ...")
            geo_image = load_geo_image(str(img_path))

            annotated, polygons, scores = process_geo_image(
                geo_image,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                min_area=min_area,
                progress_cb=None,
            )

            entries.append((geo_image, polygons))
            last_annotated = annotated

            geo_flag = ""
            if geo_image.has_geo:
                epsg = geo_image.crs.to_epsg() if geo_image.crs else "?"
                geo_flag = f" | GeoTIFF EPSG:{epsg}"

            status_lines.append(
                f"✅ {img_path.name}: {len(polygons)} Risse erkannt"
                f" | {geo_image.width}×{geo_image.height} px{geo_flag}"
            )

        except Exception as e:
            status_lines.append(f"❌ {img_path.name}: {e}")
            traceback.print_exc()

    total_cracks = sum(len(p) for _, p in entries)

    if total_cracks == 0:
        status_lines.append("⚠️ Keine Risse erkannt. Schwellwerte anpassen oder Bild-Qualität prüfen.")
        return last_annotated, "\n".join(status_lines), None, None

    # ── Export ────────────────────────────────────────────────────────────────
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    geojson_path = str(OUTPUT_DIR / f"cracks_{ts}.geojson")
    dxf_path     = str(OUTPUT_DIR / f"cracks_{ts}.dxf")

    progress(0.92, "Exportiere GeoJSON ...")
    try:
        export_geojson(entries, geojson_path)
        status_lines.append(f"📄 GeoJSON: cracks_{ts}.geojson ({total_cracks} Features)")
    except Exception as e:
        status_lines.append(f"⚠️ GeoJSON-Export fehlgeschlagen: {e}")
        geojson_path = None

    progress(0.96, "Exportiere DXF ...")
    try:
        if HAS_EZDXF:
            export_dxf(entries, dxf_path)
            status_lines.append(f"📐 DXF: cracks_{ts}.dxf ({total_cracks} Polygone)")
        else:
            status_lines.append("⚠️ DXF-Export übersprungen (ezdxf nicht installiert)")
            dxf_path = None
    except Exception as e:
        status_lines.append(f"⚠️ DXF-Export fehlgeschlagen: {e}")
        dxf_path = None

    progress(1.0, "Fertig!")
    return last_annotated, "\n".join(status_lines), geojson_path, dxf_path


def build_ui() -> gr.Blocks:
    """Baut die Gradio-Oberfläche."""

    css = """
    .gradio-container { max-width: 1300px; }
    #status-box textarea { font-family: monospace; font-size: 12px; line-height: 1.6; }
    #run-btn { font-size: 16px; }
    """

    with gr.Blocks(title="CrackDetect", theme=gr.themes.Soft(), css=css) as app:

        gr.Markdown(
            "# 🔍 CrackDetect\n"
            "**Automatische Riss-Erkennung** in Fotos und georeferenzierten Orthofotos  \n"
            "_Grounding DINO + SAM2.1 · Export als GeoJSON & DXF_"
        )

        with gr.Row():

            # ── Linke Spalte: Eingabe ─────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):

                gr.Markdown("### Eingabe")

                uploaded = gr.File(
                    label="Bild(er) hochladen",
                    file_types=[".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"],
                    file_count="multiple",
                    type="filepath",
                )

                folder_path = gr.Textbox(
                    label="Oder Pfad zu Datei / Ordner eingeben",
                    placeholder=r"C:\Orthofotos\baustelle.tif  oder  C:\Orthofotos\kacheln",
                    info="Einzelbild, GeoTIFF, oder Ordner mit gekachelten Orthofotos. "
                         "Unterstützt auch JPG/PNG mit World-File (.jgw/.pgw) + .prj",
                )

                gr.Markdown("---")
                gr.Markdown("### Erkennungs-Einstellungen")

                text_prompt = gr.Textbox(
                    value="crack",
                    label="Text-Prompt",
                    info="Was erkannt wird. Mehrere Begriffe mit ' . ' trennen z.B. 'crack . fracture'",
                )

                box_threshold = gr.Slider(
                    0.1, 0.9, value=0.3, step=0.05,
                    label="Box-Schwellwert",
                    info="↑ Weniger aber sicherere Boxen  |  ↓ Mehr, auch unsichere",
                )

                text_threshold = gr.Slider(
                    0.1, 0.9, value=0.25, step=0.05,
                    label="Text-Schwellwert",
                )

                min_area = gr.Slider(
                    10, 5000, value=100, step=10,
                    label="Mindestfläche (px²)",
                    info="Risse kleiner als dieser Wert werden ignoriert",
                )

                run_btn = gr.Button(
                    "🔍  Risse erkennen",
                    variant="primary",
                    size="lg",
                    elem_id="run-btn",
                )

                gr.Markdown(
                    "---\n"
                    "**Tiling:** Bilder >2500 px werden automatisch in  \n"
                    "1024×1024 px Kacheln mit 256 px Überlappung aufgeteilt.  \n\n"
                    "**Ordner-Batch:** Ordner mit gekachelten Orthofotos  \n"
                    "(z.B. aus Pix4D/Agisoft) werden Kachel für Kachel verarbeitet.  \n\n"
                    "**Geo-Formate:** GeoTIFF, JPG/PNG mit World-File (.jgw/.pgw) + .prj  \n"
                    "Export-Dateien liegen im `output/` Ordner."
                )

            # ── Rechte Spalte: Ausgabe ────────────────────────────────────────
            with gr.Column(scale=2):

                gr.Markdown("### Ergebnis")

                output_image = gr.Image(
                    label="Erkannte Risse (rot markiert)",
                    type="numpy",
                    height=580,
                    show_download_button=True,
                )

                status_box = gr.Textbox(
                    label="Status & Protokoll",
                    lines=7,
                    interactive=False,
                    elem_id="status-box",
                )

                with gr.Row():
                    geojson_dl = gr.File(label="📄 GeoJSON herunterladen", interactive=False)
                    dxf_dl     = gr.File(label="📐 DXF herunterladen",     interactive=False)

        run_btn.click(
            fn=run_detection,
            inputs=[uploaded, folder_path, text_prompt, box_threshold, text_threshold, min_area],
            outputs=[output_image, status_box, geojson_dl, dxf_dl],
        )

    return app


# ══════════════════════════════════════════════════════════════════════════════
#  EINSTIEGSPUNKT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print(" ================================================================")
    print("   CrackDetect – Automatische Riss-Erkennung")
    print("   Powered by Grounded-SAM2 (Grounding DINO + SAM2.1)")
    print(" ================================================================")
    print()

    print("[INFO] Lade KI-Modelle ...")
    try:
        load_models()
    except Exception as e:
        print()
        print(f"  [FEHLER] Modell-Laden fehlgeschlagen:")
        print(f"  {e}")
        print()
        print("  Bitte start.bat erneut ausführen um Modelle zu installieren.")
        print()
        input("  Enter drücken zum Beenden ...")
        sys.exit(1)

    print()
    print("[INFO] Starte Gradio UI auf http://127.0.0.1:7861 ...")
    print()

    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        show_error=True,
        quiet=False,
    )
