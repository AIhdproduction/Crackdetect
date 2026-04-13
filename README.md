# CrackDetect

CrackDetect ist eine lokale Desktop-App zur **automatischen Riss-Erkennung** in Bauwerks-Fotos und georeferenzierten Orthofotos. Sie läuft komplett **offline auf dem eigenen PC** (keine Cloud, keine API-Kosten) und nutzt modernste Open-Source KI-Modelle.

## Features

- **Vollautomatische Erkennung**: Nutzt SAM3 für automatisches Erkennen von Rissen via Text-Prompt – ohne manuelle Maskierung.
- **Formate**: Unterstützt JPG, PNG, TIFF, BMP, WebP.
- **Georeferenzierung**:
  - GeoTIFF-Orthofotos mit echten Koordinaten (CRS, Affin-Transform).
  - World-File-Unterstützung: JPG/PNG mit Sidecar-Dateien (`.jgw` / `.pgw` / `.tfw` + `.prj`).
- **Skelettlinien statt Flächen**: Riss-Masken werden zu 1-px-breiten Mittellinien reduziert – eine Linie pro Riss-Kante, keine überlappenden Flächen.
- **Tiling für große Bilder**: Bilder >2500 px werden automatisch gekachelt (1024×1024 px, 256 px Überlappung). Überlappende Erkennungen werden zusammengeführt.
- **Batch-Verarbeitung**: Komplette Ordner auf einmal verarbeiten.
- **CAD & GIS Export** – wird automatisch neben dem Eingabebild gespeichert:
  - **Annotiertes Bild** (`<name>_cracks.png`): Original mit blauen Risslinien (skaliert auf max. 2000 px).
  - **GeoJSON**: LineStrings mit echten Weltkoordinaten (CRS) und Länge in px.
  - **DXF**: Export für CAD (Layer `CRACKS` mit LWPOLYLINE, Layer `CRACK_LABELS`).
- **Desktop-App**: Natives Windows-Fenster (customtkinter), kein Browser nötig.

## Installation & Start

CrackDetect ist für Windows konzipiert und bietet ein automatisches Setup-Skript.

1. Stelle sicher, dass **Python 3.10–3.12** und **Git** installiert sind.
2. Führe die Datei `start.bat` aus.
   - *Beim ersten Start:* Die virtuelle Umgebung wird erstellt, PyTorch mit CUDA sowie alle Pakete heruntergeladen (~5 GB Checkpoints). Dieser Vorgang kann 15–40 Minuten dauern.
   - *Bei weiteren Starts:* Das Desktop-Fenster öffnet sich direkt.

## Technologie-Stack

- **Erkennung + Segmentierung**: SAM3 (Meta) – Text-Prompt `"crack"` → direkte Pixel-Masken.
- **Skelettierung**: `scikit-image skeletonize` → 1-px-Mittellinie pro Riss.
- **Geo-Verarbeitung**: `rasterio` & `shapely`.
- **Export**: `ezdxf` & `json`.
- **UI**: `customtkinter` (natives Desktop-Fenster).

## Nutzung

1. Klicke auf **„Bild(er) auswählen"** oder **„Ordner auswählen"**.
2. Wähle den Erkennungs-Typ (Strassenrisse, Betonrisse, …) und passe Confidence / Min.-Fläche an.
3. Klicke auf **„▶ Risse erkennen"**.
4. Die Ergebnisse werden automatisch im Ordner `output/` neben dem Eingabebild gespeichert:
   - `<name>_cracks.png` – Bild mit blauen Risslinien
   - `cracks_<ts>.geojson` – Vektordaten (LineStrings)
   - `cracks_<ts>.dxf` – CAD-Export
5. Mit **„📁 Output öffnen"** öffnet sich der Ausgabeordner direkt im Explorer.
