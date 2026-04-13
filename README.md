# CrackDetect 🔍

CrackDetect ist eine lokale Desktop-App zur **automatischen Riss-Erkennung** in Bauwerks-Fotos und georeferenzierten Orthofotos. Sie läuft komplett **offline auf dem eigenen PC** (keine Cloud, keine API-Kosten) und nutzt modernste Open-Source KI-Modelle.

## Features ✨

- **Vollautomatische Erkennung**: Nutzt Grounding DINO + SAM2.1 für automatisches Erkennen von Rissen ohne manuelle Maskierung.
- **Formate**: Unterstützt JPG, PNG, TIFF, BMP, WebP.
- **Georeferenzierung**: 
  - GeoTIFF-Orthofotos mit echten Koordinaten (CRS, Affin-Transform).
  - World-File-Unterstützung: JPG/PNG mit Sidecar-Dateien (`.jgw` / `.pgw` / `.tfw` + `.prj`).
- **Tiling für große Bilder**: Bilder >2500 px werden automatisch gekachelt (1024×1024 px mit 256 px Überlappung) und Duplikate per NMS (Non-Maximum Suppression) entfernt.
- **Batch-Verarbeitung**: Komplette Ordner mit Bildern oder gekachelten Orthofotos auf einmal verarbeiten.
- **CAD & GIS Export**: 
  - **GeoJSON**: Export mit echten Weltkoordinaten (CRS), Fläche in px² und Dateiname.
  - **DXF**: Export für CAD-Programme (Layer "CRACKS" mit Polygonen, Layer "CRACK_LABELS" mit Nummerierung).
- **Lokale Web-UI**: Einfache Benutzeroberfläche über Gradio (öffnet im Browser).

## Installation & Start 🚀

CrackDetect ist für Windows konzipiert und bietet ein automatisches Setup-Skript.

1. Stelle sicher, dass **Python (3.10 oder 3.11)** und **Git** installiert sind.
2. Für die Grounding DINO Kompilierung werden die **Visual Studio C++ Build Tools** benötigt.
3. Führe die Datei `start.bat` aus.
   - *Beim ersten Start:* Die virtuelle Umgebung wird erstellt, PyTorch mit CUDA (ca. 3 GB) sowie alle Modelle und Pakete heruntergeladen (~1.6 GB Checkpoints). Dieser Vorgang kann 15-40 Minuten dauern. Bitte das Fenster nicht schließen!
   - *Bei weiteren Starts:* Das Tool öffnet sich direkt im Browser unter `http://127.0.0.1:7861`.

## Technologie-Stack 🛠️

- **Erkennung (Detection)**: Grounding DINO (`groundingdino-swint-ogc`) - Text-Prompt „crack“ → automatische Bounding-Boxes.
- **Segmentierung**: SAM 2.1 Hiera Large (`sam2.1_hiera_large.pt`) - Präzise Pixel-Masken aus Bounding-Boxes.
- **Geo-Verarbeitung**: `rasterio` & `shapely`.
- **Export**: `ezdxf` & `json`.
- **UI**: `gradio`.

## Nutzung 💡

1. Lade ein Bild hoch oder gib den Pfad zu einem Ordner/GeoTIFF ein.
2. (Optional) Passe die Schwellwerte für Box- & Text-Threshold sowie die Mindestfläche an.
3. Klicke auf "Risse erkennen".
4. Lade die generierten `.geojson` und `.dxf` Dateien herunter, die im Ordner `output/` gespeichert werden.