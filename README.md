# CrackDetect

CrackDetect ist eine lokale Desktop-App zur **automatischen Riss-Erkennung** in Bauwerks-Fotos und georeferenzierten Orthofotos. Sie laeuft komplett **offline auf dem eigenen PC** (keine Cloud, keine API-Kosten) und nutzt ein speziell trainiertes U-Net Modell fuer praezise Riss-Segmentierung.

## Features

- **Vollautomatische Erkennung**: Nutzt ein trainiertes U-Net Modell (ResNet34-Backbone) fuer direkte Pixel-genaue Riss-Segmentierung.
- **Formate**: Unterstuetzt JPG, PNG, TIFF, BMP, WebP.
- **Georeferenzierung**:
  - GeoTIFF-Orthofotos mit echten Koordinaten (CRS, Affin-Transform).
  - World-File-Unterstuetzung: JPG/PNG mit Sidecar-Dateien (`.jgw` / `.pgw` / `.tfw` + `.prj`).
- **Riss-Konturen**: Die erkannten Risse werden als geschlossene Umrandungen (Konturen) exportiert - die exakte Kante jedes Risses.
- **Two-Pass Tiling fuer grosse Bilder**: Bilder werden automatisch in 512x512 px Kacheln aufgeteilt (= Modell-Trainingsgroesse, kein Qualitaetsverlust). Overlap ist adaptiv (Minimum 30%, automatisch erhoht damit Tiles das Bild gleichmaessig aufteilen). Nach Pass 1 werden zusaetzlich zentrierte Refine-Tiles direkt ueber erkannte Risse gelegt (Pass 2), damit Risse am Kachelrand vollstaendig erfasst werden. Ueberlappende Erkennungen werden zusammengefuehrt.
- **Batch-Verarbeitung**: Komplette Ordner auf einmal verarbeiten.
- **CAD & GIS Export** - wird automatisch neben dem Eingabebild gespeichert:
  - **Annotiertes Bild** (`<name>_cracks.png`): Original mit blauen Risslinien (skaliert auf max. 2000 px).
  - **GeoJSON**: Riss-Konturen als LineStrings mit echten Weltkoordinaten (CRS) und Breite in px.
  - **DXF**: Export fuer CAD (Layer `CRACKS` mit LWPOLYLINE, Layer `CRACK_LABELS`).
- **Desktop-App**: Natives Windows-Fenster (customtkinter), kein Browser noetig.

## Installation & Start

CrackDetect ist fuer Windows konzipiert und bietet ein automatisches Setup-Skript.

1. Stelle sicher, dass **Python 3.10-3.12** und **Git** installiert sind.
2. Fuehre die Datei `start.bat` aus.
   - *Beim ersten Start:* Die virtuelle Umgebung wird erstellt, PyTorch mit CUDA sowie alle Pakete heruntergeladen (~3 GB fuer PyTorch + Abhaengigkeiten). Dieser Vorgang kann 15-40 Minuten dauern.
   - *Bei weiteren Starts:* Das Desktop-Fenster oeffnet sich direkt.

## Technologie-Stack

- **Erkennung + Segmentierung**: U-Net mit ResNet34-Backbone (PyTorch/ONNX) - trainiert auf einem proprietaeren Riss-Datensatz mit **124.796 Bildern** - direkte Pixel-Masken.
- **Kontur-Extraktion**: Polygon-Umrandung der erkannten Riss-Regionen.
- **Geo-Verarbeitung**: `rasterio` & `shapely`.
- **Export**: `ezdxf` & `json`.
- **UI**: `customtkinter` (natives Desktop-Fenster).

## Nutzung

1. Klicke auf **"Bild(er) auswaehlen"** oder **"Ordner auswaehlen"**.
2. Passe **Confidence** und **Min.-Flaeche** an.
3. Klicke auf **"Risse erkennen"**.
4. Die Ergebnisse werden automatisch im Unterordner `output/` neben dem Eingabebild gespeichert:
   - `<name>_cracks.png` - Bild mit blauen Risslinien
   - `cracks_<ts>.geojson` - Vektordaten (LineStrings)
   - `cracks_<ts>.dxf` - CAD-Export
5. Mit **"Output"** oeffnet sich der Ausgabeordner direkt im Explorer.

## Mitgeliefertes Basismodell

Das Repository enthaelt ein **vortrainiertes U-Net Modell** (`model/crack_unet.onnx`), das direkt genutzt werden kann.

- Trainiert auf **124.796 annotierten Rissbildern** aus einem proprietaeren Datensatz.
- Der Datensatz gehoert ausschliesslich dem Entwickler und wird **nicht veroeffentlicht oder weitergegeben**.
- Das Modell funktioniert sehr gut und ist sofort einsatzbereit.

Wie bei jedem KI-Modell gilt: Es erkennt am zuverlaessigsten, wofuer es trainiert wurde. Fuer andere Materialien, Kameraperspektiven oder Lichtsituationen sind eigene Bilder immer die beste Grundlage, um das Modell weiter zu verbessern.

## Fine-Tuning mit eigenen Bildern

Das Modell kann jederzeit mit eigenen Aufnahmen weiter trainiert werden, um die Erkennungsgenauigkeit fuer den jeweiligen Anwendungsfall zu steigern.

**CrackDetect erstellt die Trainingsmasken direkt mit:**

Die App exportiert bei jeder Erkennung automatisch binaere Rissmasken (`*_cracks.png`). Diese koennen direkt als Trainingsmasken fuer das Fine-Tuning verwendet werden - kein separates Annotierungstool noetig.

**So funktioniert das Fine-Tuning:**

1. Eigene Bilder mit CrackDetect analysieren - die erzeugten Masken als Trainingsgrundlage verwenden.
2. Das ONNX-Modell als PyTorch-Checkpoint laden und mit den vortrainierten Gewichten initialisieren.
3. Weitertraining auf den eigenen Daten (Transfer Learning).
4. Fertiges Modell erneut als ONNX exportieren und unter `model/crack_unet.onnx` ablegen.
5. CrackDetect startet beim naechsten Mal automatisch mit dem verbesserten Modell.
