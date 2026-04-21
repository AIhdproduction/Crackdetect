# CrackDetect

CrackDetect ist eine lokale Desktop-App zur **automatischen Riss-Erkennung** in Bauwerks-Fotos und georeferenzierten Orthofotos. Sie laeuft komplett **offline auf dem eigenen PC** (keine Cloud, keine API-Kosten) und nutzt ein speziell trainiertes U-Net Modell fuer praezise Riss-Segmentierung.

## Features

- **Vollautomatische Erkennung**: Nutzt ein trainiertes U-Net Modell (ResNet34-Backbone) fuer direkte Pixel-genaue Riss-Segmentierung.
- **Formate**: Unterstützt JPG, PNG, TIFF, BMP, WebP.
- **Georeferenzierung**:
  - GeoTIFF-Orthofotos mit echten Koordinaten (CRS, Affin-Transform).
  - World-File-Unterstützung: JPG/PNG mit Sidecar-Dateien (`.jgw` / `.pgw` / `.tfw` + `.prj`).
- **Riss-Konturen**: Die erkannten Risse werden als geschlossene Umrandungen (Konturen) exportiert – die exakte Kante jedes Risses.
- **Two-Pass Tiling fuer grosse Bilder**: Bilder werden automatisch in 512x512 px Kacheln aufgeteilt (= Modell-Trainingsgroesse, kein Qualitaetsverlust). Overlap ist adaptiv (Minimum 30%, automatisch erhoht damit Tiles das Bild gleichmaessig aufteilen). Nach Pass 1 werden zusaetzlich zentrierte Refine-Tiles direkt ueber erkannte Risse gelegt (Pass 2), damit Risse am Kachelrand vollstaendig erfasst werden. Ueberlappende Erkennungen werden zusammengefuehrt.
- **Batch-Verarbeitung**: Komplette Ordner auf einmal verarbeiten.
- **CAD & GIS Export** – wird automatisch neben dem Eingabebild gespeichert:
  - **Annotiertes Bild** (`<name>_cracks.png`): Original mit blauen Risslinien (skaliert auf max. 2000 px).
  - **GeoJSON**: Riss-Konturen als LineStrings mit echten Weltkoordinaten (CRS) und Breite in px.
  - **DXF**: Export für CAD (Layer `CRACKS` mit LWPOLYLINE, Layer `CRACK_LABELS`).
- **Desktop-App**: Natives Windows-Fenster (customtkinter), kein Browser nötig.

## Installation & Start

CrackDetect ist für Windows konzipiert und bietet ein automatisches Setup-Skript.

1. Stelle sicher, dass **Python 3.10–3.12** und **Git** installiert sind.
2. Führe die Datei `start.bat` aus.
   - *Beim ersten Start:* Die virtuelle Umgebung wird erstellt, PyTorch mit CUDA sowie alle Pakete heruntergeladen (~3 GB fuer PyTorch + Abhaengigkeiten). Dieser Vorgang kann 15–40 Minuten dauern.
   - *Bei weiteren Starts:* Das Desktop-Fenster öffnet sich direkt.

## Technologie-Stack

- **Erkennung + Segmentierung**: U-Net mit ResNet34-Backbone (PyTorch/ONNX) – trainiert auf einem proprietaeren Riss-Datensatz mit **124.796 Bildern** → direkte Pixel-Masken.
- **Kontur-Extraktion**: Polygon-Umrandung der erkannten Riss-Regionen.
- **Geo-Verarbeitung**: `rasterio` & `shapely`.
- **Export**: `ezdxf` & `json`.
- **UI**: `customtkinter` (natives Desktop-Fenster).

## Nutzung

1. Klicke auf **„Bild(er) auswaehlen"** oder **„Ordner auswaehlen"**.
2. Passe **Confidence** und **Min.-Flaeche** an.
3. Klicke auf **„Risse erkennen"**.
4. Die Ergebnisse werden automatisch im Unterordner `output/` neben dem Eingabebild gespeichert:
   - `<name>_cracks.png` – Bild mit blauen Risslinien
   - `cracks_<ts>.geojson` – Vektordaten (LineStrings)
   - `cracks_<ts>.dxf` – CAD-Export
5. Mit **"Output"** öffnet sich der Ausgabeordner direkt im Explorer.

## Mitgeliefertes Basismodell

Das Repository enthaelt ein **vortrainiertes U-Net Modell** (`model/crack_unet.onnx`), das direkt als Basismodell fuer eigene Anwendungen genutzt werden kann.

**Trainingsdaten:**
- Trainiert auf **124.796 annotierten Rissbildern** aus einem proprietaeren Datensatz.
- Der Datensatz gehoert ausschliesslich dem Entwickler und wird **nicht veroeffentlicht oder weitergegeben**.

**Als Ausgangsbasis verwenden (Fine-Tuning):**

Das mitgelieferte Modell liefert bereits solide Ergebnisse und kann als Startpunkt fuer eigenes Fine-Tuning genutzt werden – besonders wenn die eigenen Aufnahmesituationen sich vom trainierten Kontext unterscheiden:

1. Lade das ONNX-Modell in deine Trainingsumgebung und exportiere es als PyTorch-Checkpoint.
2. Initialisiere das U-Net mit den vortrainierten Gewichten (ResNet34 Encoder).
3. Trainiere weiter auf deinen eigenen annotierten Bildern (Transfer Learning).
4. Exportiere das fertige Modell erneut als ONNX und lege es unter `model/crack_unet.onnx` ab.

## Eigenes Modell von Grund auf trainieren

Alternativ kann ein komplett neues Modell trainiert werden:

1. **Datensatz zusammenstellen** – Bilder + binaere Masken (weiss = Riss, schwarz = Hintergrund) in `images/` und `masks/` Ordner sortieren.
2. **U-Net trainieren** – z. B. mit [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) (ResNet34 Encoder, DiceFocalLoss).
3. **Als ONNX exportieren** – `torch.onnx.export()` mit Eingabegroesse 512x512.
4. **Modell ablegen** – die fertige `.onnx` Datei muss hier liegen:
   ```
   model\crack_unet.onnx
   ```
5. **App starten** – CrackDetect erkennt das Modell beim naechsten Start automatisch.

**Eigene Daten liefern die besten Ergebnisse:**

Ein Modell erkennt nur zuverlaessig, wofuer es trainiert wurde. Deshalb gilt: Sammle eigene Fotos von genau dem Material und dem Blickwinkel, den du in der Praxis verwendest, und annotiere diese manuell. Bereits 200-500 sauber annotierte Bilder von deiner spezifischen Aufnahmesituation uebertreffen in der Regel grosse oeffentliche Datensaetze aus anderen Kontexten.

Oeffentliche Datensaetze koennen als **Ausgangsbasis oder Ergaenzung** dienen, sollten aber nicht der einzige Trainingsinput sein:

| Datensatz | Quelle | Inhalt |
|---|---|---|
| Crack500 | [GitHub](https://github.com/fyangneil/pavement-crack-detection) | Asphalt-Strassenrisse (Bodenperspektive) |
| khanhha/crack_segmentation | [GitHub](https://github.com/khanhha/crack_segmentation) | 11.200 Bilder aus 12 verschiedenen Quellen |
| UAV Crack Dataset | [Google Drive](https://drive.google.com/open?id=1RMf0GYXn7Mq1s9STGFG5iByavTr05SjF) | 11.298 Drohnen-/Luftaufnahmen mit Masken |
| SDNET2018 | [Kaggle](https://www.kaggle.com/datasets/anitarostami/structural-defects-network-sdnet-2018) | Bruecken, Waende, Pflaster (Nahaufnahmen) |
