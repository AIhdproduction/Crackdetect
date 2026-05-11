# CrackDetect - Risk Management Tool

> **Important notice:** CrackDetect is a **risk management aid**, not a certified inspection or crack detection system. The results show what the underlying model has learned so far and where it has identified potential anomalies. They are intended to help prioritize areas for further review - not to replace professional structural assessment.

CrackDetect is a local desktop application that **marks anomalies and potential damage patterns** in structure photos and georeferenced orthophotos. It runs completely **offline on your own PC** (no cloud, no API costs) and uses a specially trained U-Net model to visually highlight areas that may require closer inspection.

---

## What this tool is - and what it is not

CrackDetect **does not perform certified crack detection**. It is designed as a **risk management tool** to:

- Visualize where the model has identified potential surface anomalies
- Give a first orientation about which areas deserve manual follow-up
- Help track model learning progress: what is already detected reliably, and where gaps remain
- Support field teams in prioritizing inspection routes

The output is **not a structural assessment**. Every flagged area must be reviewed by a qualified person on-site. False positives and missed detections are expected - especially in conditions the model has not yet seen enough of.

---

## Features

- **Anomaly visualization**: Uses a trained U-Net model (ResNet34 backbone) to highlight potential problem areas at pixel level.
- **Formats**: Supports JPG, PNG, TIFF, BMP, WebP.
- **Georeferencing**:
  - GeoTIFF orthophotos with real coordinates (CRS, affine transform).
  - World file support: JPG/PNG with sidecar files (`.jgw` / `.pgw` / `.tfw` + `.prj`).
- **Contour export**: Detected areas are exported as closed contours - the precise boundary of each flagged region.
- **Two-pass tiling for large images**: Images are split into 512x512 px tiles (the model training size, no quality loss). Overlap is adaptive (minimum 30%, automatically increased for even coverage). After pass 1, additional centered refinement tiles are placed directly over detected areas (pass 2) to catch anomalies near tile edges. Overlapping detections are merged.
- **Batch processing**: Process entire folders at once.
- **CAD and GIS export** - saved automatically alongside the input image:
  - **Annotated image** (`<name>_cracks.png`): Original with blue marked areas (scaled to max. 2000 px).
  - **GeoJSON**: Contours as LineStrings with real world coordinates (CRS) and width in px.
  - **DXF**: Export for CAD (layer `CRACKS` with LWPOLYLINE, layer `CRACK_LABELS`).
- **Desktop app**: Native Windows window (customtkinter), no browser needed.

---

## Example results

Original image:

![Original image](assets/Fine_crack_orig.jpg)

Marked anomaly lines (blue):

![Anomaly detection result](assets/Fine_crack_cracks.png)

Binary anomaly mask (model segmentation output):

![Anomaly segmentation mask](assets/Fine_crack_mask.png)

> The results shown were produced with the following settings, which allowed the model to reliably flag very fine surface anomalies:
> - **Contour smoothing**: 0.0010 (very low = maximum detail)
> - **Tile size**: 512 px (matches model training size, best result)
> - **Fine cracks (multi-scale)**: enabled
>
> The image used is **not part of the training data** and is shown purely to illustrate what the model currently detects.

---

## Installation and startup

CrackDetect is designed for Windows and comes with an automatic setup script.

1. Make sure **Python 3.10-3.12** and **Git** are installed.
2. Run the file `start.bat`.
   - *First run:* The virtual environment is created, PyTorch with CUDA and all packages are downloaded (~3 GB for PyTorch + dependencies). This can take 15-40 minutes.
   - *Subsequent runs:* The desktop window opens directly.

---

## Technology stack

- **Segmentation**: U-Net with ResNet34 backbone (PyTorch/ONNX) - trained on a proprietary annotated dataset with **124,796 images** - direct pixel masks.
- **Contour extraction**: Polygon boundary of detected anomaly regions.
- **Geo processing**: `rasterio` and `shapely`.
- **Export**: `ezdxf` and `json`.
- **UI**: `customtkinter` (native desktop window).

---

## Usage

1. Click **"Select image(s)"** or **"Select folder"**.
2. Adjust **confidence** and **minimum area**.
3. Click **"Detect anomalies"**.
4. Results are saved automatically in the `output/` subfolder next to the input image:
   - `<name>_cracks.png` - image with marked areas
   - `cracks_<ts>.geojson` - vector data (LineStrings)
   - `cracks_<ts>.dxf` - CAD export
5. Use **"Output"** to open the output folder directly in Explorer.

---

## Included base model

The repository includes a **pre-trained U-Net model** in two formats:

| File | Purpose |
|---|---|
| `model/crack_unet.onnx` | Ready to use in CrackDetect (inference) |
| `model/best_model.pth` | PyTorch checkpoint for fine-tuning |

- Trained on **124,796 annotated images** from a proprietary dataset.
- The dataset belongs exclusively to the developer and is **not published or shared**.
- The model works well as a starting point and is immediately usable.

As with any AI model: it performs most reliably on what it was trained for. For other materials, camera angles, or lighting conditions, adding your own images is the best way to improve coverage.

---

## Understanding the model's current state

One of the core use cases of CrackDetect is to **see what the model has already learned** and where it still struggles. Running the tool on a set of representative images gives a practical picture of:

- Which damage types and scales it reliably flags
- Where it tends to produce false positives (marking areas that are not damage)
- Where it misses anomalies entirely (blind spots)

This insight is the basis for targeted fine-tuning. The goal is not a perfect result from day one, but a tool that continuously improves as more real-world data is added.

---

## Fine-tuning with your own images

The model can be further trained with your own photos. This way it learns to recognize exactly what matters in your specific use case - whether fine hairline anomalies, wider damage areas, or particular materials.

### Step 1 - Generate mask automatically

Run your image through CrackDetect. The app automatically creates two files in the `output/` folder:
- `<name>_cracks.png` - original image with marked areas
- `<name>_mask.png` - binary anomaly mask (white = flagged, black = background)

### Step 2 - Correct the mask manually (GIMP or Photoshop)

Open the generated mask in GIMP or Photoshop and adjust it until it precisely shows what the model should detect in the future:
- Areas the model missed: **paint white**
- Areas incorrectly flagged: **paint black**
- Fine anomalies that should be highlighted more clearly: adjust brush hardness and size

> The mask must always remain purely binary - only pure white (`#FFFFFF`) and pure black (`#000000`). Grayscale or antialiasing distort training. In GIMP: `Image > Mode > Grayscale`, then work with the paint brush (hardness 100%). In Photoshop: create layer as bitmap or set brush to hard edge.

The more carefully corrected masks you have, the more targeted the model learns. Even 20-30 well-annotated images can noticeably improve detection quality for a specific use case.

This cycle (detect -> review mask -> correct -> train) can be repeated as often as needed until the model reliably covers what it needs to cover.

### Step 3 - Fine-tuning

The training pipeline is not part of this repository. There are two options:

- **Submit training data**: Anyone who provides their own annotated images and masks can have the model improved for everyone. The data is incorporated, the model is retrained, and the improved version is published.
- **Request your own pipeline**: Anyone who wants to train the model independently can request the training pipeline directly. Just get in touch - briefly describe the requirements and the specific use case.
