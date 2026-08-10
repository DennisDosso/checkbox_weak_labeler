# Local Weak Labeler (`local_weak_labeler.py`)

## Overview

`local_weak_labeler.py` is a standalone command-line script that automates the generation of "weak" COCO-format annotations for a directory of images using a **locally-loaded RF-DETR Nano model** (`.pth` weights file).

Unlike `weak_labeler.py` — which relies on a remote Roboflow API — this script runs inference entirely on the local machine, with no network connection required. It is designed to detect **signatures** in document images and produce an output JSON file in the same COCO format as the remote labeler, making the two tools interchangeable in downstream pipelines.

---

## Prerequisites

1. **Model weights:** A trained RF-DETR Nano `.pth` checkpoint file must be available on the local filesystem.
2. **Configuration file:** A `config.local.yaml` file must exist at the project root (or an alternative path must be provided via `--config`). See [Configuration](#configuration) for details.
3. **Dependencies:** Requires `rfdetr`, `torch`, `pydantic-settings`, `pyyaml`, `pillow`, and `tqdm`. All are listed in `pyproject.toml`.
4. **Image format:** Supports `.jpg`, `.jpeg`, and `.png` files (case-insensitive).

---

## Configuration

The script is configured via a YAML file. By default it reads `config.local.yaml` from the project root.

```yaml
# config.local.yaml

# Path to the RF-DETR .pth weights file (REQUIRED)
model_path: "models/rfdetr-nano.pth"

# Confidence threshold for detections (0.0 - 1.0)
confidence_threshold: 0.35
```

### Configuration resolution priority

Settings are resolved in the following order (highest to lowest priority):

| Priority | Source | Example |
|----------|--------|---------|
| 1 (highest) | Environment variables (`LOCAL_LABELER_*`) | `LOCAL_LABELER_MODEL_PATH=models/my_model.pth` |
| 2 | YAML config file | `config.local.yaml` |
| 3 (lowest) | Python defaults | `confidence_threshold = 0.35` |

### Available environment variables

| Variable | Description |
|----------|-------------|
| `LOCAL_LABELER_MODEL_PATH` | Path to the `.pth` weights file |
| `LOCAL_LABELER_CONFIDENCE_THRESHOLD` | Confidence threshold (0.0–1.0) |
| `LOCAL_LABELER_DEVICE` | Torch device (`"cpu"` or `"cuda"`) |
| `LOCAL_LABELER_CONFIG_PATH` | Path to an alternative YAML config file |

---

## How It Works

1. **Configuration loading:** Settings are loaded from `config.local.yaml` (and/or environment variables) via `LocalSettings` in `src/local_labeling/local_config.py`.
2. **Model loading:** The RF-DETR Nano model is instantiated from the `.pth` checkpoint using `RFDetrDetector` (`src/local_labeling/rfdetr_detector.py`). Inference optimization is attempted and gracefully skipped if no NVIDIA GPU is available.
3. **Scanning:** All valid image files (JPG and PNG) in the input directory are collected.
4. **Inference loop:** For each image, the script:
   - Opens the image with PIL to retrieve its dimensions (`width`, `height`).
   - Runs local RF-DETR inference via `detector.predict(image)`.
   - Filters detections to keep only those with `label == "signature"`.
5. **Bbox conversion:** RF-DETR returns bounding boxes in `[xmin, ymin, xmax, ymax]` format (absolute pixel coordinates). These are converted to COCO format `[x, y, width, height]`:
   ```
   width  = xmax - xmin
   height = ymax - ymin
   coco_bbox = [xmin, ymin, width, height]
   ```
6. **Export:** A COCO JSON file is written to the specified output path.

---

## Output Format

The output file is a standard COCO JSON with the following structure:

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "document_001.jpg",
            "width": 1654,
            "height": 2339
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 0,
            "bbox": [120, 340, 210, 55],
            "area": 11550,
            "iscrowd": 0,
            "confidence": 0.87341
        }
    ],
    "categories": [
        {"id": 0, "name": "signature", "supercategory": "signature"}
    ]
}
```

The output format is identical to that produced by `weak_labeler.py`, ensuring full compatibility with downstream tools.

---

## Usage

Run the script from the project root using the `-m` module flag:

```bash
python -m src.local_labeling.local_weak_labeler --input "path/to/images" --output "path/to/save/annotations.json"
```

### Arguments

| Argument | Short Flag | Required | Default | Description |
|----------|-----------|----------|---------|-------------|
| `--input` | `-i` | Yes | — | Path to the directory containing the images to label. |
| `--output` | `-o` | Yes | — | Path and filename where the final COCO JSON will be saved. |
| `--confidence` | — | No | *(from config)* | Override the confidence threshold (float between 0.0 and 1.0). |
| `--config` | — | No | `config.local.yaml` | Path to an alternative YAML configuration file. |

### Examples

**Basic usage** (reads `config.local.yaml` from the project root):
```bash
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_2" \
    -o "data/datasets/weak_labels_batch_2.json"
```

**Override confidence threshold at runtime:**
```bash
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_2" \
    -o "data/datasets/weak_labels_batch_2.json" \
    --confidence 0.50
```

**Use a custom config file:**
```bash
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_2" \
    -o "data/datasets/weak_labels_batch_2.json" \
    --config "configs/my_custom_config.yaml"
```

**Use a GPU and a specific model via environment variables:**
```bash
LOCAL_LABELER_MODEL_PATH="models/my_finetuned.pth" \
LOCAL_LABELER_DEVICE="cuda" \
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_2" \
    -o "data/datasets/weak_labels_batch_2.json"
```

---

## Error Handling

If an image cannot be opened or inference fails, the script catches the exception, prints an error message using `tqdm.write()` (so it does not break the progress bar display), and continues to the next image. A single failed image does not interrupt the entire labeling job.

---

## Module Structure

```
src/local_labeling/
    __init__.py                 # Package init
    local_config.py             # LocalSettings (pydantic-settings + YAML)
    rfdetr_detector.py          # RFDetrDetector — model loading and inference
    local_weak_labeler.py       # CLI entry point and labeling pipeline
```