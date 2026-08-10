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
2. **Model loading:** The RF-DETR Nano model is instantiated from the `.pth` checkpoint using `RFDetrDetector` (`src/local_labeling/rfdetr_detector.py`). Inference optimization is attempted and gracefully skipped if no NVIDIA GPU is available. In dry-run mode the model is never loaded.
3. **Scanning:** All valid image files (JPG and PNG) in the input directory are collected, optionally scanning subdirectories recursively with `--recursive`.
4. **Resume/Overwrite check:** If `--resume` is set and the output file already exists, the script loads existing annotations and skips already-processed images. If `--overwrite` is set, it proceeds silently without prompting. Otherwise, if the output file exists, it asks for interactive confirmation.
5. **Inference loop:** For each new image, the script:
   - Opens the image with PIL to retrieve its dimensions (`width`, `height`).
   - Runs local RF-DETR inference via `detector.predict(image)`.
   - Filters detections to keep only those with `label == "signature"`.
6. **Bbox conversion:** RF-DETR returns bounding boxes in `[xmin, ymin, xmax, ymax]` format (absolute pixel coordinates). These are converted to COCO format `[x, y, width, height]`:
   ```
   width  = xmax - xmin
   height = ymax - ymin
   coco_bbox = [xmin, ymin, width, height]
   ```
7. **Export:** A COCO JSON file is written to the specified output path.

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
| `--recursive` | `-r` | No | `False` | Recursively scan subdirectories for images. |
| `--resume` | — | No | `False` | Resume from an existing output file, skipping already-processed images. |
| `--overwrite` | — | No | `False` | Overwrite the output file without interactive confirmation. |
| `--dry-run` | — | No | `False` | Simulate execution: scan images and report counts without loading the model, running inference, or writing any files. |

> **Note:** `--resume` and `--overwrite` are mutually exclusive. Using both together will produce an error.

### Examples

**Basic usage** (reads `config.local.yaml` from the project root):
```bash
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_2" \
    -o "data/datasets/weak_labels_batch_2.json"
```

**Recursive scan of subdirectories:**
```bash
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/" \
    -o "data/datasets/weak_labels_all.json" \
    --recursive
```

**Dry run (preview without inference):**
```bash
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_2" \
    -o "data/datasets/weak_labels_batch_2.json" \
    --dry-run
```

**Override confidence threshold at runtime:**
```bash
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_2" \
    -o "data/datasets/weak_labels_batch_2.json" \
    --confidence 0.50
```

**Resume an interrupted job:**
```bash
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_2" \
    -o "data/datasets/weak_labels_batch_2.json" \
    --resume
```

**Overwrite existing output without prompting (useful in CI pipelines):**
```bash
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_2" \
    -o "data/datasets/weak_labels_batch_2.json" \
    --overwrite
```

**Use a GPU and a specific model via environment variables:**
```bash
LOCAL_LABELER_MODEL_PATH="models/my_finetuned.pth" \
LOCAL_LABELER_DEVICE="cuda" \
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_2" \
    -o "data/datasets/weak_labels_batch_2.json"
```

**Use a custom config file:**
```bash
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_2" \
    -o "data/datasets/weak_labels_batch_2.json" \
    --config "configs/my_custom_config.yaml"
```

---

## Recursive Directory Scanning

By default, only the top-level of the input directory is scanned. With `--recursive`, the script walks all subdirectories and collects images at any nesting depth.

```bash
python -m src.local_labeling.local_weak_labeler -i "data/" -o "out.json" -r
```

The log will indicate the scanning mode:
```
Scanning directory (recursive): data/
```

---

## Dry Run Mode

When `--dry-run` is passed, the script scans images and reports what would happen without loading the model, performing any inference, or writing any files.

Sample output:
```
[DRY RUN] Input directory: data/raw/batch_2
[DRY RUN] Output file: data/datasets/weak_labels_batch_2.json
[DRY RUN] Images found: 150
[DRY RUN] Already processed (would skip): 50
[DRY RUN] Images that would be processed: 100
[DRY RUN] No inference was run. No files were written.
```

The "Already processed (would skip)" line appears only when `--dry-run` is combined with `--resume` and the output file already exists.

---

## Resume / Checkpoint Support

If a labeling job is interrupted (e.g. due to a system shutdown), it can be resumed without reprocessing already-labeled images.

```bash
python -m src.local_labeling.local_weak_labeler -i "data/raw/batch_2" -o "out.json" --resume
```

**How it works:**
- If the output file exists: existing images and annotations are loaded; only images whose `file_name` is not already in the output are processed; `image_id` and `annotation_id` counters continue from the last saved values.
- If the output file does not exist: the script starts fresh without errors.
- At the end, the output file contains both previously saved and newly generated annotations.

---

## Output File Protection

To prevent accidental data loss, the script handles existing output files as follows:

| Flags used | Output file exists? | Behavior |
|------------|---------------------|----------|
| *(none)* | No | Starts normally |
| *(none)* | Yes | Asks interactively: `"Output file already exists. Overwrite? [y/N]"` |
| `--overwrite` | No | Starts normally |
| `--overwrite` | Yes | Overwrites silently (no prompt) |
| `--resume` | No | Starts fresh (no error) |
| `--resume` | Yes | Loads existing data, appends new annotations |
| `--resume --overwrite` | any | **Error** — flags are mutually exclusive |

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