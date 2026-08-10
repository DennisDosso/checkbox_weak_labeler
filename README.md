# checkbox-weak-labeler

A toolkit for automatically generating **weak COCO-format annotations** on document images. It supports two independent labeling strategies: a **remote** approach via the Roboflow API, and a **local** approach using a RF-DETR Nano model loaded from a `.pth` weights file — no internet connection required.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Labeling Strategies](#labeling-strategies)
  - [Remote Labeler (Roboflow API)](#1-remote-labeler-roboflow-api)
  - [Local Labeler (RF-DETR Nano)](#2-local-labeler-rf-detr-nano)
- [Utilities](#utilities)
  - [PDF to PNG Converter](#pdf-to-png-converter)
  - [COCO Dataset Analyzer](#coco-dataset-analyzer)
- [Output Format](#output-format)
- [Documentation](#documentation)

---

## Overview

Manually annotating large datasets of document images is slow and expensive. This project automates the process by running object detection models on images and saving the predictions as COCO-format JSON files — ready to be reviewed, corrected, and used for supervised training.

**Supported detection classes (remote labeler):**

| ID | Name | Description |
|----|------|-------------|
| 0 | `signature` | Handwritten signature |
| 1 | `checked` | Filled checkbox |
| 2 | `unchecked` | Empty checkbox |

**Supported detection classes (local labeler):**

| ID | Name | Description |
|----|------|-------------|
| 0 | `signature` | Handwritten signature |

---

## Project Structure

```
checkbox-weak-labeler/
├── src/
│   ├── config.py                        # Shared config: category mappings, Roboflow credentials
│   ├── weak_labeling/
│   │   ├── roboflow_client.py           # Roboflow API client factory
│   │   └── weak_labeler.py              # Remote labeler CLI script
│   ├── local_labeling/
│   │   ├── local_config.py              # Local labeler config (pydantic-settings + YAML)
│   │   ├── rfdetr_detector.py           # RF-DETR Nano model wrapper
│   │   └── local_weak_labeler.py        # Local labeler CLI script
│   └── utils/
│       ├── pdf_to_png.py                # PDF → PNG conversion utility
│       └── coco_dataset_analyzer.py     # COCO JSON analysis and QA
├── scripts/
│   ├── analyze_dataset.py               # CLI for COCO dataset analysis
│   └── apply_detection_to_one_image.py  # Run detection on a single image
├── docs/
│   └── src/                             # Detailed documentation per module
├── config.local.yaml                    # Config for the local RF-DETR labeler
├── pyproject.toml
└── README.md
```

---

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install all dependencies
uv sync
```

Alternatively, install with pip:

```bash
pip install -e .
```

---

## Labeling Strategies

### 1. Remote Labeler (Roboflow API)

Sends each image to a hosted Roboflow model and collects the predictions. Requires an active internet connection and a valid Roboflow API key.

#### Setup

Create a `.env.local` file at the project root:

```env
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_API_URL=https://serverless.roboflow.com
```

#### Usage

```bash
python -m src.weak_labeling.weak_labeler \
    -i "data/raw/batch_1" \
    -o "data/annotations/weak_labels_batch_1.json" \
    -m "checkbox-0fyo0/1"
```

#### Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--input` | `-i` | Yes | — | Directory containing source images |
| `--output` | `-o` | Yes | — | Output path for the COCO JSON file |
| `--model` | `-m` | No | `checkbox-0fyo0/1` | Roboflow model ID and version |

---

### 2. Local Labeler (RF-DETR Nano)

Runs inference entirely on the local machine using a RF-DETR Nano model loaded from a `.pth` checkpoint file. No internet connection or API key required.

#### Setup

Edit `config.local.yaml` at the project root and set the path to your model weights:

```yaml
# Path to the RF-DETR .pth weights file (REQUIRED)
model_path: "models/rfdetr-nano.pth"

# Confidence threshold for detections (0.0 - 1.0)
confidence_threshold: 0.35
```

Any value can be overridden at runtime via environment variables with the prefix `LOCAL_LABELER_` (e.g. `LOCAL_LABELER_CONFIDENCE_THRESHOLD=0.5`).

#### Usage

```bash
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_1" \
    -o "data/annotations/weak_labels_batch_1.json"
```

#### Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--input` | `-i` | Yes | — | Directory containing source images |
| `--output` | `-o` | Yes | — | Output path for the COCO JSON file |
| `--confidence` | — | No | *(from config)* | Override confidence threshold at runtime |
| `--config` | — | No | `config.local.yaml` | Path to an alternative YAML config file |

#### Examples

```bash
# Override confidence threshold at runtime
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_1" \
    -o "data/annotations/weak_labels.json" \
    --confidence 0.50

# Use a custom config file
python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_1" \
    -o "data/annotations/weak_labels.json" \
    --config "configs/production.yaml"

# Run on GPU
LOCAL_LABELER_DEVICE=cuda python -m src.local_labeling.local_weak_labeler \
    -i "data/raw/batch_1" \
    -o "data/annotations/weak_labels.json"
```

---

## Utilities

### PDF to PNG Converter

Converts multi-page PDF documents into individual PNG images at high resolution (default: 300 DPI). This is a required preprocessing step before running either labeler on PDF-sourced data.

```bash
python src/utils/pdf_to_png.py \
    -i "data/raw/pdfs" \
    -o "data/processed/images" \
    --dpi 300
```

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--input` | `-i` | Yes | — | Directory containing PDF files |
| `--output` | `-o` | Yes | — | Directory where PNG images will be saved |
| `--dpi` | — | No | `300` | Output image resolution |

---

### COCO Dataset Analyzer

Validates and summarizes a COCO JSON annotation file. Checks for:
- Class distribution and instance counts
- Images with no annotations (unannotated images)
- Bounding boxes that extend outside image boundaries

```bash
python scripts/analyze_dataset.py -i "data/annotations/weak_labels.json"
```

**Example output:**
```
==================================================
             COCO DATASET REPORT
==================================================
Total Images:               1,240
Total Annotations:          3,872
Avg Annotations / Image:    3.12
Total Unique Classes:       3
--------------------------------------------------
Class Name                       | Instance Count
--------------------------------------------------
signature                        | 1,204
checked                          | 1,530
unchecked                        | 1,138
==================================================
✅ QA Check: All images contain at least one annotation.
✅ QA Check: All bounding box coordinates fit safely within image dimensions.
```

---

## Output Format

Both labelers produce the same COCO-format JSON structure:

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "document_page_1.jpg",
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
        {"id": 0, "name": "signature", "supercategory": "signature"},
        {"id": 1, "name": "checked", "supercategory": "checkbox"},
        {"id": 2, "name": "unchecked", "supercategory": "checkbox"}
    ]
}
```

> **Note:** The local labeler only outputs annotations for the `signature` class (id=0). Its `categories` array contains only that entry.

Bounding boxes use the COCO format: `[x_min, y_min, width, height]` in absolute pixel coordinates.

---

## Documentation

Detailed documentation for each module is available in the `docs/src/` folder:

| File | Description |
|------|-------------|
| [`docs/src/config.md`](docs/src/config.md) | Project-wide configuration and category mappings |
| [`docs/src/weak_labeler.md`](docs/src/weak_labeler.md) | Remote Roboflow labeler |
| [`docs/src/local_weak_labeler.md`](docs/src/local_weak_labeler.md) | Local RF-DETR labeler |
| [`docs/src/pdf_to_png.md`](docs/src/pdf_to_png.md) | PDF to PNG conversion utility |