# Weak Labeler (`weak_labeler.py`)

## Overview

`weak_labeler.py` is a command-line executable script that automates the process of generating "weak" annotations for an unlabelled image dataset. It scans a directory of images, sends them to a hosted Roboflow model for inference, and compiles the predictions into a standard COCO format JSON file.

This is highly useful for pre-labeling large datasets before manual review, drastically speeding up the data preparation phase.

## Prerequisites

1. **API Key Setup:** Ensure your `.env.local` file is populated with your `ROBOFLOW_API_KEY`.
2. **Image Format:** The script supports `.jpg`, `.jpeg`, and `.png` files (case-insensitive).
3. **Dependencies:** Requires the `inference_sdk`, `python-dotenv`, and `tqdm` packages.

## How It Works

1. **Initialization:** The script imports configuration variables from `src.config` and initializes the inference client via `src.weak_labeling.roboflow_client`.
2. **Scanning:** It uses `glob` to locate all valid image files (JPG and PNG) in the provided input directory.
3. **Inference Loop:** For each image, it queries the Roboflow model while displaying a clean `tqdm` progress bar in the terminal.
4. **Data Transformation:** It converts Roboflow's center-based bounding boxes `[center_x, center_y, width, height]` into COCO's top-left format `[top_left_x, top_left_y, width, height]`.
5. **Filtering:** Predictions are checked against the `CATEGORY_MAPPING` in `config.py`. Unknown classes are safely ignored.
6. **Export:** A complete COCO JSON dictionary is saved to the specified output path.

## Usage

Run the script from the root of the project using the command line.

```bash
python -m src.weak_labeling.weak_labeler --input "path/to/images" --output "path/to/save/annotations.json" --model "model-id/version"
```

### Arguments

| Argument | Short Flag | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `--input` | `-i` | Yes | None | Path to the directory containing the images to be labeled. |
| `--output` | `-o` | Yes | None | Path and filename where the final COCO JSON should be saved. |
| `--model` | `-m` | No | `checkbox-0fyo0/1` | The Roboflow model ID to use for inference. |

### Example

```bash
python -m src/weak_labeling/weak_labeler -i "data/processed/batch_1" -o "data/datasets/weak_labels_batch_1.json"

```

## Error Handling

If an image is corrupted or the API connection drops, the script catches the exception, outputs an error safely using `tqdm.write()` (so it doesn't break the progress bar UI), and continues to the next image. This ensures a single failure does not crash a labeling job that might contain thousands of files.