# Weak Labeler (`weak_labeler.py`)

## Overview

`weak_labeler.py` is a command-line executable script that automates the process of generating "weak" annotations for an unlabelled image dataset. It scans a directory of images, sends them to a hosted Roboflow model for inference, and compiles the predictions into a standard COCO format JSON file.

This is highly useful for pre-labeling large datasets before manual review, drastically speeding up the data preparation phase.

## Prerequisites

1. **API Key Setup:** Ensure your `.env.local` file is populated with your `ROBOFLOW_API_KEY`.
2. **Image Format:** The script supports `.jpg`, `.jpeg`, and `.png` files (case-insensitive).
3. **Dependencies:** Requires the `inference_sdk`, `python-dotenv`, and `tqdm` packages.

## How It Works

1. **Initialization:** The script imports configuration variables from `src.config` and initializes the inference client via `src.weak_labeling.roboflow_client` (only if not in dry-run mode).
2. **Scanning:** It uses `glob` to locate all valid image files (JPG and PNG) in the provided input directory, optionally scanning subdirectories recursively with `--recursive`.
3. **Resume/Overwrite check:** If `--resume` is set and the output file already exists, the script loads existing annotations and skips already-processed images. If `--overwrite` is set, it proceeds silently without prompting. Otherwise, if the output file exists, it asks for interactive confirmation.
4. **Inference loop:** For each new image, it queries the Roboflow model while displaying a clean `tqdm` progress bar in the terminal.
5. **Data Transformation:** It converts Roboflow's center-based bounding boxes `[center_x, center_y, width, height]` into COCO's top-left format `[top_left_x, top_left_y, width, height]`.
6. **Filtering:** Predictions are checked against the `CATEGORY_MAPPING` in `config.py`. Unknown classes are safely ignored.
7. **Export:** A complete COCO JSON dictionary is saved to the specified output path.

## Usage

Run the script from the root of the project using the command line.

```bash
python -m src.weak_labeling.weak_labeler --input "path/to/images" --output "path/to/save/annotations.json"
```

### Arguments

| Argument | Short Flag | Required | Default | Description |
|----------|-----------|----------|---------|-------------|
| `--input` | `-i` | Yes | — | Path to the directory containing the images to be labeled. |
| `--output` | `-o` | Yes | — | Path and filename where the final COCO JSON should be saved. |
| `--model` | `-m` | No | `checkbox-0fyo0/1` | The Roboflow model ID to use for inference. |
| `--recursive` | `-r` | No | `False` | Recursively scan subdirectories for images. |
| `--resume` | — | No | `False` | Resume from an existing output file, skipping already-processed images. |
| `--overwrite` | — | No | `False` | Overwrite the output file without interactive confirmation. |
| `--dry-run` | — | No | `False` | Simulate execution: scan images and report counts without running inference or writing any files. |

> **Note:** `--resume` and `--overwrite` are mutually exclusive. Using both together will produce an error.

### Examples

**Basic usage:**
```bash
python -m src.weak_labeling.weak_labeler \
    -i "data/processed/batch_1" \
    -o "data/datasets/weak_labels_batch_1.json"
```

**Recursive scan of subdirectories:**
```bash
python -m src.weak_labeling.weak_labeler \
    -i "data/processed/" \
    -o "data/datasets/weak_labels_all.json" \
    --recursive
```

**Dry run (preview without inference):**
```bash
python -m src.weak_labeling.weak_labeler \
    -i "data/processed/batch_1" \
    -o "data/datasets/weak_labels_batch_1.json" \
    --dry-run
```

**Resume an interrupted job:**
```bash
python -m src.weak_labeling.weak_labeler \
    -i "data/processed/batch_1" \
    -o "data/datasets/weak_labels_batch_1.json" \
    --resume
```

**Overwrite existing output without prompting (useful in CI pipelines):**
```bash
python -m src.weak_labeling.weak_labeler \
    -i "data/processed/batch_1" \
    -o "data/datasets/weak_labels_batch_1.json" \
    --overwrite
```

---

## Recursive Directory Scanning

By default, only the top-level of the input directory is scanned. With `--recursive`, the script walks all subdirectories and collects images at any nesting depth.

```bash
python -m src.weak_labeling.weak_labeler -i "data/" -o "out.json" -r
```

The log will indicate the scanning mode:
```
Scanning directory (recursive): data/
```

---

## Dry Run Mode

When `--dry-run` is passed, the script scans images and reports what would happen without performing any inference or writing any files. The Roboflow client is never initialized in this mode.

Sample output:
```
[DRY RUN] Input directory: data/processed/batch_1
[DRY RUN] Output file: data/datasets/weak_labels_batch_1.json
[DRY RUN] Images found: 150
[DRY RUN] Already processed (would skip): 50
[DRY RUN] Images that would be processed: 100
[DRY RUN] No inference was run. No files were written.
```

The "Already processed (would skip)" line appears only when `--dry-run` is combined with `--resume` and the output file already exists.

---

## Resume / Checkpoint Support

If a labeling job is interrupted (e.g. due to a network error or system shutdown), it can be resumed without reprocessing already-labeled images.

```bash
python -m src.weak_labeling.weak_labeler -i "data/batch_1" -o "out.json" --resume
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

If an image is corrupted or the API connection drops, the script catches the exception, outputs an error safely using `tqdm.write()` (so it does not break the progress bar UI), and continues to the next image. This ensures a single failure does not crash a labeling job that might contain thousands of files.