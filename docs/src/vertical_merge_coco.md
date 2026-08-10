# vertical_merge_coco — Vertical COCO Dataset Merge Utility

## Overview

`src/utils/vertical_merge_coco.py` performs a **vertical merge** of two COCO
datasets: it combines datasets that contain **different images** but **identical
or compatible category schemas** into a single, flat output directory.

### Horizontal vs. Vertical merge

| | `merge_coco.py` (horizontal) | `vertical_merge_coco.py` (vertical) |
|---|---|---|
| **Image sets** | Same images in both files | Different images in each file |
| **Categories** | Different categories per file | Same (or compatible) categories |
| **Output** | Single merged JSON | Single directory with copied images + JSON |
| **Image deduplication** | By `file_name` (keep one) | Filename conflicts → renamed with suffix |
| **Category conflicts** | Resolved automatically by name | Strict — any mismatch stops execution |

---

## Prerequisites

No additional dependencies beyond those already listed in `pyproject.toml`.
The script uses only the Python standard library (`shutil`, `pathlib`, `json`,
`logging`, `argparse`, `sys`, `datetime`).

---

## How It Works

### 1. Pre-flight validation (before any file is created or copied)

The script validates all inputs before touching the filesystem:

- **Output directory**: if it already exists → error and exit (no overwrite, no prompt).
- **Input image directories**: if either does not exist → error and exit.
- **Input JSON files**: if either does not exist or contains invalid JSON → error and exit.
- **Category compatibility**: see step 2 below.

### 2. Strict category validation

Category identity is determined by `id`. The two datasets must be compatible:

| Situation | Behavior |
|-----------|----------|
| Same `id`, same `name`, same `supercategory` | OK — keep one entry |
| Same `id`, **different `name`** | **Error — exit 1** |
| Same `id`, same `name`, **different `supercategory`** | **Error — exit 1** |
| Category `id` only in one file | OK — included in output (additive) |

The merged category list is the union of both files, sorted by `id`.

### 3. Filename conflict handling

Before copying any files, the script compares the `file_name` fields of both
datasets. If the same filename appears in both:

- Both images are treated as **different images** (not duplicates).
- Dataset 1's copy is renamed: `document_001.png` → `document_001_1.png`
- Dataset 2's copy is renamed: `document_001.png` → `document_001_2.png`
- A warning is printed for each conflict.
- The `file_name` field in the output JSON is updated to the renamed filename.

Non-conflicting filenames are copied as-is.

### 4. Image copying

Images are copied from both source directories into the output directory using
`shutil.copy2` (preserves file metadata).

- Output layout is **flat** (no subdirectories).
- If an image referenced in the JSON is **not found** in its directory → warning
  log, image skipped, and all its annotations are excluded from the output.
- All images receive **new sequential `id` values** starting from `1`
  (dataset 1's images first, then dataset 2's).

### 5. Annotation merging

- All annotations from both files are combined.
- Each annotation's `image_id` is rewritten to match the new global image ID.
- Annotations for skipped images (file not found) are excluded automatically.
- All other fields (`bbox`, `area`, `iscrowd`, `confidence`, etc.) are preserved
  exactly as-is.
- Annotations receive **new sequential `id` values** starting from `1`.

### 6. Output

The output directory contains:
- All copied image files (renamed where necessary).
- `_annotations.coco.json` — the merged COCO JSON.

The JSON structure follows the standard COCO format:

```json
{
    "info": { ... },
    "licenses": [],
    "images": [ ... ],
    "annotations": [ ... ],
    "categories": [ ... ]
}
```

---

## CLI Usage

```bash
python -m src.utils.vertical_merge_coco \
    --input1-images /path/to/dataset1/images \
    --input1-json   /path/to/dataset1/_annotations.coco.json \
    --input2-images /path/to/dataset2/images \
    --input2-json   /path/to/dataset2/_annotations.coco.json \
    --output        /path/to/output_dir
```

### Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--input1-images` | `-i1` | **Yes** | Path to dataset 1 image directory |
| `--input1-json`   | `-j1` | **Yes** | Path to dataset 1 COCO JSON file |
| `--input2-images` | `-i2` | **Yes** | Path to dataset 2 image directory |
| `--input2-json`   | `-j2` | **Yes** | Path to dataset 2 COCO JSON file |
| `--output`        | `-o`  | **Yes** | Output directory path (must not exist) |

### Example

```bash
python -m src.utils.vertical_merge_coco \
    -i1 data/train_batch1 \
    -j1 data/train_batch1/_annotations.coco.json \
    -i2 data/train_batch2 \
    -j2 data/train_batch2/_annotations.coco.json \
    -o  data/train_combined
```

---

## Console Output

After merging, a summary is always printed to stdout:

```
=== Vertical COCO Merge Summary ===
Dataset 1       : data/train_batch1 (250 images, 3120 annotations)
Dataset 2       : data/train_batch2 (200 images, 2930 annotations)
Filename conflicts : 3 (renamed with _1 / _2 suffix)
Total images    : 450
Total annotations: 6050
Categories      : 3 (signature, checked, unchecked)
Output directory: data/train_combined
```

Filename conflicts are also logged as warnings during the copy phase:

```
WARNING: Filename conflict 'document_001.png' — renamed to 'document_001_1.png' and 'document_001_2.png'
```

---

## Error Handling

| Situation | Behavior |
|-----------|----------|
| Output directory already exists | stderr + exit 1 (no prompt, no overwrite) |
| Input JSON not found | stderr + exit 1 |
| Input JSON is not valid JSON | stderr + exit 1 |
| Input image directory not found | stderr + exit 1 |
| Category `id` conflict (different name) | stderr + exit 1 (before any copy) |
| Category `id` conflict (different supercategory) | stderr + exit 1 (before any copy) |
| Image file referenced in JSON but not found | Warning log, image and its annotations skipped |

Category validation errors are reported **before any files are created or
copied**, so the output directory is never left in a partial state due to a
category conflict.

---

## Output Structure

```
output_dir/
  document_001.png
  document_002.png
  document_003_1.png        ← renamed (conflict from dataset 1)
  document_003_2.png        ← renamed (conflict from dataset 2)
  invoice_001.png
  ...
  _annotations.coco.json
```

---

## Module Structure

```
src/utils/
  vertical_merge_coco.py    # Main script and all helper functions
  merge_coco.py             # Horizontal merge utility
  coco_dataset_analyzer.py
  pdf_to_png.py
  __init__.py
```

Key functions exposed by the module:

| Function | Description |
|----------|-------------|
| `vertical_merge(input1_images, input1_json, input2_images, input2_json, output)` | Main entry point |
| `validate_and_merge_categories(categories1, categories2)` | Strict category validation + union |
| `detect_filename_conflicts(images1, images2)` | Find conflicting file_names |
| `copy_images_and_build_map(images, image_dir, output_dir, conflicts, suffix)` | Copy images with renaming |
| `assign_sequential_image_ids(images1, images2)` | Assign new sequential IDs |
| `merge_annotations(...)` | Merge and remap annotations |
| `load_coco_file(path)` | Load and validate a COCO JSON file |
| `validate_image_dir(path)` | Validate an image directory exists |