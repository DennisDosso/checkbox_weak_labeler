# filter_coco — COCO Category Filter and Remapper

## Overview

`src/utils/filter_coco.py` filters and remaps a COCO JSON annotation file
according to a user-supplied category mapping.

- **Categories in the mapping** are kept, with their `id` remapped to the value
  specified in the mapping.
- **Categories NOT in the mapping** are dropped, along with all their annotations.

This is useful for standardising category IDs across multiple datasets or for
removing unwanted classes (e.g. overly generic "container" classes) before training.

---

## Prerequisites

No additional dependencies beyond those already listed in `pyproject.toml`.
The script uses only the Python standard library (`pathlib`, `json`, `logging`,
`argparse`, `sys`, `datetime`).

---

## Mapping File Format

The mapping file is a plain JSON object mapping **category name** to the
**desired output `id`**:

```json
{
    "signature": 0,
    "checked": 1,
    "unchecked": 2
}
```

Rules:
- Values must be integers.
- No two names may share the same output `id` (duplicate IDs → error and exit).
- Categories in the input COCO but absent from the mapping are **dropped**.
- Categories in the mapping but absent from the input COCO are included in the
  output categories list (they simply match no annotations).

---

## How It Works

### 1. Load inputs
- Validates that both the input COCO JSON and the mapping file exist and are
  valid JSON. Exits with code 1 on any error.

### 2. Build category remapping
For each category in the input COCO:

| Situation | Behavior |
|-----------|----------|
| Name is in the mapping | Kept; `id` remapped to `mapping[name]`; `supercategory` preserved from input |
| Name is **not** in the mapping | **Dropped** — all its annotations are also dropped |

Categories listed in the mapping but absent from the input are still added to the
output with `supercategory: "none"` (they just won't match any annotation).

The output category list is sorted by new `id`.

### 3. Filter annotations
- Only annotations whose `category_id` corresponds to a kept category are kept.
- `category_id` is rewritten to the new id.
- `image_id` is rewritten to the new sequential image id.
- Annotation `id` values are reassigned sequentially from `1`.

### 4. Handle images
- By default, **all images are kept** in the output (even those with zero
  remaining annotations after filtering). This is valid per the COCO standard —
  unannotated images can serve as hard negatives during training.
- With `--drop-unannotated`: images with zero kept annotations are removed.
- Image `id` values are reassigned sequentially from `1`.

### 5. Write output
- The output file is **silently overwritten** if it already exists.
- Parent directories are created automatically if needed.
- Output follows the standard COCO structure: `info`, `licenses: []`, `images`,
  `annotations`, `categories`.

---

## CLI Usage

```bash
python -m src.utils.filter_coco \
    -i  input/_annotations.merged.coco.json \
    -m  category_mapping.json \
    -o  output/_annotations.filtered.coco.json \
    [--drop-unannotated]
```

### Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--input` | `-i` | **Yes** | Path to input COCO JSON file |
| `--mapping` | `-m` | **Yes** | Path to category mapping JSON file |
| `--output` | `-o` | **No** | Path to output COCO JSON file (silently overwritten). If missing, it is used the same value of `--input` |
| `--drop-unannotated` | — | No | Drop images with zero annotations after filtering |

### Example — user scenario

Input categories (from merged dataset):

| Name | Input id |
|------|----------|
| signature | 0 |
| checkbox | 1 |
| checked | 2 |
| unchecked | 3 |

Mapping file (`category_mapping.json`):
```json
{
    "signature": 0,
    "checked": 1,
    "unchecked": 2
}
```

Command:
```bash
python -m src.utils.filter_coco \
    -i _annotations.merged.coco.json \
    -m category_mapping.json \
    -o _annotations.filtered.coco.json
```

Result:

| Name | Input id | Output id |
|------|----------|-----------|
| signature | 0 | 0 — no change |
| checkbox | 1 | **DROPPED** |
| checked | 2 | 1 — remapped |
| unchecked | 3 | 2 — remapped |

---

## Console Output

A summary is always printed to stdout:

```
=== COCO Filter Summary ===
Input file      : _annotations.merged.coco.json
Mapping file    : category_mapping.json
Input categories: 4 (signature, checkbox, checked, unchecked)
Kept categories : 3 (signature, checked, unchecked)
Dropped categories: 1 (checkbox)
Input annotations: 498
Kept annotations : 420
Dropped annotations: 78
Input images    : 30
Output images   : 30
Output written  : _annotations.filtered.coco.json

Category remapping:
  'signature' : old id 0 -> new id 0  [no change]
  'checked'   : old id 2 -> new id 1  [REMAPPED]
  'unchecked' : old id 3 -> new id 2  [REMAPPED]
  'checkbox'  : old id 1 -> DROPPED
```

---

## Error Handling

| Situation | Behavior |
|-----------|----------|
| Input JSON not found | stderr + exit 1 |
| Input JSON is invalid | stderr + exit 1 |
| Mapping JSON not found | stderr + exit 1 |
| Mapping JSON is not a dict | stderr + exit 1 |
| Mapping value is not an integer | stderr + exit 1 |
| Duplicate output IDs in mapping | stderr + exit 1 |

---

## Unannotated Images — COCO Standard Note

The COCO format explicitly allows images without annotations. In the official COCO
dataset, unannotated images are used as background/hard-negative examples. Most
training frameworks (YOLO, Detectron2, MMDetection, etc.) handle them correctly.

- **Default** (`--drop-unannotated` not set): all images are kept — safe and
  COCO-compliant.
- **With `--drop-unannotated`**: images that have no remaining annotations after
  filtering are excluded — useful when you want a "clean" annotation-only dataset.

---

## Module Structure

```
src/utils/
  filter_coco.py              # Main script and all helper functions
  merge_coco.py               # Horizontal merge utility
  vertical_merge_coco.py      # Vertical merge utility
  coco_dataset_analyzer.py
  pdf_to_png.py
  __init__.py
```

Key functions:

| Function | Description |
|----------|-------------|
| `filter_coco(input, mapping, output, drop_unannotated)` | Main entry point |
| `load_mapping(path)` | Load and validate the category mapping JSON |
| `build_category_remap(input_categories, mapping)` | Build old_id → new_id remap |
| `filter_annotations(annotations, old_id_to_new_id, image_id_map)` | Filter and remap annotations |
| `build_image_id_map(images, drop_unannotated, kept_image_ids)` | Build image list and id map |
| `load_json_file(path, label)` | Generic JSON loader with error handling |