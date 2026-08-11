# coco_dataset_analyzer

## Purpose

The `coco_dataset_analyzer` module provides tools for parsing, validating, and comparing
datasets in [COCO JSON format](https://cocodataset.org/#format-data). It is useful for
quality-assurance (QA) workflows where you need to inspect class distributions, detect
images without annotations, identify out-of-bounds bounding boxes, and compare multiple
dataset splits side-by-side.

**Location:** `src/utils/coco_dataset_analyzer.py`

---

## `CocoDatasetAnalyzer` class

### Constructor

```python
CocoDatasetAnalyzer(file_path: str | Path)
```

Loads the COCO JSON file and validates that it contains the mandatory root keys
(`images`, `annotations`, `categories`).

**Args:**

| Parameter   | Type           | Description                        |
|-------------|----------------|------------------------------------|
| `file_path` | `str \| Path`  | Path to the COCO JSON file to load |

**Raises:**

| Exception           | Condition                                          |
|---------------------|----------------------------------------------------|
| `FileNotFoundError` | The specified file does not exist on disk          |
| `KeyError`          | The JSON is missing one or more mandatory root keys |

---

### `get_summary() -> CocoAnalysisSummary`

Computes advanced metrics and runs spatial validation checks on the loaded dataset.

**Returns** a `CocoAnalysisSummary` TypedDict with the following fields:

| Field                      | Type              | Description                                                                                          |
|----------------------------|-------------------|------------------------------------------------------------------------------------------------------|
| `total_images`             | `int`             | Number of entries in the `images` array                                                              |
| `total_annotations`        | `int`             | Number of entries in the `annotations` array                                                         |
| `avg_annotations_per_image`| `float`           | `total_annotations / total_images`, rounded to 2 decimal places; `0.0` if there are no images       |
| `classes`                  | `List[str]`       | All category names sorted alphabetically                                                             |
| `class_counts`             | `Dict[str, int]`  | Annotation count per class name (includes zero-count classes)                                        |
| `unannotated_image_paths`  | `List[str]`       | `file_name` values of images that have no associated annotations                                     |
| `out_of_bounds_errors`     | `List[str]`       | Human-readable error strings for bounding boxes that extend outside the image dimensions             |
| `categories_with_ids`      | `List[Dict]`      | List of `{"id": int, "name": str}` dicts for every category, sorted by `id` ascending               |

**Example:**

```python
from src.utils.coco_dataset_analyzer import CocoDatasetAnalyzer

analyzer = CocoDatasetAnalyzer("dataset/_annotations.coco.json")
summary = analyzer.get_summary()

print(summary["total_images"])          # e.g. 1200
print(summary["class_counts"])          # e.g. {"checked": 400, "signature": 300}
print(summary["categories_with_ids"])   # e.g. [{"id": 0, "name": "signature"}, ...]
```

---

## `compare_datasets()` function

```python
compare_datasets(paths: list[str | Path]) -> list[dict]
```

Analyzes and compares multiple COCO JSON datasets in a single call.

### Args

| Parameter | Type                    | Description                                 |
|-----------|-------------------------|---------------------------------------------|
| `paths`   | `list[str \| Path]`     | Ordered list of paths to COCO JSON files    |

### Return format

Returns a `list[dict]`, one entry per input path, **in the same order as `paths`**.

Each dict has one of two shapes:

**Success:**
```python
{
    "file": str,                  # file path as supplied
    "summary": CocoAnalysisSummary
}
```

**Error:**
```python
{
    "file": str,                  # file path as supplied
    "error": str                  # error message (e.g. FileNotFoundError, KeyError)
}
```

### Error handling

If a file raises any exception (not found, invalid JSON, missing COCO keys, etc.),
`compare_datasets` catches it, stores the error message in `"error"`, and continues
processing the remaining files. It never raises.

### Example

```python
from src.utils.coco_dataset_analyzer import compare_datasets

results = compare_datasets(["train.json", "val.json", "test.json"])

for r in results:
    if "error" in r:
        print(f"{r['file']}: ERROR — {r['error']}")
    else:
        s = r["summary"]
        print(f"{r['file']}: {s['total_images']} images, {s['total_annotations']} annotations")
```

---

## CLI usage

The CLI is provided by `scripts/analyze_dataset.py`.

### Single-file mode

When a single file is provided, the existing detailed per-file report is printed:

```bash
python scripts/analyze_dataset.py -i path/to/dataset.json
```

**Example output:**

```
Parsing and running data-integrity checks on: path/to/dataset.json...

==================================================
             COCO DATASET REPORT
==================================================
Total Images:               1 200
Total Annotations:          8 400
Avg Annotations / Image:    7.0
Total Unique Classes:       3
--------------------------------------------------
Class Name                       | Instance Count
--------------------------------------------------
checked                          | 4 200
signature                        | 2 800
unchecked                        | 1 400
==================================================

✅ QA Check: All images contain at least one annotation.
✅ QA Check: All bounding box coordinates fit safely within image dimensions.
```

### Multi-file comparison mode

When two or more files are provided, a side-by-side comparison table is printed:

```bash
python scripts/analyze_dataset.py -i train.json val.json test.json
```

**Example output:**

```
=== COCO DATASET COMPARISON REPORT ===

File                           |   Images | Categories   |  Total Ann.
--------------------------------------------------------------------
dataset_train.json             |    1 200 | 3            |       8 400
dataset_val.json               |      300 | 3            |       2 100
dataset_test.json              |      150 | 2            |         980

--- Per-class annotation counts ---

Class                          |   ID | dataset_train.json | dataset_val.json | dataset_test.json
checked                        |    1 |              4 200 |            1 050 |               490
signature                      |    0 |              2 800 |              700 |               350
unchecked                      |    2 |              1 400 |              350 |               —

--- Class balance (annotations per class / total annotations) ---

Class                          | dataset_train.json | dataset_val.json | dataset_test.json
checked                        |            50.00 % |          50.00 % |           50.00 %
signature                      |            33.33 % |          33.33 % |           35.71 %
unchecked                      |            16.67 % |          16.67 % |                —

--- QA Warnings ---
dataset_train.json : 0 unannotated images, 0 out-of-bounds errors
dataset_val.json   : 0 unannotated images, 0 out-of-bounds errors
dataset_test.json  : 2 unannotated images, 1 out-of-bounds error
```

**Notes on the comparison table:**

- Classes are unioned across all files and sorted alphabetically. If a class is absent from a file, `—` (em-dash) is shown for count and percentage.
- File column headers use only the **basename** of the path, truncated to 20 characters if longer.
- Numbers are formatted with thousands separators (e.g. `1 200`).
- If any input file fails to load, an error is printed to `stderr` and the script exits with code `1`.

### How to run

```bash
# Single file
python scripts/analyze_dataset.py -i dataset/_annotations.coco.json

# Multiple files (comparison)
python scripts/analyze_dataset.py -i train.json val.json test.json

# Using long-form flag
python scripts/analyze_dataset.py --input train.json val.json test.json
```

---

## Running the tests

```bash
pytest tests/test_coco_analyzer.py -v
```

Test classes:

| Class                                | Description                                            |
|--------------------------------------|--------------------------------------------------------|
| `TestCocoDatasetAnalyzerValid`       | Image/annotation/class counts on well-formed datasets  |
| `TestCocoDatasetAnalyzerUnannotated` | Unannotated image detection                            |
| `TestCocoDatasetAnalyzerOutOfBounds` | Bounding box spatial validation                        |
| `TestCocoDatasetAnalyzerErrors`      | Error handling for bad/missing files                   |
| `TestCocoAnalysisSummaryNewFields`   | `categories_with_ids` field presence, content, sorting |
| `TestCompareDatasets`                | `compare_datasets()` single-file, multi-file, errors   |