# Testing Guide

## Overview

The test suite for `checkbox-weak-labeler` is located in the `tests/` directory and is built with [pytest](https://docs.pytest.org/). It covers the core logic of the project — bounding box conversion, COCO dataset analysis, Roboflow client initialization, and the local RF-DETR detector — using synthetic data and mocks wherever external dependencies (models, APIs, GPUs) would otherwise be required.

No real model weights, API keys, or image files are needed to run the tests.

---

## Prerequisites

All test dependencies are declared in `pyproject.toml`. Make sure the project is installed before running the suite:

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e .
```

---

## Running the Tests

From the **project root**, run:

```bash
pytest tests/
```

To get a more detailed output with individual test names:

```bash
pytest tests/ -v
```

To see a short summary of failures only (useful in CI):

```bash
pytest tests/ -q
```

To run a single test file:

```bash
pytest tests/test_coco_analyzer.py -v
```

To run a single test class or method:

```bash
# Run a specific class
pytest tests/test_rfdetr_detector.py::TestRFDetrDetectorLoad -v

# Run a specific test method
pytest tests/test_rfdetr_detector.py::TestRFDetrDetectorLoad::test_load_raises_file_not_found_for_missing_path -v
```

---

## Test Files

### `tests/test_bbox_conversion.py`

Tests the bounding box conversion utilities used by both labelers.

| Test | What it verifies |
|------|-----------------|
| `TestConvertToCocobbox::test_basic_conversion` | Center x/y/w/h → COCO `[x_min, y_min, w, h]` produces correct coordinates |
| `TestConvertToCocobbox::test_width_and_height_preserved` | Width and height pass through unchanged |
| `TestConvertToCocobbox::test_zero_bbox` | All-zero input produces all-zero output |
| `TestConvertToCocobbox::test_float_coordinates_remain_float` | Remote labeler outputs float coordinates |
| `TestConvertToCocobbox::test_large_values` | Handles large image-scale coordinates correctly |
| `TestConvertXyxyToCocobbox::test_basic_conversion` | RF-DETR xyxy → COCO `[x_min, y_min, w, h]` produces correct coordinates |
| `TestConvertXyxyToCocobbox::test_zero_area_bbox` | Degenerate bbox (min == max) produces zero width/height |
| `TestConvertXyxyToCocobbox::test_integer_coordinates` | Local labeler outputs integer coordinates |
| `TestConvertXyxyToCocobbox::test_width_height_correct` | Width and height equal `xmax-xmin` and `ymax-ymin` |
| `TestConvertXyxyToCocobbox::test_top_left_corner_preserved` | `x_min` and `y_min` match the input values |

**Expected output:** all tests pass with no warnings.

---

### `tests/test_coco_analyzer.py`

Tests `CocoDatasetAnalyzer` using synthetic COCO JSON files created in memory via `tempfile`. No real dataset files needed.

| Test | What it verifies |
|------|-----------------|
| `test_image_count` | `total_images` equals the number of entries in `images` |
| `test_annotation_count` | `total_annotations` equals the number of entries in `annotations` |
| `test_class_counts` | `class_counts` correctly tallies annotations per category name |
| `test_classes_list` | `classes` lists all category names sorted alphabetically |
| `test_unannotated_image_detected` | Images with no annotations appear in `unannotated_image_paths` |
| `test_all_annotated_no_unannotated` | When all images are annotated, `unannotated_image_paths` is empty |
| `test_out_of_bounds_detected` | A bbox extending beyond the image boundary is flagged |
| `test_bbox_at_exact_boundary_is_valid` | A bbox that exactly touches the image edge is NOT flagged |
| `test_negative_origin_detected` | A bbox with negative x or y is flagged |
| `test_in_bounds_annotation_not_flagged` | A valid in-bounds bbox produces zero errors |
| `test_file_not_found` | Passing a non-existent path raises `FileNotFoundError` |
| `test_missing_root_keys_raises_key_error` | A COCO JSON missing mandatory keys raises `KeyError` |

**Expected output:** all 12 tests pass. Temporary files are cleaned up automatically by `tempfile`.

---

### `tests/test_roboflow_client.py`

Tests `get_inference_client()` in isolation. Uses `unittest.mock` to control environment variables and patch `InferenceHTTPClient` — no real Roboflow connection is made.

| Test | What it verifies |
|------|-----------------|
| `test_client_created_when_api_key_present` | When `ROBOFLOW_API_KEY` is set, the client is created without error |
| `test_raises_value_error_when_api_key_missing` | When the key is absent, `ValueError` is raised |
| `test_raises_value_error_when_api_key_empty_string` | An empty string for the key also triggers `ValueError` |

**Expected output:** all 3 tests pass. No network calls are made.

> **Note:** These tests use `importlib.reload` to re-evaluate module-level variables after patching the environment. This is necessary because `src/config.py` reads env vars at import time.

---

### `tests/test_rfdetr_detector.py`

Tests `RFDetrDetector` with `RFDETRNano` fully mocked — no real model weights or GPU are required. A temporary dummy `.pth` file is created by a `pytest` fixture for path-existence checks.

| Test | What it verifies |
|------|-----------------|
| `TestRFDetrDetectorLoad::test_load_raises_file_not_found_for_missing_path` | `load()` raises `FileNotFoundError` when the weights file does not exist |
| `TestRFDetrDetectorLoad::test_load_succeeds_with_valid_path` | `load()` completes without error when the file exists; `from_checkpoint` is called with correct arguments |
| `TestRFDetrDetectorLoad::test_load_skips_optimize_on_nvidia_runtime_error` | If `optimize_for_inference()` raises a NVIDIA-related `RuntimeError`, it is caught and logged as a warning — execution continues |
| `TestRFDetrDetectorLoad::test_load_reraises_non_nvidia_runtime_error` | A non-NVIDIA `RuntimeError` from `optimize_for_inference()` is re-raised |
| `TestRFDetrDetectorPredict::test_predict_raises_if_not_loaded` | Calling `predict()` before `load()` raises `RuntimeError` |
| `TestRFDetrDetectorPredict::test_predict_returns_filtered_detections` | Detections above the threshold are returned with integer bbox coords and correctly rounded confidence |
| `TestRFDetrDetectorPredict::test_predict_skips_out_of_range_class_id` | Detections with a `class_id` outside the model's `class_names` list are silently skipped |
| `TestRFDetrDetectorPredict::test_predict_returns_empty_list_for_no_detections` | When the model returns no detections, `predict()` returns an empty list |

**Expected output:** all 8 tests pass. No model is loaded; no GPU is required.

---

## What to Expect

A full successful run looks like this:

```
============================= test session starts ==============================
platform ... -- Python 3.13.x, pytest-8.x.x
rootdir: /path/to/checkbox-weak-labeler
collected 33 items

tests/test_bbox_conversion.py ..........                                 [ 30%]
tests/test_coco_analyzer.py ............                                 [ 66%]
tests/test_roboflow_client.py ...                                        [ 75%]
tests/test_rfdetr_detector.py ........                                   [100%]

============================== 33 passed in X.XXs ==============================
```

All 33 tests should pass. No external network connections, no GPU, and no real model files are needed.

---

## Adding New Tests

When extending the project, follow these conventions:

- Place new test files in `tests/` with the prefix `test_`.
- Use `unittest.mock.patch` and `MagicMock` to isolate external dependencies (models, APIs, file I/O).
- Use `tempfile` to create synthetic input files; never rely on real project data files in tests.
- Group related tests into classes named `TestSomethingSpecific`.
- Each test method should test exactly one behavior and have a descriptive name.