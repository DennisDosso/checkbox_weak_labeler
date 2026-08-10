"""
tests/test_coco_analyzer.py
----------------------------
Unit tests for CocoDatasetAnalyzer in src/utils/coco_dataset_analyzer.py.

Uses tempfile and json to create synthetic COCO JSON files in memory —
no real dataset files are required.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.utils.coco_dataset_analyzer import CocoDatasetAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_coco_json(data: dict) -> Path:
    """Write a COCO dict to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


def _minimal_coco(
    images=None,
    annotations=None,
    categories=None,
) -> dict:
    """Return a minimal valid COCO dict with sensible defaults."""
    return {
        "images": images or [],
        "annotations": annotations or [],
        "categories": categories or [{"id": 0, "name": "signature"}],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCocoDatasetAnalyzerValid:
    """Tests against well-formed COCO datasets."""

    def test_image_count(self):
        """total_images must equal the number of entries in 'images'."""
        data = _minimal_coco(
            images=[
                {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "b.jpg", "width": 200, "height": 200},
            ],
        )
        path = _write_coco_json(data)
        analyzer = CocoDatasetAnalyzer(path)
        summary = analyzer.get_summary()
        assert summary["total_images"] == 2

    def test_annotation_count(self):
        """total_annotations must equal the number of entries in 'annotations'."""
        data = _minimal_coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]},
                {"id": 2, "image_id": 1, "category_id": 0, "bbox": [5, 5, 10, 10]},
            ],
        )
        path = _write_coco_json(data)
        analyzer = CocoDatasetAnalyzer(path)
        summary = analyzer.get_summary()
        assert summary["total_annotations"] == 2

    def test_class_counts(self):
        """class_counts must correctly tally annotations per category name."""
        data = _minimal_coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 200, "height": 200}],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]},
                {"id": 2, "image_id": 1, "category_id": 0, "bbox": [20, 20, 10, 10]},
            ],
            categories=[{"id": 0, "name": "signature"}],
        )
        path = _write_coco_json(data)
        summary = CocoDatasetAnalyzer(path).get_summary()
        assert summary["class_counts"]["signature"] == 2

    def test_classes_list(self):
        """classes must list all category names sorted alphabetically."""
        data = _minimal_coco(
            categories=[
                {"id": 1, "name": "unchecked"},
                {"id": 0, "name": "checked"},
                {"id": 2, "name": "signature"},
            ],
        )
        path = _write_coco_json(data)
        summary = CocoDatasetAnalyzer(path).get_summary()
        assert summary["classes"] == ["checked", "signature", "unchecked"]


class TestCocoDatasetAnalyzerUnannotated:
    """Tests that unannotated images are correctly identified."""

    def test_unannotated_image_detected(self):
        """An image with no annotations must appear in unannotated_image_paths."""
        data = _minimal_coco(
            images=[
                {"id": 1, "file_name": "annotated.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "empty.jpg", "width": 100, "height": 100},
            ],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]},
            ],
        )
        path = _write_coco_json(data)
        summary = CocoDatasetAnalyzer(path).get_summary()
        assert "empty.jpg" in summary["unannotated_image_paths"]
        assert "annotated.jpg" not in summary["unannotated_image_paths"]

    def test_all_annotated_no_unannotated(self):
        """When every image has at least one annotation, unannotated_image_paths is empty."""
        data = _minimal_coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]},
            ],
        )
        path = _write_coco_json(data)
        summary = CocoDatasetAnalyzer(path).get_summary()
        assert summary["unannotated_image_paths"] == []


class TestCocoDatasetAnalyzerOutOfBounds:
    """Tests for out-of-bounds bbox detection."""

    def test_out_of_bounds_detected(self):
        """A bbox that extends beyond the image boundary must be flagged."""
        data = _minimal_coco(
            images=[{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
            annotations=[
                # x=80, y=80, w=30, h=30 -> x+w=110 > 100 (out of bounds)
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [80, 80, 30, 30]},
            ],
        )
        path = _write_coco_json(data)
        summary = CocoDatasetAnalyzer(path).get_summary()
        assert len(summary["out_of_bounds_errors"]) == 1

    def test_bbox_at_exact_boundary_is_valid(self):
        """A bbox that exactly touches the image edge must NOT be flagged."""
        data = _minimal_coco(
            images=[{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
            annotations=[
                # x=0, y=0, w=100, h=100 -> x+w=100 == image width (valid)
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 100, 100]},
            ],
        )
        path = _write_coco_json(data)
        summary = CocoDatasetAnalyzer(path).get_summary()
        assert summary["out_of_bounds_errors"] == []

    def test_negative_origin_detected(self):
        """A bbox with negative x or y must be flagged as out of bounds."""
        data = _minimal_coco(
            images=[{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [-5, 10, 20, 20]},
            ],
        )
        path = _write_coco_json(data)
        summary = CocoDatasetAnalyzer(path).get_summary()
        assert len(summary["out_of_bounds_errors"]) == 1

    def test_in_bounds_annotation_not_flagged(self):
        """A valid in-bounds bbox must produce zero errors."""
        data = _minimal_coco(
            images=[{"id": 1, "file_name": "img.jpg", "width": 200, "height": 200}],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 50, 50]},
            ],
        )
        path = _write_coco_json(data)
        summary = CocoDatasetAnalyzer(path).get_summary()
        assert summary["out_of_bounds_errors"] == []


class TestCocoDatasetAnalyzerErrors:
    """Tests for error handling on bad inputs."""

    def test_file_not_found(self):
        """Passing a non-existent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CocoDatasetAnalyzer("/nonexistent/path/dataset.json")

    def test_missing_root_keys_raises_key_error(self):
        """A COCO JSON missing mandatory root keys must raise KeyError."""
        # Only 'images' present — 'annotations' and 'categories' missing
        data = {"images": []}
        path = _write_coco_json(data)
        with pytest.raises(KeyError):
            CocoDatasetAnalyzer(path)