"""
tests/test_bbox_conversion.py
------------------------------
Unit tests for bounding-box conversion functions used in both the remote
weak labeler (center x/y/w/h -> COCO) and the local weak labeler (xyxy -> COCO).
"""

import pytest

from src.weak_labeling.weak_labeler import convert_to_coco_bbox


# ---------------------------------------------------------------------------
# Helpers — inline the local xyxy->COCO conversion (mirrors local_weak_labeler.py)
# ---------------------------------------------------------------------------

def convert_xyxy_to_coco_bbox(xmin: int, ymin: int, xmax: int, ymax: int) -> list:
    """Convert RF-DETR xyxy bbox to COCO [x_min, y_min, width, height]."""
    width = xmax - xmin
    height = ymax - ymin
    return [xmin, ymin, width, height]


# ---------------------------------------------------------------------------
# Tests for convert_to_coco_bbox (remote weak labeler — center x/y format)
# ---------------------------------------------------------------------------

class TestConvertToCocobbox:
    """Tests for weak_labeler.convert_to_coco_bbox (center-based format)."""

    def test_basic_conversion(self):
        """Standard center x/y/w/h -> COCO [x_min, y_min, w, h]."""
        result = convert_to_coco_bbox(x=100.0, y=200.0, width=50.0, height=40.0)
        assert result == [75.0, 180.0, 50.0, 40.0]

    def test_width_and_height_preserved(self):
        """Width and height must be passed through unchanged."""
        result = convert_to_coco_bbox(x=60.0, y=80.0, width=30.0, height=20.0)
        assert result[2] == 30.0
        assert result[3] == 20.0

    def test_zero_bbox(self):
        """All-zero input should produce all-zero output."""
        result = convert_to_coco_bbox(x=0.0, y=0.0, width=0.0, height=0.0)
        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_float_coordinates_remain_float(self):
        """Remote labeler returns float coordinates — output must be float."""
        result = convert_to_coco_bbox(x=10.5, y=20.3, width=8.2, height=6.4)
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
        # Verify correctness
        assert abs(result[0] - (10.5 - 8.2 / 2)) < 1e-9
        assert abs(result[1] - (20.3 - 6.4 / 2)) < 1e-9

    def test_large_values(self):
        """Should handle large image-scale coordinates without overflow."""
        result = convert_to_coco_bbox(x=1920.0, y=1080.0, width=200.0, height=100.0)
        assert result == [1820.0, 1030.0, 200.0, 100.0]


# ---------------------------------------------------------------------------
# Tests for xyxy -> COCO conversion (local weak labeler — RF-DETR format)
# ---------------------------------------------------------------------------

class TestConvertXyxyToCocobbox:
    """Tests for the xyxy->COCO conversion used in local_weak_labeler.py."""

    def test_basic_conversion(self):
        """Standard xyxy -> COCO [x_min, y_min, w, h]."""
        result = convert_xyxy_to_coco_bbox(10, 20, 60, 80)
        assert result == [10, 20, 50, 60]

    def test_zero_area_bbox(self):
        """A degenerate bbox where min == max should produce zero width/height."""
        result = convert_xyxy_to_coco_bbox(0, 0, 0, 0)
        assert result == [0, 0, 0, 0]

    def test_integer_coordinates(self):
        """Local labeler uses rounded int coords — output must be int."""
        result = convert_xyxy_to_coco_bbox(5, 10, 15, 25)
        assert all(isinstance(v, int) for v in result)

    def test_width_height_correct(self):
        """Width and height must equal xmax-xmin and ymax-ymin respectively."""
        xmin, ymin, xmax, ymax = 30, 40, 130, 190
        result = convert_xyxy_to_coco_bbox(xmin, ymin, xmax, ymax)
        assert result[2] == xmax - xmin   # width
        assert result[3] == ymax - ymin   # height

    def test_top_left_corner_preserved(self):
        """x_min and y_min in the output must match the input xmin/ymin."""
        result = convert_xyxy_to_coco_bbox(7, 13, 50, 80)
        assert result[0] == 7
        assert result[1] == 13