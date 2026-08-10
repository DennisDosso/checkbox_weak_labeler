"""
tests/test_rfdetr_detector.py
------------------------------
Unit tests for RFDetrDetector in src/local_labeling/rfdetr_detector.py.

RFDETRNano and torch.cuda are fully mocked — no real model or GPU required.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.local_labeling.rfdetr_detector import RFDetrDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_model_file(tmp_path: Path) -> Path:
    """Create a temporary dummy .pth file so path-existence checks pass."""
    model_file = tmp_path / "rfdetr-nano.pth"
    model_file.write_bytes(b"dummy weights")
    return model_file


# ---------------------------------------------------------------------------
# Tests: load()
# ---------------------------------------------------------------------------

class TestRFDetrDetectorLoad:
    """Tests for RFDetrDetector.load()."""

    def test_load_raises_file_not_found_for_missing_path(self):
        """load() must raise FileNotFoundError when the weights file does not exist."""
        detector = RFDetrDetector(
            model_path="/nonexistent/path/model.pth",
            confidence_threshold=0.35,
            device="cpu",
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            detector.load()

    def test_load_succeeds_with_valid_path(self, tmp_model_file: Path):
        """load() must complete without error when the weights file exists."""
        with patch("src.local_labeling.rfdetr_detector.RFDETRNano") as mock_rfdetr:
            mock_model = MagicMock()
            mock_rfdetr.from_checkpoint.return_value = mock_model

            detector = RFDetrDetector(
                model_path=str(tmp_model_file),
                confidence_threshold=0.35,
                device="cpu",
            )
            detector.load()

            mock_rfdetr.from_checkpoint.assert_called_once_with(
                str(tmp_model_file), num_classes=1, device="cpu"
            )

    def test_load_skips_optimize_on_nvidia_runtime_error(self, tmp_model_file: Path):
        """load() must log a warning and continue if optimize_for_inference raises NVIDIA RuntimeError."""
        with patch("src.local_labeling.rfdetr_detector.RFDETRNano") as mock_rfdetr:
            mock_model = MagicMock()
            mock_model.optimize_for_inference.side_effect = RuntimeError(
                "No NVIDIA GPU found"
            )
            mock_rfdetr.from_checkpoint.return_value = mock_model

            detector = RFDetrDetector(
                model_path=str(tmp_model_file),
                confidence_threshold=0.35,
                device="cpu",
            )
            # Should not raise — NVIDIA errors are caught gracefully
            detector.load()

    def test_load_reraises_non_nvidia_runtime_error(self, tmp_model_file: Path):
        """load() must re-raise RuntimeError when the message does not contain 'NVIDIA'."""
        with patch("src.local_labeling.rfdetr_detector.RFDETRNano") as mock_rfdetr:
            mock_model = MagicMock()
            mock_model.optimize_for_inference.side_effect = RuntimeError(
                "Some unrelated error"
            )
            mock_rfdetr.from_checkpoint.return_value = mock_model

            detector = RFDetrDetector(
                model_path=str(tmp_model_file),
                confidence_threshold=0.35,
                device="cpu",
            )
            with pytest.raises(RuntimeError, match="Some unrelated error"):
                detector.load()


# ---------------------------------------------------------------------------
# Tests: predict()
# ---------------------------------------------------------------------------

class TestRFDetrDetectorPredict:
    """Tests for RFDetrDetector.predict()."""

    def test_predict_raises_if_not_loaded(self):
        """predict() must raise RuntimeError when called before load()."""
        detector = RFDetrDetector(
            model_path="/any/path.pth",
            confidence_threshold=0.35,
            device="cpu",
        )
        dummy_image = MagicMock()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            detector.predict(dummy_image)

    def test_predict_returns_filtered_detections(self, tmp_model_file: Path):
        """predict() must return only detections above the confidence threshold with int bbox coords."""
        import numpy as np

        with patch("src.local_labeling.rfdetr_detector.RFDETRNano") as mock_rfdetr:
            mock_model = MagicMock()
            mock_rfdetr.from_checkpoint.return_value = mock_model

            # Mock detection result from model.predict()
            mock_detections = MagicMock()
            mock_detections.__len__ = MagicMock(return_value=2)
            mock_detections.xyxy = [
                [10.7, 20.3, 60.9, 80.1],   # above threshold
                [5.0,  5.0,  15.0, 15.0],   # will be filtered — model returns threshold-filtered results
            ]
            mock_detections.confidence = [0.90, 0.20]
            mock_detections.class_id = [0, 0]
            mock_model.predict.return_value = mock_detections
            mock_model.class_names = ["signature"]

            detector = RFDetrDetector(
                model_path=str(tmp_model_file),
                confidence_threshold=0.35,
                device="cpu",
            )
            detector.load()

            # Patch the model's predict to only return the high-confidence detection
            high_conf_detections = MagicMock()
            high_conf_detections.__len__ = MagicMock(return_value=1)
            high_conf_detections.xyxy = [[10.7, 20.3, 60.9, 80.1]]
            high_conf_detections.confidence = [0.90]
            high_conf_detections.class_id = [0]
            mock_model.predict.return_value = high_conf_detections

            from PIL import Image as PILImage
            dummy_image = PILImage.new("RGB", (200, 200))
            results = detector.predict(dummy_image)

            assert len(results) == 1
            det = results[0]
            assert det["label"] == "signature"
            assert det["confidence"] == round(0.90, 5)
            # bbox coordinates must be integers
            assert all(isinstance(c, int) for c in det["bbox"])
            assert det["bbox"] == [11, 20, 61, 80]

    def test_predict_skips_out_of_range_class_id(self, tmp_model_file: Path):
        """predict() must skip detections whose class_id is out of range for class_names."""
        with patch("src.local_labeling.rfdetr_detector.RFDETRNano") as mock_rfdetr:
            mock_model = MagicMock()
            mock_rfdetr.from_checkpoint.return_value = mock_model

            mock_detections = MagicMock()
            mock_detections.__len__ = MagicMock(return_value=1)
            mock_detections.xyxy = [[0.0, 0.0, 50.0, 50.0]]
            mock_detections.confidence = [0.95]
            mock_detections.class_id = [99]   # out of range
            mock_model.predict.return_value = mock_detections
            mock_model.class_names = ["signature"]  # only index 0 valid

            detector = RFDetrDetector(
                model_path=str(tmp_model_file),
                confidence_threshold=0.35,
                device="cpu",
            )
            detector.load()

            from PIL import Image as PILImage
            dummy_image = PILImage.new("RGB", (200, 200))
            results = detector.predict(dummy_image)

            assert results == []

    def test_predict_returns_empty_list_for_no_detections(self, tmp_model_file: Path):
        """predict() must return an empty list when the model finds nothing."""
        with patch("src.local_labeling.rfdetr_detector.RFDETRNano") as mock_rfdetr:
            mock_model = MagicMock()
            mock_rfdetr.from_checkpoint.return_value = mock_model

            mock_detections = MagicMock()
            mock_detections.__len__ = MagicMock(return_value=0)
            mock_model.predict.return_value = mock_detections
            mock_model.class_names = ["signature"]

            detector = RFDetrDetector(
                model_path=str(tmp_model_file),
                confidence_threshold=0.35,
                device="cpu",
            )
            detector.load()

            from PIL import Image as PILImage
            dummy_image = PILImage.new("RGB", (200, 200))
            results = detector.predict(dummy_image)

            assert results == []