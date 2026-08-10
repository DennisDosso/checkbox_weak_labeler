"""
local_labeling.rfdetr_detector
-------------------------------
Simplified RF-DETR model wrapper for local weak labeling.

Loads a RF-DETR Nano model from a ``.pth`` checkpoint file and runs
inference on PIL images, returning plain-dict detection results.

Example
-------
>>> from src.local_labeling.rfdetr_detector import RFDetrDetector
>>>
>>> detector = RFDetrDetector("models/rfdetr-nano.pth", confidence_threshold=0.35)
>>> detector.load()
>>> detections = detector.predict(image)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
from PIL import Image
from rfdetr import RFDETRNano

logger = logging.getLogger(__name__)


class RFDetrDetector:
    """
    RF-DETR Nano model wrapper for local inference.

    Parameters
    ----------
    model_path:
        Path to the ``.pth`` weights file.
    confidence_threshold:
        Minimum confidence score for a detection to be included (default: 0.35).
    device:
        Torch device string (e.g. ``"cpu"``, ``"cuda"``).
        Defaults to ``"cuda"`` if available, otherwise ``"cpu"``.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        device: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Validate that the weights file exists, then load the RF-DETR model.

        Raises
        ------
        FileNotFoundError
            If the weights file does not exist at ``model_path``.
        RuntimeError
            If the model fails to load for any other reason.
        """
        path = Path(self._model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"RF-DETR weights file not found: '{path}'. "
                "Please set 'model_path' in config.local.yaml or via "
                "the LOCAL_LABELER_MODEL_PATH environment variable."
            )

        logger.info(
            "Loading RF-DETR Nano model from '%s' on device '%s' …",
            path,
            self._device,
        )

        self._model = RFDETRNano.from_checkpoint(
            str(path), num_classes=1, device=self._device
        )

        # Attempt inference optimization; gracefully skip if no NVIDIA GPU is present
        try:
            self._model.optimize_for_inference()
            logger.info("Model optimized for inference.")
        except RuntimeError as exc:
            if "NVIDIA" in str(exc):
                logger.warning(
                    "optimize_for_inference() skipped: no NVIDIA GPU found. "
                    "Continuing with unoptimized CPU inference."
                )
            else:
                raise

        logger.info("RF-DETR model loaded successfully.")

    def predict(self, image: Image.Image) -> List[dict]:
        """
        Run RF-DETR inference on *image* and return filtered detections.

        Parameters
        ----------
        image:
            A PIL ``Image`` to run inference on.

        Returns
        -------
        list[dict]
            Each dict has the keys:

            * ``"label"`` (str) — class name
            * ``"confidence"`` (float) — detection confidence, rounded to 5 dp
            * ``"bbox"`` (list[int]) — ``[xmin, ymin, xmax, ymax]`` in pixels

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called before :meth:`predict`.
        """
        if self._model is None:
            raise RuntimeError(
                "Model has not been loaded. Call load() before predict()."
            )

        detections_sv = self._model.predict(
            image,
            threshold=self._confidence_threshold,
        )

        results: List[dict] = []

        if len(detections_sv) == 0:
            return results

        for bbox, conf, cls_id in zip(
            detections_sv.xyxy,
            detections_sv.confidence,
            detections_sv.class_id,
        ):
            try:
                label = self._model.class_names[cls_id]
            except IndexError:
                # class_id out of range — skip (can happen with transformer models)
                logger.debug(
                    "Skipping detection with out-of-range class_id=%s", cls_id
                )
                continue

            results.append(
                {
                    "label": label,
                    "confidence": round(float(conf), 5),
                    "bbox": [int(round(float(c))) for c in bbox],
                }
            )

        return results

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RFDetrDetector("
            f"model_path={self._model_path!r}, "
            f"confidence_threshold={self._confidence_threshold}, "
            f"device={self._device!r})"
        )