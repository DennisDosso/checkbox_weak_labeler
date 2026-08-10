"""
apply_detection_to_one_image.py
--------------------------------
CLI script that runs Roboflow inference on a single image and draws
bounding boxes around detected objects.

Usage
-----
From the project root::

    python scripts/apply_detection_to_one_image.py -i /path/to/image.jpg
    python scripts/apply_detection_to_one_image.py -i /path/to/image.jpg -o /path/to/output.jpg
    python scripts/apply_detection_to_one_image.py -i /path/to/image.jpg -m checkbox-0fyo0/1

If ``--output`` is not specified, the annotated image is displayed interactively
using ``cv2.imshow``.
"""

import argparse
import logging
import sys

import cv2

from src.config import ROBOFLOW_API_KEY, ROBOFLOW_API_URL
from src.weak_labeling.roboflow_client import get_inference_client

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color map (OpenCV uses BGR)
# ---------------------------------------------------------------------------
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "checked": (0, 255, 0),      # Green
    "unchecked": (0, 0, 255),    # Red
    "signature": (255, 0, 0),    # Blue
}
DEFAULT_COLOR: tuple[int, int, int] = (128, 128, 128)  # Grey for unknown classes


# ---------------------------------------------------------------------------
# Core detection + drawing function
# ---------------------------------------------------------------------------

def detect_and_draw(image_path: str, output_path: str | None, model_id: str) -> None:
    """
    Run Roboflow inference on a single image and draw bounding boxes.

    Parameters
    ----------
    image_path:
        Path to the source image file.
    output_path:
        If provided, save the annotated image to this path instead of
        displaying it interactively.
    model_id:
        Roboflow model ID to use for inference (e.g. ``"checkbox-0fyo0/1"``).
    """
    # Load image with OpenCV
    image = cv2.imread(image_path)
    if image is None:
        logger.error(
            "OpenCV could not read the image at: '%s'. "
            "Please check that the file exists and the path is correct.",
            image_path,
        )
        sys.exit(1)

    # Run inference
    logger.info("Running inference with model '%s' on '%s' …", model_id, image_path)
    client = get_inference_client()
    result = client.infer(image, model_id=model_id)

    predictions = result.get("predictions", [])  # type: ignore[union-attr]
    logger.info("Received %d prediction(s).", len(predictions))

    # Draw bounding boxes
    for prediction in predictions:
        center_x: float = prediction["x"]
        center_y: float = prediction["y"]
        width: float = prediction["width"]
        height: float = prediction["height"]
        label: str = prediction["class"]
        confidence: float = prediction["confidence"]

        # Convert center x/y to top-left / bottom-right corner coordinates
        x_min = int(center_x - (width / 2))
        y_min = int(center_y - (height / 2))
        x_max = int(center_x + (width / 2))
        y_max = int(center_y + (height / 2))

        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Draw label + confidence text
        text = f"{label} ({confidence:.2f})"
        cv2.putText(
            image,
            text,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # Output: save or display
    if output_path:
        cv2.imwrite(output_path, image)
        logger.info("Annotated image saved to: '%s'", output_path)
    else:
        cv2.imshow("Roboflow Detections", image)
        print("Press any key on the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Roboflow inference on a single image and draw bounding boxes "
            "around detected objects."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Path where the annotated image will be saved. "
            "If omitted, the image is displayed interactively."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        default="checkbox-0fyo0/1",
        help="Roboflow model ID to use for inference (default: 'checkbox-0fyo0/1').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    detect_and_draw(args.input, args.output, args.model)