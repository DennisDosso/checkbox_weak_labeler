"""
local_labeling.local_weak_labeler
----------------------------------
Standalone CLI script that generates COCO-format weak labels for a directory
of images using a locally-loaded RF-DETR Nano model.

Usage
-----
From the project root::

    python -m src.local_labeling.local_weak_labeler \\
        -i /path/to/images \\
        -o /path/to/output/annotations.json \\
        [--confidence 0.40] \\
        [--config config.local.yaml]

Environment variables (all optional, prefix ``LOCAL_LABELER_``):
    LOCAL_LABELER_MODEL_PATH        Path to the .pth weights file
    LOCAL_LABELER_CONFIDENCE_THRESHOLD  Confidence threshold (0.0–1.0)
    LOCAL_LABELER_CONFIG_PATH       Path to the YAML config file
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports work when running as a
# script directly (python -m src.local_labeling.local_weak_labeler) as well
# as when run via `python src/local_labeling/local_weak_labeler.py`.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.local_labeling.local_config import LOCAL_COCO_CATEGORIES, LocalSettings  # noqa: E402
from src.local_labeling.rfdetr_detector import RFDetrDetector  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VALID_EXTENSIONS = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]


# ---------------------------------------------------------------------------
# Core labeling function
# ---------------------------------------------------------------------------

def generate_local_weak_labels(
    input_dir: str,
    output_json: str,
    detector: RFDetrDetector,
    settings: LocalSettings,
) -> None:
    """
    Scan *input_dir* for images, run local RF-DETR inference, and save a
    COCO-format JSON file to *output_json*.

    Parameters
    ----------
    input_dir:
        Directory containing the source images.
    output_json:
        Destination path for the output COCO JSON file.
    detector:
        A loaded :class:`RFDetrDetector` instance.
    settings:
        The active :class:`LocalSettings` configuration.
    """
    coco_output: dict = {
        "images": [],
        "annotations": [],
        "categories": LOCAL_COCO_CATEGORIES,
    }

    image_id_counter = 1
    annotation_id_counter = 1

    # ------------------------------------------------------------------
    # Collect image files
    # ------------------------------------------------------------------
    print(f"Scanning directory: {input_dir}")
    image_files: list[str] = []
    for ext in _VALID_EXTENSIONS:
        pattern = os.path.join(input_dir, ext)
        image_files.extend(glob.glob(pattern))

    if not image_files:
        print("No JPG or PNG images found in the specified directory.")
        return

    print(
        f"Found {len(image_files)} image(s). "
        f"Running local RF-DETR inference "
        f"(confidence threshold: {settings.confidence_threshold}) …"
    )

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    for img_path in tqdm(image_files, desc="Labeling Images", unit="img"):
        filename = os.path.basename(img_path)

        try:
            with Image.open(img_path) as pil_image:
                img_width, img_height = pil_image.size
                # Keep the image open during inference
                pil_image_rgb = pil_image.convert("RGB")

            detections = detector.predict(pil_image_rgb)

            coco_output["images"].append(
                {
                    "id": image_id_counter,
                    "file_name": filename,
                    "width": img_width,
                    "height": img_height,
                }
            )

            for detection in detections:
                label = detection["label"]

                # Only include signature detections (category_id = 0)
                if label != "signature":
                    continue

                xmin, ymin, xmax, ymax = detection["bbox"]

                # Convert RF-DETR [xmin, ymin, xmax, ymax] → COCO [x, y, w, h]
                coco_width = xmax - xmin
                coco_height = ymax - ymin
                coco_bbox = [xmin, ymin, coco_width, coco_height]

                coco_output["annotations"].append(
                    {
                        "id": annotation_id_counter,
                        "image_id": image_id_counter,
                        "category_id": 0,
                        "bbox": coco_bbox,
                        "area": coco_width * coco_height,
                        "iscrowd": 0,
                        "confidence": detection["confidence"],
                    }
                )
                annotation_id_counter += 1

            image_id_counter += 1

        except Exception as exc:
            # Use tqdm.write so error messages don't break the progress bar
            tqdm.write(f"[ERROR] Failed to process '{filename}': {exc}")

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(coco_output, fp, indent=4)

    total_images = image_id_counter - 1
    total_annotations = annotation_id_counter - 1
    print(
        f"\nDone! Processed {total_images} image(s), "
        f"generated {total_annotations} annotation(s)."
    )
    print(f"COCO annotations saved to: {output_json}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate COCO-format weak labels for a directory of images "
            "using a local RF-DETR Nano model."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the directory containing images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to save the resulting COCO JSON file.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help=(
            "Override the confidence threshold from config "
            "(float between 0.0 and 1.0)."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the YAML configuration file "
            "(default: config.local.yaml in project root)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Allow --config to override the YAML path via env var consumed by LocalSettings
    if args.config:
        os.environ["LOCAL_LABELER_CONFIG_PATH"] = args.config

    # Load settings (env vars + YAML + defaults)
    settings = LocalSettings()

    # CLI --confidence flag overrides config value
    if args.confidence is not None:
        settings = LocalSettings(
            model_path=settings.model_path,
            confidence_threshold=args.confidence,
            device=settings.device,
        )

    logger.info("Using model: %s", settings.model_path)
    logger.info("Confidence threshold: %s", settings.confidence_threshold)
    logger.info("Device: %s", settings.device)

    # Instantiate and load the detector
    detector = RFDetrDetector(
        model_path=settings.model_path,
        confidence_threshold=settings.confidence_threshold,
        device=settings.device,
    )
    detector.load()

    # Run weak labeling
    generate_local_weak_labels(args.input, args.output, detector, settings)