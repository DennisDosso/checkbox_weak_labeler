"""
local_labeling.local_weak_labeler
----------------------------------
Standalone CLI script that generates COCO-format weak labels for a directory
of images using a locally-loaded RF-DETR Nano model.

Usage
-----
From the project root::

    python -m src.local_labeling.local_weak_labeler \
        -i /path/to/images \
        -o /path/to/output/annotations.json \
        [--confidence 0.40] \
        [--config config.local.yaml] \
        [--recursive] \
        [--resume] \
        [--overwrite] \
        [--dry-run]

Environment variables (all optional, prefix ``LOCAL_LABELER_``):
    LOCAL_LABELER_MODEL_PATH        Path to the .pth weights file
    LOCAL_LABELER_CONFIDENCE_THRESHOLD  Confidence threshold (0.0-1.0)
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
# Resume helper
# ---------------------------------------------------------------------------

def load_existing_coco(output_json: str) -> tuple[dict, set, int, int]:
    """Load an existing COCO JSON file to support resume functionality.

    Parameters
    ----------
    output_json:
        Path to the existing COCO JSON file.

    Returns
    -------
    tuple of:
        - coco_output (dict): the loaded COCO structure (or empty skeleton)
        - processed_filenames (set): set of file_name values already in images
        - next_image_id (int): next available image id (max existing + 1, or 1)
        - next_annotation_id (int): next available annotation id (max existing + 1, or 1)
    """
    output_path = Path(output_json)
    empty: tuple[dict, set, int, int] = (
        {"images": [], "annotations": [], "categories": LOCAL_COCO_CATEGORIES},
        set(),
        1,
        1,
    )

    if not output_path.exists():
        return empty

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Could not read existing output file '%s': %s. Starting fresh.",
            output_json,
            exc,
        )
        return empty

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    processed_filenames = {img["file_name"] for img in images}
    next_image_id = max((img["id"] for img in images), default=0) + 1
    next_annotation_id = max((ann["id"] for ann in annotations), default=0) + 1

    if "categories" not in coco_data:
        coco_data["categories"] = LOCAL_COCO_CATEGORIES

    return coco_data, processed_filenames, next_image_id, next_annotation_id


# ---------------------------------------------------------------------------
# Core labeling function
# ---------------------------------------------------------------------------

def generate_local_weak_labels(
    input_dir: str,
    output_json: str,
    detector: RFDetrDetector | None,
    settings: LocalSettings,
    recursive: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
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
        A loaded :class:`RFDetrDetector` instance, or None when dry_run=True.
    settings:
        The active :class:`LocalSettings` configuration.
    recursive:
        If True, scan subdirectories recursively for images.
    resume:
        If True, load existing output and skip already-processed images.
    dry_run:
        If True, scan and report what would be processed without loading the
        model, running inference, or writing any files.
    overwrite:
        If True, overwrite existing output file without interactive confirmation.
    """
    output_path = Path(output_json)

    # ------------------------------------------------------------------
    # Collect image files
    # ------------------------------------------------------------------
    if recursive:
        logger.info("Scanning directory (recursive): %s", input_dir)
    else:
        logger.info("Scanning directory: %s", input_dir)

    image_files: list[str] = []
    for ext in _VALID_EXTENSIONS:
        if recursive:
            pattern = os.path.join(input_dir, "**", ext)
            image_files.extend(glob.glob(pattern, recursive=True))
        else:
            pattern = os.path.join(input_dir, ext)
            image_files.extend(glob.glob(pattern))

    total_found = len(image_files)

    # ------------------------------------------------------------------
    # Dry-run mode: report and exit without inference or file writes
    # ------------------------------------------------------------------
    if dry_run:
        already_processed = 0
        if resume and output_path.exists():
            _, processed_filenames_dry, _, _ = load_existing_coco(output_json)
            already_processed = sum(
                1 for p in image_files
                if os.path.basename(p) in processed_filenames_dry
            )

        would_process = total_found - already_processed

        print(f"[DRY RUN] Input directory: {input_dir}")
        print(f"[DRY RUN] Output file: {output_json}")
        print(f"[DRY RUN] Images found: {total_found}")
        if resume and output_path.exists():
            print(f"[DRY RUN] Already processed (would skip): {already_processed}")
        print(f"[DRY RUN] Images that would be processed: {would_process}")
        print("[DRY RUN] No inference was run. No files were written.")
        return

    # ------------------------------------------------------------------
    # Resume / overwrite / interactive logic
    # ------------------------------------------------------------------
    if resume:
        coco_output, processed_filenames, image_id_counter, annotation_id_counter = (
            load_existing_coco(output_json)
        )
        if processed_filenames:
            logger.info(
                "Resuming from existing output: found %d already-processed image(s).",
                len(processed_filenames),
            )
        else:
            logger.info("--resume set but no existing output found. Starting fresh.")
    elif overwrite:
        # Proceed without interactive confirmation
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": LOCAL_COCO_CATEGORIES,
        }
        processed_filenames: set = set()
        image_id_counter = 1
        annotation_id_counter = 1
    else:
        if output_path.exists():
            answer = input(
                f"Output file '{output_json}' already exists. Overwrite? [y/N] "
            ).strip().lower()
            if answer != "y":
                logger.info("Aborted by user. Existing output file was not modified.")
                return
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": LOCAL_COCO_CATEGORIES,
        }
        processed_filenames = set()
        image_id_counter = 1
        annotation_id_counter = 1

    # Filter out already-processed images when resuming
    if processed_filenames:
        original_count = len(image_files)
        image_files = [
            p for p in image_files
            if os.path.basename(p) not in processed_filenames
        ]
        skipped = original_count - len(image_files)
        if skipped:
            logger.info("Skipping %d already-processed image(s).", skipped)

    if not image_files:
        logger.info("No new JPG or PNG images to process.")
        if resume and coco_output["images"]:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fp:
                json.dump(coco_output, fp, indent=4)
            logger.info("Existing output preserved at: %s", output_json)
        return

    logger.info(
        "Found %d new image(s). Running local RF-DETR inference "
        "(confidence threshold: %s) ...",
        len(image_files),
        settings.confidence_threshold,
    )

    # ------------------------------------------------------------------
    # Inference loop (only reached when not dry_run)
    # ------------------------------------------------------------------
    for img_path in tqdm(image_files, desc="Labeling Images", unit="img"):
        filename = os.path.basename(img_path)

        try:
            with Image.open(img_path) as pil_image:
                img_width, img_height = pil_image.size
                pil_image_rgb = pil_image.convert("RGB")

            detections = detector.predict(pil_image_rgb)  # type: ignore[union-attr]

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

                if label != "signature":
                    continue

                xmin, ymin, xmax, ymax = detection["bbox"]

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
            tqdm.write(f"[ERROR] Failed to process '{filename}': {exc}")

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(coco_output, fp, indent=4)

    total_images = len(coco_output["images"])
    total_annotations = len(coco_output["annotations"])
    logger.info(
        "Done! Total images in output: %d, total annotations: %d.",
        total_images,
        total_annotations,
    )
    logger.info("COCO annotations saved to: %s", output_json)


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
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively scan subdirectories for images.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume from an existing output file, skipping already-processed images. "
            "If the output file does not exist, starts fresh."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the output file without interactive confirmation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help=(
            "Simulate execution: scan images and report what would be processed "
            "without loading the model, running inference, or writing any files."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Validate mutually exclusive flags
    if args.resume and args.overwrite:
        import sys as _sys
        print(
            "error: --resume and --overwrite are mutually exclusive. Use one or the other.",
            file=_sys.stderr,
        )
        _sys.exit(2)

    # Allow --config to override the YAML path via env var consumed by LocalSettings
    if args.config:
        os.environ["LOCAL_LABELER_CONFIG_PATH"] = args.config

    # Load settings (env vars + YAML + defaults)
    # model_path is supplied by LOCAL_LABELER_MODEL_PATH or YAML config.
    settings = LocalSettings()  # type: ignore[call-arg]

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

    # In dry-run mode, skip loading the detector entirely
    if args.dry_run:
        detector = None
    else:
        detector = RFDetrDetector(
            model_path=settings.model_path,
            confidence_threshold=settings.confidence_threshold,
            device=settings.device,
        )
        detector.load()

    # Run weak labeling
    generate_local_weak_labels(
        args.input,
        args.output,
        detector,
        settings,
        recursive=args.recursive,
        resume=args.resume,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )