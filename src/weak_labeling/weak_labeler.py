import os
import json
import glob
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Import our modularized code
from src.config import CATEGORY_MAPPING, COCO_CATEGORIES
from src.weak_labeling.roboflow_client import get_inference_client

logger = logging.getLogger(__name__)


def convert_to_coco_bbox(x: float, y: float, width: float, height: float) -> list:
    """Helper function to convert center x/y to COCO top-left x/y."""
    x_min = x - (width / 2)
    y_min = y - (height / 2)
    return [x_min, y_min, width, height]


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
    empty = (
        {"images": [], "annotations": [], "categories": COCO_CATEGORIES},
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
        coco_data["categories"] = COCO_CATEGORIES

    return coco_data, processed_filenames, next_image_id, next_annotation_id


def generate_weak_labels(
    input_dir: str,
    output_json: str,
    model_id: str,
    recursive: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    """Scans a directory of images and generates weak COCO annotations via Roboflow.

    Parameters
    ----------
    input_dir:
        Path to the directory containing images to label.
    output_json:
        Path where the resulting COCO JSON file will be saved.
    model_id:
        Roboflow model ID to use for inference.
    recursive:
        If True, scan subdirectories recursively for images.
    resume:
        If True, load existing output and skip already-processed images.
    dry_run:
        If True, scan and report what would be processed without running inference
        or writing any files.
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

    image_files = []
    valid_extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]

    for ext in valid_extensions:
        if recursive:
            search_pattern = os.path.join(input_dir, "**", ext)
            image_files.extend(glob.glob(search_pattern, recursive=True))
        else:
            search_pattern = os.path.join(input_dir, ext)
            image_files.extend(glob.glob(search_pattern))

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
    # Resume / overwrite logic
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
            "categories": COCO_CATEGORIES,
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
            "categories": COCO_CATEGORIES,
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
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(coco_output, f, indent=4)
            logger.info("Existing output preserved at: %s", output_json)
        return

    logger.info(
        "Found %d new image(s). Starting inference with model '%s' ...",
        len(image_files),
        model_id,
    )

    # ------------------------------------------------------------------
    # Inference loop (only reached when not dry_run)
    # ------------------------------------------------------------------
    client = get_inference_client()

    for img_path in tqdm(image_files, desc="Labeling Images", unit="img"):
        filename = os.path.basename(img_path)

        try:
            result = client.infer(img_path, model_id=model_id)

            img_width = result['image']['width']  # type: ignore
            img_height = result['image']['height']  # type: ignore

            coco_output["images"].append({
                "id": image_id_counter,
                "file_name": filename,
                "width": img_width,
                "height": img_height
            })

            for prediction in result.get('predictions', []):  # type: ignore
                label = prediction['class']

                if label in CATEGORY_MAPPING:
                    bbox = convert_to_coco_bbox(
                        prediction['x'], prediction['y'],
                        prediction['width'], prediction['height']
                    )

                    coco_output["annotations"].append({
                        "id": annotation_id_counter,
                        "image_id": image_id_counter,
                        "category_id": CATEGORY_MAPPING[label],
                        "bbox": bbox,
                        "area": prediction['width'] * prediction['height'],
                        "iscrowd": 0,
                        "confidence": prediction['confidence']
                    })

                    annotation_id_counter += 1

            image_id_counter += 1

        except Exception as e:
            tqdm.write(f"Failed to process {filename}. Error: {e}")

    # ------------------------------------------------------------------
    # Save the final output
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=4)

    logger.info("Done! Successfully saved COCO annotations to: %s", output_json)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Weakly label images using a Roboflow model.")
    parser.add_argument("-i", "--input", required=True, help="Path to the directory containing images.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the resulting COCO JSON.")
    parser.add_argument("-m", "--model", default="checkbox-0fyo0/1", help="Roboflow model ID to use for inference.")
    parser.add_argument(
        "-r", "--recursive",
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
            "without running inference or writing any files."
        ),
    )

    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive. Use one or the other.")

    generate_weak_labels(
        args.input,
        args.output,
        args.model,
        recursive=args.recursive,
        resume=args.resume,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )