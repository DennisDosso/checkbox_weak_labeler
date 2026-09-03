"""
azure_labeling.azure_di_labeler
---------------------------------
Core labeling logic: iterates a directory of images, calls Azure Document
Intelligence with ``prebuilt-layout``, extracts checkbox selectionMarks, and
writes a COCO-format JSON annotation file.

This module mirrors the structure of ``src.local_labeling.local_weak_labeler``
so the two pipelines stay consistent.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from tqdm import tqdm

from src.azure_labeling.azure_config import (
    AZURE_COCO_CATEGORIES,
    AZURE_STATE_TO_CATEGORY_ID,
    AZURE_STATE_TO_LABEL,
    AzureSettings,
)

logger = logging.getLogger(__name__)

_VALID_EXTENSIONS = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]


# ---------------------------------------------------------------------------
# Azure DI client factory
# ---------------------------------------------------------------------------

def build_client(settings: AzureSettings) -> DocumentIntelligenceClient:
    """Instantiate and return an Azure Document Intelligence client."""
    return DocumentIntelligenceClient(
        endpoint=settings.endpoint,
        credential=AzureKeyCredential(settings.key),
    )


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

def analyze_image(
    client: DocumentIntelligenceClient,
    image_path: str,
    settings: AzureSettings,
) -> list[dict[str, Any]]:
    """
    Run Azure DI ``prebuilt-layout`` on one image and return raw checkbox data.

    Parameters
    ----------
    client:
        Authenticated :class:`DocumentIntelligenceClient`.
    image_path:
        Absolute path to the image file.
    settings:
        Active :class:`AzureSettings`.

    Returns
    -------
    list of dicts, each with keys:
        ``state``        – ``"selected"`` or ``"unselected"``
        ``confidence``   – float in [0, 1]
        ``polygon``      – list of floats [x0,y0, x1,y1, ... x3,y3] in pixels
        ``page_width``   – page width in pixels (for bbox normalisation)
        ``page_height``  – page height in pixels (for bbox normalisation)
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    poller = client.begin_analyze_document(
        model_id=settings.model_id,
        body=image_bytes,
        content_type="application/octet-stream",
    )
    result = poller.result()

    checkboxes: list[dict[str, Any]] = []

    if not result.pages:
        return checkboxes

    for page in result.pages:
        page_width = page.width or 0.0
        page_height = page.height or 0.0

        if not page.selection_marks:
            continue

        for mark in page.selection_marks:
            state_str = str(mark.state).lower() if mark.state else "unselected"
            # Normalise to "selected"/"unselected"
            if "select" in state_str and "un" not in state_str:
                state_str = "selected"
            else:
                state_str = "unselected"

            checkboxes.append(
                {
                    "state": state_str,
                    "confidence": float(mark.confidence or 0.0),
                    "polygon": list(mark.polygon) if mark.polygon else [],
                    "page_width": float(page_width),
                    "page_height": float(page_height),
                }
            )

    return checkboxes


# ---------------------------------------------------------------------------
# Bounding-box conversion
# ---------------------------------------------------------------------------

def polygon_to_coco_bbox(
    polygon: list[float],
    page_width: float,
    page_height: float,
    img_width: int,
    img_height: int,
) -> list[float]:
    """
    Convert an Azure DI polygon (8 floats, normalised 0-1 per page unit)
    to a COCO bounding box in absolute pixel coordinates [x, y, w, h].

    Azure DI returns polygon coordinates in inches when the page unit is
    ``inch``, or in pixels when the unit is ``pixel``.  The ``page_width``
    and ``page_height`` values from the API correspond to the same unit as
    the polygon coordinates.

    We normalise to [0, 1] relative to the page, then scale to the actual
    image pixel dimensions.

    Parameters
    ----------
    polygon:
        Flat list of (x, y) pairs: [x0, y0, x1, y1, x2, y2, x3, y3].
    page_width, page_height:
        Page dimensions in the same unit as the polygon coordinates.
    img_width, img_height:
        Actual image dimensions in pixels.

    Returns
    -------
    [x_min_px, y_min_px, width_px, height_px]
    """
    if len(polygon) < 8 or page_width == 0 or page_height == 0:
        return [0.0, 0.0, 0.0, 0.0]

    xs = [polygon[i]     for i in range(0, len(polygon), 2)]
    ys = [polygon[i + 1] for i in range(0, len(polygon), 2)]

    x_min_norm = min(xs) / page_width
    y_min_norm = min(ys) / page_height
    x_max_norm = max(xs) / page_width
    y_max_norm = max(ys) / page_height

    x_min_px = x_min_norm * img_width
    y_min_px = y_min_norm * img_height
    w_px     = (x_max_norm - x_min_norm) * img_width
    h_px     = (y_max_norm - y_min_norm) * img_height

    return [round(x_min_px, 2), round(y_min_px, 2), round(w_px, 2), round(h_px, 2)]


# ---------------------------------------------------------------------------
# Resume helper (identical contract to local_weak_labeler.load_existing_coco)
# ---------------------------------------------------------------------------

def load_existing_coco(output_json: str) -> tuple[dict, set, int, int]:
    """Load an existing COCO JSON file to support --resume functionality."""
    output_path = Path(output_json)
    empty: tuple[dict, set, int, int] = (
        {"images": [], "annotations": [], "categories": AZURE_COCO_CATEGORIES},
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

    images      = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    processed_filenames   = {img["file_name"] for img in images}
    next_image_id         = max((img["id"] for img in images),      default=0) + 1
    next_annotation_id    = max((ann["id"] for ann in annotations),  default=0) + 1

    if "categories" not in coco_data:
        coco_data["categories"] = AZURE_COCO_CATEGORIES

    return coco_data, processed_filenames, next_image_id, next_annotation_id


# ---------------------------------------------------------------------------
# Core labeling function
# ---------------------------------------------------------------------------

def generate_azure_weak_labels(
    input_dir: str,
    output_json: str,
    client: DocumentIntelligenceClient | None,
    settings: AzureSettings,
    recursive: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Scan *input_dir* for images, run Azure DI checkbox inference, and save a
    COCO-format JSON file to *output_json*.

    Parameters
    ----------
    input_dir:
        Directory containing the source images.
    output_json:
        Destination path for the output COCO JSON file.
    client:
        Authenticated :class:`DocumentIntelligenceClient`, or ``None`` when
        ``dry_run=True``.
    settings:
        Active :class:`AzureSettings` configuration.
    recursive:
        If True, scan subdirectories recursively.
    resume:
        If True, load existing output and skip already-processed images.
    dry_run:
        If True, scan and report without running inference or writing files.
    overwrite:
        If True, overwrite the output file without interactive confirmation.
    """
    output_path = Path(output_json)

    # ------------------------------------------------------------------
    # Collect image files
    # ------------------------------------------------------------------
    log_msg = "Scanning directory%s: %s"
    logger.info(log_msg, " (recursive)" if recursive else "", input_dir)

    raw_image_files: list[str] = []
    for ext in _VALID_EXTENSIONS:
        if recursive:
            pattern = os.path.join(input_dir, "**", ext)
            raw_image_files.extend(glob.glob(pattern, recursive=True))
        else:
            pattern = os.path.join(input_dir, ext)
            raw_image_files.extend(glob.glob(pattern))

    # Deduplicate (case-insensitive OS may return dupes)
    seen_basenames: set[str] = set()
    image_files: list[str] = []
    for p in raw_image_files:
        abs_path = os.path.abspath(p)
        basename = os.path.basename(abs_path)
        if basename not in seen_basenames:
            seen_basenames.add(basename)
            image_files.append(abs_path)

    total_found = len(image_files)

    # ------------------------------------------------------------------
    # Dry-run mode
    # ------------------------------------------------------------------
    if dry_run:
        already_processed = 0
        if resume and output_path.exists():
            _, processed_dry, _, _ = load_existing_coco(output_json)
            already_processed = sum(
                1 for p in image_files
                if os.path.basename(p) in processed_dry
            )
        print(f"[DRY RUN] Input directory  : {input_dir}")
        print(f"[DRY RUN] Output file      : {output_json}")
        print(f"[DRY RUN] Azure DI model   : {settings.model_id}")
        print(f"[DRY RUN] Confidence thresh: {settings.confidence_threshold}")
        print(f"[DRY RUN] Images found     : {total_found}")
        if resume and output_path.exists():
            print(f"[DRY RUN] Already processed (skip): {already_processed}")
        print(f"[DRY RUN] Would process    : {total_found - already_processed}")
        print("[DRY RUN] No inference was run. No files were written.")
        return

    # ------------------------------------------------------------------
    # Resume / overwrite / interactive
    # ------------------------------------------------------------------
    if resume:
        coco_output, processed_filenames, image_id_counter, annotation_id_counter = (
            load_existing_coco(output_json)
        )
        if processed_filenames:
            logger.info(
                "Resuming: %d image(s) already processed.",
                len(processed_filenames),
            )
    elif overwrite:
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": AZURE_COCO_CATEGORIES,
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
                logger.info("Aborted. Existing file not modified.")
                return
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": AZURE_COCO_CATEGORIES,
        }
        processed_filenames = set()
        image_id_counter = 1
        annotation_id_counter = 1

    # Filter already-processed images when resuming
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
        logger.info("No new images to process.")
        if resume and coco_output["images"]:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fp:
                json.dump(coco_output, fp, indent=4)
            logger.info("Existing output preserved at: %s", output_json)
        return

    logger.info(
        "Processing %d image(s) with Azure DI model '%s' "
        "(confidence threshold: %.2f) …",
        len(image_files),
        settings.model_id,
        settings.confidence_threshold,
    )

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    from PIL import Image as PILImage  # imported here to keep top-level imports lean

    for img_path in tqdm(image_files, desc="Azure DI Labeling", unit="img"):
        filename = os.path.basename(img_path)

        try:
            with PILImage.open(img_path) as pil_img:
                img_width, img_height = pil_img.size

            checkboxes = analyze_image(client, img_path, settings)  # type: ignore[arg-type]

            # Filter by confidence threshold
            checkboxes = [
                c for c in checkboxes
                if c["confidence"] >= settings.confidence_threshold
            ]

            coco_output["images"].append(
                {
                    "id":        image_id_counter,
                    "file_name": filename,
                    "width":     img_width,
                    "height":    img_height,
                }
            )

            for checkbox in checkboxes:
                state      = checkbox["state"]
                category_id = AZURE_STATE_TO_CATEGORY_ID.get(state, 2)
                label       = AZURE_STATE_TO_LABEL.get(state, "unchecked")

                bbox = polygon_to_coco_bbox(
                    checkbox["polygon"],
                    checkbox["page_width"],
                    checkbox["page_height"],
                    img_width,
                    img_height,
                )
                area = bbox[2] * bbox[3]

                coco_output["annotations"].append(
                    {
                        "id":          annotation_id_counter,
                        "image_id":    image_id_counter,
                        "category_id": category_id,
                        "label":       label,
                        "bbox":        bbox,
                        "area":        round(area, 2),
                        "iscrowd":     0,
                        "score":       checkbox["confidence"],
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

    total_images      = len(coco_output["images"])
    total_annotations = len(coco_output["annotations"])
    logger.info(
        "Done. Images: %d, Annotations: %d. Saved to: %s",
        total_images,
        total_annotations,
        output_json,
    )