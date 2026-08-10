"""
src.utils.vertical_merge_coco
------------------------------
Utility to perform a "vertical merge" of two COCO datasets containing
**different images** but **identical or compatible category schemas**.

The output is a single flat directory with all images from both datasets
plus a merged ``_annotations.coco.json``.

This differs from ``merge_coco.py`` ("horizontal merge"), which handles the
same images annotated with different classes.

Usage
-----
From the project root::

    python -m src.utils.vertical_merge_coco \\
        --input1-images /path/to/dataset1/images \\
        --input1-json   /path/to/dataset1/_annotations.coco.json \\
        --input2-images /path/to/dataset2/images \\
        --input2-json   /path/to/dataset2/_annotations.coco.json \\
        --output        /path/to/output_dir
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_coco_file(path: str) -> dict:
    """Load and validate a COCO JSON file.

    Parameters
    ----------
    path:
        Filesystem path to the COCO JSON file.

    Returns
    -------
    dict
        Parsed COCO data.

    Raises
    ------
    SystemExit
        If the file does not exist or is not valid JSON (exits with code 1).
    """
    p = Path(path)
    if not p.exists():
        print(f"error: input JSON file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in '{path}': {exc}", file=sys.stderr)
        sys.exit(1)

    return data


def validate_image_dir(path: str) -> Path:
    """Validate that an image directory exists.

    Parameters
    ----------
    path:
        Filesystem path to the image directory.

    Returns
    -------
    Path
        Resolved Path object.

    Raises
    ------
    SystemExit
        If the directory does not exist (exits with code 1).
    """
    p = Path(path)
    if not p.is_dir():
        print(f"error: input image directory not found: {path}", file=sys.stderr)
        sys.exit(1)
    return p


# ---------------------------------------------------------------------------
# Category validation
# ---------------------------------------------------------------------------

def validate_and_merge_categories(
    categories1: list[dict],
    categories2: list[dict],
) -> list[dict]:
    """Validate that two category lists are compatible and return the merged union.

    Category identity is determined by ``id``. Two categories with the same
    ``id`` must have identical ``name`` and ``supercategory`` values. Categories
    that only appear in one file are allowed (additive).

    Parameters
    ----------
    categories1:
        Category list from dataset 1.
    categories2:
        Category list from dataset 2.

    Returns
    -------
    list of dict
        Union of all categories from both datasets, sorted by ``id``.

    Raises
    ------
    SystemExit
        If a category ``id`` has conflicting ``name`` or ``supercategory``
        between the two datasets (exits with code 1).
    """
    by_id: dict[int, dict] = {}

    for cat in categories1:
        by_id[cat["id"]] = dict(cat)

    for cat in categories2:
        cat_id = cat["id"]
        if cat_id in by_id:
            existing = by_id[cat_id]
            if existing["name"] != cat["name"]:
                print(
                    f"error: category id {cat_id} has conflicting names: "
                    f"'{existing['name']}' vs '{cat['name']}'.",
                    file=sys.stderr,
                )
                sys.exit(1)
            existing_super = existing.get("supercategory", "")
            new_super = cat.get("supercategory", "")
            if existing_super != new_super:
                print(
                    f"error: category id {cat_id} ('{cat['name']}') has conflicting "
                    f"supercategory: '{existing_super}' vs '{new_super}'.",
                    file=sys.stderr,
                )
                sys.exit(1)
            # Compatible — keep existing entry
        else:
            by_id[cat_id] = dict(cat)

    return sorted(by_id.values(), key=lambda c: c["id"])


# ---------------------------------------------------------------------------
# Filename conflict detection
# ---------------------------------------------------------------------------

def detect_filename_conflicts(
    images1: list[dict],
    images2: list[dict],
) -> set[str]:
    """Find file_name values that appear in both datasets.

    Parameters
    ----------
    images1:
        Image list from dataset 1.
    images2:
        Image list from dataset 2.

    Returns
    -------
    set of str
        Set of conflicting file_name values.
    """
    names1 = {img["file_name"] for img in images1}
    names2 = {img["file_name"] for img in images2}
    return names1 & names2


# ---------------------------------------------------------------------------
# Image copying and ID mapping
# ---------------------------------------------------------------------------

def copy_images_and_build_map(
    images: list[dict],
    image_dir: Path,
    output_dir: Path,
    conflicts: set[str],
    dataset_suffix: str,
) -> tuple[list[dict], dict[int, int]]:
    """Copy image files to the output directory and build an old->new ID map.

    For filenames in ``conflicts``, the file is renamed with the given
    ``dataset_suffix`` before copying (e.g. ``doc.png`` -> ``doc_1.png``).
    For non-conflicting filenames, the file is copied as-is.

    Images whose source file is not found are skipped with a warning.

    Parameters
    ----------
    images:
        Image entries from the COCO JSON for this dataset.
    image_dir:
        Directory containing the source image files.
    output_dir:
        Destination directory for copied images.
    conflicts:
        Set of file_name values that conflict between the two datasets.
    dataset_suffix:
        Suffix to append to conflicting filenames (e.g. ``"1"`` or ``"2"``).

    Returns
    -------
    output_images:
        List of new image dicts (with updated ``file_name``); IDs are
        temporary (original) — sequential IDs are assigned later.
    old_to_temp_map:
        Dict mapping original image id -> temporary image id (identity map
        here; sequential reassignment happens after both datasets are combined).
    skipped_ids:
        Set of original image IDs that were skipped (file not found).
    """
    output_images: list[dict] = []
    skipped_ids: set[int] = set()

    for img in images:
        fname = img["file_name"]
        src_path = image_dir / fname

        if not src_path.exists():
            logger.warning(
                "Image file not found in dataset directory, skipping: %s", src_path
            )
            skipped_ids.add(img["id"])
            continue

        if fname in conflicts:
            stem = Path(fname).stem
            ext = Path(fname).suffix
            new_fname = f"{stem}_{dataset_suffix}{ext}"
            logger.warning(
                "Filename conflict '%s' — renamed to '%s'",
                fname,
                new_fname,
            )
        else:
            new_fname = fname

        dest_path = output_dir / new_fname
        shutil.copy2(src_path, dest_path)
        logger.debug("Copied: %s -> %s", src_path, dest_path)

        new_img = dict(img)
        new_img["file_name"] = new_fname
        output_images.append(new_img)

    return output_images, skipped_ids


# ---------------------------------------------------------------------------
# Sequential ID reassignment
# ---------------------------------------------------------------------------

def assign_sequential_image_ids(
    images1: list[dict],
    images2: list[dict],
) -> tuple[list[dict], dict[int, int], dict[int, int]]:
    """Combine two image lists and assign new sequential IDs from 1.

    Parameters
    ----------
    images1:
        Processed image list from dataset 1 (with updated file_names).
    images2:
        Processed image list from dataset 2 (with updated file_names).

    Returns
    -------
    global_images:
        Combined image list with new sequential IDs.
    id_map1:
        Old image id (dataset 1) -> new global image id.
    id_map2:
        Old image id (dataset 2) -> new global image id.
    """
    id_map1: dict[int, int] = {}
    id_map2: dict[int, int] = {}
    global_images: list[dict] = []

    new_id = 1
    for img in images1:
        old_id = img["id"]
        new_img = dict(img)
        new_img["id"] = new_id
        id_map1[old_id] = new_id
        global_images.append(new_img)
        new_id += 1

    for img in images2:
        old_id = img["id"]
        new_img = dict(img)
        new_img["id"] = new_id
        id_map2[old_id] = new_id
        global_images.append(new_img)
        new_id += 1

    return global_images, id_map1, id_map2


# ---------------------------------------------------------------------------
# Annotation merging
# ---------------------------------------------------------------------------

def merge_annotations(
    annotations1: list[dict],
    annotations2: list[dict],
    id_map1: dict[int, int],
    id_map2: dict[int, int],
    skipped_ids1: set[int],
    skipped_ids2: set[int],
) -> list[dict]:
    """Merge annotations from both datasets, remapping image IDs.

    Annotations whose image was skipped (file not found) are excluded.

    Parameters
    ----------
    annotations1:
        Annotation list from dataset 1.
    annotations2:
        Annotation list from dataset 2.
    id_map1:
        Old image id (dataset 1) -> new global image id.
    id_map2:
        Old image id (dataset 2) -> new global image id.
    skipped_ids1:
        Set of original image IDs from dataset 1 that were skipped.
    skipped_ids2:
        Set of original image IDs from dataset 2 that were skipped.

    Returns
    -------
    list of annotation dicts with new sequential IDs starting from 1.
    """
    merged: list[dict] = []
    ann_id = 1

    for ann in annotations1:
        old_img_id = ann.get("image_id")
        if old_img_id in skipped_ids1:
            logger.debug(
                "Skipping annotation id=%d (image id=%d was not found).",
                ann["id"],
                old_img_id,
            )
            continue
        new_ann = dict(ann)
        new_ann["id"] = ann_id
        new_ann["image_id"] = id_map1.get(old_img_id, old_img_id)
        merged.append(new_ann)
        ann_id += 1

    for ann in annotations2:
        old_img_id = ann.get("image_id")
        if old_img_id in skipped_ids2:
            logger.debug(
                "Skipping annotation id=%d (image id=%d was not found).",
                ann["id"],
                old_img_id,
            )
            continue
        new_ann = dict(ann)
        new_ann["id"] = ann_id
        new_ann["image_id"] = id_map2.get(old_img_id, old_img_id)
        merged.append(new_ann)
        ann_id += 1

    return merged


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(
    input1_images_path: str,
    input2_images_path: str,
    coco1: dict,
    coco2: dict,
    conflict_count: int,
    global_images: list[dict],
    merged_annotations: list[dict],
    global_categories: list[dict],
    output_dir: str,
) -> None:
    """Print the vertical merge summary report to stdout.

    Parameters
    ----------
    input1_images_path:
        Path to dataset 1 image directory (for display).
    input2_images_path:
        Path to dataset 2 image directory (for display).
    coco1:
        Raw COCO data from dataset 1.
    coco2:
        Raw COCO data from dataset 2.
    conflict_count:
        Number of filename conflicts that were resolved.
    global_images:
        Final merged image list.
    merged_annotations:
        Final merged annotation list.
    global_categories:
        Final merged category list.
    output_dir:
        Path to the output directory.
    """
    n1_images = len(coco1.get("images", []))
    n1_annotations = len(coco1.get("annotations", []))
    n2_images = len(coco2.get("images", []))
    n2_annotations = len(coco2.get("annotations", []))
    cat_names = ", ".join(cat["name"] for cat in global_categories)

    print("")
    print("=== Vertical COCO Merge Summary ===")
    print(
        f"Dataset 1       : {input1_images_path}"
        f" ({n1_images} images, {n1_annotations} annotations)"
    )
    print(
        f"Dataset 2       : {input2_images_path}"
        f" ({n2_images} images, {n2_annotations} annotations)"
    )
    print(
        f"Filename conflicts : {conflict_count}"
        f" (renamed with _1 / _2 suffix)"
    )
    print(f"Total images    : {len(global_images)}")
    print(f"Total annotations: {len(merged_annotations)}")
    print(f"Categories      : {len(global_categories)} ({cat_names})")
    print(f"Output directory: {output_dir}")


# ---------------------------------------------------------------------------
# Core merge function
# ---------------------------------------------------------------------------

def vertical_merge(
    input1_images: str,
    input1_json: str,
    input2_images: str,
    input2_json: str,
    output: str,
) -> None:
    """Perform a vertical merge of two COCO datasets.

    "Vertical merge" combines two datasets with **different images** but
    **compatible category schemas** into a single flat output directory.

    Parameters
    ----------
    input1_images:
        Path to the image directory of dataset 1.
    input1_json:
        Path to the COCO JSON file of dataset 1.
    input2_images:
        Path to the image directory of dataset 2.
    input2_json:
        Path to the COCO JSON file of dataset 2.
    output:
        Path to the output directory (must not already exist).
    """
    output_dir = Path(output)

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    if output_dir.exists():
        print(
            f"error: output directory already exists: {output}",
            file=sys.stderr,
        )
        sys.exit(1)

    img_dir1 = validate_image_dir(input1_images)
    img_dir2 = validate_image_dir(input2_images)
    coco1 = load_coco_file(input1_json)
    coco2 = load_coco_file(input2_json)

    # ------------------------------------------------------------------
    # Category validation (before any file copying)
    # ------------------------------------------------------------------
    logger.info("Validating category schemas ...")
    global_categories = validate_and_merge_categories(
        coco1.get("categories", []),
        coco2.get("categories", []),
    )
    logger.info(
        "Categories validated: %d category(ies) in merged schema.",
        len(global_categories),
    )

    # ------------------------------------------------------------------
    # Detect filename conflicts
    # ------------------------------------------------------------------
    images1 = coco1.get("images", [])
    images2 = coco2.get("images", [])
    conflicts = detect_filename_conflicts(images1, images2)
    conflict_count = len(conflicts)
    if conflicts:
        logger.info(
            "%d filename conflict(s) detected — will rename with _1 / _2 suffix.",
            conflict_count,
        )

    # ------------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True)
    logger.info("Created output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # Copy images from dataset 1
    # ------------------------------------------------------------------
    logger.info("Copying images from dataset 1 ...")
    processed_images1, skipped_ids1 = copy_images_and_build_map(
        images1,
        img_dir1,
        output_dir,
        conflicts,
        dataset_suffix="1",
    )

    # ------------------------------------------------------------------
    # Copy images from dataset 2
    # ------------------------------------------------------------------
    logger.info("Copying images from dataset 2 ...")
    processed_images2, skipped_ids2 = copy_images_and_build_map(
        images2,
        img_dir2,
        output_dir,
        conflicts,
        dataset_suffix="2",
    )

    # ------------------------------------------------------------------
    # Assign sequential image IDs
    # ------------------------------------------------------------------
    logger.info("Assigning sequential image IDs ...")
    global_images, id_map1, id_map2 = assign_sequential_image_ids(
        processed_images1,
        processed_images2,
    )

    # ------------------------------------------------------------------
    # Merge annotations
    # ------------------------------------------------------------------
    logger.info("Merging annotations ...")
    merged_annotations = merge_annotations(
        coco1.get("annotations", []),
        coco2.get("annotations", []),
        id_map1,
        id_map2,
        skipped_ids1,
        skipped_ids2,
    )

    # ------------------------------------------------------------------
    # Build output COCO JSON
    # ------------------------------------------------------------------
    now_iso = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    coco_output = {
        "info": {
            "description": "Vertically merged COCO dataset",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "contributor": "vertical_merge_coco.py",
            "date_created": now_iso,
        },
        "licenses": [],
        "images": global_images,
        "annotations": merged_annotations,
        "categories": global_categories,
    }

    annotations_path = output_dir / "_annotations.coco.json"
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=4)
    logger.info("Annotations written to: %s", annotations_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(
        input1_images,
        input2_images,
        coco1,
        coco2,
        conflict_count,
        global_images,
        merged_annotations,
        global_categories,
        output,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Vertically merge two COCO datasets (different images, compatible "
            "category schemas) into a single output directory."
        )
    )
    parser.add_argument(
        "-i1",
        "--input1-images",
        required=True,
        metavar="DIR",
        help="Path to dataset 1 image directory.",
    )
    parser.add_argument(
        "-j1",
        "--input1-json",
        required=True,
        metavar="FILE",
        help="Path to dataset 1 COCO JSON file.",
    )
    parser.add_argument(
        "-i2",
        "--input2-images",
        required=True,
        metavar="DIR",
        help="Path to dataset 2 image directory.",
    )
    parser.add_argument(
        "-j2",
        "--input2-json",
        required=True,
        metavar="FILE",
        help="Path to dataset 2 COCO JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory path (must not already exist).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()

    vertical_merge(
        input1_images=args.input1_images,
        input1_json=args.input1_json,
        input2_images=args.input2_images,
        input2_json=args.input2_json,
        output=args.output,
    )