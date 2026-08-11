"""
src.utils.filter_coco
----------------------
Utility to filter and remap a COCO JSON annotation file according to a
user-supplied category mapping.

Categories not present in the mapping are dropped along with all their
annotations. Categories present in the mapping are kept and their IDs are
remapped to the values specified in the mapping.

Usage
-----
From the project root::

    python -m src.utils.filter_coco \\
        -i input/_annotations.merged.coco.json \\
        -m category_mapping.json \\
        -o output/_annotations.filtered.coco.json \\
        [--drop-unannotated]

Mapping file format
-------------------
A simple JSON dict mapping category name to desired output id::

    {
        "signature": 0,
        "checked": 1,
        "unchecked": 2
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json_file(path: str, label: str = "file") -> dict | list:
    """Load and parse a JSON file.

    Parameters
    ----------
    path:
        Filesystem path to the JSON file.
    label:
        Human-readable label used in error messages (e.g. ``"input COCO"``).

    Returns
    -------
    dict or list
        Parsed JSON content.

    Raises
    ------
    SystemExit
        If the file does not exist or is not valid JSON (exits with code 1).
    """
    p = Path(path)
    if not p.exists():
        print(f"error: {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {label} file '{path}': {exc}", file=sys.stderr)
        sys.exit(1)

    return data


def load_mapping(path: str) -> dict[str, int]:
    """Load and validate a category mapping JSON file.

    The mapping must be a JSON object mapping category names (str) to new
    category IDs (int). Duplicate output IDs are not permitted.

    Parameters
    ----------
    path:
        Filesystem path to the mapping JSON file.

    Returns
    -------
    dict mapping category name -> new category id.

    Raises
    ------
    SystemExit
        If the file is missing, invalid JSON, not a dict, contains non-integer
        values, or contains duplicate output IDs (exits with code 1).
    """
    raw = load_json_file(path, label="mapping")

    if not isinstance(raw, dict):
        print(
            f"error: mapping file must be a JSON object (dict), got {type(raw).__name__}.",
            file=sys.stderr,
        )
        sys.exit(1)

    validated: dict[str, int] = {}
    for name, new_id in raw.items():
        if not isinstance(new_id, int):
            print(
                f"error: mapping value for '{name}' must be an integer, "
                f"got {type(new_id).__name__} ({new_id!r}).",
                file=sys.stderr,
            )
            sys.exit(1)
        validated[name] = new_id

    # Check for duplicate output IDs
    seen_ids: dict[int, str] = {}
    for name, new_id in validated.items():
        if new_id in seen_ids:
            print(
                f"error: duplicate new id {new_id} in mapping — "
                f"assigned to both '{seen_ids[new_id]}' and '{name}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        seen_ids[new_id] = name

    return validated


# ---------------------------------------------------------------------------
# Category remapping
# ---------------------------------------------------------------------------

def build_category_remap(
    input_categories: list[dict],
    mapping: dict[str, int],
) -> tuple[list[dict], dict[int, int], list[dict]]:
    """Build the output category list and an old_id -> new_id remapping.

    Parameters
    ----------
    input_categories:
        Category list from the input COCO JSON.
    mapping:
        Dict mapping category name -> desired output id.

    Returns
    -------
    output_categories:
        List of category dicts for the output COCO JSON, sorted by new id.
    old_id_to_new_id:
        Dict mapping old category id -> new category id (only for kept cats).
    dropped_categories:
        List of input category dicts that were dropped (not in mapping).
    """
    # Build a lookup: name -> input category dict
    input_by_name: dict[str, dict] = {
        cat["name"]: cat for cat in input_categories
    }

    output_categories: list[dict] = []
    old_id_to_new_id: dict[int, int] = {}

    for name, new_id in mapping.items():
        if name in input_by_name:
            src_cat = input_by_name[name]
            supercategory = src_cat.get("supercategory", "none") or "none"
            output_categories.append(
                {
                    "id": new_id,
                    "name": name,
                    "supercategory": supercategory,
                }
            )
            old_id_to_new_id[src_cat["id"]] = new_id
            logger.debug(
                "Category '%s': old id %d -> new id %d",
                name,
                src_cat["id"],
                new_id,
            )
        else:
            # Category in mapping but not in input — still include in output
            # (it just won't match any annotation)
            output_categories.append(
                {
                    "id": new_id,
                    "name": name,
                    "supercategory": "none",
                }
            )
            logger.debug(
                "Category '%s' (new id %d) is in mapping but not in input COCO.",
                name,
                new_id,
            )

    output_categories.sort(key=lambda c: c["id"])

    dropped_categories = [
        cat for cat in input_categories if cat["name"] not in mapping
    ]

    return output_categories, old_id_to_new_id, dropped_categories


# ---------------------------------------------------------------------------
# Annotation filtering
# ---------------------------------------------------------------------------

def filter_annotations(
    input_annotations: list[dict],
    old_id_to_new_id: dict[int, int],
    image_id_map: dict[int, int],
) -> list[dict]:
    """Filter and remap annotations.

    Keeps only annotations whose ``category_id`` has a mapping to a new id.
    Reassigns ``id`` sequentially from 1 and rewrites ``image_id`` and
    ``category_id`` using the provided maps.

    Parameters
    ----------
    input_annotations:
        Annotation list from the input COCO JSON.
    old_id_to_new_id:
        Old category id -> new category id (only kept categories).
    image_id_map:
        Old image id -> new sequential image id.

    Returns
    -------
    list of filtered annotation dicts with new sequential IDs.
    """
    output: list[dict] = []
    ann_id = 1

    for ann in input_annotations:
        old_cat_id = ann.get("category_id")
        if old_cat_id not in old_id_to_new_id:
            logger.debug(
                "Dropping annotation id=%d (category_id=%d not in mapping).",
                ann.get("id"),
                old_cat_id,
            )
            continue

        new_ann = dict(ann)
        new_ann["id"] = ann_id
        new_ann["category_id"] = old_id_to_new_id[old_cat_id]

        old_img_id = ann.get("image_id")
        new_ann["image_id"] = image_id_map.get(old_img_id, old_img_id)

        output.append(new_ann)
        ann_id += 1

    return output


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

def build_image_id_map(
    input_images: list[dict],
    drop_unannotated: bool,
    kept_image_ids: set[int],
) -> tuple[list[dict], dict[int, int]]:
    """Build the output image list and old_id -> new_id map.

    Parameters
    ----------
    input_images:
        Image list from the input COCO JSON.
    drop_unannotated:
        If True, exclude images whose id is not in ``kept_image_ids``.
    kept_image_ids:
        Set of original image IDs that have at least one kept annotation.

    Returns
    -------
    output_images:
        Filtered and sequentially re-IDed image list.
    image_id_map:
        Old image id -> new sequential image id.
    """
    output_images: list[dict] = []
    image_id_map: dict[int, int] = {}
    new_id = 1

    for img in input_images:
        old_id = img["id"]
        if drop_unannotated and old_id not in kept_image_ids:
            logger.debug(
                "Dropping unannotated image id=%d file_name='%s'.",
                old_id,
                img.get("file_name", ""),
            )
            continue
        new_img = dict(img)
        new_img["id"] = new_id
        image_id_map[old_id] = new_id
        output_images.append(new_img)
        new_id += 1

    return output_images, image_id_map


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(
    input_path: str,
    mapping_path: str,
    input_categories: list[dict],
    output_categories: list[dict],
    dropped_categories: list[dict],
    input_annotations: list[dict],
    output_annotations: list[dict],
    input_images: list[dict],
    output_images: list[dict],
    drop_unannotated: bool,
    old_id_to_new_id: dict[int, int],
    mapping: dict[str, int],
    output_path: str,
) -> None:
    """Print the filter summary report to stdout.

    Parameters
    ----------
    input_path:
        Path to the input COCO JSON file.
    mapping_path:
        Path to the category mapping JSON file.
    input_categories:
        Category list from the input COCO JSON.
    output_categories:
        Final output category list.
    dropped_categories:
        Categories that were dropped.
    input_annotations:
        Original annotation list.
    output_annotations:
        Filtered output annotation list.
    input_images:
        Original image list.
    output_images:
        Output image list (after optional unannotated drop).
    drop_unannotated:
        Whether unannotated images were dropped.
    old_id_to_new_id:
        Old category id -> new category id map.
    mapping:
        The category mapping dict (name -> new id).
    output_path:
        Path where output was written.
    """
    input_cat_names = ", ".join(cat["name"] for cat in input_categories)
    kept_cat_names = ", ".join(cat["name"] for cat in output_categories)
    dropped_cat_names = ", ".join(cat["name"] for cat in dropped_categories)

    dropped_images_count = len(input_images) - len(output_images)

    print("")
    print("=== COCO Filter Summary ===")
    print(f"Input file      : {input_path}")
    print(f"Mapping file    : {mapping_path}")
    print(f"Input categories: {len(input_categories)} ({input_cat_names})")
    print(f"Kept categories : {len(output_categories)} ({kept_cat_names})")
    print(f"Dropped categories: {len(dropped_categories)} ({dropped_cat_names})")
    print(f"Input annotations: {len(input_annotations)}")
    print(f"Kept annotations : {len(output_annotations)}")
    print(f"Dropped annotations: {len(input_annotations) - len(output_annotations)}")
    print(f"Input images    : {len(input_images)}")

    if drop_unannotated and dropped_images_count > 0:
        print(
            f"Output images   : {len(output_images)}"
            f"  ({dropped_images_count} dropped as unannotated)"
        )
    else:
        print(f"Output images   : {len(output_images)}")

    print(f"Output written  : {output_path}")

    # Category remapping table
    # Build old_id lookup from input
    input_by_name: dict[str, dict] = {cat["name"]: cat for cat in input_categories}

    print("")
    print("Category remapping:")
    # Show kept categories
    for name, new_id in sorted(mapping.items(), key=lambda x: x[1]):
        if name in input_by_name:
            old_id = input_by_name[name]["id"]
            tag = "[no change]" if old_id == new_id else "[REMAPPED]"
            print(f"  '{name}' : old id {old_id} -> new id {new_id}  {tag}")
    # Show dropped categories (in input but not in mapping)
    for cat in input_categories:
        if cat["name"] not in mapping:
            print(f"  '{cat['name']}' : old id {cat['id']} -> DROPPED")


# ---------------------------------------------------------------------------
# Core filter function
# ---------------------------------------------------------------------------

def filter_coco(
    input_path: str,
    mapping_path: str,
    output_path: str,
    drop_unannotated: bool = False,
) -> None:
    """Filter and remap a COCO JSON file according to a category mapping.

    Parameters
    ----------
    input_path:
        Path to the input COCO JSON file.
    mapping_path:
        Path to the category mapping JSON file.
    output_path:
        Path for the output COCO JSON file (silently overwritten if exists).
    drop_unannotated:
        If True, images with zero kept annotations are excluded from output.
    """
    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    logger.info("Loading input COCO: %s", input_path)
    coco_data = load_json_file(input_path, label="input COCO")
    if not isinstance(coco_data, dict):
        print(
            f"error: input COCO file must be a JSON object, "
            f"got {type(coco_data).__name__}.",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("Loading mapping: %s", mapping_path)
    mapping = load_mapping(mapping_path)

    input_categories: list[dict] = coco_data.get("categories", [])
    input_annotations: list[dict] = coco_data.get("annotations", [])
    input_images: list[dict] = coco_data.get("images", [])

    # ------------------------------------------------------------------
    # Build category remap
    # ------------------------------------------------------------------
    logger.info("Building category remapping ...")
    output_categories, old_id_to_new_id, dropped_categories = build_category_remap(
        input_categories, mapping
    )

    # ------------------------------------------------------------------
    # Determine which image IDs have at least one kept annotation
    # (needed for --drop-unannotated; compute before image_id_map)
    # ------------------------------------------------------------------
    kept_image_ids: set[int] = {
        ann["image_id"]
        for ann in input_annotations
        if ann.get("category_id") in old_id_to_new_id
    }

    # ------------------------------------------------------------------
    # Build image list and ID map
    # ------------------------------------------------------------------
    logger.info("Building image list ...")
    output_images, image_id_map = build_image_id_map(
        input_images, drop_unannotated, kept_image_ids
    )

    # ------------------------------------------------------------------
    # Filter annotations (using the image_id_map built above)
    # ------------------------------------------------------------------
    logger.info("Filtering annotations ...")
    output_annotations = filter_annotations(
        input_annotations, old_id_to_new_id, image_id_map
    )

    # ------------------------------------------------------------------
    # Build output COCO JSON
    # ------------------------------------------------------------------
    now_iso = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    coco_output = {
        "info": {
            "description": "Filtered COCO dataset",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "contributor": "filter_coco.py",
            "date_created": now_iso,
        },
        "licenses": [],
        "images": output_images,
        "annotations": output_annotations,
        "categories": output_categories,
    }

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=4)
    logger.info("Filtered output written to: %s", output_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(
        input_path,
        mapping_path,
        input_categories,
        output_categories,
        dropped_categories,
        input_annotations,
        output_annotations,
        input_images,
        output_images,
        drop_unannotated,
        old_id_to_new_id,
        mapping,
        output_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter and remap a COCO JSON annotation file according to a "
            "user-supplied category mapping."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        metavar="FILE",
        help="Path to input COCO JSON file.",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        required=True,
        metavar="FILE",
        help="Path to category mapping JSON file (name -> new_id).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=None,
        metavar="FILE",
        help=(
            "Path to output COCO JSON file (silently overwritten if exists). "
            "Defaults to the input file path (in-place overwrite) if not specified."
        ),
    )
    parser.add_argument(
        "--drop-unannotated",
        action="store_true",
        default=False,
        help="Drop images with zero annotations after filtering.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    output_path = args.output if args.output is not None else args.input
    filter_coco(
        input_path=args.input,
        mapping_path=args.mapping,
        output_path=output_path,
        drop_unannotated=args.drop_unannotated,
    )