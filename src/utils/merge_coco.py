"""
src.utils.merge_coco
---------------------
Utility to merge two or more COCO JSON files into a single output file.

Features
--------
- Category conflict resolution by name (first occurrence wins canonical ID).
- Image deduplication by file_name (prefer larger resolution if duplicated).
- Sequential ID reassignment for images, annotations, and categories.
- Optional verbose report of category remappings.

Usage
-----
From the project root::

    python -m src.utils.merge_coco \
        -i file1.json file2.json [file3.json ...] \
        -o merged_output.json \
        [--verbose]
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
        print(f"error: input file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in '{path}': {exc}", file=sys.stderr)
        sys.exit(1)

    return data


# ---------------------------------------------------------------------------
# Category conflict resolution
# ---------------------------------------------------------------------------

def build_global_categories(
    all_files: list[tuple[str, dict]],
) -> tuple[list[dict], dict[str, dict[int, int]]]:
    """Build a global conflict-resolved category list across all input files.

    Category identity is determined by **name**. The first occurrence of a name
    wins the canonical global ID. If two files share a name with different IDs,
    both are remapped to the same canonical global ID. If two files have
    different names sharing the same numeric ID, one keeps the ID and the other
    gets the next available unused ID.

    Parameters
    ----------
    all_files:
        List of (file_path, coco_data) pairs.

    Returns
    -------
    global_categories:
        Deduplicated, conflict-resolved list of category dicts sorted by id.
    per_file_category_maps:
        Dict keyed by file_path, value is a dict mapping old category id ->
        new global category id for that file.
    """
    # Global registry: name -> canonical global category dict
    name_to_global: dict[str, dict] = {}
    # Track which global IDs are in use (to detect numeric conflicts)
    used_ids: set[int] = set()
    # Result: per-file old_id -> new_global_id
    per_file_maps: dict[str, dict[int, int]] = {}

    def next_free_id() -> int:
        candidate = 0
        while candidate in used_ids:
            candidate += 1
        return candidate

    for file_path, coco_data in all_files:
        per_file_maps[file_path] = {}
        categories = coco_data.get("categories", [])

        for cat in categories:
            old_id: int = cat["id"]
            name: str = cat["name"]

            if name in name_to_global:
                # Name already registered — remap to existing canonical ID
                canonical_id = name_to_global[name]["id"]
                per_file_maps[file_path][old_id] = canonical_id
                logger.debug(
                    "File '%s': category '%s' old_id=%d -> canonical_id=%d (name match)",
                    file_path, name, old_id, canonical_id,
                )
            else:
                # New name — determine the global ID to assign
                if old_id not in used_ids:
                    # No numeric conflict: keep the original ID
                    new_id = old_id
                else:
                    # Numeric conflict: assign the next free ID
                    new_id = next_free_id()
                    logger.debug(
                        "File '%s': category '%s' old_id=%d conflicts -> new_id=%d",
                        file_path, name, old_id, new_id,
                    )

                global_cat = {
                    "id": new_id,
                    "name": name,
                    "supercategory": cat.get("supercategory", name),
                }
                name_to_global[name] = global_cat
                used_ids.add(new_id)
                per_file_maps[file_path][old_id] = new_id

    global_categories = sorted(name_to_global.values(), key=lambda c: c["id"])
    return global_categories, per_file_maps


# ---------------------------------------------------------------------------
# Image deduplication
# ---------------------------------------------------------------------------

def build_global_images(
    all_files: list[tuple[str, dict]],
) -> tuple[list[dict], dict[str, dict[int, int]]]:
    """Deduplicate images across all files and assign new sequential IDs.

    Images are identified by ``file_name``. If the same ``file_name`` appears
    in multiple files, prefer the entry with larger ``width * height``; on a
    tie, keep the first seen.

    Parameters
    ----------
    all_files:
        List of (file_path, coco_data) pairs.

    Returns
    -------
    global_images:
        Deduplicated image list with new sequential IDs starting from 1.
    per_file_image_maps:
        Dict keyed by file_path, value maps old image id -> new global image id.
    """
    # file_name -> best image dict seen so far (still uses original IDs internally)
    best_by_filename: dict[str, dict] = {}
    # file_name -> list of (file_path, old_id) that map to this filename
    filename_sources: dict[str, list[tuple[str, int]]] = {}

    for file_path, coco_data in all_files:
        for img in coco_data.get("images", []):
            fname = img["file_name"]
            area = img.get("width", 0) * img.get("height", 0)

            if fname not in best_by_filename:
                best_by_filename[fname] = dict(img)
                filename_sources[fname] = [(file_path, img["id"])]
            else:
                existing_area = (
                    best_by_filename[fname].get("width", 0)
                    * best_by_filename[fname].get("height", 0)
                )
                if area > existing_area:
                    best_by_filename[fname] = dict(img)
                filename_sources[fname].append((file_path, img["id"]))

    # Assign new sequential IDs
    global_images: list[dict] = []
    # file_path -> (old_id -> new_global_id)
    per_file_maps: dict[str, dict[int, int]] = {
        fp: {} for fp, _ in all_files
    }

    new_id = 1
    for fname, img_data in best_by_filename.items():
        new_img = dict(img_data)
        new_img["id"] = new_id

        global_images.append(new_img)

        # Map every (file_path, old_id) that pointed to this filename to new_id
        for file_path, old_id in filename_sources[fname]:
            per_file_maps[file_path][old_id] = new_id

        new_id += 1

    return global_images, per_file_maps


# ---------------------------------------------------------------------------
# Annotation merging
# ---------------------------------------------------------------------------

def merge_annotations(
    all_files: list[tuple[str, dict]],
    image_maps: dict[str, dict[int, int]],
    category_maps: dict[str, dict[int, int]],
) -> list[dict]:
    """Merge all annotations, remapping image_id and category_id.

    Parameters
    ----------
    all_files:
        List of (file_path, coco_data) pairs.
    image_maps:
        Per-file old image id -> new global image id.
    category_maps:
        Per-file old category id -> new global category id.

    Returns
    -------
    list of annotation dicts with new sequential IDs starting from 1.
    """
    merged: list[dict] = []
    ann_id = 1

    for file_path, coco_data in all_files:
        img_map = image_maps.get(file_path, {})
        cat_map = category_maps.get(file_path, {})

        for ann in coco_data.get("annotations", []):
            new_ann = dict(ann)
            new_ann["id"] = ann_id

            old_img_id = ann.get("image_id")
            new_ann["image_id"] = img_map.get(old_img_id, old_img_id)

            old_cat_id = ann.get("category_id")
            new_ann["category_id"] = cat_map.get(old_cat_id, old_cat_id)

            merged.append(new_ann)
            ann_id += 1

    return merged


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _count_duplicates(all_files: list[tuple[str, dict]]) -> int:
    """Count how many image file_names appear in more than one input file."""
    seen: dict[str, int] = {}
    for _, coco_data in all_files:
        for img in coco_data.get("images", []):
            fname = img["file_name"]
            seen[fname] = seen.get(fname, 0) + 1
    return sum(1 for count in seen.values() if count > 1)


def _count_conflicts(
    all_files: list[tuple[str, dict]],
    per_file_category_maps: dict[str, dict[int, int]],
) -> int:
    """Count how many category IDs were actually remapped (i.e. old != new)."""
    conflicts = 0
    for file_path, coco_data in all_files:
        cat_map = per_file_category_maps.get(file_path, {})
        for cat in coco_data.get("categories", []):
            old_id = cat["id"]
            new_id = cat_map.get(old_id, old_id)
            if old_id != new_id:
                conflicts += 1
    return conflicts


def print_summary(
    all_files: list[tuple[str, dict]],
    global_categories: list[dict],
    global_images: list[dict],
    merged_annotations: list[dict],
    per_file_category_maps: dict[str, dict[int, int]],
    output_path: str,
    verbose: bool,
) -> None:
    """Print the merge summary report to stdout.

    Parameters
    ----------
    all_files:
        List of (file_path, coco_data) pairs.
    global_categories:
        The final resolved global category list.
    global_images:
        The final deduplicated global image list.
    merged_annotations:
        The final merged annotation list.
    per_file_category_maps:
        Per-file old_id -> new_global_id maps.
    output_path:
        Path where the output was written.
    verbose:
        If True, also print per-file category remapping tables.
    """
    total_raw_images = sum(
        len(coco_data.get("images", [])) for _, coco_data in all_files
    )
    duplicated_count = _count_duplicates(all_files)
    conflicts = _count_conflicts(all_files, per_file_category_maps)

    # Build per-category source file list
    cat_sources: dict[int, list[str]] = {cat["id"]: [] for cat in global_categories}
    for file_path, coco_data in all_files:
        cat_map = per_file_category_maps.get(file_path, {})
        fname = Path(file_path).name
        for cat in coco_data.get("categories", []):
            new_id = cat_map.get(cat["id"], cat["id"])
            if new_id in cat_sources and fname not in cat_sources[new_id]:
                cat_sources[new_id].append(fname)

    print("")
    print("=== COCO Merge Summary ===")
    print(f"Input files     : {len(all_files)}")
    print(
        f"Total images    : {len(global_images)}"
        f" (deduplicated: {duplicated_count})"
    )
    print(f"Total annotations: {len(merged_annotations)}")
    print("Categories in output:")
    for cat in global_categories:
        sources_str = ", ".join(cat_sources.get(cat["id"], []))
        print(f"  [{cat['id']}] {cat['name']:<16} (from: {sources_str})")
    print(f"Category conflicts resolved: {conflicts}")
    print(f"Output written  : {output_path}")

    if verbose:
        print("")
        for file_path, coco_data in all_files:
            fname = Path(file_path).name
            cat_map = per_file_category_maps.get(file_path, {})
            print(f"Category remapping for {fname}:")
            for cat in coco_data.get("categories", []):
                old_id = cat["id"]
                new_id = cat_map.get(old_id, old_id)
                name = cat["name"]
                tag = "[no change]" if old_id == new_id else "[REMAPPED]"
                print(
                    f"  old id {old_id} ({name}) -> new id {new_id} ({name})  {tag}"
                )


# ---------------------------------------------------------------------------
# Core merge function
# ---------------------------------------------------------------------------

def merge_coco_files(
    input_paths: list[str],
    output_path: str,
    verbose: bool = False,
) -> None:
    """Merge multiple COCO JSON files into a single output file.

    Parameters
    ----------
    input_paths:
        List of paths to input COCO JSON files (at least 2 required).
    output_path:
        Destination path for the merged output JSON file.
    verbose:
        If True, print a detailed remapping report after merging.
    """
    if len(input_paths) < 2:
        print(
            "error: at least 2 input files are required for merging.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load all files
    logger.info("Loading %d input file(s) ...", len(input_paths))
    all_files: list[tuple[str, dict]] = []
    for path in input_paths:
        logger.info("  Loading: %s", path)
        data = load_coco_file(path)
        all_files.append((path, data))

    # Resolve categories
    logger.info("Resolving category conflicts ...")
    global_categories, per_file_category_maps = build_global_categories(all_files)

    # Deduplicate images
    logger.info("Deduplicating images ...")
    global_images, per_file_image_maps = build_global_images(all_files)

    # Merge annotations
    logger.info("Merging annotations ...")
    merged_annotations = merge_annotations(
        all_files, per_file_image_maps, per_file_category_maps
    )

    # Build output structure
    now_iso = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    coco_output = {
        "info": {
            "description": "Merged COCO dataset",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "contributor": "merge_coco.py",
            "date_created": now_iso,
        },
        "licenses": [],
        "images": global_images,
        "annotations": merged_annotations,
        "categories": global_categories,
    }

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=4)
    logger.info("Merged output written to: %s", output_path)

    # Print summary
    print_summary(
        all_files,
        global_categories,
        global_images,
        merged_annotations,
        per_file_category_maps,
        output_path,
        verbose,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two or more COCO JSON files into a single output file.",
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Two or more input COCO JSON file paths.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="FILE",
        help="Path for the merged output JSON file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print a detailed conflict/merge report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()

    if len(args.inputs) < 2:
        print(
            "error: at least 2 input files are required. Got: "
            + str(len(args.inputs)),
            file=sys.stderr,
        )
        sys.exit(1)

    merge_coco_files(args.inputs, args.output, verbose=args.verbose)