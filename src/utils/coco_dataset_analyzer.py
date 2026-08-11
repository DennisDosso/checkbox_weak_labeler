"""Module for analyzing and validating datasets formatted in the COCO JSON format."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, TypedDict, Union


class CocoAnalysisSummary(TypedDict):
    """Type definition for the advanced COCO analysis summary dictionary."""

    total_images: int
    total_annotations: int
    avg_annotations_per_image: float
    classes: List[str]
    class_counts: Dict[str, int]
    unannotated_image_paths: List[str]
    out_of_bounds_errors: List[str]
    categories_with_ids: List[Dict]


class CocoDatasetAnalyzer:
    """Parses, computes structural statistics, and validates a COCO format JSON file."""

    def __init__(self, file_path: Union[str, Path]):
        """Initializes the analyzer, loads JSON, and validates root structure.

        Args:
            file_path: Path to the COCO JSON file.
        """
        self.file_path = Path(file_path)
        self._data = self._load_json()
        self._validate_coco_structure()

    def _load_json(self) -> dict:
        """Loads the JSON file from disk."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"COCO file not found at: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _validate_coco_structure(self) -> None:
        """Validates that the fundamental COCO structural keys exist."""
        required_keys = ["images", "annotations", "categories"]
        missing_keys = [key for key in required_keys if key not in self._data]
        if missing_keys:
            raise KeyError(
                f"Invalid COCO format. Missing mandatory root keys: {missing_keys}"
            )

    def get_summary(self) -> CocoAnalysisSummary:
        """Computes advanced metrics and runs spatial validation checks on the dataset.

        Returns:
            A CocoAnalysisSummary dictionary containing advanced metrics and validation reports.
        """
        images = self._data["images"]
        annotations = self._data["annotations"]
        categories = self._data["categories"]

        # 1. Map lookups for performance
        category_map = {cat["id"]: cat["name"] for cat in categories}
        image_map = {img["id"]: img for img in images}
        classes = sorted(list(category_map.values()))

        # Build categories_with_ids: sorted by id
        categories_with_ids = sorted(
            [{"id": cat["id"], "name": cat["name"]} for cat in categories],
            key=lambda c: c["id"],
        )

        # 2. Count instances per class and map annotations to images
        annotation_counts = Counter()
        image_annotation_tracker = {img_id: 0 for img_id in image_map.keys()}
        out_of_bounds_errors = []

        for ann in annotations:
            category_id = ann.get("category_id")
            category_name = category_map.get(category_id, f"Unknown (ID: {category_id})")
            annotation_counts[category_name] += 1

            # Track annotations per image
            img_id = ann.get("image_id")
            if img_id in image_annotation_tracker:
                image_annotation_tracker[img_id] += 1

            # Spatial Out-of-Bounds Validation
            bbox = ann.get("bbox")  # COCO format: [x_min, y_min, width, height]
            if bbox and img_id in image_map:
                img_meta = image_map[img_id]
                img_w, img_h = img_meta.get("width", 0), img_meta.get("height", 0)
                x, y, w, h = bbox

                # Check if coordinates cross image limits
                if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
                    out_of_bounds_errors.append(
                        f"Annotation ID {ann.get('id')} in image '{img_meta.get('file_name')}' "
                        f"is out of bounds. BBox: [{x}, {y}, {w}, {h}] on {img_w}x{img_h} image."
                    )

        # 3. Identify images with completely missing annotations
        unannotated_image_paths = [
            image_map[img_id].get("file_name", f"Unknown_ID_{img_id}")
            for img_id, count in image_annotation_tracker.items()
            if count == 0
        ]

        # 4. Enforce explicit inclusion of zero-count classes
        class_counts = {cls_name: annotation_counts[cls_name] for cls_name in classes}

        # 5. Compute averages safely
        total_images = len(images)
        total_anns = len(annotations)
        avg_annotations = total_anns / total_images if total_images > 0 else 0.0

        return {
            "total_images": total_images,
            "total_annotations": total_anns,
            "avg_annotations_per_image": round(avg_annotations, 2),
            "classes": classes,
            "class_counts": class_counts,
            "unannotated_image_paths": unannotated_image_paths,
            "out_of_bounds_errors": out_of_bounds_errors,
            "categories_with_ids": categories_with_ids,
        }


def compare_datasets(paths: list[str | Path]) -> list[dict]:
    """Analyze and compare multiple COCO JSON datasets.

    Accepts a list of N file paths (N >= 1). For each path, instantiates a
    ``CocoDatasetAnalyzer`` and calls ``get_summary()``. If a file raises an
    exception (not found, bad JSON, missing keys, etc.), the result entry
    contains an ``"error"`` key instead of ``"summary"``.

    Args:
        paths: List of paths to COCO JSON files.

    Returns:
        A list of result dicts, one per file, in the same order as ``paths``.
        Each dict has:

        - ``"file"`` (str): the file path as supplied.
        - ``"summary"`` (CocoAnalysisSummary): the full summary, **or**
        - ``"error"`` (str): error message if loading/analysis failed.

    Example::

        results = compare_datasets(["train.json", "val.json"])
        for r in results:
            if "error" in r:
                print(f"{r['file']}: ERROR — {r['error']}")
            else:
                print(f"{r['file']}: {r['summary']['total_images']} images")
    """
    results: list[dict] = []
    for path in paths:
        file_str = str(path)
        try:
            analyzer = CocoDatasetAnalyzer(path)
            summary = analyzer.get_summary()
            results.append({"file": file_str, "summary": summary})
        except Exception as exc:  # noqa: BLE001
            results.append({"file": file_str, "error": str(exc)})
    return results