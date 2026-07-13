"""Module for analyzing and validating datasets formatted in the COCO JSON format."""

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
        }