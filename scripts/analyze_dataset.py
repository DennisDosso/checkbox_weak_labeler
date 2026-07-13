#!/usr/bin/env python3
"""CLI script to run and display data analysis on a COCO JSON file."""

import argparse
import sys
from src.utils.coco_dataset_analyzer import CocoDatasetAnalyzer


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and validate a COCO JSON file for class distributions and data leaks."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to the COCO format JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution block."""
    args = parse_arguments()

    try:
        print(f"Parsing and running data-integrity checks on: {args.input}...")
        analyzer = CocoDatasetAnalyzer(args.input)
        summary = analyzer.get_summary()

        # Print Base Statistics
        print("\n" + "=" * 50)
        print("             COCO DATASET REPORT              ")
        print("=" * 50)
        print(f"Total Images:               {summary['total_images']:,}")
        print(f"Total Annotations:          {summary['total_annotations']:,}")
        print(f"Avg Annotations / Image:    {summary['avg_annotations_per_image']}")
        print(f"Total Unique Classes:       {len(summary['classes'])}")
        print("-" * 50)
        print(f"{'Class Name':<32} | {'Instance Count':<15}")
        print("-" * 50)
        
        for class_name, count in summary["class_counts"].items():
            print(f"{class_name:<32} | {count:<15,}")
        print("=" * 50)

        # Print Unannotated Images Check
        unannotated_count = len(summary["unannotated_image_paths"])
        if unannotated_count > 0:
            print(f"\n⚠️  WARNING: Found {unannotated_count} images completely missing annotations.")
            # Show a small preview snippet if there are many
            preview_limit = 5
            for path in summary["unannotated_image_paths"][:preview_limit]:
                print(f"   - Missing labels: {path}")
            if unannotated_count > preview_limit:
                print(f"   - ... and {unannotated_count - preview_limit} more files.")
        else:
            print("\n✅ QA Check: All images contain at least one annotation.")

        # Print Spatial Boundary Validation Check
        error_count = len(summary["out_of_bounds_errors"])
        if error_count > 0:
            print(f"\n❌ CRITICAL: Found {error_count} bounding boxes extending outside image borders!")
            preview_limit = 5
            for error in summary["out_of_bounds_errors"][:preview_limit]:
                print(f"   - {error}")
            if error_count > preview_limit:
                print(f"   - ... and {error_count - preview_limit} more boundary errors.")
        else:
            print("✅ QA Check: All bounding box coordinates fit safely within image dimensions.")
        print()

    except Exception as e:
        print(f"Error during validation pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()