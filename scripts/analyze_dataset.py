#!/usr/bin/env python3
"""CLI script to run and display data analysis on one or more COCO JSON files.

Single-file mode: prints the existing detailed per-file report.
Multi-file mode: prints a side-by-side comparison table across all files.
"""

from __future__ import annotations

import argparse
import os
import sys

from src.utils.coco_dataset_analyzer import CocoDatasetAnalyzer, compare_datasets


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Analyze and validate one or more COCO JSON files for class "
            "distributions and data quality issues."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        nargs="+",
        metavar="FILE",
        help="Path(s) to COCO format JSON file(s). Pass multiple paths to get a comparison report.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single-file report (unchanged look)
# ---------------------------------------------------------------------------

def _print_single_report(file_path: str) -> None:
    """Print the full single-file COCO dataset report.

    Args:
        file_path: Path to the COCO JSON file to analyze.

    Raises:
        SystemExit: If analysis fails (exits with code 1).
    """
    try:
        print(f"Parsing and running data-integrity checks on: {file_path}...")
        analyzer = CocoDatasetAnalyzer(file_path)
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


# ---------------------------------------------------------------------------
# Multi-file comparison report
# ---------------------------------------------------------------------------

def _truncate(s: str, max_len: int = 20) -> str:
    """Truncate a string to max_len characters, appending '…' if needed.

    Args:
        s: The input string.
        max_len: Maximum allowed length (default 20).

    Returns:
        Truncated string.
    """
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _print_comparison_report(file_paths: list[str]) -> None:
    """Print a multi-file side-by-side comparison report.

    Args:
        file_paths: List of paths to COCO JSON files.

    Raises:
        SystemExit: If any file fails to load (exits with code 1).
    """
    results = compare_datasets(file_paths)

    # Check for errors
    has_error = False
    for r in results:
        if "error" in r:
            print(
                f"Error loading '{r['file']}': {r['error']}",
                file=sys.stderr,
            )
            has_error = True
    if has_error:
        sys.exit(1)

    summaries = [r["summary"] for r in results]

    # Column headers: basename truncated to 20 chars
    headers = [_truncate(os.path.basename(r["file"]), 20) for r in results]

    # Union of all class names across files, sorted alphabetically
    all_classes: list[str] = sorted(
        {cls for s in summaries for cls in s["classes"]}
    )

    # Union of categories_with_ids: build name -> id mapping (first occurrence wins)
    class_id_map: dict[str, int] = {}
    for s in summaries:
        for cat in s["categories_with_ids"]:
            if cat["name"] not in class_id_map:
                class_id_map[cat["name"]] = cat["id"]

    # ----------------------------------------------------------------
    # Table 1: Overview
    # ----------------------------------------------------------------
    col_file_w = 30
    col_img_w = 8
    col_cat_w = 13
    col_ann_w = 12

    separator = "-" * (col_file_w + 2 + col_img_w + 2 + col_cat_w + 2 + col_ann_w + 1)

    print("")
    print("=== COCO DATASET COMPARISON REPORT ===")
    print("")
    print(
        f"{'File':<{col_file_w}} | {'Images':>{col_img_w}} | {'Categories':<{col_cat_w}} | {'Total Ann.':>{col_ann_w}}"
    )
    print(separator)
    for r in results:
        s = r["summary"]
        label = _truncate(os.path.basename(r["file"]), col_file_w)
        print(
            f"{label:<{col_file_w}} | {s['total_images']:>{col_img_w},} | "
            f"{len(s['classes']):<{col_cat_w}} | {s['total_annotations']:>{col_ann_w},}"
        )

    # ----------------------------------------------------------------
    # Table 2: Per-class annotation counts
    # ----------------------------------------------------------------
    n = len(results)
    count_col_w = 20

    # Build column widths for per-file count columns
    file_col_ws = [max(len(h), 10) for h in headers]

    print("")
    print("--- Per-class annotation counts ---")
    print("")

    # Header row
    header_row = f"{'Class':<{col_file_w}} | {'ID':>4}"
    for i, h in enumerate(headers):
        header_row += f" | {h:>{file_col_ws[i]}}"
    print(header_row)

    sep_row = "-" * col_file_w + "-|-" + "-" * 4
    for w in file_col_ws:
        sep_row += "-|-" + "-" * w
    print(sep_row)

    for cls_name in all_classes:
        cls_id = class_id_map.get(cls_name, "?")
        row = f"{cls_name:<{col_file_w}} | {cls_id!s:>4}"
        for i, s in enumerate(summaries):
            if cls_name in s["class_counts"]:
                count_str = f"{s['class_counts'][cls_name]:,}"
            else:
                count_str = "\u2014"  # em-dash
            row += f" | {count_str:>{file_col_ws[i]}}"
        print(row)

    # ----------------------------------------------------------------
    # Table 3: Class balance (% of total annotations)
    # ----------------------------------------------------------------
    print("")
    print("--- Class balance (annotations per class / total annotations) ---")
    print("")

    header_row = f"{'Class':<{col_file_w}}"
    for i, h in enumerate(headers):
        header_row += f" | {h:>{file_col_ws[i]}}"
    print(header_row)

    sep_row = "-" * col_file_w
    for w in file_col_ws:
        sep_row += "-|-" + "-" * w
    print(sep_row)

    for cls_name in all_classes:
        row = f"{cls_name:<{col_file_w}}"
        for i, s in enumerate(summaries):
            total = s["total_annotations"]
            if cls_name in s["class_counts"] and total > 0:
                pct = s["class_counts"][cls_name] / total * 100
                pct_str = f"{pct:.2f} %"
            elif cls_name not in s["class_counts"]:
                pct_str = "\u2014"
            else:
                pct_str = "0.00 %"
            row += f" | {pct_str:>{file_col_ws[i]}}"
        print(row)

    # ----------------------------------------------------------------
    # QA Warnings
    # ----------------------------------------------------------------
    print("")
    print("--- QA Warnings ---")
    max_label_len = max(len(os.path.basename(r["file"])) for r in results)
    for r in results:
        s = r["summary"]
        label = os.path.basename(r["file"])
        unannotated = len(s["unannotated_image_paths"])
        oob = len(s["out_of_bounds_errors"])
        img_word = "image" if unannotated == 1 else "images"
        err_word = "error" if oob == 1 else "errors"
        print(
            f"{label:<{max_label_len}} : {unannotated} unannotated {img_word}, "
            f"{oob} out-of-bounds {err_word}"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main execution block."""
    args = parse_arguments()

    if len(args.input) == 1:
        _print_single_report(args.input[0])
    else:
        _print_comparison_report(args.input)


if __name__ == "__main__":
    main()