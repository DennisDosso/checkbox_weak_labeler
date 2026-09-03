"""
scripts/azure_di_weak_labeler.py
---------------------------------
CLI script that generates COCO-format weak labels for a directory of images
by calling Azure Document Intelligence with the ``prebuilt-layout`` model
to detect checkboxes (selectionMarks).

Usage
-----
From the project root::

    python scripts/azure_di_weak_labeler.py \\
        -i /path/to/images \\
        -o /path/to/output/annotations.json \\
        [--confidence 0.80] \\
        [--model prebuilt-layout] \\
        [--recursive] \\
        [--resume] \\
        [--overwrite] \\
        [--dry-run]

Environment / .env.local variables:
    AZURE_DI_ENDPOINT               Azure DI resource endpoint URL (required)
    AZURE_DI_KEY                    Azure DI API key (required)
    AZURE_DI_MODEL_ID               Model to use (default: prebuilt-layout)
    AZURE_DI_CONFIDENCE_THRESHOLD   Minimum confidence threshold (default: 0.80)
    AZURE_DI_TIMEOUT                HTTP timeout in seconds (default: 600)

IMPORTANT: Do NOT change AZURE_DI_MODEL_ID to ``prebuilt-read`` — that model
does not detect checkboxes.  Only ``prebuilt-layout`` supports selectionMarks.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``src.*`` imports resolve whether
# the script is run as ``python scripts/azure_di_weak_labeler.py`` or
# ``python -m scripts.azure_di_weak_labeler``.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.azure_labeling.azure_config import AzureSettings   # noqa: E402
from src.azure_labeling.azure_di_labeler import (            # noqa: E402
    build_client,
    generate_azure_weak_labels,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate COCO-format weak labels for a directory of images "
            "using Azure Document Intelligence (prebuilt-layout) to detect checkboxes."
        )
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the directory containing the source images.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to save the resulting COCO JSON annotations file.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help=(
            "Override the confidence threshold from .env.local "
            "(float between 0.0 and 1.0; default: 0.80)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override the Azure DI model ID (default: prebuilt-layout). "
            "WARNING: do not use prebuilt-read — it does not detect checkboxes."
        ),
    )
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
            "without calling Azure DI or writing any files."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    # Validate mutually exclusive flags
    if args.resume and args.overwrite:
        print(
            "error: --resume and --overwrite are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Allow CLI flags to override env/settings
    if args.model:
        os.environ["AZURE_DI_MODEL_ID"] = args.model
    if args.confidence is not None:
        os.environ["AZURE_DI_CONFIDENCE_THRESHOLD"] = str(args.confidence)

    # Load settings (reads .env.local automatically via AzureSettings)
    try:
        settings = AzureSettings()  # type: ignore[call-arg]
    except Exception as exc:
        print(
            f"error: Could not load Azure DI settings: {exc}\n"
            "Make sure AZURE_DI_ENDPOINT and AZURE_DI_KEY are set in .env.local "
            "or as environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("Azure DI endpoint  : %s", settings.endpoint)
    logger.info("Azure DI model     : %s", settings.model_id)
    logger.info("Confidence threshold: %.2f", settings.confidence_threshold)

    # In dry-run mode, skip building the client
    if args.dry_run:
        client = None
    else:
        client = build_client(settings)

    generate_azure_weak_labels(
        input_dir=args.input,
        output_json=args.output,
        client=client,
        settings=settings,
        recursive=args.recursive,
        resume=args.resume,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )