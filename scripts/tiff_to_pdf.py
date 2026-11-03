"""Convert a TIFF image to PDF from the command line."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.tiff_to_pdf import convert_tiff_to_pdf

logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a single- or multi-page TIFF image to PDF."
    )
    parser.add_argument("input", type=Path, help="Path to the input TIFF file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PDF path (defaults to the input path with a .pdf extension).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        output_path = convert_tiff_to_pdf(args.input, args.output)
    except (OSError, ValueError) as error:
        logger.error("Conversion failed: %s", error)
        return 1

    logger.info("Conversion completed: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())