"""TIFF-to-PDF conversion utilities."""

from pathlib import Path

from PIL import Image


def convert_tiff_to_pdf(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a TIFF image, including all its pages, to a PDF file.

    When ``output_path`` is omitted, the PDF is written beside the input with
    the same name and a ``.pdf`` extension.
    """
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")

    destination = (
        Path(output_path) if output_path is not None else source.with_suffix(".pdf")
    )
    if destination.suffix.lower() != ".pdf":
        raise ValueError(f"Output file must have a .pdf extension: {destination}")
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output paths must be different.")

    with Image.open(source) as image:
        if image.format != "TIFF":
            raise ValueError(f"Input file is not a TIFF image: {source}")

        pages: list[Image.Image] = []
        try:
            for page_number in range(getattr(image, "n_frames", 1)):
                image.seek(page_number)
                pages.append(image.convert("RGB"))

            destination.parent.mkdir(parents=True, exist_ok=True)
            pages[0].save(
                destination,
                format="PDF",
                save_all=len(pages) > 1,
                append_images=pages[1:],
            )
        finally:
            for page in pages:
                page.close()

    return destination