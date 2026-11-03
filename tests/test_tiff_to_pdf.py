"""Tests for TIFF-to-PDF conversion and its command-line entry point."""

import fitz
from PIL import Image

from scripts.tiff_to_pdf import main
from src.utils.tiff_to_pdf import convert_tiff_to_pdf


def test_cli_defaults_output_to_input_directory(tmp_path):
    input_path = tmp_path / "document.tiff"
    Image.new("RGB", (16, 16), "white").save(input_path)

    assert main([str(input_path)]) == 0

    output_path = tmp_path / "document.pdf"
    assert output_path.is_file()
    with fitz.open(output_path) as document:
        assert document.page_count == 1


def test_convert_preserves_all_tiff_pages_and_creates_output_directory(tmp_path):
    input_path = tmp_path / "multipage.tif"
    first_page = Image.new("RGB", (16, 16), "white")
    second_page = Image.new("RGB", (16, 16), "black")
    first_page.save(input_path, save_all=True, append_images=[second_page])
    output_path = tmp_path / "nested" / "result.pdf"

    result = convert_tiff_to_pdf(input_path, output_path)

    assert result == output_path
    with fitz.open(output_path) as document:
        assert document.page_count == 2


def test_convert_rejects_non_tiff_input(tmp_path):
    input_path = tmp_path / "not-a-tiff.png"
    Image.new("RGB", (16, 16), "white").save(input_path)

    try:
        convert_tiff_to_pdf(input_path)
    except ValueError as error:
        assert "not a TIFF" in str(error)
    else:
        raise AssertionError("Expected a non-TIFF input to be rejected")