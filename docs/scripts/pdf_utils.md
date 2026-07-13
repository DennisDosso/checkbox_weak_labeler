# PDF to Image Utility (`pdf_utils.py`)

## Why We Need It
Document Object Detection models (like YOLO or those hosted on Roboflow) process images, not raw PDF documents. To train a model to detect specific elements on a page—such as checkboxes, signatures, or printed text—we must first convert our training data from multi-page PDFs into high-quality, individual image files. 

This utility standardizes that pipeline by ensuring:
* **Resolution is preserved:** It defaults to 300 DPI to maintain the crisp edges required to detect small objects like checkboxes accurately.
* **Quality is lossless:** It exports strictly to PNG format, avoiding the compression artifacts (fuzziness) commonly introduced by JPGs.
* **Multi-page documents are tracked:** It automatically iterates through multi-page PDFs, converting each page into a standalone image and appending the page number to the filename for easy traceability back to the source document.

## How to Use It

You can use this utility either as a standalone command-line script or by importing its core function into another Python module.

### 1. As a Command-Line Tool
Run the script directly from your terminal, passing your input and output directories as arguments:

```bash
python src/pdf_utils.py --input "data/raw" --output "data/processed" --dpi 300

```

**Arguments:**

* `-i`, `--input` *(Required)*: Path to the folder containing your raw PDF files.
* `-o`, `--output` *(Required)*: Path where the converted PNGs will be saved.
* `--dpi` *(Optional)*: The resolution of the output images. Defaults to 300.

### 2. As an Imported Module

If you are writing a larger pipeline script, you can import the function directly:

```python
from src.pdf_utils import convert_pdfs_to_images

convert_pdfs_to_images(
    pdf_dir="path/to/pdfs", 
    output_dir="path/to/save/images", 
    dpi=300
)