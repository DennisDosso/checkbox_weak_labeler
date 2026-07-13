import os
import glob
import argparse
import fitz  # PyMuPDF

def convert_pdfs_to_images(pdf_dir: str, output_dir: str, dpi: int = 300) -> None:
    """
    Converts all PDF files in a directory to PNG images.
    """
    os.makedirs(output_dir, exist_ok=True)
    search_pattern = os.path.join(pdf_dir, "*.[pP][dD][fF]")
    pdf_files = glob.glob(search_pattern)
    
    print(f"Found {len(pdf_files)} PDF files to convert.")
    zoom_factor = dpi / 72.0
    matrix = fitz.Matrix(zoom_factor, zoom_factor)
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(filename)[0]
        
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                output_filename = f"{base_name}_page_{page_num + 1}.png"
                output_path = os.path.join(output_dir, output_filename)
                pix.save(output_path)
                
            print(f"Successfully converted: {filename} ({len(doc)} pages)")
            doc.close()
            
        except Exception as e:
            print(f"Failed to convert {filename}. Error: {e}")

if __name__ == "__main__":
    # Set up argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="Convert PDF documents to PNG images for Object Detection.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input directory containing PDF files.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output directory to save PNG images.")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution of the output images (default: 300).")
    
    args = parser.parse_args()
    
    # Run the function using the provided arguments
    convert_pdfs_to_images(args.input, args.output, dpi=args.dpi)