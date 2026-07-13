import os
import json
import glob
import argparse
from tqdm import tqdm  # NEW: Import the progress bar

# Import our modularized code
from src.config import CATEGORY_MAPPING, COCO_CATEGORIES
from src.weak_labeling.roboflow_client import get_inference_client

def convert_to_coco_bbox(x: float, y: float, width: float, height: float) -> list:
    """Helper function to convert center x/y to COCO top-left x/y."""
    x_min = x - (width / 2)
    y_min = y - (height / 2)
    return [x_min, y_min, width, height]

def generate_weak_labels(input_dir: str, output_json: str, model_id: str) -> None:
    """Scans a directory of images and generates weak COCO annotations via Roboflow."""
    
    client = get_inference_client()
    
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": COCO_CATEGORIES
    }

    image_id_counter = 1
    annotation_id_counter = 1

    print(f"Scanning directory: {input_dir}")
    
    # Create a list to hold all found images (both JPG and PNG)
    image_files = []
    valid_extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    
    for ext in valid_extensions:
        search_pattern = os.path.join(input_dir, ext)
        image_files.extend(glob.glob(search_pattern))
    
    if not image_files:
        print("No JPG or PNG images found in the specified directory.")
        return

    print(f"Found {len(image_files)} images. Starting inference with model '{model_id}'...")

    # NEW: Wrap the image_files list with tqdm for a beautiful progress bar
    for img_path in tqdm(image_files, desc="Labeling Images", unit="img"):
        filename = os.path.basename(img_path)
        
        try:
            result = client.infer(img_path, model_id=model_id)
            
            img_width = result['image']['width'] # type: ignore
            img_height = result['image']['height'] # type: ignore
            
            coco_output["images"].append({
                "id": image_id_counter,
                "file_name": filename,
                "width": img_width,
                "height": img_height
            })
            
            for prediction in result.get('predictions', []): # type: ignore
                label = prediction['class']
                
                if label in CATEGORY_MAPPING:
                    bbox = convert_to_coco_bbox(
                        prediction['x'], prediction['y'], 
                        prediction['width'], prediction['height']
                    )
                    
                    coco_output["annotations"].append({
                        "id": annotation_id_counter,
                        "image_id": image_id_counter,
                        "category_id": CATEGORY_MAPPING[label],
                        "bbox": bbox,
                        "area": prediction['width'] * prediction['height'],
                        "iscrowd": 0,
                        "confidence": prediction['confidence'] 
                    })
                    
                    annotation_id_counter += 1
            
            image_id_counter += 1
            
        except Exception as e:
            # tqdm.write() ensures error messages don't break the progress bar's formatting
            tqdm.write(f"Failed to process {filename}. Error: {e}")

    # Save the final output
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"\nDone! Successfully saved COCO annotations to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weakly label images using a Roboflow model.")
    parser.add_argument("-i", "--input", required=True, help="Path to the directory containing images.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the resulting COCO JSON.")
    parser.add_argument("-m", "--model", default="checkbox-0fyo0/1", help="Roboflow model ID to use for inference.")
    
    args = parser.parse_args()
    
    generate_weak_labels(args.input, args.output, args.model)