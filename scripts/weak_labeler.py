import os
import json
import glob
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

load_dotenv(".env.local")

API_KEY = os.getenv("ROBOFLOW_API_KEY")
API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com") # Defaults if not found

if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY is missing. Please check your .env.local file.")

# 1. Set up the client
CLIENT = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
)

# 2. Configuration
# Set the path to the folder containing your JPGs
IMAGES_DIR = r"C:\Users\dennis.dosso\Projects\OBJECT_DETECTION\datasets\checkbox_datasets\checkbox.v6i.coco\valid"
# Set where you want the final JSON to be saved
OUTPUT_JSON_PATH = os.path.join(IMAGES_DIR, "roboflow_annotations.json")

# Map the string labels to your specific Category IDs (0 is saved for signatures)
CATEGORY_MAPPING = {
    "checked": 1,
    "unchecked": 2
}

# 3. Initialize the COCO dictionary structure
coco_output = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "checked", "supercategory": "checkbox"},
        {"id": 2, "name": "unchecked", "supercategory": "checkbox"}
    ]
}

# Counters for COCO IDs (must be unique across the whole dataset)
image_id_counter = 1
annotation_id_counter = 1

print(f"Scanning directory: {IMAGES_DIR}")

# 4. Find all JPG files (ignoring JSONs, PNGs, etc.)
# This pattern matches .jpg, .JPG, .jpeg, etc.
search_pattern = os.path.join(IMAGES_DIR, "*.[jJ][pP][gG]")
image_files = glob.glob(search_pattern)

print(f"Found {len(image_files)} JPG images. Starting inference...")

# 5. Process each image
for img_path in image_files:
    filename = os.path.basename(img_path)
    print(f"Processing: {filename}...")
    
    try:
        # Run inference
        result = CLIENT.infer(img_path, model_id="checkbox-0fyo0/1")
        
        # Extract image dimensions directly from the Roboflow response
        img_width = result['image']['width'] # type: ignore 
        img_height = result['image']['height'] # type: ignore 
        
        # Add image data to COCO 'images' list
        coco_output["images"].append({
            "id": image_id_counter,
            "file_name": filename,
            "width": img_width,
            "height": img_height
        })
        
        # Process predictions for this specific image
        for prediction in result.get('predictions', []): #type: ignore
            label = prediction['class']
            
            # Only process if the label is in our mapping (ignores unknown classes)
            if label in CATEGORY_MAPPING:
                category_id = CATEGORY_MAPPING[label]
                
                # COCO bbox format requires [top_left_x, top_left_y, width, height]
                width = prediction['width']
                height = prediction['height']
                x_min = prediction['x'] - (width / 2)
                y_min = prediction['y'] - (height / 2)
                
                # Add annotation data to COCO 'annotations' list
                coco_output["annotations"].append({
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    # Optional: Include confidence score as extra metadata
                    "confidence": prediction['confidence'] 
                })
                
                annotation_id_counter += 1
        
        # Increment image ID for the next file
        image_id_counter += 1
        
    except Exception as e:
        # Catch errors (like a corrupted image) so the whole script doesn't crash
        print(f"Failed to process {filename}. Error: {e}")

# 6. Save the final COCO dictionary to a JSON file
with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(coco_output, f, indent=4)

print(f"\nDone! Successfully saved COCO annotations to: {OUTPUT_JSON_PATH}")