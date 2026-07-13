import os
import sys

import cv2
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


# image_path = r"C:\Users\dennis.dosso\Projects\OBJECT_DETECTION\datasets\checkbox_datasets\checkbox_ws.coco\test\Screenshot-2024-07-17-113456_png.rf.bbe169609435e428ad3aedb22bdf9028.jpg" # Make sure this matches your image file
image_path = r"C:\Users\dennis.dosso\Projects\OBJECT_DETECTION\datasets\checkbox_datasets\checkpoint_2024\Materiale CASSA di TRENTO lavorato\1_B_ROSA.pdf" # Make sure this matches your image file

# 2. Load the image using OpenCV FIRST
image = cv2.imread(image_path)

# Add a safety check: if the path is wrong, OpenCV returns None
if image is None:
    print(f"Error: OpenCV could not read the image at:\n{image_path}\nPlease check if the file exists and the path is correct.")
    sys.exit(1)

# 3. Run the model (pass the loaded 'image' array instead of the 'image_path' string)
print("Running inference...")
result = CLIENT.infer(image, model_id="checkbox-0fyo0/1")

# 4. Loop through the predictions and draw boxes
for prediction in result['predictions']:  # type: ignore
    # Extract bounding box data
    center_x = prediction['x']
    center_y = prediction['y']
    width = prediction['width']
    height = prediction['height']
    label = prediction['class']
    confidence = prediction['confidence']

    # Calculate top-left and bottom-right coordinates
    x_min = int(center_x - (width / 2))
    y_min = int(center_y - (height / 2))
    x_max = int(center_x + (width / 2))
    y_max = int(center_y + (height / 2))

    # Choose color based on class (OpenCV uses BGR format: Blue, Green, Red)
    if label == "checked":
        color = (0, 255, 0) # Green for checked
    else:
        color = (0, 0, 255) # Red for unchecked

    # Draw the bounding box rectangle
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    # Prepare and draw the text label with confidence score
    text = f"{label} ({confidence:.2f})"
    cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 5. Display the final image
cv2.imshow("Roboflow Detections", image)

# Wait for the user to press any key, then close the window
print("Press any key on the image window to close it.")
cv2.waitKey(0)
cv2.destroyAllWindows()