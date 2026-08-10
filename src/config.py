import os
from dotenv import load_dotenv

# Load environment variables once, here.
load_dotenv(".env.local")

# API Configuration
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

# Dataset Category Mappings
CATEGORY_MAPPING = {
    "signature": 0,
    "checked": 1,
    "unchecked": 2
}

# Base COCO Categories
COCO_CATEGORIES = [
    {"id": 0, "name": "signature", "supercategory": "signature"},
    {"id": 1, "name": "checked", "supercategory": "checkbox"},
    {"id": 2, "name": "unchecked", "supercategory": "checkbox"}
]