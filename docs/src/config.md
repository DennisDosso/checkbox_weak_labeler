# Configuration Guide (`config.py`)

## Overview
The `src/config.py` file acts as the single source of truth for all project-wide settings, environment variables, and dataset mappings. By centralizing these variables, we ensure that sensitive information (like API keys) is kept out of the executable scripts and that adding new classes or changing endpoints only requires an update in one place.

## Environment Variables (`.env.local`)
To run this project, you must have a `.env.local` file at the root of the repository. `config.py` uses the `python-dotenv` library to securely load these values into the application.

**Required Variables:**
* `ROBOFLOW_API_KEY`: Your private Roboflow API key. **(Never commit this to version control)**.
* `ROBOFLOW_API_URL`: The endpoint for inference (defaults to `https://serverless.roboflow.com`).

If the API key is missing, `config.py` will immediately raise a `ValueError` to prevent downstream errors.

## Category Mappings
When the Roboflow model returns predictions as string labels (e.g., "checked"), we need to convert them into integer IDs for the COCO JSON format. 

`config.py` contains two crucial dictionaries for this:
1. **`CATEGORY_MAPPING`**: Maps the string label to its corresponding integer ID.
2. **`COCO_CATEGORIES`**: The exact dictionary structure required by the COCO format to define the classes.

### How to Add a New Class
If you train a new model to also detect "signatures", you only need to update `config.py`:

```python
CATEGORY_MAPPING = {
    "checked": 1,
    "unchecked": 2,
    "signature": 3  # <-- Added new mapping
}

COCO_CATEGORIES = [
    {"id": 1, "name": "checked", "supercategory": "checkbox"},
    {"id": 2, "name": "unchecked", "supercategory": "checkbox"},
    {"id": 3, "name": "signature", "supercategory": "document"} # <-- Added new COCO definition
]
```

No changes are required in the executable scripts when adding new classes.