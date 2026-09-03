"""
azure_labeling
--------------
Weak labeling module that uses Azure Document Intelligence (prebuilt-layout)
to detect and annotate checkboxes in images, producing COCO-format output.
"""

from src.azure_labeling.azure_di_labeler import generate_azure_weak_labels
from src.azure_labeling.azure_config import AzureSettings, AZURE_COCO_CATEGORIES

__all__ = [
    "generate_azure_weak_labels",
    "AzureSettings",
    "AZURE_COCO_CATEGORIES",
]