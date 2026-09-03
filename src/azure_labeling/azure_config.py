"""
azure_labeling.azure_config
----------------------------
Central configuration for the Azure Document Intelligence weak labeler.

Values can be overridden via:
1. Environment variables with prefix ``AZURE_DI_`` (highest priority)
2. ``.env.local`` file in the project root (loaded automatically)
3. Python defaults (lowest priority)

Required environment variables (no defaults):
    AZURE_DI_ENDPOINT    Azure DI resource endpoint URL
    AZURE_DI_KEY         Azure DI API key

Optional environment variables:
    AZURE_DI_MODEL_ID               Azure DI model to use (default: prebuilt-layout)
    AZURE_DI_CONFIDENCE_THRESHOLD   Minimum confidence to include a detection (default: 0.80)
    AZURE_DI_TIMEOUT                HTTP timeout in seconds (default: 600)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Type

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Project root: two levels up from this file (src/azure_labeling/azure_config.py)
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]

# ---------------------------------------------------------------------------
# COCO category definitions — checkboxes only
# These match the global CATEGORY_MAPPING in src/config.py:
#   checked=1, unchecked=2
# ---------------------------------------------------------------------------
AZURE_COCO_CATEGORIES = [
    {"id": 1, "name": "checked",   "supercategory": "checkbox"},
    {"id": 2, "name": "unchecked", "supercategory": "checkbox"},
]

# Maps Azure DI selection mark state strings to COCO category ids
AZURE_STATE_TO_CATEGORY_ID: dict[str, int] = {
    "selected":   1,  # checked
    "unselected": 2,  # unchecked
}

# Maps Azure DI selection mark state strings to human-readable label
AZURE_STATE_TO_LABEL: dict[str, str] = {
    "selected":   "checked",
    "unselected": "unchecked",
}


class AzureSettings(BaseSettings):
    """
    Settings schema for the Azure Document Intelligence weak labeler.

    Attributes
    ----------
    endpoint:
        Azure DI resource endpoint URL (required).
    key:
        Azure DI API key (required).
    model_id:
        Azure DI model to use for analysis. Must support selectionMarks.
        Use ``prebuilt-layout`` (default) — do NOT use ``prebuilt-read``
        as it does not detect checkboxes.
    confidence_threshold:
        Minimum confidence score [0.0, 1.0] for a detection to be included
        in the COCO output (default: 0.80).
    timeout:
        HTTP timeout for Azure DI polling, in seconds (default: 600).
    """

    endpoint: str
    key: str
    model_id: str = "prebuilt-layout"
    confidence_threshold: float = 0.80
    timeout: int = 600

    model_config = SettingsConfigDict(
        env_prefix="AZURE_DI_",
        env_nested_delimiter="__",
        env_file=str(PROJECT_ROOT / ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """
        Priority (highest to lowest):
            1. Environment variables (``AZURE_DI_*``)
            2. ``.env.local`` file
            3. Python defaults
        """
        return (env_settings, dotenv_settings, init_settings)