"""
local_labeling.local_config
----------------------------
Central configuration for the local RF-DETR weak labeler.

Values can be overridden via:
1. Environment variables with prefix ``LOCAL_LABELER_`` (highest priority)
2. YAML config file (default: ``config.local.yaml`` in project root,
   overridable via ``LOCAL_LABELER_CONFIG_PATH`` env var)
3. Python defaults (lowest priority)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Type

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Project root: two levels up from this file (src/local_labeling/local_config.py)
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]

# ---------------------------------------------------------------------------
# COCO category definitions for this labeler (signature only)
# ---------------------------------------------------------------------------
LOCAL_COCO_CATEGORIES = [
    {"id": 0, "name": "signature", "supercategory": "signature"}
]


class LocalSettings(BaseSettings):
    """
    Settings schema for the local RF-DETR weak labeler.

    Attributes
    ----------
    model_path:
        Path to the ``.pth`` weights file (required, no default).
    confidence_threshold:
        Minimum confidence score for a detection to be included (default: 0.35).
    device:
        Torch device to run inference on (default: ``"cpu"``).
        At runtime, ``"cuda"`` is used automatically if a GPU is available
        and this field is not overridden.
    """

    model_path: str
    confidence_threshold: float = 0.35
    device: str = "cpu"

    model_config = SettingsConfigDict(
        env_prefix="LOCAL_LABELER_",
        env_nested_delimiter="__",
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
        Customize setting resolution order.

        Priority (highest to lowest):
            1. Environment variables (``LOCAL_LABELER_*``)
            2. YAML config file
            3. Python defaults
        """
        # Determine YAML config file path
        config_path = os.getenv("LOCAL_LABELER_CONFIG_PATH", "config.local.yaml")
        resolved_config = Path(config_path)

        # Resolve relative paths against the project root
        if not resolved_config.is_absolute():
            resolved_config = (PROJECT_ROOT / resolved_config).resolve()

        if resolved_config.exists():
            yaml_source = YamlConfigSettingsSource(
                settings_cls, yaml_file=resolved_config
            )
            return (
                env_settings,
                yaml_source,
                init_settings,
            )

        # Fallback: no YAML file found
        return (env_settings, init_settings)