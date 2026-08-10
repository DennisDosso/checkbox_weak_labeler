"""
tests/test_roboflow_client.py
------------------------------
Unit tests for get_inference_client() in src/weak_labeling/roboflow_client.py.

After Task 2, the API-key check lives inside get_inference_client(), not in
src/config.py, so these tests patch the env var and verify behaviour accordingly.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestGetInferenceClient:
    """Tests for src.weak_labeling.roboflow_client.get_inference_client."""

    def test_client_created_when_api_key_present(self):
        """When ROBOFLOW_API_KEY is set, the client must be created without error."""
        with patch.dict(
            "os.environ",
            {
                "ROBOFLOW_API_KEY": "test-api-key-123",
                "ROBOFLOW_API_URL": "https://serverless.roboflow.com",
            },
        ):
            # Re-import after patching env so config picks up the new values
            import importlib
            import src.config as cfg_module
            importlib.reload(cfg_module)

            import src.weak_labeling.roboflow_client as client_module
            importlib.reload(client_module)

            with patch(
                "src.weak_labeling.roboflow_client.InferenceHTTPClient",
                autospec=True,
            ) as mock_client_cls:
                mock_instance = MagicMock()
                mock_client_cls.return_value = mock_instance

                from src.weak_labeling.roboflow_client import get_inference_client
                client = get_inference_client()

                mock_client_cls.assert_called_once()
                assert client is mock_instance

    def test_raises_value_error_when_api_key_missing(self):
        """When ROBOFLOW_API_KEY is absent, get_inference_client must raise ValueError."""
        # Remove the key from the environment entirely
        with patch.dict("os.environ", {}, clear=True):
            import importlib
            import src.config as cfg_module
            importlib.reload(cfg_module)

            import src.weak_labeling.roboflow_client as client_module
            importlib.reload(client_module)

            from src.weak_labeling.roboflow_client import get_inference_client
            with pytest.raises(ValueError, match="ROBOFLOW_API_KEY"):
                get_inference_client()

    def test_raises_value_error_when_api_key_empty_string(self):
        """An empty string for ROBOFLOW_API_KEY must also trigger ValueError."""
        with patch.dict("os.environ", {"ROBOFLOW_API_KEY": ""}, clear=False):
            import importlib
            import src.config as cfg_module
            importlib.reload(cfg_module)

            import src.weak_labeling.roboflow_client as client_module
            importlib.reload(client_module)

            from src.weak_labeling.roboflow_client import get_inference_client
            with pytest.raises(ValueError, match="ROBOFLOW_API_KEY"):
                get_inference_client()