from inference_sdk import InferenceHTTPClient
from src.config import ROBOFLOW_API_KEY, ROBOFLOW_API_URL

def get_inference_client() -> InferenceHTTPClient:
    """
    Initializes and returns the Roboflow Inference client using 
    credentials from the environment configuration.
    """
    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_API_KEY
    )
    return client