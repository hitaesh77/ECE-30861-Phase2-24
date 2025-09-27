# name.py
import time
from urllib.parse import urlparse

def compute(model_url: str, code_url: str, dataset_url: str) -> dict:
    """
    Returns the name of the model.
    """
    startTime = time.time()

    if not model_url:
        return ""

    path = urlparse(model_url).path.strip("/")
    parts = path.split("/")

    if len(parts) >= 2:
        latency_ms = (time.time() - startTime) * 1000
        return parts[-1], latency_ms   # model name
    elif parts:
        latency_ms = (time.time() - startTime) * 1000
        return parts[0], latency_ms    # fallback if only one component
    
    latency_ms = (time.time() - startTime) * 1000
    return "", latency_ms
