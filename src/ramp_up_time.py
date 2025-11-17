import math
import time
import requests
from urllib.parse import urlparse
from typing import Optional

ERROR_VALUE = -1.0

def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """Keep a value within [min_value, max_value]."""
    return max(min_value, min(value, max_value))

import requests

def get_downloads(model_url: str) -> int:

    # Extract model ID from the URL
    if not model_url.startswith("https://huggingface.co/"):
        raise ValueError("Invalid Hugging Face model URL")
    
    model_id = model_url.replace("https://huggingface.co/", "").strip("/")
    
    # Hugging Face API endpoint
    api_url = f"https://huggingface.co/api/models/{model_id}"
    
    # Send request
    response = requests.get(api_url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch model info: {response.status_code} - {response.text}")
    
    data = response.json()
    
    # Some models may not have 'downloads' key
    downloads = data.get("downloads")
    if downloads is None:
        raise KeyError(f"No 'downloads' field found for model: {model_id}")
    
    return downloads


async def compute(model_url: str, code_url: str, dataset_url: str) -> float:
    """
    Calculates a ramp-up subscore (0–1) for the given model, code, and dataset URLs.
    Higher means faster ramp-up (popular and responsive).
    """
    subscores = []

    start_time = time.perf_counter()

    downloads = get_downloads(model_url)
    if downloads is None or downloads <= 0:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return 0, latency_ms

    ramp_score = clamp(math.log10(downloads) / 15)
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    # Average across all three URLs
    return ramp_score, latency_ms


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    print("TEST 1")
    code_url = "https://github.com/google-research/bert"
    dataset_url = "https://huggingface.co/datasets/bookcorpus/bookcorpus"
    model_url = "https://huggingface.co/google-bert/bert-base-uncased"
    score, latency = compute(model_url, code_url, dataset_url)
    print(f"Ramp-up score: {score}")
    print(f"Computation time: {latency:.2f} ms")

    print("\nTEST 2")
    code_url = "https://github.com/huggingface/transformers"  
    dataset_url = "https://huggingface.co/datasets/none"  
    model_url = "https://huggingface.co/roberta-base"
    score, latency = compute(model_url, code_url, dataset_url)
    print(f"Ramp-up score: {score}")
    print(f"Computation time: {latency:.2f} ms")


    print("\nTEST 3")
    code_url    = "https://huggingface.co/chiedo/hello-world"  
    dataset_url = "https://huggingface.co/datasets/chiedo/hello-world"  
    model_url   = "https://huggingface.co/chiedo/hello-world"
    score, latency = compute(model_url, code_url, dataset_url)
    print(f"Ramp-up score: {score}")
    print(f"Computation time: {latency:.2f} ms")