# category.py

import time
from run import UrlCategory

def compute(model_url: str, code_url: str, dataset_url: str) -> str:
    """
    Returns the category of model.
    """
    startTime = time.time()

    if model_url and "huggingface.co" in model_url:
        category = "MODEL"
    elif code_url and "github.com" in code_url:
        category = "CODE"
    elif dataset_url and "huggingface.co/datasets" in dataset_url:
        category = "DATASET"
    else:
        category = "OTHER"

    latency_ms = (time.time() - startTime) * 1000
    return category, latency_ms
