# category.py

import time

def compute(model_url: str, code_url: str, dataset_url: str) -> dict:
    """
    Returns the category of model.
    """
    startTime = time.time()
    latency_ms = (time.time() - startTime) * 1000
    return "MODEL", latency_ms
