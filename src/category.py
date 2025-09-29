# category.py

import time
from typing import Tuple

async def compute(model_url: str, code_url: str, dataset_url: str) -> Tuple[str, int]:
    """
    Returns the category of model.
    """
    startTime = time.time()
    latency_ms = (int)((time.time() - startTime) * 1000)
    return "MODEL", latency_ms
