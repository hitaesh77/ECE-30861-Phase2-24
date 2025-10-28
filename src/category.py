# category.py

import time
from typing import Optional, Tuple

async def compute(model_url: str, code_url: Optional[str], dataset_url: Optional[str]) -> Tuple[float, int]:
    """
    Returns the category of model.
    """
    startTime = time.time()
    latency_ms = (int)((time.time() - startTime) * 1000)
    return "MODEL", latency_ms
