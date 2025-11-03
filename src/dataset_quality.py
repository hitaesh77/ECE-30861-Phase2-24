# dataset_quality.py
import time
from typing import Optional, Tuple

async def compute(model_url: str, code_url: Optional[str], dataset_url: Optional[str]) -> Tuple[float, int]:
    """
    Lightweight dataset quality score using Hugging Face Hub metadata.
    """

    from huggingface_hub import HfApi
    
    startTime = time.time()

    if not dataset_url or "huggingface.co/datasets" not in dataset_url:
        latency_ms = int((time.time() - startTime) * 1000)
        return 0.0, latency_ms

    parts = dataset_url.strip("/").split("/")
    if len(parts) < 5:
        latency_ms = int((time.time() - startTime) * 1000)
        return 0.0, latency_ms

    owner, name = parts[-2], parts[-1]
    repo_id = f"{owner}/{name}"

    api = HfApi()
    try:
        info = api.dataset_info(repo_id)
    except Exception:
        latency_ms = int((time.time() - startTime) * 1000)
        return 0.0, latency_ms

    has_description = bool(info.card_data.get("description"))
    has_license = bool(info.card_data.get("license"))
    has_files = bool(info.siblings)
    
    score = round((sum([has_description, has_license, has_files]) / 3), 2)
    latency_ms = (int)((time.time() - startTime) * 1000)
    
    return score, latency_ms
