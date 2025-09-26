# metrics/dataset_quality_metric.py

import asyncio
from huggingface_hub import HfApi

async def compute(payload: dict) -> float:
    """
    Compute dataset quality using Hugging Face Hub metadata.
    Metric score is based on presence of description, license, and size info.
    """
    url = payload.get("url", "")

    if "huggingface.co/datasets" not in url:
        return 0.0
    owner = payload.get("owner")
    name = payload.get("name")
    repo_id = f"{owner}/{name}" if owner and name else name

    api = HfApi()
    try:
        info = api.dataset_info(repo_id)
    except Exception:
        return 0.0  # if fetch fails, consider quality unknown

    # metadata checks
    has_description = bool(info.card_data.get("description"))
    has_license = bool(info.card_data.get("license"))
    has_size = bool(info.siblings)  # if there are files listed, then dataset content exists

    checks = [has_description, has_license, has_size]
    score = sum(checks) / len(checks)

    return score
