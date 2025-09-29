# name.py
import time
from urllib.parse import urlparse
from typing import Tuple

# path segments that indicate we've gone past the repo id into files/views
_RESERVED = {"resolve", "blob", "tree", "commit", "commits", "discussions", "revision", "files"}

async def compute(model_url: str, code_url: str, dataset_url: str) -> Tuple[str, int]:
    """
    Returns the model name from a Hugging Face model URL.
    """
    startTime = time.time()

    # Default
    model_name = ""

    if model_url:
        url = model_url.strip()
        if "://" not in url:  # i.e. "huggingface.co/org/model"
            url = "https://" + url

        parsed = urlparse(url)
        path = (parsed.path or "").strip("/")
        if path:
            parts = [p for p in path.split("/") if p]
            if not (len(parts) == 1 and parts[0].lower() in {"models", "datasets", "spaces"}): # listing/landing pages like "/models" but no model name
                cut = next((i for i, p in enumerate(parts) if p.lower() in _RESERVED), len(parts))
                repo_parts = parts[:cut]
                if repo_parts and repo_parts[0].lower() in {"datasets", "spaces"}: # Catch for "datasets" prefix
                    repo_parts = repo_parts[1:]
                if repo_parts:
                    model_name = repo_parts[-1]

    latency_ms = (int)((time.time() - startTime) * 1000)
    return model_name, latency_ms
