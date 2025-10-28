# metrics/size_score.py
import time
import logging
from typing import Optional, Tuple, Dict

# Default score if metric fails
ERROR_VALUE: Dict[str, float] = {
    "raspberry_pi": 0.0,
    "jetson_nano": 0.0,
    "desktop_pc": 0.0,
    "aws_server": 0.0,
}

# Device capacity thresholds in GB
DEVICE_LIMITS: Dict[str, float] = {
    "raspberry_pi": 0.5,
    "jetson_nano": 1.5,
    "desktop_pc": 10.0,
    "aws_server": 100.0,
}


def _estimate_model_size_gb(model_url: str) -> float:
    """
    Estimate the size of a Hugging Face model in GB.

    Strategy:
    1. Try Hugging Face API (`siblings` list sizes).
    2. Fallback: heuristic guesses based on model name keywords.
    3. Default to ~5 GB if nothing found.
    """
    
    import requests

    # --- API attempt ---
    try:
        model_id = model_url.rstrip("/").split("huggingface.co/")[-1]
        resp = requests.get(f"https://huggingface.co/api/models/{model_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        total_size_bytes = sum(f.get("size", 0) for f in data.get("siblings", []))
        if total_size_bytes > 0:
            logging.debug(f"[size_score] HF API size for {model_url}: {total_size_bytes} bytes")
            return total_size_bytes / (1024**3)  # bytes → GB
    except Exception as e:
        logging.debug(f"[size_score] HF API failed for {model_url}: {e}")

    # --- Heuristic fallback ---
    lower = model_url.lower()
    if "tiny" in lower or "small" in lower:
        return 0.2
    if "base" in lower:
        return 0.8
    if "medium" in lower:
        return 2.0
    if "large" in lower:
        return 8.0
    if "xl" in lower or "xxl" in lower:
        return 20.0

    # --- Default ---
    return 5.0


async def compute(model_url: str, code_url: Optional[str], dataset_url: Optional[str]) -> Tuple[float, int]:
    """
    Compute hardware compatibility (size_score) for a model.

    Args:
        model_url: Hugging Face model URL.
        code_url: GitHub repo URL (unused in this metric, but required for consistency).
        dataset_url: Dataset URL (unused in this metric, but required for consistency).

    Returns:
        (scores: dict[str, float], latency_ms: float)
    """
    start = time.perf_counter()

    if not model_url or "huggingface.co" not in model_url:
        logging.warning("[size_score] No valid Hugging Face model URL.")
        return ERROR_VALUE, (time.perf_counter() - start) * 1000

    try:
        # Step 1. Estimate model size
        model_size_gb = _estimate_model_size_gb(model_url)

        # Step 2. Apply thresholds to produce device-specific scores
        scores: Dict[str, float] = {}
        for device, limit in DEVICE_LIMITS.items():
            if model_size_gb <= limit * 0.5:
                score = 1.0
            elif model_size_gb <= limit:
                score = 0.5
            else:
                score = 0.0
            scores[device] = score

        latency_ms = (time.perf_counter() - start) * 1000
        return scores, round(latency_ms)

    except Exception as e:
        logging.error(f"[size_score] Error computing for {model_url}: {e}")
        latency_ms = (time.perf_counter() - start) * 1000
        return ERROR_VALUE, round(latency_ms)