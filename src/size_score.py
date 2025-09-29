# metrics/size_score.py
import asyncio
import time
import logging

ERROR_VALUE = {
    "raspberry_pi": 0.0,
    "jetson_nano": 0.0,
    "desktop_pc": 0.0,
    "aws_server": 0.0,
}

# Device memory thresholds in GB
DEVICE_LIMITS = {
    "raspberry_pi": 0.5,
    "jetson_nano": 1.5,
    "desktop_pc": 10.0,
    "aws_server": 100.0,
}

async def compute(model_url: str | None, code_url: str | None, dataset_url: str | None) -> tuple[dict, float]:
    """
    Compute hardware compatibility scores for a model.

    The score for each device is a heuristic:
      - 1.0 if the model size is well below the device limit
      - 0.5 if it's close to the limit
      - 0.0 if it exceeds the limit or size is unknown
    """
    start = time.perf_counter()

    try:
        # Naive heuristic: extract model name and guess size
        model_size_gb = 0.0

        if model_url:
            # Example heuristic: smaller models like "tiny" or "base" are small,
            # "large" or "xl" are big. This avoids needing the API.
            lower = model_url.lower()
            if "tiny" in lower or "small" in lower:
                model_size_gb = 0.2
            elif "base" in lower:
                model_size_gb = 0.8
            elif "medium" in lower:
                model_size_gb = 2.0
            elif "large" in lower:
                model_size_gb = 8.0
            elif "xl" in lower or "xxl" in lower:
                model_size_gb = 20.0
            else:
                model_size_gb = 5.0  # default heuristic

        scores = {}
        for device, limit in DEVICE_LIMITS.items():
            if model_size_gb == 0.0:
                score = 0.0
            elif model_size_gb <= limit * 0.5:
                score = 1.0
            elif model_size_gb <= limit:
                score = 0.5
            else:
                score = 0.0
            scores[device] = score

        latency_ms = (time.perf_counter() - start) * 1000
        return scores, round(latency_ms)

    except Exception as e:
        logging.error(f"[size_score] Error computing size score: {e}")
        latency_ms = (time.perf_counter() - start) * 1000
        return ERROR_VALUE, round(latency_ms)