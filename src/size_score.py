import requests
import time  

ERROR_VALUE = {"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.0, "aws_server": 0.0}

def compute(payload: dict) -> tuple:
    """
    Compute hardware compatibility scores for a Hugging Face model link.
    
    Args:
        hf_link (str): Hugging Face model link, e.g. "https://huggingface.co/gpt2"
        
    Returns:
        dict: Mapping {device: score (0.0–1.0)}
    """
    start_time = time.time()

    try:
        # Extract model id from link
        model_id = (payload.get("url")).strip("/").split("huggingface.co/")[-1]

        # Call Hugging Face API
        url = f"https://huggingface.co/api/models/{model_id}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        
        # Try to infer size from file list
        total_size_bytes = 0
        for f in data.get("siblings", []):
            if "size" in f:
                total_size_bytes += f["size"]
        
        # Convert to GB
        model_size_gb = total_size_bytes / (1024**3) if total_size_bytes > 0 else 0.0

        # Try to estimate parameter count if not given
        param_count = data.get("modelParameters", {}).get("parameterCount", None)
        if param_count is None and model_size_gb > 0:
            # Approximate: 1 param ≈ 4 bytes (FP32)
            param_count = int((model_size_gb * (1024**3)) / 4)
        
        # Device-specific thresholds
        device_limits = {
            "raspberry_pi": 0.5,   # GB
            "jetson_nano": 1.5,
            "desktop_pc": 10.0,
            "aws_server": 100.0
        }
        
        scores = {}
        for device, limit in device_limits.items():
            if model_size_gb == 0:  # If no size info, assume unusable
                score = 0.0
            else:
                score = max(0.0, 1.0 - (model_size_gb / limit))
            scores[device] = round(score, 2)
        
        return scores, (time.time() - start_time) * 1000

    except Exception as e:
        print(f"Error: {e}")
        return ERROR_VALUE, (time.time() - start_time) * 1000