import requests
import time  
from typing import Tuple

ERROR_VALUE = -1.0  # global fallback value


async def compute(model_url: str, code_url: str | None, dataset_url: str | None) -> Tuple[float, float]:
    """
    Compute a license score for a Hugging Face model.
    
    Args:
        hf_link (str): Full Hugging Face model link (e.g., "https://huggingface.co/gpt2")
        
    Returns:
        float: License score between 0.0 (very restrictive/unknown) and 1.0 (very permissive).
    """
    start_time = time.time()

    try:
        # Extract model_id from link
        model_id = model_url.replace("https://huggingface.co/", "").strip("/")
        url = f"https://huggingface.co/api/models/{model_id}"
        
        # Fetch model metadata
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        
        # Get license field (default to "unknown")
        license_name = data.get("cardData", {}).get("license", "").lower()
        if not license_name:
            return 0.0, (time.time() - start_time) * 1000
        
        # Define categories
        permissive = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "mpl-2.0", "cc-by-4.0"}
        restrictive = {"gpl-2.0", "gpl-3.0", "lgpl-3.0", "cc-by-sa-4.0"}
        non_commercial = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0", "research-only"}
        custom = {"openrail-m", "bigscience-openrail-m", "bigscience-openrail", "custom"}
        unknown = {"unknown", "other"}
        
        # Score mapping
        
        if license_name in permissive:
            score = 1.0, 
        elif license_name in restrictive:
            score = 0.7
        elif license_name in non_commercial:
            score = 0.4
        elif license_name in custom:
            score = 0.6
        elif license_name in unknown:
            score = 0.0
        else:
            # fallback: partial credit for unrecognized licenses
            score = 0.5
        return score, (time.time() - start_time) * 1000
    except Exception as e:
        print(f"Error computing license score: {e}")
        return ERROR_VALUE, (time.time() - start_time) * 1000
    