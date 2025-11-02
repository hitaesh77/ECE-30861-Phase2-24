import requests
import time
import re

def compute_reproducibility(model_url: str):
    """
    Compute the Reproducibility score for a Hugging Face model.
    1.0 = runs as-is using demo code
    0.5 = runs with debugging/adjustments
    0.0 = no demo code or cannot run
    """
    start_time = time.time()
    score = 0.0

    # Check if it's a Hugging Face model URL
    match = re.match(r"https?://huggingface\.co/([^/]+/[^/]+)", model_url)
    if not match:
        return 0.0, time.time() - start_time

    model_id = match.group(1)

    try:
        # Try to fetch the model card metadata
        api_url = f"https://huggingface.co/api/models/{model_id}"
        resp = requests.get(api_url, timeout=10)
        if resp.status_code != 200:
            return 0.0, time.time() - start_time

        data = resp.json()

        # Check if there is demo or usage code
        has_demo = False
        if "cardData" in data and data["cardData"]:
            card = data["cardData"]
            text = str(card).lower()
            if "pipeline(" in text or "from_pretrained" in text:
                has_demo = True

        # Determine reproducibility score
        if has_demo:
            score = 1.0
        else:
            # Might still have a repo or model file but no demo
            score = 0.5 if "files" in data and data["files"] else 0.0

    except Exception as e:
        print(f"Error: {e}")
        score = 0.0

    latency = time.time() - start_time
    return score, latency


# ------------------ Test run ------------------
if __name__ == "__main__":
    code_url = "https://github.com/google-research/bert"
    dataset_url = "https://huggingface.co/datasets/bookcorpus/bookcorpus"
    model_url = "https://huggingface.co/google-bert/bert-base-uncased"

    score, latency = compute_reproducibility(model_url)
    print(f"Reproducibility score: {score}")
    print(f"Computation time: {latency:.2f} seconds")
