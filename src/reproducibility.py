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
        # Fetch model metadata
        api_url = f"https://huggingface.co/api/models/{model_id}"
        resp = requests.get(api_url, timeout=10)
        if resp.status_code != 200:
            return 0.0, time.time() - start_time

        data = resp.json()
        has_demo = False

        # 1️⃣ Check cardData (README) for usage/demo code
        card_texts = []
        if "cardData" in data and data["cardData"]:
            card_texts.append(str(data["cardData"]).lower())

        # 2️⃣ Check README if available
        if "readme" in data and data["readme"]:
            card_texts.append(data["readme"].lower())

        # 3️⃣ Check files for .py examples
        file_names = [f["rfilename"] for f in data.get("siblings", []) if "rfilename" in f]
        py_files = [f for f in file_names if f.endswith(".py")]

        # 4️⃣ Look for demo code keywords
        demo_keywords = ["from_pretrained", "pipeline(", "example", "demo"]
        for text in card_texts:
            if any(k in text for k in demo_keywords):
                has_demo = True
                break

        # If no demo in text, but example .py files exist, treat as demo
        if not has_demo and py_files:
            has_demo = True

        # Determine score
        if has_demo:
            score = 1.0
        elif file_names:
            score = 0.5
        else:
            score = 0.0

    except Exception as e:
        print(f"Error: {e}")
        score = 0.0

    latency = time.time() - start_time
    return score, latency

# ------------------ Test run ------------------
if __name__ == "__main__":
    print("TEST 1")
    code_url = "https://github.com/google-research/bert"
    dataset_url = "https://huggingface.co/datasets/bookcorpus/bookcorpus"
    model_url = "https://huggingface.co/google-bert/bert-base-uncased"

    score, latency = compute_reproducibility(model_url)
    print(f"Reproducibility score: {score}")
    print(f"Computation time: {latency:.2f} seconds")

    print("\nTEST 2")
    code_url    = "https://huggingface.co/chiedo/hello-world"  
    dataset_url = "https://huggingface.co/datasets/chiedo/hello-world"  
    model_url   = "https://huggingface.co/chiedo/hello-world"

    score, latency = compute_reproducibility(model_url)
    print(f"Reproducibility score: {score}")
    print(f"Computation time: {latency:.2f} seconds")

    print("\nTEST 3")
    code_url = "https://github.com/huggingface/transformers"  
    dataset_url = "https://huggingface.co/datasets/none"  
    model_url = "https://huggingface.co/roberta-base"

    score, latency = compute_reproducibility(model_url)
    print(f"Reproducibility score: {score}")
    print(f"Computation time: {latency:.2f} seconds")
