import os
import requests
import json
import time

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from dotenv import load_dotenv
load_dotenv()


GENAI_BASE_URL  = "https://genai.rcac.purdue.edu/api/chat/completions"
GENAI_MODEL     = "llama3.1:latest"
API_KEY_ENV     = "GEN_AI_STUDIO_API_KEY"
TIMEOUT_SEC     = int(os.getenv("GENAI_TIMEOUT_SEC", "90"))
README_MAX_CHARS= int(os.getenv("README_MAX_CHARS", "30000"))

def get_model_readme(model_url: str) -> str:
    """
    Fetches the README (model card) content for a Hugging Face model.
    """
    if not model_url.startswith("https://huggingface.co/"):
        raise ValueError("Invalid Hugging Face model URL")
    
    model_id = model_url.replace("https://huggingface.co/", "").strip("/")
    api_url = f"https://huggingface.co/api/models/{model_id}"
    response = requests.get(api_url, timeout=30)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch model info: {response.status_code}")
    
    data = response.json()
    card_data = data.get("cardData", {})
    readme = card_data.get("content", "")
    
    if not readme:
        # fallback: fetch README.md from the repo directly
        alt_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        r2 = requests.get(alt_url, timeout=30)
        if r2.status_code == 200:
            readme = r2.text
    
    return readme[:README_MAX_CHARS]


def evaluate_performance_claims(readme_text: str) -> dict:
    """
    Uses Purdue GenAI to evaluate the README text for performance scoring.
    """
    if not readme_text.strip():
        return {"presence": 0, "detail": 0, "evidence": 0, "confirmation": 0, "final_score": 0}

    payload = {
        "model": GENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert model card auditor. "
                    "Given the README text of a Hugging Face model, evaluate its benchmark claims "
                    "using the rubric below and return a JSON object with subscores and a final score."
                    "\n\nRubric:\n"
                    "- presence (45%): 1 if any numeric benchmark claims (README or model-index); else 0.\n"
                    "- detail (15%): scale by clarity/coverage of dataset/task/split/metric/value.\n"
                    "- evidence (10%): strength of supporting material.\n"
                    "- confirmation (30%): authoritative links or model-index corroboration.\n\n"
                    "Respond ONLY with JSON in the format:\n"
                    "{'presence': float, 'detail': float, 'evidence': float, 'confirmation': float, 'final_score': float}"
                )
            },
            {"role": "user", "content": readme_text}
        ],
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {os.getenv(API_KEY_ENV)}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        GENAI_BASE_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=TIMEOUT_SEC,
        verify=False
    )

    if response.status_code != 200:
        raise ValueError(f"GenAI request failed: {response.status_code} - {response.text}")

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Strip Markdown code fences if present
        if content.startswith("```"):
            content = content.strip("`")           # remove backticks
            content = content.replace("json", "", 1).strip()  # remove optional 'json' tag

        # Now parse the JSON inside
        scores = json.loads(content)
        for k in ["presence", "detail", "evidence", "confirmation"]:
            scores[k] = max(0.0, min(1.0, scores[k] / 10.0)) if scores[k] > 1 else scores[k]

        scores["final_score"] = round(
            0.45 * scores.get("presence", 0) +
            0.15 * scores.get("detail", 0) +
            0.10 * scores.get("evidence", 0) +
            0.30 * scores.get("confirmation", 0)
        , 2) # rounds final score to 2 decimals

    except Exception as e:
        raise ValueError(f"Could not parse GenAI output: {e}\nRaw content:\n{content}")


    return scores


async def compute(model_url: str, code_url: str, dataset_url: str) -> dict:
    # start = time.time()
    # readme = get_model_readme(model_url)
    # result = evaluate_performance_claims(readme)
    # latency_ms = (time.time() - start) * 1000
    # return result["final_score"], latency_ms
    return 0.0, 0  # Disabled for testing without API access

if __name__ == "__main__":
    print("TEST 1")
    code_url = "https://github.com/google-research/bert"
    dataset_url = "https://huggingface.co/datasets/bookcorpus/bookcorpus"
    model_url = "https://huggingface.co/google-bert/bert-base-uncased"
    score, latency = compute(model_url, code_url, dataset_url)
    print(f"Score: {score}")
    print(f"Computation time: {latency:.2f} ms")

    print("\nTEST 2")
    code_url = "https://github.com/huggingface/transformers"  
    dataset_url = "https://huggingface.co/datasets/none"  
    model_url = "https://huggingface.co/roberta-base"
    score, latency = compute(model_url, code_url, dataset_url)
    print(f"Score: {score}")
    print(f"Computation time: {latency:.2f} ms")


    print("\nTEST 3")
    code_url    = "https://huggingface.co/chiedo/hello-world"  
    dataset_url = "https://huggingface.co/datasets/chiedo/hello-world"  
    model_url   = "https://huggingface.co/chiedo/hello-world"
    score, latency = compute(model_url, code_url, dataset_url)
    print(f"Score: {score}")
    print(f"Computation time: {latency:.2f} ms")
