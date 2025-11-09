import os
import re
import json
import requests
import warnings
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load environment variables (like API keys)
load_dotenv()

# LLM Configuration
GENAI_BASE_URL = "https://genai.rcac.purdue.edu/api/chat/completions"
GENAI_MODEL = "llama3.1:latest"
API_KEY_ENV = "GEN_AI_STUDIO_API_KEY"
TIMEOUT_SEC = 90
SYSTEM_PROMPT = "You are a strict evaluator. Output valid minified JSON only. No commentary."

# LLM Prompt for extracting lineage
USER_PROMPT = """
You are given unstructured README text and selected Hub metadata.
Extract the model lineage (parent models), and return ONLY valid minified JSON:

{
  "lineage": ["parent_model1", "parent_model2", "base_model"]
}

README_AND_METADATA:
<<<
{TEXT}
>>>
""".strip()

# ================================
# Helper Functions

def get_config(model_id: str) -> dict:
    """Fetch config.json from Hugging Face if it exists."""
    try:
        path = hf_hub_download(model_id, "config.json")
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
            return config
    except Exception as e:
        return {}

def get_model_card(model_id: str) -> str:
    """Fetch README/model card text from Hugging Face."""
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.text
        else:
            return ""
    except Exception as e:
        return ""

def _post_chat(base_url: str, api_key: str, model: str, system: str, user: str, timeout: int = TIMEOUT_SEC) -> str:
    """Send the user prompt to the LLM and return the response."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0, "stream": False, "max_tokens": 800
    }
    try:
        r = requests.post(base_url, headers=headers, json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return ""

def extract_lineage_with_llm(model_id: str) -> list:
    """Use LLM to extract the lineage of a model from README and metadata."""
    readme = get_model_card(model_id)
    config = get_config(model_id)
    
    # Include metadata if available
    metadata = json.dumps(config) if config else ""
    
    # Combine README and metadata
    text = readme + "\n\n### MODEL_METADATA ###\n" + metadata
    
    user_prompt = USER_PROMPT.replace("{TEXT}", text)
    
    # Make LLM request
    llm_response = _post_chat(GENAI_BASE_URL, os.getenv(API_KEY_ENV), GENAI_MODEL, SYSTEM_PROMPT, user_prompt)
    
    try:
        # Parse response from LLM (expected in JSON format)
        parsed = json.loads(llm_response)
        return parsed.get("lineage", [])
    except Exception as e:
        return []

def lineage_graph_with_llm(model_url_or_id: str) -> list:
    """Return the model lineage from root → current model using LLM, as URLs."""
    # Extract model ID
    if model_url_or_id.startswith("https://huggingface.co/"):
        model_id = model_url_or_id.split("https://huggingface.co/")[1].strip("/")
    else:
        model_id = model_url_or_id.strip("/")

    lineage = []
    visited = set()
    current_model = model_id

    while current_model and current_model not in visited:
        lineage.append(f"https://huggingface.co/{current_model}")  # Add URL instead of model name
        visited.add(current_model)

        # Use LLM to get the parent model
        parent_models = extract_lineage_with_llm(current_model)

        # If no parent model found, break
        if not parent_models:
            break
        
        # Assuming the first model in the lineage is the most direct parent
        current_model = parent_models[0]

    return lineage[::-1]  # Reverse to get the lineage from root → current model

# ================================
# Example usage:

if __name__ == "__main__":
    # Suppress all warnings from huggingface_hub (you can customize it more if needed)
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

    # Get the lineage for a model and print only the lineage output as URLs
    lineage = lineage_graph_with_llm("https://huggingface.co/google-bert/bert-base-uncased")
    print("Lineage:", " → ".join(lineage))

    lineage = lineage_graph_with_llm("https://huggingface.co/textattack/bert-base-uncased-imdb")
    print("Lineage:", " → ".join(lineage))

    lineage = lineage_graph_with_llm("https://huggingface.co/justinlamlamlam/open_orca_chat")
    print("Lineage:", " → ".join(lineage))
