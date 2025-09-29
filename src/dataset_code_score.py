import requests
import openai
from urllib.parse import urlparse
from typing import Dict, Tuple
import time
import os

ERROR_VALUE = -1.0

async def compute(model_url: str, code_url: str | None, dataset_url: str | None) -> Tuple[float, int]:
    """
    Grade how well a Hugging Face model's code and dataset are documented.
    
    Args:
        hf_link (str): Full Hugging Face link (e.g., "https://huggingface.co/bert-base-uncased").
        api_key (str): API key for the LLM used to evaluate documentation.
        
    Returns:
        float: Score between 0.0 and 1.0, or ERROR_VALUE if something fails.
    """
    start_time = time.time()
    # --- Step 1: Parse model_id ---
    try:
        model_id = model_url.replace("https://huggingface.co/", "").strip("/") #removing the huggingface part of the link to get the model id
    except Exception as e:
        print(f"Error parsing Hugging Face link: {e}") #exception if huggingface is wrong
        return ERROR_VALUE, (int)((time.time() - start_time) * 1000)

    # --- Step 2: Fetch model card & datasets ---
    try:
        url = f"https://huggingface.co/api/models/{model_id}" #fetching model card, check previous documentation for explaination
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        
        model_card = data.get("cardData", {}).get("content", "")
        datasets = data.get("datasets", [])
    except Exception as e:
        print(f"Error fetching model info: {e}")
        return ERROR_VALUE, (int)((time.time() - start_time) * 1000)

    # Handle missing dataset
    dataset_text = "\n".join(datasets) if datasets else "NO DATASET PROVIDED" #if no datasets provided, return no dataset provided

    # --- Step 3: Fetch repository files ---
    try:
        files_url = f"https://huggingface.co/api/models/{model_id}/tree/main" #fetching model files from huggingface api
        files_resp = requests.get(files_url) #requesting files based on the url
        files_resp.raise_for_status() #checking if the url is valid
        all_files = [f["path"] for f in files_resp.json()] #parsing the json to get the file paths
    except Exception: #if exception thrown, just return empty list, implies that there is no code linked
        all_files = []

    # Accept any text-like files (not just .py)
    text_like_ext = (".py", ".md", ".txt", ".json", ".yaml", ".yml") #accepting any text like files
    code_files = [f for f in all_files if f.endswith(text_like_ext)] #filtering files to only include text like files

    code_texts = []
    if code_files:
        for f in code_files[:3]:  # limit to 3 files
            try:
                raw_url = f"https://huggingface.co/{model_id}/resolve/main/{f}" #creating url to later fetch raw file contents from huggingface
                file_resp = requests.get(raw_url) #fetching raw file
                if file_resp.status_code == 200:
                    # take at most ~2000 chars to keep LLM input efficient
                    code_texts.append(file_resp.text[:2000])
            except Exception:
                continue
    else:
        code_texts.append("NO CODE FILES PROVIDED")

    # --- Step 4: Build combined evaluation text ---
    joined = "\n\n".join(code_texts)
    combined_text = f"""
=== MODEL CARD === in case of no model card, tell no model card provided. Ideally there will be a model card if it is a valid link but private models, new/experiemntal models and minimal repos that are just weights will not hav a card
{model_card if model_card else "NO MODEL CARD PROVIDED"} 

=== DATASETS ===
{dataset_text} 

=== CODE SNIPPETS ===
{joined}"""

    if not combined_text.strip():  # if no text at all, return error
        return ERROR_VALUE, (int)((time.time() - start_time) * 1000)

    # --- Step 5: Prompt LLM for grading ---
    prompt = f"""
You are an expert evaluator of ML repositories.

Grade the following Hugging Face repository on documentation quality.
Criteria:
1. Code documentation (inline comments, docstrings, clarity of usage).
2. Dataset documentation (description, licensing, intended use).
3. Model card completeness (instructions, examples, limitations).

If dataset or code files are missing, assume they score 0.0 and only evaluate what's available.

Return ONLY a floating-point number between 0.0 (very poorly documented) and 1.0 (excellent documentation).

Repository content:
\"\"\" 
{combined_text}
\"\"\"
"""

    try:  # call LLM to grade documentation based off prompt, check prior documentation for explaination
        openai.api_key = os.getenv("GEN_AI_STUDIO_API_KEY")
        openai.api_base = "https://genai.rcac.purdue.edu/api"

        response = openai.ChatCompletion.create(
            model= "llama3.1:latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        score_text = response["choices"][0]["message"]["content"].strip()
        score = float(score_text)
        return max(0.0, min(1.0, score)), (int)((time.time() - start_time) * 1000)  # clamp to [0,1]
    except Exception as e:
        print(f"Error grading documentation: {e}")
        return ERROR_VALUE, (int)((time.time() - start_time) * 1000)
