import asyncio
import concurrent.futures
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch #using pytorch framwework for model manipulation. chose pytorch vs tensorflow because of its variability and similarity to python syntax and simplicity (easier ramp up)
import time
import requests
import openai
from urllib.parse import urlparse
from run import UrlCategory, Provider
import logging
from typing import Dict, Tuple

ERROR_VALUE = -1.0

def bus_factor(model_link: str, eval_prompts=None, eval_answers=None) -> tuple:
    """
    Calculates the bus factor (robustness to ablation) for a Hugging Face model.
    Returns (score, latency_ms).
    """

    # Start timing
    start_time = time.time()

    # Try to load the model + tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_link) #need tokenizer to convert text to language LLM can understand. using huggingface tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_link)  #loading model from huggingface for analysis
        model.eval()
    except Exception as e:
        #print(f"[Error] Could not load model: {e}") //log file will print to stdout
        return ERROR_VALUE, 0.0  # return error value and 0 ms latency

    # Default prompts if none provided
    if eval_prompts is None: #tehnically eval prompts will always be none however fucnitonality exists in case spesific eval prompts would like to be used
        eval_prompts = [
            "The capital of France is",
            "The chemical symbol for water is",
            "The largest planet in our solar system is",
            "The author of '1984' is",
            "The square root of 16 is"
        ]

    # Default answers
    if eval_answers is None:
        eval_answers = ["Paris", "H2O", "Jupiter", "George Orwell", "4"]

    # Pick CPU or GPU | Not sure if this is needed for the scope of this project
    device = "cuda" if torch.cuda.is_available() else "cpu" #if a GPU is available, use it, if not use CPU
    model.to(device)

    # Function to evaluate accuracy
    def eval_model(m):
        correct = 0
        for prompt, ans in zip(eval_prompts, eval_answers): #for each prompt with each answer (zipping them together to input to model)
            inputs = tokenizer(prompt, return_tensors="pt").to(device) #tokenized inputs, return_tensors simply tells our oenizer to retun in pyTorch, and the toDevice tells the tokenizer to keep using the gpu/cpu
            with torch.no_grad(): #with torch.no_grad() to tell our code that we do not need gradients since we are not trainnig a model
                outputs = m.generate(**inputs, max_new_tokens=5) #function to ask the model to generate relevant outputs by inpacking our iniputs using ** and asking it to at most send 5 tokens
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True) #decoding tokenizied output back to plainitext
            if ans.lower() in decoded.lower(): #if the decoded answer is in the response
                correct += 1 #increment correct
        return correct / len(eval_prompts) #normalize output

    # Baseline accuracy (no ablation)
    baseline_acc = eval_model(model) #evaluate the model using the baseline model

    # Function to apply random ablation
    def ablate_model(m, fraction=0.1):#function to ablate model (turning off some of the weights)
        def hook_fn(module, input, output): #functoin to turn off weights
            mask = (torch.rand_like(output) > fraction).float() #creates a tensor of the shape of output with random numbers. if each number has an equally likely chance of being generated then we get rid of each value under .1. we cast ths to a float which ini theory gives us eiither 1.0 or 0.0
            return output * mask #muuliply output by mask, so fraction% of values will be zeroed out
        handles = []
        for _, mod in m.named_modules(): # m.named_modules() returns the names for th emoduels and the module layer objects for each layer
            if isinstance(mod, torch.nn.Linear): #checking the type of module to only look for linear layers, nonlinear layers are skipped
                handles.append(mod.register_forward_hook(hook_fn)) #run the hook_fn function to the layer so that every forward pass on that layer randomy cancells out some neurons
        return handles

    # Ablation fractions
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
    robustness_scores = []

    # Evaluate robustness under ablation
    for frac in fractions:
        handles = ablate_model(model, fraction=frac) #create ablated model
        ablated_acc = eval_model(model) #evaluate the ablated model
        for h in handles:
            h.remove()  # remove hooks after use

        drop = max(0.0, baseline_acc - ablated_acc) #comparing ablated scores to baseline scores 0.0 means nothing really changed, 1.0 means completely changed
        robustness = max(0.0, min(1.0, 1.0 - drop / max(1e-5, baseline_acc))) #robustness score is 1.0 - drop normalized by baseline accuracy (max with 1e-5 to avoid div by zero). robustness score to check how well our model does in these new conditions
        robustness_scores.append(robustness)

    # Average robustness score
    score = sum(robustness_scores) / len(robustness_scores)

    # Stop timing and calculate latency in ms
    latency_ms = (time.time() - start_time) * 1000

    return score, latency_ms

def size():
    return 1

def ramp_up_time(hf_url: str, api_key: str) -> tuple:
    """
    Fetch a Hugging Face model card from a full link and grade its informational value.
    
    Args:
        hf_url (str): Full Hugging Face model URL (e.g., "https://huggingface.co/gpt2").
        api_key (str): API key for the LLM used to grade the card.
        
    Returns:
        float: Score between 0.0 (useless) and 1.0 (extremely helpful), or ERROR_VALUE on failure.
    """
    # Start timing
    start_time = time.time()

    # Step 0: Extract model_id from full URL
    try:
        parsed = urlparse(hf_url) #parsing the url using urlparse from urllib.parse
        path_parts = parsed.path.strip("/").split("/") #splititing up url to get huggingface Key
        if len(path_parts) == 0: #if no path parts, return error
            return ERROR_VALUE, 0.0
        model_id = "/".join(path_parts)  # handles cases like "username/modelname"
    except Exception as e: #exception handling and error message
        print(f"Error parsing Hugging Face URL: {e}")
        return ERROR_VALUE, 0.0

    # Step 1: Fetch model card
    url = f"https://huggingface.co/api/models/{model_id}" #
    try:
        resp = requests.get(url) #fetching the model card from huggingface api
        resp.raise_for_status() #checks if the URL is a valid url if not an exception will be raise
        data = resp.json() #converting response to json
        card_text = data.get("cardData", {}).get("content", "") #parsing neccessary information from json
        
        if not card_text.strip(): #if not card text
            return ERROR_VALUE, 0.0 #return error value
    except Exception as e: #exception handling and error message
        print(f"Error fetching model card: {e}")
        return ERROR_VALUE, 0.0

    # Step 2: Prepare prompt for grading LLM
    prompt = f"""
You are an expert evaluator of AI model documentation. Please grade the following Hugging Face model card
on a scale from 0 to 1. Base your grade on:
1. Helpfulness: How well does the card enable a new user to start using the model?
2. Breadth: How comprehensive is the card in covering usage, inputs, outputs, limitations, and license?

Provide ONLY a single floating point number between 0.0 and 1.0.

Model card content:
\"\"\" 
{card_text} 
\"\"\" 
"""
    # Step 3: Call secondary LLM
    try:
        openai.api_key = api_key #set api key for openai
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # desired model endpoint for grading
            messages=[{"role": "user", "content": prompt}], #formatting prompt for LLM model
            temperature=0.0 #temperature 0.0 for deterministic output
        )
        score_text = response["choices"][0]["message"]["content"].strip() #response is the dict provided by openai, of choices choose the first one, grab the message -> content that is returned and strip extra whitespace
        score = float(score_text) #convert score to float
        score = max(0.0, min(1.0, score))  # clamp to [0.0, 1.0] in the case where the LLM returns a value outside this range
        return score, ((time.time() - start_time) * 1000)  # return latency in ms
    except Exception as e:
        print(f"Error grading model card: {e}")
        return ERROR_VALUE, 0.0

def correctness():
    return 1

def license():
    return 1

def netscore():
    return 1

# async functions -- for api bound calculations
async def performance_claims():
    await asyncio.sleep(0.2)  # simulates api latency
    return 1

async def responsive_maintainer():
    await asyncio.sleep(0.4)
    return 1

async def code_quality():
    await asyncio.sleep(0.3)
    return 1

async def dataset_quality():
    await asyncio.sleep(0.6)
    return 1
def dataset_code_score(hf_link: str, api_key: str) -> float:
    """
    Grade how well a Hugging Face model's code and dataset are documented.
    
    Args:
        hf_link (str): Full Hugging Face link (e.g., "https://huggingface.co/bert-base-uncased").
        api_key (str): API key for the LLM used to evaluate documentation.
        
    Returns:
        float: Score between 0.0 and 1.0, or ERROR_VALUE if something fails.
    """

    # --- Step 1: Parse model_id ---
    try:
        if not hf_link.startswith("https://huggingface.co/"): #if model not huggingface link throw error
            return ERROR_VALUE
        model_id = hf_link.replace("https://huggingface.co/", "").strip("/") #removing the huggingface part of the link to get the model id
    except Exception as e:
        print(f"Error parsing Hugging Face link: {e}") #exception if huggingface is wrong
        return ERROR_VALUE

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
        return ERROR_VALUE

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
    combined_text = f"""
=== MODEL CARD === in case of no model card, tell no model card provided. Ideally there will be a model card if it is a valid link but private models, new/experiemntal models and minimal repos that are just weights will not hav a card
{model_card if model_card else "NO MODEL CARD PROVIDED"} 

=== DATASETS ===
{dataset_text} 

=== CODE SNIPPETS ===
{"\n\n".join(code_texts)}
"""

    if not combined_text.strip(): #if no text at all, return error
        return ERROR_VALUE

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

    try: #call LLM to grade documentation based off prompt, check prior documentation for explaination
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        score_text = response["choices"][0]["message"]["content"].strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))  # clamp to [0,1]
    except Exception as e:
        print(f"Error grading documentation: {e}")
        return ERROR_VALUE


# run tasks
async def run_metrics(category: UrlCategory, provider: Provider, ids: Dict[str, str], metric_scores: Dict) -> Dict[str, float]:
    """Run all relevant metrics based on URL classification."""
    tasks = []
    task_names = []

    # Common metrics for all types
    tasks.extend([
        responsive_maintainer(),
        code_quality()
    ])
    task_names.extend(['responsive_maintainer', 'code_quality'])

    # Add category-specific metrics
    if category == UrlCategory.MODEL:
        tasks.extend([
            performance_claims(),
            bus_factor(ids['url']),
            size(),
            ramp_up_time(),
            license()
        ])
        task_names.extend([
            'performance_claims',
            'bus_factor',
            'size',
            'ramp_up_time',
            'license'
        ])

    elif category == UrlCategory.DATASET:
        tasks.extend([
            dataset_quality(),
            dataset_code_score()
        ])
        task_names.extend(['dataset_quality', 'dataset_code_score'])

    elif category == UrlCategory.CODE:
        tasks.extend([
            code_quality(),
            license()
        ])
        task_names.extend(['code_quality', 'license'])

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Store results in metric_scores dictionary
    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            logging.error(f"Error in metric {name}: {result}")
            metric_scores[name] = ERROR_VALUE
            metric_scores[f"{name}_latency"] = 0.0
        else:
            # All metrics now return (score, latency)
            score, latency = result
            metric_scores[name] = score
            metric_scores[f"{name}_latency"] = latency

    return metric_scores
