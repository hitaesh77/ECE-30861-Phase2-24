import time  
import requests
import openai
from urllib.parse import urlparse

ERROR_VALUE = -1.0

def ramp_up_time(payload : dict, api_key: str) -> float:
    """
    Fetch a Hugging Face model card from a full link and grade its informational value.
    
    Args:
        hf_url (str): Full Hugging Face model URL (e.g., "https://huggingface.co/gpt2").
        api_key (str): API key for the LLM used to grade the card.
        
    Returns:
        float: Score between 0.0 (useless) and 1.0 (extremely helpful), or ERROR_VALUE on failure.
    """
    
    start_time = time.time()

    # Step 0: Extract model_id from full URL
    try:
        parsed = urlparse(payload.get(url)) #parsing the url using urlparse from urllib.parse
        path_parts = parsed.path.strip("/").split("/") #splititing up url to get huggingface Key
        if len(path_parts) == 0: #if no path parts, return error
            return ERROR_VALUE, (time.time() - start_time) * 1000
        model_id = "/".join(path_parts)  # handles cases like "username/modelname"
    except Exception as e: #exception handling and error message
        print(f"Error parsing Hugging Face URL: {e}")
        return ERROR_VALUE, (time.time() - start_time) * 1000

    # Step 1: Fetch model card
    url = f"https://huggingface.co/api/models/{model_id}" #
    try:
        resp = requests.get(url) #fetching the model card from huggingface api
        resp.raise_for_status() #checks if the URL is a valid url if not an exception will be raise
        data = resp.json() #converting response to json
        card_text = data.get("cardData", {}).get("content", "") #parsing neccessary information from json
        
        if not card_text.strip(): #if not card text
            return ERROR_VALUE, (time.time() - start_time) * 1000 #return error value
    except Exception as e: #exception handling and error message
        print(f"Error fetching model card: {e}")
        return ERROR_VALUE, (time.time() - start_time) * 1000

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
        return score, (time.time() - start_time) * 1000
    except Exception as e:
        print(f"Error grading model card: {e}")
        return ERROR_VALUE, (time.time() - start_time) * 1000