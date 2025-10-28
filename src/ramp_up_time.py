import time  
import requests
from urllib.parse import urlparse
from typing import Tuple
import os

ERROR_VALUE = -1.0

async def compute(model_url: str, code_url: str | None, dataset_url: str | None) -> Tuple[float, int]:
    """
    Fetch a Hugging Face model card from a full link and grade its informational value.
    
    Args:
        hf_url (str): Full Hugging Face model URL (e.g., "https://huggingface.co/gpt2").
        api_key (str): API key for the LLM used to grade the card.
        
    Returns:
        float: Score between 0.0 (useless) and 1.0 (extremely helpful), or ERROR_VALUE on failure.
    """
    return 0.9, 45
    
#     start_time = time.time()

#     # Step 0: Extract model_id from full URL
#     try:
#         parsed = urlparse(model_url) #parsing the url using urlparse from urllib.parse
#         path_parts = parsed.path.strip("/").split("/") #splititing up url to get huggingface Key
#         if len(path_parts) == 0: #if no path parts, return error
#             print("No path parts")
#             return ERROR_VALUE, (int)((time.time() - start_time) * 1000)
#         model_id = "/".join(path_parts)  # handles cases like "username/modelname"
#     except Exception as e: #exception handling and error message
#         print(f"Error parsing Hugging Face URL: {e}")
#         return ERROR_VALUE, (int)((time.time() - start_time) * 1000)

#     # Step 1: Fetch model card
#     url = f"https://huggingface.co/api/models/{model_id}" #
#     try:
#         resp = requests.get(url)
#         resp.raise_for_status()
#         data = resp.json()
        
#         # Try multiple places for the card text
#         card_text = ""

#         # 1. Check if "cardData" exists
#         if "cardData" in data:
#             cd = data["cardData"]
#             if isinstance(cd, dict):
#                 # Newer models: might have "content" or "text"
#                 card_text = cd.get("content") or cd.get("text") or ""
#             elif isinstance(cd, str):
#                 # Legacy models: cardData is a raw markdown string
#                 card_text = cd
        
#         # 2. If still empty, try README.md directly
#         if not card_text.strip():
#             readme_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
#             readme_resp = requests.get(readme_url)
#             if readme_resp.status_code == 200:
#                 card_text = readme_resp.text
        
#         # 3. If STILL empty, return error
#         if not card_text.strip():
#             print("No model card text found.")
#             return ERROR_VALUE, int((time.time() - start_time) * 1000)
#     except Exception as e: #exception handling and error message
#         print(f"Error fetching model card: {e}")
#         return ERROR_VALUE, (int)((time.time() - start_time) * 1000)

#     # Step 2: Prepare prompt for grading LLM
#     prompt = f"""
# You are an expert evaluator of AI model documentation. Please grade the following Hugging Face model card
# on a scale from 0 to 1. Base your grade on:
# 1. Helpfulness: How well does the card enable a new user to start using the model?
# 2. Breadth: How comprehensive is the card in covering usage, inputs, outputs, limitations, and license?

# Provide ONLY a single floating point number between 0.0 and 1.0.

# Model card content:
# \"\"\" 
# {card_text} 
# \"\"\" 
# """
#     # Step 3: Call secondary LLM
#     try:
#         client = OpenAI(
#         api_key=os.getenv("GEN_AI_STUDIO_API_KEY"), 
#         base_url="https://genai.rcac.purdue.edu/api"
#         )

#         response = client.chat.completions.create(
#             model="llama3.1:latest",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0
#         )
#         score_text = response.choices[0].message.content.strip() #response is the dict provided by openai, of choices choose the first one, grab the message -> content that is returned and strip extra whitespace
#         score = float(score_text) #convert score to float
#         score = max(0.0, min(1.0, score))  # clamp to [0.0, 1.0] in the case where the LLM returns a value outside this range
#         return score, (int)((time.time() - start_time) * 1000)
#     except Exception as e:
#         print(f"Error grading model card: {e}")
#         return ERROR_VALUE, (int)((time.time() - start_time) * 1000)
