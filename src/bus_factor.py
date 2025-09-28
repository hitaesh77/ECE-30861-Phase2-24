from transformers import AutoModelForCausalLM, AutoTokenizer
import torch #using pytorch framwework for model manipulation. chose pytorch vs tensorflow because of its variability and similarity to python syntax and simplicity (easier ramp up)
import time
from typing import Tuple

ERROR_VALUE = -1.0

async def compute(model_url: str, code_url: str | None, dataset_url: str | None) -> Tuple[float, int]:
    """
    Calculates the bus factor (robustness to ablation) for a Hugging Face model.
    Returns (score, latency_ms).
    """

    # Start timing
    start_time = time.time()

    # Try to load the model + tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_url) #need tokenizer to convert text to language LLM can understand. using huggingface tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_url)  #loading model from huggingface for analysis
        model.eval()
    except Exception as e:
        #print(f"[Error] Could not load model: {e}") //log file will print to stdout
        return ERROR_VALUE, (time.time() - start_time) * 1000  # return error value and 0 ms latency

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
    latency_ms = (int)((time.time() - start_time) * 1000)

    return score, latency_ms
