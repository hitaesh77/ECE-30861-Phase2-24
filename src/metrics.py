# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch #using pytorch framwework for model manipulation. chose pytorch vs tensorflow because of its variability and similarity to python syntax and simplicity (easier ramp up)
# import time
# import requests
# import openai
import asyncio
from urllib.parse import urlparse
import logging
from typing import Dict, TypedDict, Literal
from enum import Enum

# Metric function imports
from name import compute as name
from category import compute as category
from netscore import compute as net_score
from ramp_up_time import compute as ramp_up_time
from bus_factor import compute as bus_factor
from performance_claims import compute as performance_claims
from license import compute as license
from size_score import compute as size
from dataset_code_score import compute as dataset_and_code_score
from dataset_quality import compute as dataset_quality
from code_quality import compute as code_quality

ERROR_VALUE = -1.0


class UrlCategory(str, Enum):
    MODEL = "MODEL"
    DATASET = "DATASET"
    CODE = "CODE"


# Optional: if you want to branch logic later


class Provider(str, Enum):
    HUGGINGFACE = "huggingface"
    GITHUB = "github"
    OTHER = "other"


# ---- Domain: NDJSON output schema for MODEL lines ----


class SizeScore(TypedDict):
    raspberry_pi: float
    jetson_nano: float
    desktop_pc: float
    aws_server: float


class GradeResult(TypedDict):
    name: str
    category: Literal["MODEL"]
    net_score: float
    net_score_latency: int
    ramp_up_time: float
    ramp_up_time_latency: int
    bus_factor: float
    bus_factor_latency: int
    performance_claims: float
    performance_claims_latency: int
    license: float
    license_latency: int
    size_score: SizeScore
    size_score_latency: int
    dataset_and_code_score: float
    dataset_and_code_score_latency: int
    dataset_quality: float
    dataset_quality_latency: int
    code_quality: float
    code_quality_latency: int


# run tasks
async def run_metrics(urls: Dict[UrlCategory, str]) -> GradeResult:
    print(f"running all metrics {urls}")

    model_url = urls.get(UrlCategory.MODEL)
    dataset_url = urls.get(UrlCategory.DATASET)
    code_url = urls.get(UrlCategory.CODE)

    # List of (metric_name, metric_func) pairs
    metric_funcs = [
        ("name", name, 1),
        ("category", category, 0),
        ("code_quality", code_quality, 0),
        ("performance_claims", performance_claims, 0),
        ("bus_factor", bus_factor, 0),
        ("size", size, 0),
        ("ramp_up_time", ramp_up_time, 0),
        ("license", license, 0),
        ("dataset_quality", dataset_quality, 0),
        ("dataset_and_code_score", dataset_and_code_score, 0),
    ]

    # Build tasks and names in sync
    task_names = [name for name, _, en in metric_funcs if en]
    tasks = [func(model_url, code_url, dataset_url) for _, func, en in metric_funcs if en]

    print("running tasks: ", task_names)

    # Run them concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    metric_scores: GradeResult = {}

    # Store results
    for n, result in zip(task_names, results):
        if n == "name" or n == "category":
            if isinstance(result, Exception):
                logging.error(f"Error in metric {n}: {result}")
                metric_scores[n] = ERROR_VALUE
            else:
                score, latency = result
                metric_scores[n] = score
        else:
            if isinstance(result, Exception):
                logging.error(f"Error in metric {n}: {result}")
                metric_scores[n] = ERROR_VALUE
                metric_scores[f"{n}_latency"] = 0.0
            else:
                score, latency = result
                metric_scores[n] = score
                metric_scores[f"{n}_latency"] = latency

    # Compute net score last
    net, net_latency = net_score(metric_scores)
    metric_scores["net_score"] = net
    metric_scores["net_score_latency"] = net_latency

    return metric_scores
