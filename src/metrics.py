import asyncio
from urllib.parse import urlparse
import logging
from typing import Dict, TypedDict, Literal
from utils import UrlCategory, Provider

# Metric function imports
from name import compute as name
from category import compute as category
from netscore import compute as net_score
from ramp_up_time import compute as ramp_up_time
from bus_factor import compute as bus_factor
from performance_claims import compute as performance_claims
from license import compute as license
from size_score import compute as size_score
from dataset_code_score import compute as dataset_and_code_score
from dataset_quality import compute as dataset_quality
from code_quality import compute as code_quality

ERROR_VALUE = -1.0

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

    model_url = urls.get(UrlCategory.MODEL)['url']
    dataset_url = urls.get(UrlCategory.DATASET) and urls.get(UrlCategory.DATASET)['url']
    code_url = urls.get(UrlCategory.CODE) and urls.get(UrlCategory.CODE)['url']

    # List of (metric_name, metric_func) pairs
    metric_funcs = [
        ("name", name, 1),
        ("category", category, 1),
        ("code_quality", code_quality, 1),
        ("performance_claims", performance_claims, 1),
        ("bus_factor", bus_factor, 1),
        ("size_score", size_score, 1),
        ("ramp_up_time", ramp_up_time, 1),
        ("license", license, 1),
        ("dataset_quality", dataset_quality, 1),
        ("dataset_and_code_score", dataset_and_code_score, 1),
    ]

    # Build tasks and names in sync
    task_names = [name for name, _, en in metric_funcs if en]
    tasks = [func(model_url, code_url, dataset_url) for _, func, en in metric_funcs if en]

    # Run them concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    metric_scores: GradeResult = {}

    # Store results
    for n, result in zip(task_names, results):
        if n == "name" or n == "category":
            if isinstance(result, Exception):
                logging.error(f"Error in metric {n}:{result}")
                metric_scores[n] = ERROR_VALUE
            else:
                score, latency = result
                metric_scores[n] = score
        else:
            if isinstance(result, Exception):
                logging.error(f"Error in metric {n}:{result}")
                metric_scores[n] = ERROR_VALUE
                metric_scores[f"{n}_latency"] = 0.0
            else:
                score, latency = result
                metric_scores[n] = score
                metric_scores[f"{n}_latency"] = latency

    # TEMPORARY FIX, NEED TO CHANGE
    net_en = 1  # enabled
    if net_en:
        net_score_input = {}
        for k, v in metric_scores.items():
            # Only include keys that are intended to be numeric *scores*
            if not k.endswith("_latency") and k not in ["name", "category", "size_score"]:
                net_score_input[k] = v

        net, net_latency = net_score(net_score_input)
        metric_scores["net_score"] = net
        metric_scores["net_score_latency"] = net_latency


    final_ordered_scores: GradeResult = {}
    
    # 1. Name and Category
    final_ordered_scores["name"] = metric_scores.get("name")
    final_ordered_scores["category"] = metric_scores.get("category")

    # 2. Net Score (REQUIRED POSITION)
    final_ordered_scores["net_score"] = metric_scores.get("net_score")
    final_ordered_scores["net_score_latency"] = metric_scores.get("net_score_latency")

    # 3. The Rest of the Scores (following GradeResult TypedDict order)
    final_ordered_scores["ramp_up_time"] = metric_scores.get("ramp_up_time")
    final_ordered_scores["ramp_up_time_latency"] = metric_scores.get("ramp_up_time_latency")
    final_ordered_scores["bus_factor"] = metric_scores.get("bus_factor")
    final_ordered_scores["bus_factor_latency"] = metric_scores.get("bus_factor_latency")
    final_ordered_scores["performance_claims"] = metric_scores.get("performance_claims")
    final_ordered_scores["performance_claims_latency"] = metric_scores.get("performance_claims_latency")
    final_ordered_scores["license"] = metric_scores.get("license")
    final_ordered_scores["license_latency"] = metric_scores.get("license_latency")
    
    final_ordered_scores["size_score"] = metric_scores.get("size_score")
    final_ordered_scores["size_score_latency"] = metric_scores.get("size_score_latency")
    
    final_ordered_scores["dataset_and_code_score"] = metric_scores.get("dataset_and_code_score")
    final_ordered_scores["dataset_and_code_score_latency"] = metric_scores.get("dataset_and_code_score_latency")
    final_ordered_scores["dataset_quality"] = metric_scores.get("dataset_quality")
    final_ordered_scores["dataset_quality_latency"] = metric_scores.get("dataset_quality_latency")
    final_ordered_scores["code_quality"] = metric_scores.get("code_quality")
    final_ordered_scores["code_quality_latency"] = metric_scores.get("code_quality_latency")

    # The final dictionary 'final_ordered_scores' now has the keys inserted 
    # in the desired order, satisfying the requirement.
    return final_ordered_scores
