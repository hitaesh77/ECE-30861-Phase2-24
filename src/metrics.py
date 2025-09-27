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


def bus_factor():
    return 1


def correctness():
    return 1


def license():
    return 1


def netscore():
    return 1


def ramp_up_time():
    return 1


def size():
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


async def dataset_code_score():
    await asyncio.sleep(0.6)
    return 1


# run tasks
async def run_metrics(urls: Dict[str, str]) -> GradeResult:
    print(f"running all metrics {urls}")
    """Run all relevant metrics based on URL classification."""
    tasks = []
    task_names = []

    # Common metrics for all types
    tasks.extend([
        responsive_maintainer(urls),
        code_quality(urls),
        performance_claims(urls),
        bus_factor(urls),
        size(urls),
        ramp_up_time(urls),
        license(urls),
        dataset_quality(urls),
        dataset_code_score(urls),
        code_quality(urls),
    ])
    task_names.extend(['responsive_maintainer', 
                        'code_quality', 
                        'performance_claims',
                        'bus_factor',
                        'size',
                        'ramp_up_time',
                        'license',
                        'dataset_quality',
                        'dataset_code_score',
                        'code_quality'])

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    metric_scores: GradeResult = {}
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
