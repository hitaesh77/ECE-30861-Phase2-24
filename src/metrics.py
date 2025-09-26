import asyncio
import concurrent.futures
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch #using pytorch framwework for model manipulation. chose pytorch vs tensorflow because of its variability and similarity to python syntax and simplicity (easier ramp up)
# import time
# import requests
# import openai
from urllib.parse import urlparse
from run import UrlCategory, Provider
import logging
from typing import Dict, Tuple

ERROR_VALUE = -1.0

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
