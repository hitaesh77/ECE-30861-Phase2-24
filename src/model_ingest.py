from typing import Dict, TypedDict, Literal
from metrics import GradeResult, SizeScore

def model_ingest(metric_scores: GradeResult) -> bool:
    # Add all the metrics that need to be checked here. (EXCEPT SIZE)
    # It should include all non-latency metrics that have subscores.
    metrics_to_check = [
        "net_score",
        "ramp_up_time",
        "bus_factor",
        "performance_claims",
        "license",
        "dataset_and_code_score",
        "dataset_quality",
        "code_quality",
    ]

    for metric in metrics_to_check:
        if metric_scores[metric] < 0.5:
            return False

    # Size has its own subscores so do that separately
    for size_metric, size_subscore in metric_scores["size_score"].items():
        if size_subscore < 0.5:
            return False

    return True

# --- Test Data ---
sample_good_result: GradeResult = {
    "name": "Good Model",
    "category": "MODEL",
    "net_score": 0.9,
    "net_score_latency": 120,
    "ramp_up_time": 0.8,
    "ramp_up_time_latency": 90,
    "bus_factor": 0.75,
    "bus_factor_latency": 100,
    "performance_claims": 0.88,
    "performance_claims_latency": 110,
    "license": 1.0,
    "license_latency": 80,
    "size_score": {
        "raspberry_pi": 0.9,
        "jetson_nano": 0.85,
        "desktop_pc": 0.95,
        "aws_server": 0.98,
    },
    "size_score_latency": 60,
    "dataset_and_code_score": 0.77,
    "dataset_and_code_score_latency": 70,
    "dataset_quality": 0.8,
    "dataset_quality_latency": 55,
    "code_quality": 0.92,
    "code_quality_latency": 50,
}

sample_bad_result: GradeResult = {
    "name": "Bad Model",
    "category": "MODEL",
    "net_score": 0.25,
    "net_score_latency": 150,
    "ramp_up_time": 0.7,
    "ramp_up_time_latency": 95,
    "bus_factor": 0.9,
    "bus_factor_latency": 80,
    "performance_claims": 0.6,
    "performance_claims_latency": 85,
    "license": 0.9,
    "license_latency": 70,
    "size_score": {
        "raspberry_pi": 0.4,
        "jetson_nano": 0.6,
        "desktop_pc": 0.8,
        "aws_server": 0.9,
    },
    "size_score_latency": 65,
    "dataset_and_code_score": 0.55,
    "dataset_and_code_score_latency": 90,
    "dataset_quality": 0.75,
    "dataset_quality_latency": 60,
    "code_quality": 0.88,
    "code_quality_latency": 55,
}

import random
sample_random_result: GradeResult = {
    "name": "Random Model",
    "category": "MODEL",
    "net_score": random.random() ** 0.25,
    "net_score_latency": random.randint(30, 300),
    "ramp_up_time": random.random() ** 0.25,
    "ramp_up_time_latency": random.randint(30, 300),
    "bus_factor": random.random() ** 0.25,
    "bus_factor_latency": random.randint(30, 300),
    "performance_claims": random.random() ** 0.25,
    "performance_claims_latency": random.randint(30, 300),
    "license": random.random() ** 0.25,
    "license_latency": random.randint(30, 300),
    "size_score": {
        "raspberry_pi": random.random() ** 0.25,
        "jetson_nano": random.random() ** 0.25,
        "desktop_pc": random.random() ** 0.25,
        "aws_server": random.random() ** 0.25,
    },
    "size_score_latency": random.randint(30, 300),
    "dataset_and_code_score": random.random() ** 0.25,
    "dataset_and_code_score_latency": random.randint(30, 300),
    "dataset_quality": random.random() ** 0.25,
    "dataset_quality_latency": random.randint(30, 300),
    "code_quality": random.random() ** 0.25,
    "code_quality_latency": random.randint(30, 300),
}

print(sample_random_result)
print()
print("Expected True from sample_good_result: ", model_ingest(sample_good_result))
print("Expected False from sample_bad_result: ", model_ingest(sample_bad_result))
print("Manually verify for sample_random_result: ", model_ingest(sample_random_result))
