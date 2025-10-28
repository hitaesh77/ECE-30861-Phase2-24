from enum import Enum
from typing import Dict, TypedDict, Literal

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

class UrlCategory(str, Enum):
    MODEL = "MODEL"
    DATASET = "DATASET"
    CODE = "CODE"
    OTHER = "OTHER"

class Provider(str, Enum):
    HUGGINGFACE = "huggingface"
    GITHUB = "github"
    OTHER = "other"