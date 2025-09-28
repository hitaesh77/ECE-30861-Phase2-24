import time
from typing import Dict, Mapping, Tuple

WEIGHTS: Dict[str,float]={
    
    "license":0.18,
    "ramp_up":0.15,
    "bus_factor":0.12,
    "performance_claims":0.1,
    "dataset_and_codescore":0.15,
    "dataset_quality":0.1,
    "code_quality":0.1,
    "size_score":0.1,
}

# We use this function to ensure our score will always be in a range from 0 to 1, inclusive.
async def bounds(x: float, bottom: float = 0, top: float = 1) -> float:
    if x < bottom:
        return bottom
    if x > top:
        return top
    return x

async def compute(metrics: Mapping[str, float]) -> Tuple[float, int]:
    startTime = time.perf_counter_ns()
    
    net = float(0)
    for key, w in WEIGHTS.items():
        net += w * bounds(metrics.get(key,0.0))
        
    net = bounds(net)
    
    latency_ms = int((time.perf_counter_ns() - startTime)/(1000000))
    
    return net, latency_ms
    
    