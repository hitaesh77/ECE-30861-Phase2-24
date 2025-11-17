import asyncio
from src.bus_factor import compute

def test_compute_contract_basic():
    score, latency_ms = asyncio.run(compute("https://huggingface.co/org/model", None, None))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert isinstance(latency_ms, int)
    assert latency_ms >= 0

def test_compute_is_deterministic_for_same_inputs():
    s1, t1 = asyncio.run(compute("https://huggingface.co/org/model", None, None))
    s2, t2 = asyncio.run(compute("https://huggingface.co/org/model", None, None))
    assert s1 == s2
    assert 0.0 <= s1 <= 1.0
    assert t1 >= 0 and t2 >= 0

def test_compute_ignores_aux_urls_for_now():
    s0, _ = asyncio.run(compute("https://huggingface.co/org/model", None, None))
    s1, _ = asyncio.run(compute("https://huggingface.co/org/model", "https://example.com/code", None))
    s2, _ = asyncio.run(compute("https://huggingface.co/org/model", None, "https://example.com/dataset"))
    s3, _ = asyncio.run(compute("https://huggingface.co/org/model", "https://example.com/code", "https://example.com/dataset"))
    assert s0 == s1 == s2 == s3
