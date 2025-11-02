import math
import pytest

# If your file is at src/src/bus_factor.py:
from src.bus_factor import normalize_model_url, compute, ERROR_VALUE

# ---------------- normalize_model_url ----------------

def test_normalize_full_hf_url_basic():
    url = "https://huggingface.co/google-bert/bert-base-uncased"
    assert normalize_model_url(url) == "google-bert/bert-base-uncased"

def test_normalize_idempotent_for_repo_id():
    assert normalize_model_url("openai-community/gpt2") == "openai-community/gpt2"

def test_normalize_trailing_slash():
    url = "https://huggingface.co/t5-small/"
    assert normalize_model_url(url) == "t5-small"

def test_normalize_extra_segments_current_behavior():
    # Current implementation returns the full path after domain, even with extras.
    # If you later change normalize_model_url to trim extras, update this expected value.
    url = "https://huggingface.co/org/model/blob/main/README.md"
    assert normalize_model_url(url) == "org/model/blob/main/README.md"

# ---------------- compute (async) ----------------

@pytest.mark.asyncio
async def test_compute_contract_basic():
    score, latency_ms = await compute("https://huggingface.co/org/model", None, None)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert isinstance(latency_ms, int)
    assert latency_ms >= 0

@pytest.mark.asyncio
async def test_compute_is_deterministic_for_same_inputs():
    s1, t1 = await compute("https://huggingface.co/org/model", None, None)
    s2, t2 = await compute("https://huggingface.co/org/model", None, None)
    # Deterministic score (latency may vary in real impl, but in the stub it's constant)
    assert s1 == s2
    assert 0.0 <= s1 <= 1.0
    assert t1 >= 0 and t2 >= 0

@pytest.mark.asyncio
async def test_compute_ignores_aux_urls_for_now():
    # Given the current stub implementation, code_url/dataset_url do not affect output.
    s0, _ = await compute("https://huggingface.co/org/model", None, None)
    s1, _ = await compute("https://huggingface.co/org/model", "https://example.com/code", None)
    s2, _ = await compute("https://huggingface.co/org/model", None, "https://example.com/dataset")
    s3, _ = await compute("https://huggingface.co/org/model", "https://example.com/code", "https://example.com/dataset")
    assert s0 == s1 == s2 == s3
