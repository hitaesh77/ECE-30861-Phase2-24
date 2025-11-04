import math
import pytest

import dataset_code_score as dcs

# ---------------- extract_score ----------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("0.0", 0.0),
        ("1.0", 1.0),
        ("0.85", 0.85),
        ("Score: 0.85/1.0", 0.85),          # first number wins
        ("Leading int 1 then 0.4", 1.0),     # first number wins (1)
        ("2.5 (should clamp)", 1.0),         # clamp high
        ("-0.5 (regex grabs 0.5)", 0.5),     # current regex doesn't capture '-' sign
        ("No decimal but int 7", 1.0),       # clamp high after cast
        ("0", 0.0),
    ],
)
def test_extract_score_values(text, expected):
    assert dcs.extract_score(text) == expected

def test_extract_score_raises_on_no_number():
    with pytest.raises(ValueError):
        dcs.extract_score("no digits here; N/A")

# ---------------- compute (current stub) ----------------

@pytest.mark.asyncio
async def test_compute_contract_basic():
    score, latency_ms = await dcs.compute(
        "https://huggingface.co/org/model", "https://example.com/code", "https://example.com/ds"
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert isinstance(latency_ms, int)
    assert latency_ms >= 0

@pytest.mark.asyncio
async def test_compute_is_deterministic_for_same_inputs():
    s1, t1 = await dcs.compute("https://huggingface.co/org/model", None, None)
    s2, t2 = await dcs.compute("https://huggingface.co/org/model", None, None)
    assert s1 == s2
    assert t1 >= 0 and t2 >= 0

@pytest.mark.asyncio
async def test_compute_ignores_aux_urls_for_now():
    s0, _ = await dcs.compute("https://huggingface.co/org/model", None, None)
    s1, _ = await dcs.compute("https://huggingface.co/org/model", "https://x/code", None)
    s2, _ = await dcs.compute("https://huggingface.co/org/model", None, "https://x/ds")
    s3, _ = await dcs.compute("https://huggingface.co/org/model", "https://x/code", "https://x/ds")
    assert s0 == s1 == s2 == s3
