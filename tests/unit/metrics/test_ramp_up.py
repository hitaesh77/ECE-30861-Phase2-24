import math
import pytest
from src.ramp_up_time import compute, ERROR_VALUE

@pytest.mark.asyncio
async def test_basic_success(monkeypatch, sample_md, sample_api_json, frozen_time, env_log):
    monkeypatch.setattr("src.ramp_up_time._extract_repo_id", lambda u: "org/model")
    monkeypatch.setattr("src.ramp_up_time._fetch_readme_text", lambda rid: sample_md)
    monkeypatch.setattr("src.ramp_up_time._fetch_api_card", lambda rid: sample_api_json)
    monkeypatch.setattr("src.ramp_up_time._reachable", lambda url: True)

    score, latency_ms = await compute("https://huggingface.co/org/model", "https://x/code", "https://x/ds")
    assert 0.0 <= score <= 1.0
    assert not math.isnan(score)
    assert latency_ms >= 0

@pytest.mark.asyncio
async def test_error_on_bad_url(monkeypatch):
    monkeypatch.setattr("src.ramp_up_time._extract_repo_id", lambda u: None)
    score, _ = await compute("https://not-hf.example.com", None, None)
    assert score == ERROR_VALUE
