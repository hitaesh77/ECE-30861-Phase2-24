import sys
import types
import pytest

import size_score as ss

# ----------------- helpers -----------------

def fake_hf_response(siblings_sizes_bytes):
    """Return a minimal fake requests response for HF API."""
    class Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"siblings": [{"size": s} for s in siblings_sizes_bytes]}
    return Resp()

# ----------------- tests -----------------

@pytest.mark.asyncio
async def test_early_invalid_url_returns_error_value_and_latency(caplog):
    scores, latency = await ss.compute("", None, None)
    assert scores == ss.ERROR_VALUE
    assert isinstance(latency, (int, float)) and latency >= 0
    assert any("[size_score] No valid Hugging Face model URL." in r.message for r in caplog.records)

    scores2, latency2 = await ss.compute("https://example.com/not-hf", None, None)
    assert scores2 == ss.ERROR_VALUE and latency2 >= 0

@pytest.mark.asyncio
@pytest.mark.parametrize("device,limit", list(ss.DEVICE_LIMITS.items()))
async def test_thresholds_by_device(monkeypatch, device, limit):
    # ≤ 0.5*limit => 1.0
    monkeypatch.setattr(ss, "_estimate_model_size_gb", lambda url: 0.5 * limit)
    scores, _ = await ss.compute("https://huggingface.co/org/model", None, None)
    assert scores[device] == 1.0

    # ≤ limit (but > 0.5*limit) => 0.5
    monkeypatch.setattr(ss, "_estimate_model_size_gb", lambda url: limit * 0.9)
    scores, _ = await ss.compute("https://huggingface.co/org/model", None, None)
    assert scores[device] == 0.5

    # > limit => 0.0
    monkeypatch.setattr(ss, "_estimate_model_size_gb", lambda url: limit * 1.01)
    scores, _ = await ss.compute("https://huggingface.co/org/model", None, None)
    assert scores[device] == 0.0

@pytest.mark.asyncio
async def test_api_size_path_bytes_to_gb(monkeypatch):
    import size_score as m
    import requests

    calls = {}

    def fake_get(url, timeout=10):
        calls["url"] = url
        # 1 GiB + 512 MiB = 1.5 GiB
        class Resp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"siblings": [
                    {"size": 1024**3},
                    {"size": 512 * 1024**2},
                ]}
        return Resp()

    # Patch the real requests.get so the local `import requests` sees this
    monkeypatch.setattr(requests, "get", fake_get, raising=True)

    size_gb = m._estimate_model_size_gb("https://huggingface.co/org/model")
    assert 1.49 < size_gb < 1.51  # ~1.5 GB
    assert "api/models/org/model" in calls["url"]


@pytest.mark.asyncio
async def test_api_failure_falls_back_to_heuristics(monkeypatch):
    import size_score as m

    # Force API path to raise so code hits heuristics
    def boom(*a, **k): 
        raise RuntimeError("net")
    m.requests = types.SimpleNamespace(get=boom)

    # Heuristic: keywords in URL → sizes
    assert m._estimate_model_size_gb("https://huggingface.co/org/tiny-model") == 0.2
    assert m._estimate_model_size_gb("https://huggingface.co/org/small-model") == 0.2
    assert m._estimate_model_size_gb("https://huggingface.co/org/base-model") == 0.8
    assert m._estimate_model_size_gb("https://huggingface.co/org/medium-model") == 2.0
    assert m._estimate_model_size_gb("https://huggingface.co/org/large-model") == 8.0
    assert m._estimate_model_size_gb("https://huggingface.co/org/xxl-model") == 20.0
    # Default when no keywords:
    assert m._estimate_model_size_gb("https://huggingface.co/org/unknown") == 5.0

@pytest.mark.asyncio
async def test_success_path_latency_is_int(monkeypatch):
    # Any fixed size to trigger success path
    monkeypatch.setattr(ss, "_estimate_model_size_gb", lambda url: 1.0)
    scores, latency = await ss.compute("https://huggingface.co/org/model", None, None)
    assert isinstance(scores, dict) and set(scores) == set(ss.DEVICE_LIMITS)
    assert isinstance(latency, int) and latency >= 0

@pytest.mark.asyncio
async def test_exception_in_compute_returns_error_value(monkeypatch, caplog):
    def raise_in_estimator(url): 
        raise RuntimeError("boom")
    monkeypatch.setattr(ss, "_estimate_model_size_gb", raise_in_estimator)

    scores, latency = await ss.compute("https://huggingface.co/org/model", None, None)
    assert scores == ss.ERROR_VALUE
    assert isinstance(latency, int) and latency >= 0
    assert any("[size_score] Error computing for https://huggingface.co/org/model" in r.message for r in caplog.records)
