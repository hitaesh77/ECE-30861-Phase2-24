import pytest
import types

# === Pick ONE of these imports, comment the other ===
# If file is at src/license.py:
import license as license_metric  # noqa: F401
# If file is at src/metrics/license.py:
# from metrics import license as license_metric  # noqa: E402, F401

# -------------------------------------------------------------------
# Helpers: fake requests.get responses, captured URL, and time control
# -------------------------------------------------------------------

class FakeResp:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self._data = data or {}
    def json(self):
        return self._data
    def raise_for_status(self):
        if self.status_code >= 400:
            # mimic requests.HTTPError without importing requests
            raise RuntimeError(f"HTTP {self.status_code}")

@pytest.fixture
def capture_get(monkeypatch):
    """Monkeypatch requests.get to return controlled payloads and capture the URL."""
    state = {"last_url": None, "payload": {}, "status": 200}
    def fake_get(url, *a, **k):
        state["last_url"] = url
        return FakeResp(status=state["status"], data=state["payload"])
    # Patch the module's requests.get (not global requests) to avoid conflicts
    monkeypatch.setattr(license_metric, "requests", types.SimpleNamespace(get=fake_get), raising=True)
    return state

@pytest.fixture
def fixed_time(monkeypatch):
    """Make time monotonic but deterministic; we only assert non-negative ints."""
    t = {"now": 1_700_000_000.0}
    def time_now():
        t["now"] += 0.001
        return t["now"]
    monkeypatch.setattr(license_metric, "time", types.SimpleNamespace(time=time_now))
    return t

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_url_is_built_from_full_hf_link(capture_get, fixed_time):
    capture_get["payload"] = {"cardData": {"license": "mit"}}
    score, ms = await license_metric.compute("https://huggingface.co/org/model", None, None)
    assert score == 1.0 and isinstance(ms, int) and ms >= 0
    assert capture_get["last_url"] == "https://huggingface.co/api/models/org/model"

@pytest.mark.asyncio
async def test_permissive_mit(capture_get, fixed_time):
    capture_get["payload"] = {"cardData": {"license": "mit"}}
    score, _ = await license_metric.compute("https://huggingface.co/gpt2", None, None)
    assert score == 1.0

@pytest.mark.asyncio
async def test_restrictive_gpl3(capture_get, fixed_time):
    capture_get["payload"] = {"cardData": {"license": "gpl-3.0"}}
    score, _ = await license_metric.compute("https://huggingface.co/foo/bar", None, None)
    assert score == 0.7

@pytest.mark.asyncio
async def test_non_commercial_nc(capture_get, fixed_time):
    capture_get["payload"] = {"cardData": {"license": "cc-by-nc-4.0"}}
    score, _ = await license_metric.compute("https://huggingface.co/x/y", None, None)
    assert score == 0.4

@pytest.mark.asyncio
async def test_custom_openrail(capture_get, fixed_time):
    capture_get["payload"] = {"cardData": {"license": "openrail-m"}}
    score, _ = await license_metric.compute("https://huggingface.co/a/b", None, None)
    assert score == 0.6

@pytest.mark.asyncio
async def test_unknown_other_zero(capture_get, fixed_time):
    capture_get["payload"] = {"cardData": {"license": "other"}}
    score, _ = await license_metric.compute("https://huggingface.co/a/b", None, None)
    assert score == 0.0

@pytest.mark.asyncio
async def test_unrecognized_license_falls_back_to_point5(capture_get, fixed_time):
    capture_get["payload"] = {"cardData": {"license": "weird-license-1.0"}}
    score, _ = await license_metric.compute("https://huggingface.co/a/b", None, None)
    assert score == 0.5

@pytest.mark.asyncio
async def test_license_missing_returns_zero(capture_get, fixed_time):
    # No cardData.license at all
    capture_get["payload"] = {"cardData": {}}
    score, _ = await license_metric.compute("https://huggingface.co/a/b", None, None)
    assert score == 0.0

@pytest.mark.asyncio
async def test_empty_license_string_returns_zero(capture_get, fixed_time):
    capture_get["payload"] = {"cardData": {"license": ""}}
    score, _ = await license_metric.compute("https://huggingface.co/a/b", None, None)
    assert score == 0.0

@pytest.mark.asyncio
async def test_http_error_returns_error_value(capture_get, fixed_time):
    capture_get["status"] = 404  # raise_for_status will trigger
    score, ms = await license_metric.compute("https://huggingface.co/a/b", None, None)
    assert score == license_metric.ERROR_VALUE
    assert isinstance(ms, int) and ms >= 0

@pytest.mark.asyncio
async def test_case_insensitive_license_handling(capture_get, fixed_time):
    # Uppercase should still map to permissive
    capture_get["payload"] = {"cardData": {"license": "Apache-2.0"}}
    score, _ = await license_metric.compute("https://huggingface.co/a/b", None, None)
    assert score == 1.0
