# tests/unit/metrics/test_dataset_quality.py
import sys
import types
import pytest

import dataset_quality as dq  # from src/dataset_quality.py

# ---------- Helpers ----------

def make_info(has_desc: bool, has_lic: bool, files_count: int):
    """Mimic the object returned by HfApi.dataset_info()."""
    card_data = {}
    if has_desc:
        card_data["description"] = "desc"
    if has_lic:
        card_data["license"] = "mit"
    siblings = [types.SimpleNamespace(path="file.txt")] * files_count
    return types.SimpleNamespace(card_data=card_data, siblings=siblings)

# Replace (or insert) a stub huggingface_hub module for all tests
@pytest.fixture(autouse=True)
def stub_hf_module(monkeypatch):
    class _DefaultHfApi:
        def dataset_info(self, repo_id: str):
            # default: empty info (0/3 signals)
            return make_info(False, False, 0)
    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = _DefaultHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)

# Make time.time deterministic (we only assert latency is a non-negative int)
@pytest.fixture
def fixed_time(monkeypatch):
    t = {"now": 1_700_000_000.0}
    def now():
        t["now"] += 0.001
        return t["now"]
    # Patch stdlib so it works regardless of how module imports time
    monkeypatch.setattr("time.time", now)
    return t

# ---------- Early returns (expect tuple everywhere) ----------

@pytest.mark.asyncio
async def test_early_no_dataset_url_returns_zero_tuple(fixed_time):
    score, ms = await dq.compute("hf://org/model", None, None)
    assert score == 0.0
    assert isinstance(ms, int) and ms >= 0

@pytest.mark.asyncio
async def test_early_non_hf_url_returns_zero_tuple(fixed_time):
    score, ms = await dq.compute("hf://org/model", None, "https://example.com/datasets/owner/name")
    assert score == 0.0 and ms >= 0

@pytest.mark.asyncio
async def test_early_short_path_returns_zero_tuple(fixed_time):
    # len(parts) < 5
    score, ms = await dq.compute("hf://org/model", None, "https://huggingface.co/datasets/owner")
    assert score == 0.0 and ms >= 0

# ---------- Exception path ----------

@pytest.mark.asyncio
async def test_hf_api_exception_returns_zero_tuple(monkeypatch, fixed_time):
    import huggingface_hub
    class RaisingApi:
        def dataset_info(self, repo_id: str):
            raise RuntimeError("hf error")
    monkeypatch.setattr(huggingface_hub, "HfApi", RaisingApi, raising=False)

    score, ms = await dq.compute("hf://org/model", None, "https://huggingface.co/datasets/owner/name")
    assert score == 0.0 and ms >= 0

# ---------- Happy path scoring ----------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "has_desc, has_lic, files, expected",
    [
        (False, False, 0, 0.00),
        (True,  False, 0, 0.33),
        (False, True,  0, 0.33),
        (False, False, 1, 0.33),
        (True,  True,  0, 0.67),
        (True,  False, 1, 0.67),
        (False, True,  1, 0.67),
        (True,  True,  2, 1.00),
    ],
)
async def test_scoring_matrix(monkeypatch, fixed_time, has_desc, has_lic, files, expected):
    import huggingface_hub
    class HfApiStub:
        def dataset_info(self, repo_id: str):
            return make_info(has_desc, has_lic, files)
    monkeypatch.setattr(huggingface_hub, "HfApi", HfApiStub, raising=False)

    score, ms = await dq.compute("hf://org/model", None, "https://huggingface.co/datasets/owner/name")
    assert score == expected
    assert isinstance(ms, int) and ms >= 0

@pytest.mark.asyncio
async def test_owner_name_parsing_and_repo_id_used(monkeypatch, fixed_time):
    import huggingface_hub
    captured = {"repo_id": None}
    class HfApiSpy:
        def dataset_info(self, repo_id: str):
            captured["repo_id"] = repo_id
            return make_info(True, True, 1)
    monkeypatch.setattr(huggingface_hub, "HfApi", HfApiSpy, raising=False)

    score, ms = await dq.compute("hf://org/model", None, "https://huggingface.co/datasets/owner/name")
    assert captured["repo_id"] == "owner/name"
    assert score == 1.00 and ms >= 0
