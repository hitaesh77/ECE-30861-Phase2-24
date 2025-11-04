# tests/unit/metrics/test_performance_claims.py
import sys
import types
import re
from pathlib import Path
import pytest

import performance_claims as pc  # adjust if your path differs

# ---------- helpers to stub git and control temp dir / cleanup ----------

def install_git_stub(clone_impl):
    """
    Ensure a 'git' module exists and its Repo.clone_from uses our implementation.
    clone_impl(url:str, path:Path, depth:int) -> any
    """
    git_stub = sys.modules.get("git") or types.ModuleType("git")

    class Repo:
        @staticmethod
        def clone_from(url, path, depth=30):
            return clone_impl(url, Path(path), depth)

    git_stub.Repo = Repo
    sys.modules["git"] = git_stub

@pytest.fixture
def patch_tmpdir(monkeypatch, tmp_path):
    """
    Route tempfile.mkdtemp to a stable path; intercept shutil.rmtree calls to assert cleanup.
    """
    fixed = tmp_path / "perf_claims_tmp"
    fixed.mkdir(parents=True, exist_ok=True)

    # mkdtemp -> our fixed path
    monkeypatch.setattr(pc.tempfile, "mkdtemp", lambda prefix="perf_claims_": str(fixed))

    calls = {"args": None, "kwargs": None}
    def fake_rmtree(path, ignore_errors=False):
        calls["args"] = (path,)
        calls["kwargs"] = {"ignore_errors": ignore_errors}
    monkeypatch.setattr(pc.shutil, "rmtree", fake_rmtree)

    return fixed, calls

@pytest.fixture
def fixed_time(monkeypatch):
    # Keep perf_counter monotonic but deterministic
    t = {"p": 1000.0}
    def perf():
        t["p"] += 0.001
        return t["p"]
    monkeypatch.setattr(pc.time, "perf_counter", perf)
    return t

# --------------------------- tests ------------------------------------

@pytest.mark.asyncio
async def test_early_return_no_valid_code_url(fixed_time):
    score, ms = await pc.compute("hf://org/model", code_url=None, dataset_url=None)
    assert score == 0.0 and isinstance(ms, int) and ms >= 0

    score2, ms2 = await pc.compute("hf://org/model", code_url="https://example.com/not-github", dataset_url=None)
    assert score2 == 0.0 and ms2 >= 0

@pytest.mark.asyncio
async def test_happy_path_some_keywords(monkeypatch, patch_tmpdir, fixed_time):
    tmp_root, rm_calls = patch_tmpdir

    # We'll capture exactly what we write, to replicate the metric's regex computation.
    readme_text = (
        "# Project\n"
        "This achieves state-of-the-art accuracy on GLUE.\n"
        "Also reports F1 and recall.\n"
    )
    notes_text = "Benchmarks: SQuAD and CIFAR discussed here; eval pipeline described."

    def clone_impl(url, path: Path, depth):
        assert "github.com" in url and depth == 30
        path.mkdir(parents=True, exist_ok=True)
        (path / "README.md").write_text(readme_text, encoding="utf-8")
        (path / "docs").mkdir(exist_ok=True)
        (path / "docs" / "notes.txt").write_text(notes_text, encoding="utf-8")
        return object()

    install_git_stub(clone_impl)

    # Run metric
    score, ms = await pc.compute("hf://org/model", code_url="https://github.com/org/repo", dataset_url=None)
    assert isinstance(ms, int) and ms >= 0

    # Compute expected using the SAME regex logic as the implementation on the SAME text
    text = readme_text + "\n" + notes_text
    matches = sum(
        1
        for kw in pc.BENCHMARK_KEYWORDS
        if re.search(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE)
    )
    expected = round(min(matches / len(pc.BENCHMARK_KEYWORDS), 1.0), 2)
    assert score == expected

    # Cleanup happened
    assert rm_calls["args"] == (str(tmp_root),)
    assert rm_calls["kwargs"].get("ignore_errors") is True

@pytest.mark.asyncio
async def test_case_insensitive_and_hyphen_handling(monkeypatch, patch_tmpdir, fixed_time):
    tmp_root, _ = patch_tmpdir

    def clone_impl(url, path: Path, depth):
        path.mkdir(parents=True, exist_ok=True)
        # Uppercase "SOTA" and hyphenated "mt-bench"
        (path / "readme.md").write_text("SOTA results on MT-BENCH.\n", encoding="utf-8")
        return object()

    install_git_stub(clone_impl)

    score, _ = await pc.compute("hf://org/model", code_url="https://github.com/org/repo", dataset_url=None)

    # At least 'sota' and 'mt-bench' should match
    min_hits = 2 / len(pc.BENCHMARK_KEYWORDS)
    assert score >= round(min_hits, 2)

@pytest.mark.asyncio
async def test_zero_hits_results_in_zero_score(monkeypatch, patch_tmpdir, fixed_time):
    tmp_root, _ = patch_tmpdir

    def clone_impl(url, path: Path, depth):
        path.mkdir(parents=True, exist_ok=True)
        (path / "README.md").write_text("No performance claims here.\nJust prose.", encoding="utf-8")
        (path / "docs").mkdir(exist_ok=True)
        (path / "docs" / "guide.txt").write_text("Usage only; no benchmarks mentioned.", encoding="utf-8")
        return object()

    install_git_stub(clone_impl)

    score, _ = await pc.compute("hf://org/model", code_url="https://github.com/org/repo", dataset_url=None)
    assert score == 0.0

@pytest.mark.asyncio
async def test_clone_exception_logs_and_returns_zero(monkeypatch, patch_tmpdir, fixed_time, caplog):
    tmp_root, rm_calls = patch_tmpdir

    def clone_impl(url, path: Path, depth):
        raise RuntimeError("network error")

    install_git_stub(clone_impl)

    with caplog.at_level("ERROR"):
        score, ms = await pc.compute("hf://org/model", code_url="https://github.com/org/repo", dataset_url=None)

    assert score == 0.0 and isinstance(ms, int) and ms >= 0
    assert any("Error analyzing performance_claims" in rec.message for rec in caplog.records)
    assert rm_calls["args"] == (str(tmp_root),)
