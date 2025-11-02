import sys
import types
import pytest

# Import the module from src/
import code_quality

# ---------- Helpers for stubbing ----------
def make_commit(ts, email="a@ex.com"):
    author = types.SimpleNamespace(email=email)
    return types.SimpleNamespace(committed_date=ts, author=author)

class MockRepo:
    def __init__(self, commits):
        self._commits = commits
    def iter_commits(self):
        return list(self._commits)

def install_git_stub(repo_obj_factory):
    """
    Provide a 'git' module with a Repo.clone_from that returns a mock repo.
    """
    git_stub = sys.modules.get("git") or types.ModuleType("git")
    class DummyRepo:
        @staticmethod
        def clone_from(url, path, depth=50):
            return repo_obj_factory(url, path, depth)
    git_stub.Repo = DummyRepo
    sys.modules["git"] = git_stub   # inject/overwrite

# ---------- Time & tmp fixtures ----------
@pytest.fixture
def fixed_time(monkeypatch):
    now = 1_700_000_000  # arbitrary epoch seconds
    # code_quality uses time.time and time.perf_counter
    monkeypatch.setattr(code_quality.time, "time", lambda: now)
    counter = {"t": 1000.0}
    monkeypatch.setattr(code_quality.time, "perf_counter", lambda: (counter.__setitem__("t", counter["t"] + 0.001) or counter["t"]))
    return now

@pytest.fixture
def tmpdir_fixed(monkeypatch, tmp_path):
    fixed_root = tmp_path / "cq_tmp"
    fixed_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(code_quality.tempfile, "mkdtemp", lambda prefix="code_quality_": str(fixed_root))
    calls = {"args": None, "kwargs": None}
    def fake_rmtree(path, ignore_errors=False):
        calls["args"] = (path,)
        calls["kwargs"] = {"ignore_errors": ignore_errors}
    monkeypatch.setattr(code_quality.shutil, "rmtree", fake_rmtree)
    return fixed_root, calls

# ---------- Tests ----------

@pytest.mark.asyncio
async def test_no_valid_code_url_returns_zero(fixed_time):
    score, latency = await code_quality.compute("hf://org/model", None, None)
    assert score == 0.0 and isinstance(latency, int) and latency >= 0

    score2, latency2 = await code_quality.compute("hf://org/model", "https://example.com/not-github", None)
    assert score2 == 0.0 and latency2 >= 0

@pytest.mark.asyncio
async def test_happy_path_recent_fresh_commits(monkeypatch, fixed_time, tmpdir_fixed):
    fixed_root, rm_calls = tmpdir_fixed
    now = fixed_time

    # 150 commits, 7 authors, last commit 10 days ago -> all three sub-scores hit their caps (1.0)
    commits = [make_commit(now - 10*86400 - i, email=f"user{i%7}@ex.com") for i in range(150)]
    repo_obj = MockRepo(commits)

    # Install a git stub that returns our mock repo
    def factory(url, path, depth):
        assert "github.com" in url and depth == 50
        return repo_obj
    install_git_stub(factory)

    score, latency = await code_quality.compute("hf://org/model", "https://github.com/org/repo", None)
    assert score == 1.0
    assert latency >= 0

    # Ensure cleanup attempted
    assert rm_calls["args"] == (str(fixed_root),)
    assert rm_calls["kwargs"].get("ignore_errors") is True

@pytest.mark.asyncio
async def test_mid_and_old_freshness(monkeypatch, fixed_time, tmpdir_fixed):
    now = fixed_time

    # mid: 60 commits (0.6), 3 authors (0.6), last commit 100 days ago (0.5)
    repo_mid = MockRepo([make_commit(now - 100*86400, email=f"a{i%3}@ex.com") for i in range(60)])
    install_git_stub(lambda u, p, d: repo_mid)
    score_mid, _ = await code_quality.compute("hf://org/model", "https://github.com/org/repo", None)
    assert score_mid == round((0.6 + 0.6 + 0.5) / 3, 2)

    # old: 20 commits (0.2), 4 authors (0.8), last commit 400 days ago (0.0)
    repo_old = MockRepo([make_commit(now - 400*86400, email=f"a{i%4}@ex.com") for i in range(20)])
    install_git_stub(lambda u, p, d: repo_old)
    score_old, _ = await code_quality.compute("hf://org/model", "https://github.com/org/repo", None)
    assert score_old == round((0.2 + 0.8 + 0.0) / 3, 2)

@pytest.mark.asyncio
async def test_clone_exception_returns_zero(monkeypatch, fixed_time, tmpdir_fixed, caplog):
    fixed_root, rm_calls = tmpdir_fixed

    # Make the stub raise on clone
    def factory_raises(url, path, depth):
        raise RuntimeError("network issue")
    install_git_stub(factory_raises)

    with caplog.at_level("ERROR"):
        score, latency = await code_quality.compute("hf://org/model", "https://github.com/org/repo", None)

    assert score == 0.0 and latency >= 0
    assert any("Error analyzing code repo" in rec.message for rec in caplog.records)
    assert rm_calls["args"] == (str(fixed_root),)
