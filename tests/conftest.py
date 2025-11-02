# tests/conftest.py
import os
import pytest
from freezegun import freeze_time

def pytest_addoption(parser):
    parser.addoption(
        "--offline",
        action="store_true",
        default=False,
        help="Disable network access in tests and use offline fixtures.",
    )

@pytest.fixture(scope="session")
def offline(request):
    # Also respect environment variable OFFLINE_MODE=1
    env_flag = os.environ.get("OFFLINE_MODE", "")
    cmd_flag = request.config.getoption("--offline")
    return cmd_flag or env_flag in {"1", "true", "True"}

@pytest.fixture
def sample_md():
    # A tiny README with signals for quick ramp-up (uses indented code block to avoid backticks)
    return """# Model Title

## Quickstart
    from transformers import pipeline
    pipe = pipeline("text-generation", model="org/model")

Install:
pip install transformers
"""

@pytest.fixture
def sample_api_json():
    return {
        "pipeline_tag": "text-generation",
        "cardData": {"language": "en"},
        "tags": ["transformers", "pytorch"],
    }

@pytest.fixture
def frozen_time():
    with freeze_time("2025-11-01 10:00:00"):
        yield

@pytest.fixture
def env_log(tmp_path, monkeypatch):
    # Centralize logging env so tests don't write to project root
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "app.log"))
    yield

@pytest.fixture(autouse=True)
def maybe_disable_network(offline, monkeypatch):
    """If --offline (or OFFLINE_MODE=1) is set, block all outgoing HTTP(S).
    This keeps unit tests deterministic and fast by default.
    Integration tests can override by not using --offline.
    """
    if not offline:
        return

    # Monkeypatch the low-level request method used by requests
    import requests.sessions

    def _blocked(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("Network disabled in --offline mode (use without --offline to enable).")

    monkeypatch.setattr(requests.sessions.Session, "request", _blocked, raising=True)
