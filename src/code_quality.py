# metrics/code_quality.py
import tempfile
import time
import shutil
from pathlib import Path
from typing import Tuple
import logging
import git  # from GitPython


def compute(model_url: str, code_url: str | None, dataset_url: str | None) -> Tuple[float, float]:
    """
    Compute code quality metric based on GitHub repo metadata.
    Returns: (score [0-1], latency_ms)
    """
    start = time.perf_counter()

    if not code_url or "github.com" not in code_url:
        logging.warning("No valid code_url provided, defaulting code_quality=0.0")
        return 0.0, (time.perf_counter() - start) * 1000

    tmpdir = tempfile.mkdtemp(prefix="code_quality_")
    score = 0.0

    try:
        # Shallow clone for speed
        repo_path = Path(tmpdir) / "repo"
        logging.info(f"Cloning {code_url} into {repo_path}")
        repo = git.Repo.clone_from(code_url, repo_path, depth=50)

        commits = list(repo.iter_commits())
        num_commits = len(commits)
        authors = {c.author.email for c in commits}
        num_authors = len(authors)

        # Commit recency
        last_commit = commits[0].committed_date if commits else 0
        age_days = (time.time() - last_commit) / 86400 if last_commit else 9999

        # Naive normalization heuristics
        commit_score = min(num_commits / 100.0, 1.0)        # cap at 100 commits
        author_score = min(num_authors / 5.0, 1.0)          # cap at 5 contributors
        freshness_score = 1.0 if age_days < 30 else 0.5 if age_days < 365 else 0.0

        score = round((commit_score + author_score + freshness_score) / 3, 2)

    except Exception as e:
        logging.error(f"Error analyzing code repo {code_url}: {e}")
        score = 0.0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    latency_ms = (time.perf_counter() - start) * 1000
    return score, latency_ms
