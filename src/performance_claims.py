# metrics/performance_claims.py
import time
import logging
import re
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple

# A list of keywords/phrases we’ll scan for in repo text
BENCHMARK_KEYWORDS = [
    "accuracy", "precision", "recall", "f1", "f-1", "bleu", "rouge",
    "state-of-the-art", "sota", "outperform", "benchmark", "eval",
    "imagenet", "cifar", "glue", "squad", "msmarco", "wikitext", "mt-bench"
]

async def compute(model_url: str, code_url: Optional[str], dataset_url: Optional[str]) -> Tuple[float, int]:
    """
    Heuristic metric: detect performance/benchmark claims in a repo.
    Score = fraction of benchmark keywords found, capped at 1.0.
    
    Returns: (score ∈ [0,1], latency_ms)
    """

    import git  # GitPython

    start = time.perf_counter()

    if not code_url or "github.com" not in code_url:
        logging.warning("No valid code_url provided, defaulting performance_claims=0.0")
        return 0.0, (int)((time.perf_counter() - start) * 1000)

    tmpdir = tempfile.mkdtemp(prefix="perf_claims_")
    score = 0.0

    try:
        repo_path = Path(tmpdir) / "repo"
        logging.info(f"[performance_claims] Cloning {code_url} into {repo_path}")
        repo = git.Repo.clone_from(code_url, repo_path, depth=30)

        # Collect candidate text files (README, docs/, markdown, notebooks, etc.)
        candidates = []
        for pattern in ["README.md", "readme.md", "docs", "*.md", "*.rst", "*.txt"]:
            candidates.extend(repo_path.rglob(pattern))

        # Concatenate contents (limit for safety)
        all_text = ""
        for f in candidates:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                all_text += "\n" + text
            except Exception:
                continue

        # Count matches
        matches = 0
        for kw in BENCHMARK_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", all_text, flags=re.IGNORECASE):
                matches += 1

        # Normalize to [0,1] with simple heuristic
        if matches > 0:
            score = min(matches / len(BENCHMARK_KEYWORDS), 1.0)

    except Exception as e:
        logging.error(f"Error analyzing performance_claims in {code_url}: {e}")
        score = 0.0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    latency_ms = (int)((time.perf_counter() - start) * 1000)
    return round(score, 2), latency_ms
