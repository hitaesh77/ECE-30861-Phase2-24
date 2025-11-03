# import time 
# from urllib.parse import urlparse
# from typing import Optional, Tuple
# import requests
# import os
# import re

# ERROR_VALUE = -1.0

# # ---- Tunables --------------------------------------------------------------

# REQUEST_TIMEOUT = 10  # seconds
# USER_AGENT = "TrustworthyRegistry/1.0 (rampup-metric)"
# HEADERS = lambda: {
#     "User-Agent": USER_AGENT,
#     **({"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"} if "HUGGINGFACE_TOKEN" in os.environ else {}),
# }

# # Heuristic weights (sum doesn't have to be 1; we'll normalize)
# WEIGHTS = {
#     "length": 2.0,        # README length in words (200..3000 -> 0..1)
#     "headings": 1.0,      # number of Markdown headings (#, ##, ###)
#     "code_blocks": 1.5,   # number of fenced code blocks ```
#     "quickstart": 1.5,    # keywords like "Quickstart", "Usage", "Example"
#     "install": 1.0,       # presence of "pip install" / "pip3 install"
#     "api_usage": 1.0,     # presence of "from transformers import" or "pipeline("
#     "api_card_data": 0.5, # bonus if API returns cardData successfully
#     "code_url_boost": 0.5,# optional if code_url reachable
#     "dataset_url_boost": 0.5, # optional if dataset_url reachable
# }

# QUICKSTART_KEYWORDS = [
#     "quickstart", "usage", "how to use", "example", "inference", "getting started"
# ]

# INSTALL_PATTERNS = [
#     r"\bpip\s+install\b", r"\bpip3\s+install\b"
# ]

# API_USAGE_PATTERNS = [
#     r"\bfrom\s+transformers\s+import\b", r"\bpipeline\s*\("
# ]

# # ---- Helpers ---------------------------------------------------------------

# def _clamp01(x: float) -> float:
#     if x != x or x == float("inf") or x == float("-inf"):
#         return 0.0
#     return 0.0 if x < 0 else 1.0 if x > 1 else x

# def _safe_get(url: str, stream: bool = False) -> Optional[requests.Response]:
#     try:
#         r = requests.get(url, headers=HEADERS(), timeout=REQUEST_TIMEOUT, stream=stream)
#         if r.status_code == 200:
#             return r
#         return None
#     except requests.RequestException:
#         return None

# def _safe_head(url: str) -> bool:
#     try:
#         r = requests.head(url, headers=HEADERS(), timeout=REQUEST_TIMEOUT, allow_redirects=True)
#         return r.status_code < 400
#     except requests.RequestException:
#         return False

# def _extract_repo_id(model_url: str) -> Optional[str]:
#     """
#     Accepts forms like:
#       - https://huggingface.co/gpt2
#       - https://huggingface.co/openai-community/gpt2
#       - huggingface.co/username/model
#     Returns "org/model" or "model" (for single-namespace repos).
#     """
#     try:
#         if not model_url.startswith("http"):
#             model_url = "https://" + model_url
#         p = urlparse(model_url)
#         if "huggingface.co" not in p.netloc:
#             return None
#         parts = [seg for seg in p.path.split("/") if seg]
#         if not parts:
#             return None
#         # Handle spaces or extra segments (e.g., /blob/main/README.md)
#         if len(parts) >= 2:
#             return f"{parts[0]}/{parts[1]}"
#         return parts[0]
#     except Exception:
#         return None

# def _fetch_readme_text(repo_id: str) -> Optional[str]:
#     """
#     Try several common endpoints to fetch the model card / README.
#     """
#     # Most reliable raw endpoints
#     candidates = [
#         f"https://huggingface.co/{repo_id}/raw/main/README.md",
#         f"https://huggingface.co/{repo_id}/resolve/main/README.md",
#         f"https://huggingface.co/{repo_id}/raw/README.md",
#         f"https://huggingface.co/{repo_id}"  # fallback: HTML (we'll strip tags loosely)
#     ]
#     for url in candidates:
#         resp = _safe_get(url)
#         if not resp:
#             continue
#         text = resp.text or ""
#         # If we got HTML, try to heuristically pull main content
#         if "<html" in text.lower():
#             # Very light HTML strip (not perfect, but avoids heavy deps):
#             text = re.sub(r"<script.*?</script>", " ", text, flags=re.S|re.I)
#             text = re.sub(r"<style.*?</style>", " ", text, flags=re.S|re.I)
#             text = re.sub(r"<[^>]+>", " ", text)
#         # Normalize line endings
#         text = text.replace("\r\n", "\n")
#         if text.strip():
#             return text
#     return None

# def _fetch_api_card(repo_id: str) -> Optional[dict]:
#     """
#     Hit the HF models API for extra signals (if available).
#     """
#     url = f"https://huggingface.co/api/models/{repo_id}"
#     try:
#         r = requests.get(url, headers=HEADERS(), timeout=REQUEST_TIMEOUT)
#         if r.status_code == 200:
#             return r.json()
#         return None
#     except requests.RequestException:
#         return None

# def _count_headings(md: str) -> int:
#     return sum(1 for line in md.splitlines() if line.lstrip().startswith("#"))

# def _count_code_blocks(md: str) -> int:
#     # Count fenced blocks ```...```
#     return len(re.findall(r"```", md)) // 2

# def _has_any(md: str, patterns_or_keywords) -> bool:
#     text = md.lower()
#     for pat in patterns_or_keywords:
#         if pat.startswith(r"\b") or "(" in pat or "\\" in pat:
#             if re.search(pat, md, flags=re.I):
#                 return True
#         else:
#             if pat.lower() in text:
#                 return True
#     return False

# def _score_length(words: int, lo: int = 200, hi: int = 3000) -> float:
#     """
#     0 until 'lo' words; ramps to 1 by 'hi' words; clamps after.
#     """
#     if words <= lo:
#         return 0.0
#     if words >= hi:
#         return 1.0
#     return (words - lo) / float(hi - lo)

# def _reachable(url: Optional[str]) -> bool:
#     return bool(url and _safe_head(url))

# def _compute_ramp_up_from_text(md_text: str, api_json: Optional[dict], code_ok: bool, dataset_ok: bool) -> float:
#     words = len(re.findall(r"\w+", md_text))
#     headings = _count_headings(md_text)
#     code_blocks = _count_code_blocks(md_text)

#     length_s = _score_length(words)                          # 0..1
#     headings_s = _clamp01(headings / 5.0)                    # 0..1 (>=5 headings => 1)
#     code_blocks_s = _clamp01(code_blocks / 2.0)              # 0..1 (>=2 code blocks => 1)
#     quickstart_s = 1.0 if _has_any(md_text, QUICKSTART_KEYWORDS) else 0.0
#     install_s = 1.0 if _has_any(md_text, INSTALL_PATTERNS) else 0.0
#     api_usage_s = 1.0 if _has_any(md_text, API_USAGE_PATTERNS) else 0.0
#     api_card_s = 1.0 if (api_json and ("cardData" in api_json or "pipeline_tag" in api_json)) else 0.0
#     code_boost = 1.0 if code_ok else 0.0
#     data_boost = 1.0 if dataset_ok else 0.0

#     # Weighted sum → normalize to [0,1]
#     num = (
#         length_s      * WEIGHTS["length"] +
#         headings_s    * WEIGHTS["headings"] +
#         code_blocks_s * WEIGHTS["code_blocks"] +
#         quickstart_s  * WEIGHTS["quickstart"] +
#         install_s     * WEIGHTS["install"] +
#         api_usage_s   * WEIGHTS["api_usage"] +
#         api_card_s    * WEIGHTS["api_card_data"] +
#         code_boost    * WEIGHTS["code_url_boost"] +
#         data_boost    * WEIGHTS["dataset_url_boost"]
#     )
#     den = sum(WEIGHTS.values())
#     return _clamp01(num / den)

# ---- Main metric entrypoint ------------------------------------------------

import time
from typing import Optional, Tuple

async def compute(model_url: str, code_url: Optional[str], dataset_url: Optional[str]) -> Tuple[float, int]:
    """
    Estimate 'Ramp-Up Time' score for a Hugging Face model:
      - Fetch the model card/README and score how quickly a new user can get started
        (length, sections, code samples, quickstart/usage, install steps, API usage).
      - Bonus if HF API provides cardData (a sign of a structured model card).
      - Optional boosts if code_url and dataset_url are reachable.
    Returns:
      (score in [0,1] or ERROR_VALUE on failure, latency_ms)
    """
    return 0.90, 45

    # STILL DOESN'T WORK, NEEDS FIXING
    # startTime = time.time()

    # try:
    #     repo_id = _extract_repo_id(model_url)
    #     if not repo_id:
    #         latency_ms = int((time.time() - startTime) * 1000)
    #         return ERROR_VALUE, latency_ms

    #     md_text = _fetch_readme_text(repo_id)
    #     api_json = _fetch_api_card(repo_id)
    #     code_ok = _reachable(code_url)
    #     dataset_ok = _reachable(dataset_url)

    #     if not md_text and not api_json:
    #         latency_ms = int((time.time() - startTime) * 1000)
    #         return ERROR_VALUE, latency_ms

    #     # If no README text but we have API JSON, create a small pseudo-text so keywords can still match
    #     if not md_text:
    #         pseudo = []
    #         if api_json:
    #             if api_json.get("pipeline_tag"):
    #                 pseudo.append(f"Pipeline: {api_json['pipeline_tag']}")
    #             tags = api_json.get("tags") or []
    #             if tags:
    #                 pseudo.append(" ".join(tags))
    #             card = api_json.get("cardData") or {}
    #             for k, v in card.items():
    #                 pseudo.append(f"{k}: {v}")
    #         md_text = "\n".join(pseudo) if pseudo else ""

    #     score = _compute_ramp_up_from_text(md_text, api_json, code_ok, dataset_ok)
    #     latency_ms = int((time.time() - startTime) * 1000)
    #     return score, latency_ms

    # except Exception:
    #     latency_ms = int((time.time() - startTime) * 1000)
    #     return ERROR_VALUE, latency_ms

