import requests
import time
import re

def compute_reviewedness(code_url: str):
    """
    Compute the Reviewedness score for a GitHub repository.
    Reviewedness = (# of commits merged via reviewed PRs) / (total commits)
    If not a GitHub URL, return -1.
    """
    start_time = time.time()

    # Check if this is a GitHub repo
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+)", code_url)
    if not match:
        return -1, time.time() - start_time

    owner, repo = match.groups()

    try:
        # Get total commits
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=1"
        commits_resp = requests.get(commits_url)
        if "Link" in commits_resp.headers:
            # GitHub pagination: extract last page number to estimate total commits
            link = commits_resp.headers["Link"]
            last_page = int(re.search(r'page=(\d+)>; rel="last"', link).group(1))
        else:
            last_page = len(commits_resp.json())

        total_commits = last_page

        # Get all merged PRs
        prs_url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=closed&per_page=100"
        merged_count = 0
        page = 1
        while True:
            prs_resp = requests.get(prs_url + f"&page={page}")
            prs = prs_resp.json()
            if not prs or not isinstance(prs, list):
                break
            for pr in prs:
                if pr.get("merged_at"):
                    merged_count += 1
            if len(prs) < 100:
                break
            page += 1

        # Reviewedness: fraction of commits from merged PRs
        score = merged_count / total_commits if total_commits > 0 else 0

    except Exception as e:
        print(f"Error: {e}")
        score = -1

    latency_ms = (time.time() - start_time) * 1000
    return score, latency_ms


# ------------------ Test run ------------------
if __name__ == "__main__":
    print("TEST 1")
    code_url = "https://github.com/google-research/bert"
    dataset_url = "https://huggingface.co/datasets/bookcorpus/bookcorpus"
    model_url = "https://huggingface.co/google-bert/bert-base-uncased"
    score, latency = compute_reviewedness(code_url)
    print(f"Reviewedness score: {score}")
    print(f"Computation time: {latency:.2f} ms")

    print("\nTEST 2")
    code_url    = "https://huggingface.co/chiedo/hello-world"  
    dataset_url = "https://huggingface.co/datasets/chiedo/hello-world"  
    model_url   = "https://huggingface.co/chiedo/hello-world"
    score, latency = compute_reviewedness(code_url)
    print(f"Reviewedness score: {score}")
    print(f"Computation time: {latency:.2f} ms")

