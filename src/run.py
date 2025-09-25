#!/usr/bin/env python3
# run.py
import sys
from pathlib import Path
import click
from enum import Enum
from typing import TypedDict, Literal, Dict
import subprocess
import asyncio

# ---- Domain: URL Classification -----


class UrlCategory(str, Enum):
    MODEL = "MODEL"
    DATASET = "DATASET"
    CODE = "CODE"


# Optional: if you want to branch logic later


class Provider(str, Enum):
    HUGGINGFACE = "huggingface"
    GITHUB = "github"
    OTHER = "other"


# ---- Domain: NDJSON output schema for MODEL lines ----


class SizeScore(TypedDict):
    raspberry_pi: float
    jetson_nano: float
    desktop_pc: float
    aws_server: float


class GradeResult(TypedDict):
    name: str
    category: Literal["MODEL"]
    net_score: float
    net_score_latency: int
    ramp_up_time: float
    ramp_up_time_latency: int
    bus_factor: float
    bus_factor_latency: int
    performance_claims: float
    performance_claims_latency: int
    license: float
    license_latency: int
    size_score: SizeScore
    size_score_latency: int
    dataset_and_code_score: float
    dataset_and_code_score_latency: int
    dataset_quality: float
    dataset_quality_latency: int
    code_quality: float
    code_quality_latency: int


# ---- Ingest: URL parsing & classification (stub) ----
def classify_url(raw: str) -> tuple[UrlCategory, Provider, Dict[str, str]]:
    """Return (category, provider, ids) for a URL string. Improved dataset detection."""
    s = raw.strip()
    if "huggingface.co" in s:
        if "/datasets/" in s or s.rstrip("/").endswith("/datasets"):
            # Hugging Face dataset URL
            return UrlCategory.DATASET, Provider.HUGGINGFACE, {"url": s}
        else:
            # Hugging Face model URL (default)
            return UrlCategory.MODEL, Provider.HUGGINGFACE, {"url": s}
    if "github.com" in s:
        return UrlCategory.CODE, Provider.GITHUB, {"url": s}
    return UrlCategory.OTHER, Provider.OTHER, {"url": s}


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
def cli(ctx):
    """
    LLM Grader CLI

    Usage:
      python run.py install    # Install dependencies
      python run.py test      # Run tests
      python run.py FILE      # Grade URLs from file (or '-' for stdin)
    """
    if ctx.invoked_subcommand is None:
        if not ctx.args:
            click.echo(ctx.get_help())
            ctx.exit(1)
        # treat the first leftover token as a filename
        urls_file = ctx.args[0]
        ctx.invoke(urls_command, urls_file=urls_file)


@cli.command(short_help="Run tests and print required summary line.")
@click.option("--min-coverage", type=int, default=80, show_default=True, help="Minimum coverage to pass.")
def test(min_coverage: int):
    # Minimal placeholder
    passed = True
    coverage = 100
    click.echo(f"X/Y test cases passed. {coverage}% line coverage achieved.")
    sys.exit(0 if (passed and coverage >= min_coverage) else 1)


@cli.command(short_help="Install project/runtime dependencies.")
def install():
    """Install all project dependencies from pyproject.toml."""
    try:
        # Change to root directory
        root_dir = Path(__file__).parent.parent
        
        # Update pip first
        click.echo("Updating pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install dependencies from project root
        click.echo("Installing dependencies...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=str(root_dir)
        )
        click.echo("Installation completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        click.echo(f"Error installing dependencies: {e}", err=True)
        return 1


# Change the urls command name to make it private
@cli.command("_urls", hidden=True)
@click.argument("urls_file", required=True)
def urls_command(urls_file: str):
    """Process a newline-delimited URL file (or '-' for stdin)."""
    p = Path(urls_file)
    if not p.exists():
        click.echo(f"Error: file not found: {p}", err=True)
        sys.exit(1)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    source = str(p)
    click.echo(f"Read {len(lines)} URL(s) from {source}. (grading stub)")


if __name__ == "__main__":
    raise SystemExit(cli())
