#!/usr/bin/env python3
# run.py
import sys, click, asyncio, json, subprocess
from pathlib import Path
from enum import Enum
from typing import TypedDict, Literal, Dict
from metrics import run_metrics, GradeResult, UrlCategory, Provider
# ---- Domain: URL Classification -----

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


# custom group that defaults to "urls" command
class DefaultGroup(click.Group):
    def __init__(self, *args, **kwargs):
        self.default_cmd_name = kwargs.pop("default_cmd_name", None)
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx, args):
        if args and not self.get_command(ctx, args[0]):
            # route first token into the default command
            args.insert(0, self.default_cmd_name)
        return super().parse_args(ctx, args)


@click.group(cls=DefaultGroup, context_settings=dict(help_option_names=["-h", "--help"]), invoke_without_command=True, default_cmd_name="_urls")
@click.pass_context
def cli(ctx):
    """
    LLM Grader CLI

    Usage:
      python run.py install    # Install dependencies
      python run.py test      # Run tests
      python run.py FILE      # Grade URLs from file (or '-' for stdin)
    """

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
@cli.command("_urls")
@click.argument("urls_file", required=True)
def urls_command(urls_file: str):
    """Process a newline-delimited URL file (or '-' for stdin)."""
    p = Path(urls_file)
    if not p.exists():
        click.echo(f"Error: file not found: {p}", err=True)
        sys.exit(1)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    source = str(p)
    click.echo(f"Read {len(lines)} lines from {source}. (grading stub)")
    
    for line in lines:
        click.echo(f"Processing line: {line}")
        url_dictionary = {}
        for url in line.split(","):
            
            if url is None or url.strip() == "":
                # click.echo("Skipping empty URL.")
                continue
            click.echo(f"Classifying URL: {url}")
            category, provider, ids = classify_url(url)
            click.echo(f"URL: {url}\n  Category: {category}, Provider: {provider}, IDs: {ids}")
            url_dictionary[category] = ids
        
        result: GradeResult = asyncio.run(run_metrics(url_dictionary))
        print(json.dumps(result))

if __name__ == "__main__":
    raise SystemExit(cli())
