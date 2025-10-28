#!/usr/bin/env python3
# run.py
import coverage
import logging
import sys, click, asyncio, json, subprocess, logging
from pathlib import Path
from enum import Enum
from typing import TypedDict, Literal, Dict, Tuple
from src.metrics import run_metrics, GradeResult, UrlCategory, Provider
from test import setup_logger
# ---- Domain: URL Classification -----

logger = setup_logger()
cov = coverage.Coverage()

# ---- Ingest: URL parsing & classification (stub) ----
def classify_url(raw: str) -> Tuple[UrlCategory, Provider, Dict[str, str]]:
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

def read_enter_delimited_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
    return data

@cli.command(short_help="Run tests and print required summary line.")
@click.option("--min-coverage", type=int, default=80, show_default=True, help="Minimum coverage to pass.")
def test(min_coverage: int):
    cov.start()

    test_inputs = read_enter_delimited_file("test_inputs.txt")
    total = len(test_inputs)
    passed = 0

    for idx, input_str in enumerate(test_inputs, start=1):
        logger.info(f"Running Test {idx} with input: {input_str}")

        try:
            result = urls_command(input_str)

            logger.info(f"Test {idx} completed successfully.")
            logger.debug(f"Test {idx} output: {result}")

            passed += 1
        except Exception as e:
            logger.error(f"Test {idx} failed with input={input_str}, error={e}", exc_info=True)

    cov.stop()
    cov.save()

    # Generate a numeric report
    coverage_percent = cov.report(show_missing=False)

    print(f"{passed}/{total} test cases passed. {coverage_percent:.1f}% line coverage achieved")
    sys.exit((coverage_percent > .8 and passed/total > .8))


@cli.command(short_help="Install project/runtime dependencies.")
def install():
    """Install all project dependencies from pyproject.toml."""
    try:
        # Change to root directory
        root_dir = Path(__file__).parent.parent
        
        # Update pip first
        logging.info("Updating pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install dependencies from project root
        logging.info("Installing project dependencies...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=str(root_dir)
        )
        logging.info("Installation completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing dependencies: {e}")
        return 1


# Change the urls command name to make it private
@cli.command("_urls")
@click.argument("urls_file", required=True)
def urls_command(urls_file: str) -> dict:
    """Process a newline-delimited URL file (or '-' for stdin)."""
    p = Path(urls_file)
    if not p.exists():
        logging.error(f"Error: file not found: {p}")
        sys.exit(1)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    source = str(p)
    logging.info(f"Read {len(lines)} lines from {source}. (grading stub)")
    
    for line in lines:
        # click.echo(f"Processing line: {line}")
        url_dictionary = {}
        for url in line.split(","):
            
            if url is None or url.strip() == "":
                # click.echo("Skipping empty URL.")
                continue
            # click.echo(f"Classifying URL: {url}")
            category, provider, ids = classify_url(url)
            # click.echo(f"URL: {url}\n  Category: {category}, Provider: {provider}, IDs: {ids}")
            url_dictionary[category] = ids
        
        if url_dictionary.get(UrlCategory.MODEL) is None:
            logging.error("Error: No MODEL URL found, skipping line.")
            continue
        result: GradeResult = asyncio.run(run_metrics(url_dictionary))
        print(json.dumps(result))
    return result

if __name__ == "__main__":
    raise SystemExit(cli())
