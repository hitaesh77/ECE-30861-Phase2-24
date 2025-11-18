import logging
import sys
import asyncio
import json
import os
import subprocess
from pathlib import Path
from enum import Enum
from typing import Dict, Tuple
from utils import UrlCategory, Provider


def setup_logger():
    log_file = os.getenv("LOG_FILE")#, "llm_logs.log")
    log_level = int(os.getenv("LOG_LEVEL", "1"))  # default to INFO

    if log_file == None:
        print(f"Error: Invalid log file path")
        sys.exit(1)

    if log_level == 0:
        level = logging.disable(logging.CRITICAL + 1)  # silence
    elif log_level == 1:
        level = logging.INFO
    elif log_level == 2:
        level = logging.DEBUG
    else:
        level = logging.INFO

    try:
        logging.basicConfig(
            filename=log_file,
            filemode="w",  # overwrite for each run
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    except Exception:
        print(f"Error: Invalid log file path '{log_file}'")
        sys.exit(1)

    return logging.getLogger("testbench")

# --- Domain: URL Classification ---

# logger = setup_logger()

# --- Ingest: URL parsing & classification ---
def classify_url(raw: str) -> Tuple[UrlCategory, Provider, Dict[str, str]]:
    from metrics import UrlCategory, Provider
    
    """Return (category, provider, ids) for a URL string. Improved dataset detection."""

    s = raw.strip()
    if not s:
        return UrlCategory.OTHER, Provider.OTHER, {"url": ""}
    s_lower = s.lower()
    
    if "huggingface.co" in s_lower:
        if "/datasets/" in s_lower or s.rstrip("/").endswith("/datasets"):
            # Hugging Face dataset URL
            return UrlCategory.DATASET, Provider.HUGGINGFACE, {"url": s}
        else:
            # Hugging Face model URL (default)
            return UrlCategory.MODEL, Provider.HUGGINGFACE, {"url": s}
    if "github.com" in s_lower:
        return UrlCategory.CODE, Provider.GITHUB, {"url": s}
    return UrlCategory.OTHER, Provider.OTHER, {"url": s}

# --- Core Logic Functions ---

def read_enter_delimited_file(filename: str) -> list[str]:
    """Reads a file and returns a list of non-empty lines."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = [line.strip() for line in f if line.strip()]
        return data
    except FileNotFoundError:
        logger.error(f"Error: File not found: {filename}")
        # Re-raise the error to be caught by the caller for proper exit
        raise

# def urls_processor(urls_file: str) -> Dict:
#     """Process a newline-delimited URL file."""

#     # Note: run_metrics, GradeResult, UrlCategory, Provider are imported at the top now
#     from metrics import run_metrics, GradeResult, UrlCategory, Provider

#     p = Path(urls_file)
#     if not p.exists():
#         logger.error(f"Error: file not found: {p}")
#         sys.exit(1)
        
#     lines = read_enter_delimited_file(urls_file)
#     source = str(p)
#     logger.info(f"Read {len(lines)} lines from {source}. (grading stub)")

#     last_result = {}
    
#     for line in lines:
#         url_dictionary = {}
#         # Allows for multiple comma-separated URLs on one line, treating them all as part of one "repo group"
#         for url in line.split(","):
#             url = url.strip()
#             if not url:
#                 continue
#             category, provider, ids = classify_url(url)
#             # Store the info for this group
#             url_dictionary[category] = ids
        
#         if not url_dictionary.get(UrlCategory.MODEL):
#             logger.error("Error: No MODEL URL found in line, skipping.")
#             continue
            
#         try:
#             result = asyncio.run(run_metrics(url_dictionary))
#             # Guaranteed NDJSON output: explicitly write to stdout with a newline
#             # sys.stdout.write(json.dumps(result) + '\n') 
#             sys.stdout.write(json.dumps(result, separators=(',', ':')) + '\n') 
#             last_result = result

#         except Exception as e:
#             logger.error(f"Error running metrics for line '{line}': {e}", exc_info=True)

#     return last_result

def urls_processor(urls_file: str) -> Dict:
    """Process a newline-delimited URL file."""

    # Note: run_metrics, GradeResult, UrlCategory, Provider are imported at the top now
    from metrics import run_metrics, GradeResult, UrlCategory, Provider

    p = Path(urls_file)
    if not p.exists():
        logger.error(f"Error: file not found: {p}")
        sys.exit(1)
        
    lines = read_enter_delimited_file(urls_file)
    source = str(p)
    logger.info(f"Read {len(lines)} lines from {source}.")

    all_results = []
    
    for line_num, line in enumerate(lines, 1):
        url_dictionary = {}
        
        # Skip empty lines
        if not line.strip():
            continue
        
        # Split by comma and process each URL
        url_parts = [u.strip() for u in line.split(",")]
        
        # Remove empty strings from the list
        url_parts = [u for u in url_parts if u]
        
        if not url_parts:
            logger.warning(f"Line {line_num} has no valid URLs, skipping.")
            continue
        
        # Process each URL in the line
        for url in url_parts:
            try:
                category, provider, ids = classify_url(url)
                # Store the info for this group
                url_dictionary[category] = ids
            except Exception as e:
                logger.error(f"Error classifying URL '{url}' on line {line_num}: {e}")
                continue
        
        # Skip if no MODEL URL found
        if not url_dictionary.get(UrlCategory.MODEL):
            logger.error(f"Error: No MODEL URL found on line {line_num}, skipping.")
            continue
            
        try:
            logger.info(f"Processing line {line_num}: {url_dictionary.get(UrlCategory.MODEL).get('url', 'N/A')}")
            result = asyncio.run(run_metrics(url_dictionary))
            
            # Write result as NDJSON to stdout
            sys.stdout.write(json.dumps(result, separators=(',', ':')) + '\n')
            sys.stdout.flush()  # Ensure immediate output
            
            all_results.append(result)

        except Exception as e:
            logger.error(f"Error running metrics for line {line_num} '{line}': {e}", exc_info=True)
            continue

    logger.info(f"Processed {len(all_results)} URLs successfully out of {len(lines)} total lines.")
    
    # Return the last result for compatibility (or return all_results if you need all)
    return all_results[-1] if all_results else {}

def run_test(min_coverage: int = 80) -> bool:
    import coverage

    """Runs tests, reports coverage, and exits with a status code."""
    try:
        test_inputs = read_enter_delimited_file("test_inputs.txt")
    except FileNotFoundError:
        logger.error("Error: test_inputs.txt not found.")
        print("0/0 test cases passed. 0.0% line coverage achieved")
        return False
    
    
    cov = coverage.Coverage(data_file=".coverage_run", auto_data=True)    
    cov.start()

    total = len(test_inputs)
    passed = 0

    for idx, input_str in enumerate(test_inputs, start=1):
        logger.info(f"Running Test {idx} with input: {input_str}")
        
        try:
            # urls_processor expects a file path, but in the original test command 
            # `urls_command` was called directly with the input string.
            # This suggests a change in how `urls_command` was used in the test.
            # We need to simulate the original `urls_command` call from the test
            # or refactor the test to use a temporary file.
            
            # **SIMULATING ORIGINAL BEHAVIOR (UrlsCommand called with a URL line)**
            # Since the original test function called `urls_command(input_str)`, 
            # and `urls_command` expects a FILE path, the original logic for `test` 
            # calling `urls_command` was flawed/untested, or `urls_command` was 
            # expected to handle the string as a path.
            # To preserve the *intent* of testing the grading logic, we should probably
            # move the processing logic out of the command and call it.
            
            # ***Refactored Test Logic (Using a temporary file to match `urls_processor` signature)***
            temp_file = Path("temp_test_input.txt")
            temp_file.write_text(input_str, encoding="utf-8")
            result = urls_processor(str(temp_file))
            temp_file.unlink() # Clean up

            logger.info(f"Test {idx} completed successfully.")
            logger.debug(f"Test {idx} output: {result}")
            
            passed += 1
        except Exception as e:
            logger.error(f"Test {idx} failed with input='{input_str}', error={e}", exc_info=True)
            
    cov.stop()
    cov.save()

    # Generate a numeric report (using a simpler method for a single file)
    try:
        # Get the coverage percentage for this file
        analysis = cov.analysis(sys.argv[0]) # Analyze 'run.py' itself
        covered, total_lines = len(analysis[1]), len(analysis[1]) + len(analysis[2])
        coverage_percent = (covered / total_lines) * 100 if total_lines > 0 else 0.0
    except Exception as e:
        logger.warning(f"Could not generate coverage report: {e}")
        coverage_percent = 0.0
        
    logger.info(f"Test run finished. Passed: {passed}/{total}. Coverage: {coverage_percent:.1f}%")

    # Print the required summary line
    print(f"{passed}/{total} test cases passed. {coverage_percent:.1f}% line coverage achieved")
    
    # Check for success (original exit logic: coverage > 80% AND passed/total > 80%)
    success = (coverage_percent >= min_coverage) and (passed / total) >= 0.8
    return success


def run_install(req_path: Path = None) -> int:
    """Install all project dependencies."""
    try:
        # Change to root directory
        # Since `run.py` is likely in a 'src' dir, the root is up one level.
        root_dir = Path(__file__).resolve().parent.parent 
        
        # Install dependencies from project root
        logger.info("Installing project dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
            check=True
        )
        logger.info("Installation completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred during installation: {e}")
        return 1

# --- Main Entry Point ---

def incorrect():
    """Prints usage and exits."""
    print("Incorrect Use of CLI -> Try: ./run.py <install|test|url_file>", file=sys.stderr)
    sys.exit(1)

def main():

    logger = setup_logger()
    
    token = os.getenv("GITHUB_TOKEN")
    if not token or not token.strip() or token == None:
        print("Error: Invalid or missing GITHUB_TOKEN")
        sys.exit(1)

    """Handles command-line arguments (install, test, or urls_file)."""

    if len(sys.argv) < 2:
        logger.critical("Error in usage: Missing argument. Exiting.")
        incorrect()
        
    arg: str = sys.argv[1].lower()
    
    if arg == "install":
        repo_root = Path(__file__).parent.parent.resolve()
        req_file = repo_root / "requirements.txt"
        exit_code = run_install(req_file)
        sys.exit(exit_code)

    elif arg == "test":
        import subprocess
        import json
        import xml.etree.ElementTree as ET

        try:
            # Erase old coverage data
            subprocess.run([sys.executable, "-m", "coverage", "erase"], check=True)

            # Run pytest under coverage, run all tests, suppress warnings
            output = subprocess.run(
                [sys.executable, "-m", "coverage", "run", "-m", "pytest", "--disable-warnings", "-q", "--maxfail=0", "--tb=no"],
                check=False,
                capture_output=True,
            )
            output = output.stdout.decode("utf-8")

            # Generate coverage XML
            subprocess.run([sys.executable, "-m", "coverage", "xml"], check=True)

            # Parse coverage.xml to get overall line coverage
            tree = ET.parse("coverage.xml")
            root = tree.getroot()
            coverage = round(float(root.get("line-rate", 0.0)) * 100, 2)

            # Collect test functions
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "--tb=no", "-q"],
                capture_output=True,
                text=True,
            )

            test_count = 0
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                test_count += int(line.split(":")[1].strip())
            passed_count = test_count - output.count("FAILED")

            if test_count == 0:
                print("No tests found. Check the pytest collection output.", file=sys.stderr)
                sys.exit(1)

            # Output the results as JSON
            print(json.dumps({
                "test_count": test_count,
                "coverage": coverage
            }))
            json.dumps({
                "test_count": test_count,
                "coverage": coverage
            })

            print(f"{passed_count}/{test_count} test cases passed. {coverage}% line coverage achieved.")

            sys.exit(0)

        except Exception as e:
            print(f"Error running tests: {e}", file=sys.stderr)
            sys.exit(1)

    else:

        # Assume it's a file path for the urls_processor
        urls_file = arg
        logger.info(f"Processing URLs from file: {urls_file}")
        try:
            # urls_processor handles logging and printing JSON output
            urls_processor(urls_file)
            sys.exit(0)
        except Exception as e:
            logger.critical(f"A critical error occurred during URL processing: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()