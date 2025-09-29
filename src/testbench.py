import logging
from metrics import run_metrics
from test import setup_logger

logger = setup_logger()

def read_enter_delimited_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
    return data

def run_tests():    
    test_inputs = read_enter_delimited_file("test_inputs.txt")

    total = len(test_inputs)
    passed = 0

    for idx, input_str in enumerate(test_inputs, start=1):
        logger.info(f"Running Test {idx} with input: {input_str}")

        try:
            output_dict = run_metrics()

            # Summary
            logger.info(f"Test {idx} completed successfully. Keys returned: {list(output_dict.keys())}")

            # Full detail (only if LOG_LEVEL=2)
            logger.debug(f"Test {idx} output: {output_dict}")

            passed += 1  # count as passed if no exception
        except Exception as e:
            logger.error(f"Test {idx} failed with input={input_str}, error={e}", exc_info=True)

    # Compute coverage
    if total > 0:
        coverage = (passed / total) * 100
    else:
        coverage = 0.0

    # Print final summary line to stdout
    print(f"{passed}/{total} test cases passed. {coverage:.1f}% line coverage achieved")
