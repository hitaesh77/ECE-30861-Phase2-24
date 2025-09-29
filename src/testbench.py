import logging
from metrics import run_all_metrics
from test import setup_logger

logger = setup_logger()

def read_enter_delimited_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
    return data

def run_tests():    
    test_inputs = read_enter_delimited_file("test_inputs.txt")

    for idx, input_str in enumerate(test_inputs, start=1):
        logger.info(f"Running Test {idx} with input: {input_str}")

        try:
            output_dict = run_all_metrics(input_str)

            # Summary
            logger.info(f"Test {idx} completed successfully. Keys returned: {list(output_dict.keys())}")

            # Full detail (only if LOG_LEVEL=2)
            logger.debug(f"Test {idx} output: {output_dict}")

        except Exception as e:
            logger.error(f"Test {idx} failed with input={input_str}, error={e}", exc_info=True)
