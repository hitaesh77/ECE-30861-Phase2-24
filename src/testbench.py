import logging
import asyncio
import coverage
from metrics import run_metrics as run_all_metrics
from test import setup_logger

logger = setup_logger()
cov = coverage.Coverage()
cov.start()

def read_enter_delimited_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
    return data

async def run_tests():    
    test_inputs = read_enter_delimited_file("test_inputs.txt")

    total = len(test_inputs)
    passed = 0

    for idx, input_str in enumerate(test_inputs, start=1):
        logger.info(f"Running Test {idx} with input: {input_str}")

        try:
            output_dict = await run_all_metrics(input_str)   # <-- must await async fn
            logger.info(f"Test {idx} completed successfully. Keys returned: {list(output_dict.keys())}")
            logger.debug(f"Test {idx} output: {output_dict}")
            passed += 1
        except Exception as e:
            logger.error(f"Test {idx} failed with input={input_str}, error={e}", exc_info=True)

    cov.stop()
    cov.save()
    coverage_percent = cov.report(show_missing=False)
    print(f"{passed}/{total} test cases passed. {coverage_percent:.1f}% line coverage achieved")
