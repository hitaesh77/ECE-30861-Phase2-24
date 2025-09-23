import os
import logging

def setup_logger():
    # Log file from environment, or default
    log_file = os.getenv("LOG_FILE", "testbench.log")
    log_level = int(os.getenv("LOG_LEVEL", "1"))  # default to INFO

    if log_level == 0:
        level = logging.CRITICAL + 1  # silence
    elif log_level == 1:
        level = logging.INFO
    elif log_level == 2:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        filename=log_file,
        filemode="w",  # overwrite for each run
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger("testbench")