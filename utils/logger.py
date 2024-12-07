
import os
import sys
from loguru import logger
from utils.register import registry


"""
This module configures and initializes the logging system for the application using the Loguru library. It provides a customizable logging setup that can be tailored to different runtime environments, such as development or production. The logger is designed to support various logging levels and formats, and it can output logs to different destinations, including standard error output and log files. The module also demonstrates the integration of the logging system with the application's registry for easy access across different parts of the application.

Functions:
    formatter(record): Defines the log format based on the log record's attributes.
    setup_logger(training_args): Configures and initializes the logger based on the provided training arguments. It registers the logger in the application's registry for global access.

The logger setup includes colorized output for better readability and supports conditional logging to different destinations based on the execution context (e.g., main process vs. other processes in distributed training environments).

Example Usage:
    from utils.logger import setup_logger
    training_args = TrainingArguments(...)
    logger = setup_logger(training_args)
    logger.info("Logging setup complete.")
"""


def formatter(record):
    """
    Custom log format based on the log record's attributes.

    Args:
        record: Log record object containing log attributes.

    Returns:
        str: Formatted log string.

    """
    # default format
    time_format = "<green>{time:MM-DD/HH:mm:ss}</>"
    lvl_format = "<lvl><i>{level:^5}</></>"
    rcd_format = "<cyan>{file}:{line:}</>"
    msg_format = "<lvl>{message}</>"

    if record["level"].name in ["WARNING", "CRITICAL"]:
        lvl_format = "<l>" + lvl_format + "</>"

    return "|".join([time_format, lvl_format, rcd_format, msg_format]) + "\n"


def setup_logger(training_args):
    """
    Configures and initializes the logger based on the provided training arguments.

    Args:
        training_args: TrainingArguments object containing training arguments.

    Returns:
        logger: Configured and initialized logger object.

    """
    logger.remove()

    main_process = training_args.local_rank == 0

    if main_process:
        logger.add(
            sys.stderr, format=formatter,
            colorize=True, enqueue=True
        )
    else:
        logger.add(
            os.path.join(training_args.save_dir, "log.txt"), format=formatter,
            colorize=True, enqueue=True
        )

    registry.register("logger", logger)

    return logger
