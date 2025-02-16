import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Configure and return a logger with both file and console handlers.

    Args:
        name: Name for the logger and log file
        log_dir: Directory where log files will be stored

    Returns:
        Configured logger instance
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # create formatters and handlers
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # file handler
    file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
