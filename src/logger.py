import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Returns a module-level logger with a consistent structured format."""
    logger = logging.getLogger(name)
    if not logger.handlers:                        # avoid duplicate handlers
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
