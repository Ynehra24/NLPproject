import logging

try:
    from post_generation.core.logging_utils import setup_logger
except ModuleNotFoundError:
    from core.logging_utils import setup_logger


logger = logging.getLogger()


__all__ = ["logger", "setup_logger"]
