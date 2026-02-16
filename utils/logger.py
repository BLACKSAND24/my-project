import logging
import sys
from financial_organism.config import CONFIG

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = getattr(logging, str(CONFIG.get("LOG_LEVEL","INFO")).upper(), logging.INFO)
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(level)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False
    return logger
