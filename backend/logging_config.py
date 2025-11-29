# backend/logging_config.py
import logging
import os

def get_logger(path="logs/requests.log"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    logger = logging.getLogger("fraud_agent")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(path)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger
