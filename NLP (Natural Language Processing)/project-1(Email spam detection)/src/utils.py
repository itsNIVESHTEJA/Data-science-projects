"""
Shared utility helpers for the Spam Detection project.
"""

import os
import logging
from datetime import datetime

import config


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to both console and a timestamped log file."""
    os.makedirs(config.LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = os.path.join(config.LOG_DIR,
                            f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def ensure_dirs():
    """Create all required project directories if they do not yet exist."""
    for path in (config.MODEL_DIR, config.LOG_DIR,
                 config.PLOTS_DIR, config.REPORTS_DIR):
        os.makedirs(path, exist_ok=True)
