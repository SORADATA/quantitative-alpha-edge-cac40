import logging
import sys
from datetime import datetime
from const import LOG_DIR


def setup_logger(name: str = "PortfolioPipeline") -> logging.Logger:
    """Configures production-grade logging."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s | %(levelname)8s | %(message)s')
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        # File Handler
        # We ensure that the forlder exists
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
