"""
Logging utilities
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str = "results/logs", verbose: bool = False, run_name: str = None):
    """Setup logging with improved formatting and error handling"""
    try:
        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Set log level
        level = logging.DEBUG if verbose else logging.INFO
        
        # Set log format with timestamp and level
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # File handler with run_name as filename
        if run_name:
            log_filename = f"{run_name}.log"
        else:
            timestamp = datetime.now().strftime('%H%M%S')
            log_filename = f"run_{timestamp}.log"
        log_file = Path(log_dir) / log_filename
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Console handler with simplified format for better readability
        console_formatter = logging.Formatter('%(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        
        # Set root logger
        root_logger.setLevel(level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        return log_file
        
    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logging.warning(f"Failed to setup advanced logging: {e}. Using basic logging.")
        return None


def get_logger(name: str) -> logging.Logger:
    """Get logger"""
    return logging.getLogger(name)