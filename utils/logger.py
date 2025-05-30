# utils/logger.py
"""
Centralized logging configuration for the application.
"""
import logging
import sys
from config.settings import LOG_FORMAT, LOG_LEVEL

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger, typically __name__ of the calling module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if logger is already configured
    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(console_handler)
        
        # Optional: File Handler (Uncomment to enable logging to a file)
        # file_handler = logging.FileHandler("app.log")
        # file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        # logger.addHandler(file_handler)

    # Set propagation to False if you don't want messages to go to the root logger,
    # especially if the root logger has its own handlers.
    # logger.propagate = False
    
    return logger

# Example usage in other modules:
# from utils.logger import get_logger
# logger = get_logger(__name__)
# logger.info("This is an info message.")
