# app/core/logging.py
"""
Logging configuration for the application.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import os

from app.core.config import settings

# Try to import JSON logger
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False
    jsonlogger = None


def setup_logging():
    """Set up logging configuration."""
    # Set logging level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove default handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Set formatter based on configuration
    if settings.LOG_FORMAT == "json" and JSON_LOGGER_AVAILABLE:
        # JSON formatter for production
        formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(funcName)s %(message)s",
            timestamp=True
        )
    else:
        # Human-readable formatter for development (or fallback)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
    
    console_handler.setFormatter(formatter)
    
    # Add console handler
    root_logger.addHandler(console_handler)
    
    # Only add file handler if not in demo mode
    if settings.ENVIRONMENT not in ("demo", "test"):
        try:
            # Create logs directory relative to current working directory
            log_dir = Path("./logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # File handler with rotation
            file_handler = RotatingFileHandler(
                log_dir / "app.log",
                maxBytes=10_000_000,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            logging.warning(f"Could not set up file logging: {e}")
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured - Level: {settings.LOG_LEVEL}, "
        f"Format: {settings.LOG_FORMAT}, "
        f"Environment: {settings.ENVIRONMENT}"
    )