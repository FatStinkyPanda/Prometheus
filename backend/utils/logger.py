# backend/utils/logger.py

import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional

# Attempt to import the ConfigLoader. If it fails, we'll fall back to default logging.
try:
    from backend.utils.config_loader import ConfigLoader
except ImportError:
    ConfigLoader = None

class Logger:
    """
    A factory class for creating and configuring logger instances for the application.

    This class ensures that logging is configured only once in a thread-safe manner.
    It sets up handlers for both console and file logging based on settings
    retrieved from the configuration files.

    Usage:
        logger = Logger(__name__)
        logger.info("This is an informational message.")
        logger.error("This is an error message.", exc_info=True)
    """
    _logging_configured = False
    _lock = threading.Lock()

    def __new__(cls, name: str) -> logging.Logger:
        """
        Creates or retrieves a logger instance after ensuring logging is configured.

        Args:
            name (str): The name for the logger, typically __name__.

        Returns:
            logging.Logger: A configured logger instance.
        """
        # Use a lock for thread-safety during the first-time setup.
        with cls._lock:
            if not cls._logging_configured:
                cls._setup_logging()
                cls._logging_configured = True
        
        return logging.getLogger(name)

    @classmethod
    def _parse_log_size(cls, size_str: str) -> int:
        """
        Parses a log size string (e.g., "10MB", "500KB", "1GB") into bytes.

        Args:
            size_str (str): The size string.

        Returns:
            int: The size in bytes. Returns a default of 10MB if parsing fails.
        """
        size_str = size_str.strip().upper()
        try:
            if size_str.endswith('KB'):
                return int(size_str[:-2]) * 1024
            elif size_str.endswith('MB'):
                return int(size_str[:-2]) * 1024 * 1024
            elif size_str.endswith('GB'):
                return int(size_str[:-2]) * 1024 * 1024 * 1024
            else:
                return int(size_str)
        except (ValueError, TypeError):
            # Log a warning to stderr as the main logger isn't fully configured yet.
            print(f"Warning: Could not parse log size '{size_str}'. Defaulting to 10MB.", file=sys.stderr)
            return 10 * 1024 * 1024  # Default to 10MB

    @classmethod
    def _setup_logging(cls):
        """
        Sets up the root logger with handlers and formatters.
        
        Reads configuration from files via ConfigLoader. If this fails,
        it falls back to a basic, console-only configuration.
        """
        log_config = {}
        try:
            if ConfigLoader:
                # Load configuration using the utility we created
                full_config = ConfigLoader.load_config(
                    primary_config="prometheus_config.yaml",
                    merge_configs=["logging_config.yaml"]
                )
                log_config = full_config.get('logging', {})
            else:
                raise RuntimeError("ConfigLoader not available.")

        except Exception as e:
            # Fallback configuration if ConfigLoader fails for any reason
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - [FALLBACK] %(message)s',
                stream=sys.stdout
            )
            fallback_logger = logging.getLogger(__name__)
            fallback_logger.error(
                "Failed to load logging configuration. Using fallback basicConfig. Error: %s",
                e,
                exc_info=True
            )
            return

        # --- Get configuration with robust defaults ---
        log_level_str = log_config.get('level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        log_format = log_config.get(
            'format',
            '%(asctime)s - %(name)s - [%(levelname)s] - (%(threadName)s) - %(message)s'
        )
        
        log_file = log_config.get('file', 'logs/prometheus.log')
        max_size_str = log_config.get('max_size', '100MB')
        backup_count = log_config.get('backup_count', 5)

        # --- Configure the root logger ---
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear any existing handlers to prevent duplicate logs
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
            
        # --- Create formatter ---
        formatter = logging.Formatter(log_format)

        # --- Create Console Handler ---
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        except Exception as e:
            print(f"FATAL: Could not set up console logging handler: {e}", file=sys.stderr)

        # --- Create Rotating File Handler ---
        try:
            log_file_path = Path(log_file)
            # Ensure the directory for the log file exists
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            max_bytes = cls._parse_log_size(max_size_str)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            # Use the newly configured logger to announce success
            init_logger = logging.getLogger(__name__)
            init_logger.info("Logging configured successfully. Level: %s. Output file: %s", log_level_str, log_file_path)

        except (PermissionError, OSError) as e:
            # If we can't write the log file, we must report it.
            # We can use the console logger which should already be set up.
            error_logger = logging.getLogger(__name__)
            error_logger.error(
                "Could not create or write to log file at '%s'. "
                "Logging to file is disabled. Error: %s",
                log_file, e, exc_info=True
            )
        except Exception as e:
            # Catch any other unexpected errors during file handler setup
            error_logger = logging.getLogger(__name__)
            error_logger.error(
                "An unexpected error occurred while setting up the file logger. Error: %s",
                e, exc_info=True
            )