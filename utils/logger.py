"""
Logging Module
==============

This module provides structured logging functionality for the Supply Chain Optimization system.

Features:
    - JSON-formatted logs for production
    - Human-readable logs for development
    - Multiple log handlers (console, file, rotating file)
    - Context processors for adding correlation IDs
    - Performance timing decorators
    - Log level configuration per environment

Usage:
    from utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Optimization started", product_count=50, service_level=0.95)
    logger.error("Optimization failed", error=str(e), product_id="PROD_001")
"""

import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from functools import wraps
import json
from datetime import datetime

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

from config import settings


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def configure_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    use_json: bool = False
) -> None:
    """
    Configure logging for the application.

    Parameters
    ----------
    level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        If None, uses settings.LOG_LEVEL
    log_file : str, optional
        Path to log file. If None, uses settings.LOG_FILE
    use_json : bool, default=False
        Use JSON formatting for logs (recommended for production)
    """
    log_level = level or settings.LOG_LEVEL
    log_file_path = log_file or settings.LOG_FILE

    # Configure structlog if available
    if STRUCTLOG_AVAILABLE and use_json:
        _configure_structlog(log_level, log_file_path)
    else:
        _configure_standard_logging(log_level, log_file_path)


def _configure_structlog(level: str, log_file: Optional[str]) -> None:
    """Configure structlog for structured logging."""
    # Determine if we're in production
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.is_production():
        # JSON formatting for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Pretty console formatting for development
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging as backend
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level),
        stream=sys.stdout,
    )

    # Add file handler if log file specified
    if log_file:
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        handler.setFormatter(logging.Formatter('%(message)s'))
        logging.root.addHandler(handler)


def _configure_standard_logging(level: str, log_file: Optional[str]) -> None:
    """Configure standard Python logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create formatters
    console_formatter = logging.Formatter(log_format, date_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # Remove existing handlers
    root_logger.handlers = []

    # Add console handler
    if settings.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file and settings.LOG_TO_FILE:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)


# ============================================================================
# LOGGER FACTORY
# ============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module)

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting optimization")
    >>> logger.debug("Detailed debug information", extra={'product_id': 'PROD_001'})
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


# ============================================================================
# CONTEXT MANAGEMENT
# ============================================================================

class LogContext:
    """
    Context manager for adding context to log messages.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> with LogContext(correlation_id="abc-123"):
    ...     logger.info("Processing request")
    """

    def __init__(self, **context):
        """Initialize context with key-value pairs."""
        self.context = context
        self.old_factory = None

    def __enter__(self):
        """Enter context."""
        if STRUCTLOG_AVAILABLE:
            structlog.contextvars.bind_contextvars(**self.context)
        else:
            # For standard logging, we'll use a filter
            self.old_factory = logging.getLogRecordFactory()

            def record_factory(*args, **kwargs):
                record = self.old_factory(*args, **kwargs)
                for key, value in self.context.items():
                    setattr(record, key, value)
                return record

            logging.setLogRecordFactory(record_factory)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if STRUCTLOG_AVAILABLE:
            structlog.contextvars.unbind_contextvars(*self.context.keys())
        elif self.old_factory:
            logging.setLogRecordFactory(self.old_factory)


# ============================================================================
# DECORATORS
# ============================================================================

def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. If None, creates logger from function module.

    Examples
    --------
    >>> @log_execution_time()
    ... def optimize_inventory(products):
    ...     # optimization logic
    ...     pass
    """

    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = func.__name__

            logger.debug(f"Starting {func_name}")

            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time

                logger.info(
                    f"Completed {func_name}",
                    elapsed_seconds=f"{elapsed_time:.2f}",
                    function=func_name
                )

                return result

            except Exception as e:
                elapsed_time = time.time() - start_time

                logger.error(
                    f"Failed {func_name}",
                    elapsed_seconds=f"{elapsed_time:.2f}",
                    function=func_name,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise

        return wrapper

    return decorator


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls with arguments.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. If None, creates logger from function module.

    Examples
    --------
    >>> @log_function_call()
    ... def calculate_eoq(demand, order_cost, holding_cost):
    ...     return np.sqrt(2 * demand * order_cost / holding_cost)
    """

    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # Log function call
            logger.debug(
                f"Calling {func_name}",
                function=func_name,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()) if kwargs else []
            )

            result = func(*args, **kwargs)

            return result

        return wrapper

    return decorator


# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

class PerformanceTimer:
    """
    Context manager for timing code blocks.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> with PerformanceTimer(logger, "data_loading"):
    ...     data = load_large_dataset()
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation_name: str,
        log_level: str = "INFO"
    ):
        """
        Initialize performance timer.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance
        operation_name : str
            Name of the operation being timed
        log_level : str, default="INFO"
            Log level for timing message
        """
        self.logger = logger
        self.operation_name = operation_name
        self.log_level = log_level.upper()
        self.start_time = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log elapsed time."""
        elapsed = time.time() - self.start_time

        log_method = getattr(self.logger, self.log_level.lower())
        log_method(
            f"{self.operation_name} completed",
            operation=self.operation_name,
            elapsed_seconds=f"{elapsed:.2f}"
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log_dataframe_info(
    logger: logging.Logger,
    df: Any,
    name: str = "DataFrame"
) -> None:
    """
    Log information about a pandas DataFrame.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    df : pd.DataFrame
        DataFrame to log info about
    name : str, default="DataFrame"
        Name to identify the DataFrame in logs
    """
    logger.info(
        f"{name} info",
        dataframe=name,
        shape=df.shape,
        columns=list(df.columns),
        memory_mb=f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"
    )


def log_optimization_start(
    logger: logging.Logger,
    **params: Any
) -> None:
    """
    Log optimization start with parameters.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    **params : Any
        Optimization parameters to log
    """
    logger.info("Optimization started", **params)


def log_optimization_result(
    logger: logging.Logger,
    baseline_cost: float,
    optimized_cost: float,
    method: str,
    **additional_metrics: Any
) -> None:
    """
    Log optimization results.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    baseline_cost : float
        Baseline total cost
    optimized_cost : float
        Optimized total cost
    method : str
        Optimization method used
    **additional_metrics : Any
        Additional metrics to log
    """
    savings = baseline_cost - optimized_cost
    savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

    logger.info(
        "Optimization completed",
        method=method,
        baseline_cost=f"${baseline_cost:,.2f}",
        optimized_cost=f"${optimized_cost:,.2f}",
        savings=f"${savings:,.2f}",
        savings_percent=f"{savings_pct:.1f}%",
        **additional_metrics
    )


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log error with context information.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    error : Exception
        Exception that occurred
    context : dict, optional
        Additional context information
    """
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    if context:
        error_info.update(context)

    logger.error("Error occurred", **error_info, exc_info=True)


# ============================================================================
# INITIALIZATION
# ============================================================================

# Configure logging on module import if not already configured
if not logging.root.handlers:
    configure_logging()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    configure_logging(level="DEBUG")

    logger = get_logger(__name__)

    logger.info("Application started")
    logger.debug("Debug information", extra_data="some value")

    # Using context manager
    with LogContext(correlation_id="abc-123", user_id="user-456"):
        logger.info("Processing request")

    # Using performance timer
    with PerformanceTimer(logger, "heavy_computation"):
        time.sleep(0.1)  # Simulate work

    logger.info("Application finished")
