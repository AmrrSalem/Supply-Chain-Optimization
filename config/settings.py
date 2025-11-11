"""
Settings Module
===============

This module provides configuration settings that can be loaded from:
1. Environment variables (highest priority)
2. .env files
3. Default values from constants.py

Usage:
    from config import settings

    # Access settings
    print(settings.SERVICE_LEVEL)
    print(settings.ENVIRONMENT)
"""

import os
from typing import Optional
from pathlib import Path

# Try to import python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from config.constants import (
    DEFAULT_SERVICE_LEVEL,
    DEFAULT_HOLDING_COST_RATE,
    DEFAULT_N_PRODUCTS,
    DEFAULT_N_PERIODS,
    DEFAULT_DEMAND_SCENARIOS,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_LOG_LEVEL,
    STREAMLIT_PORT,
    STREAMLIT_ADDRESS,
    CACHE_TTL,
    VERSION,
)


class Settings:
    """
    Configuration settings for the Supply Chain Optimization system.

    Settings are loaded in this priority order:
    1. Environment variables
    2. .env file
    3. Default values

    Attributes
    ----------
    ENVIRONMENT : str
        Current environment (development, staging, production)
    DEBUG : bool
        Debug mode flag
    SERVICE_LEVEL : float
        Default service level target
    HOLDING_COST_RATE : float
        Default holding cost rate
    LOG_LEVEL : str
        Logging level
    ... and more
    """

    def __init__(self):
        """Initialize settings by loading from environment and defaults."""
        # Load .env file if available
        self._load_env_file()

        # ====================================================================
        # ENVIRONMENT SETTINGS
        # ====================================================================

        self.ENVIRONMENT = self._get_env(
            "ENVIRONMENT",
            default="development",
            allowed_values=["development", "staging", "production"]
        )

        self.DEBUG = self._get_bool_env("DEBUG", default=True)

        self.VERSION = VERSION

        # ====================================================================
        # OPTIMIZATION SETTINGS
        # ====================================================================

        self.SERVICE_LEVEL = self._get_float_env(
            "SERVICE_LEVEL",
            default=DEFAULT_SERVICE_LEVEL,
            min_value=0.5,
            max_value=0.999
        )

        self.HOLDING_COST_RATE = self._get_float_env(
            "HOLDING_COST_RATE",
            default=DEFAULT_HOLDING_COST_RATE,
            min_value=0.01,
            max_value=1.0
        )

        self.N_PRODUCTS = self._get_int_env(
            "N_PRODUCTS",
            default=DEFAULT_N_PRODUCTS,
            min_value=1,
            max_value=10000
        )

        self.N_PERIODS = self._get_int_env(
            "N_PERIODS",
            default=DEFAULT_N_PERIODS,
            min_value=30,
            max_value=3650  # 10 years
        )

        self.DEMAND_SCENARIOS = self._get_int_env(
            "DEMAND_SCENARIOS",
            default=DEFAULT_DEMAND_SCENARIOS,
            min_value=100,
            max_value=10000
        )

        self.FORECAST_HORIZON = self._get_int_env(
            "FORECAST_HORIZON",
            default=DEFAULT_FORECAST_HORIZON,
            min_value=1,
            max_value=365
        )

        # ====================================================================
        # LOGGING SETTINGS
        # ====================================================================

        self.LOG_LEVEL = self._get_env(
            "LOG_LEVEL",
            default=DEFAULT_LOG_LEVEL,
            allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )

        self.LOG_FILE = self._get_env("LOG_FILE", default=None)

        self.LOG_TO_CONSOLE = self._get_bool_env("LOG_TO_CONSOLE", default=True)

        self.LOG_TO_FILE = self._get_bool_env(
            "LOG_TO_FILE",
            default=self.LOG_FILE is not None
        )

        # ====================================================================
        # STREAMLIT SETTINGS
        # ====================================================================

        self.STREAMLIT_PORT = self._get_int_env(
            "STREAMLIT_PORT",
            default=STREAMLIT_PORT,
            min_value=1024,
            max_value=65535
        )

        self.STREAMLIT_ADDRESS = self._get_env(
            "STREAMLIT_ADDRESS",
            default=STREAMLIT_ADDRESS
        )

        # ====================================================================
        # CACHE SETTINGS
        # ====================================================================

        self.CACHE_ENABLED = self._get_bool_env("CACHE_ENABLED", default=True)

        self.CACHE_TTL = self._get_int_env(
            "CACHE_TTL",
            default=CACHE_TTL,
            min_value=0
        )

        self.REDIS_URL = self._get_env(
            "REDIS_URL",
            default="redis://localhost:6379/0"
        )

        # ====================================================================
        # DATABASE SETTINGS (Future use)
        # ====================================================================

        self.DATABASE_URL = self._get_env("DATABASE_URL", default=None)

        # ====================================================================
        # SECURITY SETTINGS
        # ====================================================================

        self.SECRET_KEY = self._get_env(
            "SECRET_KEY",
            default="changeme-in-production-" + os.urandom(24).hex()
        )

        self.ENABLE_AUTH = self._get_bool_env("ENABLE_AUTH", default=False)

        self.AUTH_USERNAME = self._get_env("AUTH_USERNAME", default="admin")

        self.AUTH_PASSWORD = self._get_env("AUTH_PASSWORD", default=None)

        # ====================================================================
        # PERFORMANCE SETTINGS
        # ====================================================================

        self.PARALLEL_PROCESSING = self._get_bool_env(
            "PARALLEL_PROCESSING",
            default=True
        )

        self.N_JOBS = self._get_int_env(
            "N_JOBS",
            default=-1,  # Use all CPU cores
            min_value=-1
        )

        # ====================================================================
        # FILE PATHS
        # ====================================================================

        self.BASE_DIR = Path(__file__).resolve().parent.parent

        self.DATA_DIR = self.BASE_DIR / self._get_env("DATA_DIR", default="data")
        self.MODELS_DIR = self.BASE_DIR / self._get_env("MODELS_DIR", default="models")
        self.LOGS_DIR = self.BASE_DIR / self._get_env("LOGS_DIR", default="logs")
        self.REPORTS_DIR = self.BASE_DIR / self._get_env("REPORTS_DIR", default="reports")

        # Create directories if they don't exist
        self._create_directories()

        # ====================================================================
        # MONITORING SETTINGS (Future use)
        # ====================================================================

        self.ENABLE_MONITORING = self._get_bool_env(
            "ENABLE_MONITORING",
            default=False
        )

        self.METRICS_PORT = self._get_int_env(
            "METRICS_PORT",
            default=9090,
            min_value=1024,
            max_value=65535
        )

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _load_env_file(self):
        """Load environment variables from .env file if available."""
        if DOTENV_AVAILABLE:
            env_file = Path(__file__).resolve().parent.parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)

    @staticmethod
    def _get_env(
        key: str,
        default: Optional[str] = None,
        allowed_values: Optional[list] = None
    ) -> str:
        """
        Get string environment variable.

        Parameters
        ----------
        key : str
            Environment variable name
        default : str, optional
            Default value if not found
        allowed_values : list, optional
            List of allowed values

        Returns
        -------
        str
            Environment variable value or default

        Raises
        ------
        ValueError
            If value not in allowed_values
        """
        value = os.getenv(key, default)

        if allowed_values and value not in allowed_values:
            raise ValueError(
                f"Invalid value '{value}' for {key}. "
                f"Must be one of: {allowed_values}"
            )

        return value

    @staticmethod
    def _get_bool_env(key: str, default: bool = False) -> bool:
        """
        Get boolean environment variable.

        Accepts: true, yes, 1, on (case insensitive) for True
        """
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', 'yes', '1', 'on')

    @staticmethod
    def _get_int_env(
        key: str,
        default: int = 0,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> int:
        """
        Get integer environment variable with optional bounds checking.

        Parameters
        ----------
        key : str
            Environment variable name
        default : int
            Default value if not found
        min_value : int, optional
            Minimum allowed value
        max_value : int, optional
            Maximum allowed value

        Returns
        -------
        int
            Environment variable value or default

        Raises
        ------
        ValueError
            If value is out of bounds or not a valid integer
        """
        value = os.getenv(key)
        if value is None:
            return default

        try:
            int_value = int(value)
        except ValueError:
            raise ValueError(f"Invalid integer value for {key}: {value}")

        if min_value is not None and int_value < min_value:
            raise ValueError(
                f"Value for {key} ({int_value}) is below minimum ({min_value})"
            )

        if max_value is not None and int_value > max_value:
            raise ValueError(
                f"Value for {key} ({int_value}) exceeds maximum ({max_value})"
            )

        return int_value

    @staticmethod
    def _get_float_env(
        key: str,
        default: float = 0.0,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> float:
        """
        Get float environment variable with optional bounds checking.

        Parameters
        ----------
        key : str
            Environment variable name
        default : float
            Default value if not found
        min_value : float, optional
            Minimum allowed value
        max_value : float, optional
            Maximum allowed value

        Returns
        -------
        float
            Environment variable value or default

        Raises
        ------
        ValueError
            If value is out of bounds or not a valid float
        """
        value = os.getenv(key)
        if value is None:
            return default

        try:
            float_value = float(value)
        except ValueError:
            raise ValueError(f"Invalid float value for {key}: {value}")

        if min_value is not None and float_value < min_value:
            raise ValueError(
                f"Value for {key} ({float_value}) is below minimum ({min_value})"
            )

        if max_value is not None and float_value > max_value:
            raise ValueError(
                f"Value for {key} ({float_value}) exceeds maximum ({max_value})"
            )

        return float_value

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR, self.REPORTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"

    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.ENVIRONMENT == "staging"

    def __repr__(self) -> str:
        """Return string representation of settings."""
        return f"Settings(environment={self.ENVIRONMENT}, debug={self.DEBUG})"

    def to_dict(self) -> dict:
        """
        Convert settings to dictionary.

        Returns
        -------
        dict
            Dictionary of all settings
        """
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith('_') and not callable(getattr(self, key))
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_settings() -> Settings:
    """Get the global settings instance."""
    return Settings()


def print_settings():
    """Print all current settings (for debugging)."""
    settings = Settings()
    print("=" * 60)
    print("CURRENT SETTINGS")
    print("=" * 60)

    for key, value in settings.to_dict().items():
        # Don't print sensitive values
        if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key']):
            print(f"{key}: ***REDACTED***")
        else:
            print(f"{key}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    # Print settings when run directly
    print_settings()
