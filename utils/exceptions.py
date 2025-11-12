"""
Custom Exceptions Module
========================

This module defines custom exception classes for the Supply Chain Optimization system.

Exception Hierarchy:
    SupplyChainError (base)
    ├── ValidationError
    │   ├── InvalidParameterError
    │   └── DataQualityError
    ├── OptimizationError
    │   ├── ConvergenceError
    │   └── InfeasibleError
    ├── DataError
    │   ├── DataLoadError
    │   ├── DataFormatError
    │   └── MissingDataError
    └── ConfigurationError

Usage:
    from utils.exceptions import ValidationError, OptimizationError

    if service_level < 0.5:
        raise ValidationError("Service level must be at least 50%")
"""


# ============================================================================
# BASE EXCEPTION
# ============================================================================

class SupplyChainError(Exception):
    """
    Base exception for all supply chain optimization errors.

    All custom exceptions inherit from this class.
    """

    def __init__(self, message: str, details: dict = None):
        """
        Initialize exception.

        Parameters
        ----------
        message : str
            Error message
        details : dict, optional
            Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        """Return string representation."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# ============================================================================
# VALIDATION EXCEPTIONS
# ============================================================================

class ValidationError(SupplyChainError):
    """
    Exception raised for validation errors.

    Raised when input data or parameters fail validation checks.
    """
    pass


class InvalidParameterError(ValidationError):
    """
    Exception raised for invalid parameter values.

    Examples
    --------
    >>> raise InvalidParameterError(
    ...     "Service level out of range",
    ...     details={"value": 1.5, "min": 0.5, "max": 0.999}
    ... )
    """
    pass


class DataQualityError(ValidationError):
    """
    Exception raised for data quality issues.

    Raised when data contains anomalies, outliers, or quality problems.

    Examples
    --------
    >>> raise DataQualityError(
    ...     "Negative demand detected",
    ...     details={"product_id": "PROD_001", "demand": -15.2}
    ... )
    """
    pass


# ============================================================================
# OPTIMIZATION EXCEPTIONS
# ============================================================================

class OptimizationError(SupplyChainError):
    """
    Exception raised for optimization errors.

    Base class for all optimization-related errors.
    """
    pass


class ConvergenceError(OptimizationError):
    """
    Exception raised when optimization fails to converge.

    Examples
    --------
    >>> raise ConvergenceError(
    ...     "Stochastic optimization did not converge",
    ...     details={"iterations": 200, "product_id": "PROD_042"}
    ... )
    """
    pass


class InfeasibleError(OptimizationError):
    """
    Exception raised when optimization problem is infeasible.

    Raised when constraints cannot be satisfied.

    Examples
    --------
    >>> raise InfeasibleError(
    ...     "Budget constraint cannot be met",
    ...     details={"required": 100000, "available": 80000}
    ... )
    """
    pass


# ============================================================================
# DATA EXCEPTIONS
# ============================================================================

class DataError(SupplyChainError):
    """
    Exception raised for data-related errors.

    Base class for all data handling errors.
    """
    pass


class DataLoadError(DataError):
    """
    Exception raised when data cannot be loaded.

    Examples
    --------
    >>> raise DataLoadError(
    ...     "Failed to read CSV file",
    ...     details={"file_path": "data.csv", "error": "File not found"}
    ... )
    """
    pass


class DataFormatError(DataError):
    """
    Exception raised for data format errors.

    Raised when data is not in the expected format.

    Examples
    --------
    >>> raise DataFormatError(
    ...     "Missing required columns",
    ...     details={"missing": ["product_id", "demand"]}
    ... )
    """
    pass


class MissingDataError(DataError):
    """
    Exception raised when required data is missing.

    Examples
    --------
    >>> raise MissingDataError(
    ...     "Product data not loaded",
    ...     details={"expected": "product_summary", "actual": None}
    ... )
    """
    pass


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================

class ConfigurationError(SupplyChainError):
    """
    Exception raised for configuration errors.

    Raised when configuration is invalid or incomplete.

    Examples
    --------
    >>> raise ConfigurationError(
    ...     "Invalid environment setting",
    ...     details={"value": "prod", "allowed": ["development", "production"]}
    ... )
    """
    pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def raise_for_invalid_range(
    value: float,
    min_value: float,
    max_value: float,
    param_name: str
):
    """
    Raise InvalidParameterError if value is out of range.

    Parameters
    ----------
    value : float
        Value to check
    min_value : float
        Minimum allowed value
    max_value : float
        Maximum allowed value
    param_name : str
        Parameter name for error message

    Raises
    ------
    InvalidParameterError
        If value is out of range
    """
    if value < min_value or value > max_value:
        raise InvalidParameterError(
            f"{param_name} must be between {min_value} and {max_value}",
            details={
                "parameter": param_name,
                "value": value,
                "min": min_value,
                "max": max_value
            }
        )


def raise_for_negative_value(value: float, param_name: str):
    """
    Raise InvalidParameterError if value is negative.

    Parameters
    ----------
    value : float
        Value to check
    param_name : str
        Parameter name for error message

    Raises
    ------
    InvalidParameterError
        If value is negative
    """
    if value < 0:
        raise InvalidParameterError(
            f"{param_name} cannot be negative",
            details={"parameter": param_name, "value": value}
        )


def raise_for_missing_columns(df, required_columns: list, data_name: str = "DataFrame"):
    """
    Raise DataFormatError if required columns are missing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    required_columns : list
        List of required column names
    data_name : str, default="DataFrame"
        Name of the data for error message

    Raises
    ------
    DataFormatError
        If required columns are missing
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise DataFormatError(
            f"{data_name} is missing required columns",
            details={
                "missing_columns": list(missing),
                "available_columns": list(df.columns)
            }
        )
