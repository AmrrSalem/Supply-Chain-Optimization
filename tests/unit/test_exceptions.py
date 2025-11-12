"""
Unit Tests for Custom Exceptions
=================================

Tests for exceptions.py module including custom exception classes.
"""

import pytest

from utils.exceptions import (
    SupplyChainError,
    ValidationError,
    InvalidParameterError,
    DataQualityError,
    OptimizationError,
    ConvergenceError,
    InfeasibleError,
    DataError,
    DataLoadError,
    DataFormatError,
    MissingDataError,
    ConfigurationError,
    raise_for_invalid_range,
    raise_for_negative_value,
    raise_for_missing_columns
)


@pytest.mark.unit
@pytest.mark.fast
class TestBaseException:
    """Test suite for base SupplyChainError exception."""

    def test_base_exception_with_message(self):
        """Test base exception with message only."""
        error = SupplyChainError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}

    def test_base_exception_with_details(self):
        """Test base exception with details."""
        error = SupplyChainError(
            "Test error",
            details={"param": "value", "count": 42}
        )

        assert "Test error" in str(error)
        assert error.details == {"param": "value", "count": 42}
        # String representation should include details
        error_str = str(error)
        assert "param=value" in error_str
        assert "count=42" in error_str


@pytest.mark.unit
@pytest.mark.fast
class TestValidationExceptions:
    """Test suite for validation exception classes."""

    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError("Invalid input")
        assert isinstance(error, SupplyChainError)
        assert "Invalid input" in str(error)

    def test_invalid_parameter_error(self):
        """Test InvalidParameterError exception."""
        error = InvalidParameterError(
            "Parameter out of range",
            details={"value": 1.5, "min": 0.5, "max": 0.999}
        )

        assert isinstance(error, ValidationError)
        assert "out of range" in str(error)
        assert error.details['value'] == 1.5

    def test_data_quality_error(self):
        """Test DataQualityError exception."""
        error = DataQualityError(
            "Negative demand detected",
            details={"product_id": "PROD_001", "demand": -15.2}
        )

        assert isinstance(error, ValidationError)
        assert "Negative demand" in str(error)


@pytest.mark.unit
@pytest.mark.fast
class TestOptimizationExceptions:
    """Test suite for optimization exception classes."""

    def test_optimization_error(self):
        """Test OptimizationError exception."""
        error = OptimizationError("Optimization failed")
        assert isinstance(error, SupplyChainError)
        assert "failed" in str(error)

    def test_convergence_error(self):
        """Test ConvergenceError exception."""
        error = ConvergenceError(
            "Did not converge",
            details={"iterations": 200, "product_id": "PROD_042"}
        )

        assert isinstance(error, OptimizationError)
        assert "converge" in str(error)
        assert error.details['iterations'] == 200

    def test_infeasible_error(self):
        """Test InfeasibleError exception."""
        error = InfeasibleError(
            "Budget constraint cannot be met",
            details={"required": 100000, "available": 80000}
        )

        assert isinstance(error, OptimizationError)
        assert "Budget" in str(error)


@pytest.mark.unit
@pytest.mark.fast
class TestDataExceptions:
    """Test suite for data exception classes."""

    def test_data_error(self):
        """Test DataError exception."""
        error = DataError("Data error occurred")
        assert isinstance(error, SupplyChainError)
        assert "Data error" in str(error)

    def test_data_load_error(self):
        """Test DataLoadError exception."""
        error = DataLoadError(
            "Failed to read file",
            details={"file_path": "data.csv", "error": "File not found"}
        )

        assert isinstance(error, DataError)
        assert "Failed to read" in str(error)

    def test_data_format_error(self):
        """Test DataFormatError exception."""
        error = DataFormatError(
            "Missing required columns",
            details={"missing": ["product_id", "demand"]}
        )

        assert isinstance(error, DataError)
        assert "Missing" in str(error)

    def test_missing_data_error(self):
        """Test MissingDataError exception."""
        error = MissingDataError(
            "Product data not loaded",
            details={"expected": "product_summary"}
        )

        assert isinstance(error, DataError)
        assert "not loaded" in str(error)


@pytest.mark.unit
@pytest.mark.fast
class TestConfigurationException:
    """Test suite for configuration exception."""

    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        error = ConfigurationError(
            "Invalid environment",
            details={"value": "prod", "allowed": ["development", "production"]}
        )

        assert isinstance(error, SupplyChainError)
        assert "Invalid environment" in str(error)


@pytest.mark.unit
@pytest.mark.fast
class TestHelperFunctions:
    """Test suite for exception helper functions."""

    def test_raise_for_invalid_range_within_range(self):
        """Test that no exception is raised for valid value."""
        # Should not raise
        raise_for_invalid_range(
            value=0.95,
            min_value=0.5,
            max_value=0.999,
            param_name="service_level"
        )

    def test_raise_for_invalid_range_below_minimum(self):
        """Test exception is raised for value below minimum."""
        with pytest.raises(InvalidParameterError) as exc_info:
            raise_for_invalid_range(
                value=0.3,
                min_value=0.5,
                max_value=0.999,
                param_name="service_level"
            )

        assert "between 0.5 and 0.999" in str(exc_info.value)
        assert exc_info.value.details['value'] == 0.3

    def test_raise_for_invalid_range_above_maximum(self):
        """Test exception is raised for value above maximum."""
        with pytest.raises(InvalidParameterError):
            raise_for_invalid_range(
                value=1.5,
                min_value=0.5,
                max_value=0.999,
                param_name="service_level"
            )

    def test_raise_for_negative_value_positive(self):
        """Test that no exception is raised for positive value."""
        # Should not raise
        raise_for_negative_value(
            value=100.0,
            param_name="demand"
        )

    def test_raise_for_negative_value_negative(self):
        """Test exception is raised for negative value."""
        with pytest.raises(InvalidParameterError) as exc_info:
            raise_for_negative_value(
                value=-50.0,
                param_name="demand"
            )

        assert "cannot be negative" in str(exc_info.value)
        assert exc_info.value.details['value'] == -50.0

    def test_raise_for_missing_columns_all_present(self, sample_product_data):
        """Test that no exception is raised when all columns present."""
        required = ['product_id', 'demand', 'unit_cost']

        # Should not raise
        raise_for_missing_columns(
            sample_product_data,
            required,
            "Test Data"
        )

    def test_raise_for_missing_columns_some_missing(self, sample_product_data):
        """Test exception is raised when columns are missing."""
        required = ['product_id', 'demand', 'nonexistent_column']

        with pytest.raises(DataFormatError) as exc_info:
            raise_for_missing_columns(
                sample_product_data,
                required,
                "Test Data"
            )

        assert "missing required columns" in str(exc_info.value).lower()
        assert 'nonexistent_column' in exc_info.value.details['missing_columns']


@pytest.mark.unit
class TestExceptionInheritance:
    """Test suite for exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from SupplyChainError."""
        exceptions = [
            ValidationError,
            InvalidParameterError,
            DataQualityError,
            OptimizationError,
            ConvergenceError,
            InfeasibleError,
            DataError,
            DataLoadError,
            DataFormatError,
            MissingDataError,
            ConfigurationError
        ]

        for exc_class in exceptions:
            instance = exc_class("test message")
            assert isinstance(instance, SupplyChainError)
            assert isinstance(instance, Exception)

    def test_validation_errors_hierarchy(self):
        """Test validation error hierarchy."""
        error = InvalidParameterError("test")
        assert isinstance(error, ValidationError)
        assert isinstance(error, SupplyChainError)

        error = DataQualityError("test")
        assert isinstance(error, ValidationError)
        assert isinstance(error, SupplyChainError)

    def test_optimization_errors_hierarchy(self):
        """Test optimization error hierarchy."""
        error = ConvergenceError("test")
        assert isinstance(error, OptimizationError)
        assert isinstance(error, SupplyChainError)

        error = InfeasibleError("test")
        assert isinstance(error, OptimizationError)
        assert isinstance(error, SupplyChainError)

    def test_data_errors_hierarchy(self):
        """Test data error hierarchy."""
        for exc_class in [DataLoadError, DataFormatError, MissingDataError]:
            error = exc_class("test")
            assert isinstance(error, DataError)
            assert isinstance(error, SupplyChainError)
