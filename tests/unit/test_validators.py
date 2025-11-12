"""
Unit Tests for Input Validation
================================

Tests for validators.py module including Pydantic models and validation functions.
"""

import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError as PydanticValidationError

from utils.validators import (
    OptimizationParams,
    ProductData,
    ProductDataset,
    validate_product_dataframe,
    check_data_quality,
    sanitize_product_id,
    sanitize_dataframe,
    validate_optimization_results
)
from utils.exceptions import ValidationError


@pytest.mark.unit
@pytest.mark.fast
class TestOptimizationParams:
    """Test suite for OptimizationParams Pydantic model."""

    def test_valid_optimization_params(self):
        """Test creating OptimizationParams with valid values."""
        params = OptimizationParams(
            service_level=0.95,
            holding_cost_rate=0.25
        )

        assert params.service_level == 0.95
        assert params.holding_cost_rate == 0.25

    @pytest.mark.parametrize("service_level,should_fail", [
        (0.5, False),   # Minimum valid
        (0.95, False),  # Normal
        (0.999, False), # Maximum valid
        (0.49, True),   # Too low
        (1.0, True),    # Too high
        (1.5, True),    # Way too high
    ])
    def test_service_level_validation(self, service_level, should_fail):
        """Test service level bounds checking."""
        if should_fail:
            with pytest.raises(PydanticValidationError):
                OptimizationParams(
                    service_level=service_level,
                    holding_cost_rate=0.25
                )
        else:
            params = OptimizationParams(
                service_level=service_level,
                holding_cost_rate=0.25
            )
            assert params.service_level == service_level

    @pytest.mark.parametrize("holding_cost_rate,should_fail", [
        (0.01, False),  # Minimum valid
        (0.25, False),  # Normal
        (1.0, False),   # Maximum valid
        (0.005, True),  # Too low
        (1.5, True),    # Too high
    ])
    def test_holding_cost_rate_validation(self, holding_cost_rate, should_fail):
        """Test holding cost rate bounds checking."""
        if should_fail:
            with pytest.raises(PydanticValidationError):
                OptimizationParams(
                    service_level=0.95,
                    holding_cost_rate=holding_cost_rate
                )
        else:
            params = OptimizationParams(
                service_level=0.95,
                holding_cost_rate=holding_cost_rate
            )
            assert params.holding_cost_rate == holding_cost_rate


@pytest.mark.unit
@pytest.mark.fast
class TestProductData:
    """Test suite for ProductData Pydantic model."""

    def test_valid_product_data(self):
        """Test creating ProductData with valid values."""
        product = ProductData(
            product_id="PROD_001",
            demand=100.5,
            unit_cost=45.0,
            order_cost=200.0,
            lead_time=7
        )

        assert product.product_id == "PROD_001"
        assert product.demand == 100.5
        assert product.unit_cost == 45.0
        assert product.order_cost == 200.0
        assert product.lead_time == 7

    def test_negative_demand_rejected(self):
        """Test that negative demand is rejected."""
        with pytest.raises(PydanticValidationError):
            ProductData(
                product_id="PROD_001",
                demand=-10.0,
                unit_cost=45.0,
                order_cost=200.0,
                lead_time=7
            )

    def test_zero_unit_cost_rejected(self):
        """Test that zero or negative unit cost is rejected."""
        with pytest.raises(PydanticValidationError):
            ProductData(
                product_id="PROD_001",
                demand=100.0,
                unit_cost=0.0,
                order_cost=200.0,
                lead_time=7
            )

    def test_product_id_whitespace_trimmed(self):
        """Test that product ID whitespace is trimmed."""
        product = ProductData(
            product_id="  PROD_001  ",
            demand=100.0,
            unit_cost=45.0,
            order_cost=200.0,
            lead_time=7
        )

        assert product.product_id == "PROD_001"

    def test_empty_product_id_rejected(self):
        """Test that empty product ID is rejected."""
        with pytest.raises(PydanticValidationError):
            ProductData(
                product_id="   ",
                demand=100.0,
                unit_cost=45.0,
                order_cost=200.0,
                lead_time=7
            )


@pytest.mark.unit
class TestDataFrameValidation:
    """Test suite for DataFrame validation functions."""

    def test_validate_product_dataframe_success(self, sample_product_data):
        """Test validating a correct DataFrame."""
        # Should not raise
        result = validate_product_dataframe(sample_product_data)
        assert isinstance(result, pd.DataFrame)

    def test_validate_missing_columns(self):
        """Test validation fails with missing columns."""
        df = pd.DataFrame({
            'product_id': ['PROD_001'],
            'demand': [100.0]
            # Missing: unit_cost, order_cost, lead_time
        })

        with pytest.raises(ValidationError) as exc_info:
            validate_product_dataframe(df)

        assert "Missing required columns" in str(exc_info.value)

    def test_validate_empty_dataframe(self):
        """Test validation fails with empty DataFrame."""
        df = pd.DataFrame(columns=[
            'product_id', 'demand', 'unit_cost', 'order_cost', 'lead_time'
        ])

        with pytest.raises(ValidationError) as exc_info:
            validate_product_dataframe(df)

        assert "empty" in str(exc_info.value).lower()


@pytest.mark.unit
class TestDataQualityChecking:
    """Test suite for data quality checking."""

    def test_clean_data_has_no_issues(self, sample_product_data):
        """Test that clean data reports no issues."""
        report = check_data_quality(sample_product_data)

        assert isinstance(report, dict)
        assert 'has_issues' in report
        # Clean sample data should have minimal issues
        # (might have some outliers due to random generation)

    def test_missing_values_detected(self):
        """Test that missing values are detected."""
        df = pd.DataFrame({
            'product_id': ['PROD_001', 'PROD_002', 'PROD_003'],
            'demand': [100.0, None, 200.0],
            'unit_cost': [45.0, 50.0, None],
            'order_cost': [200.0, 200.0, 200.0],
            'lead_time': [7, 7, 7]
        })

        report = check_data_quality(df)

        assert report['has_issues'] == True
        assert 'missing_values' in report
        assert report['total_issues'] > 0

    def test_duplicates_detected(self):
        """Test that duplicate product IDs are detected."""
        df = pd.DataFrame({
            'product_id': ['PROD_001', 'PROD_001', 'PROD_002'],
            'demand': [100.0, 150.0, 200.0],
            'unit_cost': [45.0, 50.0, 55.0],
            'order_cost': [200.0, 200.0, 200.0],
            'lead_time': [7, 7, 7]
        })

        report = check_data_quality(df)

        assert report['has_issues'] == True
        assert 'duplicates' in report
        assert len(report['duplicates']) > 0

    def test_negative_values_detected(self):
        """Test that negative values in critical columns are detected."""
        df = pd.DataFrame({
            'product_id': ['PROD_001', 'PROD_002'],
            'demand': [-100.0, 200.0],  # Negative demand
            'unit_cost': [45.0, 50.0],
            'order_cost': [200.0, 200.0],
            'lead_time': [7, 7]
        })

        report = check_data_quality(df)

        assert report['has_issues'] == True
        assert 'negative_values' in report


@pytest.mark.unit
@pytest.mark.fast
class TestSanitization:
    """Test suite for input sanitization functions."""

    def test_sanitize_product_id_removes_dangerous_chars(self):
        """Test that dangerous characters are removed from product ID."""
        dangerous_id = "PROD<script>alert('xss')</script>_001"
        sanitized = sanitize_product_id(dangerous_id)

        assert '<' not in sanitized
        assert '>' not in sanitized
        assert 'script' in sanitized  # Word remains but tags removed

    def test_sanitize_product_id_limits_length(self):
        """Test that product ID length is limited."""
        long_id = "P" * 100
        sanitized = sanitize_product_id(long_id)

        assert len(sanitized) <= 50

    def test_sanitize_dataframe(self, sample_product_data):
        """Test DataFrame sanitization."""
        # Add some issues
        df = sample_product_data.copy()
        df.loc[0, 'product_id'] = "<script>alert('xss')</script>"
        df.loc[1, 'demand'] = None

        sanitized = sanitize_dataframe(df)

        # Check that dangerous chars are removed
        assert '<script>' not in sanitized.loc[0, 'product_id']


@pytest.mark.unit
class TestOptimizationResultsValidation:
    """Test suite for optimization results validation."""

    def test_valid_optimization_results(self):
        """Test validation passes for reasonable results."""
        # Should not raise
        validate_optimization_results(
            baseline_cost=100000.0,
            optimized_cost=80000.0,
            service_level=0.95
        )

    def test_negative_baseline_cost_rejected(self):
        """Test that negative baseline cost is rejected."""
        with pytest.raises(ValidationError):
            validate_optimization_results(
                baseline_cost=-100000.0,
                optimized_cost=80000.0,
                service_level=0.95
            )

    def test_optimization_made_costs_worse(self):
        """Test that significant cost increase is flagged."""
        with pytest.raises(ValidationError) as exc_info:
            validate_optimization_results(
                baseline_cost=100000.0,
                optimized_cost=120000.0,  # 20% increase
                service_level=0.95
            )

        assert "increased costs" in str(exc_info.value).lower()

    def test_unrealistic_improvement_rejected(self):
        """Test that unrealistically high improvement is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_optimization_results(
                baseline_cost=100000.0,
                optimized_cost=10000.0,  # 90% improvement - unrealistic
                service_level=0.95
            )

        assert "unrealistic" in str(exc_info.value).lower()
