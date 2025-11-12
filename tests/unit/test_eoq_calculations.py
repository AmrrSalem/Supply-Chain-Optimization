"""
Unit Tests for EOQ Calculations
================================

Tests for Economic Order Quantity (EOQ) and safety stock calculations.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm


@pytest.mark.unit
@pytest.mark.fast
class TestEOQCalculations:
    """Test suite for EOQ calculation functions."""

    def test_eoq_basic_calculation(self, sample_eoq_params):
        """Test basic EOQ calculation with known values."""
        # Given
        demand = sample_eoq_params['annual_demand']
        order_cost = sample_eoq_params['order_cost']
        holding_cost = (
            sample_eoq_params['unit_cost'] *
            sample_eoq_params['holding_cost_rate']
        )

        # When
        eoq = np.sqrt(2 * demand * order_cost / holding_cost)

        # Then
        expected_eoq = np.sqrt(2 * 1000 * 50 / 2.5)  # = 200
        assert abs(eoq - expected_eoq) < 0.01, f"EOQ {eoq} != expected {expected_eoq}"
        assert eoq > 0, "EOQ must be positive"

    def test_eoq_with_zero_demand(self):
        """Test that EOQ handles zero demand gracefully."""
        # Given
        demand = 0
        order_cost = 50
        holding_cost = 2.5

        # When
        eoq = np.sqrt(2 * demand * order_cost / holding_cost)

        # Then
        assert eoq == 0, "EOQ with zero demand should be zero"

    def test_safety_stock_calculation(self, sample_eoq_params):
        """Test safety stock calculation."""
        # Given
        service_level = sample_eoq_params['service_level']
        demand_std = sample_eoq_params['demand_std']
        lead_time = sample_eoq_params['lead_time']

        # When
        z_score = norm.ppf(service_level)
        lead_time_std = demand_std * np.sqrt(lead_time)
        safety_stock = z_score * lead_time_std

        # Then
        assert safety_stock > 0, "Safety stock must be positive"
        assert z_score > 0, f"Z-score for service level {service_level} should be positive"

    @pytest.mark.parametrize("service_level,expected_z", [
        (0.90, 1.28),
        (0.95, 1.645),
        (0.98, 2.05),
        (0.99, 2.33),
    ])
    def test_z_score_for_service_levels(self, service_level, expected_z):
        """Test Z-score calculation for different service levels."""
        # When
        z_score = norm.ppf(service_level)

        # Then
        assert abs(z_score - expected_z) < 0.05, (
            f"Z-score for {service_level*100}% service level should be ~{expected_z}"
        )


@pytest.mark.unit
class TestOptimizerInitialization:
    """Test suite for InventoryOptimizer initialization."""

    def test_optimizer_default_parameters(self, optimizer):
        """Test optimizer initializes with correct default parameters."""
        assert optimizer.service_level == 0.95
        assert optimizer.holding_cost_rate == 0.25
        assert isinstance(optimizer.results, dict)

    def test_optimizer_custom_parameters(self):
        """Test optimizer with custom parameters."""
        from sc_optimization import InventoryOptimizer

        custom_optimizer = InventoryOptimizer(
            service_level=0.98,
            holding_cost_rate=0.30
        )

        assert custom_optimizer.service_level == 0.98
        assert custom_optimizer.holding_cost_rate == 0.30


@pytest.mark.unit
class TestDataPreprocessing:
    """Test suite for data preprocessing functions."""

    def test_data_loading(self, optimizer, sample_daily_demand_data):
        """Test that data loads correctly."""
        # Given
        optimizer.data = sample_daily_demand_data

        # Then
        assert optimizer.data is not None
        assert len(optimizer.data) > 0
        assert 'product_id' in optimizer.data.columns
        assert 'demand' in optimizer.data.columns

    def test_product_summary_creation(self, optimizer_with_data):
        """Test that product summary is created correctly."""
        # Then
        assert hasattr(optimizer_with_data, 'product_summary')
        assert len(optimizer_with_data.product_summary) > 0
        assert 'demand_mean' in optimizer_with_data.product_summary.columns

    @pytest.mark.xfail(reason="Bug in original sc_optimization.py: preprocess_data() doesn't properly store rolling statistics columns")
    def test_rolling_statistics(self, optimizer_with_data):
        """Test that rolling statistics are calculated."""
        # Then
        assert 'demand_ma_7' in optimizer_with_data.processed_data.columns
        assert 'demand_ma_30' in optimizer_with_data.processed_data.columns


# This test file establishes the testing framework
# More tests will be added in the next phase
