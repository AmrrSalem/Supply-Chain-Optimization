"""
Pytest Configuration and Shared Fixtures
========================================

This module contains pytest configuration and shared fixtures that can be used
across all test modules.

Fixtures:
    sample_data: Sample inventory data for testing
    optimizer: Inventory optimizer instance
    mock_logger: Mocked logger for testing
    temp_data_file: Temporary CSV file with test data
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import logging

# Import modules to test
from sc_optimization import InventoryOptimizer


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual functions"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for workflows"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time"
    )


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_product_data():
    """
    Fixture providing sample product data for testing.

    Returns
    -------
    pd.DataFrame
        Sample product data with required columns
    """
    np.random.seed(42)

    data = {
        'product_id': [f'TEST_PROD_{i:03d}' for i in range(10)],
        'annual_demand': np.random.uniform(1000, 5000, 10),
        'demand_std': np.random.uniform(50, 200, 10),
        'unit_cost': np.random.uniform(10, 100, 10),
        'order_cost': np.random.uniform(50, 200, 10),
        'lead_time': np.random.randint(1, 15, 10),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_daily_demand_data():
    """
    Fixture providing sample daily demand time series data.

    Returns
    -------
    pd.DataFrame
        Daily demand data for multiple products
    """
    np.random.seed(42)

    products = []
    n_products = 5
    n_days = 100

    for i in range(n_products):
        base_demand = np.random.lognormal(3, 0.5)

        for day in range(n_days):
            products.append({
                'product_id': f'TEST_PROD_{i:03d}',
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=day),
                'demand': max(0, np.random.normal(base_demand, base_demand * 0.2)),
                'unit_cost': 50.0,
                'order_cost': 100.0,
                'lead_time': 7,
                'stockout': 0
            })

    return pd.DataFrame(products)


@pytest.fixture
def sample_eoq_params():
    """
    Fixture providing sample EOQ calculation parameters.

    Returns
    -------
    dict
        Dictionary with EOQ parameters
    """
    return {
        'annual_demand': 1000.0,
        'order_cost': 50.0,
        'unit_cost': 10.0,
        'holding_cost_rate': 0.25,
        'lead_time': 7,
        'demand_std': 50.0,
        'service_level': 0.95
    }


# ============================================================================
# OPTIMIZER FIXTURES
# ============================================================================

@pytest.fixture
def optimizer():
    """
    Fixture providing a basic InventoryOptimizer instance.

    Returns
    -------
    InventoryOptimizer
        Optimizer instance with default parameters
    """
    return InventoryOptimizer(service_level=0.95, holding_cost_rate=0.25)


@pytest.fixture
def optimizer_with_data(optimizer, sample_daily_demand_data):
    """
    Fixture providing an optimizer instance with loaded data.

    Returns
    -------
    InventoryOptimizer
        Optimizer with sample data loaded
    """
    optimizer.data = sample_daily_demand_data
    optimizer.preprocess_data()
    return optimizer


@pytest.fixture
def optimizer_with_baseline(optimizer_with_data):
    """
    Fixture providing an optimizer with baseline calculations complete.

    Returns
    -------
    InventoryOptimizer
        Optimizer with baseline EOQ calculated
    """
    optimizer_with_data.calculate_eoq_baseline()
    return optimizer_with_data


# ============================================================================
# FILE FIXTURES
# ============================================================================

@pytest.fixture
def temp_csv_file(sample_product_data, tmp_path):
    """
    Fixture providing a temporary CSV file with sample data.

    Parameters
    ----------
    sample_product_data : pd.DataFrame
        Sample data to write
    tmp_path : Path
        Pytest temporary directory

    Returns
    -------
    Path
        Path to temporary CSV file
    """
    file_path = tmp_path / "test_data.csv"
    sample_product_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def temp_excel_file(sample_product_data, tmp_path):
    """
    Fixture providing a temporary Excel file with sample data.

    Parameters
    ----------
    sample_product_data : pd.DataFrame
        Sample data to write
    tmp_path : Path
        Pytest temporary directory

    Returns
    -------
    Path
        Path to temporary Excel file
    """
    file_path = tmp_path / "test_data.xlsx"
    sample_product_data.to_excel(file_path, index=False, engine='openpyxl')
    return file_path


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_logger(mocker):
    """
    Fixture providing a mocked logger.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Pytest mocker fixture

    Returns
    -------
    MagicMock
        Mocked logger object
    """
    mock = mocker.MagicMock(spec=logging.Logger)
    return mock


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def test_config(monkeypatch):
    """
    Fixture providing test configuration overrides.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture

    Returns
    -------
    dict
        Test configuration dictionary
    """
    # Set test environment variables
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CACHE_ENABLED", "false")

    return {
        'environment': 'development',
        'debug': True,
        'log_level': 'DEBUG',
        'cache_enabled': False
    }


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def reset_random_seed():
    """
    Fixture to reset random seed before each test for reproducibility.
    """
    np.random.seed(42)
    yield
    # Cleanup if needed


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Session-wide fixture to set up test environment.
    """
    # Setup code
    print("\n=== Setting up test environment ===")

    yield

    # Teardown code
    print("\n=== Tearing down test environment ===")


# ============================================================================
# PARAMETRIZE HELPERS
# ============================================================================

# Common parameter sets for parametrized tests
SERVICE_LEVELS = [0.90, 0.95, 0.98, 0.99]
HOLDING_COST_RATES = [0.15, 0.20, 0.25, 0.30]
PRODUCT_COUNTS = [10, 25, 50]


# ============================================================================
# ASSERTION HELPERS
# ============================================================================

def assert_dataframe_equal(df1, df2, **kwargs):
    """
    Helper function to assert DataFrame equality with better error messages.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame
    df2 : pd.DataFrame
        Second DataFrame
    **kwargs
        Additional arguments for pd.testing.assert_frame_equal
    """
    try:
        pd.testing.assert_frame_equal(df1, df2, **kwargs)
    except AssertionError as e:
        print(f"\nDataFrame 1:\n{df1}")
        print(f"\nDataFrame 2:\n{df2}")
        raise e


def assert_optimization_improved(baseline_cost, optimized_cost, min_improvement=0.01):
    """
    Helper function to assert that optimization improved costs.

    Parameters
    ----------
    baseline_cost : float
        Baseline total cost
    optimized_cost : float
        Optimized total cost
    min_improvement : float, default=0.01
        Minimum improvement percentage (1%)
    """
    improvement = (baseline_cost - optimized_cost) / baseline_cost

    assert improvement >= min_improvement, (
        f"Optimization did not improve enough. "
        f"Baseline: ${baseline_cost:,.2f}, "
        f"Optimized: ${optimized_cost:,.2f}, "
        f"Improvement: {improvement * 100:.2f}% "
        f"(minimum: {min_improvement * 100:.2f}%)"
    )


# ============================================================================
# PYTEST HOOKS
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection.

    Add markers automatically based on test file location.
    """
    for item in items:
        # Add 'unit' marker to tests in tests/unit/
        if "tests/unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add 'integration' marker to tests in tests/integration/
        elif "tests/integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


def pytest_report_header(config):
    """Add custom header to pytest report."""
    return [
        "Supply Chain Optimization Test Suite",
        "======================================",
        f"Test directory: {config.rootdir / 'tests'}",
    ]
