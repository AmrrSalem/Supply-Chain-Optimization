"""
Constants Module
================

This module defines all constants used throughout the Supply Chain Optimization system.
Extract magic numbers and hardcoded values here for easy configuration and maintenance.

All constants are in SCREAMING_SNAKE_CASE.
"""

# ============================================================================
# DEFAULT OPTIMIZATION PARAMETERS
# ============================================================================

# Default service level target (0-1, where 1 = 100%)
DEFAULT_SERVICE_LEVEL = 0.95

# Default annual holding cost rate as fraction of unit value
DEFAULT_HOLDING_COST_RATE = 0.25

# Default shortage cost multiplier (relative to unit cost)
# Shortage cost = unit_cost * SHORTAGE_COST_MULTIPLIER
DEFAULT_SHORTAGE_COST_MULTIPLIER = 2.0

# ============================================================================
# DATA GENERATION PARAMETERS
# ============================================================================

# Default number of products for sample data generation
DEFAULT_N_PRODUCTS = 50

# Number of time periods (days) in sample data
DEFAULT_N_PERIODS = 365

# Random seed for reproducible sample data generation
DEFAULT_RANDOM_SEED = 42

# Demand generation parameters
MIN_BASE_DEMAND = 3  # Lognormal distribution mean (log scale)
MAX_BASE_DEMAND = 1
MIN_SEASONALITY = 0.1  # Minimum seasonal variation
MAX_SEASONALITY = 0.3  # Maximum seasonal variation
MIN_TREND = -0.001  # Minimum growth/decline trend
MAX_TREND = 0.001  # Maximum growth/decline trend
NOISE_STD = 0.2  # Standard deviation of demand noise

# Cost parameters for sample data
MIN_UNIT_COST = 10.0
MAX_UNIT_COST = 500.0
MIN_ORDER_COST = 50.0
MAX_ORDER_COST = 500.0
MIN_LEAD_TIME = 1  # days
MAX_LEAD_TIME = 21  # days

# Stockout probability range for sample data
MIN_STOCKOUT_PROB = 0.02
MAX_STOCKOUT_PROB = 0.08

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

# Stochastic optimization parameters
DEFAULT_DEMAND_SCENARIOS = 1000  # Number of scenarios to simulate
STOCHASTIC_SCENARIOS_REDUCED = 10  # Reduced scenarios for efficiency (/ 100)
MAX_OPTIMIZATION_ITERATIONS = 200  # Maximum iterations for optimization solver

# Multi-product optimization parameters
DEFAULT_BUDGET_MULTIPLIER = 0.8  # Budget = baseline * this multiplier
MIN_ORDER_QUANTITY_WEEKS = 52  # Minimum order quantity (as weeks of demand)

# Safety stock calculation
SAFETY_STOCK_MULTIPLIER = 2  # Multiplier for minimum safety stock constraint

# Penalty for invalid optimization parameters
INVALID_PARAM_PENALTY = 1e10

# Penalty for service level violations (stochastic optimization)
SERVICE_LEVEL_VIOLATION_PENALTY = 1e6

# ============================================================================
# MACHINE LEARNING PARAMETERS
# ============================================================================

# ML forecasting parameters
DEFAULT_FORECAST_HORIZON = 30  # Days ahead to forecast
DEFAULT_N_ESTIMATORS = 100  # Number of trees in Random Forest
ML_RANDOM_STATE = 42  # Random state for ML models
ML_TRAIN_TEST_SPLIT = 0.8  # Train/test split ratio (80/20)
ML_MIN_DATA_POINTS = 100  # Minimum data points required for ML training
ML_MIN_DATA_AFTER_CLEANING = 50  # Minimum data points after removing NaN

# Feature engineering parameters
ROLLING_WINDOW_SHORT = 7  # Short-term rolling window (days)
ROLLING_WINDOW_LONG = 30  # Long-term rolling window (days)
LAG_1_DAY = 1  # 1-day lag for features
LAG_7_DAYS = 7  # 7-day lag for features

# ML forecasting - products to process (for demo)
ML_DEMO_PRODUCT_LIMIT = 10  # Limit ML forecasting to first N products

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Figure size for matplotlib plots
FIGURE_SIZE = (18, 12)
FIGURE_ROWS = 2
FIGURE_COLS = 3

# Histogram parameters
HISTOGRAM_BINS = 20
HISTOGRAM_ALPHA = 0.7

# Color schemes
COLOR_BASELINE = 'lightblue'
COLOR_OPTIMIZED = 'lightgreen'
COLOR_COMPARISON = ['skyblue', 'lightgreen', 'orange']
COLOR_PIE_CHART = ['lightcoral', 'lightskyblue']

# ============================================================================
# STREAMLIT APP PARAMETERS
# ============================================================================

# Service level slider range
STREAMLIT_SERVICE_LEVEL_MIN = 85  # Minimum service level (%)
STREAMLIT_SERVICE_LEVEL_MAX = 99  # Maximum service level (%)
STREAMLIT_SERVICE_LEVEL_DEFAULT = 95  # Default service level (%)
STREAMLIT_SERVICE_LEVEL_STEP = 1  # Step size for slider

# Holding cost rate slider range
STREAMLIT_HOLDING_COST_MIN = 10  # Minimum holding cost rate (%)
STREAMLIT_HOLDING_COST_MAX = 50  # Maximum holding cost rate (%)
STREAMLIT_HOLDING_COST_DEFAULT = 25  # Default holding cost rate (%)
STREAMLIT_HOLDING_COST_STEP = 5  # Step size for slider

# Number of products options
STREAMLIT_N_PRODUCTS_OPTIONS = [10, 25, 50, 100]
STREAMLIT_N_PRODUCTS_DEFAULT_INDEX = 1  # Index in options list (25 products)

# Budget constraint slider range
STREAMLIT_BUDGET_MIN = 60  # Minimum budget (% of baseline)
STREAMLIT_BUDGET_MAX = 120  # Maximum budget (% of baseline)
STREAMLIT_BUDGET_DEFAULT = 80  # Default budget (% of baseline)
STREAMLIT_BUDGET_STEP = 5  # Step size for slider

# Streamlit port and address
STREAMLIT_PORT = 8501
STREAMLIT_ADDRESS = "0.0.0.0"

# ============================================================================
# REPORTING PARAMETERS
# ============================================================================

# Top opportunities to display
TOP_OPPORTUNITIES_COUNT = 10

# Sample products to display in visualizations
SAMPLE_PRODUCTS_DISPLAY = 10  # Show first N products in charts

# ============================================================================
# DATA VALIDATION PARAMETERS
# ============================================================================

# Valid ranges for input validation
MIN_SERVICE_LEVEL = 0.5  # 50%
MAX_SERVICE_LEVEL = 0.999  # 99.9%

MIN_HOLDING_COST_RATE = 0.01  # 1%
MAX_HOLDING_COST_RATE = 1.0  # 100%

MIN_UNIT_COST_VALIDATION = 0.01  # Minimum positive value
MAX_UNIT_COST_VALIDATION = 1_000_000  # Maximum reasonable value

MIN_DEMAND = 0  # Demand cannot be negative
MAX_DEMAND = 1_000_000  # Maximum reasonable daily demand

MIN_LEAD_TIME_VALIDATION = 0  # Same-day delivery possible
MAX_LEAD_TIME_VALIDATION = 365  # Maximum 1 year lead time

MIN_N_PRODUCTS = 1
MAX_N_PRODUCTS = 10000

# ============================================================================
# PERFORMANCE PARAMETERS
# ============================================================================

# Cache settings
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)

# Parallel processing
USE_PARALLEL_PROCESSING = True
N_JOBS = -1  # -1 means use all available CPU cores

# Progress bar settings
PROGRESS_BAR_ENABLED = True

# ============================================================================
# FILE SYSTEM PARAMETERS
# ============================================================================

# Directory paths
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"
REPORTS_DIR = "reports"
TEMPLATES_DIR = "templates"

# File extensions
CSV_EXTENSION = ".csv"
EXCEL_EXTENSION = ".xlsx"
PICKLE_EXTENSION = ".pkl"

# ============================================================================
# LOGGING PARAMETERS
# ============================================================================

# Default log level
DEFAULT_LOG_LEVEL = "INFO"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log file settings
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5

# ============================================================================
# DATABASE PARAMETERS (Future use)
# ============================================================================

# Connection pool settings
DB_POOL_SIZE = 5
DB_MAX_OVERFLOW = 10
DB_POOL_TIMEOUT = 30  # seconds

# ============================================================================
# MONITORING PARAMETERS (Future use)
# ============================================================================

# Health check
HEALTH_CHECK_INTERVAL = 60  # seconds

# Metrics
METRICS_PORT = 9090

# ============================================================================
# SECURITY PARAMETERS
# ============================================================================

# Rate limiting
RATE_LIMIT_CALLS = 10  # Maximum calls
RATE_LIMIT_PERIOD = 60  # Per period in seconds

# File upload limits
MAX_FILE_SIZE_MB = 100  # Maximum file size for uploads

# ============================================================================
# VERSION INFORMATION
# ============================================================================

VERSION = "0.1.0"
VERSION_INFO = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "status": "alpha"
}
