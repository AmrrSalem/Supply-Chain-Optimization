"""
Input Validation Module
=======================

This module provides input validation using Pydantic models.

Features:
    - Type validation
    - Range validation
    - Business rule validation
    - Data quality checks
    - Automatic error messages

Usage:
    from utils.validators import OptimizationParams, ProductData

    # Validate optimization parameters
    params = OptimizationParams(service_level=0.95, holding_cost_rate=0.25)

    # Validate product data
    product = ProductData(
        product_id="PROD_001",
        demand=100.5,
        unit_cost=45.0,
        order_cost=200.0,
        lead_time=7
    )
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator, root_validator
import pandas as pd
import numpy as np

from config.constants import (
    MIN_SERVICE_LEVEL,
    MAX_SERVICE_LEVEL,
    MIN_HOLDING_COST_RATE,
    MAX_HOLDING_COST_RATE,
    MIN_DEMAND,
    MIN_UNIT_COST_VALIDATION,
    MAX_UNIT_COST_VALIDATION,
    MIN_LEAD_TIME_VALIDATION,
    MAX_LEAD_TIME_VALIDATION,
    MIN_N_PRODUCTS,
    MAX_N_PRODUCTS,
)
from utils.exceptions import ValidationError, DataQualityError


# ============================================================================
# OPTIMIZATION PARAMETER MODELS
# ============================================================================

class OptimizationParams(BaseModel):
    """
    Validation model for optimization parameters.

    Attributes
    ----------
    service_level : float
        Target service level (0.5 to 0.999)
    holding_cost_rate : float
        Annual holding cost rate (0.01 to 1.0)
    n_products : int, optional
        Number of products to optimize (1 to 10000)
    demand_scenarios : int, optional
        Number of demand scenarios for stochastic optimization
    """

    service_level: float = Field(
        ...,
        ge=MIN_SERVICE_LEVEL,
        le=MAX_SERVICE_LEVEL,
        description="Target service level (probability of not stocking out)"
    )

    holding_cost_rate: float = Field(
        ...,
        ge=MIN_HOLDING_COST_RATE,
        le=MAX_HOLDING_COST_RATE,
        description="Annual holding cost as fraction of unit value"
    )

    n_products: Optional[int] = Field(
        default=None,
        ge=MIN_N_PRODUCTS,
        le=MAX_N_PRODUCTS,
        description="Number of products to optimize"
    )

    demand_scenarios: Optional[int] = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of demand scenarios for stochastic optimization"
    )

    @validator('service_level')
    def validate_service_level(cls, v):
        """Validate service level is reasonable."""
        if v < 0.5:
            raise ValueError(
                f"Service level {v*100:.1f}% is too low. "
                f"Minimum is {MIN_SERVICE_LEVEL*100:.1f}%"
            )
        if v > 0.999:
            raise ValueError(
                f"Service level {v*100:.1f}% is too high. "
                f"Maximum is {MAX_SERVICE_LEVEL*100:.1f}%"
            )
        return v

    @validator('holding_cost_rate')
    def validate_holding_cost_rate(cls, v):
        """Validate holding cost rate is reasonable."""
        if v > 1.0:
            raise ValueError(
                f"Holding cost rate {v*100:.1f}% cannot exceed 100%"
            )
        return v

    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = 'forbid'


# ============================================================================
# PRODUCT DATA MODELS
# ============================================================================

class ProductData(BaseModel):
    """
    Validation model for individual product data.

    Attributes
    ----------
    product_id : str
        Unique product identifier
    demand : float
        Demand quantity (must be non-negative)
    unit_cost : float
        Cost per unit (must be positive)
    order_cost : float
        Fixed cost per order (must be positive)
    lead_time : int
        Lead time in days (0 to 365)
    demand_std : float, optional
        Standard deviation of demand
    """

    product_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique product identifier"
    )

    demand: float = Field(
        ...,
        ge=MIN_DEMAND,
        description="Demand quantity"
    )

    unit_cost: float = Field(
        ...,
        gt=MIN_UNIT_COST_VALIDATION,
        le=MAX_UNIT_COST_VALIDATION,
        description="Cost per unit"
    )

    order_cost: float = Field(
        ...,
        gt=0.0,
        description="Fixed cost per order"
    )

    lead_time: int = Field(
        ...,
        ge=MIN_LEAD_TIME_VALIDATION,
        le=MAX_LEAD_TIME_VALIDATION,
        description="Lead time in days"
    )

    demand_std: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Standard deviation of demand"
    )

    @validator('product_id')
    def validate_product_id(cls, v):
        """Validate product ID format."""
        if not v or v.isspace():
            raise ValueError("Product ID cannot be empty or whitespace")
        return v.strip()

    @validator('demand')
    def validate_demand(cls, v):
        """Validate demand is non-negative."""
        if v < 0:
            raise ValueError(f"Demand cannot be negative: {v}")
        return v

    @validator('demand_std')
    def validate_demand_std(cls, v, values):
        """Validate demand standard deviation is reasonable."""
        if v is not None and 'demand' in values:
            # Standard deviation shouldn't be larger than mean demand
            if v > values['demand'] * 2:
                raise ValueError(
                    f"Demand std ({v:.2f}) is suspiciously large "
                    f"compared to mean demand ({values['demand']:.2f})"
                )
        return v

    class Config:
        """Pydantic config."""
        validate_assignment = True


# ============================================================================
# BULK DATA VALIDATION
# ============================================================================

class ProductDataset(BaseModel):
    """
    Validation model for bulk product data.

    Attributes
    ----------
    products : List[ProductData]
        List of product data records
    """

    products: List[ProductData]

    @validator('products')
    def validate_unique_product_ids(cls, v):
        """Validate that product IDs are unique."""
        product_ids = [p.product_id for p in v]
        duplicates = set([pid for pid in product_ids if product_ids.count(pid) > 1])

        if duplicates:
            raise ValueError(
                f"Duplicate product IDs found: {', '.join(duplicates)}"
            )

        return v

    @validator('products')
    def validate_minimum_products(cls, v):
        """Validate minimum number of products."""
        if len(v) < 1:
            raise ValueError("At least one product is required")
        return v


# ============================================================================
# DATAFRAME VALIDATION FUNCTIONS
# ============================================================================

def validate_product_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a DataFrame containing product data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with product data

    Returns
    -------
    pd.DataFrame
        Validated DataFrame

    Raises
    ------
    ValidationError
        If validation fails
    """
    required_columns = ['product_id', 'demand', 'unit_cost', 'order_cost', 'lead_time']

    # Check for required columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValidationError(
            f"Missing required columns: {', '.join(missing)}",
            details={"missing_columns": list(missing)}
        )

    # Check for empty DataFrame
    if len(df) == 0:
        raise ValidationError("DataFrame is empty")

    # Validate each row
    errors = []
    for idx, row in df.iterrows():
        try:
            ProductData(
                product_id=row['product_id'],
                demand=row['demand'],
                unit_cost=row['unit_cost'],
                order_cost=row['order_cost'],
                lead_time=row['lead_time'],
                demand_std=row.get('demand_std', None)
            )
        except Exception as e:
            errors.append(f"Row {idx}: {str(e)}")

    if errors:
        raise ValidationError(
            f"Data validation failed for {len(errors)} rows",
            details={"errors": errors[:10]}  # Show first 10 errors
        )

    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Check data quality and return report.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check

    Returns
    -------
    dict
        Data quality report with issues found

    Examples
    --------
    >>> report = check_data_quality(product_df)
    >>> if report['has_issues']:
    ...     print(f"Found {report['total_issues']} issues")
    """
    issues = {
        'has_issues': False,
        'total_issues': 0,
        'missing_values': {},
        'outliers': {},
        'duplicates': [],
        'negative_values': {},
        'warnings': []
    }

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues['has_issues'] = True
        issues['missing_values'] = missing[missing > 0].to_dict()
        issues['total_issues'] += sum(missing)

    # Check for duplicates
    if 'product_id' in df.columns:
        duplicates = df[df.duplicated('product_id', keep=False)]['product_id'].unique()
        if len(duplicates) > 0:
            issues['has_issues'] = True
            issues['duplicates'] = list(duplicates)
            issues['total_issues'] += len(duplicates)

    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['demand', 'unit_cost', 'order_cost']:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues['has_issues'] = True
                issues['negative_values'][col] = negative_count
                issues['total_issues'] += negative_count

    # Check for outliers (using IQR method)
    for col in ['demand', 'unit_cost']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outlier_count > 0:
                issues['outliers'][col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(df) * 100)
                }
                if outlier_count / len(df) > 0.05:  # More than 5% outliers
                    issues['has_issues'] = True
                    issues['total_issues'] += outlier_count

    # Check for suspiciously low variance
    for col in ['demand']:
        if col in df.columns:
            cv = df[col].std() / df[col].mean() if df[col].mean() > 0 else 0
            if cv < 0.01:  # Very low variance
                issues['warnings'].append(
                    f"{col} has very low variance (CV={cv:.4f}), "
                    f"data may be artificial or incorrect"
                )

    return issues


# ============================================================================
# SANITIZATION FUNCTIONS
# ============================================================================

def sanitize_product_id(product_id: str) -> str:
    """
    Sanitize product ID to prevent injection attacks.

    Parameters
    ----------
    product_id : str
        Raw product ID

    Returns
    -------
    str
        Sanitized product ID
    """
    # Remove dangerous characters
    dangerous_chars = ['<', '>', '&', '"', "'", '\\', '/', ';', '--']
    sanitized = product_id

    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')

    # Limit length
    sanitized = sanitized[:50]

    # Strip whitespace
    sanitized = sanitized.strip()

    if not sanitized:
        raise ValidationError("Product ID is empty after sanitization")

    return sanitized


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize DataFrame by removing/fixing problematic data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame

    Returns
    -------
    pd.DataFrame
        Sanitized DataFrame
    """
    df = df.copy()

    # Sanitize product IDs
    if 'product_id' in df.columns:
        df['product_id'] = df['product_id'].apply(sanitize_product_id)

    # Remove rows with all NaN
    df = df.dropna(how='all')

    # Fill NaN in numeric columns with 0 (or appropriate default)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['demand', 'unit_cost', 'order_cost']:
            # Don't fill these - they must be valid
            pass
        else:
            df[col] = df[col].fillna(0)

    # Remove duplicate product IDs (keep first)
    if 'product_id' in df.columns:
        df = df.drop_duplicates(subset='product_id', keep='first')

    # Ensure numeric types
    for col in ['demand', 'unit_cost', 'order_cost', 'lead_time']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_optimization_results(
    baseline_cost: float,
    optimized_cost: float,
    service_level: float
) -> None:
    """
    Validate optimization results are reasonable.

    Parameters
    ----------
    baseline_cost : float
        Baseline total cost
    optimized_cost : float
        Optimized total cost
    service_level : float
        Target service level

    Raises
    ------
    ValidationError
        If results are unreasonable
    """
    if baseline_cost <= 0:
        raise ValidationError(
            "Baseline cost must be positive",
            details={"baseline_cost": baseline_cost}
        )

    if optimized_cost <= 0:
        raise ValidationError(
            "Optimized cost must be positive",
            details={"optimized_cost": optimized_cost}
        )

    # Check if optimization made things worse by more than 10%
    if optimized_cost > baseline_cost * 1.1:
        raise ValidationError(
            "Optimization increased costs by more than 10%",
            details={
                "baseline_cost": baseline_cost,
                "optimized_cost": optimized_cost,
                "increase_pct": ((optimized_cost - baseline_cost) / baseline_cost * 100)
            }
        )

    # Check if improvement is suspiciously high (> 80%)
    improvement = (baseline_cost - optimized_cost) / baseline_cost
    if improvement > 0.8:
        raise ValidationError(
            "Optimization improvement > 80% seems unrealistic",
            details={
                "baseline_cost": baseline_cost,
                "optimized_cost": optimized_cost,
                "improvement_pct": improvement * 100
            }
        )
