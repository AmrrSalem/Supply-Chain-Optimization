"""
Data Loading Module
===================

This module provides utilities for loading inventory data from various sources.

Supported Formats:
    - CSV files
    - Excel files (.xlsx, .xls)
    - Pandas DataFrames
    - JSON files
    - Future: Database connections

Features:
    - Automatic schema validation
    - Data quality checking
    - Missing value handling
    - Type conversion
    - Error reporting

Usage:
    from utils.data_loader import DataLoader

    # Load from CSV
    loader = DataLoader()
    data = loader.load_from_csv('inventory_data.csv')

    # Load from Excel
    data = loader.load_from_excel('inventory_data.xlsx')

    # Validate and clean
    clean_data = loader.validate_and_clean(data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict
import json

from utils.exceptions import (
    DataLoadError,
    DataFormatError,
    MissingDataError,
    ValidationError
)
from utils.validators import (
    validate_product_dataframe,
    check_data_quality,
    sanitize_dataframe
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# DATA LOADER CLASS
# ============================================================================

class DataLoader:
    """
    Data loader for inventory optimization data.

    This class handles loading data from various file formats and validates
    the data structure and quality.

    Attributes
    ----------
    required_columns : List[str]
        Minimum required columns for product data
    optional_columns : List[str]
        Optional columns that enhance optimization
    """

    # Required columns for inventory data
    REQUIRED_COLUMNS = [
        'product_id',
        'demand',
        'unit_cost',
        'order_cost',
        'lead_time'
    ]

    # Optional but useful columns
    OPTIONAL_COLUMNS = [
        'demand_std',
        'date',
        'stockout',
        'base_demand',
        'category',
        'supplier_id'
    ]

    def __init__(
        self,
        validate: bool = True,
        clean: bool = True,
        check_quality: bool = True
    ):
        """
        Initialize DataLoader.

        Parameters
        ----------
        validate : bool, default=True
            Validate data after loading
        clean : bool, default=True
            Clean and sanitize data
        check_quality : bool, default=True
            Check data quality and report issues
        """
        self.validate = validate
        self.clean = clean
        self.check_quality = check_quality

    # ========================================================================
    # CSV LOADING
    # ========================================================================

    def load_from_csv(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load inventory data from CSV file.

        Parameters
        ----------
        file_path : str or Path
            Path to CSV file
        **kwargs
            Additional arguments for pd.read_csv

        Returns
        -------
        pd.DataFrame
            Loaded and validated data

        Raises
        ------
        DataLoadError
            If file cannot be read
        DataFormatError
            If data format is invalid

        Examples
        --------
        >>> loader = DataLoader()
        >>> data = loader.load_from_csv('inventory.csv')
        >>> print(f"Loaded {len(data)} products")
        """
        file_path = Path(file_path)

        logger.info(f"Loading data from CSV: {file_path}")

        # Check file exists
        if not file_path.exists():
            raise DataLoadError(
                f"File not found: {file_path}",
                details={"file_path": str(file_path)}
            )

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            logger.warning(
                f"Large file detected: {file_size_mb:.1f} MB",
                file_path=str(file_path)
            )

        # Load CSV
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(
                f"Successfully loaded CSV",
                rows=len(df),
                columns=len(df.columns)
            )
        except Exception as e:
            raise DataLoadError(
                f"Failed to read CSV file: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            )

        # Validate and clean
        return self._process_dataframe(df, source=str(file_path))

    # ========================================================================
    # EXCEL LOADING
    # ========================================================================

    def load_from_excel(
        self,
        file_path: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load inventory data from Excel file.

        Parameters
        ----------
        file_path : str or Path
            Path to Excel file
        sheet_name : str or int, default=0
            Sheet name or index to read
        **kwargs
            Additional arguments for pd.read_excel

        Returns
        -------
        pd.DataFrame
            Loaded and validated data

        Raises
        ------
        DataLoadError
            If file cannot be read

        Examples
        --------
        >>> loader = DataLoader()
        >>> data = loader.load_from_excel('inventory.xlsx', sheet_name='Products')
        """
        file_path = Path(file_path)

        logger.info(
            f"Loading data from Excel: {file_path}",
            sheet_name=sheet_name
        )

        # Check file exists
        if not file_path.exists():
            raise DataLoadError(
                f"File not found: {file_path}",
                details={"file_path": str(file_path)}
            )

        # Check file extension
        if file_path.suffix.lower() not in ['.xlsx', '.xls', '.xlsm']:
            raise DataFormatError(
                f"Invalid Excel file extension: {file_path.suffix}",
                details={"file_path": str(file_path)}
            )

        # Load Excel
        try:
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                engine='openpyxl' if file_path.suffix == '.xlsx' else None,
                **kwargs
            )
            logger.info(
                f"Successfully loaded Excel",
                rows=len(df),
                columns=len(df.columns),
                sheet=sheet_name
            )
        except Exception as e:
            raise DataLoadError(
                f"Failed to read Excel file: {str(e)}",
                details={
                    "file_path": str(file_path),
                    "sheet_name": sheet_name,
                    "error": str(e)
                }
            )

        # Validate and clean
        return self._process_dataframe(df, source=f"{file_path}:{sheet_name}")

    # ========================================================================
    # JSON LOADING
    # ========================================================================

    def load_from_json(
        self,
        file_path: Union[str, Path],
        orient: str = 'records'
    ) -> pd.DataFrame:
        """
        Load inventory data from JSON file.

        Parameters
        ----------
        file_path : str or Path
            Path to JSON file
        orient : str, default='records'
            JSON orientation (records, columns, index, etc.)

        Returns
        -------
        pd.DataFrame
            Loaded and validated data

        Examples
        --------
        >>> loader = DataLoader()
        >>> data = loader.load_from_json('inventory.json')
        """
        file_path = Path(file_path)

        logger.info(f"Loading data from JSON: {file_path}")

        if not file_path.exists():
            raise DataLoadError(
                f"File not found: {file_path}",
                details={"file_path": str(file_path)}
            )

        try:
            df = pd.read_json(file_path, orient=orient)
            logger.info(
                f"Successfully loaded JSON",
                rows=len(df),
                columns=len(df.columns)
            )
        except Exception as e:
            raise DataLoadError(
                f"Failed to read JSON file: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            )

        return self._process_dataframe(df, source=str(file_path))

    # ========================================================================
    # DATAFRAME VALIDATION
    # ========================================================================

    def _process_dataframe(
        self,
        df: pd.DataFrame,
        source: str = "unknown"
    ) -> pd.DataFrame:
        """
        Process DataFrame: validate, clean, and check quality.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame
        source : str
            Data source for logging

        Returns
        -------
        pd.DataFrame
            Processed DataFrame
        """
        logger.info(f"Processing data from {source}")

        # Check if DataFrame is empty
        if df.empty:
            raise DataFormatError(
                "DataFrame is empty",
                details={"source": source}
            )

        # Clean column names (strip whitespace, lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Sanitize if requested
        if self.clean:
            df = sanitize_dataframe(df)
            logger.debug("Data sanitized")

        # Validate schema
        if self.validate:
            self._validate_schema(df, source)
            logger.debug("Schema validated")

        # Check data quality
        if self.check_quality:
            quality_report = check_data_quality(df)

            if quality_report['has_issues']:
                logger.warning(
                    f"Data quality issues found: {quality_report['total_issues']} issues",
                    **quality_report
                )

                # Log specific issues
                if quality_report['missing_values']:
                    logger.warning(
                        "Missing values detected",
                        missing_values=quality_report['missing_values']
                    )

                if quality_report['duplicates']:
                    logger.warning(
                        f"Duplicate product IDs: {len(quality_report['duplicates'])}",
                        duplicates=quality_report['duplicates'][:10]
                    )

                if quality_report['negative_values']:
                    logger.error(
                        "Negative values detected in critical columns",
                        negative_values=quality_report['negative_values']
                    )
                    raise DataFormatError(
                        "Negative values found in critical columns",
                        details=quality_report['negative_values']
                    )

        logger.info(
            f"Data processing complete",
            final_rows=len(df),
            final_columns=len(df.columns)
        )

        return df

    def _validate_schema(self, df: pd.DataFrame, source: str):
        """Validate DataFrame schema."""
        # Check required columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise DataFormatError(
                f"Missing required columns: {', '.join(missing)}",
                details={
                    "missing_columns": list(missing),
                    "available_columns": list(df.columns),
                    "source": source
                }
            )

        # Validate using Pydantic
        try:
            validate_product_dataframe(df)
        except Exception as e:
            raise DataFormatError(
                f"Data validation failed: {str(e)}",
                details={"source": source, "error": str(e)}
            )

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def get_schema_info(self) -> Dict[str, List[str]]:
        """
        Get information about required and optional columns.

        Returns
        -------
        dict
            Schema information
        """
        return {
            'required_columns': self.REQUIRED_COLUMNS,
            'optional_columns': self.OPTIONAL_COLUMNS,
            'all_columns': self.REQUIRED_COLUMNS + self.OPTIONAL_COLUMNS
        }

    def create_template_dataframe(self, n_rows: int = 5) -> pd.DataFrame:
        """
        Create a template DataFrame with example data.

        Parameters
        ----------
        n_rows : int, default=5
            Number of example rows

        Returns
        -------
        pd.DataFrame
            Template DataFrame

        Examples
        --------
        >>> loader = DataLoader()
        >>> template = loader.create_template_dataframe()
        >>> template.to_csv('template.csv', index=False)
        """
        np.random.seed(42)

        data = {
            'product_id': [f'PROD_{i:03d}' for i in range(n_rows)],
            'demand': np.random.uniform(50, 500, n_rows).round(2),
            'unit_cost': np.random.uniform(10, 100, n_rows).round(2),
            'order_cost': np.random.uniform(50, 200, n_rows).round(2),
            'lead_time': np.random.randint(1, 21, n_rows),
            'demand_std': np.random.uniform(10, 50, n_rows).round(2)
        }

        return pd.DataFrame(data)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_inventory_data(
    file_path: Union[str, Path],
    file_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Load inventory data from file (auto-detect format).

    Parameters
    ----------
    file_path : str or Path
        Path to data file
    file_type : str, optional
        File type ('csv', 'excel', 'json'). Auto-detected if None.

    Returns
    -------
    pd.DataFrame
        Loaded inventory data

    Examples
    --------
    >>> data = load_inventory_data('inventory.csv')
    >>> data = load_inventory_data('inventory.xlsx')
    """
    file_path = Path(file_path)
    loader = DataLoader()

    # Auto-detect file type
    if file_type is None:
        suffix = file_path.suffix.lower()
        if suffix == '.csv':
            file_type = 'csv'
        elif suffix in ['.xlsx', '.xls', '.xlsm']:
            file_type = 'excel'
        elif suffix == '.json':
            file_type = 'json'
        else:
            raise DataFormatError(
                f"Unsupported file extension: {suffix}",
                details={"file_path": str(file_path)}
            )

    # Load based on type
    if file_type == 'csv':
        return loader.load_from_csv(file_path)
    elif file_type == 'excel':
        return loader.load_from_excel(file_path)
    elif file_type == 'json':
        return loader.load_from_json(file_path)
    else:
        raise DataFormatError(
            f"Unknown file type: {file_type}",
            details={"file_path": str(file_path)}
        )


def create_data_template(
    output_path: Union[str, Path],
    n_rows: int = 10,
    file_format: str = 'csv'
) -> None:
    """
    Create a data template file.

    Parameters
    ----------
    output_path : str or Path
        Where to save the template
    n_rows : int, default=10
        Number of example rows
    file_format : str, default='csv'
        Output format ('csv' or 'excel')

    Examples
    --------
    >>> create_data_template('template.csv')
    >>> create_data_template('template.xlsx', file_format='excel')
    """
    loader = DataLoader()
    template = loader.create_template_dataframe(n_rows)

    output_path = Path(output_path)

    if file_format == 'csv':
        template.to_csv(output_path, index=False)
    elif file_format == 'excel':
        template.to_excel(output_path, index=False, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    logger.info(f"Created template file: {output_path}")
