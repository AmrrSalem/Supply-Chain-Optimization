"""
Unit Tests for Data Loading
============================

Tests for data_loader.py module including CSV, Excel, and JSON loading.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from utils.data_loader import DataLoader, load_inventory_data, create_data_template
from utils.exceptions import DataLoadError, DataFormatError


@pytest.mark.unit
class TestDataLoaderCSV:
    """Test suite for CSV data loading."""

    def test_load_from_csv_success(self, temp_csv_file):
        """Test successful CSV loading."""
        loader = DataLoader()
        data = loader.load_from_csv(temp_csv_file)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'product_id' in data.columns

    def test_load_from_csv_file_not_found(self):
        """Test error handling for missing file."""
        loader = DataLoader()

        with pytest.raises(DataLoadError) as exc_info:
            loader.load_from_csv('nonexistent_file.csv')

        assert "not found" in str(exc_info.value).lower()

    def test_load_from_csv_validates_data(self, tmp_path):
        """Test that CSV loading validates data."""
        # Create invalid CSV (missing required columns)
        invalid_file = tmp_path / "invalid.csv"
        df = pd.DataFrame({
            'product_id': ['PROD_001'],
            'demand': [100.0]
            # Missing other required columns
        })
        df.to_csv(invalid_file, index=False)

        loader = DataLoader(validate=True)

        with pytest.raises(DataFormatError):
            loader.load_from_csv(invalid_file)

    def test_load_from_csv_without_validation(self, tmp_path):
        """Test CSV loading without validation."""
        # Create CSV with minimal columns
        file_path = tmp_path / "minimal.csv"
        df = pd.DataFrame({
            'product_id': ['PROD_001'],
            'demand': [100.0]
        })
        df.to_csv(file_path, index=False)

        loader = DataLoader(validate=False, clean=False, check_quality=False)
        data = loader.load_from_csv(file_path)

        assert len(data) == 1
        assert 'product_id' in data.columns


@pytest.mark.unit
class TestDataLoaderExcel:
    """Test suite for Excel data loading."""

    def test_load_from_excel_success(self, temp_excel_file):
        """Test successful Excel loading."""
        loader = DataLoader()
        data = loader.load_from_excel(temp_excel_file)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'product_id' in data.columns

    def test_load_from_excel_specific_sheet(self, tmp_path, sample_product_data):
        """Test loading specific sheet from Excel."""
        file_path = tmp_path / "multi_sheet.xlsx"

        # Create multi-sheet Excel
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            sample_product_data.to_excel(writer, sheet_name='Sheet1', index=False)
            sample_product_data.to_excel(writer, sheet_name='Products', index=False)

        loader = DataLoader()
        data = loader.load_from_excel(file_path, sheet_name='Products')

        assert len(data) == len(sample_product_data)

    def test_load_from_excel_file_not_found(self):
        """Test error handling for missing Excel file."""
        loader = DataLoader()

        with pytest.raises(DataLoadError):
            loader.load_from_excel('nonexistent_file.xlsx')

    def test_load_from_excel_invalid_extension(self, tmp_path):
        """Test error for invalid file extension."""
        file_path = tmp_path / "data.txt"
        file_path.write_text("not an excel file")

        loader = DataLoader()

        with pytest.raises(DataFormatError):
            loader.load_from_excel(file_path)


@pytest.mark.unit
class TestDataLoaderJSON:
    """Test suite for JSON data loading."""

    def test_load_from_json_success(self, tmp_path, sample_product_data):
        """Test successful JSON loading."""
        file_path = tmp_path / "data.json"
        sample_product_data.to_json(file_path, orient='records')

        loader = DataLoader()
        data = loader.load_from_json(file_path)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_load_from_json_file_not_found(self):
        """Test error handling for missing JSON file."""
        loader = DataLoader()

        with pytest.raises(DataLoadError):
            loader.load_from_json('nonexistent_file.json')


@pytest.mark.unit
class TestDataLoaderHelpers:
    """Test suite for DataLoader helper methods."""

    def test_get_schema_info(self):
        """Test getting schema information."""
        loader = DataLoader()
        schema = loader.get_schema_info()

        assert 'required_columns' in schema
        assert 'optional_columns' in schema
        assert 'all_columns' in schema
        assert 'product_id' in schema['required_columns']
        assert 'demand' in schema['required_columns']

    def test_create_template_dataframe(self):
        """Test template DataFrame creation."""
        loader = DataLoader()
        template = loader.create_template_dataframe(n_rows=5)

        assert len(template) == 5
        assert 'product_id' in template.columns
        assert 'demand' in template.columns
        assert 'unit_cost' in template.columns


@pytest.mark.unit
@pytest.mark.fast
class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_load_inventory_data_csv(self, temp_csv_file):
        """Test convenience function for CSV."""
        data = load_inventory_data(temp_csv_file)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_load_inventory_data_excel(self, temp_excel_file):
        """Test convenience function for Excel."""
        data = load_inventory_data(temp_excel_file)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_load_inventory_data_auto_detect(self, temp_csv_file):
        """Test auto-detection of file type."""
        data = load_inventory_data(temp_csv_file, file_type=None)

        assert isinstance(data, pd.DataFrame)

    def test_create_data_template_csv(self, tmp_path):
        """Test creating CSV template."""
        output_path = tmp_path / "template.csv"
        create_data_template(output_path, n_rows=10, file_format='csv')

        assert output_path.exists()

        # Load and verify
        df = pd.read_csv(output_path)
        assert len(df) == 10
        assert 'product_id' in df.columns

    def test_create_data_template_excel(self, tmp_path):
        """Test creating Excel template."""
        output_path = tmp_path / "template.xlsx"
        create_data_template(output_path, n_rows=10, file_format='excel')

        assert output_path.exists()

        # Load and verify
        df = pd.read_excel(output_path, engine='openpyxl')
        assert len(df) == 10
        assert 'product_id' in df.columns


@pytest.mark.unit
class TestDataCleaning:
    """Test suite for data cleaning and sanitization."""

    def test_column_name_normalization(self, tmp_path):
        """Test that column names are normalized."""
        file_path = tmp_path / "messy_columns.csv"

        # Create DataFrame with messy column names
        df = pd.DataFrame({
            ' Product ID ': ['PROD_001'],
            'DEMAND': [100.0],
            'Unit Cost': [45.0],
            'Order_Cost': [200.0],
            'Lead Time': [7]
        })
        df.to_csv(file_path, index=False)

        loader = DataLoader()
        data = loader.load_from_csv(file_path)

        # Check that columns are normalized
        assert 'product_id' in data.columns
        assert 'demand' in data.columns
        assert 'unit_cost' in data.columns

    def test_sanitization_removes_dangerous_content(self, tmp_path):
        """Test that sanitization removes dangerous content."""
        file_path = tmp_path / "dangerous.csv"

        df = pd.DataFrame({
            'product_id': ["<script>alert('xss')</script>"],
            'demand': [100.0],
            'unit_cost': [45.0],
            'order_cost': [200.0],
            'lead_time': [7]
        })
        df.to_csv(file_path, index=False)

        loader = DataLoader(clean=True)
        data = loader.load_from_csv(file_path)

        # Check that dangerous content is removed
        assert '<script>' not in data.iloc[0]['product_id']
