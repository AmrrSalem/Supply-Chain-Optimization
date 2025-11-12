"""
Unit Tests for Results Export
==============================

Tests for exporter.py module including CSV, Excel, and JSON exports.
"""

import pytest
import pandas as pd
import json
from pathlib import Path

from utils.exporter import ResultsExporter, export_optimization_results


@pytest.mark.unit
class TestResultsExporterCSV:
    """Test suite for CSV export."""

    def test_export_to_csv_success(self, sample_product_data, tmp_path):
        """Test successful CSV export."""
        exporter = ResultsExporter()
        output_path = tmp_path / "results.csv"

        result_path = exporter.export_to_csv(sample_product_data, output_path)

        assert result_path.exists()
        assert result_path == output_path

        # Verify content
        df = pd.read_csv(result_path)
        assert len(df) == len(sample_product_data)

    def test_export_to_csv_creates_directory(self, sample_product_data, tmp_path):
        """Test that CSV export creates directory if needed."""
        exporter = ResultsExporter()
        output_path = tmp_path / "subdir" / "results.csv"

        result_path = exporter.export_to_csv(sample_product_data, output_path)

        assert result_path.exists()
        assert result_path.parent.exists()

    def test_export_to_csv_with_timestamp(self, sample_product_data, tmp_path):
        """Test CSV export with timestamp in filename."""
        exporter = ResultsExporter(add_timestamp=True)
        output_path = tmp_path / "results.csv"

        result_path = exporter.export_to_csv(sample_product_data, output_path)

        assert result_path.exists()
        assert "results_" in result_path.name  # Contains timestamp


@pytest.mark.unit
class TestResultsExporterExcel:
    """Test suite for Excel export."""

    def test_export_to_excel_success(self, sample_product_data, tmp_path):
        """Test successful Excel export."""
        exporter = ResultsExporter()
        output_path = tmp_path / "results.xlsx"

        result_path = exporter.export_to_excel(sample_product_data, output_path)

        assert result_path.exists()

        # Verify content
        df = pd.read_excel(result_path, engine='openpyxl')
        assert len(df) == len(sample_product_data)

    def test_export_to_excel_custom_sheet_name(self, sample_product_data, tmp_path):
        """Test Excel export with custom sheet name."""
        exporter = ResultsExporter()
        output_path = tmp_path / "results.xlsx"

        exporter.export_to_excel(
            sample_product_data,
            output_path,
            sheet_name='CustomSheet'
        )

        # Verify sheet name
        df = pd.read_excel(output_path, sheet_name='CustomSheet', engine='openpyxl')
        assert len(df) == len(sample_product_data)

    def test_export_to_excel_multi_sheet(self, sample_product_data, tmp_path):
        """Test multi-sheet Excel export."""
        exporter = ResultsExporter()
        output_path = tmp_path / "results.xlsx"

        data_dict = {
            'Sheet1': sample_product_data,
            'Sheet2': sample_product_data.head(5),
            'Sheet3': sample_product_data.tail(5)
        }

        result_path = exporter.export_to_excel_multi_sheet(data_dict, output_path)

        assert result_path.exists()

        # Verify sheets
        df1 = pd.read_excel(result_path, sheet_name='Sheet1', engine='openpyxl')
        df2 = pd.read_excel(result_path, sheet_name='Sheet2', engine='openpyxl')
        df3 = pd.read_excel(result_path, sheet_name='Sheet3', engine='openpyxl')

        assert len(df1) == len(sample_product_data)
        assert len(df2) == 5
        assert len(df3) == 5


@pytest.mark.unit
class TestResultsExporterJSON:
    """Test suite for JSON export."""

    def test_export_to_json_dataframe(self, sample_product_data, tmp_path):
        """Test JSON export with DataFrame."""
        exporter = ResultsExporter()
        output_path = tmp_path / "results.json"

        result_path = exporter.export_to_json(sample_product_data, output_path)

        assert result_path.exists()

        # Verify content
        with open(result_path, 'r') as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == len(sample_product_data)

    def test_export_to_json_dict(self, tmp_path):
        """Test JSON export with dictionary."""
        exporter = ResultsExporter()
        output_path = tmp_path / "results.json"

        test_dict = {
            'baseline_cost': 100000.0,
            'optimized_cost': 80000.0,
            'improvement': 20.0
        }

        result_path = exporter.export_to_json(test_dict, output_path)

        assert result_path.exists()

        # Verify content
        with open(result_path, 'r') as f:
            data = json.load(f)

        assert data == test_dict


@pytest.mark.unit
class TestSummaryExport:
    """Test suite for summary report export."""

    def test_export_summary_report_txt(self, tmp_path):
        """Test summary report export to text file."""
        exporter = ResultsExporter()
        output_path = tmp_path / "summary.txt"

        comparison = {
            'baseline': {
                'total_cost': 100000.0,
                'total_investment': 500000.0,
                'avg_service_level': 0.95
            },
            'stochastic': {
                'total_cost': 80000.0,
                'cost_reduction_pct': 20.0
            }
        }

        result_path = exporter.export_summary_report(
            comparison,
            output_path,
            format='txt'
        )

        assert result_path.exists()

        # Verify content
        content = result_path.read_text()
        assert 'OPTIMIZATION SUMMARY' in content
        assert 'BASELINE PERFORMANCE' in content
        assert '100,000.00' in content or '100000' in content

    def test_export_summary_report_json(self, tmp_path):
        """Test summary report export to JSON."""
        exporter = ResultsExporter()
        output_path = tmp_path / "summary.json"

        comparison = {
            'baseline': {'total_cost': 100000.0},
            'stochastic': {'cost_reduction_pct': 20.0}
        }

        result_path = exporter.export_summary_report(
            comparison,
            output_path,
            format='json'
        )

        assert result_path.exists()

        # Verify content
        with open(result_path, 'r') as f:
            data = json.load(f)

        assert data == comparison


@pytest.mark.unit
class TestConvenienceExport:
    """Test suite for convenience export function."""

    def test_export_optimization_results(self, sample_product_data, tmp_path):
        """Test complete optimization results export."""
        baseline_df = sample_product_data.copy()
        optimized_df = sample_product_data.copy()
        optimized_df['demand'] = optimized_df['demand'] * 0.9  # Simulate optimization

        comparison = {
            'baseline': {
                'total_cost': 100000.0,
                'total_investment': 500000.0
            },
            'stochastic': {
                'total_cost': 80000.0,
                'cost_reduction_pct': 20.0
            }
        }

        paths = export_optimization_results(
            baseline_df,
            optimized_df,
            comparison,
            tmp_path,
            base_name='test_results'
        )

        # Verify all files created
        assert 'excel' in paths
        assert 'baseline_csv' in paths
        assert 'optimized_csv' in paths
        assert 'summary' in paths

        # Verify files exist
        assert paths['excel'].exists()
        assert paths['baseline_csv'].exists()
        assert paths['optimized_csv'].exists()
        assert paths['summary'].exists()

    def test_export_optimization_results_baseline_only(self, sample_product_data, tmp_path):
        """Test export with baseline results only."""
        baseline_df = sample_product_data.copy()

        comparison = {
            'baseline': {
                'total_cost': 100000.0
            }
        }

        paths = export_optimization_results(
            baseline_df,
            None,  # No optimized results
            comparison,
            tmp_path
        )

        # Should still create files
        assert paths['excel'].exists()
        assert paths['baseline_csv'].exists()
        assert paths['summary'].exists()

        # Should not create optimized CSV
        assert 'optimized_csv' not in paths


@pytest.mark.unit
class TestFilePathHandling:
    """Test suite for file path handling."""

    def test_adds_extension_if_missing(self, sample_product_data, tmp_path):
        """Test that extension is added if missing."""
        exporter = ResultsExporter()
        output_path = tmp_path / "results"  # No extension

        result_path = exporter.export_to_excel(sample_product_data, output_path)

        assert result_path.suffix == '.xlsx'

    def test_timestamp_format(self, sample_product_data, tmp_path):
        """Test timestamp format in filename."""
        exporter = ResultsExporter(add_timestamp=True)
        output_path = tmp_path / "results.csv"

        result_path = exporter.export_to_csv(sample_product_data, output_path)

        # Filename should contain timestamp in format: results_YYYYMMDD_HHMMSS.csv
        assert '_' in result_path.stem
        assert result_path.suffix == '.csv'
