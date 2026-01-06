# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from unittest import mock

import pytest

from dbt_osmosis.core.validation import (
    ModelValidationResult,
    ModelValidationStatus,
    ValidationReport,
    validate_models,
)


class TestModelValidationResult:
    """Tests for ModelValidationResult dataclass."""

    def test_validation_result_success(self):
        """Create a successful validation result."""
        result = ModelValidationResult(
            model_name="test_model",
            unique_id="model.project.test_model",
            status=ModelValidationStatus.SUCCESS,
            compiled_sql="SELECT 1 AS col",
            execution_time_seconds=0.5,
            row_count=1,
        )
        assert result.model_name == "test_model"
        assert result.status == ModelValidationStatus.SUCCESS
        assert result.row_count == 1
        assert result.execution_time_seconds == 0.5

    def test_validation_result_compile_error(self):
        """Create a compile error validation result."""
        result = ModelValidationResult(
            model_name="test_model",
            unique_id="model.project.test_model",
            status=ModelValidationStatus.COMPILE_ERROR,
            error_message="Syntax error near SELECT",
        )
        assert result.status == ModelValidationStatus.COMPILE_ERROR
        assert result.error_message == "Syntax error near SELECT"
        assert result.compiled_sql is None


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_empty_report(self):
        """Create an empty validation report."""
        report = ValidationReport()
        assert report.total_models == 0
        assert report.successful == 0
        assert report.failed == 0
        assert report.get_success_rate() == 0.0

    def test_add_successful_result(self):
        """Add a successful result to the report."""
        report = ValidationReport()
        result = ModelValidationResult(
            model_name="test_model",
            unique_id="model.project.test_model",
            status=ModelValidationStatus.SUCCESS,
            execution_time_seconds=0.5,
            row_count=10,
        )
        report.add_result(result)

        assert report.total_models == 1
        assert report.successful == 1
        assert report.failed == 0
        assert report.get_success_rate() == 100.0
        assert len(report.results) == 1

    def test_add_failed_result(self):
        """Add a failed result to the report."""
        report = ValidationReport()
        result = ModelValidationResult(
            model_name="test_model",
            unique_id="model.project.test_model",
            status=ModelValidationStatus.EXECUTION_ERROR,
            error_message="Table not found",
        )
        report.add_result(result)

        assert report.total_models == 1
        assert report.successful == 0
        assert report.failed == 1
        assert report.get_success_rate() == 0.0

    def test_mixed_results(self):
        """Add mixed success and failure results."""
        report = ValidationReport()

        report.add_result(
            ModelValidationResult(
                model_name="model1",
                unique_id="model.project.model1",
                status=ModelValidationStatus.SUCCESS,
                execution_time_seconds=0.1,
            )
        )
        report.add_result(
            ModelValidationResult(
                model_name="model2",
                unique_id="model.project.model2",
                status=ModelValidationStatus.EXECUTION_ERROR,
                error_message="Error",
            )
        )
        report.add_result(
            ModelValidationResult(
                model_name="model3",
                unique_id="model.project.model3",
                status=ModelValidationStatus.SUCCESS,
                execution_time_seconds=0.2,
            )
        )

        assert report.total_models == 3
        assert report.successful == 2
        assert report.failed == 1
        assert report.get_success_rate() == pytest.approx(66.6667, rel=1e-3)

    def test_get_failed_models(self):
        """Get only failed models from the report."""
        report = ValidationReport()

        report.add_result(
            ModelValidationResult(
                model_name="model1",
                unique_id="model.project.model1",
                status=ModelValidationStatus.SUCCESS,
                execution_time_seconds=0.1,
            )
        )
        report.add_result(
            ModelValidationResult(
                model_name="model2",
                unique_id="model.project.model2",
                status=ModelValidationStatus.COMPILE_ERROR,
                error_message="Syntax error",
            )
        )
        report.add_result(
            ModelValidationResult(
                model_name="model3",
                unique_id="model.project.model3",
                status=ModelValidationStatus.EXECUTION_ERROR,
                error_message="Runtime error",
            )
        )

        failed = report.get_failed_models()
        assert len(failed) == 2
        assert [r.model_name for r in failed] == ["model2", "model3"]


class TestValidateModels:
    """Tests for the validate_models function."""

    def test_validate_empty_list(self, yaml_context):
        """Validate with an empty list of models."""
        report = validate_models(yaml_context.project, [])
        assert report.total_models == 0
        assert report.successful == 0
        assert report.failed == 0

    def test_validate_single_model_success(self, yaml_context):
        """Validate a single model successfully."""
        # Create a mock node
        mock_node = mock.Mock()
        mock_node.name = "test_model"
        mock_node.raw_code = "SELECT 1 AS col"
        mock_node.raw_sql = "SELECT 1 AS col"

        with (
            mock.patch("dbt_osmosis.core.validation.compile_sql_code") as mock_compile,
            mock.patch("dbt_osmosis.core.validation.execute_sql_code") as mock_execute,
        ):
            # Mock compile to return a compiled node
            compiled_node = mock.Mock()
            compiled_node.compiled_code = "SELECT 1 AS col"
            compiled_node.raw_code = "SELECT 1 AS col"
            mock_compile.return_value = compiled_node

            # Mock execute to return a result
            mock_adapter_response = mock.Mock()
            mock_table = mock.Mock()
            mock_table.rows = [(1,), (2,)]
            mock_execute.return_value = (mock_adapter_response, mock_table)

            report = validate_models(
                yaml_context.project,
                [("model.project.test_model", mock_node)],
            )

        assert report.total_models == 1
        assert report.successful == 1
        assert report.failed == 0

    def test_validate_model_compile_error(self, yaml_context):
        """Validate a model that fails to compile."""
        mock_node = mock.Mock()
        mock_node.name = "test_model"
        mock_node.raw_code = "SELEC 1"  # Syntax error
        mock_node.raw_sql = "SELEC 1"

        with mock.patch("dbt_osmosis.core.validation.compile_sql_code") as mock_compile:
            mock_compile.side_effect = Exception("Syntax error")

            report = validate_models(
                yaml_context.project,
                [("model.project.test_model", mock_node)],
                quiet=True,
            )

        assert report.total_models == 1
        assert report.successful == 0
        assert report.failed == 1
        assert report.results[0].status == ModelValidationStatus.COMPILE_ERROR

    def test_validate_model_execution_error(self, yaml_context):
        """Validate a model that fails during execution."""
        mock_node = mock.Mock()
        mock_node.name = "test_model"
        mock_node.raw_code = "SELECT * FROM nonexistent_table"
        mock_node.raw_sql = "SELECT * FROM nonexistent_table"

        with (
            mock.patch("dbt_osmosis.core.validation.compile_sql_code") as mock_compile,
            mock.patch("dbt_osmosis.core.validation.execute_sql_code") as mock_execute,
        ):
            compiled_node = mock.Mock()
            compiled_node.compiled_code = "SELECT * FROM nonexistent_table"
            compiled_node.raw_code = "SELECT * FROM nonexistent_table"
            mock_compile.return_value = compiled_node

            mock_execute.side_effect = Exception("Table not found")

            report = validate_models(
                yaml_context.project,
                [("model.project.test_model", mock_node)],
                quiet=True,
            )

        assert report.total_models == 1
        assert report.successful == 0
        assert report.failed == 1
        assert report.results[0].status == ModelValidationStatus.EXECUTION_ERROR
