# pyright: reportPrivateImportUsage=false, reportUnknownVariableType=false, reportUnknownMemberType=false

"""Tests for source and staging generators.

This module contains tests for the generators module, which provides:
- SourceGenerationResult and DocumentationCheckResult dataclasses
- generate_sources_from_database for source generation
- generate_staging_from_source for staging model generation
- check_documentation for documentation completeness checks
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from dbt_osmosis.core.generators import (
    DocumentationCheckResult,
    SourceGenerationResult,
    _get_source_definition,
    check_documentation,
    generate_sources_from_database,
    generate_staging_from_source,
)


class TestSourceGenerationResult:
    """Tests for the SourceGenerationResult dataclass."""

    def test_result_creation(self):
        """Test creating a source generation result."""
        result = SourceGenerationResult(
            source_name="raw",
            table_count=5,
            yaml_content="version: 2\nsources: []",
            yaml_path=Path("/path/to/sources.yml"),
        )

        assert result.source_name == "raw"
        assert result.table_count == 5
        assert result.yaml_content.startswith("version:")
        assert result.yaml_path == Path("/path/to/sources.yml")
        assert result.error is None

    def test_result_with_error(self):
        """Test creating a result with an error."""
        error = Exception("Connection failed")
        result = SourceGenerationResult(
            source_name="raw",
            table_count=0,
            yaml_content="",
            yaml_path=Path("/path/to/sources.yml"),
            error=error,
        )

        assert result.error == error
        assert result.table_count == 0


class TestDocumentationCheckResult:
    """Tests for the DocumentationCheckResult dataclass."""

    def test_result_creation(self):
        """Test creating a documentation check result."""
        result = DocumentationCheckResult(
            total_models=10,
            models_with_descriptions=8,
            models_without_descriptions=2,
            total_columns=100,
            documented_columns=75,
            undocumented_columns=25,
            gaps=[],
        )

        assert result.total_models == 10
        assert result.models_with_descriptions == 8
        assert result.models_without_descriptions == 2
        assert result.total_columns == 100
        assert result.documented_columns == 75
        assert result.undocumented_columns == 25

    def test_coverage_calculation(self):
        """Test calculating documentation coverage."""
        result = DocumentationCheckResult(
            total_models=10,
            models_with_descriptions=8,
            models_without_descriptions=2,
            total_columns=100,
            documented_columns=75,
            undocumented_columns=25,
            gaps=[],
        )

        coverage = (result.documented_columns / result.total_columns) * 100
        assert coverage == 75.0


class TestGenerateSourcesFromDatabase:
    """Tests for the generate_sources_from_database function."""

    @mock.patch("dbt_core_interface.source_generator.SourceGenerator")
    @mock.patch("dbt_core_interface.source_generator.to_yaml")
    def test_generate_sources_basic(self, mock_to_yaml, mock_generator_class, yaml_context):
        """Test basic source generation."""
        # Mock the source generator
        mock_generator = mock.MagicMock()
        mock_generator_class.return_value = mock_generator

        # Mock source definitions
        mock_source_def = mock.MagicMock()
        mock_source_def.tables = [mock.MagicMock() for _ in range(3)]
        mock_generator.generate_sources.return_value = [mock_source_def]

        # Mock to_yaml function
        mock_to_yaml.return_value = "version: 2\nsources: []"

        result = generate_sources_from_database(
            context=yaml_context,
            source_name="raw",
            schema_name="public",
        )

        assert isinstance(result, SourceGenerationResult)
        assert result.source_name == "raw"
        assert result.table_count == 3
        assert result.error is None

    @mock.patch("dbt_core_interface.source_generator.SourceGenerator")
    def test_generate_sources_no_tables(self, mock_generator_class, yaml_context):
        """Test source generation with no tables found."""
        mock_generator = mock.MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_sources.return_value = []

        result = generate_sources_from_database(
            context=yaml_context,
            source_name="raw",
        )

        assert result.table_count == 0
        assert result.yaml_content == ""

    @mock.patch("dbt_core_interface.source_generator.SourceGenerator")
    def test_generate_sources_with_excludes(self, mock_generator_class, yaml_context):
        """Test source generation with exclusions."""
        mock_generator = mock.MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_sources.return_value = []

        generate_sources_from_database(
            context=yaml_context,
            source_name="raw",
            exclude_schemas=["scratch", "tmp"],
            exclude_tables=["_temp", "_test"],
        )

        # Should call generate_sources
        assert mock_generator.generate_sources.called

    @mock.patch("dbt_core_interface.source_generator.SourceGenerator")
    def test_generate_sources_custom_path(self, mock_generator_class, yaml_context):
        """Test source generation with custom output path."""
        mock_generator = mock.MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_sources.return_value = []

        custom_path = Path("/custom/path/sources.yml")
        result = generate_sources_from_database(
            context=yaml_context,
            source_name="raw",
            output_path=custom_path,
        )

        assert result.yaml_path == custom_path

    @mock.patch("dbt_core_interface.source_generator.SourceGenerator")
    def test_generate_sources_with_exception(self, mock_generator_class, yaml_context):
        """Test source generation with exception."""
        mock_generator_class.side_effect = Exception("Database connection failed")

        with pytest.raises(Exception, match="Database connection failed"):
            generate_sources_from_database(
                context=yaml_context,
                source_name="raw",
            )


class TestGenerateStagingFromSource:
    """Tests for the generate_staging_from_source function."""

    def test_generate_staging_ai_enabled(self, yaml_context):
        """Test generating staging model with AI."""
        # Mock the source definition
        mock_source = mock.MagicMock()
        mock_source.source_name = "raw"
        mock_source.name = "users"

        # Mock _get_source_definition
        with mock.patch(
            "dbt_osmosis.core.generators._get_source_definition", return_value=mock_source
        ):
            # Mock generate_staging_for_source
            mock_spec = mock.MagicMock()
            mock_spec.staging_name = "stg_users"
            mock_spec.description = "Staging users model"
            mock_spec.columns = []
            mock_spec.error = None

            mock_result = mock.MagicMock()
            mock_result.spec = mock_spec
            mock_result.error = None

            with mock.patch(
                "dbt_osmosis.core.generators.generate_staging_for_source", return_value=mock_result
            ):
                # Mock write_staging_files
                with mock.patch("dbt_osmosis.core.generators.write_staging_files"):
                    result = generate_staging_from_source(
                        context=yaml_context,
                        source_name="raw",
                        table_name="users",
                        use_ai=True,
                    )

                    assert result.staging_name == "stg_users"
                    assert result.spec == mock_spec

    def test_generate_staging_interface_based(self, yaml_context):
        """Test generating staging model with interface generator."""
        # Mock the source definition
        mock_source = mock.MagicMock()
        mock_source.source_name = "raw"
        mock_source.name = "orders"

        with mock.patch(
            "dbt_osmosis.core.generators._get_source_definition", return_value=mock_source
        ):
            # Mock interface generator
            result_dict = {
                "staging_name": "stg_orders",
                "sql": "select * from {{ source('raw', 'orders') }}",
                "yaml": "version: 2\nmodels: []",
            }

            with mock.patch(
                "dbt_osmosis.core.generators.generate_staging_model_from_source",
                return_value=result_dict,
            ):
                result = generate_staging_from_source(
                    context=yaml_context,
                    source_name="raw",
                    table_name="orders",
                    use_ai=False,
                )

                assert result.staging_name == "stg_orders"
                assert result.sql_content == result_dict["sql"]

    def test_generate_staging_source_not_found(self, yaml_context):
        """Test staging generation when source not found."""
        with mock.patch("dbt_osmosis.core.generators._get_source_definition", return_value=None):
            with pytest.raises(ValueError, match="Source raw.users not found"):
                generate_staging_from_source(
                    context=yaml_context,
                    source_name="raw",
                    table_name="users",
                )

    def test_generate_staging_with_custom_path(self, yaml_context):
        """Test staging generation with custom output path."""
        mock_source = mock.MagicMock()
        mock_source.source_name = "raw"
        mock_source.name = "users"

        with mock.patch(
            "dbt_osmosis.core.generators._get_source_definition", return_value=mock_source
        ):
            result_dict = {
                "staging_name": "stg_users",
                "sql": "select 1",
                "yaml": "version: 2",
            }

            with mock.patch(
                "dbt_osmosis.core.generators.generate_staging_model_from_source",
                return_value=result_dict,
            ):
                custom_path = Path("/custom/staging")
                result = generate_staging_from_source(
                    context=yaml_context,
                    source_name="raw",
                    table_name="users",
                    use_ai=False,
                    staging_path=custom_path,
                )

                assert result.yaml_path.parent == custom_path

    def test_generate_staging_ai_error_fallback(self, yaml_context):
        """Test that AI generation errors are propagated."""
        mock_source = mock.MagicMock()
        mock_source.source_name = "raw"
        mock_source.name = "users"

        with mock.patch(
            "dbt_osmosis.core.generators._get_source_definition", return_value=mock_source
        ):
            mock_result = mock.MagicMock()
            mock_result.error = Exception("AI generation failed")
            mock_result.spec = None

            with mock.patch(
                "dbt_osmosis.core.generators.generate_staging_for_source", return_value=mock_result
            ):
                with pytest.raises(Exception, match="AI generation failed"):
                    generate_staging_from_source(
                        context=yaml_context,
                        source_name="raw",
                        table_name="users",
                        use_ai=True,
                    )


class TestCheckDocumentation:
    """Tests for the check_documentation function."""

    @mock.patch("dbt_osmosis.core.generators.DocumentationChecker")
    def test_check_documentation_basic(self, mock_checker_class, yaml_context):
        """Test basic documentation check."""
        # Mock the checker
        mock_checker = mock.MagicMock()
        mock_checker_class.return_value = mock_checker

        # Mock the report
        mock_report = mock.MagicMock()
        mock_report.total_models = 10
        mock_report.models_with_descriptions = 8
        mock_report.models_without_descriptions = 2
        mock_report.total_columns = 100
        mock_report.documented_columns = 75
        mock_report.undocumented_columns = 25
        mock_report.all_gaps = []

        mock_checker.check_project.return_value = mock_report

        result = check_documentation(
            context=yaml_context,
        )

        assert isinstance(result, DocumentationCheckResult)
        assert result.total_models == 10
        assert result.documented_columns == 75
        assert result.undocumented_columns == 25

    @mock.patch("dbt_osmosis.core.generators.DocumentationChecker")
    def test_check_documentation_with_filter(self, mock_checker_class, yaml_context):
        """Test documentation check with model filter."""
        mock_checker = mock.MagicMock()
        mock_checker_class.return_value = mock_checker

        mock_report = mock.MagicMock()
        mock_report.total_models = 1
        mock_report.models_with_descriptions = 1
        mock_report.models_without_descriptions = 0
        mock_report.total_columns = 10
        mock_report.documented_columns = 10
        mock_report.undocumented_columns = 0
        mock_report.all_gaps = []

        mock_checker.check_project.return_value = mock_report

        result = check_documentation(
            context=yaml_context,
            model_filter="users",
        )

        assert result.total_models == 1

    @mock.patch("dbt_osmosis.core.generators.DocumentationChecker")
    def test_check_documentation_custom_thresholds(self, mock_checker_class, yaml_context):
        """Test documentation check with custom thresholds."""
        mock_checker = mock.MagicMock()
        mock_checker_class.return_value = mock_checker

        mock_report = mock.MagicMock()
        mock_report.total_models = 5
        mock_report.models_with_descriptions = 3
        mock_report.models_without_descriptions = 2
        mock_report.total_columns = 50
        mock_report.documented_columns = 30
        mock_report.undocumented_columns = 20
        mock_report.all_gaps = []

        mock_checker.check_project.return_value = mock_report

        check_documentation(
            context=yaml_context,
            min_model_length=20,
            min_column_length=10,
        )

        # Should instantiate checker with correct thresholds
        mock_checker_class.assert_called_once()
        call_kwargs = mock_checker_class.call_args.kwargs
        assert call_kwargs["min_model_description_length"] == 20
        assert call_kwargs["min_column_description_length"] == 10

    @mock.patch("dbt_osmosis.core.generators.DocumentationChecker")
    def test_check_documentation_with_gaps(self, mock_checker_class, yaml_context):
        """Test documentation check with gaps identified."""
        mock_checker = mock.MagicMock()
        mock_checker_class.return_value = mock_checker

        # Create mock gaps
        mock_gap1 = mock.MagicMock()
        mock_gap1.model_name = "users"
        mock_gap1.column_name = "email"

        mock_gap2 = mock.MagicMock()
        mock_gap2.model_name = "orders"
        mock_gap2.column_name = "amount"

        mock_report = mock.MagicMock()
        mock_report.total_models = 2
        mock_report.models_with_descriptions = 2
        mock_report.models_without_descriptions = 0
        mock_report.total_columns = 10
        mock_report.documented_columns = 8
        mock_report.undocumented_columns = 2
        mock_report.all_gaps = [mock_gap1, mock_gap2]

        mock_checker.check_project.return_value = mock_report

        result = check_documentation(
            context=yaml_context,
        )

        assert len(result.gaps) == 2

    @mock.patch("dbt_osmosis.core.generators.DocumentationChecker")
    def test_check_documentation_zero_coverage(self, mock_checker_class, yaml_context):
        """Test documentation check with zero coverage."""
        mock_checker = mock.MagicMock()
        mock_checker_class.return_value = mock_checker

        mock_report = mock.MagicMock()
        mock_report.total_models = 5
        mock_report.models_with_descriptions = 0
        mock_report.models_without_descriptions = 5
        mock_report.total_columns = 50
        mock_report.documented_columns = 0
        mock_report.undocumented_columns = 50
        mock_report.all_gaps = []

        mock_checker.check_project.return_value = mock_report

        result = check_documentation(
            context=yaml_context,
        )

        assert result.documented_columns == 0
        assert result.undocumented_columns == 50


class TestGetSourceDefinition:
    """Tests for the _get_source_definition helper function."""

    def test_get_source_found(self, yaml_context):
        """Test getting a source that exists."""
        # Mock source in manifest
        mock_source = mock.MagicMock()
        mock_source.source_name = "raw"
        mock_source.name = "users"

        yaml_context.manifest.sources = {
            "source.raw.users": mock_source,
        }

        result = _get_source_definition(yaml_context, "raw", "users")

        assert result == mock_source

    def test_get_source_not_found(self, yaml_context):
        """Test getting a source that doesn't exist."""
        yaml_context.manifest.sources = {}

        result = _get_source_definition(yaml_context, "raw", "nonexistent")

        assert result is None

    def test_get_source_wrong_schema(self, yaml_context):
        """Test getting a source with wrong schema name."""
        mock_source = mock.MagicMock()
        mock_source.source_name = "prod"
        mock_source.name = "users"

        yaml_context.manifest.sources = {
            "source.prod.users": mock_source,
        }

        result = _get_source_definition(yaml_context, "raw", "users")

        assert result is None

    def test_get_source_multiple_sources(self, yaml_context):
        """Test getting correct source when multiple exist."""
        mock_source1 = mock.MagicMock()
        mock_source1.source_name = "raw"
        mock_source1.name = "users"

        mock_source2 = mock.MagicMock()
        mock_source2.source_name = "raw"
        mock_source2.name = "orders"

        yaml_context.manifest.sources = {
            "source.raw.users": mock_source1,
            "source.raw.orders": mock_source2,
        }

        result = _get_source_definition(yaml_context, "raw", "orders")

        assert result == mock_source2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @mock.patch("dbt_osmosis.core.generators.SourceGenerator")
    def test_generate_sources_empty_yaml(self, mock_generator_class, yaml_context):
        """Test handling of empty YAML output."""
        mock_generator = mock.MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_sources.return_value = []

        with mock.patch("dbt_osmosis.core.generators.to_yaml", return_value=""):
            result = generate_sources_from_database(
                context=yaml_context,
                source_name="raw",
            )

            assert result.yaml_content == ""

    def test_generate_staging_no_columns(self, yaml_context):
        """Test staging generation with no columns."""
        mock_source = mock.MagicMock()
        mock_source.source_name = "raw"
        mock_source.name = "empty_table"

        with mock.patch(
            "dbt_osmosis.core.generators._get_source_definition", return_value=mock_source
        ):
            result_dict = {
                "staging_name": "stg_empty_table",
                "sql": "select 1 as id",
                "yaml": "version: 2",
            }

            with mock.patch(
                "dbt_osmosis.core.generators.generate_staging_model_from_source",
                return_value=result_dict,
            ):
                result = generate_staging_from_source(
                    context=yaml_context,
                    source_name="raw",
                    table_name="empty_table",
                    use_ai=False,
                )

                assert result.staging_name == "stg_empty_table"

    @mock.patch("dbt_osmosis.core.generators.DocumentationChecker")
    def test_check_documentation_exception(self, mock_checker_class, yaml_context):
        """Test that exceptions are propagated."""
        mock_checker_class.side_effect = Exception("Checker failed")

        with pytest.raises(Exception, match="Checker failed"):
            check_documentation(
                context=yaml_context,
            )

    @mock.patch("dbt_osmosis.core.generators.SourceGenerator")
    def test_generate_sources_invalid_yaml(self, mock_generator_class, yaml_context):
        """Test handling of invalid YAML generation."""
        mock_generator = mock.MagicMock()
        mock_generator_class.return_value = mock_generator

        mock_source_def = mock.MagicMock()
        mock_source_def.tables = [mock.MagicMock()]
        mock_generator.generate_sources.return_value = [mock_source_def]

        # Mock to_yaml to raise an exception
        with mock.patch(
            "dbt_osmosis.core.generators.to_yaml", side_effect=Exception("YAML generation failed")
        ):
            with pytest.raises(Exception, match="YAML generation failed"):
                generate_sources_from_database(
                    context=yaml_context,
                    source_name="raw",
                )

    def test_get_source_empty_manifest(self, yaml_context):
        """Test getting source from empty manifest."""
        yaml_context.manifest.sources = {}

        result = _get_source_definition(yaml_context, "raw", "users")

        assert result is None
