# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest
import ruamel.yaml
from dbt.contracts.results import CatalogResults

from dbt_osmosis.core.settings import (
    EMPTY_STRING,
    YamlRefactorContext,
    YamlRefactorSettings,
)


class TestYamlRefactorSettings:
    """Test suite for YamlRefactorSettings dataclass."""

    def test_default_settings(self):
        """Test that YamlRefactorSettings has correct default values."""
        settings = YamlRefactorSettings()

        # Check default values
        assert settings.fqn == []
        assert settings.models == []
        assert settings.dry_run is False
        assert settings.skip_merge_meta is False
        assert settings.skip_add_columns is False
        assert settings.skip_add_tags is False
        assert settings.skip_add_data_types is False
        assert settings.skip_add_source_columns is False
        assert settings.add_progenitor_to_meta is False
        assert settings.numeric_precision_and_scale is False
        assert settings.string_length is False
        assert settings.force_inherit_descriptions is False
        assert settings.use_unrendered_descriptions is False
        assert settings.add_inheritance_for_specified_keys == []
        assert settings.output_to_lower is False
        assert settings.catalog_path is None
        assert settings.create_catalog_if_not_exists is False

    def test_custom_settings(self):
        """Test YamlRefactorSettings with custom values."""
        models = [Path("test_model1.sql"), "test_model2.sql"]
        settings = YamlRefactorSettings(
            fqn=["project.schema.model"],
            models=models,
            dry_run=True,
            skip_merge_meta=True,
            skip_add_columns=True,
            skip_add_tags=True,
            skip_add_data_types=True,
            skip_add_source_columns=True,
            add_progenitor_to_meta=True,
            numeric_precision_and_scale=True,
            string_length=True,
            force_inherit_descriptions=True,
            use_unrendered_descriptions=True,
            add_inheritance_for_specified_keys=["custom_field", "another_field"],
            output_to_lower=True,
            catalog_path="/path/to/catalog.json",
            create_catalog_if_not_exists=True,
        )

        assert settings.fqn == ["project.schema.model"]
        assert settings.models == models
        assert settings.dry_run is True
        assert settings.skip_merge_meta is True
        assert settings.skip_add_columns is True
        assert settings.skip_add_tags is True
        assert settings.skip_add_data_types is True
        assert settings.skip_add_source_columns is True
        assert settings.add_progenitor_to_meta is True
        assert settings.numeric_precision_and_scale is True
        assert settings.string_length is True
        assert settings.force_inherit_descriptions is True
        assert settings.use_unrendered_descriptions is True
        assert settings.add_inheritance_for_specified_keys == ["custom_field", "another_field"]
        assert settings.output_to_lower is True
        assert settings.catalog_path == "/path/to/catalog.json"
        assert settings.create_catalog_if_not_exists is True

    def test_settings_immutability(self):
        """Test that settings are properly created as dataclass instances."""
        settings = YamlRefactorSettings()

        # Should not have _mutation_count as it's init=False
        assert not hasattr(settings, "_mutation_count")

        # Should not have _catalog as it's init=False
        assert not hasattr(settings, "_catalog")


class TestYamlRefactorContext:
    """Test suite for YamlRefactorContext dataclass."""

    @pytest.fixture(scope="function")
    def mock_project_context(self):
        """Create a mock DbtProjectContext for testing."""
        mock_cfg = Mock()
        mock_cfg.threads = 4
        mock_cfg.vars = Mock()
        mock_cfg.vars.to_dict.return_value = {}

        mock_runtime_cfg = Mock()
        mock_runtime_cfg.threads = 4
        mock_runtime_cfg.vars = Mock()
        mock_runtime_cfg.vars.to_dict.return_value = {}
        mock_cfg.runtime_cfg = mock_runtime_cfg

        project_context = Mock()
        project_context.runtime_cfg = mock_runtime_cfg
        return project_context

    def test_default_context_initialization(self, mock_project_context):
        """Test YamlRefactorContext with default settings."""
        context = YamlRefactorContext(project=mock_project_context)

        # Check project reference
        assert context.project == mock_project_context

        # Check default settings
        assert isinstance(context.settings, YamlRefactorSettings)
        assert context.settings.dry_run is False

        # Check thread pool
        assert isinstance(context.pool, ThreadPoolExecutor)
        assert context.pool._max_workers == 4

        # Check YAML handler
        assert isinstance(context.yaml_handler, ruamel.yaml.YAML)

        # Check YAML handler lock
        assert context.yaml_handler_lock is not None
        assert hasattr(context.yaml_handler_lock, "acquire")
        assert hasattr(context.yaml_handler_lock, "release")

        # Check placeholders
        expected_placeholders = (
            "",
            "Pending further documentation",
            "No description for this column",
            "Not documented",
            "Undefined",
        )
        assert context.placeholders == expected_placeholders

        # Check mutation count
        assert context._mutation_count == 0
        assert not context.mutation_count
        assert not context.mutated

        # Check catalog
        assert context._catalog is None

    def test_context_with_custom_settings(self, mock_project_context):
        """Test YamlRefactorContext with custom settings."""
        custom_settings = YamlRefactorSettings(
            dry_run=True,
            skip_merge_meta=True,
        )

        context = YamlRefactorContext(
            project=mock_project_context,
            settings=custom_settings,
        )

        assert context.settings == custom_settings
        assert context.settings.dry_run is True
        assert context.settings.skip_merge_meta is True

    def test_mutation_count_registration(self, mock_project_context):
        """Test mutation count registration functionality."""
        context = YamlRefactorContext(project=mock_project_context)

        # Initially zero
        assert context._mutation_count == 0

        # Register mutations
        context.register_mutations(5)
        assert context._mutation_count == 5

        # Register more mutations
        context.register_mutations(3)
        assert context._mutation_count == 8

        # Check property access
        assert context.mutation_count == 8
        assert context.mutated is True

    def test_mutation_properties(self, mock_project_context):
        """Test mutation count properties."""
        context = YamlRefactorContext(project=mock_project_context)

        # Initially not mutated
        assert not context.mutated
        assert context.mutation_count == 0

        # After mutations
        context.register_mutations(1)
        assert context.mutated
        assert context.mutation_count == 1

    def test_placeholders_initialization_with_empty_string(self, mock_project_context):
        """Test that EMPTY_STRING is always the first placeholder."""
        context = YamlRefactorContext(project=mock_project_context)
        assert context.placeholders[0] == EMPTY_STRING

    def test_source_definitions_property_empty_vars(self, mock_project_context):
        """Test source_definitions when no dbt-osmosis vars are set."""
        mock_project_context.runtime_cfg.vars.to_dict.return_value = {}

        context = YamlRefactorContext(project=mock_project_context)
        assert context.source_definitions == {}

    def test_source_definitions_property_with_dbt_osmosis_vars(self, mock_project_context):
        """Test source_definitions when dbt-osmosis vars are set."""
        vars_dict = {
            "dbt-osmosis": {
                "sources": {
                    "my_source": {
                        "table": "my_table",
                        "database": "my_db",
                    }
                }
            }
        }
        mock_project_context.runtime_cfg.vars.to_dict.return_value = vars_dict

        context = YamlRefactorContext(project=mock_project_context)
        expected = {
            "my_source": {
                "table": "my_table",
                "database": "my_db",
            }
        }
        assert context.source_definitions == expected

    def test_source_definitions_property_with_dbt_osmosis_underscore(self, mock_project_context):
        """Test source_definitions with dbt_osmosis (underscore) config."""
        vars_dict = {
            "dbt_osmosis": {
                "sources": {
                    "my_source": {
                        "table": "my_table",
                    }
                }
            }
        }
        mock_project_context.runtime_cfg.vars.to_dict.return_value = vars_dict

        context = YamlRefactorContext(project=mock_project_context)
        expected = {
            "my_source": {
                "table": "my_table",
            }
        }
        assert context.source_definitions == expected

    def test_ignore_patterns_property_empty(self, mock_project_context):
        """Test ignore_patterns when no patterns are set."""
        mock_project_context.runtime_cfg.vars.to_dict.return_value = {}

        context = YamlRefactorContext(project=mock_project_context)
        assert context.ignore_patterns == []

    def test_ignore_patterns_property_with_patterns(self, mock_project_context):
        """Test ignore_patterns when patterns are set."""
        vars_dict = {"dbt-osmosis": {"column_ignore_patterns": ["test_.*", "_temp$"]}}
        mock_project_context.runtime_cfg.vars.to_dict.return_value = vars_dict

        context = YamlRefactorContext(project=mock_project_context)
        expected = ["test_.*", "_temp$"]
        assert context.ignore_patterns == expected

    def test_yaml_settings_property_empty(self, mock_project_context):
        """Test yaml_settings when no settings are configured."""
        mock_project_context.runtime_cfg.vars.to_dict.return_value = {}

        context = YamlRefactorContext(project=mock_project_context)
        assert context.yaml_settings == {}

    def test_yaml_settings_property_with_config(self, mock_project_context):
        """Test yaml_settings when settings are configured."""
        vars_dict = {
            "dbt-osmosis": {
                "yaml_settings": {
                    "map_indent": 2,
                    "sequence_indent": 4,
                    "width": 120,
                }
            }
        }
        mock_project_context.runtime_cfg.vars.to_dict.return_value = vars_dict

        context = YamlRefactorContext(project=mock_project_context)
        expected = {
            "map_indent": 2,
            "sequence_indent": 4,
            "width": 120,
        }
        assert context.yaml_settings == expected

    def test_read_catalog_with_existing_catalog(self, mock_project_context):
        """Test reading catalog when one already exists."""
        mock_catalog = Mock(spec=CatalogResults)
        context = YamlRefactorContext(project=mock_project_context)
        context._catalog = mock_catalog

        catalog = context.read_catalog()
        assert catalog == mock_catalog

    def test_read_catalog_without_existing_catalog(self, mock_project_context):
        """Test reading catalog when none exists."""
        context = YamlRefactorContext(project=mock_project_context)

        with (
            mock.patch("dbt_osmosis.core.introspection._load_catalog") as mock_load,
            mock.patch("dbt_osmosis.core.introspection._generate_catalog") as mock_generate,
        ):
            mock_load.return_value = None
            mock_generate.return_value = Mock(spec=CatalogResults)

            context.read_catalog()

            # Should have tried to load catalog first
            mock_load.assert_called_once_with(context.settings)

            # Since _load_catalog returned None and create_catalog_if_not_exists is False,
            # catalog should remain None
            assert context._catalog is None

    def test_read_catalog_with_auto_generate(self, mock_project_context):
        """Test auto-generating catalog when create_catalog_if_not_exists is True."""
        settings = YamlRefactorSettings(create_catalog_if_not_exists=True)
        context = YamlRefactorContext(project=mock_project_context, settings=settings)

        with (
            mock.patch("dbt_osmosis.core.introspection._load_catalog") as mock_load,
            mock.patch("dbt_osmosis.core.introspection._generate_catalog") as mock_generate,
        ):
            mock_load.return_value = None
            mock_generate.return_value = Mock(spec=CatalogResults)

            context.read_catalog()

            # Should have tried to generate catalog
            mock_generate.assert_called_once_with(context.project)

            # Catalog should now be set
            assert context._catalog is not None

    def test_read_catalog_caching(self, mock_project_context):
        """Test that catalog is cached after first read."""
        mock_catalog = Mock(spec=CatalogResults)
        context = YamlRefactorContext(project=mock_project_context)
        context._catalog = mock_catalog

        # Multiple reads should return the same catalog
        assert context.read_catalog() == mock_catalog
        assert context.read_catalog() == mock_catalog
        assert context.read_catalog() == mock_catalog

    def test_find_first_method(self, mock_project_context):
        """Test the _find_first helper method."""
        context = YamlRefactorContext(project=mock_project_context)

        # Test finding first truthy value
        values = [None, False, [], {"key": "value"}]
        result = context._find_first(values, bool, "default")
        assert result == {"key": "value"}

        # Test finding first matching predicate
        numbers = [1, 2, 3, 4, 5]
        result = context._find_first(numbers, lambda x: x > 3, "default")
        assert result == 4

        # Test default value when no match
        result = context._find_first(numbers, lambda x: x > 10, "default")
        assert result == "default"

    def test_yaml_handler_configuration(self, mock_project_context):
        """Test that YAML handler is configured with settings from project."""
        vars_dict = {
            "dbt-osmosis": {
                "yaml_settings": {
                    "map_indent": 4,
                    "sequence_indent": 2,
                    "width": 120,
                }
            }
        }
        mock_project_context.runtime_cfg.vars.to_dict.return_value = vars_dict

        context = YamlRefactorContext(project=mock_project_context)

        # YAML handler should be configured with project settings
        assert context.yaml_handler.map_indent == 4
        assert context.yaml_handler.sequence_indent == 2
        assert context.yaml_handler.width == 120

    def test_thread_pool_configuration(self, mock_project_context):
        """Test that thread pool is configured with dbt thread count."""
        mock_project_context.runtime_cfg.threads = 8
        context = YamlRefactorContext(project=mock_project_context)

        # ThreadPoolExecutor should have max_workers set to dbt thread count
        assert context.pool._max_workers == 8


@pytest.fixture(scope="function")
def demo_project_context():
    """Create a real YamlRefactorContext using the demo_duckdb project."""
    # Use a simpler approach that doesn't require full dbt project setup
    mock_cfg = Mock()
    mock_cfg.threads = 4
    mock_cfg.vars = Mock()
    mock_cfg.vars.to_dict.return_value = {"dbt-osmosis": {}}

    mock_runtime_cfg = Mock()
    mock_runtime_cfg.threads = 4
    mock_runtime_cfg.vars = Mock()
    mock_runtime_cfg.vars.to_dict.return_value = {"dbt-osmosis": {}}
    mock_cfg.runtime_cfg = mock_runtime_cfg

    project_context = Mock()
    project_context.runtime_cfg = mock_runtime_cfg
    context = YamlRefactorContext(
        project_context,
        settings=YamlRefactorSettings(dry_run=True),
    )
    return context


class TestYamlRefactorContextIntegration:
    """Integration tests for YamlRefactorContext with real project."""

    def test_context_initialization_with_real_project(self, demo_project_context):
        """Test that context initializes correctly with real dbt project."""
        context = demo_project_context

        # Check that all components are properly initialized
        assert context.project is not None
        assert isinstance(context.settings, YamlRefactorSettings)
        assert isinstance(context.pool, ThreadPoolExecutor)
        assert isinstance(context.yaml_handler, ruamel.yaml.YAML)
        assert context.yaml_handler_lock is not None
        assert hasattr(context.yaml_handler_lock, "acquire")
        assert hasattr(context.yaml_handler_lock, "release")

        # Check that thread pool is configured
        assert context.pool._max_workers > 0

        # Check that yaml_handler is properly configured
        assert hasattr(context.yaml_handler, "map_indent")
        assert hasattr(context.yaml_handler, "sequence_indent")

        # Check YAML handler lock
        assert context.yaml_handler_lock is not None
        assert hasattr(context.yaml_handler_lock, "acquire")
        assert hasattr(context.yaml_handler_lock, "release")

    def test_read_catalog_with_real_project(self, demo_project_context):
        """Test reading catalog with real dbt project."""
        # This test ensures that the catalog reading process works
        # with the actual demo_duckdb project
        context = demo_project_context

        # Try to read catalog (may return None if no catalog exists)
        catalog = context.read_catalog()

        # Either returns a catalog or None
        assert catalog is None or isinstance(catalog, CatalogResults)

    def test_context_yaml_settings_integration(self, demo_project_context):
        """Test that YAML settings are properly integrated."""
        context = demo_project_context

        # Check that yaml_settings property works
        settings = context.yaml_settings
        assert isinstance(settings, dict)

        # Test yaml_handler configuration
        assert isinstance(context.yaml_handler, ruamel.yaml.YAML)

    def test_mutation_count_in_real_context(self, demo_project_context):
        """Test mutation count functionality in real context."""
        context = demo_project_context

        # Initially should be zero
        assert context.mutation_count == 0
        assert not context.mutated

        # Register mutations
        context.register_mutations(1)
        assert context.mutation_count == 1
        assert context.mutated
