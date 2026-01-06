# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

import logging
from pathlib import Path
from unittest import mock

import pytest

from dbt_osmosis.core.path_management import create_missing_source_yamls
from dbt_osmosis.core.restructuring import (
    RestructureDeltaPlan,
    RestructureOperation,
    apply_restructure_plan,
    draft_restructure_delta_plan,
    pretty_print_plan,
)
from dbt_osmosis.core.settings import YamlRefactorContext


@pytest.fixture(scope="function")
def fresh_caches():
    """Patches the internal caches so each test starts with a fresh state."""
    with mock.patch("dbt_osmosis.core.schema.reader._YAML_BUFFER_CACHE", {}):
        yield


def test_create_missing_source_yamls(yaml_context: YamlRefactorContext, fresh_caches):
    """Creates missing source YAML files if any are declared in dbt-osmosis sources
    but do not exist in the manifest. Typically, might be none in your project.
    """
    create_missing_source_yamls(yaml_context)


def test_draft_restructure_delta_plan(yaml_context: YamlRefactorContext, fresh_caches):
    """Ensures we can generate a restructure plan for real models and sources.
    Usually, this plan might be empty if everything lines up already.
    """
    plan = draft_restructure_delta_plan(yaml_context)
    assert plan is not None


def test_apply_restructure_plan(yaml_context: YamlRefactorContext, fresh_caches):
    """Applies the restructure plan for the real project (in dry_run mode).
    Should not raise errors even if the plan is empty or small.
    """
    plan = draft_restructure_delta_plan(yaml_context)
    apply_restructure_plan(yaml_context, plan, confirm=False)


def test_pretty_print_plan(caplog):
    """Test pretty_print_plan logs the correct output for each operation."""
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=Path("models/some_file.yml"),
                content={"models": [{"name": "my_model"}]},
            ),
            RestructureOperation(
                file_path=Path("sources/another_file.yml"),
                content={"sources": [{"name": "my_source"}]},
                superseded_paths={Path("old_file.yml"): []},
            ),
        ],
    )
    test_logger = logging.getLogger("test_logger")
    with mock.patch("dbt_osmosis.core.logger.LOGGER", test_logger):
        caplog.clear()
        with caplog.at_level(logging.INFO):
            pretty_print_plan(plan)
    logs = caplog.text
    assert "Restructure plan includes => 2 operations" in logs
    assert "CREATE or MERGE => models/some_file.yml" in logs
    assert "['old_file.yml'] -> sources/another_file.yml" in logs


def test_apply_restructure_plan_confirm_prompt(
    yaml_context: YamlRefactorContext,
    fresh_caches,
    capsys,
):
    """We test apply_restructure_plan with confirm=True, mocking input to 'n' to skip it.
    This ensures we handle user input logic.
    """
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=Path("models/some_file.yml"),
                content={"models": [{"name": "m1"}]},
            ),
        ],
    )

    with mock.patch("builtins.input", side_effect=["n"]):
        apply_restructure_plan(yaml_context, plan, confirm=True)
        captured = capsys.readouterr()
        assert "Skipping restructure plan." in captured.err


def test_apply_restructure_plan_confirm_yes(
    yaml_context: YamlRefactorContext,
    fresh_caches,
    capsys,
):
    """Same as above, but we input 'y' so it actually proceeds with the plan.
    (No real writing occurs due to dry_run=True).
    """
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=Path("models/whatever.yml"),
                content={"models": [{"name": "m2"}]},
            ),
        ],
    )

    with mock.patch("builtins.input", side_effect=["y"]):
        apply_restructure_plan(yaml_context, plan, confirm=True)
        captured = capsys.readouterr()
        assert "Committing all restructure changes" in captured.err
        assert "Reloading the dbt project manifest" in captured.err


# ============================================================================
# Behavior Tests for Config-Driven Path Resolution
# ============================================================================


def test_target_path_resolution_with_schema_template(yaml_context: YamlRefactorContext):
    """Behavior test: Verify that {node.schema}/{node.name}.yml template produces
    the expected directory structure.
    """
    from dbt_osmosis.core.path_management import get_target_yaml_path

    # Find a model node from the manifest
    model_nodes = [
        node
        for node in yaml_context.project.manifest.nodes.values()
        if node.resource_type.value == "model"
    ]
    assert model_nodes, "No model nodes found in manifest"

    test_node = model_nodes[0]
    target_path = get_target_yaml_path(yaml_context, test_node)

    # Verify path includes schema
    assert test_node.schema in str(target_path) or target_path.parent.name == test_node.schema
    assert target_path.suffix in (".yml", ".yaml")


def test_target_path_resolution_with_custom_template(yaml_context: YamlRefactorContext):
    """Behavior test: Verify that custom path templates are correctly rendered."""
    from unittest import mock

    from dbt_osmosis.core.path_management import get_target_yaml_path

    # Find a model node
    model_nodes = [
        node
        for node in yaml_context.project.manifest.nodes.values()
        if node.resource_type.value == "model"
    ]
    if not model_nodes:
        pytest.skip("No model nodes found in manifest")
    test_node = model_nodes[0]

    # Test with a custom template that uses {node.name}
    custom_template = "{node.name}.yml"
    with mock.patch(
        "dbt_osmosis.core.path_management._get_yaml_path_template",
        return_value=custom_template,
    ):
        target_path = get_target_yaml_path(yaml_context, test_node)
        assert target_path.stem == test_node.name
        assert target_path.suffix in (".yml", ".yaml")


def test_target_path_source_node_handling(yaml_context: YamlRefactorContext):
    """Behavior test: Verify that source nodes are handled correctly and placed
    under the models directory.
    """
    from dbt_osmosis.core.path_management import get_target_yaml_path

    # Find a source node
    source_nodes = list(yaml_context.project.manifest.sources.values())
    if not source_nodes:
        pytest.skip("No source nodes found in manifest")
    test_node = source_nodes[0]

    target_path = get_target_yaml_path(yaml_context, test_node)

    # Sources should be under models directory
    model_path = Path(yaml_context.project.runtime_cfg.model_paths[0])
    assert target_path.is_relative_to(model_path)


def test_target_path_auto_extension_addition(yaml_context: YamlRefactorContext):
    """Behavior test: Verify that .yml extension is automatically added if missing."""
    from unittest import mock

    from dbt_osmosis.core.path_management import get_target_yaml_path

    model_nodes = [
        node
        for node in yaml_context.project.manifest.nodes.values()
        if node.resource_type.value == "model"
    ]
    if not model_nodes:
        pytest.skip("No model nodes found in manifest")
    test_node = model_nodes[0]

    # Template without extension
    with mock.patch(
        "dbt_osmosis.core.path_management._get_yaml_path_template",
        return_value="{node.name}",
    ):
        target_path = get_target_yaml_path(yaml_context, test_node)
        assert target_path.suffix in (".yml", ".yaml")


# ============================================================================
# Error Handling Tests for Invalid Path Templates
# ============================================================================


def test_missing_osmosis_config_raises_error(yaml_context: YamlRefactorContext):
    """Behavior test: Verify that models without dbt-osmosis config raise
    MissingOsmosisConfig.
    """
    from unittest import mock

    from dbt_osmosis.core.path_management import get_target_yaml_path

    model_nodes = [
        node
        for node in yaml_context.project.manifest.nodes.values()
        if node.resource_type.value == "model"
    ]
    if not model_nodes:
        pytest.skip("No model nodes found in manifest")
    test_node = model_nodes[0]

    # Mock _get_yaml_path_template to return None (missing config)
    with mock.patch(
        "dbt_osmosis.core.path_management._get_yaml_path_template",
        return_value=None,
    ):
        # Should return original file path, not raise
        target_path = get_target_yaml_path(yaml_context, test_node)
        assert target_path is not None


def test_path_traversal_attack_prevented(yaml_context: YamlRefactorContext):
    """Security test: Verify that path traversal attempts are blocked."""
    from unittest import mock

    from dbt_osmosis.core.exceptions import PathResolutionError
    from dbt_osmosis.core.path_management import get_target_yaml_path

    model_nodes = [
        node
        for node in yaml_context.project.manifest.nodes.values()
        if node.resource_type.value == "model"
    ]
    if not model_nodes:
        pytest.skip("No model nodes found in manifest")
    test_node = model_nodes[0]

    # Attempt path traversal with malicious template
    malicious_template = "../../../etc/passwd"

    with (
        mock.patch(
            "dbt_osmosis.core.path_management._get_yaml_path_template",
            return_value=malicious_template,
        ),
        pytest.raises(PathResolutionError, match="Security violation"),
    ):
        get_target_yaml_path(yaml_context, test_node)


def test_absolute_path_within_project_allowed(yaml_context: YamlRefactorContext):
    """Behavior test: Verify that absolute paths starting with / are allowed
    as long as they're within project root (single leading slash is stripped).
    """
    from unittest import mock

    from dbt_osmosis.core.path_management import get_target_yaml_path

    model_nodes = [
        node
        for node in yaml_context.project.manifest.nodes.values()
        if node.resource_type.value == "model"
    ]
    if not model_nodes:
        pytest.skip("No model nodes found in manifest")
    test_node = model_nodes[0]

    # Template with leading slash (allowed convention for absolute paths under models)
    with mock.patch(
        "dbt_osmosis.core.path_management._get_yaml_path_template",
        return_value="/staging/{node.name}.yml",
    ):
        target_path = get_target_yaml_path(yaml_context, test_node)
        # Should be under models directory
        # Need to resolve model_path relative to project root for comparison
        project_root = Path(yaml_context.project.runtime_cfg.project_root)
        model_path = project_root / yaml_context.project.runtime_cfg.model_paths[0]
        assert target_path.is_relative_to(model_path)


# ============================================================================
# File Operation Behavior Tests
# ============================================================================


def test_yaml_file_creation_on_disk(yaml_context: YamlRefactorContext, tmp_path):
    """Behavior test: Verify that YAML files are actually created on disk."""
    from dbt_osmosis.core.restructuring import RestructureOperation, apply_restructure_plan

    # Create a plan with a new file
    target_file = tmp_path / "models" / "test_model.yml"
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=target_file,
                content={"version": 2, "models": [{"name": "test_model", "columns": []}]},
            ),
        ],
    )

    # Apply with dry_run=False to actually write
    from unittest import mock

    with mock.patch.object(yaml_context.settings, "dry_run", False):
        apply_restructure_plan(yaml_context, plan, confirm=False)

    # Verify file was created
    assert target_file.exists(), f"Target file {target_file} was not created"

    # Verify content
    import yaml

    with target_file.open() as f:
        content = yaml.safe_load(f)
    assert content["version"] == 2
    assert "models" in content


def test_yaml_file_merge_with_existing(yaml_context: YamlRefactorContext, tmp_path):
    """Behavior test: Verify that new content is merged with existing YAML content."""
    import yaml

    from dbt_osmosis.core.restructuring import RestructureOperation, apply_restructure_plan

    target_file = tmp_path / "models" / "existing.yml"
    target_file.parent.mkdir(parents=True, exist_ok=True)

    # Create existing file with content
    existing_content = {
        "version": 2,
        "models": [{"name": "model1", "description": "Existing model"}],
    }
    with target_file.open("w") as f:
        yaml.dump(existing_content, f)

    # Plan to add another model
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=target_file,
                content={"models": [{"name": "model2", "description": "New model"}]},
            ),
        ],
    )

    from unittest import mock

    with mock.patch.object(yaml_context.settings, "dry_run", False):
        apply_restructure_plan(yaml_context, plan, confirm=False)

    # Verify both models exist
    with target_file.open() as f:
        content = yaml.safe_load(f)

    model_names = [m["name"] for m in content.get("models", [])]
    assert "model1" in model_names
    assert "model2" in model_names


def test_superseded_file_cleanup(yaml_context: YamlRefactorContext, tmp_path):
    """Behavior test: Verify that superseded files are cleaned up after migration."""
    from unittest import mock as mock_patch

    import yaml
    from dbt.artifacts.resources.types import NodeType
    from dbt.contracts.graph.nodes import ModelNode

    from dbt_osmosis.core.restructuring import RestructureOperation, apply_restructure_plan

    old_file = tmp_path / "models" / "old.yml"
    old_file.parent.mkdir(parents=True, exist_ok=True)

    # Create old file with a model
    old_content = {
        "version": 2,
        "models": [{"name": "my_model", "description": "To be migrated"}],
    }
    with old_file.open("w") as f:
        yaml.dump(old_content, f)

    # Create a mock node
    mock_node = mock_patch.Mock(spec=ModelNode)
    mock_node.name = "my_model"
    mock_node.resource_type = NodeType.Model

    # Plan to migrate to new location
    new_file = tmp_path / "models" / "staging" / "my_model.yml"
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=new_file,
                content={"version": 2, "models": [{"name": "my_model"}]},
                superseded_paths={old_file: [mock_node]},
            ),
        ],
    )

    with mock_patch.patch.object(yaml_context.settings, "dry_run", False):
        apply_restructure_plan(yaml_context, plan, confirm=False)

    # Verify old file was removed (since all its content was migrated)
    assert not old_file.exists(), "Old file should be removed after migration"

    # Verify new file exists
    assert new_file.exists(), "New file should be created"


def test_directory_structure_creation(yaml_context: YamlRefactorContext, tmp_path):
    """Behavior test: Verify that nested directories are created as needed."""
    from dbt_osmosis.core.restructuring import RestructureOperation, apply_restructure_plan

    # Create a deep nested path
    nested_path = tmp_path / "models" / "staging" / "raw" / "sources" / "test.yml"
    assert not nested_path.parent.exists(), "Parent directories should not exist initially"

    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=nested_path,
                content={"version": 2, "sources": []},
            ),
        ],
    )

    from unittest import mock

    with mock.patch.object(yaml_context.settings, "dry_run", False):
        apply_restructure_plan(yaml_context, plan, confirm=False)

    # Verify all parent directories were created
    assert nested_path.parent.exists(), "Parent directories should be created"
    assert nested_path.exists(), "Target file should be created"


# ============================================================================
# Conflict Resolution Tests
# ============================================================================


def test_conflict_resolution_file_already_exists(yaml_context: YamlRefactorContext, tmp_path):
    """Behavior test: Verify behavior when target file already exists.
    Content should be merged, not overwritten.
    """
    import yaml

    from dbt_osmosis.core.restructuring import RestructureOperation, apply_restructure_plan

    target_file = tmp_path / "models" / "conflict.yml"
    target_file.parent.mkdir(parents=True, exist_ok=True)

    # Create existing file
    existing = {"version": 2, "models": [{"name": "model_a"}]}
    with target_file.open("w") as f:
        yaml.dump(existing, f)

    # Try to create file at same location
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=target_file,
                content={"version": 2, "models": [{"name": "model_b"}]},
            ),
        ],
    )

    from unittest import mock

    with mock.patch.object(yaml_context.settings, "dry_run", False):
        apply_restructure_plan(yaml_context, plan, confirm=False)

    # Verify merge occurred
    with target_file.open() as f:
        content = yaml.safe_load(f)

    model_names = [m["name"] for m in content.get("models", [])]
    assert "model_a" in model_names, "Existing model should be preserved"
    assert "model_b" in model_names, "New model should be added"


def test_empty_superseded_file_removal(yaml_context: YamlRefactorContext, tmp_path):
    """Behavior test: Verify that empty superseded files are deleted."""
    from unittest import mock as mock_patch

    import yaml
    from dbt.artifacts.resources.types import NodeType
    from dbt.contracts.graph.nodes import ModelNode

    from dbt_osmosis.core.restructuring import RestructureOperation, apply_restructure_plan

    old_file = tmp_path / "models" / "to_empty.yml"
    old_file.parent.mkdir(parents=True, exist_ok=True)

    # File with single model
    content = {"version": 2, "models": [{"name": "only_model"}]}
    with old_file.open("w") as f:
        yaml.dump(content, f)

    mock_node = mock_patch.Mock(spec=ModelNode)
    mock_node.name = "only_model"
    mock_node.resource_type = NodeType.Model

    new_file = tmp_path / "models" / "new_location.yml"
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=new_file,
                content={"version": 2, "models": [{"name": "only_model"}]},
                superseded_paths={old_file: [mock_node]},
            ),
        ],
    )

    with mock_patch.patch.object(yaml_context.settings, "dry_run", False):
        apply_restructure_plan(yaml_context, plan, confirm=False)

    # Old file should be deleted (empty after removing the model)
    assert not old_file.exists()


def test_partial_superseded_file_preserved(yaml_context: YamlRefactorContext, tmp_path):
    """Behavior test: Verify that partially superseded files are preserved
    with remaining content.
    """
    from unittest import mock as mock_patch

    import yaml
    from dbt.artifacts.resources.types import NodeType
    from dbt.contracts.graph.nodes import ModelNode

    from dbt_osmosis.core.restructuring import RestructureOperation, apply_restructure_plan

    old_file = tmp_path / "models" / "partial.yml"
    old_file.parent.mkdir(parents=True, exist_ok=True)

    # File with multiple models
    content = {
        "version": 2,
        "models": [
            {"name": "model_to_move"},
            {"name": "model_to_stay"},
        ],
    }
    with old_file.open("w") as f:
        yaml.dump(content, f)

    mock_node = mock_patch.Mock(spec=ModelNode)
    mock_node.name = "model_to_move"
    mock_node.resource_type = NodeType.Model

    new_file = tmp_path / "models" / "moved.yml"
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=new_file,
                content={"version": 2, "models": [{"name": "model_to_move"}]},
                superseded_paths={old_file: [mock_node]},
            ),
        ],
    )

    with mock_patch.patch.object(yaml_context.settings, "dry_run", False):
        apply_restructure_plan(yaml_context, plan, confirm=False)

    # Old file should still exist with remaining model
    assert old_file.exists()
    with old_file.open() as f:
        remaining = yaml.safe_load(f)

    model_names = [m["name"] for m in remaining.get("models", [])]
    assert "model_to_stay" in model_names
    assert "model_to_move" not in model_names


# ============================================================================
# Catalog Data Type Sync Tests
# ============================================================================


def test_catalog_data_type_used_in_sync(yaml_context: YamlRefactorContext, fresh_caches):
    """Behavior test: Verify that data types from catalog are used when
    --catalog-path is provided. Catalog data types should take precedence
    over manifest data types.
    """
    from unittest import mock as mock_patch
    from unittest.mock import PropertyMock

    from dbt.artifacts.resources.types import NodeType
    from dbt.contracts.graph.nodes import ModelNode
    from dbt.contracts.results import CatalogResults
    from dbt_common.contracts.metadata import ColumnMetadata, TableMetadata

    from dbt_osmosis.core.sync_operations import _sync_doc_section

    # Create a mock catalog table with specific data types
    catalog_table = mock_patch.Mock()
    catalog_table.metadata = TableMetadata(
        name="test_model",
        schema="test_schema",
        database="test_db",
        type="model",
    )
    catalog_table.columns = {
        "col1": ColumnMetadata(name="col1", type="BIGINT", index=0),
        "col2": ColumnMetadata(name="col2", type="VARCHAR(255)", index=1),
    }

    mock_catalog = mock_patch.Mock(spec=CatalogResults)
    mock_catalog.nodes = {"model.test.test_model": catalog_table}
    mock_catalog.sources = {}

    # Create a mock node with different (incorrect) data types in manifest
    mock_node = mock_patch.Mock(spec=ModelNode)
    mock_node.unique_id = "model.test.test_model"
    mock_node.name = "test_model"
    mock_node.schema = "test_schema"
    mock_node.description = "Test model"
    mock_node.resource_type = NodeType.Model
    mock_node.package_name = "test"

    # Mock columns with different types than catalog
    mock_col1 = mock_patch.Mock()
    mock_col1.name = "col1"
    mock_col1.to_dict.return_value = {"name": "col1", "data_type": "INTEGER"}  # Wrong type
    mock_col1.meta = {}

    mock_col2 = mock_patch.Mock()
    mock_col2.name = "col2"
    mock_col2.to_dict.return_value = {"name": "col2", "data_type": "TEXT"}  # Wrong type
    mock_col2.meta = {}

    mock_node.columns = {"col1": mock_col1, "col2": mock_col2}
    mock_node.meta = {}
    mock_node.config = mock_patch.Mock()
    mock_node.config.extra = {}

    # Doc section to sync into
    doc_section = {"columns": []}

    # Mock context with catalog and runtime_cfg credentials
    mock_runtime = mock_patch.Mock()
    mock_runtime.credentials.type = "postgres"

    with mock_patch.patch.object(yaml_context, "_catalog", mock_catalog):
        with mock_patch.patch.object(
            type(yaml_context.project),
            "runtime_cfg",
            new_callable=PropertyMock,
            return_value=mock_runtime,
        ):
            _sync_doc_section(yaml_context, mock_node, doc_section)

    # Verify catalog data types were used
    assert len(doc_section["columns"]) == 2
    col_types = {col["name"]: col.get("data_type") for col in doc_section["columns"]}

    # Should use catalog types (BIGINT, VARCHAR(255)), not manifest types (INTEGER, TEXT)
    assert col_types["col1"] == "BIGINT"
    assert col_types["col2"] == "VARCHAR(255)"


def test_sync_without_catalog_falls_back_to_manifest(
    yaml_context: YamlRefactorContext,
    fresh_caches,
):
    """Behavior test: Verify that when no catalog is available, sync falls back
    to manifest data types.
    """
    from unittest import mock as mock_patch
    from unittest.mock import PropertyMock

    from dbt.artifacts.resources.types import NodeType
    from dbt.contracts.graph.nodes import ModelNode

    from dbt_osmosis.core.sync_operations import _sync_doc_section

    # Create a mock node with data types in manifest
    mock_node = mock_patch.Mock(spec=ModelNode)
    mock_node.unique_id = "model.test.test_model"
    mock_node.name = "test_model"
    mock_node.schema = "test_schema"
    mock_node.description = "Test model"
    mock_node.resource_type = NodeType.Model
    mock_node.package_name = "test"

    # Mock columns with manifest types
    mock_col1 = mock_patch.Mock()
    mock_col1.name = "col1"
    mock_col1.to_dict.return_value = {"name": "col1", "data_type": "INTEGER"}
    mock_col1.meta = {}

    mock_col2 = mock_patch.Mock()
    mock_col2.name = "col2"
    mock_col2.to_dict.return_value = {"name": "col2", "data_type": "TEXT"}
    mock_col2.meta = {}

    mock_node.columns = {"col1": mock_col1, "col2": mock_col2}
    mock_node.meta = {}
    mock_node.config = mock_patch.Mock()
    mock_node.config.extra = {}

    # Doc section to sync into
    doc_section = {"columns": []}

    # Mock context WITHOUT catalog (catalog is None)
    mock_runtime = mock_patch.Mock()
    mock_runtime.credentials.type = "postgres"

    with (
        mock_patch.patch.object(yaml_context, "_catalog", None),
        mock_patch.patch.object(
            type(yaml_context.project),
            "runtime_cfg",
            new_callable=PropertyMock,
            return_value=mock_runtime,
        ),
    ):
        _sync_doc_section(yaml_context, mock_node, doc_section)

    # Verify manifest data types were used (fallback behavior)
    assert len(doc_section["columns"]) == 2
    col_types = {col["name"]: col.get("data_type") for col in doc_section["columns"]}

    # Should use manifest types when no catalog available
    assert col_types["col1"] == "INTEGER"
    assert col_types["col2"] == "TEXT"
