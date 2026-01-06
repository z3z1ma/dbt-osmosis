# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from unittest import mock

import pytest

from dbt_osmosis.core.diff import (
    ChangeCategory,
    ChangeSeverity,
    ColumnAdded,
    ColumnRemoved,
    SchemaDiffResult,
)
from dbt_osmosis.core.migration import (
    MigrationFormat,
    MigrationPlan,
    MigrationPlanner,
    MigrationStep,
)
from dbt_osmosis.core.settings import YamlRefactorContext


@pytest.fixture(scope="function")
def fresh_caches():
    """Patches the internal caches so each test starts with a fresh state."""
    with (
        mock.patch("dbt_osmosis.core.introspection._COLUMN_LIST_CACHE", {}),
        mock.patch("dbt_osmosis.core.schema.reader._YAML_BUFFER_CACHE", {}),
    ):
        yield


def test_migration_planner_initialization(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that MigrationPlanner can be initialized with a context."""
    planner = MigrationPlanner(yaml_context)
    assert planner._context == yaml_context
    assert planner._dry_run is False
    assert planner._format == MigrationFormat.SQL


def test_migration_planner_dry_run(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that MigrationPlanner accepts dry_run flag."""
    planner = MigrationPlanner(yaml_context, dry_run=True)
    assert planner._dry_run is True


def test_migration_planner_format(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that MigrationPlanner accepts different formats."""
    planner = MigrationPlanner(yaml_context, format=MigrationFormat.JSON)
    assert planner._format == MigrationFormat.JSON

    planner = MigrationPlanner(yaml_context, format=MigrationFormat.MARKDOWN)
    assert planner._format == MigrationFormat.MARKDOWN


def test_migration_format_enum():
    """Test MigrationFormat enum values."""
    assert MigrationFormat.SQL.value == "sql"
    assert MigrationFormat.JSON.value == "json"
    assert MigrationFormat.MARKDOWN.value == "markdown"


def test_migration_step_creation(yaml_context: YamlRefactorContext):
    """Test MigrationStep creation."""
    from dbt.artifacts.resources.types import NodeType

    node = next(
        (n for n in yaml_context.manifest.nodes.values() if n.resource_type == NodeType.Model),
        None,
    )
    if not node:
        pytest.skip("No model nodes found in manifest")

    change = ColumnAdded(
        category=ChangeCategory.COLUMN_ADDED,
        severity=ChangeSeverity.SAFE,
        node=node,
        description="",
        column_name="new_col",
        data_type="VARCHAR(255)",
    )

    step = MigrationStep(
        description="Add column new_col",
        sql='ALTER TABLE "test_schema"."test_model" ADD COLUMN "new_col" VARCHAR(255);',
        rollback_sql='ALTER TABLE "test_schema"."test_model" DROP COLUMN "new_col";',
        change=change,
        is_breaking=False,
    )

    assert step.description == "Add column new_col"
    assert "ALTER TABLE" in step.sql
    assert "ALTER TABLE" in step.rollback_sql
    assert step.is_breaking is False
    assert str(step) == "Add column new_col"


def test_migration_plan_properties(yaml_context: YamlRefactorContext):
    """Test MigrationPlan properties."""
    from dbt.artifacts.resources.types import NodeType

    node = next(
        (n for n in yaml_context.manifest.nodes.values() if n.resource_type == NodeType.Model),
        None,
    )
    if not node:
        pytest.skip("No model nodes found in manifest")

    safe_change = ColumnAdded(
        category=ChangeCategory.COLUMN_ADDED,
        severity=ChangeSeverity.SAFE,
        node=node,
        description="",
        column_name="safe_col",
        data_type="VARCHAR(255)",
    )

    breaking_change = ColumnRemoved(
        category=ChangeCategory.COLUMN_REMOVED,
        severity=ChangeSeverity.MODERATE,
        node=node,
        description="",
        column_name="breaking_col",
        data_type="INTEGER",
    )

    safe_step = MigrationStep(
        description="Add column safe_col",
        sql="ALTER TABLE test ADD COLUMN safe_col VARCHAR(255);",
        rollback_sql="ALTER TABLE test DROP COLUMN safe_col;",
        change=safe_change,
        is_breaking=False,
    )

    breaking_step = MigrationStep(
        description="Drop column breaking_col",
        sql="ALTER TABLE test DROP COLUMN breaking_col;",
        rollback_sql="-- ROLLBACK: Cannot restore column",
        change=breaking_change,
        is_breaking=True,
    )

    plan = MigrationPlan(
        node_id=node.unique_id,
        node_name=node.name,
        steps=[safe_step, breaking_step],
    )

    assert plan.has_breaking_changes is True
    assert len(plan.safe_steps) == 1
    assert len(plan.breaking_steps) == 1
    assert plan.safe_steps[0] == safe_step
    assert plan.breaking_steps[0] == breaking_step


def test_migration_plan_to_sql(yaml_context: YamlRefactorContext):
    """Test MigrationPlan.to_sql() method."""
    from dbt.artifacts.resources.types import NodeType

    node = next(
        (n for n in yaml_context.manifest.nodes.values() if n.resource_type == NodeType.Model),
        None,
    )
    if not node:
        pytest.skip("No model nodes found in manifest")

    change = ColumnAdded(
        category=ChangeCategory.COLUMN_ADDED,
        severity=ChangeSeverity.SAFE,
        node=node,
        description="",
        column_name="new_col",
        data_type="VARCHAR(255)",
    )

    step = MigrationStep(
        description="Add column new_col",
        sql="ALTER TABLE test ADD COLUMN new_col VARCHAR(255);",
        rollback_sql="ALTER TABLE test DROP COLUMN new_col;",
        change=change,
        is_breaking=False,
    )

    plan = MigrationPlan(
        node_id=node.unique_id,
        node_name=node.name,
        steps=[step],
    )

    sql = plan.to_sql(include_rollback=True)

    assert "BEGIN;" in sql
    assert "COMMIT;" in sql
    assert "ALTER TABLE test ADD COLUMN new_col VARCHAR(255);" in sql
    assert "ALTER TABLE test DROP COLUMN new_col;" in sql
    assert "Rollback Script" in sql


def test_migration_plan_to_markdown(yaml_context: YamlRefactorContext):
    """Test MigrationPlan.to_markdown() method."""
    from dbt.artifacts.resources.types import NodeType

    node = next(
        (n for n in yaml_context.manifest.nodes.values() if n.resource_type == NodeType.Model),
        None,
    )
    if not node:
        pytest.skip("No model nodes found in manifest")

    change = ColumnAdded(
        category=ChangeCategory.COLUMN_ADDED,
        severity=ChangeSeverity.SAFE,
        node=node,
        description="",
        column_name="new_col",
        data_type="VARCHAR(255)",
    )

    step = MigrationStep(
        description="Add column new_col",
        sql="ALTER TABLE test ADD COLUMN new_col VARCHAR(255);",
        rollback_sql="ALTER TABLE test DROP COLUMN new_col;",
        change=change,
        is_breaking=False,
    )

    plan = MigrationPlan(
        node_id=node.unique_id,
        node_name=node.name,
        steps=[step],
    )

    markdown = plan.to_markdown()

    assert f"# Migration Plan: {node.name}" in markdown
    assert node.unique_id in markdown
    assert "Safe changes: 1" in markdown
    assert "Add column new_col" in markdown
    assert "ALTER TABLE test ADD COLUMN new_col VARCHAR(255);" in markdown


def test_migration_plan_to_dict(yaml_context: YamlRefactorContext):
    """Test MigrationPlan.to_dict() method."""
    from dbt.artifacts.resources.types import NodeType

    node = next(
        (n for n in yaml_context.manifest.nodes.values() if n.resource_type == NodeType.Model),
        None,
    )
    if not node:
        pytest.skip("No model nodes found in manifest")

    change = ColumnAdded(
        category=ChangeCategory.COLUMN_ADDED,
        severity=ChangeSeverity.SAFE,
        node=node,
        description="",
        column_name="new_col",
        data_type="VARCHAR(255)",
    )

    step = MigrationStep(
        description="Add column new_col",
        sql="ALTER TABLE test ADD COLUMN new_col VARCHAR(255);",
        rollback_sql="ALTER TABLE test DROP COLUMN new_col;",
        change=change,
        is_breaking=False,
    )

    plan = MigrationPlan(
        node_id=node.unique_id,
        node_name=node.name,
        steps=[step],
    )

    data = plan.to_dict()

    assert data["node_id"] == node.unique_id
    assert data["node_name"] == node.name
    assert "created_at" in data
    assert data["summary"]["total_steps"] == 1
    assert data["summary"]["safe_steps"] == 1
    assert data["summary"]["breaking_steps"] == 0
    assert len(data["steps"]) == 1
    assert data["steps"][0]["description"] == "Add column new_col"
    assert data["steps"][0]["is_breaking"] is False


def test_migration_planner_dialect_detection(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that MigrationPlanner detects the correct dialect."""
    planner = MigrationPlanner(yaml_context)

    # DuckDB is the default for demo_duckdb
    assert planner._dialect in ("duckdb", "postgres")


def test_migration_planner_quote_identifier(yaml_context: YamlRefactorContext, fresh_caches):
    """Test identifier quoting for different SQL dialects."""
    # Test DuckDB (double quotes)
    planner = MigrationPlanner(yaml_context)

    if planner._dialect == "duckdb":
        quoted = planner._quote_identifier("my_schema.my_table")
        assert '"my_schema"."my_table"' == quoted

        quoted = planner._quote_identifier("my_table")
        assert '"my_table"' == quoted


def test_migration_planner_plan_for_result_empty(yaml_context: YamlRefactorContext, fresh_caches):
    """Test planning a result with no changes."""
    planner = MigrationPlanner(yaml_context)

    from dbt.artifacts.resources.types import NodeType

    node = next(
        (n for n in yaml_context.manifest.nodes.values() if n.resource_type == NodeType.Model),
        None,
    )
    if not node:
        pytest.skip("No model nodes found in manifest")

    result = SchemaDiffResult(
        node=node,
        yaml_columns={},
        database_columns={},
        changes=[],
    )

    plan = planner.plan_for_result(result)

    assert plan.node_id == node.unique_id
    assert plan.node_name == node.name
    assert len(plan.steps) == 0


def test_migration_planner_plan_for_result_with_changes(
    yaml_context: YamlRefactorContext, fresh_caches
):
    """Test planning a result with changes."""
    planner = MigrationPlanner(yaml_context)

    from dbt.artifacts.resources.types import NodeType

    node = next(
        (n for n in yaml_context.manifest.nodes.values() if n.resource_type == NodeType.Model),
        None,
    )
    if not node:
        pytest.skip("No model nodes found in manifest")

    change = ColumnAdded(
        category=ChangeCategory.COLUMN_ADDED,
        severity=ChangeSeverity.SAFE,
        node=node,
        description="",
        column_name="new_col",
        data_type="VARCHAR(255)",
    )

    result = SchemaDiffResult(
        node=node,
        yaml_columns={},
        database_columns={},
        changes=[change],
    )

    plan = planner.plan_for_result(result)

    assert plan.node_id == node.unique_id
    assert len(plan.steps) == 1
    assert "new_col" in plan.steps[0].description
    assert "ALTER TABLE" in plan.steps[0].sql


def test_migration_planner_plan_for_results(yaml_context: YamlRefactorContext, fresh_caches):
    """Test planning multiple results."""
    planner = MigrationPlanner(yaml_context)

    from dbt.artifacts.resources.types import NodeType

    nodes = [n for n in yaml_context.manifest.nodes.values() if n.resource_type == NodeType.Model]

    if len(nodes) < 2:
        pytest.skip("Need at least 2 model nodes for this test")

    node1, node2 = nodes[0], nodes[1]

    change1 = ColumnAdded(
        category=ChangeCategory.COLUMN_ADDED,
        severity=ChangeSeverity.SAFE,
        node=node1,
        description="",
        column_name="col1",
        data_type="VARCHAR(255)",
    )

    change2 = ColumnAdded(
        category=ChangeCategory.COLUMN_ADDED,
        severity=ChangeSeverity.SAFE,
        node=node2,
        description="",
        column_name="col2",
        data_type="INTEGER",
    )

    results = {
        node1.unique_id: SchemaDiffResult(
            node=node1,
            yaml_columns={},
            database_columns={},
            changes=[change1],
        ),
        node2.unique_id: SchemaDiffResult(
            node=node2,
            yaml_columns={},
            database_columns={},
            changes=[change2],
        ),
    }

    plans = planner.plan_for_results(results)

    assert len(plans) == 2
    assert node1.unique_id in plans
    assert node2.unique_id in plans
    assert plans[node1.unique_id].node_name == node1.name
    assert plans[node2.unique_id].node_name == node2.name
