# pyright: reportUnknownVariableType=false, reportPrivateImportUsage=false, reportAny=false, reportUnknownMemberType=false

"""dbt-osmosis core module with backwards compatibility imports."""

from __future__ import annotations

# Core configuration and project management
from dbt_osmosis.core.config import (
    DbtConfiguration,
    DbtProjectContext,
    config_to_namespace,
    create_dbt_project_context,
    discover_project_dir,
    discover_profiles_dir,
    _reload_manifest,
)

# Inheritance functionality
from dbt_osmosis.core.inheritance import (
    _build_node_ancestor_tree,
    _get_node_yaml,
    _build_column_knowledge_graph,
)

# Introspection utilities
from dbt_osmosis.core.introspection import (
    get_columns,
    normalize_column_name,
    _find_first,
    _get_setting_for_node,
    _maybe_use_precise_dtype,
    _COLUMN_LIST_CACHE,
)

# Node filtering and sorting
from dbt_osmosis.core.node_filters import (
    _topological_sort,
)

# Path management
from dbt_osmosis.core.path_management import (
    build_yaml_file_mapping,
    create_missing_source_yamls,
    get_current_yaml_path,
    get_target_yaml_path,
    MissingOsmosisConfig,
    _get_yaml_path_template,
)

# Plugin system
from dbt_osmosis.core.plugins import (
    get_plugin_manager,
    FuzzyCaseMatching,
    FuzzyPrefixMatching,
)

# Restructuring operations
from dbt_osmosis.core.restructuring import (
    apply_restructure_plan,
    draft_restructure_delta_plan,
    pretty_print_plan,
    RestructureOperation,
    RestructureDeltaPlan,
)

# Schema parsing and writing
from dbt_osmosis.core.schema.parser import (
    create_yaml_instance,
)
from dbt_osmosis.core.schema.reader import (
    _YAML_BUFFER_CACHE,
)
from dbt_osmosis.core.schema.writer import (
    commit_yamls as _commit_yamls_impl,
)

# Settings and context
from dbt_osmosis.core.settings import (
    EMPTY_STRING,
    YamlRefactorContext,
    YamlRefactorSettings,
)

# SQL operations
from dbt_osmosis.core.sql_operations import (
    compile_sql_code,
    execute_sql_code,
)

# Import SqlCompileRunner for test compatibility
from dbt.task.sql import SqlCompileRunner

# Sync operations
from dbt_osmosis.core.sync_operations import (
    sync_node_to_yaml,
)

# Transform operations
from dbt_osmosis.core.transforms import (
    inherit_upstream_column_knowledge,
    inject_missing_columns,
    remove_columns_not_in_database,
    sort_columns_alphabetically,
    sort_columns_as_configured,
    sort_columns_as_in_database,
    synchronize_data_types,
    synthesize_missing_documentation_with_openai,
)

# Note: process_node is imported in sql_operations.py where it's used


# Backwards compatibility wrapper for commit_yamls
def commit_yamls(context: YamlRefactorContext) -> None:
    """Backwards compatible wrapper for commit_yamls that accepts only a context."""
    _commit_yamls_impl(
        yaml_handler=context.yaml_handler,
        yaml_handler_lock=context.yaml_handler_lock,
        dry_run=context.settings.dry_run,
        mutation_tracker=context.register_mutations,
    )


# Backwards compatibility exports
__all__ = [
    "discover_project_dir",
    "discover_profiles_dir",
    "DbtConfiguration",
    "DbtProjectContext",
    "create_dbt_project_context",
    "create_yaml_instance",
    "YamlRefactorSettings",
    "YamlRefactorContext",
    "EMPTY_STRING",
    "compile_sql_code",
    "execute_sql_code",
    "normalize_column_name",
    "get_columns",
    "create_missing_source_yamls",
    "get_current_yaml_path",
    "get_target_yaml_path",
    "build_yaml_file_mapping",
    "commit_yamls",
    "draft_restructure_delta_plan",
    "pretty_print_plan",
    "sync_node_to_yaml",
    "apply_restructure_plan",
    "inherit_upstream_column_knowledge",
    "inject_missing_columns",
    "remove_columns_not_in_database",
    "sort_columns_as_in_database",
    "sort_columns_alphabetically",
    "sort_columns_as_configured",
    "synchronize_data_types",
    "synthesize_missing_documentation_with_openai",
    "config_to_namespace",
    "_reload_manifest",
    "_find_first",
    "_get_setting_for_node",
    "_maybe_use_precise_dtype",
    "_topological_sort",
    "MissingOsmosisConfig",
    "_get_yaml_path_template",
    "RestructureOperation",
    "RestructureDeltaPlan",
    "get_plugin_manager",
    "FuzzyCaseMatching",
    "FuzzyPrefixMatching",
    "_build_node_ancestor_tree",
    "_get_node_yaml",
    "_build_column_knowledge_graph",
    "_COLUMN_LIST_CACHE",
    "_YAML_BUFFER_CACHE",
    "SqlCompileRunner",
]

if __name__ == "__main__":
    c = DbtConfiguration(
        project_dir="demo_duckdb", profiles_dir="demo_duckdb", vars={"dbt-osmosis": {}}
    )
    project = create_dbt_project_context(c)
    yaml_context = YamlRefactorContext(
        project, settings=YamlRefactorSettings(use_unrendered_descriptions=True)
    )
    create_missing_source_yamls(yaml_context)
    plan = draft_restructure_delta_plan(yaml_context)
    apply_restructure_plan(yaml_context, plan, confirm=True)

    from dbt_osmosis.core.transforms import TransformPipeline

    pipeline = TransformPipeline()
    pipeline >> inject_missing_columns
    pipeline >> remove_columns_not_in_database
    pipeline >> inherit_upstream_column_knowledge
    pipeline >> sort_columns_as_configured
    pipeline >> synchronize_data_types
    pipeline.commit_mode = "atomic"
    _ = pipeline(context=yaml_context)
