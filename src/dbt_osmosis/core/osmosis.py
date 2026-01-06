# pyright: reportUnknownVariableType=false, reportPrivateImportUsage=false, reportAny=false, reportUnknownMemberType=false
"""dbt-osmosis core module with backwards compatibility imports."""

from __future__ import annotations

# Import SqlCompileRunner for test compatibility
from dbt.task.sql import SqlCompileRunner

# Core configuration and project management
from dbt_osmosis.core.config import (
    DbtConfiguration,
    DbtProjectContext,
    _reload_manifest,
    config_to_namespace,
    create_dbt_project_context,
    discover_profiles_dir,
    discover_project_dir,
)

# Inheritance functionality
from dbt_osmosis.core.inheritance import (
    _build_column_knowledge_graph,
    _build_node_ancestor_tree,
    _get_node_yaml,
)

# Introspection utilities
from dbt_osmosis.core.introspection import (
    _COLUMN_LIST_CACHE,
    PropertyAccessor,
    SettingsResolver,
    _find_first,
    _get_setting_for_node,
    _maybe_use_precise_dtype,
    get_columns,
    normalize_column_name,
)

# Node filtering and sorting
from dbt_osmosis.core.node_filters import (
    _topological_sort,
)

# Path management
from dbt_osmosis.core.path_management import (
    MissingOsmosisConfig,
    _get_yaml_path_template,
    build_yaml_file_mapping,
    create_missing_source_yamls,
    get_current_yaml_path,
    get_target_yaml_path,
)

# Plugin system
from dbt_osmosis.core.plugins import (
    FuzzyCaseMatching,
    FuzzyPrefixMatching,
    get_plugin_manager,
)

# Restructuring operations
from dbt_osmosis.core.restructuring import (
    RestructureDeltaPlan,
    RestructureOperation,
    apply_restructure_plan,
    draft_restructure_delta_plan,
    pretty_print_plan,
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

# Sync operations
from dbt_osmosis.core.sync_operations import (
    sync_node_to_yaml,
)

# Transform operations
from dbt_osmosis.core.transforms import (
    apply_semantic_analysis,
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
    "_COLUMN_LIST_CACHE",
    "_YAML_BUFFER_CACHE",
    "DbtConfiguration",
    "DbtProjectContext",
    "EMPTY_STRING",
    "FuzzyCaseMatching",
    "FuzzyPrefixMatching",
    "MissingOsmosisConfig",
    "PropertyAccessor",
    "RestructureDeltaPlan",
    "RestructureOperation",
    "SettingsResolver",
    "SqlCompileRunner",
    "YamlRefactorContext",
    "YamlRefactorSettings",
    "_build_column_knowledge_graph",
    "_build_node_ancestor_tree",
    "_find_first",
    "_get_node_yaml",
    "_get_setting_for_node",
    "_get_yaml_path_template",
    "_maybe_use_precise_dtype",
    "_reload_manifest",
    "_topological_sort",
    "apply_restructure_plan",
    "apply_semantic_analysis",
    "build_yaml_file_mapping",
    "commit_yamls",
    "compile_sql_code",
    "config_to_namespace",
    "create_dbt_project_context",
    "create_missing_source_yamls",
    "create_yaml_instance",
    "discover_profiles_dir",
    "discover_project_dir",
    "draft_restructure_delta_plan",
    "execute_sql_code",
    "get_columns",
    "get_current_yaml_path",
    "get_plugin_manager",
    "get_target_yaml_path",
    "inherit_upstream_column_knowledge",
    "inject_missing_columns",
    "normalize_column_name",
    "pretty_print_plan",
    "remove_columns_not_in_database",
    "sort_columns_alphabetically",
    "sort_columns_as_configured",
    "sort_columns_as_in_database",
    "sync_node_to_yaml",
    "synchronize_data_types",
    "synthesize_missing_documentation_with_openai",
]
