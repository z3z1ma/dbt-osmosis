# pyright: reportUnknownVariableType=false, reportPrivateImportUsage=false, reportUnknownMemberType=false
# ruff: noqa: E402
"""dbt-osmosis core module with backwards compatibility imports."""

from __future__ import annotations

# LLM functions (require openai extra)
import importlib.util

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

# Natural language generation (from llm.py) - conditional on openai availability
_llm_available = importlib.util.find_spec("openai") is not None

if _llm_available:
    from dbt_osmosis.core.llm import (
        generate_dbt_model_from_nl,
        generate_sql_from_nl,
    )
else:
    # Stub functions that raise helpful errors
    def generate_dbt_model_from_nl(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "Natural language features require OpenAI. "
            "Install with: pip install 'dbt-osmosis[openai]'"
        )

    def generate_sql_from_nl(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "Natural language features require OpenAI. "
            "Install with: pip install 'dbt-osmosis[openai]'"
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

# SQL linting
from dbt_osmosis.core.sql_lint import (
    KeywordCapitalizationRule,
    LintLevel,
    LintResult,
    LintRule,
    LintViolation,
    LineLengthRule,
    QuotedIdentifierRule,
    SQLLinter,
    SelectStarRule,
    TableAliasRule,
    lint_sql_code,
)

# Staging operations - conditional on openai availability
if _llm_available:
    from dbt_osmosis.core.staging import (
        StagingGenerationResult,
        generate_staging_for_all_sources,
        generate_staging_for_source,
        write_staging_files,
    )
else:
    # Stub classes/functions
    class StagingGenerationResult:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise ImportError(
                "Staging generation requires OpenAI. "
                "Install with: pip install 'dbt-osmosis[openai]'"
            )

    def generate_staging_for_all_sources(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "Staging generation requires OpenAI. Install with: pip install 'dbt-osmosis[openai]'"
        )

    def generate_staging_for_source(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "Staging generation requires OpenAI. Install with: pip install 'dbt-osmosis[openai]'"
        )

    def write_staging_files(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "Staging generation requires OpenAI. Install with: pip install 'dbt-osmosis[openai]'"
        )


# Sync operations
from dbt_osmosis.core.sync_operations import (
    sync_node_to_yaml,
)

# Test suggestion operations - conditional on openai availability
if _llm_available:
    from dbt_osmosis.core.test_suggestions import (
        AITestSuggester,
        ModelTestAnalysis,
        TestPatternExtractor,
        TestSuggestion,
        suggest_tests_for_model,
        suggest_tests_for_project,
    )
else:
    # Stub classes/functions
    class AITestSuggester:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise ImportError(
                "AI test suggestions require OpenAI. "
                "Install with: pip install 'dbt-osmosis[openai]'"
            )

    class ModelTestAnalysis:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise ImportError(
                "AI test analysis requires OpenAI. Install with: pip install 'dbt-osmosis[openai]'"
            )

    class TestPatternExtractor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise ImportError(
                "Test pattern extraction requires OpenAI. "
                "Install with: pip install 'dbt-osmosis[openai]'"
            )

    class TestSuggestion:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise ImportError(
                "Test suggestions require OpenAI. Install with: pip install 'dbt-osmosis[openai]'"
            )

    def suggest_tests_for_model(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "AI test suggestions require OpenAI. Install with: pip install 'dbt-osmosis[openai]'"
        )

    def suggest_tests_for_project(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "AI test suggestions require OpenAI. Install with: pip install 'dbt-osmosis[openai]'"
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
    suggest_improved_documentation,
    synchronize_data_types,
    synthesize_missing_documentation_with_openai,
)

# Voice learning and AI co-pilot
from dbt_osmosis.core.voice_learning import (
    ProjectStyleProfile,
    analyze_project_documentation_style,
    extract_style_examples,
    find_similar_documented_nodes,
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
        strip_eof_blank_lines=context.settings.strip_eof_blank_lines,
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
    "suggest_improved_documentation",
    "config_to_namespace",
    "_reload_manifest",
    "_find_first",
    "SettingsResolver",
    "PropertyAccessor",
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
    "StagingGenerationResult",
    "TestPatternExtractor",
    "TestSuggestion",
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
    "AITestSuggester",
    "ModelTestAnalysis",
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
    "generate_dbt_model_from_nl",
    "generate_sql_from_nl",
    "generate_staging_for_all_sources",
    "generate_staging_for_source",
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
    "suggest_tests_for_model",
    "suggest_tests_for_project",
    "sync_node_to_yaml",
    "synchronize_data_types",
    "synthesize_missing_documentation_with_openai",
    "write_staging_files",
    # Voice learning and AI co-pilot
    "ProjectStyleProfile",
    "analyze_project_documentation_style",
    "extract_style_examples",
    "find_similar_documented_nodes",
    # SQL linting
    "LintLevel",
    "LintViolation",
    "LintResult",
    "LintRule",
    "SQLLinter",
    "lint_sql_code",
    "KeywordCapitalizationRule",
    "LineLengthRule",
    "SelectStarRule",
    "TableAliasRule",
    "QuotedIdentifierRule",
]

# Add LLM exports if available
if _llm_available:
    __all__.extend([
        "DocumentationSuggestion",
        "generate_style_aware_column_doc",
        "generate_style_aware_table_doc",
        "suggest_documentation_improvements",
    ])
