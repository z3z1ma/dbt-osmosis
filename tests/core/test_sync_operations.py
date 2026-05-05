# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import stat
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import ruamel.yaml
from dbt.artifacts.resources.types import NodeType

from dbt_osmosis.core.inheritance import _get_node_yaml
from dbt_osmosis.core.exceptions import YamlValidationError
from dbt_osmosis.core.schema.reader import _read_yaml
from dbt_osmosis.core.schema.writer import _write_yaml, commit_yamls
from dbt_osmosis.core.settings import YamlRefactorContext
from dbt_osmosis.core.sync_operations import (
    _finalize_synced_document,
    _get_or_create_model,
    _get_or_create_version,
    _get_or_create_source,
    _get_or_create_source_table,
    _group_sync_nodes,
    _sync_doc_section,
    sync_node_to_yaml,
)


def test_sync_node_to_yaml(yaml_context: YamlRefactorContext, fresh_caches):
    """For a single node, we can confirm that sync_node_to_yaml runs without error,
    using the real file or generating one if missing (in dry_run mode).
    """
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.customers"]
    sync_node_to_yaml(yaml_context, node, commit=False)


def test_sync_doc_section_honors_project_vars_output_to_lower() -> None:
    """Project vars should affect real sync casing behavior."""
    from collections import OrderedDict

    from dbt.contracts.graph.nodes import ColumnInfo

    context = mock.MagicMock()
    context.settings.scaffold_empty_configs = False
    context.settings.skip_add_data_types = False
    context.settings.skip_merge_meta = False
    context.settings.use_unrendered_descriptions = False
    context.settings.prefer_yaml_values = False
    context.settings.output_to_upper = False
    context.settings.output_to_lower = False
    context.fusion_compat = False
    context.placeholders = set()
    context.read_catalog.return_value = None
    context.project.runtime_cfg.credentials.type = "postgres"
    context.project.runtime_cfg.vars = {"dbt-osmosis": {"output-to-lower": True}}

    node = mock.MagicMock()
    node.unique_id = "model.test.test_model"
    node.description = ""
    node.meta = {}
    node.config.extra = {}
    node.config.meta = {}
    node.unrendered_config = {}
    node.columns = OrderedDict({
        "MIXED_CASE": ColumnInfo.from_dict({"name": "MIXED_CASE", "description": ""}),
    })

    doc_section: dict[str, object] = {"name": "test_model"}
    _sync_doc_section(context, node, doc_section)

    assert doc_section["columns"] == [{"name": "mixed_case"}]


def test_sync_doc_section_honors_project_vars_scaffold_empty_configs() -> None:
    """Project vars should keep empty scaffold fields during sync."""
    from collections import OrderedDict

    from dbt.contracts.graph.nodes import ColumnInfo

    context = SimpleNamespace(
        settings=SimpleNamespace(
            scaffold_empty_configs=False,
            skip_add_data_types=False,
            skip_merge_meta=False,
            use_unrendered_descriptions=False,
            prefer_yaml_values=False,
            output_to_upper=False,
            output_to_lower=False,
        ),
        fusion_compat=False,
        placeholders={"Add model description"},
        read_catalog=lambda: None,
        project=SimpleNamespace(
            runtime_cfg=SimpleNamespace(
                credentials=SimpleNamespace(type="postgres"),
                vars={"dbt-osmosis": {"scaffold-empty-configs": True}},
            ),
        ),
    )
    node = SimpleNamespace(
        unique_id="model.test.test_model",
        description="Add model description",
        meta={},
        config=SimpleNamespace(extra={}, meta={}),
        unrendered_config={},
        columns=OrderedDict({
            "id": ColumnInfo.from_dict({"name": "id", "description": ""}),
        }),
    )
    doc_section: dict[str, object] = {"name": "test_model"}

    _sync_doc_section(context, node, doc_section)

    assert doc_section["description"] == "Add model description"
    assert doc_section["columns"] == [{"name": "id", "description": ""}]


def test_sync_doc_section_honors_node_config_scaffold_empty_configs() -> None:
    """Node dbt-osmosis options should keep empty scaffold fields during sync."""
    from collections import OrderedDict

    from dbt.contracts.graph.nodes import ColumnInfo

    context = SimpleNamespace(
        settings=SimpleNamespace(
            scaffold_empty_configs=False,
            skip_add_data_types=False,
            skip_merge_meta=False,
            use_unrendered_descriptions=False,
            prefer_yaml_values=False,
            output_to_upper=False,
            output_to_lower=False,
        ),
        fusion_compat=False,
        placeholders={"Add model description"},
        read_catalog=lambda: None,
        project=SimpleNamespace(
            runtime_cfg=SimpleNamespace(
                credentials=SimpleNamespace(type="postgres"),
                vars={},
            ),
        ),
    )
    node = SimpleNamespace(
        unique_id="model.test.test_model",
        description="Add model description",
        meta={},
        config=SimpleNamespace(
            extra={"dbt-osmosis-options": {"scaffold-empty-configs": True}},
            meta={},
        ),
        unrendered_config={},
        columns=OrderedDict({
            "id": ColumnInfo.from_dict({"name": "id", "description": ""}),
        }),
    )
    doc_section: dict[str, object] = {"name": "test_model"}

    _sync_doc_section(context, node, doc_section)

    assert doc_section["description"] == "Add model description"
    assert doc_section["columns"] == [{"name": "id", "description": ""}]


def test_sync_doc_section_scaffold_empty_configs_falls_back_to_cli_setting() -> None:
    """Without scoped config, sync should preserve existing CLI/default fallback behavior."""
    from collections import OrderedDict

    from dbt.contracts.graph.nodes import ColumnInfo

    def sync_columns(*, cli_scaffold_empty_configs: bool) -> list[dict[str, object]]:
        context = SimpleNamespace(
            settings=SimpleNamespace(
                scaffold_empty_configs=cli_scaffold_empty_configs,
                skip_add_data_types=False,
                skip_merge_meta=False,
                use_unrendered_descriptions=False,
                prefer_yaml_values=False,
                output_to_upper=False,
                output_to_lower=False,
            ),
            fusion_compat=False,
            placeholders=set(),
            read_catalog=lambda: None,
            project=SimpleNamespace(
                runtime_cfg=SimpleNamespace(
                    credentials=SimpleNamespace(type="postgres"),
                    vars={},
                ),
            ),
        )
        node = SimpleNamespace(
            unique_id="model.test.test_model",
            description="",
            meta={},
            config=SimpleNamespace(extra={}, meta={}),
            unrendered_config={},
            columns=OrderedDict({
                "id": ColumnInfo.from_dict({"name": "id", "description": ""}),
            }),
        )
        doc_section: dict[str, object] = {"name": "test_model"}

        _sync_doc_section(context, node, doc_section)

        return doc_section["columns"]

    assert sync_columns(cli_scaffold_empty_configs=False) == [{"name": "id"}]
    assert sync_columns(cli_scaffold_empty_configs=True) == [{"name": "id", "description": ""}]


def test_sync_node_to_yaml_versioned(yaml_context: YamlRefactorContext, fresh_caches):
    """Test syncing a versioned node to YAML."""
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v2"]
    sync_node_to_yaml(yaml_context, node, commit=False)


def test_finalize_synced_document_dry_run_commit_false_discards_cache(tmp_path: Path) -> None:
    """Dry-run sync commit=False must not leave mutated YAML in later fresh reads."""
    from dbt_osmosis.core.schema.parser import create_yaml_instance
    from dbt_osmosis.core.schema.reader import _YAML_BUFFER_CACHE, _YAML_ORIGINAL_CACHE

    yaml_handler = create_yaml_instance()
    lock = threading.Lock()
    path = tmp_path / "schema.yml"
    path.write_text(
        "version: 2\nmodels:\n  - name: customers\n    description: original\n",
        encoding="utf-8",
    )

    _YAML_BUFFER_CACHE.clear()
    _YAML_ORIGINAL_CACHE.clear()
    try:
        doc = _read_yaml(yaml_handler, lock, path)
        doc["models"][0]["description"] = "dry-run sync mutation"
        context = SimpleNamespace(settings=SimpleNamespace(dry_run=True))

        _finalize_synced_document(context, path, doc, commit=False)

        reloaded = _read_yaml(yaml_handler, lock, path)

        assert reloaded["models"][0]["description"] == "original"
    finally:
        _YAML_BUFFER_CACHE.clear()
        _YAML_ORIGINAL_CACHE.clear()


def test_sync_node_to_yaml_versioned_preserves_column_selector(
    yaml_context: YamlRefactorContext,
    fresh_caches,
):
    """Versioned sync should preserve dbt include/exclude selector entries."""
    yaml_context.settings.dry_run = False
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v2"]
    project_dir = Path(yaml_context.project.runtime_cfg.project_root)
    path = project_dir.joinpath(node.patch_path.split("://")[-1])
    yaml_doc = _read_yaml(yaml_context.yaml_handler, yaml_context.yaml_handler_lock, path)
    model = next(model for model in yaml_doc["models"] if model["name"] == node.name)
    version = next(version for version in model["versions"] if version["v"] == 2)
    version["v"] = "2"
    version["columns"].insert(0, {"include": "*", "exclude": ["internal_note"]})

    sync_node_to_yaml(yaml_context, node, commit=False)

    synced_versions = [version for version in model["versions"] if str(version.get("v")) == "2"]
    assert len(synced_versions) == 1
    synced_version = synced_versions[0]
    assert synced_version["columns"][0] == {"include": "*", "exclude": ["internal_note"]}
    assert [column["name"] for column in synced_version["columns"] if "name" in column] == [
        "id",
        "first_name",
        "last_name",
    ]


def test_get_or_create_model_rejects_duplicate_entries_without_deleting_user_content() -> None:
    """Duplicate model entries must fail closed instead of dropping later entries."""
    doc_list = [
        {"name": "customers", "description": "first user-authored description"},
        {"name": "customers", "description": "second user-authored description"},
    ]

    with pytest.raises(YamlValidationError, match="Duplicate YAML model entries.*customers"):
        _get_or_create_model(doc_list, "customers")

    assert [model["description"] for model in doc_list] == [
        "first user-authored description",
        "second user-authored description",
    ]


def test_get_or_create_version_rejects_duplicate_entries_without_deleting_user_content() -> None:
    """Duplicate version entries must fail closed instead of dropping user-authored content."""
    doc_model = {
        "name": "stg_customers",
        "versions": [
            {"v": 2, "description": "first user-authored version"},
            {"v": 2, "description": "second user-authored version"},
        ],
    }

    with pytest.raises(YamlValidationError, match="Duplicate YAML version entries.*stg_customers"):
        _get_or_create_version(doc_model, 2)

    assert [version["description"] for version in doc_model["versions"]] == [
        "first user-authored version",
        "second user-authored version",
    ]


def test_all_node_sync_preflights_duplicates_before_any_write(
    yaml_context: YamlRefactorContext,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """All-node sync should fail on duplicates before committing unrelated files."""
    good_node = SimpleNamespace(
        unique_id="model.jaffle_shop_duckdb.orders",
        package_name=yaml_context.project.runtime_cfg.project_name,
        resource_type=NodeType.Model,
        name="orders",
        version=None,
    )
    bad_node = SimpleNamespace(
        unique_id="model.jaffle_shop_duckdb.customers",
        package_name=yaml_context.project.runtime_cfg.project_name,
        resource_type=NodeType.Model,
        name="customers",
        version=None,
    )
    good_path = tmp_path / "good.yml"
    bad_path = tmp_path / "bad.yml"
    docs = {
        "orders": {"version": 2, "models": [{"name": "orders", "columns": []}]},
        "customers": {
            "version": 2,
            "models": [
                {"name": "customers", "description": "first"},
                {"name": "customers", "description": "second"},
            ],
        },
    }
    written_paths: list[Path] = []

    monkeypatch.setattr(
        "dbt_osmosis.core.sync_operations._group_sync_nodes",
        lambda context: [[good_node], [bad_node]],
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.sync_operations._resolve_sync_yaml_paths",
        lambda context, node: (None, good_path if node.name == "orders" else bad_path),
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.sync_operations._prepare_yaml_document",
        lambda context, node, current_path: docs[node.name],
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.sync_operations._finalize_synced_document",
        lambda context, target_path, doc, *, commit: written_paths.append(target_path),
    )

    with pytest.raises(YamlValidationError, match="Duplicate YAML models entries.*customers"):
        sync_node_to_yaml(yaml_context, node=None, commit=True)

    assert written_paths == []


def test_sync_node_to_yaml_all_versions_share_one_truthful_write(
    yaml_context: YamlRefactorContext,
    fresh_caches,
):
    """Syncing all candidates must update every versioned entry in the shared YAML file.

    Regression coverage for the grouped versioned-model path: older code deduplicated by
    base model name before parallel sync, which meant only one version was refreshed.
    """
    from dbt_osmosis.core.path_management import get_current_yaml_path
    from dbt_osmosis.core.schema.reader import _read_yaml

    version_one = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    version_two = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v2"]

    version_one.columns["customer_id"].description = "v1 synced description"
    version_two.columns["id"].description = "v2 synced description"
    yaml_context.settings.dry_run = False

    project_root = Path(str(yaml_context.project.runtime_cfg.project_root))
    yaml_context.settings.models = [project_root / "models" / "staging" / "jaffle_shop" / "main"]

    sync_node_to_yaml(yaml_context, commit=False)

    target_path = get_current_yaml_path(yaml_context, version_one)
    assert target_path is not None
    doc = _read_yaml(
        yaml_context.yaml_handler,
        yaml_context.yaml_handler_lock,
        target_path,
    )
    model_entry = next(model for model in doc["models"] if model["name"] == "stg_customers")
    versions = {version_doc["v"]: version_doc for version_doc in model_entry["versions"]}

    version_one_columns = {column["name"]: column for column in versions[1]["columns"]}
    version_two_columns = {column["name"]: column for column in versions[2]["columns"]}

    assert version_one_columns["customer_id"]["description"] == "v1 synced description"
    assert version_two_columns["id"]["description"] == "v2 synced description"


def test_group_sync_nodes_serializes_unversioned_nodes_sharing_target_path(
    yaml_context: YamlRefactorContext,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Same-target unversioned nodes must be one scheduled work item."""
    customers = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.customers"]
    orders = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]
    shared_target = tmp_path / "shared_schema.yml"

    def iter_same_target_nodes(context: YamlRefactorContext):
        yield customers.unique_id, customers
        yield orders.unique_id, orders

    monkeypatch.setattr(
        "dbt_osmosis.core.node_filters._iter_candidate_nodes",
        iter_same_target_nodes,
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.path_management.get_current_yaml_path",
        lambda context, node: None,
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.path_management.get_target_yaml_path",
        lambda context, node: shared_target,
    )

    groups = _group_sync_nodes(yaml_context)

    assert groups == [[customers, orders]]


def test_group_sync_nodes_serializes_mixed_versioned_and_unversioned_models(
    yaml_context: YamlRefactorContext,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Same-target versioned and unversioned models must be one scheduled work item."""
    versioned = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    unversioned = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.customers"]
    shared_target = tmp_path / "shared_models.yml"

    def iter_same_target_nodes(context: YamlRefactorContext):
        yield versioned.unique_id, versioned
        yield unversioned.unique_id, unversioned

    monkeypatch.setattr(
        "dbt_osmosis.core.node_filters._iter_candidate_nodes",
        iter_same_target_nodes,
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.path_management.get_current_yaml_path",
        lambda context, node: None,
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.path_management.get_target_yaml_path",
        lambda context, node: shared_target,
    )

    groups = _group_sync_nodes(yaml_context)

    assert groups == [[versioned, unversioned]]


def test_group_sync_nodes_serializes_sources_sharing_target_path(
    yaml_context: YamlRefactorContext,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Same-target source tables must be one scheduled work item."""
    source_nodes = [
        mock.MagicMock(
            unique_id="source.jaffle_shop_duckdb.raw.orders",
            package_name=yaml_context.project.runtime_cfg.project_name,
            resource_type=NodeType.Source,
        ),
        mock.MagicMock(
            unique_id="source.jaffle_shop_duckdb.raw.customers",
            package_name=yaml_context.project.runtime_cfg.project_name,
            resource_type=NodeType.Source,
        ),
    ]
    shared_target = tmp_path / "shared_sources.yml"

    def iter_same_target_sources(context: YamlRefactorContext):
        for source_node in source_nodes:
            yield source_node.unique_id, source_node

    monkeypatch.setattr(
        "dbt_osmosis.core.node_filters._iter_candidate_nodes",
        iter_same_target_sources,
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.path_management.get_current_yaml_path",
        lambda context, node: None,
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.path_management.get_target_yaml_path",
        lambda context, node: shared_target,
    )

    groups = _group_sync_nodes(yaml_context)

    assert groups == [source_nodes]


def test_sync_node_to_yaml_repeated_threads_same_target_preserves_model_sections(
    yaml_context: YamlRefactorContext,
    fresh_caches,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Repeated threaded sync for one target must keep every grouped model section."""
    customers = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.customers"]
    orders = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]
    shared_target = tmp_path / "shared_schema.yml"
    shared_target.write_text("version: 2\nmodels: []\n", encoding="utf-8")
    yaml_context.settings.dry_run = False

    def iter_same_target_nodes(context: YamlRefactorContext):
        yield customers.unique_id, customers
        yield orders.unique_id, orders

    monkeypatch.setattr(
        "dbt_osmosis.core.node_filters._iter_candidate_nodes",
        iter_same_target_nodes,
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.path_management.get_current_yaml_path",
        lambda context, node: None,
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.path_management.get_target_yaml_path",
        lambda context, node: shared_target,
    )

    original_pool = yaml_context.pool
    yaml_context.pool = ThreadPoolExecutor(max_workers=2)
    try:
        for attempt in range(3):
            customers.description = f"customers synced {attempt}"
            orders.description = f"orders synced {attempt}"

            sync_node_to_yaml(yaml_context, commit=False)

            doc = _read_yaml(
                yaml_context.yaml_handler,
                yaml_context.yaml_handler_lock,
                shared_target,
            )
            models = {model["name"]: model for model in doc["models"]}

            assert set(models) >= {"customers", "orders"}
    finally:
        yaml_context.pool.shutdown(wait=True)
        yaml_context.pool = original_pool


def test_write_yaml_uses_unique_temp_path_and_preserves_existing_tmp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed write must not clobber or delete another writer's temp file."""
    target_path = tmp_path / "schema.yml"
    existing_tmp = target_path.with_suffix(target_path.suffix + ".tmp")
    existing_tmp.write_bytes(b"sentinel temp from another writer")

    def fail_replace(temp_path: Path, target_path: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("dbt_osmosis.core.schema.writer._replace_atomically", fail_replace)

    yaml_handler = ruamel.yaml.YAML()
    with pytest.raises(OSError, match="simulated replace failure"):
        _write_yaml(
            yaml_handler,
            threading.Lock(),
            target_path,
            {"version": 2, "models": [{"name": "customers"}]},
        )

    assert existing_tmp.read_bytes() == b"sentinel temp from another writer"


def test_write_yaml_preserves_existing_file_mode(tmp_path: Path) -> None:
    """Atomic replacement should not make existing shared YAML owner-only."""
    target_path = tmp_path / "schema.yml"
    target_path.write_text("version: 2\n", encoding="utf-8")
    target_path.chmod(0o640)

    _write_yaml(
        ruamel.yaml.YAML(),
        threading.Lock(),
        target_path,
        {"version": 2, "models": [{"name": "customers"}]},
    )

    assert stat.S_IMODE(target_path.stat().st_mode) == 0o640


def test_commit_yamls_no_write(yaml_context: YamlRefactorContext):
    """Since dry_run=True, commit_yamls should not actually write anything to disk.
    We just ensure no exceptions are raised.
    """
    commit_yamls(
        yaml_handler=yaml_context.yaml_handler,
        yaml_handler_lock=yaml_context.yaml_handler_lock,
        dry_run=yaml_context.settings.dry_run,
        mutation_tracker=yaml_context.register_mutations,
    )


def test_get_or_create_source_reuses_unique_table_match() -> None:
    """Rendered source names should still reuse one unrendered YAML source entry."""
    doc = {
        "sources": [
            {
                "name": "{{ var('raw_source')[target.name] }}",
                "schema": "analytics",
                "database": "warehouse",
                "tables": [{"name": "orders", "identifier": "raw_orders_prod"}],
            },
        ],
    }

    matched = _get_or_create_source(
        doc,
        source_name="raw_source_prod",
        table_name="orders",
        table_identifier="raw_orders_prod",
        schema_name="analytics",
        database_name="warehouse",
    )

    assert matched["name"] == "{{ var('raw_source')[target.name] }}"
    assert len(doc["sources"]) == 1


def test_get_or_create_source_does_not_guess_across_ambiguous_table_matches() -> None:
    """When multiple sources contain the same table name, sync must not merge into one arbitrarily."""
    doc = {
        "sources": [
            {"name": "source_a", "schema": "raw_a", "tables": [{"name": "events"}]},
            {"name": "source_b", "schema": "raw_b", "tables": [{"name": "events"}]},
        ],
    }

    created = _get_or_create_source(
        doc,
        source_name="resolved_events",
        table_name="events",
        schema_name="raw_c",
    )

    assert created["name"] == "resolved_events"
    assert len(doc["sources"]) == 3


def test_get_or_create_source_uses_schema_to_break_table_match_ties() -> None:
    """Schema metadata should disambiguate repeated table names before sync creates duplicates."""
    doc = {
        "sources": [
            {"name": "source_a", "schema": "raw_a", "tables": [{"name": "events"}]},
            {"name": "source_b", "schema": "raw_b", "tables": [{"name": "events"}]},
        ],
    }

    matched = _get_or_create_source(
        doc,
        source_name="resolved_events",
        table_name="events",
        schema_name="raw_b",
    )

    assert matched["name"] == "source_b"
    assert len(doc["sources"]) == 2


def test_get_or_create_source_table_initializes_missing_tables_list() -> None:
    """Existing source YAML without tables should sync by creating an empty list."""
    doc_source: dict[str, object] = {"name": "raw"}

    table = _get_or_create_source_table(doc_source, "orders")

    assert table == {"name": "orders", "columns": []}
    assert doc_source["tables"] == [table]


def test_get_or_create_source_table_rejects_non_list_tables() -> None:
    """Malformed source tables must fail clearly instead of leaking AttributeError."""
    doc_source: dict[str, object] = {"name": "raw", "tables": {"name": "orders"}}

    with pytest.raises(YamlValidationError, match="source 'raw'.*tables.*list"):
        _get_or_create_source_table(doc_source, "orders")

    assert doc_source["tables"] == {"name": "orders"}


def test_get_or_create_source_rejects_non_list_tables_when_matching_by_table() -> None:
    """Source matching by table should fail clearly for malformed tables entries."""
    doc = {
        "sources": [
            {"name": "templated_raw", "tables": {"name": "orders"}},
        ],
    }

    with pytest.raises(YamlValidationError, match="source 'templated_raw'.*tables.*list"):
        _get_or_create_source(doc, source_name="raw_prod", table_name="orders")

    assert len(doc["sources"]) == 1


def test_preserve_unrendered_descriptions(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that when use_unrendered_descriptions is True, descriptions containing
    doc blocks ({{ doc(...) }} or {% docs %}{% enddocs %}) are preserved instead
    of being replaced with the rendered version from the manifest.

    This addresses GitHub issue #219.
    """
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]

    # Get the original YAML for this node
    original_yaml = _get_node_yaml(yaml_context, node)
    assert original_yaml is not None

    # Find a column to test with (use "status" which has a doc block reference)
    original_columns = original_yaml.get("columns", [])
    status_col = None
    for col in original_columns:
        if col.get("name") == "status":
            status_col = col
            break

    # The status column should have a doc reference in the YAML
    if status_col and "{{ doc(" in status_col.get("description", ""):
        original_description = status_col["description"]

        # Enable use_unrendered_descriptions
        yaml_context.settings.dry_run = False
        yaml_context.settings.use_unrendered_descriptions = True
        yaml_context.settings.force_inherit_descriptions = True

        # Sync the node
        sync_node_to_yaml(yaml_context, node, commit=False)

        # Get the updated YAML
        updated_yaml = _get_node_yaml(yaml_context, node)
        assert updated_yaml is not None

        # Find the status column in the updated YAML
        updated_columns = updated_yaml.get("columns", [])
        updated_status_col = None
        for col in updated_columns:
            if col.get("name") == "status":
                updated_status_col = col
                break

        # The description should still contain the doc reference (unrendered)
        assert updated_status_col is not None
        assert "{{ doc(" in updated_status_col.get("description", ""), (
            "Expected unrendered doc reference to be preserved when "
            "use_unrendered_descriptions is True"
        )
        # Verify the exact doc reference is preserved
        assert updated_status_col["description"] == original_description


def test_prefer_yaml_values_preserves_var_jinja(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that when prefer_yaml_values is True, fields containing {{ var() }}
    jinja templates are preserved instead of being replaced with rendered values.

    This addresses GitHub issue #266.
    """
    from dbt_osmosis.core.inheritance import _get_node_yaml

    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]

    # Get the original YAML and add a mock policy_tags field with var()
    original_yaml = _get_node_yaml(yaml_context, node)
    assert original_yaml is not None

    # Simulate a column with policy_tags containing {{ var() }}
    original_columns = original_yaml.get("columns", [])
    for col in original_columns:
        if col.get("name") == "order_id":
            # Add a policy_tags field with unrendered jinja
            col["policy_tags"] = ['{{ var("policy_tag_order_id") }}']
            break

    # Write the modified YAML back
    import io

    buffer = io.StringIO()
    yaml_context.yaml_handler.dump(original_yaml, buffer)
    with mock.patch("builtins.open", mock.mock_open(read_data=buffer.getvalue())):
        # Enable prefer_yaml_values
        yaml_context.settings.dry_run = False
        yaml_context.settings.prefer_yaml_values = True

        sync_node_to_yaml(yaml_context, node, commit=False)

        # Get the updated YAML
        updated_yaml = _get_node_yaml(yaml_context, node)
        assert updated_yaml is not None

        # Find the order_id column in the updated YAML
        updated_columns = updated_yaml.get("columns", [])
        updated_order_id_col = None
        for col in updated_columns:
            if col.get("name") == "order_id":
                updated_order_id_col = col
                break

        # The policy_tags should still contain the unrendered var() reference
        assert updated_order_id_col is not None
        assert "policy_tags" in updated_order_id_col
        policy_tags = updated_order_id_col["policy_tags"]
        assert any("{{ var(" in str(tag) for tag in policy_tags), (
            "Expected unrendered {{ var() }} reference to be preserved when "
            "prefer_yaml_values is True"
        )


def test_prefer_yaml_values_preserves_env_var_jinja(
    yaml_context: YamlRefactorContext,
    fresh_caches,
):
    """Test that when prefer_yaml_values is True, fields containing {{ env_var() }}
    jinja templates are preserved.
    """
    from dbt_osmosis.core.inheritance import _get_node_yaml

    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]

    # Get the original YAML and add a mock field with env_var()
    original_yaml = _get_node_yaml(yaml_context, node)
    assert original_yaml is not None

    # Simulate a column with meta containing env_var()
    original_columns = original_yaml.get("columns", [])
    for col in original_columns:
        if col.get("name") == "customer_id":
            # Add a meta field with unrendered jinja
            col["meta"] = {"pii": "{{ env_var('PII_LEVEL') }}"}
            break

    # Write the modified YAML back
    import io

    buffer = io.StringIO()
    yaml_context.yaml_handler.dump(original_yaml, buffer)
    with mock.patch("builtins.open", mock.mock_open(read_data=buffer.getvalue())):
        # Enable prefer_yaml_values; disable fusion_compat to test classic format
        yaml_context.settings.dry_run = False
        yaml_context.settings.prefer_yaml_values = True
        yaml_context.settings.fusion_compat = False

        sync_node_to_yaml(yaml_context, node, commit=False)

        # Get the updated YAML
        updated_yaml = _get_node_yaml(yaml_context, node)
        assert updated_yaml is not None

        # Find the customer_id column in the updated YAML
        updated_columns = updated_yaml.get("columns", [])
        updated_customer_id_col = None
        for col in updated_columns:
            if col.get("name") == "customer_id":
                updated_customer_id_col = col
                break

        # The meta.pii should still contain the unrendered env_var() reference
        assert updated_customer_id_col is not None
        assert "meta" in updated_customer_id_col
        assert "{{ env_var(" in updated_customer_id_col["meta"].get("pii", ""), (
            "Expected unrendered {{ env_var() }} reference to be preserved when "
            "prefer_yaml_values is True"
        )


def test_prefer_yaml_values_preserves_all_jinja_patterns(
    yaml_context: YamlRefactorContext,
    fresh_caches,
):
    """Test that prefer_yaml_values preserves all types of jinja templates including:
    - {{ doc() }}
    - {{ var() }}
    - {{ env_var() }}
    - {% docs %}{% enddocs %}
    """
    from dbt_osmosis.core.inheritance import _get_node_yaml

    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]

    # Get the original YAML
    original_yaml = _get_node_yaml(yaml_context, node)
    assert original_yaml is not None

    # Add multiple columns with different jinja patterns
    original_columns = original_yaml.get("columns", [])
    test_cases = [
        ("order_id", "policy_tags", ["{{ var('policy_tag') }}"]),
        ("customer_id", "meta", {"classification": "{{ env_var('CLASS') }}"}),
        ("status", "description", "{% docs status_desc %}{% enddocs %}"),
    ]

    for col_name, field_key, field_value in test_cases:
        for col in original_columns:
            if col.get("name") == col_name:
                col[field_key] = field_value
                break

    # Write the modified YAML back
    import io

    buffer = io.StringIO()
    yaml_context.yaml_handler.dump(original_yaml, buffer)
    with mock.patch("builtins.open", mock.mock_open(read_data=buffer.getvalue())):
        # Enable prefer_yaml_values; disable fusion_compat to test classic format
        yaml_context.settings.dry_run = False
        yaml_context.settings.prefer_yaml_values = True
        yaml_context.settings.fusion_compat = False

        sync_node_to_yaml(yaml_context, node, commit=False)

        # Get the updated YAML
        updated_yaml = _get_node_yaml(yaml_context, node)
        assert updated_yaml is not None

        # Verify all jinja patterns were preserved
        updated_columns = updated_yaml.get("columns", [])
        for col_name, field_key, field_value in test_cases:
            col = next((c for c in updated_columns if c.get("name") == col_name), None)
            assert col is not None, f"Column {col_name} not found"
            assert field_key in col, f"Field {field_key} not in column {col_name}"

            # Check that jinja patterns are preserved
            if isinstance(field_value, list):
                updated_value = col[field_key]
                assert any("{{ var(" in str(v) for v in updated_value), (
                    f"Expected {{ var() }} to be preserved in {col_name}.{field_key}"
                )
            elif isinstance(field_value, dict):
                updated_value = col[field_key]
                assert any("{{ env_var(" in str(v) for v in updated_value.values()), (
                    f"Expected {{ env_var() }} to be preserved in {col_name}.{field_key}"
                )
            else:
                updated_value = col[field_key]
                assert "{% docs " in updated_value, (
                    f"Expected {{% docs %}} to be preserved in {col_name}.{field_key}"
                )


def test_add_inheritance_for_specified_keys_still_works(
    yaml_context: YamlRefactorContext,
    fresh_caches,
):
    """Test that --add-inheritance-for-specified-keys still works for granular control.
    This ensures backward compatibility with existing functionality.
    """
    from dbt_osmosis.core.inheritance import _get_node_yaml

    # Use add_inheritance_for_specified_keys to inherit policy_tags
    yaml_context.settings.add_inheritance_for_specified_keys = ["policy_tags"]

    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]

    # Add a policy_tags field to the source YAML
    original_yaml = _get_node_yaml(yaml_context, node)
    assert original_yaml is not None

    original_columns = original_yaml.get("columns", [])
    for col in original_columns:
        if col.get("name") == "order_id":
            # Add a policy_tags field with var()
            col["policy_tags"] = ["{{ var('upstream_policy') }}"]
            break

    # Write the modified YAML back
    import io

    buffer = io.StringIO()
    yaml_context.yaml_handler.dump(original_yaml, buffer)
    with mock.patch("builtins.open", mock.mock_open(read_data=buffer.getvalue())):
        yaml_context.settings.dry_run = False
        sync_node_to_yaml(yaml_context, node, commit=False)

        # Get the updated YAML
        updated_yaml = _get_node_yaml(yaml_context, node)
        assert updated_yaml is not None

        # Verify that policy_tags was inherited (unrendered)
        updated_columns = updated_yaml.get("columns", [])
        updated_order_id_col = None
        for col in updated_columns:
            if col.get("name") == "order_id":
                updated_order_id_col = col
                break

        assert updated_order_id_col is not None
        assert "policy_tags" in updated_order_id_col
        assert any("{{ var(" in str(tag) for tag in updated_order_id_col["policy_tags"]), (
            "Expected unrendered {{ var() }} to be inherited via add-inheritance-for-specified-keys"
        )


def _make_empty_node_context():
    """Build minimal mocks for _sync_doc_section with a node that has no columns."""
    context = mock.MagicMock()
    context.settings.scaffold_empty_configs = False
    context.settings.skip_add_data_types = False
    context.settings.skip_merge_meta = False
    context.settings.use_unrendered_descriptions = False
    context.settings.prefer_yaml_values = False
    context.settings.output_to_upper = False
    context.settings.output_to_lower = False
    context.placeholders = set()
    context.project.runtime_cfg.credentials.type = "duckdb"
    context.project.is_dbt_v1_10_or_greater = False
    context.read_catalog.return_value = None

    node = mock.MagicMock()
    node.unique_id = "source.test.my_source.my_table"
    node.description = ""
    node.columns = {}

    return context, node


def test_sync_doc_section_no_columns_key_not_added():
    """When a node has no columns, _sync_doc_section must not add columns: [] to the doc_section."""
    context, node = _make_empty_node_context()
    doc_section: dict = {"name": "my_table"}

    _sync_doc_section(context, node, doc_section)

    assert "columns" not in doc_section, (
        "Expected 'columns' key to be absent when node has no columns"
    )


def test_sync_doc_section_existing_empty_columns_removed():
    """When doc_section already has columns: [] and the node has no columns,
    _sync_doc_section must remove the empty list rather than leaving it in place.

    This covers the case where osmosis previously wrote columns: [] and the user
    has skip-add-source-columns enabled.
    """
    context, node = _make_empty_node_context()
    doc_section: dict = {"name": "my_table", "columns": []}

    _sync_doc_section(context, node, doc_section)

    assert "columns" not in doc_section, (
        "Expected pre-existing 'columns: []' to be removed when node has no columns"
    )


def test_fusion_compat_pushes_meta_into_config(
    yaml_context: YamlRefactorContext,
    fresh_caches,
):
    """Test that when fusion_compat=True, top-level meta/tags are pushed into config block."""
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]

    # Set column metadata at top level
    col = node.columns["customer_id"]
    col.meta = {"owner": "analytics"}
    col.tags = ["pii", "important"]

    yaml_context.settings.fusion_compat = True
    yaml_context.settings.dry_run = False

    # fresh_caches provides a clean buffer cache for the entire test scope
    sync_node_to_yaml(yaml_context, node, commit=False)
    yaml_slice = _get_node_yaml(yaml_context, node)
    assert yaml_slice is not None

    # Find customer_id column
    yaml_col = None
    for c in yaml_slice.get("columns", []):
        if c.get("name") == "customer_id":
            yaml_col = c
            break

    assert yaml_col is not None
    # In fusion mode, meta and tags should be inside config block
    assert "config" in yaml_col, "Expected config block in fusion_compat output"
    config = yaml_col["config"]
    assert "meta" in config, "Expected meta inside config block"
    assert config["meta"]["owner"] == "analytics"
    assert "tags" in config, "Expected tags inside config block"
    assert set(config["tags"]) == {"pii", "important"}
    # Top-level meta/tags should NOT be present
    assert "meta" not in yaml_col, "Top-level meta should not exist in fusion_compat mode"
    assert "tags" not in yaml_col, "Top-level tags should not exist in fusion_compat mode"


def test_classic_mode_strips_config(
    yaml_context: YamlRefactorContext,
    fresh_caches,
):
    """Test that when fusion_compat=False, config block is stripped and meta/tags stay top-level."""
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]

    # Set column metadata at top level
    col = node.columns["customer_id"]
    col.meta = {"owner": "analytics"}
    col.tags = ["pii"]

    yaml_context.settings.fusion_compat = False
    yaml_context.settings.dry_run = False

    # fresh_caches provides a clean buffer cache for the entire test scope
    sync_node_to_yaml(yaml_context, node, commit=False)
    yaml_slice = _get_node_yaml(yaml_context, node)
    assert yaml_slice is not None

    yaml_col = None
    for c in yaml_slice.get("columns", []):
        if c.get("name") == "customer_id":
            yaml_col = c
            break

    assert yaml_col is not None
    # In classic mode, meta and tags should be at top level
    assert "meta" in yaml_col
    assert yaml_col["meta"]["owner"] == "analytics"
    assert "tags" in yaml_col
    assert "pii" in yaml_col["tags"]
    # config block should NOT be present (stripped)
    assert "config" not in yaml_col, "Config block should be stripped in classic mode"
