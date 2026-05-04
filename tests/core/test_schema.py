# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

import tempfile
from pathlib import Path

import pytest

from dbt_osmosis.core.schema.parser import create_yaml_instance
from dbt_osmosis.core.schema.reader import (
    LRUCache,
    _mark_yaml_caches_dirty,
    _read_yaml,
    _YAML_BUFFER_CACHE,
    _YAML_ORIGINAL_CACHE,
)
from dbt_osmosis.core.schema.writer import _write_yaml, _merge_preserved_sections, commit_yamls


def test_create_yaml_instance_settings():
    """Quick check that create_yaml_instance returns a configured YAML object with custom indenting."""
    y = create_yaml_instance(indent_mapping=4, indent_sequence=2, indent_offset=0)
    assert y.map_indent == 4
    assert y.sequence_indent == 2
    assert y.sequence_dash_offset == 0
    assert y.width == 100  # default
    assert y.preserve_quotes is False


def test_yaml_parser_preserves_unit_tests():
    """Test that OsmosisYAML preserves the unit_tests section when loading YAML files.
    Regression test for https://github.com/z3z1ma/dbt-osmosis/issues/293
    """
    yaml_content = """version: 2

models:
  - name: test_model
    description: "Test model"

unit_tests:
  - name: test_unit_test
    description: "Test unit test"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        temp_path = Path(f.name)

    try:
        yaml_handler = create_yaml_instance()
        data = yaml_handler.load(temp_path)

        # Verify that unit_tests section is preserved
        assert "unit_tests" in data
        assert len(data["unit_tests"]) == 1
        assert data["unit_tests"][0]["name"] == "test_unit_test"
        assert data["unit_tests"][0]["description"] == "Test unit test"
    finally:
        temp_path.unlink()


def test_yaml_parser_filters_unwanted_keys():
    """Test that OsmosisYAML filters out keys not relevant to dbt-osmosis."""
    yaml_content = """version: 2

models:
  - name: test_model

unit_tests:
  - name: test_unit_test

semantic_models:
  - name: test_semantic_model

macros:
  - name: test_macro
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        temp_path = Path(f.name)

    try:
        yaml_handler = create_yaml_instance()
        data = yaml_handler.load(temp_path)

        # Verify that only relevant keys are preserved
        assert "version" in data
        assert "models" in data
        assert "unit_tests" in data
        assert "semantic_models" not in data
        assert "macros" not in data
    finally:
        temp_path.unlink()


def test_merge_preserved_sections():
    """Test that _merge_preserved_sections correctly merges semantic_models and macros."""
    # Filtered data (what dbt-osmosis processes)
    filtered = {
        "version": 2,
        "models": [{"name": "test_model"}],
    }

    # Original data (with semantic_models and macros)
    original = {
        "version": 2,
        "models": [{"name": "test_model"}],
        "semantic_models": [{"name": "test_semantic_model"}],
        "macros": [{"name": "test_macro"}],
    }

    # Merge should restore semantic_models and macros
    merged = _merge_preserved_sections(filtered, original)

    assert "version" in merged
    assert "models" in merged
    assert "semantic_models" in merged
    assert "macros" in merged
    assert merged["semantic_models"] == original["semantic_models"]
    assert merged["macros"] == original["macros"]


def test_yaml_read_write_preserves_semantic_models():
    """Test that semantic_models are preserved through a read-write cycle.
    Regression test for https://github.com/z3z1ma/dbt-osmosis/issues/XXX
    """
    yaml_content = """version: 2

models:
  - name: test_model
    description: "Test model"

semantic_models:
  - name: test_semantic_model
    description: "Test semantic model"
    model: ref('test_model')

macros:
  - name: test_macro
    description: "Test macro"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        temp_path = Path(f.name)

    try:
        # Clear caches to ensure clean state
        if temp_path in _YAML_BUFFER_CACHE:
            del _YAML_BUFFER_CACHE[temp_path]
        if temp_path in _YAML_ORIGINAL_CACHE:
            del _YAML_ORIGINAL_CACHE[temp_path]

        # Create YAML handler
        yaml_handler = create_yaml_instance()
        yaml_handler_lock = __import__("threading").Lock()

        # Read the file (should filter out semantic_models and macros)
        data = _read_yaml(yaml_handler, yaml_handler_lock, temp_path)

        # Verify that semantic_models and macros are filtered during read
        assert "semantic_models" not in data
        assert "macros" not in data
        assert "models" in data

        # Modify the data (e.g., update a model description)
        data["models"][0]["description"] = "Updated test model"

        # Write the data back
        _write_yaml(
            yaml_handler,
            yaml_handler_lock,
            temp_path,
            data,
            dry_run=False,
        )

        # Read the file back using standard YAML to verify semantic_models are preserved
        import ruamel.yaml

        unfiltered_handler = ruamel.yaml.YAML()
        with temp_path.open("r") as f:
            written_data = unfiltered_handler.load(f)

        # Verify that semantic_models and macros are preserved in the written file
        assert "semantic_models" in written_data
        assert "macros" in written_data
        assert written_data["semantic_models"][0]["name"] == "test_semantic_model"
        assert written_data["macros"][0]["name"] == "test_macro"
        # Verify that the model description was updated
        assert written_data["models"][0]["description"] == "Updated test model"

    finally:
        # Clean up caches
        if temp_path in _YAML_BUFFER_CACHE:
            del _YAML_BUFFER_CACHE[temp_path]
        if temp_path in _YAML_ORIGINAL_CACHE:
            del _YAML_ORIGINAL_CACHE[temp_path]
        temp_path.unlink()


def test_yaml_string_representer_none_prefix_colon():
    """Test that string representer handles None prefix_colon correctly.

    Regression test for bug where f-string converted None to "None" string,
    causing incorrect threshold calculation (83 instead of 87).

    The bug caused descriptions between 83-87 characters to not use folded
    style when they should have.
    """
    import io

    yaml = create_yaml_instance()

    # Verify prefix_colon is None (default in ruamel.yaml)
    assert yaml.prefix_colon is None

    # Test that the threshold is calculated correctly
    # Should be: width - len("description: ") = 100 - 13 = 87
    # NOT: width - len("descriptionNone: ") = 100 - 17 = 83
    threshold = yaml.width - len(f"description{yaml.prefix_colon or ''}: ")
    assert threshold == 87, f"Threshold should be 87, got {threshold}"

    # Test actual YAML output
    test_cases = [
        # (length, should_use_folded_style)
        (80, False),  # Under threshold
        (87, False),  # At threshold
        (88, True),  # Over threshold
        (100, True),  # Well over threshold
    ]

    for length, should_fold in test_cases:
        data = {"version": 2, "models": [{"name": "test_model", "description": "x" * length}]}

        output = io.StringIO()
        yaml.dump(data, output)
        result = output.getvalue()

        # Check if folded style is used
        has_folded = ">" in result.split("description:")[1].split("\n")[0]

        assert has_folded == should_fold, (
            f"Description of {length} chars should {'use' if should_fold else 'not use'} "
            f"folded style, but got: {repr(result.split('description:')[1].split(chr(10))[0])}"
        )


def test_yaml_parser_allows_data_tests_and_filters_anchors():
    """Test that the parser allows data_tests but filters anchors (preserved by the writer)."""
    yaml_content = """version: 2

models:
  - name: test_model

anchors:
  - &common_tests
    - not_null
    - unique

data_tests:
  - name: test_data_test
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        temp_path = Path(f.name)

    try:
        yaml_handler = create_yaml_instance()
        data = yaml_handler.load(temp_path)

        # data_tests passes through the parser's allowed_keys
        assert "models" in data
        assert "data_tests" in data
        # anchors is filtered out by the parser but restored by the writer from the original YAML
        assert "anchors" not in data
    finally:
        temp_path.unlink()


def test_merge_preserved_sections_includes_anchors():
    """Test that _merge_preserved_sections preserves anchors alongside semantic_models/macros."""
    filtered = {
        "version": 2,
        "models": [{"name": "test_model"}],
    }

    original = {
        "version": 2,
        "models": [{"name": "test_model"}],
        "semantic_models": [{"name": "test_sm"}],
        "macros": [{"name": "test_macro"}],
        "anchors": [{"name": "common_tests"}],
    }

    merged = _merge_preserved_sections(filtered, original)

    assert "semantic_models" in merged
    assert "macros" in merged
    assert "anchors" in merged
    assert merged["anchors"] == original["anchors"]


def test_merge_preserved_sections_keeps_unknown_top_level_keys():
    """Unknown dbt top-level sections should survive writer merges unchanged."""
    filtered = {
        "version": 2,
        "models": [{"name": "test_model"}],
    }

    original = {
        "version": 2,
        "models": [{"name": "test_model"}],
        "snapshots": [{"name": "customer_snapshot"}],
        "exposures": [{"name": "executive_dashboard"}],
        "groups": [{"name": "analytics"}],
    }

    merged = _merge_preserved_sections(filtered, original)

    assert merged["snapshots"] == original["snapshots"]
    assert merged["exposures"] == original["exposures"]
    assert merged["groups"] == original["groups"]


def test_yaml_read_write_preserves_unknown_top_level_sections():
    """Read/write cycles should preserve unmanaged dbt sections beyond the hard-coded legacy list."""
    import threading

    yaml_content = """version: 2

models:
  - name: test_model
    description: "Test model"

snapshots:
  - name: customer_snapshot
    relation: ref('test_model')

exposures:
  - name: executive_dashboard
    type: dashboard
    owner:
      name: Analytics
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        temp_path = Path(f.name)

    try:
        _clear_cache(_YAML_BUFFER_CACHE, temp_path)
        _clear_cache(_YAML_ORIGINAL_CACHE, temp_path)

        yaml_handler = create_yaml_instance()
        lock = threading.Lock()

        data = _read_yaml(yaml_handler, lock, temp_path)
        data["models"][0]["description"] = "Updated test model"

        _write_yaml(yaml_handler, lock, temp_path, data, dry_run=False)

        import ruamel.yaml

        unfiltered_handler = ruamel.yaml.YAML()
        with temp_path.open("r") as f:
            written_data = unfiltered_handler.load(f)

        assert written_data["models"][0]["description"] == "Updated test model"
        assert written_data["snapshots"][0]["name"] == "customer_snapshot"
        assert written_data["exposures"][0]["name"] == "executive_dashboard"
    finally:
        _clear_cache(_YAML_BUFFER_CACHE, temp_path)
        _clear_cache(_YAML_ORIGINAL_CACHE, temp_path)
        temp_path.unlink(missing_ok=True)


def _clear_cache(cache, key):
    """Helper to safely clear a key from an LRUCache or dict."""
    if key in cache:
        del cache[key]


def test_fresh_caches_preserves_production_cache_instances(fresh_caches):
    """The central cache fixture must clear production cache objects, not replace them."""
    from dbt_osmosis.core import introspection
    from dbt_osmosis.core.schema import reader

    assert isinstance(reader._YAML_BUFFER_CACHE, LRUCache)
    assert isinstance(reader._YAML_ORIGINAL_CACHE, LRUCache)
    assert isinstance(introspection._COLUMN_LIST_CACHE, dict)


def test_write_yaml_dry_run_discards_dirty_buffer_before_fresh_read():
    """A dry-run _write_yaml must not leave mutated YAML visible to later reads."""
    import threading

    yaml_handler = create_yaml_instance()
    lock = threading.Lock()
    mutations: list[int] = []

    _YAML_BUFFER_CACHE.clear()
    _YAML_ORIGINAL_CACHE.clear()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "schema.yml"
            path.write_text(
                "version: 2\nmodels:\n  - name: customers\n    description: original\n",
                encoding="utf-8",
            )

            data = _read_yaml(yaml_handler, lock, path)
            data["models"][0]["description"] = "dry-run mutation"

            _write_yaml(
                yaml_handler,
                lock,
                path,
                data,
                dry_run=True,
                mutation_tracker=mutations.append,
            )

            reloaded = _read_yaml(yaml_handler, lock, path)

            assert mutations == [1]
            assert reloaded["models"][0]["description"] == "original"
    finally:
        _YAML_BUFFER_CACHE.clear()
        _YAML_ORIGINAL_CACHE.clear()


def test_write_yaml_dry_run_discards_buffer_even_without_mutation():
    """A no-op dry-run write should not keep stale YAML cached afterward."""
    import threading

    yaml_handler = create_yaml_instance()
    lock = threading.Lock()
    mutations: list[int] = []

    _YAML_BUFFER_CACHE.clear()
    _YAML_ORIGINAL_CACHE.clear()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "schema.yml"
            path.write_text(
                "version: 2\nmodels:\n  - name: customers\n    description: original\n",
                encoding="utf-8",
            )

            data = _read_yaml(yaml_handler, lock, path)
            _mark_yaml_caches_dirty(path)

            _write_yaml(
                yaml_handler,
                lock,
                path,
                data,
                dry_run=True,
                mutation_tracker=mutations.append,
            )

            path.write_text(
                "version: 2\nmodels:\n  - name: customers\n    description: changed on disk\n",
                encoding="utf-8",
            )
            reloaded = _read_yaml(yaml_handler, lock, path)

            assert mutations == []
            assert reloaded["models"][0]["description"] == "changed on disk"
    finally:
        _YAML_BUFFER_CACHE.clear()
        _YAML_ORIGINAL_CACHE.clear()


def test_write_yaml_allow_overwrite_false_refuses_existing_file():
    """No-clobber writes should fail even when the caller reaches the writer."""
    import threading

    yaml_handler = create_yaml_instance()
    lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "schema.yml"
        path.write_text("version: 2\nmodels: []\n", encoding="utf-8")

        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            _write_yaml(
                yaml_handler,
                lock,
                path,
                {"version": 2, "models": [{"name": "customers"}]},
                allow_overwrite=False,
            )

        assert path.read_text(encoding="utf-8") == "version: 2\nmodels: []\n"


def test_write_yaml_preserves_positional_optional_arguments():
    """Adding no-clobber support must not break existing positional callers."""
    import threading

    yaml_handler = create_yaml_instance()
    lock = threading.Lock()
    mutations: list[int] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "schema.yml"
        path.write_text("version: 2\nmodels: []\n\n", encoding="utf-8")

        _write_yaml(
            yaml_handler,
            lock,
            path,
            {"version": 2, "models": [{"name": "customers"}]},
            False,
            mutations.append,
            True,
        )

        assert mutations == [1]
        assert path.read_text(encoding="utf-8").endswith("\n")
        assert not path.read_text(encoding="utf-8").endswith("\n\n")


def test_commit_yamls_dry_run_discards_buffer_after_tracking_mutation():
    """Dry-run commit_yamls should count changes, then restore disk-backed reads."""
    import threading

    yaml_handler = create_yaml_instance()
    lock = threading.Lock()
    mutations: list[int] = []

    _YAML_BUFFER_CACHE.clear()
    _YAML_ORIGINAL_CACHE.clear()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "schema.yml"
            path.write_text(
                "version: 2\nmodels:\n  - name: customers\n    description: original\n",
                encoding="utf-8",
            )

            data = _read_yaml(yaml_handler, lock, path)
            data["models"][0]["description"] = "buffered dry-run mutation"
            _mark_yaml_caches_dirty(path)

            commit_yamls(yaml_handler, lock, dry_run=True, mutation_tracker=mutations.append)

            reloaded = _read_yaml(yaml_handler, lock, path)

            assert mutations == [1]
            assert reloaded["models"][0]["description"] == "original"
    finally:
        _YAML_BUFFER_CACHE.clear()
        _YAML_ORIGINAL_CACHE.clear()


def test_dry_run_write_does_not_leak_original_preserved_sections():
    """Original-cache state from a dry run must not restore stale preserved sections later."""
    import threading

    yaml_handler = create_yaml_instance()
    lock = threading.Lock()

    _YAML_BUFFER_CACHE.clear()
    _YAML_ORIGINAL_CACHE.clear()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "schema.yml"
            path.write_text(
                "version: 2\n"
                "models:\n"
                "  - name: customers\n"
                "semantic_models:\n"
                "  - name: stale_metricflow_model\n",
                encoding="utf-8",
            )

            data = _read_yaml(yaml_handler, lock, path)
            data["models"][0]["description"] = "dry-run mutation"

            _write_yaml(yaml_handler, lock, path, data, dry_run=True)

            path.write_text(
                "version: 2\nmodels:\n  - name: customers\n",
                encoding="utf-8",
            )

            _write_yaml(
                yaml_handler,
                lock,
                path,
                {"version": 2, "models": [{"name": "orders"}]},
                dry_run=False,
            )

            written = path.read_text(encoding="utf-8")

            assert "stale_metricflow_model" not in written
            assert "semantic_models" not in written
    finally:
        _YAML_BUFFER_CACHE.clear()
        _YAML_ORIGINAL_CACHE.clear()


def test_commit_yamls_keeps_dirty_buffer_entries_until_write():
    """Dirty YAML buffers should survive cache churn until commit writes them."""
    import threading

    yaml_handler = create_yaml_instance()
    lock = threading.Lock()

    _YAML_BUFFER_CACHE.clear()
    _YAML_ORIGINAL_CACHE.clear()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            dirty_path = Path(tmpdir) / "dirty.yml"
            dirty_path.write_text(
                "version: 2\nmodels:\n  - name: dirty_model\n    description: original\n",
            )

            dirty_doc = _read_yaml(yaml_handler, lock, dirty_path)
            dirty_doc["models"][0]["description"] = "updated"
            _mark_yaml_caches_dirty(dirty_path)

            # Force the clean-entry LRU to churn past capacity.
            for index in range(_YAML_BUFFER_CACHE.maxsize):
                clean_path = Path(tmpdir) / f"clean_{index}.yml"
                clean_path.write_text(f"version: 2\nmodels:\n  - name: model_{index}\n")
                _read_yaml(yaml_handler, lock, clean_path)

            assert dirty_path in _YAML_BUFFER_CACHE

            commit_yamls(yaml_handler, lock, dry_run=False)

            assert dirty_path not in _YAML_BUFFER_CACHE
            reloaded = _read_yaml(yaml_handler, lock, dirty_path)
            assert reloaded["models"][0]["description"] == "updated"
    finally:
        _YAML_BUFFER_CACHE.clear()
        _YAML_ORIGINAL_CACHE.clear()


def test_write_yaml_calls_written_file_tracker():
    """Test that _write_yaml calls written_file_tracker on successful write with changes."""
    import threading

    yaml_handler = create_yaml_instance()
    lock = threading.Lock()
    tracked_paths: list[Path] = []

    def tracker(path: Path) -> None:
        tracked_paths.append(path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write("version: 2\n")
        temp_path = Path(f.name)

    try:
        # Clear caches to avoid stale state
        _clear_cache(_YAML_BUFFER_CACHE, temp_path)
        _clear_cache(_YAML_ORIGINAL_CACHE, temp_path)

        # Write different content to trigger a change
        new_data = {"version": 2, "models": [{"name": "test_model"}]}
        _write_yaml(
            yaml_handler,
            lock,
            temp_path,
            new_data,
            dry_run=False,
            written_file_tracker=tracker,
        )

        assert len(tracked_paths) == 1
        assert tracked_paths[0] == temp_path
    finally:
        temp_path.unlink(missing_ok=True)


def test_write_yaml_no_tracker_when_no_changes():
    """Test that written_file_tracker is NOT called when content is unchanged."""
    import io
    import threading

    yaml_handler = create_yaml_instance()
    lock = threading.Lock()
    tracked_paths: list[Path] = []

    def tracker(path: Path) -> None:
        tracked_paths.append(path)

    data = {"version": 2, "models": [{"name": "test_model"}]}

    # Serialize the data to get the exact bytes
    with io.BytesIO() as buf:
        yaml_handler.dump(data, buf)
        content_bytes = buf.getvalue()

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".yml", delete=False) as f:
        f.write(content_bytes)
        temp_path = Path(f.name)

    try:
        # Clear caches
        _clear_cache(_YAML_BUFFER_CACHE, temp_path)
        _clear_cache(_YAML_ORIGINAL_CACHE, temp_path)

        # Write the same content — no change expected
        _write_yaml(
            yaml_handler,
            lock,
            temp_path,
            data,
            dry_run=False,
            written_file_tracker=tracker,
        )

        assert len(tracked_paths) == 0
    finally:
        temp_path.unlink(missing_ok=True)
