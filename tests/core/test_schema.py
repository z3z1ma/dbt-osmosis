# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

import tempfile
from pathlib import Path

from dbt_osmosis.core.schema.parser import create_yaml_instance
from dbt_osmosis.core.schema.reader import _read_yaml, _YAML_BUFFER_CACHE, _YAML_ORIGINAL_CACHE
from dbt_osmosis.core.schema.writer import _write_yaml, _merge_preserved_sections


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
