# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

import tempfile
from pathlib import Path

from dbt_osmosis.core.schema.parser import OsmosisYAML, create_yaml_instance


def test_create_yaml_instance_settings():
    """
    Quick check that create_yaml_instance returns a configured YAML object with custom indenting.
    """
    y = create_yaml_instance(indent_mapping=4, indent_sequence=2, indent_offset=0)
    assert y.map_indent == 4
    assert y.sequence_indent == 2
    assert y.sequence_dash_offset == 0
    assert y.width == 100  # default
    assert y.preserve_quotes is False


def test_yaml_parser_preserves_unit_tests():
    """
    Test that OsmosisYAML preserves the unit_tests section when loading YAML files.
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
    """
    Test that OsmosisYAML filters out keys not relevant to dbt-osmosis.
    """
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
