# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

import tempfile
from pathlib import Path

from dbt_osmosis.core.schema.parser import create_yaml_instance


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
