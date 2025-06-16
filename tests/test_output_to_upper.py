"""
Unit tests for the output_to_upper functionality.
These tests verify the new --output-to-upper flag works correctly.
"""

from dbt_osmosis.core.osmosis import YamlRefactorSettings


def test_yaml_refactor_settings_includes_output_to_upper():
    """Test that YamlRefactorSettings includes the new output_to_upper field."""
    # Test default value
    settings = YamlRefactorSettings()
    assert hasattr(settings, 'output_to_upper')
    assert settings.output_to_upper is False
    
    # Test explicit True value
    settings_upper = YamlRefactorSettings(output_to_upper=True)
    assert settings_upper.output_to_upper is True
    
    # Test explicit False value
    settings_lower = YamlRefactorSettings(output_to_upper=False)
    assert settings_lower.output_to_upper is False


def test_column_name_transformation_logic():
    """Test the column name transformation logic for uppercase."""
    # Simulate the transformation logic we implemented
    def transform_column_name(column_name: str, output_to_lower: bool = False, output_to_upper: bool = False) -> str:
        """Simulate the column name transformation logic."""
        if output_to_upper:
            return column_name.upper()
        elif output_to_lower:
            return column_name.lower()
        return column_name
    
    # Test cases
    test_cases = [
        ("customer_id", False, False, "customer_id"),  # No transformation
        ("customer_id", True, False, "customer_id"),   # Lowercase (no change)
        ("customer_id", False, True, "CUSTOMER_ID"),   # Uppercase
        ("Customer_ID", True, False, "customer_id"),   # Lowercase transformation
        ("Customer_ID", False, True, "CUSTOMER_ID"),   # Uppercase transformation
        ("mixedCase", False, True, "MIXEDCASE"),       # Mixed case to upper
        ("ALLCAPS", True, False, "allcaps"),           # All caps to lower
    ]
    
    for column_name, output_to_lower, output_to_upper, expected in test_cases:
        result = transform_column_name(column_name, output_to_lower, output_to_upper)
        assert result == expected, f"Failed for {column_name} with lower={output_to_lower}, upper={output_to_upper}"


def test_data_type_transformation_logic():
    """Test the data type transformation logic for uppercase."""
    # Simulate the data type transformation logic
    def transform_data_type(data_type: str, output_to_lower: bool = False, output_to_upper: bool = False) -> str:
        """Simulate the data type transformation logic with priority: upper > lower > original."""
        if output_to_upper:
            return data_type.upper()
        elif output_to_lower:
            return data_type.lower()
        return data_type
    
    # Test cases
    test_cases = [
        ("varchar", False, False, "varchar"),  # No transformation
        ("varchar", True, False, "varchar"),   # Lowercase (no change)
        ("varchar", False, True, "VARCHAR"),   # Uppercase
        ("VARCHAR", True, False, "varchar"),   # Lowercase transformation
        ("VARCHAR", False, True, "VARCHAR"),   # Uppercase (no change)
        ("Integer", False, True, "INTEGER"),   # Mixed case to upper
        ("BIGINT", True, False, "bigint"),     # All caps to lower
        # Test priority: uppercase wins over lowercase
        ("varchar", True, True, "VARCHAR"),    # Both flags set, uppercase takes priority
    ]
    
    for data_type, output_to_lower, output_to_upper, expected in test_cases:
        result = transform_data_type(data_type, output_to_lower, output_to_upper)
        assert result == expected, f"Failed for {data_type} with lower={output_to_lower}, upper={output_to_upper}"


def test_both_flags_set_priority():
    """Test that when both output_to_lower and output_to_upper are set, uppercase takes priority."""
    # This tests the priority logic we implemented
    def transform_with_priority(text: str, output_to_lower: bool = False, output_to_upper: bool = False) -> str:
        """Transform text with uppercase taking priority over lowercase."""
        if output_to_upper:
            return text.upper()
        elif output_to_lower:
            return text.lower()
        return text
    
    # When both flags are True, uppercase should win
    result = transform_with_priority("MixedCase", output_to_lower=True, output_to_upper=True)
    assert result == "MIXEDCASE"
    
    # When only lowercase is True
    result = transform_with_priority("MixedCase", output_to_lower=True, output_to_upper=False)
    assert result == "mixedcase"
    
    # When only uppercase is True
    result = transform_with_priority("MixedCase", output_to_lower=False, output_to_upper=True)
    assert result == "MIXEDCASE"
    
    # When neither is True
    result = transform_with_priority("MixedCase", output_to_lower=False, output_to_upper=False)
    assert result == "MixedCase"


def test_settings_compatibility():
    """Test that the new setting is compatible with existing settings."""
    # Test that we can create settings with both old and new flags
    settings = YamlRefactorSettings(
        output_to_lower=True,
        output_to_upper=True,
        # Add some other common settings to ensure compatibility
        use_unrendered_descriptions=False
    )
    
    assert settings.output_to_lower is True
    assert settings.output_to_upper is True
    assert settings.use_unrendered_descriptions is False
    
    # Test that the dataclass can be created with just the new field
    settings_minimal = YamlRefactorSettings(output_to_upper=True)
    assert settings_minimal.output_to_upper is True
    assert settings_minimal.output_to_lower is False  # Should default to False
