# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false, reportUnusedCallResult=false, reportCallIssue=false

"""Tests for YAML schema validation and auto-fix functionality."""

import tempfile
from pathlib import Path

from dbt_osmosis.core.schema import (
    FormattingValidator,
    ModelValidator,
    SeedValidator,
    SourceValidator,
    StructureValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    auto_fix_yaml,
    validate_yaml_file,
    validate_yaml_structure,
)


class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_is_valid_with_no_issues(self) -> None:
        """Test that ValidationResult is valid with no issues."""
        result = ValidationResult()
        assert result.is_valid is True
        assert result.issues == []

    def test_validation_result_is_invalid_with_errors(self) -> None:
        """Test that ValidationResult is invalid with errors."""
        result = ValidationResult()
        result.add_error("TEST_ERROR", "Test error message")
        assert result.is_valid is False

    def test_validation_result_valid_with_warnings_only(self) -> None:
        """Test that ValidationResult is valid with warnings only."""
        result = ValidationResult()
        result.add_warning("TEST_WARNING", "Test warning message")
        assert result.is_valid is True

    def test_validation_result_get_errors(self) -> None:
        """Test filtering errors from issues."""
        result = ValidationResult()
        result.add_error("ERROR_1", "Error 1")
        result.add_warning("WARN_1", "Warning 1")
        result.add_error("ERROR_2", "Error 2")

        errors = result.get_errors()
        assert len(errors) == 2
        assert errors[0].code == "ERROR_1"
        assert errors[1].code == "ERROR_2"

    def test_validation_result_get_warnings(self) -> None:
        """Test filtering warnings from issues."""
        result = ValidationResult()
        result.add_error("ERROR_1", "Error 1")
        result.add_warning("WARN_1", "Warning 1")
        result.add_warning("WARN_2", "Warning 2")

        warnings = result.get_warnings()
        assert len(warnings) == 2
        assert warnings[0].code == "WARN_1"
        assert warnings[1].code == "WARN_2"

    def test_validation_result_get_fixable(self) -> None:
        """Test filtering fixable issues."""
        result = ValidationResult()
        result.add_error("ERROR_1", "Error 1", fixable=True)
        result.add_warning("WARN_1", "Warning 1", fixable=False)
        result.add_error("ERROR_2", "Error 2", fixable=True)

        fixable = result.get_fixable()
        assert len(fixable) == 2
        assert all(i.fixable for i in fixable)


class TestValidationIssue:
    """Test ValidationIssue class."""

    def test_validation_issue_to_dict(self) -> None:
        """Test converting ValidationIssue to dictionary."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="TEST_CODE",
            message="Test message",
            file_path=Path("/test/path.yml"),
            line_number=42,
            column_name="test_col",
            fixable=True,
            context={"key": "value"},
        )

        d = issue.to_dict()
        assert d["severity"] == "error"
        assert d["code"] == "TEST_CODE"
        assert d["message"] == "Test message"
        assert d["file_path"] == "/test/path.yml"
        assert d["line_number"] == 42
        assert d["column_name"] == "test_col"
        assert d["fixable"] is True
        assert d["context"] == {"key": "value"}

    def test_validation_issue_str_with_location(self) -> None:
        """Test string representation with location."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="TEST_CODE",
            message="Test message",
            file_path=Path("/test/path.yml"),
            line_number=42,
        )
        s = str(issue)
        assert "❌" in s
        assert "/test/path.yml:42" in s
        assert "[TEST_CODE]" in s
        assert "Test message" in s

    def test_validation_issue_str_without_location(self) -> None:
        """Test string representation without location."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="TEST_CODE",
            message="Test message",
        )
        s = str(issue)
        assert "⚠️" in s
        assert "[TEST_CODE]" in s
        assert "Test message" in s


class TestStructureValidator:
    """Test StructureValidator class."""

    def test_validate_with_valid_version(self) -> None:
        """Test validation passes with valid version."""
        validator = StructureValidator()
        data = {"version": 2, "models": [{"name": "test"}]}
        result = validator.validate(data)

        assert result.is_valid

    def test_validate_missing_version(self) -> None:
        """Test validation fails with missing version."""
        validator = StructureValidator()
        data = {"models": [{"name": "test"}]}
        result = validator.validate(data)

        assert not result.is_valid
        errors = result.get_errors()
        assert any(e.code == "MISSING_VERSION" for e in errors)

    def test_validate_invalid_version(self) -> None:
        """Test validation warns with invalid version."""
        validator = StructureValidator()
        data = {"version": 1, "models": [{"name": "test"}]}
        result = validator.validate(data)

        # Should be valid (version is just a warning)
        assert result.is_valid
        warnings = result.get_warnings()
        assert any(e.code == "INVALID_VERSION" for e in warnings)

    def test_validate_no_resources(self) -> None:
        """Test validation fails with no resources."""
        validator = StructureValidator()
        data = {"version": 2}
        result = validator.validate(data)

        assert not result.is_valid
        errors = result.get_errors()
        assert any(e.code == "NO_RESOURCES" for e in errors)


class TestModelValidator:
    """Test ModelValidator class."""

    def test_validate_valid_model(self) -> None:
        """Test validation passes for valid model."""
        validator = ModelValidator()
        data = {
            "version": 2,
            "models": [
                {
                    "name": "customers",
                    "columns": [
                        {"name": "id", "tests": ["unique", "not_null"]},
                        {"name": "name"},
                    ],
                }
            ],
        }
        result = validator.validate(data)

        assert result.is_valid

    def test_validate_model_missing_name(self) -> None:
        """Test validation fails when model is missing name."""
        validator = ModelValidator()
        data = {"version": 2, "models": [{"description": "No name"}]}
        result = validator.validate(data)

        assert not result.is_valid
        errors = result.get_errors()
        assert any(e.code == "MISSING_MODEL_NAME" for e in errors)

    def test_validate_column_missing_name(self) -> None:
        """Test validation fails when column is missing name."""
        validator = ModelValidator()
        data = {
            "version": 2,
            "models": [{"name": "test", "columns": [{"description": "No name"}]}],
        }
        result = validator.validate(data)

        assert not result.is_valid
        errors = result.get_errors()
        assert any(e.code == "MISSING_COLUMN_NAME" for e in errors)

    def test_validate_valid_test_config(self) -> None:
        """Test validation passes for valid test configs."""
        validator = ModelValidator()
        data = {
            "version": 2,
            "models": [
                {
                    "name": "test",
                    "columns": [
                        {
                            "name": "id",
                            "tests": [
                                "unique",
                                "not_null",
                                {"relationships": {"to": "ref('other')", "field": "id"}},
                                {"accepted_values": {"values": ["a", "b", "c"]}},
                            ],
                        }
                    ],
                }
            ],
        }
        result = validator.validate(data)

        assert result.is_valid

    def test_validate_relationships_missing_field(self) -> None:
        """Test validation fails for relationships test missing required field."""
        validator = ModelValidator()
        data = {
            "version": 2,
            "models": [
                {
                    "name": "test",
                    "columns": [
                        {
                            "name": "id",
                            "tests": [{"relationships": {"to": "ref('other')"}}],
                        }
                    ],
                }
            ],
        }
        result = validator.validate(data)

        assert not result.is_valid
        errors = result.get_errors()
        assert any(e.code == "MISSING_RELATIONSHIP_FIELD" for e in errors)

    def test_validate_accepted_values_missing_values(self) -> None:
        """Test validation fails for accepted_values test missing values."""
        validator = ModelValidator()
        data = {
            "version": 2,
            "models": [
                {
                    "name": "test",
                    "columns": [{"name": "status", "tests": [{"accepted_values": {}}]}],
                }
            ],
        }
        result = validator.validate(data)

        assert not result.is_valid
        errors = result.get_errors()
        assert any(e.code == "MISSING_ACCEPTED_VALUES" for e in errors)

    def test_validate_unknown_test(self) -> None:
        """Test validation warns for unknown test name."""
        validator = ModelValidator()
        data = {
            "version": 2,
            "models": [
                {
                    "name": "test",
                    "columns": [{"name": "id", "tests": ["unknown_test"]}],
                }
            ],
        }
        result = validator.validate(data)

        # Unknown tests are warnings, not errors
        warnings = result.get_warnings()
        assert any(e.code == "UNKNOWN_TEST" for e in warnings)


class TestSourceValidator:
    """Test SourceValidator class."""

    def test_validate_valid_source(self) -> None:
        """Test validation passes for valid source."""
        validator = SourceValidator()
        data = {
            "version": 2,
            "sources": [
                {
                    "name": "raw_data",
                    "table": "customers",
                    "columns": [{"name": "id"}],
                }
            ],
        }
        result = validator.validate(data)

        assert result.is_valid

    def test_validate_source_missing_name(self) -> None:
        """Test validation fails when source is missing name."""
        validator = SourceValidator()
        data = {"version": 2, "sources": [{"table": "test"}]}
        result = validator.validate(data)

        assert not result.is_valid
        errors = result.get_errors()
        assert any(e.code == "MISSING_SOURCE_NAME" for e in errors)

    def test_validate_source_missing_table_warning(self) -> None:
        """Test validation warns when source is missing table."""
        validator = SourceValidator()
        data = {"version": 2, "sources": [{"name": "raw_data"}]}
        result = validator.validate(data)

        warnings = result.get_warnings()
        assert any(e.code == "MISSING_SOURCE_TABLE" for e in warnings)


class TestSeedValidator:
    """Test SeedValidator class."""

    def test_validate_valid_seed(self) -> None:
        """Test validation passes for valid seed."""
        validator = SeedValidator()
        data = {"version": 2, "seeds": [{"name": "raw_data"}]}
        result = validator.validate(data)

        assert result.is_valid

    def test_validate_seed_missing_name(self) -> None:
        """Test validation fails when seed is missing name."""
        validator = SeedValidator()
        data = {"version": 2, "seeds": [{"description": "No name"}]}
        result = validator.validate(data)

        assert not result.is_valid
        errors = result.get_errors()
        assert any(e.code == "MISSING_SEED_NAME" for e in errors)


class TestFormattingValidator:
    """Test FormattingValidator class."""

    def test_validate_trailing_whitespace(self) -> None:
        """Test validation detects trailing whitespace."""
        validator = FormattingValidator()
        validator.raw_content = "version: 2  \nmodels: []"
        result = validator.validate({})

        warnings = result.get_warnings()
        assert any(e.code == "TRAILING_WHITESPACE" for e in warnings)

    def test_validate_crlf_line_endings(self) -> None:
        """Test validation detects CRLF line endings."""
        validator = FormattingValidator()
        validator.raw_content = "version: 2\r\nmodels: []"
        result = validator.validate({})

        infos = [i for i in result.issues if i.severity == ValidationSeverity.INFO]
        assert any(e.code == "CRLF_LINE_ENDINGS" for e in infos)

    def test_validate_excessive_blank_lines(self) -> None:
        """Test validation detects excessive blank lines."""
        validator = FormattingValidator()
        validator.raw_content = "version: 2\n\n\n\nmodels: []"
        result = validator.validate({})

        infos = [i for i in result.issues if i.severity == ValidationSeverity.INFO]
        assert any(e.code == "EXCESSIVE_BLANK_LINES" for e in infos)


class TestValidateYamlStructure:
    """Test validate_yaml_structure function."""

    def test_validate_valid_structure(self) -> None:
        """Test validation passes for valid structure."""
        data = {
            "version": 2,
            "models": [
                {
                    "name": "customers",
                    "columns": [
                        {"name": "id", "tests": ["unique"]},
                    ],
                }
            ],
        }
        result = validate_yaml_structure(data)

        assert result.is_valid

    def test_validate_with_custom_validators(self) -> None:
        """Test validation with custom validators."""
        data = {"version": 2, "models": [{"name": "test"}]}

        custom_validator = StructureValidator()
        result = validate_yaml_structure(data, validators=[custom_validator])

        assert result.is_valid


class TestValidateYamlFile:
    """Test validate_yaml_file function."""

    def test_validate_valid_yaml_file(self) -> None:
        """Test validation passes for valid YAML file."""
        yaml_content = """version: 2
models:
  - name: customers
    columns:
      - name: id
        tests:
          - unique
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            result = validate_yaml_file(temp_path)
            assert result.is_valid
        finally:
            temp_path.unlink()

    def test_validate_invalid_yaml_file(self) -> None:
        """Test validation fails for invalid YAML file."""
        yaml_content = """models:
  - name: customers
    # Missing version field
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            result = validate_yaml_file(temp_path)
            assert not result.is_valid
            errors = result.get_errors()
            assert any(e.code == "MISSING_VERSION" for e in errors)
        finally:
            temp_path.unlink()

    def test_validate_malformed_yaml_file(self) -> None:
        """Test validation fails for malformed YAML."""
        yaml_content = """version: 2
models:
  - name: customers
    - invalid yaml structure
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            result = validate_yaml_file(temp_path)
            assert not result.is_valid
            errors = result.get_errors()
            assert any(e.code == "PARSE_ERROR" for e in errors)
        finally:
            temp_path.unlink()


class TestAutoFixYaml:
    """Test auto_fix_yaml function."""

    def test_auto_fix_missing_version(self) -> None:
        """Test auto-fix adds missing version."""
        data = {"models": [{"name": "test"}]}
        result = ValidationResult()
        result.add_error(
            "MISSING_VERSION", "Missing version", fixable=True, context={"suggested_value": 2}
        )

        fixed_data = auto_fix_yaml(data, result)

        assert fixed_data["version"] == 2
        assert len(result.fixes_applied) > 0

    def test_auto_fix_invalid_version(self) -> None:
        """Test auto-fix fixes invalid version."""
        data = {"version": 1, "models": [{"name": "test"}]}
        result = ValidationResult()
        result.add_warning(
            "INVALID_VERSION",
            "Invalid version",
            fixable=True,
            context={"suggested_value": 2},
        )

        fixed_data = auto_fix_yaml(data, result)

        assert fixed_data["version"] == 2
        assert len(result.fixes_applied) > 0

    def test_auto_fix_preserves_other_data(self) -> None:
        """Test auto-fix preserves existing data."""
        data = {
            "version": 1,
            "models": [{"name": "test", "description": "Test model"}],
        }
        result = ValidationResult()
        result.add_warning(
            "INVALID_VERSION",
            "Invalid version",
            fixable=True,
            context={"suggested_value": 2},
        )

        fixed_data = auto_fix_yaml(data, result)

        assert fixed_data["version"] == 2
        assert len(fixed_data["models"]) == 1
        assert fixed_data["models"][0]["name"] == "test"
        assert fixed_data["models"][0]["description"] == "Test model"


class TestIntegration:
    """Integration tests for validation."""

    def test_validate_and_fix_roundtrip(self) -> None:
        """Test complete validate -> fix -> validate roundtrip."""
        import threading

        from dbt_osmosis.core.schema.parser import create_yaml_instance
        from dbt_osmosis.core.schema.reader import _read_yaml
        from dbt_osmosis.core.schema.writer import _write_yaml

        # Start with invalid YAML
        yaml_content = """models:
  - name: customers
    columns:
      - name: id
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            # Initial validation should fail
            result1 = validate_yaml_file(temp_path)
            assert not result1.is_valid

            # Apply fixes
            yaml_handler = create_yaml_instance()
            yaml_handler_lock = threading.Lock()
            data = _read_yaml(yaml_handler, yaml_handler_lock, temp_path)
            fixed_data = auto_fix_yaml(data, result1)

            # Write fixed data
            _write_yaml(yaml_handler, yaml_handler_lock, temp_path, fixed_data)

            # Re-validate should pass (or have fewer errors)
            result2 = validate_yaml_file(temp_path)
            assert len(result2) == 0, f"Auto-fix should resolve all errors, but found: {result2}"
            # At minimum, version should now be present
            data2 = _read_yaml(yaml_handler, yaml_handler_lock, temp_path)
            assert "version" in data2

        finally:
            temp_path.unlink()

    def test_validate_real_world_example(self) -> None:
        """Test validation with real-world dbt YAML."""
        yaml_content = """---
version: 2
models:
  - name: customers
    description: "Customer data"
    columns:
      - name: customer_id
        description: "Unique customer identifier"
        tests:
          - unique
          - not_null
        data_type: INTEGER
      - name: first_name
        description: "Customer first name"
        data_type: VARCHAR
      - name: status
        description: "Customer status"
        tests:
          - accepted_values:
              values: [active, inactive, pending]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            result = validate_yaml_file(temp_path)
            # This should pass validation
            assert result.is_valid or all(
                e.severity != ValidationSeverity.ERROR for e in result.issues
            )

        finally:
            temp_path.unlink()
