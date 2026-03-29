"""YAML schema validation and auto-fix for dbt-osmosis.

This module provides comprehensive validation for dbt YAML schemas with
automatic fixing capabilities for common issues.
"""

from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import ruamel.yaml

from dbt_osmosis.core.schema.parser import (
    MANAGED_RESOURCE_TOP_LEVEL_KEYS,
    _partition_yaml_top_level_sections,
)

__all__ = [
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "Validator",
    "validate_yaml_file",
    "validate_yaml_structure",
    "auto_fix_yaml",
    "StructureValidator",
    "ModelValidator",
    "SourceValidator",
    "SeedValidator",
    "FormattingValidator",
]


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue found in YAML."""

    severity: ValidationSeverity
    code: str
    message: str
    file_path: Path | None = None
    line_number: int | None = None
    column_name: str | None = None
    fixable: bool = False
    context: dict[str, t.Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return a string representation of the issue."""
        location = ""
        if self.file_path:
            location = str(self.file_path)
            if self.line_number:
                location += f":{self.line_number}"
            if self.column_name:
                location += f" (column: {self.column_name})"
            location += ": "

        severity_icon = {
            ValidationSeverity.ERROR: "❌",
            ValidationSeverity.WARNING: "⚠️ ",
            ValidationSeverity.INFO: "ℹ️ ",
        }[self.severity]

        return f"{severity_icon} {location}[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, t.Any]:
        """Convert issue to dictionary for serialization."""
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "file_path": str(self.file_path) if self.file_path else None,
            "line_number": self.line_number,
            "column_name": self.column_name,
            "fixable": self.fixable,
            "context": self.context,
        }


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    issues: list[ValidationIssue] = field(default_factory=list)
    is_valid: bool = field(init=False)
    fixes_applied: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate validity based on error-level issues."""
        self.is_valid = not any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the result."""
        self.issues.append(issue)
        self.is_valid = not any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    def add_error(
        self,
        code: str,
        message: str,
        file_path: Path | None = None,
        line_number: int | None = None,
        column_name: str | None = None,
        fixable: bool = False,
        context: dict[str, t.Any] | None = None,
    ) -> None:
        """Add an error-level issue."""
        self.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code=code,
                message=message,
                file_path=file_path,
                line_number=line_number,
                column_name=column_name,
                fixable=fixable,
                context=context or {},
            )
        )

    def add_warning(
        self,
        code: str,
        message: str,
        file_path: Path | None = None,
        line_number: int | None = None,
        column_name: str | None = None,
        fixable: bool = False,
        context: dict[str, t.Any] | None = None,
    ) -> None:
        """Add a warning-level issue."""
        self.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=code,
                message=message,
                file_path=file_path,
                line_number=line_number,
                column_name=column_name,
                fixable=fixable,
                context=context or {},
            )
        )

    def add_info(
        self,
        code: str,
        message: str,
        file_path: Path | None = None,
        line_number: int | None = None,
        column_name: str | None = None,
        fixable: bool = False,
        context: dict[str, t.Any] | None = None,
    ) -> None:
        """Add an info-level issue."""
        self.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.INFO,
                code=code,
                message=message,
                file_path=file_path,
                line_number=line_number,
                column_name=column_name,
                fixable=fixable,
                context=context or {},
            )
        )

    def get_errors(self) -> list[ValidationIssue]:
        """Get all error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> list[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def get_fixable(self) -> list[ValidationIssue]:
        """Get all fixable issues."""
        return [i for i in self.issues if i.fixable]

    def __len__(self) -> int:
        """Return the number of issues in the result."""
        return len(self.issues)

    def __bool__(self) -> bool:
        """Return True if validation passed (no errors)."""
        return self.is_valid

    def summary(self) -> str:
        """Return a summary of validation results."""
        errors = len(self.get_errors())
        warnings = len(self.get_warnings())
        fixable = len(self.get_fixable())

        parts = []
        if self.is_valid:
            parts.append("✅ Validation passed")
        else:
            parts.append(f"❌ Validation failed: {errors} error(s)")

        if warnings:
            parts.append(f"{warnings} warning(s)")

        if fixable:
            parts.append(f"{fixable} fixable issue(s)")

        message = ", ".join(parts)
        if self.fixes_applied:
            message += f"\n🔧 Fixes applied: {len(self.fixes_applied)}"

        return message


class Validator:
    """Base class for YAML validators."""

    def __init__(self, auto_fix: bool = False) -> None:
        """Initialize the validator.

        Args:
            auto_fix: Whether to automatically fix issues when possible
        """
        self.auto_fix = auto_fix

    def validate(
        self,
        data: dict[str, t.Any],
        file_path: Path | None = None,
    ) -> ValidationResult:
        """Validate YAML data and return results.

        Args:
            data: Parsed YAML data
            file_path: Optional file path for error reporting

        Returns:
            ValidationResult with any issues found
        """
        result = ValidationResult()
        self._validate(data, result, file_path)
        return result

    def _validate(
        self,
        data: dict[str, t.Any],
        result: ValidationResult,
        file_path: Path | None = None,
    ) -> None:
        """Internal validation method to be overridden by subclasses."""
        raise NotImplementedError


def _default_validators(*, require_managed_resources: bool = True) -> list[Validator]:
    """Build the default validator chain for schema YAML validation."""
    return [
        StructureValidator(require_managed_resources=require_managed_resources),
        ModelValidator(),
        SourceValidator(),
        SeedValidator(),
    ]


class StructureValidator(Validator):
    """Validates basic YAML structure and required fields."""

    def __init__(
        self,
        auto_fix: bool = False,
        *,
        require_managed_resources: bool = True,
    ) -> None:
        super().__init__(auto_fix=auto_fix)
        self.require_managed_resources = require_managed_resources

    def _validate(
        self,
        data: dict[str, t.Any],
        result: ValidationResult,
        file_path: Path | None = None,
    ) -> None:
        """Validate YAML structure.

        Checks:
        - version field is present and valid
        - At least one dbt-osmosis-managed resource section is present when required
        """
        # Check version
        version = data.get("version")
        if version is None:
            result.add_error(
                code="MISSING_VERSION",
                message="Missing required 'version' field",
                file_path=file_path,
                fixable=True,
                context={"suggested_value": 2},
            )
        elif not isinstance(version, (int, float)) or version != 2:
            result.add_warning(
                code="INVALID_VERSION",
                message=f"Invalid version '{version}'. Expected 2.",
                file_path=file_path,
                fixable=True,
                context={"current_value": version, "suggested_value": 2},
            )

        # Check for at least one resource type
        resource_types = list(MANAGED_RESOURCE_TOP_LEVEL_KEYS)
        has_resources = any(data.get(rt) for rt in resource_types)

        if not has_resources and self.require_managed_resources:
            result.add_error(
                code="NO_RESOURCES",
                message=f"YAML file must contain at least one of: {', '.join(resource_types)}",
                file_path=file_path,
            )


class TestConfigValidator(Validator):
    """Shared validation helpers for dbt resource and column test configs."""

    # Valid test names for models
    VALID_TESTS = {
        "unique",
        "not_null",
        "unique_combination_of_columns",
        "relationships",
        "accepted_values",
    }

    def _validate_resource_tests(
        self,
        resource: dict[str, t.Any],
        result: ValidationResult,
        file_path: Path | None,
        resource_kind: str,
        resource_name: str,
    ) -> None:
        """Validate tests defined at the resource level."""
        tests = resource.get("data_tests", resource.get("tests", []))
        if tests:
            self._validate_tests(
                tests,
                result,
                file_path,
                resource_kind=resource_kind,
                resource_name=resource_name,
                column_name=None,
            )

    def _validate_columns(
        self,
        columns: list[dict[str, t.Any]] | t.Any,
        result: ValidationResult,
        file_path: Path | None,
        owner_kind: str,
        owner_name: str,
    ) -> None:
        """Validate column definitions on a dbt resource."""
        if not isinstance(columns, list):
            result.add_error(
                code="INVALID_COLUMNS_TYPE",
                message=f"Columns for {owner_kind} '{owner_name}' must be a list",
                file_path=file_path,
                column_name=owner_name,
            )
            return

        for idx, column in enumerate(columns):
            if not isinstance(column, dict):
                result.add_error(
                    code="INVALID_COLUMN_TYPE",
                    message=f"Column at index {idx} in {owner_kind} '{owner_name}' must be a dictionary",
                    file_path=file_path,
                    column_name=owner_name,
                )
                continue

            # Check column name
            name = column.get("name")
            if not name:
                result.add_error(
                    code="MISSING_COLUMN_NAME",
                    message=f"Column at index {idx} in {owner_kind} '{owner_name}' is missing required 'name' field",
                    file_path=file_path,
                    column_name=owner_name,
                )
            elif not isinstance(name, str):
                result.add_error(
                    code="INVALID_COLUMN_NAME",
                    message=f"Column name must be a string, got {type(name).__name__}",
                    file_path=file_path,
                    column_name=str(name),
                )

            column_name = name if isinstance(name, str) else "<unknown>"
            tests = column.get("data_tests", column.get("tests", []))
            if tests:
                self._validate_tests(
                    tests,
                    result,
                    file_path,
                    resource_kind=owner_kind,
                    resource_name=owner_name,
                    column_name=column_name,
                )

    def _validate_tests(
        self,
        tests: list[dict[str, t.Any] | str] | t.Any,
        result: ValidationResult,
        file_path: Path | None,
        resource_kind: str,
        resource_name: str,
        column_name: str | None,
    ) -> None:
        """Validate test configurations."""
        if not isinstance(tests, list):
            result.add_error(
                code="INVALID_TESTS_TYPE",
                message=f"Tests for {resource_kind} '{resource_name}' must be a list",
                file_path=file_path,
                column_name=column_name,
            )
            return

        for idx, test in enumerate(tests):
            if isinstance(test, str):
                if test not in self.VALID_TESTS:
                    location = (
                        f"column '{column_name}' of {resource_kind} '{resource_name}'"
                        if column_name
                        else f"{resource_kind} '{resource_name}'"
                    )
                    result.add_warning(
                        code="UNKNOWN_TEST",
                        message=f"Unknown test '{test}' in {location}",
                        file_path=file_path,
                        column_name=column_name,
                    )
                continue

            if not isinstance(test, dict):
                result.add_error(
                    code="INVALID_TEST_TYPE",
                    message=f"Test must be a string or dict, got {type(test).__name__}",
                    file_path=file_path,
                    column_name=column_name,
                )
                continue

            if len(test) != 1:
                result.add_warning(
                    code="INVALID_TEST_CONFIG",
                    message=f"Test configuration at index {idx} should have exactly one key",
                    file_path=file_path,
                    column_name=column_name,
                    context={"test_config": test},
                )
                continue

            test_name = next(iter(test))
            test_config = test[test_name]

            if test_name == "relationships":
                self._validate_relationships_test(
                    test_config,
                    result,
                    file_path,
                    resource_kind,
                    resource_name,
                    column_name,
                )
            elif test_name == "accepted_values":
                self._validate_accepted_values_test(
                    test_config,
                    result,
                    file_path,
                    resource_kind,
                    resource_name,
                    column_name,
                )
            elif test_name == "unique_combination_of_columns":
                self._validate_unique_combination_test(
                    test_config,
                    result,
                    file_path,
                    resource_kind,
                    resource_name,
                    column_name,
                )

    def _extract_test_arguments(
        self,
        test_name: str,
        config: t.Any,
        result: ValidationResult,
        file_path: Path | None,
        column_name: str | None,
    ) -> dict[str, t.Any] | None:
        """Support both legacy flat test args and dbt's nested `arguments` shape."""
        if not isinstance(config, dict):
            result.add_error(
                code="INVALID_TEST_CONFIG_TYPE",
                message=f"{test_name} test configuration must be a dictionary",
                file_path=file_path,
                column_name=column_name,
                context={"config_type": type(config).__name__},
            )
            return None

        arguments = config.get("arguments")
        if arguments is None:
            return config
        if not isinstance(arguments, dict):
            result.add_error(
                code="INVALID_TEST_ARGUMENTS_TYPE",
                message=f"{test_name} test 'arguments' must be a dictionary",
                file_path=file_path,
                column_name=column_name,
                context={"arguments_type": type(arguments).__name__},
            )
            return None
        return arguments

    def _validate_relationships_test(
        self,
        config: t.Any,
        result: ValidationResult,
        file_path: Path | None,
        resource_kind: str,
        resource_name: str,
        column_name: str | None,
    ) -> None:
        """Validate relationships test configuration."""
        arguments = self._extract_test_arguments(
            "relationships",
            config,
            result,
            file_path,
            column_name,
        )
        if arguments is None:
            return

        required_fields = ["to", "field"]
        for required_field in required_fields:
            if required_field not in arguments:
                result.add_error(
                    code="MISSING_RELATIONSHIP_FIELD",
                    message=f"relationships test missing required field '{required_field}'",
                    file_path=file_path,
                    column_name=column_name,
                    context={
                        "resource_kind": resource_kind,
                        "resource_name": resource_name,
                        "test": "relationships",
                        "missing_field": required_field,
                    },
                )

    def _validate_accepted_values_test(
        self,
        config: t.Any,
        result: ValidationResult,
        file_path: Path | None,
        resource_kind: str,
        resource_name: str,
        column_name: str | None,
    ) -> None:
        """Validate accepted_values test configuration."""
        arguments = self._extract_test_arguments(
            "accepted_values",
            config,
            result,
            file_path,
            column_name,
        )
        if arguments is None:
            return

        if "values" not in arguments:
            result.add_error(
                code="MISSING_ACCEPTED_VALUES",
                message="accepted_values test missing required 'values' field",
                file_path=file_path,
                column_name=column_name,
            )
            return

        values = arguments["values"]
        if not isinstance(values, list):
            result.add_error(
                code="INVALID_ACCEPTED_VALUES_TYPE",
                message="'values' field must be a list",
                file_path=file_path,
                column_name=column_name,
                context={
                    "resource_kind": resource_kind,
                    "resource_name": resource_name,
                    "values_type": type(values).__name__,
                },
            )
        elif len(values) == 0:
            result.add_warning(
                code="EMPTY_ACCEPTED_VALUES",
                message="'values' list is empty",
                file_path=file_path,
                column_name=column_name,
            )

    def _validate_unique_combination_test(
        self,
        config: t.Any,
        result: ValidationResult,
        file_path: Path | None,
        resource_kind: str,
        resource_name: str,
        column_name: str | None,
    ) -> None:
        """Validate unique_combination_of_columns test configuration."""
        arguments = self._extract_test_arguments(
            "unique_combination_of_columns",
            config,
            result,
            file_path,
            column_name,
        )
        if arguments is None:
            return

        if "combination_of_columns" not in arguments:
            result.add_error(
                code="MISSING_COMBINATION_COLUMNS",
                message="unique_combination_of_columns test missing required 'combination_of_columns' field",
                file_path=file_path,
                column_name=column_name,
            )
            return

        columns = arguments["combination_of_columns"]
        if not isinstance(columns, list):
            result.add_error(
                code="INVALID_COMBINATION_TYPE",
                message="'combination_of_columns' must be a list",
                file_path=file_path,
                column_name=column_name,
            )
        elif len(columns) < 2:
            result.add_warning(
                code="INSUFFICIENT_COMBINATION_COLUMNS",
                message="'combination_of_columns' should have at least 2 columns",
                file_path=file_path,
                column_name=column_name,
            )


class ModelValidator(TestConfigValidator):
    """Validates model definitions."""

    def _validate(
        self,
        data: dict[str, t.Any],
        result: ValidationResult,
        file_path: Path | None = None,
    ) -> None:
        """Validate model definitions.

        Checks:
        - Each model has a name
        - Column names are strings
        - Test configurations are valid
        """
        models = data.get("models", [])
        if not models:
            return

        for idx, model in enumerate(models):
            if not isinstance(model, dict):
                result.add_error(
                    code="INVALID_MODEL_TYPE",
                    message=f"Model at index {idx} must be a dictionary",
                    file_path=file_path,
                )
                continue

            # Check model name
            name = model.get("name")
            if not name:
                result.add_error(
                    code="MISSING_MODEL_NAME",
                    message=f"Model at index {idx} is missing required 'name' field",
                    file_path=file_path,
                )
            elif not isinstance(name, str):
                result.add_error(
                    code="INVALID_MODEL_NAME",
                    message=f"Model name must be a string, got {type(name).__name__}",
                    file_path=file_path,
                )

            model_name = name if isinstance(name, str) else "<unknown>"
            self._validate_resource_tests(model, result, file_path, "model", model_name)
            self._validate_columns(model.get("columns", []), result, file_path, "model", model_name)


class SourceValidator(TestConfigValidator):
    """Validates source definitions."""

    def _validate(
        self,
        data: dict[str, t.Any],
        result: ValidationResult,
        file_path: Path | None = None,
    ) -> None:
        """Validate source definitions.

        Checks:
        - Each source has a name
        - Source tables are properly defined
        - Table and column-level test configs are valid
        """
        sources = data.get("sources", [])
        if not sources:
            return

        for idx, source in enumerate(sources):
            if not isinstance(source, dict):
                result.add_error(
                    code="INVALID_SOURCE_TYPE",
                    message=f"Source at index {idx} must be a dictionary",
                    file_path=file_path,
                )
                continue

            # Check source name
            name = source.get("name")
            if not name:
                result.add_error(
                    code="MISSING_SOURCE_NAME",
                    message=f"Source at index {idx} is missing required 'name' field",
                    file_path=file_path,
                )
            elif not isinstance(name, str):
                result.add_error(
                    code="INVALID_SOURCE_NAME",
                    message=f"Source name must be a string, got {type(name).__name__}",
                    file_path=file_path,
                )

            tables = source.get("tables", [])
            if not tables:
                result.add_warning(
                    code="MISSING_SOURCE_TABLES",
                    message=f"Source '{name}' has no tables defined",
                    file_path=file_path,
                )
                continue

            if not isinstance(tables, list):
                result.add_error(
                    code="INVALID_SOURCE_TABLES_TYPE",
                    message=f"Source '{name}' must define 'tables' as a list",
                    file_path=file_path,
                )
                continue

            source_name = name if isinstance(name, str) else "<unknown>"
            for table_idx, table in enumerate(tables):
                if not isinstance(table, dict):
                    result.add_error(
                        code="INVALID_SOURCE_TABLE_TYPE",
                        message=f"Table at index {table_idx} in source '{source_name}' must be a dictionary",
                        file_path=file_path,
                    )
                    continue

                table_name = table.get("name")
                if not table_name:
                    result.add_error(
                        code="MISSING_SOURCE_TABLE_NAME",
                        message=f"Table at index {table_idx} in source '{source_name}' is missing required 'name' field",
                        file_path=file_path,
                    )
                    continue
                if not isinstance(table_name, str):
                    result.add_error(
                        code="INVALID_SOURCE_TABLE_NAME",
                        message=f"Source table name must be a string, got {type(table_name).__name__}",
                        file_path=file_path,
                    )
                    continue

                self._validate_resource_tests(table, result, file_path, "source table", table_name)
                self._validate_columns(
                    table.get("columns", []),
                    result,
                    file_path,
                    "source table",
                    table_name,
                )


class SeedValidator(TestConfigValidator):
    """Validates seed definitions."""

    def _validate(
        self,
        data: dict[str, t.Any],
        result: ValidationResult,
        file_path: Path | None = None,
    ) -> None:
        """Validate seed definitions.

        Checks:
        - Each seed has a name
        - Seed-level and column-level test configs are valid
        """
        seeds = data.get("seeds", [])
        if not seeds:
            return

        for idx, seed in enumerate(seeds):
            if not isinstance(seed, dict):
                result.add_error(
                    code="INVALID_SEED_TYPE",
                    message=f"Seed at index {idx} must be a dictionary",
                    file_path=file_path,
                )
                continue

            # Check seed name
            name = seed.get("name")
            if not name:
                result.add_error(
                    code="MISSING_SEED_NAME",
                    message=f"Seed at index {idx} is missing required 'name' field",
                    file_path=file_path,
                )
            elif not isinstance(name, str):
                result.add_error(
                    code="INVALID_SEED_NAME",
                    message=f"Seed name must be a string, got {type(name).__name__}",
                    file_path=file_path,
                )

            seed_name = name if isinstance(name, str) else "<unknown>"
            self._validate_resource_tests(seed, result, file_path, "seed", seed_name)
            self._validate_columns(seed.get("columns", []), result, file_path, "seed", seed_name)


class FormattingValidator(Validator):
    """Validates YAML formatting conventions."""

    # Common formatting issues to check
    FORMAT_CHECKS = {
        "trailing_whitespace": re.compile(r" +$"),
        "multiple_blank_lines": re.compile(r"\n\n\n+"),
    }

    def __init__(self, auto_fix: bool = False) -> None:
        """Initialize the formatting validator.

        Args:
            auto_fix: Whether to automatically fix formatting issues
        """
        super().__init__(auto_fix=auto_fix)
        self.raw_content: str | None = None

    def _validate(
        self,
        data: dict[str, t.Any],
        result: ValidationResult,
        file_path: Path | None = None,
    ) -> None:
        """Validate YAML formatting.

        Checks:
        - No trailing whitespace on lines
        - No excessive blank lines (more than 2 consecutive)
        - Proper line endings (LF, not CRLF)
        """
        if self.raw_content is None:
            return

        # Check for trailing whitespace
        for line_idx, line in enumerate(self.raw_content.split("\n"), 1):
            if self.FORMAT_CHECKS["trailing_whitespace"].search(line):
                result.add_warning(
                    code="TRAILING_WHITESPACE",
                    message=f"Line {line_idx} has trailing whitespace",
                    file_path=file_path,
                    line_number=line_idx,
                    fixable=True,
                )

        # Check for excessive blank lines
        if self.FORMAT_CHECKS["multiple_blank_lines"].search(self.raw_content):
            result.add_info(
                code="EXCESSIVE_BLANK_LINES",
                message="File has excessive blank lines (more than 2 consecutive)",
                file_path=file_path,
                fixable=True,
            )

        # Check for CRLF line endings
        if "\r" in self.raw_content:
            result.add_info(
                code="CRLF_LINE_ENDINGS",
                message="File contains CRLF line endings (should be LF)",
                file_path=file_path,
                fixable=True,
            )


def validate_yaml_structure(
    data: dict[str, t.Any],
    file_path: Path | None = None,
    validators: list[Validator] | None = None,
) -> ValidationResult:
    """Validate YAML structure using specified validators.

    Args:
        data: Parsed YAML data
        file_path: Optional file path for error reporting
        validators: List of validators to use. If None, uses default validators.

    Returns:
        ValidationResult with any issues found
    """
    if validators is None:
        validators = _default_validators()

    result = ValidationResult()
    for validator in validators:
        validator_result = validator.validate(data, file_path)
        result.issues.extend(validator_result.issues)
        result.fixes_applied.extend(validator_result.fixes_applied)
        result.is_valid = result.is_valid and validator_result.is_valid

    return result


def validate_yaml_file(
    file_path: Path,
    raw_content: str | None = None,
    validators: list[Validator] | None = None,
) -> ValidationResult:
    """Validate a YAML file.

    Args:
        file_path: Path to the YAML file
        raw_content: Optional raw file content for formatting validation
        validators: List of validators to use

    Returns:
        ValidationResult with any issues found
    """
    try:
        yaml_handler = ruamel.yaml.YAML()
        yaml_handler.preserve_quotes = True
        raw_data = yaml_handler.load(file_path)
        if raw_data is None:
            raw_data = {}
        if not isinstance(raw_data, dict):
            result = ValidationResult()
            result.add_error(
                code="INVALID_TOP_LEVEL_TYPE",
                message=f"YAML file must parse to a dictionary, got {type(raw_data).__name__}",
                file_path=file_path,
            )
            return result

        data, unmanaged_sections = _partition_yaml_top_level_sections(raw_data)
    except Exception as e:
        result = ValidationResult()
        result.add_error(
            code="PARSE_ERROR",
            message=f"Failed to parse YAML: {e}",
            file_path=file_path,
        )
        return result

    require_managed_resources = (
        any(data.get(rt) for rt in MANAGED_RESOURCE_TOP_LEVEL_KEYS) or not unmanaged_sections
    )

    # Add raw content to formatting validator if provided
    if raw_content:
        formatting_validator = FormattingValidator()
        formatting_validator.raw_content = raw_content
        if validators is None:
            validators = _default_validators(require_managed_resources=require_managed_resources)
        validators = [v for v in validators if not isinstance(v, FormattingValidator)]
        validators.append(formatting_validator)
    elif validators is None:
        validators = _default_validators(require_managed_resources=require_managed_resources)

    result = validate_yaml_structure(data, file_path, validators)

    if unmanaged_sections:
        result.add_warning(
            code="UNMANAGED_TOP_LEVEL_KEYS",
            message=(
                "dbt-osmosis preserves but does not validate or mutate these top-level keys: "
                f"{', '.join(sorted(unmanaged_sections))}"
            ),
            file_path=file_path,
            context={"keys": sorted(unmanaged_sections)},
        )

    return result


def auto_fix_yaml(
    data: dict[str, t.Any],
    result: ValidationResult,
) -> dict[str, t.Any]:
    """Apply automatic fixes to YAML data.

    Args:
        data: Parsed YAML data
        result: ValidationResult with fixable issues

    Returns:
        Fixed YAML data
    """
    import copy

    fixed_data = copy.deepcopy(data)

    for issue in result.get_fixable():
        if issue.code == "MISSING_VERSION":
            fixed_data["version"] = issue.context.get("suggested_value", 2)
            result.fixes_applied.append(f"Added version: {fixed_data['version']}")
        elif issue.code == "INVALID_VERSION":
            fixed_data["version"] = issue.context.get("suggested_value", 2)
            result.fixes_applied.append(f"Fixed version: {fixed_data['version']}")

    # Recalculate validity after fixes
    result.is_valid = not any(
        i.severity == ValidationSeverity.ERROR and not i.fixable for i in result.issues
    )

    return fixed_data
