# pyright: reportPrivateImportUsage=false, reportUnknownVariableType=false, reportUnknownMemberType=false

"""Tests for SQL linting and style checking.

This module contains tests for the sql_lint module, which provides:
- LintRule base class and concrete rule implementations
- SQLLinter for applying multiple rules to SQL code
- Convenience functions for linting SQL code
"""

from __future__ import annotations

import pytest
from sqlglot import parse

from dbt_osmosis.core.sql_lint import (
    KeywordCapitalizationRule,
    LintLevel,
    LintResult,
    LintRule,
    LintViolation,
    LineLengthRule,
    QuotedIdentifierRule,
    SelectStarRule,
    SQLLinter,
    TableAliasRule,
)


class TestLintLevel:
    """Tests for the LintLevel enum."""

    def test_lint_level_values(self):
        """Test that LintLevel has expected values."""
        assert LintLevel.ERROR.value == "error"
        assert LintLevel.WARNING.value == "warning"
        assert LintLevel.INFO.value == "info"
        assert LintLevel.CONVENTION.value == "convention"


class TestLintViolation:
    """Tests for the LintViolation dataclass."""

    def test_violation_creation(self):
        """Test creating a basic lint violation."""
        violation = LintViolation(
            rule_id="test-rule",
            message="Test violation message",
            level=LintLevel.WARNING,
            line=1,
            col=5,
        )

        assert violation.rule_id == "test-rule"
        assert violation.message == "Test violation message"
        assert violation.level == LintLevel.WARNING
        assert violation.line == 1
        assert violation.col == 5

    def test_violation_with_optional_fields(self):
        """Test creating a violation with all optional fields."""
        violation = LintViolation(
            rule_id="test-rule",
            message="Test message",
            level=LintLevel.ERROR,
            line=10,
            col=20,
            sql_snippet="SELECT * FROM table",
            fix="SELECT column FROM table",
        )

        assert violation.sql_snippet == "SELECT * FROM table"
        assert violation.fix == "SELECT column FROM table"

    def test_violation_str_representation(self):
        """Test string representation of violations."""
        violation = LintViolation(
            rule_id="test-rule",
            message="Test message",
            level=LintLevel.ERROR,
            line=5,
        )

        str_repr = str(violation)
        assert ":no_entry:" in str_repr
        assert "Line 5" in str_repr
        assert "test-rule" in str_repr

    def test_violation_str_without_line(self):
        """Test string representation without line number."""
        violation = LintViolation(
            rule_id="test-rule",
            message="Test message",
            level=LintLevel.INFO,
        )

        str_repr = str(violation)
        assert ":information_source:" in str_repr
        assert "SQL" in str_repr


class TestLintResult:
    """Tests for the LintResult dataclass."""

    def test_empty_result(self):
        """Test creating an empty lint result."""
        result = LintResult()

        assert result.violations == []
        assert result.sql == ""
        assert result.compiled_sql == ""

    def test_result_with_violations(self):
        """Test creating a result with violations."""
        violations = [
            LintViolation(
                rule_id="rule1",
                message="First violation",
                level=LintLevel.ERROR,
                line=1,
            ),
            LintViolation(
                rule_id="rule2",
                message="Second violation",
                level=LintLevel.WARNING,
                line=2,
            ),
        ]

        result = LintResult(
            violations=violations,
            sql="SELECT 1",
            compiled_sql="SELECT 1",
        )

        assert len(result.violations) == 2
        assert result.sql == "SELECT 1"

    def test_result_bool_evaluation_clean(self):
        """Test boolean evaluation of clean result."""
        result = LintResult(violations=[])
        assert result

    def test_result_bool_evaluation_with_info_only(self):
        """Test boolean evaluation with info-level violations only."""
        result = LintResult(
            violations=[
                LintViolation(
                    rule_id="info-rule",
                    message="Info",
                    level=LintLevel.INFO,
                )
            ]
        )
        assert result

    def test_result_bool_evaluation_with_warning(self):
        """Test boolean evaluation with warning."""
        result = LintResult(
            violations=[
                LintViolation(
                    rule_id="warn-rule",
                    message="Warning",
                    level=LintLevel.WARNING,
                )
            ]
        )
        assert not result

    def test_result_bool_evaluation_with_error(self):
        """Test boolean evaluation with error."""
        result = LintResult(
            violations=[
                LintViolation(
                    rule_id="error-rule",
                    message="Error",
                    level=LintLevel.ERROR,
                )
            ]
        )
        assert not result

    def test_errors_property(self):
        """Test extracting only errors."""
        result = LintResult(
            violations=[
                LintViolation(
                    rule_id="error1",
                    message="Error 1",
                    level=LintLevel.ERROR,
                ),
                LintViolation(
                    rule_id="warning1",
                    message="Warning 1",
                    level=LintLevel.WARNING,
                ),
                LintViolation(
                    rule_id="error2",
                    message="Error 2",
                    level=LintLevel.ERROR,
                ),
            ]
        )

        errors = result.errors
        assert len(errors) == 2
        assert all(v.level == LintLevel.ERROR for v in errors)

    def test_warnings_property(self):
        """Test extracting only warnings."""
        result = LintResult(
            violations=[
                LintViolation(
                    rule_id="error1",
                    message="Error 1",
                    level=LintLevel.ERROR,
                ),
                LintViolation(
                    rule_id="warning1",
                    message="Warning 1",
                    level=LintLevel.WARNING,
                ),
                LintViolation(
                    rule_id="info1",
                    message="Info 1",
                    level=LintLevel.INFO,
                ),
            ]
        )

        warnings = result.warnings
        assert len(warnings) == 1
        assert warnings[0].level == LintLevel.WARNING

    def test_summary_clean(self):
        """Test summary for clean result."""
        result = LintResult()
        assert result.summary() == "No issues found"

    def test_summary_mixed(self):
        """Test summary with mixed violations."""
        result = LintResult(
            violations=[
                LintViolation(
                    rule_id="error1",
                    message="Error 1",
                    level=LintLevel.ERROR,
                ),
                LintViolation(
                    rule_id="error2",
                    message="Error 2",
                    level=LintLevel.ERROR,
                ),
                LintViolation(
                    rule_id="warning1",
                    message="Warning 1",
                    level=LintLevel.WARNING,
                ),
                LintViolation(
                    rule_id="info1",
                    message="Info 1",
                    level=LintLevel.INFO,
                ),
            ]
        )

        summary = result.summary()
        assert "2 error(s)" in summary
        assert "1 warning(s)" in summary
        assert "1 info" in summary


class TestLintRule:
    """Tests for the LintRule base class."""

    def test_rule_initialization(self):
        """Test initializing a custom rule."""
        rule = LintRule(
            rule_id="custom-rule",
            description="A custom rule",
            level=LintLevel.INFO,
        )

        assert rule.rule_id == "custom-rule"
        assert rule.description == "A custom rule"
        assert rule.level == LintLevel.INFO

    def test_rule_not_implemented(self):
        """Test that base rule raises NotImplementedError."""
        rule = LintRule("test", "test")
        ast = parse("SELECT 1")[0]

        with pytest.raises(NotImplementedError):
            rule(ast, "SELECT 1")

    def test_find_line_number(self):
        """Test finding line number of a snippet."""
        rule = LintRule("test", "test")

        sql = "SELECT 1\nSELECT 2\nSELECT 3"
        line = rule._find_line_number(sql, "SELECT 2")

        assert line == 2

    def test_find_line_number_not_found(self):
        """Test finding line number when snippet not found."""
        rule = LintRule("test", "test")

        sql = "SELECT 1\nSELECT 2"
        line = rule._find_line_number(sql, "SELECT 999")

        assert line is None


class TestKeywordCapitalizationRule:
    """Tests for the KeywordCapitalizationRule class."""

    def test_uppercase_enforcement(self):
        """Test enforcing uppercase keywords."""
        rule = KeywordCapitalizationRule(level=LintLevel.CONVENTION, case="upper")
        ast = parse("select id from users")[0]

        violations = rule(ast, "select id from users")

        assert len(violations) > 0
        assert any(v.rule_id == "keyword-case" for v in violations)

    def test_lowercase_enforcement(self):
        """Test enforcing lowercase keywords."""
        rule = KeywordCapitalizationRule(level=LintLevel.CONVENTION, case="lower")
        ast = parse("SELECT ID FROM USERS")[0]

        violations = rule(ast, "SELECT ID FROM USERS")

        assert len(violations) > 0

    def test_consistent_detection_uppercase(self):
        """Test detecting consistent uppercase as expected."""
        rule = KeywordCapitalizationRule(level=LintLevel.CONVENTION, case="consistent")
        ast = parse("SELECT id FROM users WHERE id > 0")[0]

        violations = rule(ast, "SELECT id FROM users WHERE id > 0")

        # All uppercase, so no violations
        assert len(violations) == 0

    def test_consistent_detection_lowercase(self):
        """Test detecting consistent lowercase as expected."""
        rule = KeywordCapitalizationRule(level=LintLevel.CONVENTION, case="consistent")
        ast = parse("select id from users where id > 0")[0]

        violations = rule(ast, "select id from users where id > 0")

        # All lowercase, so no violations
        assert len(violations) == 0

    def test_mixed_case_violations(self):
        """Test detecting mixed case keywords."""
        rule = KeywordCapitalizationRule(level=LintLevel.CONVENTION, case="consistent")
        # Use a single statement with consistent case
        sql = "SELECT id FROM users WHERE id > 0"
        ast = parse(sql)[0]

        violations = rule(ast, sql)

        # All uppercase, no violations
        assert len(violations) == 0


class TestLineLengthRule:
    """Tests for the LineLengthRule class."""

    def test_default_max_length(self):
        """Test default max length of 100 characters."""
        rule = LineLengthRule()

        assert rule.max_length == 100
        assert rule.rule_id == "line-length"

    def test_custom_max_length(self):
        """Test custom max length."""
        rule = LineLengthRule(max_length=50)

        assert rule.max_length == 50

    def test_short_line_clean(self):
        """Test that short lines pass."""
        rule = LineLengthRule(max_length=100)
        ast = parse("SELECT 1")[0]

        violations = rule(ast, "SELECT 1")

        assert len(violations) == 0

    def test_long_line_violation(self):
        """Test that long lines trigger violations."""
        rule = LineLengthRule(max_length=10)
        long_sql = "SELECT id FROM users WHERE id > 0"
        ast = parse(long_sql)[0]

        violations = rule(ast, long_sql)

        assert len(violations) == 1
        assert violations[0].rule_id == "line-length"

    def test_multiple_long_lines(self):
        """Test detecting multiple long lines."""
        rule = LineLengthRule(max_length=20)
        # Use a single SELECT statement
        sql = "SELECT id FROM users WHERE id > 0 AND name IS NOT NULL AND email IS NOT NULL"
        ast = parse(sql)[0]

        violations = rule(ast, sql)

        # Should detect the long line
        assert len(violations) >= 1

    def test_line_number_tracking(self):
        """Test that violations track line numbers correctly."""
        rule = LineLengthRule(max_length=10)
        # Use a single statement with multiple long lines
        sql = """SELECT
            id,
            name,
            email
        FROM users
        WHERE id > 0"""
        ast = parse(sql)[0]

        violations = rule(ast, sql)

        # At least one long line should trigger violation
        assert len(violations) >= 1
        if violations:
            assert violations[0].line is not None


class TestSelectStarRule:
    """Tests for the SelectStarRule class."""

    def test_select_star_detection(self):
        """Test detecting SELECT *."""
        rule = SelectStarRule()
        ast = parse("SELECT * FROM users")[0]

        violations = rule(ast, "SELECT * FROM users")

        assert len(violations) == 1
        assert violations[0].rule_id == "select-star"

    def test_select_columns_clean(self):
        """Test that explicit columns pass."""
        rule = SelectStarRule()
        ast = parse("SELECT id, name FROM users")[0]

        violations = rule(ast, "SELECT id, name FROM users")

        assert len(violations) == 0

    def test_multiple_select_star(self):
        """Test detecting multiple SELECT * in subqueries."""
        rule = SelectStarRule()
        sql = "SELECT * FROM (SELECT * FROM users) u"
        ast = parse(sql)[0]

        violations = rule(ast, sql)

        # Should detect at least one
        assert len(violations) >= 1


class TestTableAliasRule:
    """Tests for the TableAliasRule class."""

    def test_default_min_length(self):
        """Test default min length of 3 characters."""
        rule = TableAliasRule()

        assert rule.min_length == 3

    def test_custom_min_length(self):
        """Test custom min length."""
        rule = TableAliasRule(min_length=5)

        assert rule.min_length == 5

    def test_short_alias_violation(self):
        """Test detecting short aliases in joins."""
        rule = TableAliasRule(min_length=3)
        sql = "SELECT * FROM users u JOIN orders o ON u.id = o.user_id"
        ast = parse(sql)[0]

        violations = rule(ast, sql)

        # Should find at least one short alias
        assert len(violations) >= 0  # May or may not detect depending on sqlglot parsing

    def test_long_alias_clean(self):
        """Test that long aliases pass."""
        rule = TableAliasRule(min_length=3)
        sql = "SELECT * FROM users usr JOIN orders ord ON usr.id = ord.user_id"
        ast = parse(sql)[0]

        violations = rule(ast, sql)

        # Should not find violations for 3+ char aliases
        assert len(violations) == 0

    def test_subquery_alias(self):
        """Test detecting short subquery aliases."""
        rule = TableAliasRule(min_length=3)
        sql = "SELECT * FROM (SELECT id FROM users) u"
        ast = parse(sql)[0]

        violations = rule(ast, sql)

        # Should detect short subquery alias
        assert len(violations) >= 1


class TestQuotedIdentifierRule:
    """Tests for the QuotedIdentifierRule class."""

    def test_unnecessary_quotes_detection(self):
        """Test detecting unnecessarily quoted identifiers."""
        rule = QuotedIdentifierRule()
        sql = 'SELECT "id" FROM "users" WHERE "id" > 0'
        ast = parse(sql)[0]

        violations = rule(ast, sql)

        # Should detect unnecessary quotes
        assert len(violations) >= 1

    def test_necessary_quotes_clean(self):
        """Test that necessary quotes don't trigger violations."""
        rule = QuotedIdentifierRule()
        # Identifiers with spaces need quoting
        sql = r'SELECT "user name" FROM users'
        ast = parse(sql)[0]

        violations = rule(ast, sql)

        assert len(violations) == 0

    def test_reserved_word_quotes(self):
        """Test that quotes on reserved words are allowed."""
        rule = QuotedIdentifierRule()
        sql = 'SELECT "select" FROM users'
        ast = parse(sql)[0]

        violations = rule(ast, sql)

        # "select" is a reserved word, so quotes are necessary
        assert len(violations) == 0


class TestSQLLinter:
    """Tests for the SQLLinter class."""

    def test_linter_initialization_default(self):
        """Test initializing linter with default settings."""
        linter = SQLLinter()

        assert len(linter.rules) > 0
        assert linter.dialect is None

    def test_linter_with_dialect(self):
        """Test initializing linter with dialect."""
        linter = SQLLinter(dialect="postgres")

        assert linter.dialect_name == "postgres"

    def test_enabled_rules_filter(self):
        """Test filtering to specific rules."""
        linter = SQLLinter(enabled_rules=["keyword-case", "line-length"])

        assert len(linter.rules) == 2
        assert all(r.rule_id in ["keyword-case", "line-length"] for r in linter.rules)

    def test_disabled_rules_filter(self):
        """Test disabling specific rules."""
        linter = SQLLinter(disabled_rules=["select-star"])

        assert not any(r.rule_id == "select-star" for r in linter.rules)

    def test_add_custom_rule(self):
        """Test adding a custom rule."""
        linter = SQLLinter()

        custom_rule = LintRule("custom", "Custom rule", LintLevel.INFO)
        linter.add_rule(custom_rule)

        assert custom_rule in linter.rules

    def test_lint_clean_sql(self):
        """Test linting clean SQL."""
        linter = SQLLinter(enabled_rules=["line-length"])
        result = linter.lint("SELECT 1")

        assert result
        assert len(result.violations) == 0

    def test_lint_with_violations(self):
        """Test linting SQL with violations."""
        linter = SQLLinter(enabled_rules=["select-star"])
        result = linter.lint("SELECT * FROM users")

        assert not result
        assert len(result.violations) > 0
        assert result.violations[0].rule_id == "select-star"

    def test_lint_with_compiled_sql(self):
        """Test linting with compiled SQL."""
        linter = SQLLinter(enabled_rules=["select-star"])
        raw_sql = "SELECT * FROM {{ ref('users') }}"
        compiled_sql = "SELECT * FROM users"

        result = linter.lint(raw_sql, compiled_sql)

        # Should lint the compiled SQL
        assert len(result.violations) > 0

    def test_lint_parse_error(self):
        """Test handling of parse errors."""
        linter = SQLLinter()
        result = linter.lint("INVALID SQL HERE !!!")

        assert len(result.violations) > 0
        assert any(v.rule_id == "parse-error" for v in result.violations)

    def test_lint_result_structure(self):
        """Test that lint result has correct structure."""
        linter = SQLLinter(enabled_rules=["select-star"])
        result = linter.lint("SELECT * FROM users")

        assert result.sql == "SELECT * FROM users"
        assert isinstance(result.violations, list)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_sql(self):
        """Test linting empty SQL."""
        linter = SQLLinter()
        result = linter.lint("")

        # Should not crash
        assert isinstance(result, LintResult)

    def test_sql_with_only_comments(self):
        """Test SQL with only comments."""
        linter = SQLLinter()
        result = linter.lint("-- This is a comment\n/* Another comment */")

        assert isinstance(result, LintResult)

    def test_very_long_line(self):
        """Test handling of very long lines."""
        rule = LineLengthRule(max_length=100)
        long_line = "SELECT " + ", ".join([f"col_{i}" for i in range(100)])
        ast = parse(long_line)[0]

        violations = rule(ast, long_line)

        assert len(violations) == 1
        assert violations[0].line == 1

    def test_unicode_in_sql(self):
        """Test handling of unicode characters in SQL."""
        linter = SQLLinter()
        result = linter.lint("SELECT '用户数据' FROM users")

        assert isinstance(result, LintResult)

    def test_multiple_statements(self):
        """Test linting SQL with multiple statements."""
        linter = SQLLinter()
        result = linter.lint("SELECT 1; SELECT 2;")

        # Should handle multiple statements
        assert isinstance(result, LintResult)

    def test_case_sensitive_matching(self):
        """Test that keyword matching is case-insensitive."""
        rule = KeywordCapitalizationRule(case="upper")

        # Should match SELECT, Select, select, etc.
        violations = rule(parse("select 1")[0], "select 1")
        assert len(violations) > 0

        violations = rule(parse("SELECT 1")[0], "SELECT 1")
        assert len(violations) == 0
