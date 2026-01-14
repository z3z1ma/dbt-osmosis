"""SQL linting and style checking for dbt-osmosis.

This module provides SQL linting functionality using sqlglot for parsing
and analysis. It supports configurable rules for SQL style, anti-patterns,
and potential bugs.
"""

from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass, field
from enum import Enum

from sqlglot import exp, parse
from sqlglot.dialects import Dialect

from dbt_osmosis.core import logger

if t.TYPE_CHECKING:
    from dbt_osmosis.core.config import DbtProjectContext

__all__ = [
    "LintLevel",
    "LintViolation",
    "LintResult",
    "SQLLinter",
    "lint_sql_code",
    "LintRule",
    "KeywordCapitalizationRule",
    "LineLengthRule",
    "SelectStarRule",
    "TableAliasRule",
    "QuotedIdentifierRule",
]


class LintLevel(Enum):
    """Severity levels for lint violations."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    CONVENTION = "convention"


@dataclass
class LintViolation:
    """A single lint violation found in SQL code."""

    rule_id: str
    """Unique identifier for the rule (e.g., 'keyword-case')."""

    message: str
    """Human-readable description of the violation."""

    level: LintLevel
    """Severity level of the violation."""

    line: int | None = None
    """Line number where the violation occurs (1-indexed)."""

    col: int | None = None
    """Column number where the violation occurs (1-indexed)."""

    sql_snippet: str | None = None
    """Snippet of SQL code that caused the violation."""

    fix: str | None = None
    """Suggested fix for the violation."""

    def __str__(self) -> str:
        location = f"Line {self.line}" if self.line else "SQL"
        level_emoji = {
            LintLevel.ERROR: ":no_entry:",
            LintLevel.WARNING: ":warning:",
            LintLevel.INFO: ":information_source:",
            LintLevel.CONVENTION: ":sparkles:",
        }
        emoji = level_emoji.get(self.level, ":white_check_mark:")
        return f"{emoji} [{location}] {self.rule_id}: {self.message}"


@dataclass
class LintResult:
    """Result of linting SQL code."""

    violations: list[LintViolation] = field(default_factory=list)
    """List of all violations found."""

    sql: str = ""
    """Original SQL code that was linted."""

    compiled_sql: str = ""
    """Compiled SQL code (after Jinja processing), if available."""

    def __bool__(self) -> bool:
        """Return True if linting passed (no errors or warnings)."""
        return not any(v.level in (LintLevel.ERROR, LintLevel.WARNING) for v in self.violations)

    @property
    def errors(self) -> list[LintViolation]:
        """Return only error-level violations."""
        return [v for v in self.violations if v.level == LintLevel.ERROR]

    @property
    def warnings(self) -> list[LintViolation]:
        """Return only warning-level violations."""
        return [v for v in self.violations if v.level == LintLevel.WARNING]

    def summary(self) -> str:
        """Return a summary string of the lint result."""
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        info_count = len(self.violations) - error_count - warning_count

        parts = []
        if error_count:
            parts.append(f"{error_count} error(s)")
        if warning_count:
            parts.append(f"{warning_count} warning(s)")
        if info_count:
            parts.append(f"{info_count} info")

        return ", ".join(parts) if parts else "No issues found"


class LintRule:
    """Base class for SQL linting rules.

    Rules are callable objects that take a sqlglot Expression and return
    a list of LintViolation objects.
    """

    def __init__(
        self,
        rule_id: str,
        description: str,
        level: LintLevel = LintLevel.WARNING,
    ) -> None:
        self.rule_id = rule_id
        self.description = description
        self.level = level

    def __call__(self, ast: exp.Expression, sql: str) -> list[LintViolation]:
        """Run the rule on a SQL AST and return violations.

        Args:
            ast: The sqlglot AST to analyze
            sql: The original SQL string for context

        Returns:
            List of violations found
        """
        raise NotImplementedError

    def _find_line_number(self, sql: str, snippet: str) -> int | None:
        """Find the line number of a snippet in SQL.

        Args:
            sql: The full SQL string
            snippet: A snippet to search for

        Returns:
            Line number (1-indexed) or None if not found
        """
        index = sql.find(snippet)
        if index == -1:
            return None
        return sql[:index].count("\n") + 1


class KeywordCapitalizationRule(LintRule):
    """Check SQL keyword capitalization consistency.

    Enforces that SQL keywords are either all uppercase or all lowercase.
    """

    def __init__(
        self,
        level: LintLevel = LintLevel.CONVENTION,
        case: t.Literal["upper", "lower", "consistent"] = "consistent",
    ) -> None:
        super().__init__(
            rule_id="keyword-case",
            description="SQL keywords should use consistent capitalization",
            level=level,
        )
        self.case = case

        # Common SQL keywords (not exhaustive)
        self.keywords = {
            "select",
            "from",
            "where",
            "join",
            "inner",
            "outer",
            "left",
            "right",
            "full",
            "on",
            "and",
            "or",
            "not",
            "in",
            "as",
            "by",
            "group",
            "order",
            "having",
            "insert",
            "update",
            "delete",
            "create",
            "drop",
            "alter",
            "table",
            "index",
            "into",
            "values",
            "set",
            "limit",
            "offset",
            "union",
            "intersect",
            "except",
            "case",
            "when",
            "then",
            "else",
            "end",
            "distinct",
            "all",
            "exists",
            "between",
            "like",
            "is",
            "null",
            "true",
            "false",
            "with",
            "recursive",
            "over",
            "partition",
        }

    def __call__(self, ast: exp.Expression, sql: str) -> list[LintViolation]:
        violations: list[LintViolation] = []

        # Find all keywords in the SQL
        keyword_pattern = r"\b(" + "|".join(self.keywords) + r")\b"
        matches = re.finditer(keyword_pattern, sql, re.IGNORECASE)

        upper_count = 0
        lower_count = 0

        for match in matches:
            keyword = match.group()
            if keyword.isupper():
                upper_count += 1
            elif keyword.islower():
                lower_count += 1

        # Determine expected case
        if self.case == "upper":
            expected_case = "upper"
        elif self.case == "lower":
            expected_case = "lower"
        else:  # consistent
            # Use the most common case as expected
            expected_case = "upper" if upper_count >= lower_count else "lower"

        # Find violations
        for match in re.finditer(keyword_pattern, sql, re.IGNORECASE):
            keyword = match.group()
            is_upper = keyword.isupper()
            is_lower = keyword.islower()

            violation = False
            fix = None

            if expected_case == "upper" and not is_upper:
                violation = True
                fix = keyword.upper()
            elif expected_case == "lower" and not is_lower:
                violation = True
                fix = keyword.lower()

            if violation:
                line = self._find_line_number(sql, keyword)
                violations.append(
                    LintViolation(
                        rule_id=self.rule_id,
                        message=f"Keyword '{keyword}' should be {expected_case}case",
                        level=self.level,
                        line=line,
                        col=match.start() - sql.rfind("\n", 0, match.start()),
                        sql_snippet=keyword,
                        fix=fix,
                    )
                )

        return violations


class LineLengthRule(LintRule):
    """Check that SQL lines don't exceed a maximum length.

    Long lines can be hard to read and may cause issues with some tools.
    """

    def __init__(self, max_length: int = 100, level: LintLevel = LintLevel.INFO) -> None:
        super().__init__(
            rule_id="line-length",
            description=f"SQL lines should not exceed {max_length} characters",
            level=level,
        )
        self.max_length = max_length

    def __call__(self, ast: exp.Expression, sql: str) -> list[LintViolation]:
        violations: list[LintViolation] = []

        for line_num, line in enumerate(sql.split("\n"), start=1):
            if len(line) > self.max_length:
                violations.append(
                    LintViolation(
                        rule_id=self.rule_id,
                        message=f"Line is {len(line)} characters (max: {self.max_length})",
                        level=self.level,
                        line=line_num,
                        col=self.max_length + 1,
                        sql_snippet=line[: self.max_length + 20] + "...",
                    )
                )

        return violations


class SelectStarRule(LintRule):
    """Check for SELECT * usage.

    SELECT * can be problematic because:
    1. It may return unexpected columns if the schema changes
    2. It can cause unnecessary data transfer
    3. It makes code less explicit about dependencies
    """

    def __init__(self, level: LintLevel = LintLevel.WARNING) -> None:
        super().__init__(
            rule_id="select-star",
            description="Avoid SELECT *; explicitly list columns instead",
            level=level,
        )

    def __call__(self, ast: exp.Expression, sql: str) -> list[LintViolation]:
        violations: list[LintViolation] = []

        # Find all SELECT expressions
        for select in ast.find_all(exp.Select):
            # Check if any selection is a star (*)
            for selection in select.expressions:
                if isinstance(selection, exp.Star):
                    line = self._find_line_number(sql, "*")
                    violations.append(
                        LintViolation(
                            rule_id=self.rule_id,
                            message="SELECT * detected; explicitly list columns for clarity",
                            level=self.level,
                            line=line,
                            sql_snippet="SELECT *",
                        )
                    )
                    break

        return violations


class TableAliasRule(LintRule):
    """Check for table alias quality.

    Discourages cryptic single-letter aliases in favor of meaningful names.
    """

    def __init__(
        self,
        min_length: int = 3,
        level: LintLevel = LintLevel.CONVENTION,
    ) -> None:
        super().__init__(
            rule_id="table-alias",
            description=f"Table aliases should be at least {min_length} characters",
            level=level,
        )
        self.min_length = min_length

    def __call__(self, ast: exp.Expression, sql: str) -> list[LintViolation]:
        violations: list[LintViolation] = []

        # Find all table aliases in JOIN and FROM clauses
        for join in ast.find_all(exp.Join):
            if join.alias and len(join.alias) < self.min_length:
                line = self._find_line_number(sql, str(join.alias))
                violations.append(
                    LintViolation(
                        rule_id=self.rule_id,
                        message=f"Table alias '{join.alias}' is too short (min: {self.min_length})",
                        level=self.level,
                        line=line,
                        sql_snippet=str(join.alias),
                    )
                )

        # Check for subquery aliases
        for subq in ast.find_all(exp.Subquery):
            if subq.alias and len(subq.alias) < self.min_length:
                line = self._find_line_number(sql, str(subq.alias))
                violations.append(
                    LintViolation(
                        rule_id=self.rule_id,
                        message=f"Subquery alias '{subq.alias}' is too short (min: {self.min_length})",
                        level=self.level,
                        line=line,
                        sql_snippet=str(subq.alias),
                    )
                )

        return violations


class QuotedIdentifierRule(LintRule):
    """Check for unnecessary quoted identifiers.

    In SQL, identifiers only need quoting if they contain special characters
    or are case-sensitive. Unnecessary quoting can make code harder to read.
    """

    def __init__(self, level: LintLevel = LintLevel.INFO) -> None:
        super().__init__(
            rule_id="quoted-identifier",
            description="Avoid unnecessary quoted identifiers",
            level=level,
        )

    def __call__(self, ast: exp.Expression, sql: str) -> list[LintViolation]:
        violations: list[LintViolation] = []

        # Find all quoted identifiers
        for identifier in ast.find_all(exp.Identifier):
            if identifier.quoted and identifier.name:
                # Check if the identifier needs quoting
                # (contains special chars, spaces, or is a reserved word)
                needs_quoting = bool(
                    re.search(r"[^a-zA-Z0-9_]", identifier.name)
                    or identifier.name.lower() in {"select", "from", "where", "group", "order"}
                )

                if not needs_quoting:
                    line = self._find_line_number(sql, f'"{identifier.name}"')
                    violations.append(
                        LintViolation(
                            rule_id=self.rule_id,
                            message=f"Unnecessarily quoted identifier '{identifier.name}'",
                            level=self.level,
                            line=line,
                            sql_snippet=f'"{identifier.name}"',
                            fix=identifier.name,
                        )
                    )

        return violations


class SQLLinter:
    """SQL linter that applies multiple rules to SQL code.

    The linter uses sqlglot to parse SQL and analyze the AST for various
    issues related to style, anti-patterns, and potential bugs.
    """

    def __init__(
        self,
        dialect: str | Dialect | None = None,
        enabled_rules: list[str] | None = None,
        disabled_rules: list[str] | None = None,
    ) -> None:
        """Initialize the SQL linter.

        Args:
            dialect: SQL dialect to use (e.g., 'postgres', 'duckdb', 'snowflake')
            enabled_rules: List of rule IDs to enable (None = all)
            disabled_rules: List of rule IDs to disable
        """
        self.dialect = dialect if isinstance(dialect, Dialect) else None
        self.dialect_name = str(dialect) if dialect else None

        # Default rules
        self._all_rules: list[LintRule] = [
            KeywordCapitalizationRule(),
            LineLengthRule(),
            SelectStarRule(),
            TableAliasRule(),
            QuotedIdentifierRule(),
        ]

        # Filter rules based on enabled/disabled lists
        self.rules = self._filter_rules(enabled_rules, disabled_rules)

    def _filter_rules(
        self,
        enabled_rules: list[str] | None,
        disabled_rules: list[str] | None,
    ) -> list[LintRule]:
        """Filter the list of rules based on enabled/disabled lists."""
        if enabled_rules is not None:
            return [r for r in self._all_rules if r.rule_id in enabled_rules]
        if disabled_rules is not None:
            return [r for r in self._all_rules if r.rule_id not in disabled_rules]
        return self._all_rules

    def add_rule(self, rule: LintRule) -> None:
        """Add a custom rule to the linter."""
        self._all_rules.append(rule)
        self.rules.append(rule)

    def lint(
        self,
        sql: str,
        compiled_sql: str | None = None,
    ) -> LintResult:
        """Lint SQL code and return violations.

        Args:
            sql: The SQL code to lint
            compiled_sql: Optional compiled SQL (after Jinja processing)

        Returns:
            LintResult containing all violations
        """
        result = LintResult(sql=sql, compiled_sql=compiled_sql or "")

        # Use compiled SQL if available, otherwise use raw SQL
        sql_to_lint = compiled_sql or sql

        try:
            # Parse SQL to AST
            parsed = parse(sql_to_lint, dialect=self.dialect_name, read=self.dialect)
            if not parsed:
                return result

            # Get the first statement if multiple
            ast = parsed[0] if isinstance(parsed, list) else parsed
            if ast is None:
                return result

            # Run all rules
            for rule in self.rules:
                try:
                    violations = rule(t.cast(exp.Expression, ast), sql_to_lint)
                    result.violations.extend(violations)
                except Exception as e:
                    logger.warning(f"Rule {rule.rule_id} failed: {e}")

        except Exception as e:
            # If parsing fails, add a parsing error violation
            result.violations.append(
                LintViolation(
                    rule_id="parse-error",
                    message=f"Failed to parse SQL: {e}",
                    level=LintLevel.ERROR,
                )
            )

        return result

    def lint_model(
        self,
        context: DbtProjectContext,
        model_name: str,
    ) -> LintResult:
        """Lint a dbt model's SQL code.

        Args:
            context: dbt project context
            model_name: Name of the model to lint

        Returns:
            LintResult containing all violations
        """
        # Find the model in the manifest
        model = None
        for node in context.manifest.nodes.values():
            if node.name == model_name:
                model = node
                break

        if model is None:
            return LintResult(
                violations=[
                    LintViolation(
                        rule_id="model-not-found",
                        message=f"Model '{model_name}' not found",
                        level=LintLevel.ERROR,
                    )
                ]
            )

        # Lint both raw and compiled SQL
        raw_sql = model.raw_code or ""
        compiled_sql = getattr(model, "compiled_code", None) or ""

        return self.lint(raw_sql, compiled_sql)

    def lint_project(
        self,
        context: DbtProjectContext,
        fqn_filter: list[str] | None = None,
    ) -> dict[str, LintResult]:
        """Lint all models in a dbt project.

        Args:
            context: dbt project context
            fqn_filter: Optional list of FQN patterns to filter models

        Returns:
            Dict mapping model names to LintResults
        """
        results: dict[str, LintResult] = {}

        for node in context.manifest.nodes.values():
            # Skip non-model nodes
            if not hasattr(node, "raw_code") or not node.raw_code:
                continue

            # Apply FQN filter if provided
            if fqn_filter:
                node_fqn = ".".join(node.fqn) if hasattr(node, "fqn") else ""
                if not any(pattern in node_fqn for pattern in fqn_filter):
                    continue

            # Lint the model
            compiled_sql = getattr(node, "compiled_code", None) or ""
            result = self.lint(node.raw_code or "", compiled_sql)
            results[node.name] = result

        return results


def lint_sql_code(
    context: DbtProjectContext,
    raw_sql: str,
    dialect: str | None = None,
    rules: list[str] | None = None,
) -> LintResult:
    """Convenience function to lint SQL code.

    Args:
        context: dbt project context
        raw_sql: Raw SQL code to lint
        dialect: Optional SQL dialect
        rules: Optional list of rule IDs to enable

    Returns:
        LintResult containing all violations
    """
    # Get dialect from adapter if not specified
    if dialect is None:
        dialect = context.adapter.type()

    # Create linter with dialect
    linter = SQLLinter(dialect=dialect, enabled_rules=rules)

    # Compile SQL first to handle Jinja
    from dbt_osmosis.core.sql_operations import compile_sql_code

    try:
        compiled_node = compile_sql_code(context, raw_sql)
        compiled_sql = compiled_node.compiled_code or raw_sql
    except Exception as e:
        logger.debug(":warning: SQL compilation failed: %s", e)
        compiled_sql = None

    return linter.lint(raw_sql, compiled_sql)
