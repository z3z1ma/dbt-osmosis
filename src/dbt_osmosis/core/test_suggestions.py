# pyright: reportUnknownVariableType=false, reportPrivateImportUsage=false, reportAny=false, reportUnknownMemberType=false
"""AI-powered test suggestion and generation for dbt models.

This module provides functionality to:
1. Analyze existing test patterns in a dbt project
2. Learn team conventions from existing tests
3. Suggest appropriate tests for models based on patterns
4. Generate test YAML using AI
"""

from __future__ import annotations

import json
import typing as t
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from textwrap import dedent

from dbt_osmosis.core.introspection import PropertyAccessor
from dbt_osmosis.core.llm import get_llm_client
from dbt_osmosis.core.settings import YamlRefactorContext
from dbt_osmosis.core.exceptions import LLMResponseError

__all__ = [
    "TestPatternExtractor",
    "TestSuggestion",
    "AITestSuggester",
    "suggest_tests_for_model",
    "suggest_tests_for_project",
]


@dataclass
class TestSuggestion:
    """A single test suggestion for a column or model."""

    test_type: str  # e.g., "unique", "not_null", "relationships", "accepted_values"
    column_name: str | None = None  # None for model-level tests
    reason: str = ""
    config: dict[str, t.Any] = field(default_factory=dict)
    confidence: float = 1.0  # 0.0 to 1.0

    def to_yaml_dict(self) -> dict[str, t.Any]:
        """Convert to YAML-serializable dict."""
        result: dict[str, t.Any] = {self.test_type: self.config} if self.config else self.test_type
        return result

    def __repr__(self) -> str:
        if self.column_name:
            return f"TestSuggestion({self.column_name}: {self.test_type})"
        return f"TestSuggestion(model: {self.test_type})"


@dataclass
class ModelTestAnalysis:
    """Analysis of test patterns for a single model."""

    model_name: str
    columns: list[str]
    existing_tests: dict[str, list[TestSuggestion]]
    suggested_tests: dict[str, list[TestSuggestion]] = field(default_factory=dict)

    def get_test_summary(self) -> dict[str, t.Any]:
        """Get a summary of tests for this model."""
        return {
            "model_name": self.model_name,
            "total_columns": len(self.columns),
            "columns_with_tests": len(self.existing_tests),
            "total_existing_tests": sum(len(tests) for tests in self.existing_tests.values()),
            "total_suggested_tests": sum(len(tests) for tests in self.suggested_tests.values()),
        }


class TestPatternExtractor:
    """Extracts and learns test patterns from a dbt project.

    This class analyzes existing tests to understand:
    - Which test types are commonly used
    - Column naming patterns that trigger specific tests
    - Test configuration patterns
    """

    def __init__(self, context: YamlRefactorContext) -> None:
        """Initialize the extractor with a dbt project context.

        Args:
            context: The YamlRefactorContext containing project information
        """
        self.context = context
        self.accessor = PropertyAccessor(context=context)

        # Track patterns across the project
        self.column_pattern_tests: defaultdict[str, Counter] = defaultdict(Counter)
        self.data_type_tests: defaultdict[str, Counter] = defaultdict(Counter)
        self.test_frequency: Counter = Counter()
        self.relationship_patterns: list[dict[str, t.Any]] = []

        # Common patterns learned from the project
        self.learned_patterns: dict[str, t.Any] = {}

    def extract_patterns(self) -> None:
        """Extract test patterns from all nodes in the project."""
        from dbt.artifacts.resources.types import NodeType

        manifest = self.context.project.manifest
        for node in manifest.nodes.values():
            # Only process model nodes
            if getattr(node, "resource_type", None) != NodeType.Model:
                continue
            self._analyze_node(node)

        self._learn_patterns()

    def _analyze_node(self, node: t.Any) -> None:
        """Analyze a single node's test patterns."""
        if not hasattr(node, "columns"):
            return

        for column in node.columns:
            col_name = getattr(column, "name", "")
            tests = getattr(column, "tests", [])

            if not tests:
                continue

            # Extract column naming pattern (e.g., "*_id", "*_date", "status")
            base_pattern = self._get_column_pattern(col_name)

            # Get data type if available
            data_type = getattr(column, "data_type", None)

            for test in tests:
                self._analyze_test(test, col_name, base_pattern, data_type)

    def _get_column_pattern(self, column_name: str) -> str:
        """Extract a naming pattern from a column name.

        Examples:
            "order_id" -> "*_id"
            "order_date" -> "*_date"
            "status" -> "status"
            "is_active" -> "is_*"
        """
        # Common suffixes
        for suffix in [
            "_id",
            "_id",
            "_date",
            "_time",
            "_at",
            "_ts",
            "_amount",
            "_count",
            "_flag",
            "_bool",
        ]:
            if column_name.endswith(suffix):
                return f"*{suffix}"

        # Common prefixes
        for prefix in ["is_", "has_", "can_", "should_"]:
            if column_name.startswith(prefix):
                return f"{prefix}*"

        # Return the name itself for specific columns like "status", "type"
        return column_name

    def _analyze_test(
        self, test: t.Any, col_name: str, pattern: str, data_type: str | None
    ) -> None:
        """Analyze a single test definition."""
        if isinstance(test, str):
            # Simple test like "unique" or "not_null"
            self.column_pattern_tests[pattern][test] += 1
            self.test_frequency[test] += 1
            if data_type:
                self.data_type_tests[data_type][test] += 1

        elif isinstance(test, dict):
            # Complex test with config
            for test_name, config in test.items():
                self.column_pattern_tests[pattern][test_name] += 1
                self.test_frequency[test_name] += 1
                if data_type:
                    self.data_type_tests[data_type][test_name] += 1

                # Track relationship patterns
                if test_name == "relationships":
                    self.relationship_patterns.append({
                        "column_pattern": pattern,
                        "config": config,
                    })

    def _learn_patterns(self) -> None:
        """Learn patterns from extracted data."""
        self.learned_patterns = {
            "id_column_tests": self._get_top_tests_for_pattern("*_id"),
            "date_column_tests": self._get_top_tests_for_pattern("*_date"),
            "amount_column_tests": self._get_top_tests_for_pattern("*_amount"),
            "status_column_tests": self._get_top_tests_for_pattern("status"),
            "is_column_tests": self._get_top_tests_for_pattern("is_*"),
            "common_tests": dict(self.test_frequency.most_common(10)),
            "data_type_tests": {
                dt: dict(tests.most_common(3)) for dt, tests in self.data_type_tests.items()
            },
        }

    def _get_top_tests_for_pattern(self, pattern: str, top_n: int = 5) -> list[str]:
        """Get the most common tests for a column pattern."""
        if pattern not in self.column_pattern_tests:
            return []
        return [test for test, _ in self.column_pattern_tests[pattern].most_common(top_n)]

    def get_suggestions_for_column(
        self, column_name: str, data_type: str | None = None
    ) -> list[TestSuggestion]:
        """Get test suggestions for a column based on learned patterns.

        Args:
            column_name: The name of the column
            data_type: Optional data type of the column

        Returns:
            List of TestSuggestion objects
        """
        suggestions: list[TestSuggestion] = []
        pattern = self._get_column_pattern(column_name)

        # Get pattern-based suggestions
        pattern_tests = self.learned_patterns.get("id_column_tests", [])
        if pattern == "*_id":
            pattern_tests = self.learned_patterns.get("id_column_tests", [])
        elif pattern == "*_date":
            pattern_tests = self.learned_patterns.get("date_column_tests", [])
        elif pattern == "*_amount":
            pattern_tests = self.learned_patterns.get("amount_column_tests", [])
        elif pattern == "status":
            pattern_tests = self.learned_patterns.get("status_column_tests", [])
        elif pattern.startswith("is_"):
            pattern_tests = self.learned_patterns.get("is_column_tests", [])

        # Get data type based suggestions
        if data_type:
            dt_tests = self.learned_patterns.get("data_type_tests", {}).get(data_type, [])
            pattern_tests.extend(dt_tests)

        # Create suggestions
        for test_type in pattern_tests:
            # Skip if it's already in suggestions
            if any(s.test_type == test_type for s in suggestions):
                continue

            suggestion = TestSuggestion(
                test_type=test_type,
                column_name=column_name,
                reason=f"Commonly used for columns matching pattern '{pattern}'",
                confidence=0.7,
            )
            suggestions.append(suggestion)

        return suggestions


class AITestSuggester:
    """AI-powered test suggestion using LLM analysis.

    This class uses an LLM to:
    1. Analyze model SQL and structure
    2. Consider existing project test patterns
    3. Generate contextually appropriate test suggestions
    """

    def __init__(
        self,
        context: YamlRefactorContext,
        pattern_extractor: TestPatternExtractor | None = None,
    ) -> None:
        """Initialize the AI test suggester.

        Args:
            context: The YamlRefactorContext containing project information
            pattern_extractor: Optional TestPatternExtractor with learned patterns
        """
        self.context = context
        self.pattern_extractor = pattern_extractor
        self.accessor = PropertyAccessor(context=context)

    def suggest_tests_for_node(
        self,
        node: t.Any,
        use_ai: bool = True,
        temperature: float = 0.3,
    ) -> ModelTestAnalysis:
        """Generate test suggestions for a single node.

        Args:
            node: The dbt model node
            use_ai: Whether to use AI for suggestions (falls back to pattern-based)
            temperature: LLM temperature for generation

        Returns:
            ModelTestAnalysis with existing and suggested tests
        """
        model_name = getattr(node, "name", "unknown")
        columns = [getattr(c, "name", "") for c in getattr(node, "columns", [])]

        # Extract existing tests
        existing_tests: dict[str, list[TestSuggestion]] = {}
        for column in getattr(node, "columns", []):
            col_name = getattr(column, "name", "")
            tests = getattr(column, "tests", [])
            if tests:
                existing_tests[col_name] = [
                    TestSuggestion(
                        test_type=t if isinstance(t, str) else list(t.keys())[0],
                        column_name=col_name,
                        confidence=1.0,
                    )
                    for t in tests
                ]

        analysis = ModelTestAnalysis(
            model_name=model_name,
            columns=columns,
            existing_tests=existing_tests,
        )

        if use_ai:
            # Use AI for smart suggestions
            suggestions = self._ai_suggest_tests(node, temperature)
        else:
            # Use pattern-based suggestions
            suggestions = self._pattern_suggest_tests(node)

        analysis.suggested_tests = suggestions
        return analysis

    def _pattern_suggest_tests(self, node: t.Any) -> dict[str, list[TestSuggestion]]:
        """Generate test suggestions based on learned patterns."""
        suggestions: dict[str, list[TestSuggestion]] = defaultdict(list)

        if not self.pattern_extractor:
            return suggestions

        for column in getattr(node, "columns", []):
            col_name = getattr(column, "name", "")
            data_type = getattr(column, "data_type", None)

            col_suggestions = self.pattern_extractor.get_suggestions_for_column(col_name, data_type)

            # Filter out already existing tests
            existing = {t.test_type for t in getattr(column, "tests", [])}
            for suggestion in col_suggestions:
                if suggestion.test_type not in existing:
                    suggestions[col_name].append(suggestion)

        return dict(suggestions)

    def _ai_suggest_tests(
        self, node: t.Any, temperature: float = 0.3
    ) -> dict[str, list[TestSuggestion]]:
        """Generate test suggestions using AI."""
        client, model_engine = get_llm_client()

        # Build the prompt
        prompt = self._create_test_suggestion_prompt(node)

        try:
            if hasattr(client, "chat"):
                response = client.chat.completions.create(
                    model=model_engine,
                    messages=prompt,
                    temperature=temperature,
                )
            else:
                response = client.ChatCompletion.create(
                    engine=model_engine,
                    messages=prompt,
                    temperature=temperature,
                )

            content = response.choices[0].message.content
            if not content:
                raise LLMResponseError("LLM returned empty response")

            return self._parse_ai_response(content)

        except Exception:
            # Fallback to pattern-based suggestions
            return self._pattern_suggest_tests(node)

    def _create_test_suggestion_prompt(self, node: t.Any) -> list[dict[str, str]]:
        """Create the LLM prompt for test suggestions."""
        model_name = getattr(node, "name", "unknown")
        raw_sql = getattr(node, "raw_sql", getattr(node, "compiled_sql", ""))
        description = self.accessor.get_description(node) or "No description"

        # Gather column info
        column_info = []
        for col in getattr(node, "columns", []):
            col_name = getattr(col, "name", "")
            col_desc = self.accessor.get_description(node, column_name=col_name) or ""
            col_type = getattr(col, "data_type", "unknown")
            existing_tests = getattr(col, "tests", [])

            column_info.append({
                "name": col_name,
                "type": col_type,
                "description": col_desc,
                "existing_tests": existing_tests,
            })

        # Build context from learned patterns
        patterns_context = ""
        if self.pattern_extractor:
            patterns = self.pattern_extractor.learned_patterns
            patterns_context = f"""
Project Test Patterns:
- ID columns typically use: {", ".join(patterns.get("id_column_tests", ["unique, not_null"]))}
- Status columns typically use: {", ".join(patterns.get("status_column_tests", ["accepted_values"]))}
- Amount columns typically use: {", ".join(patterns.get("amount_column_tests", ["not_null"]))}
"""

        system_prompt = dedent("""
            You are a dbt testing expert. Your job is to suggest appropriate tests for dbt models.

            Common dbt test types:
            - unique: ensures all values in a column are unique
            - not_null: ensures a column has no null values
            - relationships: ensures referential integrity (e.g., foreign key to another table)
            - accepted_values: ensures values are from a specific set
            - dbt_utils expressions: various utility tests from dbt_utils package

            Guidelines for test suggestions:
            1. *_id columns: suggest 'unique' and 'not_null', optionally 'relationships'
            2. Foreign key columns (*_id referencing other tables): suggest 'relationships'
            3. Status/type columns with known values: suggest 'accepted_values'
            4. Numeric/amount columns: suggest 'not_null'
            5. Date columns: suggest 'not_null' if required
            6. Don't suggest tests that already exist

            Return ONLY a valid JSON object with this structure:
            {
              "column_name": [
                {"test_type": "unique", "reason": "explanation", "config": {}},
                {"test_type": "not_null", "reason": "explanation", "config": {}}
              ]
            }

            For relationship tests, include config:
            {"test_type": "relationships", "reason": "...", "config": {"to": "ref('other_model')", "field": "id"}}

            For accepted_values tests, include config:
            {"test_type": "accepted_values", "reason": "...", "config": {"values": ["a", "b", "c"]}}
        """)

        user_prompt = dedent(f"""
            Model: {model_name}
            Description: {description}

            SQL:
            {raw_sql[:2000]}...

            Columns:
            {json.dumps(column_info, indent=2)}

            {patterns_context}

            Suggest appropriate tests for each column. Return only valid JSON.
        """)

        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

    def _parse_ai_response(self, content: str) -> dict[str, list[TestSuggestion]]:
        """Parse the AI response into TestSuggestion objects."""
        # Remove markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            content = content[content.find("{") : content.rfind("}") + 1]

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback to empty suggestions
            return {}

        suggestions: dict[str, list[TestSuggestion]] = defaultdict(list)

        for col_name, test_list in data.items():
            for test_data in test_list:
                if isinstance(test_data, str):
                    # Simple test name
                    suggestion = TestSuggestion(
                        test_type=test_data,
                        column_name=col_name,
                        reason="AI suggested",
                    )
                elif isinstance(test_data, dict):
                    # Full test data
                    suggestion = TestSuggestion(
                        test_type=test_data.get("test_type", ""),
                        column_name=col_name,
                        reason=test_data.get("reason", "AI suggested"),
                        config=test_data.get("config", {}),
                        confidence=0.8,
                    )
                else:
                    continue

                suggestions[col_name].append(suggestion)

        return dict(suggestions)


def suggest_tests_for_model(
    context: YamlRefactorContext,
    node: t.Any,
    use_ai: bool = True,
    temperature: float = 0.3,
) -> ModelTestAnalysis:
    """Suggest tests for a single model.

    This is a convenience function that creates the necessary components
    and returns test suggestions for a model.

    Args:
        context: The YamlRefactorContext containing project information
        node: The dbt model node to analyze
        use_ai: Whether to use AI for suggestions
        temperature: LLM temperature for generation

    Returns:
        ModelTestAnalysis with existing and suggested tests
    """
    # Extract patterns if using pattern-based or hybrid approach
    extractor: TestPatternExtractor | None = None
    if not use_ai or use_ai:
        extractor = TestPatternExtractor(context)
        extractor.extract_patterns()

    suggester = AITestSuggester(context, extractor)
    return suggester.suggest_tests_for_node(node, use_ai=use_ai, temperature=temperature)


def suggest_tests_for_project(
    context: YamlRefactorContext,
    use_ai: bool = True,
    temperature: float = 0.3,
) -> dict[str, ModelTestAnalysis]:
    """Suggest tests for all models in a project.

    This is a convenience function that analyzes all models in a project
    and returns test suggestions for each.

    Args:
        context: The YamlRefactorContext containing project information
        use_ai: Whether to use AI for suggestions
        temperature: LLM temperature for generation

    Returns:
        Dictionary mapping model names to ModelTestAnalysis objects
    """
    # Extract patterns once for the entire project
    extractor = TestPatternExtractor(context)
    extractor.extract_patterns()

    suggester = AITestSuggester(context, extractor)

    from dbt.artifacts.resources.types import NodeType

    results: dict[str, ModelTestAnalysis] = {}
    manifest = context.project.manifest
    for node in manifest.nodes.values():
        # Only process model nodes
        if getattr(node, "resource_type", None) != NodeType.Model:
            continue
        model_name = getattr(node, "name", "unknown")
        analysis = suggester.suggest_tests_for_node(node, use_ai=use_ai, temperature=temperature)
        results[model_name] = analysis

    return results
