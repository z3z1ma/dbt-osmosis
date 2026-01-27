# pyright: reportPrivateImportUsage=false, reportUnknownVariableType=false, reportUnknownMemberType=false

"""Tests for AI-powered test suggestion and generation.

This module contains tests for the test_suggestions module, which provides:
- TestPatternExtractor: Extracts and learns test patterns from dbt projects
- AITestSuggester: AI-powered test suggestion using LLM analysis
- Convenience functions for model and project-wide test suggestions
"""

from __future__ import annotations

import json
from collections import Counter
from unittest import mock

import pytest

from dbt_osmosis.core.test_suggestions import (
    AITestSuggester,
    ModelTestAnalysis,
    TestPatternExtractor,
    TestSuggestion,
    suggest_tests_for_model,
    suggest_tests_for_project,
)


@pytest.fixture
def mock_context(yaml_context):
    """Create a mock context for testing."""
    return yaml_context


@pytest.fixture
def sample_node(yaml_context):
    """Get a sample model node for testing."""
    manifest = yaml_context.project.manifest
    # Get a model node with columns
    for node in manifest.nodes.values():
        if hasattr(node, "columns") and node.columns:
            return node
    pytest.skip("No suitable model node found")


class TestTestSuggestion:
    """Tests for the TestSuggestion dataclass."""

    def test_test_suggestion_creation(self):
        """Test creating a basic test suggestion."""
        suggestion = TestSuggestion(
            test_type="unique",
            column_name="user_id",
            reason="ID column should be unique",
            confidence=0.9,
        )

        assert suggestion.test_type == "unique"
        assert suggestion.column_name == "user_id"
        assert suggestion.reason == "ID column should be unique"
        assert suggestion.confidence == 0.9

    def test_test_suggestion_with_config(self):
        """Test creating a test suggestion with configuration."""
        config = {"to": "ref('users')", "field": "id"}
        suggestion = TestSuggestion(
            test_type="relationships",
            column_name="customer_id",
            config=config,
        )

        assert suggestion.test_type == "relationships"
        assert suggestion.config == config

    def test_test_suggestion_to_yaml_dict(self):
        """Test converting test suggestion to YAML-serializable dict."""
        suggestion = TestSuggestion(
            test_type="unique",
            column_name="user_id",
        )

        yaml_dict = suggestion.to_yaml_dict()
        assert yaml_dict == {"unique": {}}

    def test_test_suggestion_to_yaml_dict_with_config(self):
        """Test converting test suggestion with config to YAML dict."""
        config = {"values": ["active", "inactive"]}
        suggestion = TestSuggestion(
            test_type="accepted_values",
            column_name="status",
            config=config,
        )

        yaml_dict = suggestion.to_yaml_dict()
        assert yaml_dict == {"accepted_values": config}

    def test_test_suggestion_repr_with_column(self):
        """Test string representation for column-level test."""
        suggestion = TestSuggestion(test_type="unique", column_name="user_id")
        assert repr(suggestion) == "TestSuggestion(user_id: unique)"

    def test_test_suggestion_repr_for_model(self):
        """Test string representation for model-level test."""
        suggestion = TestSuggestion(test_type="data_quality_check")
        assert repr(suggestion) == "TestSuggestion(model: data_quality_check)"


class TestModelTestAnalysis:
    """Tests for the ModelTestAnalysis dataclass."""

    def test_model_test_analysis_creation(self):
        """Test creating a model test analysis."""
        analysis = ModelTestAnalysis(
            model_name="my_model",
            columns=["id", "name", "email"],
            existing_tests={"id": [TestSuggestion(test_type="unique", column_name="id")]},
        )

        assert analysis.model_name == "my_model"
        assert analysis.columns == ["id", "name", "email"]
        assert len(analysis.existing_tests) == 1

    def test_get_test_summary(self):
        """Test getting test summary from analysis."""
        analysis = ModelTestAnalysis(
            model_name="orders",
            columns=["order_id", "customer_id", "amount"],
            existing_tests={
                "order_id": [
                    TestSuggestion(test_type="unique", column_name="order_id"),
                    TestSuggestion(test_type="not_null", column_name="order_id"),
                ],
                "customer_id": [TestSuggestion(test_type="not_null", column_name="customer_id")],
            },
            suggested_tests={
                "amount": [TestSuggestion(test_type="not_null", column_name="amount")]
            },
        )

        summary = analysis.get_test_summary()
        assert summary["model_name"] == "orders"
        assert summary["total_columns"] == 3
        assert summary["columns_with_tests"] == 2
        assert summary["total_existing_tests"] == 3
        assert summary["total_suggested_tests"] == 1


class TestTestPatternExtractor:
    """Tests for the TestPatternExtractor class."""

    def test_extractor_initialization(self, mock_context):
        """Test initializing the pattern extractor."""
        extractor = TestPatternExtractor(mock_context)

        assert extractor.context == mock_context
        assert extractor.accessor is not None
        assert extractor.column_pattern_tests == {}
        assert extractor.data_type_tests == {}
        assert extractor.test_frequency == {}
        assert extractor.learned_patterns == {}

    def test_extract_patterns_from_project(self, mock_context):
        """Test extracting patterns from a dbt project."""
        extractor = TestPatternExtractor(mock_context)
        extractor.extract_patterns()

        # Check that patterns were extracted
        assert isinstance(extractor.learned_patterns, dict)
        assert "common_tests" in extractor.learned_patterns
        assert isinstance(extractor.test_frequency, dict)

    def test_get_column_pattern_id(self):
        """Test column pattern detection for ID columns."""
        extractor = TestPatternExtractor(mock_context)

        assert extractor._get_column_pattern("user_id") == "*_id"
        assert extractor._get_column_pattern("order_id") == "*_id"
        assert extractor._get_column_pattern("customer_id") == "*_id"

    def test_get_column_pattern_date(self):
        """Test column pattern detection for date columns."""
        extractor = TestPatternExtractor(mock_context)

        assert extractor._get_column_pattern("created_date") == "*_date"
        assert extractor._get_column_pattern("updated_date") == "*_date"
        assert extractor._get_column_pattern("order_date") == "*_date"

    def test_get_column_pattern_timestamp(self):
        """Test column pattern detection for timestamp columns."""
        extractor = TestPatternExtractor(mock_context)

        assert extractor._get_column_pattern("created_at") == "*_at"
        assert extractor._get_column_pattern("updated_at") == "*_at"

    def test_get_column_pattern_boolean(self):
        """Test column pattern detection for boolean columns."""
        extractor = TestPatternExtractor(mock_context)

        assert extractor._get_column_pattern("is_active") == "is_*"
        assert extractor._get_column_pattern("has_permission") == "has_*"
        assert extractor._get_column_pattern("can_edit") == "can_*"

    def test_get_column_pattern_specific(self):
        """Test column pattern detection for specific columns."""
        extractor = TestPatternExtractor(mock_context)

        # Specific column names should return themselves
        assert extractor._get_column_pattern("status") == "status"
        assert extractor._get_column_pattern("type") == "type"

    def test_analyze_test_simple(self):
        """Test analyzing a simple test string."""
        extractor = TestPatternExtractor(mock_context)

        # Simple string test
        extractor._analyze_test("unique", "user_id", "*_id", None)

        assert extractor.test_frequency["unique"] == 1
        assert extractor.column_pattern_tests["*_id"]["unique"] == 1

    def test_analyze_test_with_config(self):
        """Test analyzing a test with configuration."""
        extractor = TestPatternExtractor(mock_context)

        # Dict test with config
        test_dict = {"relationships": {"to": "ref('users')", "field": "id"}}
        extractor._analyze_test(test_dict, "user_id", "*_id", "integer")

        assert extractor.test_frequency["relationships"] == 1
        assert extractor.column_pattern_tests["*_id"]["relationships"] == 1
        assert len(extractor.relationship_patterns) == 1

    def test_learn_patterns(self):
        """Test learning patterns from extracted data."""
        extractor = TestPatternExtractor(mock_context)

        # Add some test data - use Counter objects
        extractor.column_pattern_tests["*_id"] = Counter({"unique": 5, "not_null": 3})
        extractor.column_pattern_tests["status"] = Counter({"accepted_values": 2})
        extractor.test_frequency = Counter({"unique": 5, "not_null": 3, "accepted_values": 2})

        extractor._learn_patterns()

        assert "id_column_tests" in extractor.learned_patterns
        assert "status_column_tests" in extractor.learned_patterns
        assert "common_tests" in extractor.learned_patterns

    def test_get_suggestions_for_column_id(self):
        """Test getting suggestions for ID columns."""
        extractor = TestPatternExtractor(mock_context)
        extractor.learned_patterns = {
            "id_column_tests": ["unique", "not_null"],
        }

        suggestions = extractor.get_suggestions_for_column("user_id")

        assert len(suggestions) == 2
        assert any(s.test_type == "unique" for s in suggestions)
        assert any(s.test_type == "not_null" for s in suggestions)
        assert all(s.column_name == "user_id" for s in suggestions)

    def test_get_suggestions_for_column_no_pattern(self):
        """Test getting suggestions when no pattern matches."""
        extractor = TestPatternExtractor(mock_context)
        extractor.learned_patterns = {}

        suggestions = extractor.get_suggestions_for_column("unknown_column")

        assert len(suggestions) == 0


class TestAITestSuggester:
    """Tests for the AITestSuggester class."""

    def test_suggester_initialization(self, mock_context):
        """Test initializing the AI test suggester."""
        suggester = AITestSuggester(mock_context)

        assert suggester.context == mock_context
        assert suggester.pattern_extractor is None
        assert suggester.accessor is not None

    def test_suggester_with_pattern_extractor(self, mock_context):
        """Test initializing suggester with pattern extractor."""
        extractor = TestPatternExtractor(mock_context)
        suggester = AITestSuggester(mock_context, extractor)

        assert suggester.pattern_extractor == extractor

    def test_suggest_tests_for_node_without_ai(self, mock_context, sample_node):
        """Test generating suggestions without AI (pattern-based only)."""
        extractor = TestPatternExtractor(mock_context)
        extractor.extract_patterns()

        suggester = AITestSuggester(mock_context, extractor)
        analysis = suggester.suggest_tests_for_node(sample_node, use_ai=False)

        assert isinstance(analysis, ModelTestAnalysis)
        assert analysis.model_name == sample_node.name
        assert isinstance(analysis.existing_tests, dict)
        assert isinstance(analysis.suggested_tests, dict)

    def test_pattern_suggest_tests(self, mock_context, sample_node):
        """Test pattern-based test suggestions."""
        extractor = TestPatternExtractor(mock_context)
        extractor.extract_patterns()

        suggester = AITestSuggester(mock_context, extractor)
        suggestions = suggester._pattern_suggest_tests(sample_node)

        assert isinstance(suggestions, dict)

    def test_create_test_suggestion_prompt(self, mock_context, sample_node):
        """Test creating LLM prompt for test suggestions."""
        suggester = AITestSuggester(mock_context)
        prompt = suggester._create_test_suggestion_prompt(sample_node)

        assert isinstance(prompt, list)
        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        assert sample_node.name in prompt[1]["content"]

    def test_parse_ai_response_simple(self, mock_context, sample_node):
        """Test parsing AI response with simple test names."""
        suggester = AITestSuggester(mock_context)

        response = json.dumps({
            "user_id": ["unique", "not_null"],
            "status": ["accepted_values"],
        })

        suggestions = suggester._parse_ai_response(response)

        assert "user_id" in suggestions
        assert "status" in suggestions
        assert len(suggestions["user_id"]) == 2
        assert len(suggestions["status"]) == 1

    def test_parse_ai_response_with_config(self, mock_context):
        """Test parsing AI response with test configurations."""
        suggester = AITestSuggester(mock_context)

        response = json.dumps({
            "customer_id": [
                {
                    "test_type": "relationships",
                    "reason": "Foreign key to customers",
                    "config": {"to": "ref('customers')", "field": "id"},
                }
            ],
        })

        suggestions = suggester._parse_ai_response(response)

        assert "customer_id" in suggestions
        assert len(suggestions["customer_id"]) == 1
        assert suggestions["customer_id"][0].test_type == "relationships"
        assert suggestions["customer_id"][0].config == {"to": "ref('customers')", "field": "id"}

    def test_parse_ai_response_markdown_wrapped(self, mock_context):
        """Test parsing AI response wrapped in markdown code blocks."""
        suggester = AITestSuggester(mock_context)

        response = """```json
        {
            "user_id": ["unique", "not_null"]
        }
        ```"""

        suggestions = suggester._parse_ai_response(response)

        assert "user_id" in suggestions
        assert len(suggestions["user_id"]) == 2

    def test_parse_ai_response_invalid_json(self, mock_context):
        """Test parsing invalid AI response."""
        suggester = AITestSuggester(mock_context)

        suggestions = suggester._parse_ai_response("not valid json")

        assert suggestions == {}

    def test_ai_suggest_with_mock_client(self, mock_context, sample_node):
        """Test AI suggestion with mocked LLM client."""
        extractor = TestPatternExtractor(mock_context)
        extractor.extract_patterns()

        suggester = AITestSuggester(mock_context, extractor)

        # Mock the LLM response
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "id": ["unique"],
        })

        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with mock.patch(
            "dbt_osmosis.core.test_suggestions.get_llm_client", return_value=(mock_client, "gpt-4")
        ):
            suggestions = suggester._ai_suggest_tests(sample_node)

            # Should fall back to pattern-based if AI fails
            assert isinstance(suggestions, dict)

    def test_ai_suggest_fallback_to_pattern(self, mock_context, sample_node):
        """Test that AI suggestion falls back to pattern-based on error."""
        extractor = TestPatternExtractor(mock_context)
        extractor.extract_patterns()

        suggester = AITestSuggester(mock_context, extractor)

        # Mock get_llm_client to return a client that raises an exception
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("LLM error")

        with mock.patch(
            "dbt_osmosis.core.test_suggestions.get_llm_client", return_value=(mock_client, "gpt-4")
        ):
            suggestions = suggester._ai_suggest_tests(sample_node)

            # Should fall back to pattern-based
            assert isinstance(suggestions, dict)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_suggest_tests_for_model(self, mock_context, sample_node):
        """Test suggest_tests_for_model convenience function."""
        analysis = suggest_tests_for_model(mock_context, sample_node, use_ai=False)

        assert isinstance(analysis, ModelTestAnalysis)
        assert analysis.model_name == sample_node.name

    def test_suggest_tests_for_project(self, mock_context):
        """Test suggest_tests_for_project convenience function."""
        results = suggest_tests_for_project(mock_context, use_ai=False)

        assert isinstance(results, dict)
        # Should have at least one model
        assert len(results) > 0

        for model_name, analysis in results.items():
            assert isinstance(model_name, str)
            assert isinstance(analysis, ModelTestAnalysis)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_model_analysis(self, mock_context):
        """Test analysis of a model with no columns."""
        # Create a mock node with no columns
        mock_node = mock.MagicMock()
        mock_node.name = "empty_model"
        mock_node.columns = []
        mock_node.raw_sql = "select 1 as id"

        extractor = TestPatternExtractor(mock_context)
        suggester = AITestSuggester(mock_context, extractor)

        analysis = suggester.suggest_tests_for_node(mock_node, use_ai=False)

        assert analysis.model_name == "empty_model"
        assert analysis.columns == []

    def test_node_without_columns_attribute(self, mock_context):
        """Test handling a node without columns attribute."""
        mock_node = mock.MagicMock(spec=["name", "raw_sql"])
        mock_node.name = "weird_node"
        mock_node.raw_sql = "select 1"

        extractor = TestPatternExtractor(mock_context)
        extractor._analyze_node(mock_node)

        # Should not raise an error
        assert True

    def test_pattern_extractor_empty_project(self):
        """Test pattern extraction with minimal data."""
        # Create a minimal mock context
        mock_context = mock.MagicMock()
        mock_manifest = mock.MagicMock()
        mock_manifest.nodes = {}
        mock_context.project.manifest = mock_manifest

        extractor = TestPatternExtractor(mock_context)
        extractor.extract_patterns()

        # Should complete without error
        assert isinstance(extractor.learned_patterns, dict)
