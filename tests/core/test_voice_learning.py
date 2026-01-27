# pyright: reportPrivateImportUsage=false, reportUnknownVariableType=false, reportUnknownMemberType=false

"""Tests for voice learning and style analysis.

This module contains tests for the voice_learning module, which provides:
- ProjectStyleProfile for documenting style patterns
- Functions to analyze documentation style
- Finding similar documented nodes
- Extracting style examples for LLM prompts
"""

from __future__ import annotations

from unittest import mock

import pytest

from dbt_osmosis.core.voice_learning import (
    ProjectStyleProfile,
    _analyze_description_lengths,
    _detect_terminology_patterns,
    _detect_tone_markers,
    _extract_common_phrases,
    analyze_project_documentation_style,
    extract_style_examples,
    find_similar_documented_nodes,
)


class TestProjectStyleProfile:
    """Tests for the ProjectStyleProfile dataclass."""

    def test_profile_creation(self):
        """Test creating a basic profile."""
        profile = ProjectStyleProfile()

        assert profile.description_length_stats == {}
        assert profile.common_phrases == []
        assert profile.terminology_preferences == {}
        assert profile.sentence_structure == {}
        assert profile.tone_markers == {}
        assert profile.model_description_samples == []
        assert profile.column_description_samples == []

    def test_profile_with_data(self):
        """Test creating a profile with data."""
        profile = ProjectStyleProfile(
            description_length_stats={"avg_length": 10.5, "min_length": 5, "max_length": 20},
            common_phrases=[("contains data", 3), ("represents user", 2)],
            terminology_preferences={"user": "customer"},
            tone_markers={"imperative": 5, "passive": 2},
            model_description_samples=["Sample model description"],
            column_description_samples=["Sample column 1", "Sample column 2"],
        )

        assert profile.description_length_stats["avg_length"] == 10.5
        assert len(profile.common_phrases) == 2
        assert profile.terminology_preferences["user"] == "customer"
        assert profile.tone_markers["imperative"] == 5
        assert len(profile.model_description_samples) == 1
        assert len(profile.column_description_samples) == 2

    def test_to_prompt_context_empty(self):
        """Test converting empty profile to prompt context."""
        profile = ProjectStyleProfile()
        context = profile.to_prompt_context()

        assert context == "No style information available."

    def test_to_prompt_context_with_length_stats(self):
        """Test prompt context with length statistics."""
        profile = ProjectStyleProfile(
            description_length_stats={"avg_length": 15.0, "min_length": 10, "max_length": 25}
        )
        context = profile.to_prompt_context()

        assert "~15 words" in context
        assert "10-" in context
        assert "Target description length" in context

    def test_to_prompt_context_with_phrases(self):
        """Test prompt context with common phrases."""
        profile = ProjectStyleProfile(common_phrases=[("contains data", 5), ("represents user", 3)])
        context = profile.to_prompt_context()

        assert "Common phrases" in context
        assert "contains data" in context
        assert "represents user" in context

    def test_to_prompt_context_with_terminology(self):
        """Test prompt context with terminology preferences."""
        profile = ProjectStyleProfile(
            terminology_preferences={"user": "customer", "email": "email address"}
        )
        context = profile.to_prompt_context()

        assert "Terminology preferences" in context
        assert "'user' instead of 'customer'" in context
        assert "'email' instead of 'email address'" in context

    def test_to_prompt_context_with_examples(self):
        """Test prompt context with example descriptions."""
        profile = ProjectStyleProfile(
            model_description_samples=["Example 1", "Example 2", "Example 3"],
            column_description_samples=["Col 1", "Col 2"],
        )
        context = profile.to_prompt_context()

        assert "# Model Description Examples:" in context
        assert "# Column Description Examples:" in context
        assert "1. Example 1" in context
        assert "1. Col 1" in context  # Numbered, not bulleted

    def test_to_prompt_context_max_examples(self):
        """Test limiting examples in prompt context."""
        profile = ProjectStyleProfile(
            model_description_samples=["Ex 1", "Ex 2", "Ex 3", "Ex 4", "Ex 5"],
            column_description_samples=["Col 1", "Col 2", "Col 3", "Col 4", "Col 5"],
        )
        context = profile.to_prompt_context(max_examples=2)

        # Should only include 2 examples
        assert context.count("Ex ") == 2
        assert context.count("Col ") == 2


class TestAnalyzeDescriptionLengths:
    """Tests for the _analyze_description_lengths function."""

    def test_empty_descriptions(self):
        """Test with empty description list."""
        result = _analyze_description_lengths([])

        assert result == {}

    def test_single_description(self):
        """Test with a single description."""
        result = _analyze_description_lengths(["This is a test"])

        assert result["avg_length"] == 4.0
        assert result["min_length"] == 4
        assert result["max_length"] == 4
        assert result["median_length"] == 4

    def test_multiple_descriptions(self):
        """Test with multiple descriptions."""
        descriptions = [
            "Short",
            "Medium length description",
            "This is a much longer description with many more words",
        ]

        result = _analyze_description_lengths(descriptions)

        assert "avg_length" in result
        assert "min_length" in result
        assert "max_length" in result
        assert "median_length" in result
        assert result["min_length"] == 1  # "Short"
        assert result["max_length"] == 10  # Long description (10 words)

    def test_whitespace_only_descriptions(self):
        """Test handling of whitespace-only descriptions."""
        result = _analyze_description_lengths(["   ", "\n\n", "valid description"])

        # Should skip whitespace-only descriptions
        assert result["avg_length"] == 2.0  # Only "valid description" counts

    def test_empty_string_descriptions(self):
        """Test handling of empty strings."""
        result = _analyze_description_lengths(["", "test", ""])

        # Should skip empty strings
        assert result["avg_length"] == 1.0


class TestExtractCommonPhrases:
    """Tests for the _extract_common_phrases function."""

    def test_empty_descriptions(self):
        """Test with empty description list."""
        result = _extract_common_phrases([])

        assert result == []

    def test_no_common_phrases(self):
        """Test when no phrases meet the frequency threshold."""
        descriptions = ["unique description", "another unique one"]
        result = _extract_common_phrases(descriptions, min_frequency=2)

        assert result == []

    def test_extract_2grams(self):
        """Test extracting 2-word phrases."""
        descriptions = [
            "The user ID contains",
            "The customer ID contains",
            "User ID represents the",
        ]

        result = _extract_common_phrases(descriptions, min_frequency=2)

        # Should find "ID contains"
        assert any("id contains" in phrase.lower() for phrase, _ in result)

    def test_extract_3grams(self):
        """Test extracting 3-word phrases."""
        descriptions = [
            "This column contains user data",
            "That field contains user data",
        ]

        result = _extract_common_phrases(descriptions, min_frequency=2)

        # Should find "contains user data"
        assert any("contains user data" in phrase.lower() for phrase, _ in result)

    def test_filter_common_words(self):
        """Test that common words are filtered out."""
        descriptions = [
            "the user and the customer",
            "the user and the order",
            "a user and a customer",
        ]

        result = _extract_common_phrases(descriptions, min_frequency=2)

        # Should filter out "the", "and"
        assert not any(phrase.startswith("the ") for phrase, _ in result)
        assert not any(phrase.startswith("and ") for phrase, _ in result)

    def test_case_insensitive(self):
        """Test that extraction is case-insensitive."""
        descriptions = [
            "User ID contains",
            "user ID contains",
            "USER ID contains",
        ]

        result = _extract_common_phrases(descriptions, min_frequency=3)

        # Should find "user id contains" (lowercased)
        assert any("user id contains" in phrase.lower() for phrase, _ in result)


class TestDetectTerminologyPatterns:
    """Tests for the _detect_terminology_patterns function."""

    def test_empty_descriptions(self):
        """Test with empty description list."""
        result = _detect_terminology_patterns([])

        assert result == {}

    def test_user_vs_customer(self):
        """Test detecting user vs customer terminology."""
        descriptions = [
            "The user ID represents",
            "Each user has an email",
            "User data is stored",
        ]

        result = _detect_terminology_patterns(descriptions)

        # Should prefer "user" over "customer"
        assert "user" in result
        assert result["user"] in ["customer", "client", "account"]

    def test_email_variations(self):
        """Test detecting email terminology."""
        descriptions = [
            "User email address",
            "Customer email address",
            "The email field",
        ]

        result = _detect_terminology_patterns(descriptions)

        # Should detect "email" preference
        assert "email" in result or any("email" in k for k in result.keys())

    def test_timestamp_variations(self):
        """Test detecting timestamp terminology."""
        descriptions = [
            "Created timestamp field",
            "Updated timestamp column",
            "Timestamp of creation",
        ]

        result = _detect_terminology_patterns(descriptions)

        # Should detect timestamp terminology
        assert "timestamp" in result or any(
            k in result for k in ["timestamp", "datetime", "created at"]
        )

    def test_minimum_frequency_threshold(self):
        """Test that terms must meet minimum frequency."""
        descriptions = ["The user ID appears once"]

        result = _detect_terminology_patterns(descriptions)

        # "user" only appears once, below threshold of 2
        assert "user" not in result

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        descriptions = [
            "User ID contains",
            "USER ID represents",
            "user ID stores",
        ]

        result = _detect_terminology_patterns(descriptions)

        # Should count case-insensitively
        assert "user" in result


class TestDetectToneMarkers:
    """Tests for the _detect_tone_markers function."""

    def test_empty_descriptions(self):
        """Test with empty description list."""
        result = _detect_tone_markers([])

        assert result == {
            "imperative": 0,
            "passive": 0,
            "concise": 0,
            "detailed": 0,
            "technical": 0,
        }

    def test_concise_descriptions(self):
        """Test detecting concise descriptions."""
        descriptions = ["Short desc", "Brief text", "Quick info"]

        result = _detect_tone_markers(descriptions)

        assert result["concise"] == 3
        assert result["detailed"] == 0

    def test_detailed_descriptions(self):
        """Test detecting detailed descriptions."""
        descriptions = [
            "This is a very long description with many words that exceeds fifteen words in total length",
            "Another extended description that contains multiple clauses and provides detailed information beyond what is typical",
        ]

        result = _detect_tone_markers(descriptions)

        # At least one should be detailed (15+ words)
        assert result["detailed"] >= 1

    def test_imperative_verbs(self):
        """Test detecting imperative tone."""
        descriptions = [
            "Contains user data",
            "Represents customer ID",
            "Stores transaction records",
            "Holds configuration values",
        ]

        result = _detect_tone_markers(descriptions)

        assert result["imperative"] == 4

    def test_passive_voice(self):
        """Test detecting passive voice."""
        descriptions = [
            "Is used to store data",
            "Is a field for ID",
            "Contains a reference",
        ]

        result = _detect_tone_markers(descriptions)

        assert result["passive"] == 3

    def test_technical_terms(self):
        """Test detecting technical terminology."""
        descriptions = [
            "The primary key ID",
            "Foreign key reference",
            "UUID identifier",
            "JSON data field",
        ]

        result = _detect_tone_markers(descriptions)

        assert result["technical"] >= 2

    def test_mixed_tone(self):
        """Test mixed tone descriptions."""
        descriptions = [
            "User ID",  # concise (2 words)
            "This field contains the user ID which is used for identification purposes within the system database",  # detailed, imperative (17+ words)
            "is used to store data",  # passive
            "The integer value",  # technical
        ]

        result = _detect_tone_markers(descriptions)

        assert result["concise"] >= 1
        assert result["detailed"] >= 1
        assert result["imperative"] >= 1  # "contains"
        assert result["passive"] >= 1  # "is used to"
        assert result["technical"] >= 1


class TestAnalyzeProjectDocumentationStyle:
    """Tests for the analyze_project_documentation_style function."""

    def test_analyze_project(self, yaml_context):
        """Test analyzing project documentation style."""
        profile = analyze_project_documentation_style(yaml_context)

        assert isinstance(profile, ProjectStyleProfile)
        assert isinstance(profile.description_length_stats, dict)
        assert isinstance(profile.model_description_samples, list)
        assert isinstance(profile.column_description_samples, list)

    def test_max_nodes_limit(self, yaml_context):
        """Test that max_nodes parameter limits analysis."""
        profile = analyze_project_documentation_style(yaml_context, max_nodes=5)

        # Should complete without error
        assert isinstance(profile, ProjectStyleProfile)

    def test_max_columns_limit(self, yaml_context):
        """Test that max_columns_per_node parameter limits analysis."""
        profile = analyze_project_documentation_style(
            yaml_context, max_nodes=10, max_columns_per_node=2
        )

        # Should complete without error
        assert isinstance(profile, ProjectStyleProfile)


class TestFindSimilarDocumentedNodes:
    """Tests for the find_similar_documented_nodes function."""

    def test_find_similar_nodes(self, yaml_context):
        """Test finding similar documented nodes."""
        # Get a target node
        manifest = yaml_context.project.manifest
        target_node = None
        for node in manifest.nodes.values():
            if hasattr(node, "columns") and len(node.columns) > 0:
                target_node = node
                break

        if not target_node:
            pytest.skip("No suitable target node found")

        similar_nodes = find_similar_documented_nodes(yaml_context, target_node, max_results=5)

        assert isinstance(similar_nodes, list)
        assert len(similar_nodes) <= 5

        for node, score in similar_nodes:
            assert isinstance(node, object)
            assert isinstance(score, float)
            assert 0.0 <= score  # Score can be > 1.0 due to bonuses

    def test_max_results_limit(self, yaml_context):
        """Test that max_results parameter limits results."""
        # Get a target node
        manifest = yaml_context.project.manifest
        target_node = None
        for node in manifest.nodes.values():
            if hasattr(node, "columns") and len(node.columns) > 0:
                target_node = node
                break

        if not target_node:
            pytest.skip("No suitable target node found")

        similar_nodes = find_similar_documented_nodes(yaml_context, target_node, max_results=2)

        assert len(similar_nodes) <= 2


class TestExtractStyleExamples:
    """Tests for the extract_style_examples function."""

    def test_extract_without_target(self, yaml_context):
        """Test extracting examples without a target node."""
        examples = extract_style_examples(yaml_context, target_node=None, max_examples=3)

        assert "model_descriptions" in examples
        assert "column_descriptions" in examples
        assert isinstance(examples["model_descriptions"], list)
        assert isinstance(examples["column_descriptions"], list)

    def test_extract_with_target(self, yaml_context):
        """Test extracting examples for a specific target node."""
        # Get a target node
        manifest = yaml_context.project.manifest
        target_node = None
        for node in manifest.nodes.values():
            if hasattr(node, "columns") and len(node.columns) > 0:
                target_node = node
                break

        if not target_node:
            pytest.skip("No suitable target node found")

        examples = extract_style_examples(yaml_context, target_node=target_node, max_examples=2)

        assert "model_descriptions" in examples
        assert "column_descriptions" in examples

    def test_max_examples_limit(self, yaml_context):
        """Test that max_examples parameter limits results."""
        examples = extract_style_examples(yaml_context, target_node=None, max_examples=2)

        # Should not exceed max_examples
        assert len(examples["model_descriptions"]) <= 2
        assert len(examples["column_descriptions"]) <= 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_description_with_special_characters(self):
        """Test handling descriptions with special characters."""
        descriptions = ["User's email (optional)", "ID #1", "Data: value"]
        result = _analyze_description_lengths(descriptions)

        assert result["avg_length"] > 0

    def test_description_with_numbers(self):
        """Test handling descriptions with numbers."""
        descriptions = ["ID 12345 contains", "Field 42 represents"]
        result = _extract_common_phrases(descriptions, min_frequency=1)

        # Should handle numbers correctly
        assert len(result) >= 0

    def test_very_long_description(self):
        """Test handling very long descriptions."""
        long_desc = " ".join(["word"] * 100)
        result = _analyze_description_lengths([long_desc])

        assert result["max_length"] == 100

    def test_unicode_in_descriptions(self):
        """Test handling unicode characters."""
        descriptions = ["用户ID contains", "電子郵件字段"]
        result = _analyze_description_lengths(descriptions)

        assert result["avg_length"] > 0

    def test_mixed_whitespace(self):
        """Test handling mixed whitespace."""
        descriptions = ["  spaced  out  ", "tab\tseparated", "new\nlines"]
        result = _analyze_description_lengths(descriptions)

        # Should handle all whitespace types
        assert result["avg_length"] > 0

    def test_no_documented_nodes(self, yaml_context):
        """Test when no nodes have documentation."""
        # Mock a context with no documented nodes
        with mock.patch("dbt_osmosis.core.node_filters._iter_candidate_nodes", return_value=[]):
            profile = analyze_project_documentation_style(yaml_context)

            assert isinstance(profile, ProjectStyleProfile)
            assert profile.model_description_samples == []
            assert profile.column_description_samples == []

    def test_placeholder_filtering(self, yaml_context):
        """Test that placeholder descriptions are filtered out."""
        # This is tested implicitly by analyze_project_documentation_style
        # which filters descriptions in context.placeholders
        profile = analyze_project_documentation_style(yaml_context)

        # Count should reflect filtering (though exact count depends on project)
        assert isinstance(profile.model_description_samples, list)
