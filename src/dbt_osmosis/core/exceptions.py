"""Custom exception hierarchy for dbt-osmosis.

This module defines a comprehensive exception hierarchy for better error handling
and debugging. All dbt-osmosis specific exceptions inherit from OsmosisError.
"""

from __future__ import annotations

__all__ = [
    "OsmosisError",
    "ConfigurationError",
    "MissingOsmosisConfig",
    "ValidationError",
    "YamlValidationError",
    "YAMLError",
    "PathResolutionError",
    "DatabaseError",
    "IntrospectionError",
    "CatalogGenerationError",
    "TransformError",
    "LLMError",
    "LLMConfigurationError",
    "LLMResponseError",
]


class OsmosisError(Exception):
    """Base exception class for all dbt-osmosis errors.

    All custom exceptions in dbt-osmosis should inherit from this class.
    This allows users to catch all dbt-osmosis specific errors with a single
    except clause: `except OsmosisError`.
    """

    pass


class ConfigurationError(OsmosisError):
    """Raised when there's a configuration error.

    This includes missing or invalid configuration values, improperly
    set environment variables, or configuration parsing errors.
    """

    pass


class MissingOsmosisConfig(ConfigurationError):
    """Raised when required osmosis configuration is missing.

    Specifically raised when the `dbt-osmosis: <path>` configuration
    key is not set for a model or source node.
    """

    pass


class ValidationError(OsmosisError):
    """Raised when validation fails.

    This includes schema validation, data validation, or any
    validation checks that fail during processing.
    """

    pass


class YamlValidationError(ValidationError):
    """Raised when YAML validation fails.

    This includes structural issues, missing required fields,
    or invalid data types in YAML files.
    """

    pass


class YAMLError(OsmosisError):
    """Raised when there's a YAML processing error.

    This covers YAML parsing errors, malformed YAML syntax,
    and general YAML file handling issues.
    """

    pass


class PathResolutionError(OsmosisError):
    """Raised when path resolution or validation fails.

    This includes security violations (path traversal attempts),
    invalid path formats, or paths that don't exist when they should.
    """

    pass


class DatabaseError(OsmosisError):
    """Raised when database operations fail.

    This covers connection errors, query failures, introspection
    issues, and other database-related problems.
    """

    pass


class IntrospectionError(DatabaseError):
    """Raised when database introspection fails.

    Specifically raised when column introspection, catalog generation,
    or metadata retrieval from the database fails.
    """

    pass


class CatalogGenerationError(DatabaseError):
    """Raised when catalog generation fails.

    This is raised during dbt catalog generation when errors occur
    in the process of building the catalog artifact.
    """

    pass


class TransformError(OsmosisError):
    """Raised when a transform operation fails.

    This covers errors in the transform pipeline, including
    invalid operations, chaining errors, or transformation failures.
    """

    pass


class LLMError(OsmosisError):
    """Raised when LLM (Large Language Model) operations fail.

    This includes API errors, invalid responses, configuration issues,
    or failures in AI-generated documentation.
    """

    pass


class LLMConfigurationError(LLMError, ConfigurationError):
    """Raised when LLM configuration is invalid or missing.

    Specifically raised for missing API keys, invalid provider settings,
    or improperly configured LLM endpoints.
    """

    pass


class LLMResponseError(LLMError):
    """Raised when LLM returns an invalid or unexpected response.

    This includes empty responses, malformed JSON, or responses that
    don't match the expected format.
    """

    pass
