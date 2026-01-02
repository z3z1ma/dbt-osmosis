"""Custom exceptions for dbt-osmosis."""

from __future__ import annotations


class MissingOsmosisConfig(Exception):
    """Raised when an osmosis configuration is missing."""

    pass


class OsmosisError(Exception):
    """Base exception class for dbt-osmosis errors."""

    pass


class ConfigurationError(OsmosisError):
    """Raised when there's a configuration error."""

    pass


class ValidationError(OsmosisError):
    """Raised when there's a validation error."""

    pass


class YAMLError(OsmosisError):
    """Raised when there's a YAML processing error."""

    pass


class DatabaseError(OsmosisError):
    """Raised when there's a database operation error."""

    pass
