"""Protocol definitions for dbt types used throughout dbt-osmosis.

This module provides Protocol-based type hints for dbt interfaces, allowing
dbt-osmosis to maintain type safety without depending on dbt's internal
implementation details. Protocols are structural subtypes - any class that
matches the signature is compatible.

This isolates dbt-osmosis from breaking changes in dbt-core updates, as we only
depend on the interface shapes, not concrete classes.
"""

from __future__ import annotations

import threading
import typing as t
from concurrent.futures import ThreadPoolExecutor

import ruamel.yaml

if t.TYPE_CHECKING:
    from dbt.contracts.results import CatalogResults


class DbtAdapterProtocol(t.Protocol):
    """Protocol for dbt adapter instances.

    Covers basic connection and query methods used by dbt-osmosis.
    """

    connections: t.Any  # dbt's connection manager interface
    type: str  # Adapter type (e.g., "postgres", "bigquery")

    def acquire_connection(self) -> t.Any:
        """Acquire a connection from the pool."""
        ...

    def set_macro_resolver(self, manifest: t.Any) -> None:
        """Set the macro resolver for the adapter."""
        ...


class DbtRuntimeConfigProtocol(t.Protocol):
    """Protocol for dbt runtime configuration.

    Used to access project settings, credentials, and threading config.
    """

    project_name: str
    threads: int
    vars: t.Any  # dbt's Var container
    credentials: t.Any  # Database credentials

    def to_dict(self) -> dict[str, t.Any]:
        """Convert config to dictionary representation."""
        ...


class DbtManifestProtocol(t.Protocol):
    """Protocol for dbt manifest.

    Provides access to parsed dbt nodes and metadata.
    """

    nodes: dict[str, t.Any]  # unique_id -> Node
    sources: dict[str, t.Any]  # unique_id -> Source
    metadata: t.Any  # Manifest metadata

    def build_flat_graph(self) -> None:
        """Build the flat graph for node traversal."""
        ...


class DbtProjectContextProtocol(t.Protocol):
    """Protocol for dbt project context.

    Provides access to dbt project resources including config, manifest,
    and adapter. Used throughout dbt-osmosis for project-level operations.
    """

    config: t.Any  # DbtConfiguration
    runtime_cfg: DbtRuntimeConfigProtocol
    manifest: DbtManifestProtocol
    _adapter: t.Any | None  # DbtAdapterProtocol
    _adapter_mutex: threading.Lock

    @property
    def adapter(self) -> t.Any:
        """Get or create the adapter instance."""
        ...


class YamlRefactorContextProtocol(t.Protocol):
    """Protocol for YAML refactor context.

    The primary context object passed through the refactoring pipeline.
    Contains project context, settings, and execution resources.
    """

    project: DbtProjectContextProtocol
    settings: t.Any  # YamlRefactorSettings
    pool: ThreadPoolExecutor
    yaml_handler: ruamel.yaml.YAML
    yaml_handler_lock: threading.Lock
    placeholders: tuple[str, ...]
    _catalog: CatalogResults | None
    _mutation_count: int

    def register_mutations(self, count: int) -> None:
        """Register mutation count for tracking changes."""
        ...

    @property
    def mutation_count(self) -> int:
        """Get the total mutation count."""
        ...

    @property
    def mutated(self) -> bool:
        """Check if any mutations have been performed."""
        ...

    @property
    def source_definitions(self) -> dict[str, t.Any]:
        """Get source definitions from dbt config."""
        ...

    @property
    def ignore_patterns(self) -> list[str]:
        """Get column ignore patterns from dbt config."""
        ...

    @property
    def yaml_settings(self) -> dict[str, t.Any]:
        """Get YAML formatting settings from dbt config."""
        ...

    def read_catalog(self) -> CatalogResults | None:
        """Read and cache the catalog file."""
        ...


# Export all protocols
__all__ = [
    "DbtAdapterProtocol",
    "DbtRuntimeConfigProtocol",
    "DbtManifestProtocol",
    "DbtProjectContextProtocol",
    "YamlRefactorContextProtocol",
]
