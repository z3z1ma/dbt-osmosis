<!--
Sync Impact Report
==================
Version change: [NEW] → 1.0.0
Rationale: Initial constitution ratification for dbt-osmosis project

Modified principles: N/A (initial version)

Added sections:
- Core Principles (5 principles defined)
- Development Standards
- Governance

Removed sections: N/A (initial version)

Templates requiring updates:
  ✅ .specify/templates/plan-template.md - Constitution Check section aligns with principles
  ✅ .specify/templates/spec-template.md - Requirements format supports constraint-based development
  ✅ .specify/templates/tasks-template.md - Task organization aligns with testing and user story principles
  ✅ .claude/commands/*.md - Agent references are generic (no Claude-specific naming issues)

Follow-up TODOs: None - all placeholders filled
-->

# dbt-osmosis Constitution

## Core Principles

### I. Backwards Compatibility

**Non-negotiable rules**:

- Public API changes MUST follow semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR version increments for backward-incompatible changes to public APIs
- MINOR version increments for backward-compatible functionality additions
- PATCH version increments for backward-compatible bug fixes
- Breaking changes require migration guide documentation
- Deprecated functionality MUST remain supported for at least one MINOR version cycle
- Re-exports in `src/dbt_osmosis/core/osmosis.py` MUST be maintained for backwards compatibility

**Rationale**: dbt-osmosis is integrated into user workflows via pre-commit hooks and CI/CD pipelines. Unexpected breaking changes cause production failures and erode trust. Users depend on stable imports from the core osmosis module.

### II. Idempotency and Safety

**Non-negotiable rules**:

- YAML operations MUST be idempotent: running twice produces identical results
- YAML writing MUST preserve user formatting, comments, and anchors via ruamel.yaml
- YAML deletion operations MUST be explicit and opt-in (document command, not organize)
- Database introspection MUST be read-only; never write to warehouse
- Users MUST be able to preview changes via `--dry-run` before applying
- File operations MUST NOT corrupt YAML files if interrupted (use temp + atomic rename)

**Rationale**: dbt-osmosis operates on users' source code. Data loss or corruption is unacceptable. Idempotency ensures safe re-running, while preserving formatting respects developer autonomy over code style.

### III. dbt Integration Discipline

**Non-negotiable rules**:

- Use dbt's public APIs where available (dbt.cli.main, dbt.contracts.graph.manifest)
- Internal dbt APIs (e.g., dbt.task.sql.SqlCompileRunner) MUST be documented with version compatibility
- Support dbt-core versions specified in pyproject.toml (currently >=1.8,<1.11)
- Test against all supported dbt-core versions in CI matrix
- Prefer dbt's native parsing and manifest loading over custom implementations
- Manifest loading failures MUST produce clear error messages with resolution steps

**Rationale**: dbt-osmosis extends dbt functionality. Tight coupling to dbt's internal structure is necessary but fragile. Clear version boundaries and comprehensive testing prevent breakage when dbt updates.

### IV. Test-Driven Development (NON-NEGOTIABLE)

**Non-negotiable rules**:

- New features MUST have tests written first (Red-Green-Refactor)
- Tests MUST fail before implementation begins
- Core transformations MUST have unit tests in `tests/core/` mirroring `src/dbt_osmosis/core/`
- Integration tests MUST use the `demo_duckdb/` project fixture
- Test matrix MUST cover Python 3.10-3.12 × dbt-core 1.8-1.10
- Mutation testing (mutmut) SHOULD be used for critical path validation
- Coverage MUST remain above 70% (enforced by pyproject.toml)

**Rationale**: dbt-osmosis operates in complex environments (various dbt adapters, warehouses, Python versions). Without comprehensive tests, regressions are inevitable and costly.

### V. Observability and Debuggability

**Non-negotiable rules**:

- All CLI commands MUST support `--verbose` and `--debug` flags
- Errors MUST include actionable context (file paths, column names, suggested fixes)
- Structured logging MUST be used for operations (YAML reading/writing, database queries)
- Progress indicators MUST be shown for long-running operations
- Warnings MUST be distinguishable from errors (stderr with appropriate levels)
- Users MUST be able to export operation logs for debugging

**Rationale**: Users run dbt-osmosis across diverse environments. When issues occur, they need enough context to self-diagnose or provide useful bug reports. Text I/O ensures compatibility with logging infrastructure.

## Development Standards

### Code Quality

- **Formatting**: Ruff with `--preview` mode, 100 character line length
- **Type Hints**: Required for all public functions, use `from __future__ import annotations`
- **Imports**: Auto-sorted via ruff isort rules
- **Python Version**: 3.10+ minimum, support 3.10-3.12
- **Style**: Follow PEP 8 with project-specific conventions (2-space indentation for templates, minimal comments)

### Architecture Patterns

- **Transform Pipeline**: Use functional composition with `>>` operator for YAML transformations
- **Plugin System**: Use `pluggy` for extensibility (fuzzy matching, custom transformations)
- **Caching**: Thread-safe caching for expensive operations (column introspection, YAML buffers)
- **Separation of Concerns**:
  - `path_management.py`: YAML file location logic
  - `restructuring.py`: Move/delete planning
  - `schema/`: YAML reading/parsing/writing
  - `transforms.py`: Composable transformation pipeline
  - `introspection.py`: Database schema queries
  - `inheritance.py`: Column knowledge graph

### CLI Contract

- All commands MUST use Click for CLI interface
- Support JSON output format for programmatic use
- Support human-readable output (rich formatting)
- Exit codes MUST follow Unix conventions (0 = success, non-zero = error)
- Subcommands: `yaml` (refactor/organize/document), `sql` (compile/run), `workbench`

## Governance

### Amendment Procedure

1. Proposal MUST be documented with rationale and impact analysis
2. Changes to Core Principles require explicit version bump justification
3. Amendments MUST be reflected in `.specify/memory/constitution.md`
4. Dependent templates MUST be updated per Sync Impact Report process
5. Migration guides MUST be provided for breaking governance changes

### Versioning Policy

- **MAJOR**: Backward-incompatible governance changes, principle removals/redefinitions
- **MINOR**: New principle/section added, material guidance expansion
- **PATCH**: Clarifications, wording fixes, non-semantic refinements

### Compliance Review

- All pull requests MUST verify compliance with applicable principles
- Complexity violations MUST be justified in implementation plan's Complexity Tracking table
- Use `CLAUDE.md` and `AGENTS.md` for runtime development guidance
- Constitution supersedes all other practice documents

### Quality Gates

- `task format` MUST pass (ruff format + import sorting)
- `task lint` MUST pass (ruff check)
- `task test` MUST pass for full matrix
- Manual review of YAML output formatting for ruamel.yaml preservation

**Version**: 1.0.0 | **Ratified**: 2026-01-02 | **Last Amended**: 2026-01-02
