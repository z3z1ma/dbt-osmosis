# Implementation Plan: Unified Configuration Resolution System

**Branch**: `001-unified-config-resolution` | **Date**: 2026-01-02 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-unified-config-resolution/spec.md`

## Summary

Create a unified configuration resolution system for dbt-osmosis that:

1. **ConfigResolver**: Resolves configuration values from multiple sources (column meta, node meta, config.extra, config.meta, project vars, dbt-osmosis.yml) with documented precedence, supporting both kebab-case and snake_case variants
2. **PropertyAccessor**: Unified interface for accessing model properties from either manifest (rendered) or YAML (unrendered) sources
3. **Supplementary file support**: Optional `dbt-osmosis.yml` for project-level configuration outside dbt's hot path

This system consolidates 32+ scattered configuration access patterns, provides forward/backward compatibility across dbt versions 1.8-1.11+, and enables the unrendered jinja feature (doc blocks) to work consistently.

**Technical Approach**: Extend the existing `SettingsResolver` class in `introspection.py` to cover all sources, create new `PropertyAccessor` for model properties, and maintain backward compatibility with `_get_setting_for_node`.

## Technical Context

**Language/Version**: Python 3.10-3.12 (as specified in pyproject.toml)
**Primary Dependencies**:
- dbt-core >=1.8,<1.11 (dbt.contracts.graph.nodes, dbt.cli.main)
- ruamel.yaml (YAML reading/parsing with formatting preservation)
- pluggy (plugin system for extensibility)

**Storage**:
- YAML files (model/schema YAMLs, dbt-osmosis.yml, dbt_project.yml)
- In-memory manifest (dbt's parsed manifest.json)
- Thread-safe caches (existing _COLUMN_LIST_CACHE, _YAML_BUFFER_CACHE)

**Testing**: pytest with demo_duckdb project fixture, test matrix covering Python 3.10-3.12 × dbt-core 1.8-1.10

**Target Platform**: Cross-platform (Linux, macOS, Windows) - dbt-osmosis is a CLI tool

**Project Type**: Single project (Python library with CLI)

**Performance Goals**:
- Configuration resolution < 10ms per query (SC-001)
- Zero increase in memory footprint (SC-009)
- Use existing caches where possible

**Constraints**:
- MUST maintain backward compatibility with `_get_setting_for_node` (FR-009)
- MUST support dbt-core 1.8 through 1.11+ (FR-015, FR-016, FR-017)
- MUST preserve ruamel.yaml formatting (Constitution II)
- MUST be idempotent and safe (Constitution II)

**Scale/Scope**:
- Consolidate ~32 scattered configuration access patterns
- Extend existing `SettingsResolver` (partial implementation in introspection.py)
- Create new `PropertyAccessor` class
- Add support for `dbt-osmosis.yml` file

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Backwards Compatibility

| Principle | Compliance | Notes |
|-----------|------------|-------|
| Public API changes follow semver | ✅ PASS | New `ConfigResolver` and `PropertyAccessor` APIs are additions, not replacements |
| MAJOR for breaking changes | ✅ PASS | No breaking changes - `_get_setting_for_node` maintained (FR-009) |
| Deprecated features supported 1+ MINOR | ✅ PASS | Legacy `_get_setting_for_node` kept for backward compatibility |
| Re-exports in osmosis.py maintained | ✅ PASS | New classes will be added to `__all__` in core/osmosis.py |

### II. Idempotency and Safety

| Principle | Compliance | Notes |
|-----------|------------|-------|
| YAML operations idempotent | ✅ PASS | Configuration resolution is read-only |
| Preserve formatting via ruamel.yaml | ✅ PASS | Reads use existing YAML buffer cache |
| Database introspection read-only | ✅ PASS | Configuration resolution doesn't touch database |
| --dry-run support | ✅ PASS | Existing CLI flags preserved |
| File operations safe | ✅ PASS | `dbt-osmosis.yml` read is safe; write operations use existing safe patterns |

### III. dbt Integration Discipline

| Principle | Compliance | Notes |
|-----------|------------|-------|
| Use dbt public APIs | ✅ PASS | Uses dbt.contracts.graph.nodes, dbt.cli.main.dbtRunner |
| Internal APIs documented | ✅ PASS | Will document version compatibility for `unrendered_config` field |
| Support dbt-core >=1.8,<1.11 | ✅ PASS | FR-015, FR-016, FR-017 cover version compatibility |
| Test against all versions | ✅ PASS | Existing test matrix covers Python 3.10-3.12 × dbt 1.8-1.10 |
| Prefer dbt native parsing | ✅ PASS | Uses dbt's manifest loading |

### IV. Test-Driven Development (NON-NEGOTIABLE)

| Principle | Compliance | Notes |
|-----------|------------|-------|
| Tests written first | ✅ PASS | Plan includes test creation before implementation |
| Core transformations have unit tests | ✅ PASS | Will create tests/core/test_config_resolution.py |
| Integration tests use demo_duckdb | ✅ PASS | Will add integration tests for configuration sources |
| Test matrix covers versions | ✅ PASS | Existing CI matrix covers Python 3.10-3.12 × dbt 1.8-1.10 |
| Coverage remains above 70% | ✅ PASS | New code will be tested to maintain coverage |

### V. Observability and Debuggability

| Principle | Compliance | Notes |
|-----------|------------|-------|
| --verbose and --debug flags | ✅ PASS | Uses existing logger infrastructure |
| Errors include actionable context | ✅ PASS | FR-010 requires logging which source provided value |
| Structured logging | ✅ PASS | Will use existing logger module |
| Progress indicators | ✅ PASS | N/A for configuration resolution (fast operation) |
| Export operation logs | ✅ PASS | Uses existing logging infrastructure |

**Overall Status**: ✅ **ALL GATES PASSED** - Proceed to Phase 0 research

## Project Structure

### Documentation (this feature)

```text
specs/001-unified-config-resolution/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0: Technical research and decisions
├── data-model.md        # Phase 1: Entity definitions
├── quickstart.md        # Phase 1: Developer quickstart guide
├── contracts/           # Phase 1: API contracts
│   └── config-resolver.py  # ConfigResolver interface
└── tasks.md             # Phase 2: Implementation tasks (NOT created by plan)
```

### Source Code (repository root)

```text
src/dbt_osmosis/core/
├── config.py            # Existing: dbt project initialization
├── settings.py          # Existing: YamlRefactorSettings/Context
├── introspection.py     # MODIFY: Extend SettingsResolver, add PropertyAccessor
├── path_management.py   # Existing: YAML path resolution
├── schema/
│   ├── reader.py        # Existing: YAML buffer cache
│   └── parser.py        # Existing: ruamel.yaml instance creation
└── osmosis.py           # MODIFY: Add new classes to __all__ exports

tests/core/
├── test_config_resolution.py      # NEW: ConfigResolver tests
├── test_property_accessor.py      # NEW: PropertyAccessor tests
└── test_introspection.py          # MODIFY: Add tests for new resolution sources

demo_duckdb/
├── dbt-osmosis.yml    # NEW: Example supplementary config file
└── dbt_project.yml    # MODIFY: Add example configurations
```

**Structure Decision**: Single project structure (Python library). New classes (`ConfigResolver`, `PropertyAccessor`) will be added to `introspection.py` alongside the existing `SettingsResolver`. The `dbt-osmosis.yml` file will be read from project root using existing path resolution patterns.

## Complexity Tracking

> No violations requiring justification - all constitution gates passed.
