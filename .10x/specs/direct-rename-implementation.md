Status: active
Created: 2026-11-16
Updated: 2026-11-16

# Opt-in direct rename inheritance

Authorized implementation slice of roadmap issue #440. Existing defaults MUST remain unchanged. Research branch 0011fad is evidence, not merged authority; no research-only publication is part of this change.

## Contract
`use-sqlglot-lineage` resolved through SettingsResolver MUST default false. Initial dialect support is DuckDB and Trino only, explicitly documented. Trino catalog/schema/quoted identifiers require dialect normalization and collision rejection. A configured opt-in node with available compiled SQL can inherit through direct column references/aliases and CTEs with schema-qualified stars only when every projection on the path preserves identity and exactly one eligible ancestor relation and column resolve. Functions, casts, arithmetic, aggregates, set operations, ambiguity, and unresolved stars MUST NOT yield a donor. Semantic rejection suppresses name fallback for that target; unavailable inputs (missing compiled SQL, unsupported adapter, parser failure or resource cap) use legacy matching with diagnostic logging. Explicit progenitor override retains its existing deliberate user-authority behavior.

Target and donor column identity MUST remain separate. Target settings/explicit description and force-inherit behavior remain governed by existing pipeline; donor raw docs references and other inherited keys are read using the actual donor column. Opt-in lineage MAY compile selected needed model nodes through the established dbt compilation path once per loaded manifest generation when compiled SQL is absent. It MUST NOT execute the model SQL. Compilation runs dbt macros which MAY query the warehouse, including in dry-run; no zero-query promise is made. Dry-run MUST still preserve YAML bytes/mtime. Compilation failures MUST log per-model context and fall back explicitly. Disabled mode MUST not compile. Preserve real model context (`this`, config, package refs), manifest locking and no leaked temporary nodes. No new metadata keys.

## Bounds
No persistent cache in this first slice: prevents stale schema/relation/description reuse. Eligible relations MUST be capped at 256 and total schema columns at 4096. Input length MUST be capped before parsing (64 KiB), AST node count capped before qualification/lineage (4096), and intermediate lineage walk count capped (256). These are structural limits, NOT a hard wall-clock timeout promise. Parsing errors or recursion failures are unavailable outcomes. Tests MUST cover caps and demonstrate that disabled mode does not invoke parsing.

## Acceptance
Qualified direct rename and CTE alias resolve to unique donor; hidden transformation and unresolved star do not. Ambiguous relation mapping never selects arbitrarily. Missing SQL/unsupported dialect uses unchanged fallback. Actual inheritance tests verify explicit description, force/override, unrendered docs-block donor identity, dry-run/idempotence where applicable. Existing inheritance suite passes. All runtime changes have independent review, exact-head CI before merge and post-merge verification. #440 stays open for unimplemented dialects/derived-expression scope rather than claimed fully resolved.

## Tradeoffs
DuckDB has real compiled fixture evidence. Trino is covered by offline compiled-SQL-shaped fixtures, including quoted catalog/schema identities and case-insensitive collision rejection; no warehouse certification is claimed. Official identifier semantics: https://trino.io/docs/current/language/reserved.html (checked 2026-11-16), identifiers are not case-sensitive. No timeout executor/process service or cache abstraction is justified for this first opt-in slice. Structural caps bound accepted work size but do not guarantee runtime; documentation must state that limitation.

## Source-backed compilation amendment
A real DuckDB compile followed by create_dbt_project_context drops compiled_code, making an in-memory-only resolver ineffective in normal CLI. Parent authorized targeted opt-in compilation rather than a new artifactinput/freshness API. core/sql_operations.py establishes SqlCompileRunner and manifest_mutex, but its temporary SqlOperation context cannot substitute blindly for real model context. Compile only selected needed models; no external warehouse is used in validation.

Compilation attempts MUST be scoped to the manifest object generation: reload replaces nodes and may remove compiled SQL. Reset attempt identities when the manifest object changes, retaining an object reference to avoid id reuse. No cross-generation compiled artifact reuse.
