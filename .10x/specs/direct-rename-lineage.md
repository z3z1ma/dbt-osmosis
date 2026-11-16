Status: draft
Created: 2026-11-16
Updated: 2026-11-16

# Direct-rename lineage candidate contract

This is a proposed bounded contract for #440, **not implementation authorization or shipped behavior**. Research: `.10x/research/2026-11-16-lineage-440.md`. Existing name-based inheritance remains unchanged.

## Proposed boundary
An opt-in candidate resolver SHOULD accept existing compiled SQL, dialect, schema and eligible-ancestor relation identities; it MUST NOT compile SQL, query a warehouse, mutate YAML or copy descriptions. Return a unique `(dbt_unique_id, donor_column)` only for a fully resolved direct-column path. All intermediate expressions MUST preserve column identity; transformations, aggregates, unions, unresolved stars, multiple donors and ambiguous relation identities MUST produce a typed no-candidate reason. Unsupported/missing inputs MUST be distinguishable from an observed semantic rejection.

The integration MUST retain target-column identity separately from donor identity. Existing settings resolution, explicit descriptions, force-inherit and explicit progenitor override contracts MUST be preserved. Raw `{{ doc(...) }}` text MUST be fetched by donor identity and flow through existing PropertyAccessor/schema helpers. No new provenance YAML key is specified here.

## Given/when/then acceptance scenarios
- Given a qualified direct rename or CTE alias with complete schema and one ancestor relation, when resolving, then return that ancestor column.
- Given a function hidden inside a CTE, an aggregate or multiple origin columns, then return a semantic rejection even if one leaf remains.
- Given an unresolved star, absent compiled SQL or dialect failure, then return an explicit unavailable/unsupported reason, never a guessed donor.
- Given two manifest nodes mapping to the same quoted relation, then return ambiguous mapping.
- Given an existing explicit description, then normal inheritance preserves it; configured force/override semantics must match the existing pipeline in integration tests.
- Given a donor docs block and a renamed target, then unrendered-description mode copies the donor reference unchanged, not rendered prose or target-name metadata.
- Given changes to schema, relation mapping, dialect or manifest generation with identical SQL, then no cached donor from the earlier context is reused.
- Given dry-run or repeated refactor, then YAML bytes/mtime and idempotence follow the existing tested contracts.

## Open decisions before an executable runtime ticket
1. Fallback policy: legacy name matching after semantic rejection can undermine the no-guessed-donor promise. Recommend separating unavailable-lineage fallback from semantic rejection; approve exact behavior before wiring into inheritance.
2. Runtime scope: start with DuckDB demonstrated here; Trino requires real compiled fixtures, catalog/schema quoting and adapter mapping tests. Do not silently map unsupported adapters to generic SQL.
3. Resource bounds: benchmark parser/node growth, then choose input/node/time limits and real cancellation behavior. A thread timeout does not stop parser CPU. Do not promise a two-second timeout without cancellation evidence.
4. Reuse existing run-scoped ownership/caches; any cache key must cover SQL, schema, dialect, node mapping and generation. Bound size; cache identities only, never mutable description payloads.

## Exclusions
No transformed/aggregate inheritance, multi-source text consensus, external service or MCP dependency, Rust/Fusion compatibility claim, automatic compilation, general lineage engine, new CLI/config flags or production timeout implementation in this investigation.
