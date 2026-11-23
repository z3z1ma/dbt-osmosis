Status: open
Created: 2026-11-23
Updated: 2026-11-23
Parent: .10x/tickets/2026-11-23-quoted-catalog-resolution.md

# PostgreSQL unquoted exact-first catalog precedence

## Scope
Investigate a separate reproduced PostgreSQL unquoted resolution error before changing exact-first precedence. The quoted no-exact correction does not own this policy.

## Non-goals
No immediate source change, live warehouse, new adapter certification or global folding redesign.

## Acceptance Criteria
Define adapter-specific unquoted canonical identity precedence with mixed quote policies and compatibility controls, then independently review any bounded fix and validate normal CI.

## References
.10x/specs/catalog-exact-relation-precedence.md; .10x/specs/postgres-quoted-catalog-resolution.md; https://www.postgresql.org/docs/19/sql-syntax-lexical.html.

## Assumptions
PostgreSQL unquoted identifiers fold lowercase. Existing raw exact pass precedes folded matching. The current scope preserves that pass; a separate explicit contract is required before changing it.

## Journal
- 2026-11-23: Offline script `/tmp/osmosis-unquoted-followup.py` with actual installed PostgresRelation (dbt-postgres1.11.0/Core1.12.7) constructed catalog entries raw `Orders` (wrong_mixed_case) and `orders` (correct_lowercase). Requested relation rendered `"db"."public".Orders` with identifier quote flag false. Public get_columns returned `wrong_mixed_case`; expected PostgreSQL SQL resolution targets lowercase `orders`. Output `/tmp/osmosis-unquoted-followup.log`. No database connection was made; catalog is constructed, not real producer evidence.

## Blockers
Behavioral contract and separate implementation scope not yet established.

## Evidence
Reproduced distinct wrong result, not fixed by no-exact quoted guard.

## Review
Pending separate work.

## Retrospective
Raw exact identity is not necessarily database-resolved identity when a requested component is unquoted. Keep producer identity and SQL reference semantics distinct.
