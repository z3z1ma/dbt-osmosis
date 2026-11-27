Status: open
Created: 2026-11-27
Updated: 2026-11-27

# Investigate DuckDB column case comparison

## Scope
Bounded follow-up investigation of ASCII case differences between physical DuckDB catalog column names and existing YAML column spellings. Do not treat this record as an executable runtime contract.

## Non-goals
No global lowercasing, non-ASCII equivalence, other-adapter change, automatic spelling rewrite, warehouse or upstream SQL compiler repair. Preserve literal punctuation under duckdb-literal-column-identifiers.md.

## Acceptance Criteria
1. Reconfirm actual CLI duplicate/stale output with an original local physical table and ordinary unquoted/quoted ASCII spelling controls.
2. Establish a comparison-only contract distinct from output spelling and explicit output-to-upper/lower settings; investigate add/remove/sort/sync/inheritance/diff boundaries and non-ASCII controls before activation.
3. Preserve user docs, quote flags, literal punctuation, raw ignore patterns and deterministic missing behavior. Parent activation and independent review precede any runtime change.

## References
.10x/tickets/2026-11-27-owned-importer-boundary.md; .10x/specs/duckdb-literal-column-identifiers.md; https://duckdb.org/docs/2.0/sql/dialect/keywords_and_identifiers .

## Assumptions
Current docs describe ASCII-insensitive identifier comparison with preserved spelling. Existing literal-name contract explicitly leaves ordinary case behavior unchanged. No broader normalization default is ratified by the relation-donor correction.

## Journal
- 2026-11-27: `/tmp/osmosis-case-boundary.py`, Core1.12.7/dbt-duckdb1.11.1/adapters1.24.5/DuckDB2.0.0, unchanged genuine #487 catalog and temporary original project inputs. Existing YAML uppercase variants of id/[id]/backtick-id/[[edge]]/literal-doublequote-id produce both physical lowercase and original uppercase entries after actual document, exit0. Original descriptions/quote/meta stay on uppercase entries; those retain STALE types. Second run is byte-stable but remains semantically duplicated. Explicit output-to-upper prevents duplicates yet retains STALE types. Skip-add-columns preserves uppercase entries. Evidence root `/var/folders/m0/1lt4df1s78xgj4s6yk5c9wp00000gp/T/osmosis-case-boundary-fqdkfz6y`; dry-run bytes/mtime unchanged, consumer DB absent.

## Blockers
Comparison versus output-spelling semantics and preservation matrix require separate shaping. No runtime edit in current catalog relation-donor iteration.

## Evidence
Caller-visible observations above; no proof of all SQL/YAML case semantics or non-ASCII behavior.

## Review
Not reviewed for implementation.

## Retrospective
A literal-preserving normalization function is not automatically the right comparison key. Do not solve this by lowercasing the output and regressing the prior physical-name preservation fix.
