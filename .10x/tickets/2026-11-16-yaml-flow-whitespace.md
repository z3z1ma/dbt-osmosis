Status: active
Created: 2026-11-16
Updated: 2026-11-16

# Issue 450: YAML flow whitespace check loop

## Scope
Reproduce issue 450 on current main, then fix the smallest proven emitter/writer defect that creates trailing whitespace and repeated check changes, preserving YAML semantics.
User explicitly delegates implementation, review, draft publication and routine evidence-backed merge. This issue remains the external canonical request: https://github.com/z3z1ma/dbt-osmosis/issues/450 (open, observed 2026-11-16).

## Non-goals
Generic unchanged-section byte splicing, new preservation flags, column sorting, releases.

## Acceptance Criteria
- Record a before-fix reproduction of wrapped flow fixture trailing whitespace and the trim/rewrite loop.
- A minimal fix avoids generated trailing spaces while preserving scalar values, anchors/aliases, comments, and existing parser document/duplicate-key behavior.
- Regression coverage exercises both direct and buffered writer paths and a stable subsequent check.
- Applicable tests, lint and types pass; independent review and exact-head aggregate CI pass before merging.

## References
AGENTS.md; src/dbt_osmosis/core/AGENTS.md; historical .10x/tickets/done/2026-07-05-reduce-schema-writer-complexipy-hotspots.md. History must be verified against current code.

## Assumptions
Current base f6a57cb9 includes #456 (verified fetch). Preserve parsed meaning and existing formatting controls; user-ratified requirement. Narrow investigation may conclude no safe fix and document actionable limits.

## Journal
- Created isolated worktree from fetched main, leaving source checkout untouched.

## Blockers
None presently.

## Evidence
- Baseline ruamel 0.19.1 writes `caregiver_id: rostered, ` followed by a newline at width 60. `write_plain` inserts separation before deciding to wrap a long word. Restoring the stock RoundTripEmitter in memory makes all six direct/buffered regression cases fail (widths 40, 60, 100).
- Candidate buffers only plain-scalar emission and removes spaces immediately before emitted CR/LF. Quoted and block scalar content never enters this cleanup. The original stream and encoding are restored in `finally`; encoding occurs after plain-text cleanup. Atomic writing/cache behavior is untouched.
- Final focused `pytest tests/core/test_schema.py -q`: 48 passed. Regression coverage includes both writer paths, an unchanged dry-run after whitespace cleanup, parsed equality, anchor identity, comments, multiline/plain/quoted/literal/folded values, text and UTF-8/UTF-16LE streams, LF/CRLF/CR line breaks, emitter failure restoration, duplicate-key and multi-document rejection.
- Whole-repository Ruff 0.15.17 check passes; changed-file preview format check passes; `uv lock --check` and `git diff --check` pass.
- basedpyright 1.39.4, explicit existing venv interpreter: 38 files, 0 errors, 1990 warnings. Initial invocation without `--pythonpath` could not resolve installed dependencies and is not a source failure.
- Validation environment: Python 3.11.15, dbt-core 1.10.20, dbt-duckdb 1.10.0, ruamel.yaml 0.19.1. Full pytest is running; remote supported matrix/aggregate CI remains a merge gate.
- Generic section-splicing counterexample (parent investigation): preserving an unchanged `anchors: &original` section while changed models serialize `*renamed` produces an undefined alias, despite parsed equality of the anchor mapping. No generic preservation flag was introduced.

## Review
Independent reviewer verdict: pass. Reviewed emitter state, stream restoration, encoding, YAML semantics and tests. Differential comparison with stock RoundTripEmitter across 6,000 valid flow/block documents (widths 10/30/100, repeated spaces, newlines, Unicode/NEL/LS) changed 127 outputs with zero parsed-value differences (Python 3.11.15, ruamel 0.19.1). Minor later-word assertion corrected; reviewer also approved final CR-only delta. Full local results and exact-head remote CI will be recorded on the PR before merge.

## Retrospective
Plain-scalar emission is the narrow boundary where line-end spaces are YAML separation. Whole-document stripping would corrupt literal/folded/quoted values; parsed equality is insufficient to splice serialized sections with cross-section anchors.

Limitations: hand alignment still normalizes under ruamel. This does not promise all YAML has no trailing spaces: scalar content and comments remain intact. Existing ruamel behavior can collapse multiple internal spaces when a long following word forces a plain-scalar wrap; observed with `short  then_a_very_long_word_that_wraps_at_the_configured_width`. The same value loss occurs with the stock emitter before cleanup, so changing scalar formatting is out of this focused issue's scope; no new correction is claimed.
