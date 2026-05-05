---
id: ticket:c10anchors30
kind: ticket
status: complete_pending_acceptance
change_class: code-behavior
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-05T04:11:56Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10anchors30-yaml-anchor-validation
  critique:
    - critique:c10anchors30-yaml-anchor-review
  packets:
    - packet:ralph-ticket-c10anchors30-20260505T015001Z
    - packet:ralph-ticket-c10anchors30-20260505T020517Z
    - packet:ralph-ticket-c10anchors30-20260505T021703Z
    - packet:ralph-ticket-c10anchors30-20260505T022457Z
    - packet:ralph-ticket-c10anchors30-20260505T023255Z
    - packet:ralph-ticket-c10anchors30-20260505T024405Z
depends_on: []
---

# Summary

Decide and test whether schema reader/writer preservation supports YAML anchors and aliases that cross managed and unmanaged sections.

# Context

`src/dbt_osmosis/core/schema/parser.py`, `reader.py`, and `writer.py` parse managed and preserved sections separately, then merge preserved keys back. The audit flagged that anchors defined in unmanaged sections and referenced in managed model/source sections may duplicate anchors, move definitions, or produce undefined aliases.

# Why Now

YAML preservation is a core promise. Anchors/aliases are less common, but if users rely on them for tests or metadata, schema mutation must either preserve them or fail/document unsupported patterns.

# Scope

- Determine whether cross-section anchors/aliases are intended to be supported.
- Add tests with anchors defined in unmanaged sections and aliases used in managed sections.
- If supported, fix parser/reader/writer handling to round-trip without duplicate/undefined anchors.
- If unsupported, document and validate/fail clearly before destructive writes.

# Out Of Scope

- Supporting every advanced YAML feature if the chosen policy is explicit non-support.
- Replacing ruamel.yaml.

# Acceptance Criteria

- ACC-001: The project has an explicit supported/unsupported policy for cross-section YAML anchors and aliases.
- ACC-002: A schema file using `x-common-tests: &common` and `data_tests: *common` is covered by a regression test.
- ACC-003: If supported, write/read round-trip keeps aliases valid and parseable by ruamel/dbt.
- ACC-004: If unsupported, commands fail clearly before writing and docs/errors tell users what to change.
- ACC-005: Preserved unmanaged sections remain preserved under the chosen policy.

# Coverage

Covers:

- ticket:c10anchors30#ACC-001
- ticket:c10anchors30#ACC-002
- ticket:c10anchors30#ACC-003
- ticket:c10anchors30#ACC-004
- ticket:c10anchors30#ACC-005
- initiative:dbt-110-111-hardening#OBJ-003

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10anchors30#ACC-001 | evidence:c10anchors30-yaml-anchor-validation | critique:c10anchors30-yaml-anchor-review | covered |
| ticket:c10anchors30#ACC-002 | evidence:c10anchors30-yaml-anchor-validation | critique:c10anchors30-yaml-anchor-review | covered |
| ticket:c10anchors30#ACC-003 | evidence:c10anchors30-yaml-anchor-validation | critique:c10anchors30-yaml-anchor-review | covered |
| ticket:c10anchors30#ACC-004 | N/A - supported policy chosen | critique:c10anchors30-yaml-anchor-review | not_applicable |
| ticket:c10anchors30#ACC-005 | evidence:c10anchors30-yaml-anchor-validation | critique:c10anchors30-yaml-anchor-review | covered_with_accepted_risk |

# Execution Notes

Start with characterization tests. Do not assume support until a real ruamel/dbt round trip proves it.

# Blockers

None. Characterization showed current writes stayed parseable but flattened cross-section aliases, so the chosen policy is support valid cross-section alias preservation in normal read/write flows. If dbt-osmosis mutates an alias-bearing managed subtree in place, YAML shared-node semantics mean the unmanaged anchor object changes too; this is accepted risk because the case is unlikely, visible in git, and users can self-correct.

# Evidence

Existing evidence: evidence:oracle-backlog-scan; evidence:c10anchors30-yaml-anchor-validation. Missing evidence: remote CI for the eventual commit.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: YAML preservation edge case can cause data loss or invalid files.

Required critique profiles: code-change, data-preservation, test-coverage

Findings:

- critique:c10anchors30-yaml-anchor-review#FIND-001 resolved_by_parent: stale ticket fields were reconciled in this update.
- critique:c10anchors30-yaml-anchor-review#FIND-002 accepted_risk: quoted top-level managed keys can remain quoted; this edge-case formatting inconsistency is not data loss and preserving top-level key objects keeps the anchor-preserving merge simple.
- critique:c10anchors30-yaml-anchor-review#FIND-003 accepted_risk: dbt parse compatibility is recorded as evidence from a temporary dbt 1.10.20 / duckdb 1.10.0 fixture rather than automated in pytest.

Additional accepted risk: mutating an alias-bearing managed subtree in place can mutate the shared unmanaged anchor object because the implementation intentionally preserves YAML shared-node semantics. The operator accepted this risk as unlikely, visible in git, easy to self-correct, and preferable to extra copy-on-write complexity.

Disposition status: completed

Deferral / not-required rationale: No critique follow-up is required before acceptance.

# Retrospective / Promotion Disposition

Disposition status: not_required

Promoted: None.

Deferred / not-required rationale: No durable explanation page is needed because the supported behavior is covered by focused tests and the remaining edge cases are accepted risks in the ticket/critique record.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Pending remote CI.
Accepted at: N/A.
Basis: Local implementation, red/green evidence, dbt parse fixture, and critique are complete. Final acceptance waits for commit packaging and remote CI.
Residual risks: Accepted shared-node mutation behavior for alias-bearing managed subtrees; quoted top-level managed key formatting inconsistency; dbt parse compatibility evidenced for dbt 1.10.20 / duckdb 1.10.0 but not automated across the full dbt matrix; YAML anchor behavior can be subtle across ruamel versions.

# Dependencies

Coordinate with ticket:c10loss16 and ticket:c10gen20 if writer behavior changes.

# Journal

- 2026-05-03T21:10:43Z: Created from core architecture oracle finding.
- 2026-05-05T01:50:01Z: Promoted through ready into active after parent characterization. Current writer output for `x-common-tests: &common_tests` referenced by managed `data_tests: *common_tests` remains parseable but expands the managed alias and writes a duplicate unmanaged sequence, so the chosen policy is support valid cross-section alias preservation. Compiled packet:ralph-ticket-c10anchors30-20260505T015001Z for a test-first implementation slice.
- 2026-05-05T01:57:34Z: Ralph iteration returned stop and parent verification passed: `uv run pytest tests/core/test_schema.py -q` reported 22 passed, Ruff format/check and `git diff --check` passed, and basedpyright reported zero errors. Recorded evidence:c10anchors30-yaml-anchor-validation and moved to review_required for recommended critique.
- 2026-05-05T02:05:17Z: Parent critique found two acceptance issues before durable critique reconciliation: dbt parse compatibility needed direct evidence, and the first fix preserved managed-section quote style even when `preserve_quotes` defaults to false. Parent observed a temporary dbt parse fixture with `x-common-tests: &common_tests` and managed `data_tests: *common_tests` exits 0 under dbt 1.10.20/duckdb 1.10.0. Reopened active and compiled packet:ralph-ticket-c10anchors30-20260505T020517Z to restore managed quote normalization while preserving unmanaged quote style and anchor aliases.
- 2026-05-05T02:11:26Z: Ralph iter-02 returned stop and parent verification passed: `uv run pytest tests/core/test_schema.py -q` reported 23 passed, Ruff format/check and `git diff --check` passed, basedpyright reported zero errors, and a temporary dbt 1.10.20/duckdb 1.10.0 parse fixture for the anchor shape exited 0. Updated evidence:c10anchors30-yaml-anchor-validation and moved back to review_required for final recommended critique.
- 2026-05-05T02:17:03Z: Final critique found a blocking regression in iter-02: managed quote normalization converted all ruamel `ScalarString` values, including folded/literal block scalars, to plain strings. Reopened active and compiled packet:ralph-ticket-c10anchors30-20260505T021703Z to narrow normalization to quoted scalar styles and add block-scalar coverage.
- 2026-05-05T02:20:07Z: Ralph iter-03 returned stop and parent verification passed: schema tests reported 24 passed, Ruff format/check and `git diff --check` passed, basedpyright reported zero errors, and the temporary dbt parse fixture exited 0. Updated evidence:c10anchors30-yaml-anchor-validation and moved back to review_required for final recommended critique.
- 2026-05-05T02:24:57Z: Final critique found a high-severity preservation regression in iter-03: normalizing quoted `CommentedSeq` items via delete/insert can drop inline comments. Reopened active and compiled packet:ralph-ticket-c10anchors30-20260505T022457Z to preserve sequence item comments while normalizing quotes.
- 2026-05-05T02:28:36Z: Ralph iter-04 returned stop and parent verification passed: schema tests reported 25 passed, Ruff format/check and `git diff --check` passed, basedpyright reported zero errors, and the temporary dbt parse fixture exited 0. Updated evidence:c10anchors30-yaml-anchor-validation and moved back to review_required for final recommended critique.
- 2026-05-05T02:32:55Z: Final critique found a high-severity boolean-like scalar regression: `PlainScalarString` normalization can emit managed values such as `on` or `yes` unquoted, bypassing the existing YAML boolean-safety representer. Reopened active and compiled packet:ralph-ticket-c10anchors30-20260505T023255Z.
- 2026-05-05T02:37:42Z: Ralph iter-05 returned stop and parent verification passed: schema tests reported 26 passed, Ruff format/check and `git diff --check` passed, basedpyright reported zero errors, and the temporary dbt parse fixture exited 0. Updated evidence:c10anchors30-yaml-anchor-validation and moved back to review_required for final recommended critique.
- 2026-05-05T02:44:05Z: Final critique passed the anchor/alias design with findings and identified a remaining medium formatter-contract regression: most managed quoted scalars became `PlainScalarString`, bypassing long-string folding and multiline literal formatting. Reopened active and compiled packet:ralph-ticket-c10anchors30-20260505T024405Z to route long/multiline managed scalars through the normal string representer.
- 2026-05-05T02:48:30Z: Ralph iter-06 returned stop and parent verification passed: schema tests reported 28 passed, Ruff format/check and `git diff --check` passed, basedpyright reported zero errors, and the temporary dbt parse fixture exited 0. Updated evidence:c10anchors30-yaml-anchor-validation and moved back to review_required for final recommended critique.
- 2026-05-05T04:05:03Z: Operator accepted the residual shared-node mutation behavior as risk rather than adding guards or copy-on-write behavior. Rationale: alias-bearing managed mutations are extremely unlikely, git makes the change observable, and KISS is preferred for this edge case.
- 2026-05-05T04:11:56Z: Recorded final critique:c10anchors30-yaml-anchor-review with pass_with_findings. Dispositioned low findings and accepted risks, marked retrospective/promotion not required, and moved to complete_pending_acceptance pending packaging/remote CI.
