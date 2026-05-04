---
id: ticket:c10feed26
kind: ticket
status: closed
change_class: security-sensitive
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T23:01:41Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
    - evidence:c10feed26-workbench-feed-hardening-validation
    - evidence:c10feed26-main-ci-success
  critique:
    - critique:c10feed26-workbench-feed-hardening-review
    - critique:c10feed26-workbench-feed-hardening-final-review
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/377
depends_on: []
---

# Summary

Remove, sandbox, or harden the workbench Hacker News RSS widget so app startup is not network-dependent and external feed content is not injected as raw HTML.

# Context

`src/dbt_osmosis/workbench/app.py:348-364` fetches `https://news.ycombinator.com/rss` during app initialization and interpolates feed fields into HTML. `src/dbt_osmosis/workbench/components/feed.py:41-42` renders that HTML with `extras.InnerHTML`.

# Why Now

Workbench should be a dbt development tool, not a network-dependent dashboard with avoidable XSS and startup failure risk. Optional UI extras should not weaken trust boundaries while compatibility work is underway.

# Scope

- Decide whether the RSS widget should remain, become opt-in, or be removed.
- If retained, add timeout, caching, failure tolerance, and a user-visible disable option.
- Escape/sanitize feed fields before rendering.
- Ensure basic workbench import/start tests do not perform outbound network calls.
- Add tests for feedparser failure and malicious feed content.

# Out Of Scope

- Redesigning the whole workbench dashboard.
- Adding new external content widgets.

# Acceptance Criteria

- ACC-001: Workbench startup does not require a successful outbound RSS request.
- ACC-002: Feed content is escaped or sanitized before rendering.
- ACC-003: Users can disable the external feed or it is removed.
- ACC-004: Feed fetches are timeout-bound and failure-tolerant if retained.
- ACC-005: Tests prove a feedparser failure does not break app startup.
- ACC-006: Tests prove script/HTML content from a feed entry is not injected raw.

# Coverage

Covers:

- ticket:c10feed26#ACC-001
- ticket:c10feed26#ACC-002
- ticket:c10feed26#ACC-003
- ticket:c10feed26#ACC-004
- ticket:c10feed26#ACC-005
- ticket:c10feed26#ACC-006
- initiative:dbt-110-111-hardening#OBJ-006

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10feed26#ACC-001 | evidence:c10feed26-workbench-feed-hardening-validation; evidence:c10feed26-main-ci-success | critique:c10feed26-workbench-feed-hardening-final-review | supported |
| ticket:c10feed26#ACC-002 | evidence:c10feed26-workbench-feed-hardening-validation; evidence:c10feed26-main-ci-success | critique:c10feed26-workbench-feed-hardening-final-review | supported |
| ticket:c10feed26#ACC-003 | evidence:c10feed26-workbench-feed-hardening-validation; evidence:c10feed26-main-ci-success | critique:c10feed26-workbench-feed-hardening-final-review | supported |
| ticket:c10feed26#ACC-004 | evidence:c10feed26-workbench-feed-hardening-validation; evidence:c10feed26-main-ci-success | critique:c10feed26-workbench-feed-hardening-final-review; critique:c10feed26-workbench-feed-hardening-review#FIND-001 resolved; critique:c10feed26-workbench-feed-hardening-review#FIND-002 resolved | supported |
| ticket:c10feed26#ACC-005 | evidence:c10feed26-workbench-feed-hardening-validation; evidence:c10feed26-main-ci-success | critique:c10feed26-workbench-feed-hardening-final-review | supported |
| ticket:c10feed26#ACC-006 | evidence:c10feed26-workbench-feed-hardening-validation; evidence:c10feed26-main-ci-success | critique:c10feed26-workbench-feed-hardening-final-review | supported |

# Execution Notes

The smallest safe fix may be removing the feed widget. If keeping it, prefer escaping and a short timeout over complex sanitization libraries unless already present.

# Blockers

Potential human/product decision if removing the widget changes desired workbench polish.

# Evidence

Existing evidence: evidence:oracle-backlog-scan, evidence:c10feed26-workbench-feed-hardening-validation, and evidence:c10feed26-main-ci-success.

Evidence status: local red/green, parent validation, full pre-commit, final critique, and green remote CI support ACC-001 through ACC-006 for commit `1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c`. Missing evidence: none for this ticket's acceptance scope.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: External HTML/network behavior is security-sensitive even if the app is local.

Required critique profiles: security, code-change, test-coverage

Findings:

- critique:c10feed26-workbench-feed-hardening-review#FIND-001: resolved; verified by critique:c10feed26-workbench-feed-hardening-final-review.
- critique:c10feed26-workbench-feed-hardening-review#FIND-002: resolved; verified by critique:c10feed26-workbench-feed-hardening-final-review.

Disposition status: completed.

Review: critique:c10feed26-workbench-feed-hardening-final-review

Acceptance recommendation: no-critique-blockers.

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: not_required

Promoted: None.

Deferred / not-required rationale: The durable user-facing behavior is documented in `docs/docs/reference/cli.md`, covered by tests, and captured in ticket/evidence/critique records. No separate reusable wiki, research, spec, plan, initiative, or constitution update is needed for this bounded workbench hardening change.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: OpenCode parent acceptance gate.
Accepted at: 2026-05-04T23:01:41Z.
Basis: Implementation commit `1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c`, evidence:c10feed26-workbench-feed-hardening-validation, critique:c10feed26-workbench-feed-hardening-review, critique:c10feed26-workbench-feed-hardening-final-review, and evidence:c10feed26-main-ci-success.
Residual risks: No browser-level Streamlit rendering test was run, no real Hacker News RSS fetch was exercised, enabled fetch timeout is socket-operation bounded, and Streamlit component rendering behavior may change later.

# Dependencies

Coordinate with ticket:c10wb22 if workbench tests are added there.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench oracle finding.
- 2026-05-04T22:16:07Z: Started Ralph iteration 01 to make the workbench RSS feed disabled by default, timeout-bound when enabled, failure-tolerant, and escaped before rendering.
- 2026-05-04T22:22:20Z: Ralph iteration 01 returned stop. Parent accepted the implementation iteration after focused tests, Ruff, and basedpyright zero-error validation; moved ticket to review_required for required security/code-change/test-coverage critique.
- 2026-05-04T22:27:08Z: Required critique returned changes_required with one medium malformed-URL failure-tolerance finding and one low uncapped-response-size finding; moved ticket back to active for follow-up implementation.
- 2026-05-04T22:27:45Z: Started Ralph iteration 02 to resolve critique findings by failing closed on malformed feed URLs/entry rendering and capping RSS response reads.
- 2026-05-04T22:31:06Z: Ralph iteration 02 returned stop. Parent accepted the follow-up implementation after focused tests, Ruff, and basedpyright zero-error validation; moved ticket back to review_required for final mandatory critique verification.
- 2026-05-04T22:34:20Z: Final mandatory critique passed with no new findings and verified prior findings resolved; moved ticket to complete_pending_acceptance pending full pre-commit, commit, push, and remote CI evidence.
- 2026-05-04T22:35:45Z: Full pre-commit and focused workbench/CLI tests passed after formatting; ticket remains complete_pending_acceptance pending commit, push, and remote CI evidence.
- 2026-05-04T23:01:41Z: Commit `1d3a0cc6bb1f4ec1255b632e97057af4f8808d7c` reached green Labeler, lint, Tests, and Release workflows on `origin/main`; accepted residual risks and closed ticket.
