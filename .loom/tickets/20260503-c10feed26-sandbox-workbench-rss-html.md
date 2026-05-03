---
id: ticket:c10feed26
kind: ticket
status: ready
change_class: security-sensitive
risk_class: medium
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-03T21:10:43Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  evidence:
    - evidence:oracle-backlog-scan
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
| ticket:c10feed26#ACC-001 | evidence:oracle-backlog-scan | None | open |
| ticket:c10feed26#ACC-002 | evidence:oracle-backlog-scan | None | open |

# Execution Notes

The smallest safe fix may be removing the feed widget. If keeping it, prefer escaping and a short timeout over complex sanitization libraries unless already present.

# Blockers

Potential human/product decision if removing the widget changes desired workbench polish.

# Evidence

Existing evidence: evidence:oracle-backlog-scan. Missing evidence: tests for failure and sanitization.

# Critique Disposition

Risk class: medium

Critique policy: recommended

Policy rationale: External HTML/network behavior is security-sensitive even if the app is local.

Required critique profiles: security, code-change, test-coverage

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Not decided.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending tests and security review.
Residual risks: Streamlit component rendering behavior may change.

# Dependencies

Coordinate with ticket:c10wb22 if workbench tests are added there.

# Journal

- 2026-05-03T21:10:43Z: Created from CLI/SQL/workbench oracle finding.
