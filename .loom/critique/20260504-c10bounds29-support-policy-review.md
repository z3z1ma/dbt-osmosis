---
id: critique:c10bounds29-support-policy-review
kind: critique
status: final
created_at: 2026-05-04T23:43:46Z
updated_at: 2026-05-04T23:43:46Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10bounds29 keep-open-plus-canary support policy implementation diff"
links:
  ticket:
    - ticket:c10bounds29
  evidence:
    - evidence:c10bounds29-support-policy-validation
  packet:
    - packet:critique-ticket-c10bounds29-20260504T233947Z
    - packet:ralph-ticket-c10bounds29-20260504T233210Z
external_refs: {}
---

# Summary

Mandatory critique reviewed the `ticket:c10bounds29` implementation that keeps `dbt-core>=1.8` open, adds latest-dbt canary CI, surfaces dbt/dbt-osmosis deprecations, and updates support-policy docs/tests.

# Review Target

Review target: working tree diff from `0af38b2b7c2db6b448418525d0dbfdc3068e37bb` for `ticket:c10bounds29`, including `pyproject.toml`, `.github/workflows/tests.yml`, `tests/test_package_metadata.py`, `README.md`, `docs/docs/intro.md`, `docs/docs/tutorial-basics/installation.md`, the Ralph packet, ticket reconciliation, and `evidence:c10bounds29-support-policy-validation`.

Required profiles: release-packaging, operator-clarity, dbt-compatibility.

# Verdict

`pass_with_findings`

The implementation satisfies the ticket scope and no pre-acceptance code changes are required. Two low-severity risks remain for ticket-owned disposition before closure.

# Findings

## FIND-001: Future canary may not select an unreviewed future dbt minor

Severity: low
Confidence: medium
State: open

Observation: `.github/workflows/tests.yml:332` installs unpinned `dbt-core dbt-duckdb`, but `.github/workflows/tests.yml:346-354` only asserts installed `dbt-core >=1.8`. `tests/test_package_metadata.py:71-81` structurally checks that the canary is unpinned and meaningful, but it does not prove the resolver selected a future/latest dbt minor once one exists.

Why it matters: If adapter constraints backtrack to audited 1.11.x, the canary could pass without exercising the unreviewed future minor. That weakens the canary as proof of future-minor coverage, though the ticket intentionally treats it as a visibility signal rather than audited support.

Follow-up: Ticket should either accept this as a canary limitation or create follow-up work for a stronger latest-version detection/assertion once the project wants stricter canary semantics.

Challenges:

- ticket:c10bounds29#ACC-003: partial challenge only. The canary exists and is unpinned, but this finding limits how strongly it proves future-minor exercise.

## FIND-002: Docs could imply every Python/dbt support combination is audited

Severity: low
Confidence: medium
State: open

Observation: `README.md:32-36` says Python 3.10-3.13 and audited dbt Core 1.8.x through 1.11.x support, while `.github/workflows/tests.yml:54-69` excludes Python 3.13 for dbt Core 1.8 and 1.9. The same support-policy wording appears in `docs/docs/intro.md` and `docs/docs/tutorial-basics/installation.md`.

Why it matters: A user could read the support text as every Python/dbt combination being audited. The workflow matrix is more nuanced because older dbt/adapter pairings are not exercised on Python 3.13.

Follow-up: Ticket should either accept the low ambiguity risk or clarify docs in a follow-up if users need matrix-combination precision.

Challenges:

- ticket:c10bounds29#ACC-002: partial challenge only. Docs state the support policy, but combination-level matrix nuance is not explicit.

# Evidence Reviewed

- `packet:critique-ticket-c10bounds29-20260504T233947Z`.
- `ticket:c10bounds29`.
- `packet:ralph-ticket-c10bounds29-20260504T233210Z`.
- `evidence:c10bounds29-support-policy-validation`.
- `research:dbt-110-111-api-surfaces` context summarized in the packet.
- Working tree status, base commit, tracked diff, and changed files.
- `pyproject.toml`, `.github/workflows/tests.yml`, `tests/test_package_metadata.py`, `README.md`, `docs/docs/intro.md`, and `docs/docs/tutorial-basics/installation.md`.
- `git diff --check` on touched implementation files returned no output.

# Residual Risks

- New GitHub Actions canary has not actually run remotely.
- Non-blocking scheduled/manual canary failures may be easy to miss without monitoring.
- Warning policy surfaces dbt/dbt-osmosis deprecations but does not fail builds on them.
- Structural tests are policy guards, not proof of future dbt compatibility.

# Required Follow-up

No pre-acceptance code changes are required. The ticket should record a disposition for `critique:c10bounds29-support-policy-review#FIND-001` and `critique:c10bounds29-support-policy-review#FIND-002` before closure.

# Acceptance Recommendation

`risk-disposition-needed`

The reviewer described this as accept-with-low-findings. Ticket-owned low-risk dispositions are required before acceptance can treat the mandatory critique gate as resolved.
