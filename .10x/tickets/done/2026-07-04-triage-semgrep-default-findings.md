Status: done
Created: 2026-07-04
Updated: 2026-07-04
Parent: .10x/tickets/2026-07-04-quality-optimizer-hill-climb.md
Depends-On: .10x/tickets/done/2026-07-04-remediate-uv-audit-vulnerabilities.md

# Triage Semgrep Default Findings

## Scope

Triage the 33 Semgrep default findings from the quality baseline and split executable fixes from policy decisions.

In scope:

- Classify each Semgrep finding as true positive, false positive, policy-required, or blocked.
- For mechanical true positives, open or execute bounded child work as appropriate.
- For policy changes such as GitHub Action pinning or dependency cooldown windows, record the exact policy question before implementation.
- Preserve release workflow behavior unless the workflow security model is explicitly superseded.

Out of scope:

- Broad CI redesign.
- Adding a Semgrep baseline.
- Suppressing findings without narrow justification.
- Changing release authority or publishing behavior without a decision record.

## Acceptance Criteria

- ACC-001: All 33 baseline Semgrep findings are classified with rationale.
- ACC-002: Any implemented fix has targeted evidence and does not weaken existing CI/release behavior.
- ACC-003: Any unimplemented true positive has a durable owner or explicit no-action rationale.

## Evidence Expectations

- Semgrep JSON summary before and after any fixes.
- Workflow diffs and verification for any CI changes.
- Focused tests or static checks for Python-code findings.

## Progress and Notes

- 2026-07-04: Baseline Semgrep findings include Dependabot/uv cooldown policy, mutable GitHub Action tags, `workflow_run` checkout concerns, dynamic import, and dynamic urllib usage.
- 2026-07-04: Re-ran Semgrep through `uvx` into `/tmp/dbt-osmosis-ai-quality/semgrep-default-current.json`; current result remains 33 blocking findings.
- 2026-07-04: Classified all findings in `.10x/evidence/2026-07-04-semgrep-triage.md`.
- 2026-07-04: Opened blocked policy owners for dependency cooldowns and GitHub Actions pinning. No source/workflow changes were made in this ticket.

## Blockers

Policy findings may require user ratification before implementation.

## References

- `.10x/research/2026-07-04-quality-optimizer-baseline.md`
- `.10x/evidence/2026-07-04-quality-optimizer-baseline.md`
- `.10x/evidence/2026-07-04-semgrep-triage.md`
- `.10x/reviews/2026-07-04-semgrep-triage-review.md`
- `.10x/tickets/2026-07-04-ratify-dependency-cooldown-policy.md`
- `.10x/tickets/2026-07-04-ratify-github-actions-pinning-policy.md`
