---
id: initiative:issue-pr-zero
kind: initiative
status: active
created_at: 2026-05-05T06:02:19Z
updated_at: 2026-05-05T06:02:19Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:gh333meta
    - ticket:gh329loss
    - ticket:gh328scaf
    - ticket:gh326skip
    - ticket:gh311wrap
    - ticket:gh305wb
external_refs:
  github_issues: https://github.com/z3z1ma/dbt-osmosis/issues
  github_pulls: https://github.com/z3z1ma/dbt-osmosis/pulls
---

# Objective

Bring the repository's open GitHub issue and pull-request backlog down to a truthful zero-state: every open item is either fixed and closed with evidence, linked to a Loom execution ticket, waiting on explicit user/reporter clarification, or intentionally closed as superseded/not actionable.

# Why Now

The dbt 1.10/1.11 hardening initiative removed a large compatibility backlog and left a small set of user-reported GitHub issues plus Dependabot and external contributor PRs. Triage should happen now while the YAML preservation, config-resolution, workbench, and packaging context is fresh.

# In Scope

- Open GitHub issues present on 2026-05-05.
- Open GitHub pull requests present on 2026-05-05, including Dependabot and external contributor PRs.
- Loom tickets, comments, follow-up evidence, and closure records needed to make each external item's disposition recoverable.
- Small implementation tickets that unblock closing validated GitHub issues.
- PR disposition work: merge, close, supersede, rebase, or record a blocker with rationale.

# Out Of Scope

- Closing issues or PRs without evidence or a linked rationale.
- Publishing releases, cutting tags, or changing package support policy without an explicit human decision.
- Solving every ticket inside the initial triage pass.
- Treating Dependabot PR volume as solved by silently ignoring dependency updates.

# Delegated Authority / Autonomy Boundaries

Agents may triage open GitHub issues and PRs, create bounded Loom tickets, comment with Loom references, and implement individual tickets when the scope preserves public behavior and existing support promises.

Human decision triggers:

- Closing a user-reported issue as not planned when the behavior is plausibly still real.
- Merging an external contributor PR with broad or conflicting code changes.
- Adding, dropping, or narrowing supported extras, Python versions, dbt versions, or adapter promises.
- Publishing releases or tags.

# Objective-Level Stop Conditions

Stop and return to planning or the user if an issue implies silent data loss beyond the bounded ticket, if an open PR requires maintainer intent that is not inferable from the codebase, if dependency updates create incompatible support-policy changes, or if a GitHub item contains sensitive information that should not be copied into Loom.

# Success Metrics

- OBJ-001: Every currently open GitHub issue has a durable disposition: closed with evidence, linked to a Loom ticket, or awaiting explicit reporter/user clarification.
- OBJ-002: Every validated issue has a bounded ticket with external refs, acceptance criteria, evidence expectations, and critique policy.
- OBJ-003: Every currently open PR is merged, closed, superseded, or blocked with a recorded rationale and next move.
- OBJ-004: Main branch CI remains green after implementation and closure pushes related to this initiative.
- OBJ-005: Known user-impacting data-loss, installation, config-resolution, and YAML formatting risks from the open backlog are either fixed or tracked as explicit tickets.

# Milestones

- M1: Triage all open issues and create or close records for each.
- M2: Triage all open PRs and decide merge/close/supersede/block routes.
- M3: Implement highest-risk validated issue tickets first, especially data-loss and install blockers.
- M4: Close or reconcile external GitHub issues and PRs after evidence-backed acceptance.

# Dependencies

- GitHub issue and PR state from `gh`.
- Existing dbt 1.10/1.11 hardening records and CI evidence.
- Maintainer judgment for broad external PRs and support-policy changes.

# Risks

- A user issue may describe an older release behavior that is fixed on main but still lacks a release; comments must distinguish current-main truth from published package truth.
- Dependency PRs can interact with Python/dbt support policy and should not be bulk-merged without CI and lockfile review.
- External PRs may solve real problems but conflict with the smaller current-main architecture after hardening.

# Linked Work

Tickets:

- ticket:gh333meta - Add granular skip list for inherited column meta keys.
- ticket:gh329loss - Prevent refactor restructure from deleting schema files with unmanaged top-level content.
- ticket:gh328scaf - Honor `scaffold-empty-configs` from dbt project config during YAML sync.
- ticket:gh326skip - Add `skip-inherit-descriptions` option for column documentation inheritance.
- ticket:gh311wrap - Prevent trailing whitespace when wrapping nested column descriptions.
- ticket:gh305wb - Make workbench extra install and preflight ydata-profiling/IPython cleanly.

# Status Summary

Initial issue triage completed with `gh issue list`, `gh issue view`, code inspection, a focused YAML formatter reproduction, and Oracle review. Six ready tickets now own validated work. Open PRs have been listed but still need deeper per-PR disposition.

# Completion Basis

N/A - initiative is active.
