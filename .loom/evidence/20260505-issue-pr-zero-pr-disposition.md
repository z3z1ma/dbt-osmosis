---
id: evidence:issue-pr-zero-pr-disposition
kind: evidence
status: recorded
created_at: 2026-05-05T08:35:41Z
updated_at: 2026-05-05T08:35:41Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:issue-pr-zero
  tickets:
    - ticket:gh346white
external_refs:
  github_pulls: https://github.com/z3z1ma/dbt-osmosis/pulls
---

# Summary

Observed PR-zero disposition after issue-ticket closures and dependency PR handling.

# Procedure

Observed at: 2026-05-05T08:35:41Z
Source state: local branch `loom/dbt-110-111-hardening` after issue commits through `6578254`; `origin/main` had advanced through merged PRs.
Procedure: Used `gh pr list`, `gh pr view`, `gh pr merge`, and `gh pr close` to merge green Dependabot PRs and close failing, conflicting, superseded, or too-broad PRs with comments.
Expected result when applicable: Every open PR has a durable disposition: merged, closed with rationale, or converted into follow-up Loom work.
Actual observed result: `gh pr list --state open --limit 20` returned no open PRs after disposition.
Procedure verdict / exit code: Pass for PR-zero disposition. Main CI still had queued/in-progress runs after the final PR merges.

# Artifacts

Merged PRs:

- #391: docs npm/yarn grouped update, green and mergeable.
- #390: uv dependency group update, green and mergeable.
- #389: `@docusaurus/types` update, green and mergeable.
- #384: `@mdx-js/react` update, green and mergeable.

Closed PRs:

- #388: React 19 docs update, closed because docs builds failed on Node 18 and Node 24.
- #386: Docusaurus preset update, closed because docs builds failed on Node 18 and Node 24.
- #385: Docusaurus module type aliases update, closed as superseded/conflicting after merged docs dependency updates.
- #383: OpenAI range update, closed because uv lockfile checks failed.
- #351: pytest-cov range update, closed because uv lockfile checks failed.
- #350: Ruff upgrade, closed because uv lockfile and lint checks failed.
- #347: Hatchling update, closed because latest dbt-core compatibility validation failed.
- #346: external whitespace/fold-point/rename enrichment PR, closed because it was conflicting, failing CI, and too broad; residual whitespace/fold-idempotency claims were converted into ticket:gh346white.
- #344: older uv group update, closed as superseded by #390 and conflicting/failing compatibility.

Post-disposition command:

```bash
gh pr list --state open --limit 20
```

Observed result: no output, indicating no open PRs.

# Supports Claims

- initiative:issue-pr-zero#OBJ-003: every open PR was merged, closed, superseded, or converted to follow-up work with rationale.
- initiative:issue-pr-zero#OBJ-005: the meaningful residual behavior from PR #346 is tracked as ticket:gh346white rather than lost in a closed PR.

# Challenges Claims

- initiative:issue-pr-zero#OBJ-004 remains pending: main CI had queued/in-progress runs after the final merges.

# Environment

CLI: GitHub CLI `gh` in repository workspace.
Branch: `loom/dbt-110-111-hardening`.
Remote: `origin`.
External service / harness / data source when applicable: GitHub PR list and PR status/check metadata.

# Validity

Valid for: GitHub PR state observed at 2026-05-05T08:35:41Z.
Fresh enough for: initiative PR disposition and issue-pr-zero continuation.
Recheck when: Dependabot or maintainers open new PRs, or queued main CI completes.
Invalidated by: new open PRs or failed post-merge CI.
Supersedes / superseded by: N/A.

# Limitations

This evidence records PR disposition, not final main CI health. It does not claim dependency release readiness.

# Result

PR-zero disposition is achieved at the GitHub PR list level; initiative closure still depends on final main CI assessment.

# Interpretation

The initiative can treat PR triage/disposition as complete while keeping CI follow-through open until current main checks finish.

# Related Records

- initiative:issue-pr-zero
- ticket:gh346white
