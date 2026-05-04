---
id: critique:c10docs09-docs-ci-hardening
kind: critique
status: final
created_at: 2026-05-04T03:07:28Z
updated_at: 2026-05-04T03:07:28Z
scope:
  kind: repository
  repositories:
    - repo:root
review_target: "ticket:c10docs09 docs build CI diff"
links:
  tickets:
    - ticket:c10docs09
  packets:
    - packet:ralph-ticket-c10docs09-20260504T025259Z
  evidence:
    - evidence:c10docs09-local-docs-ci-validation
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/360
---

# Summary

Mandatory high-risk release-packaging critique for the `ticket:c10docs09` docs build CI hardening.

# Review Target

Reviewed the current uncommitted diff for docs config/dependencies, the `Tests` workflow docs job, Dependabot docs npm coverage, and the Loom ticket/evidence/packet reconciliation.

# Verdict

`pass-for-commit-push-trial`.

No blocking implementation defect was found. Do not accept or close `ticket:c10docs09` until GitHub Actions proves the Node 18/24 docs job and final evidence is persisted.

# Findings

## Low - Unscoped Lockfile Drift

Confidence: high.

`docs/package-lock.json` changes the peer-only `typescript` lock entry from `5.7.2` to `6.0.3`, even though the intended direct dependency change was React/React-DOM alignment.

This is not a blocker because TypeScript is peer-only, satisfies declared peer ranges, and local `npm --prefix docs ls` plus docs build evidence passed. Parent inspected `npm --prefix docs explain typescript`; npm selected TypeScript as an optional peer through Docusaurus' webpack/cosmiconfig dependency chain. The ticket should acknowledge this drift as a documented residual unless a later package cleanup ticket chooses to pin or otherwise constrain it.

## Gate - Node 18/24 Not Yet Observed

Confidence: high.

`.github/workflows/tests.yml` correctly encodes Node `18` and `24` for the docs job. Local evidence used Node `22.22.1`, and the ticket/evidence honestly mark Node 18/24 GitHub Actions evidence as pending.

This blocks acceptance, not commit/push.

# Evidence Reviewed

- Current worktree diff for `docs/docusaurus.config.js`, `docs/package.json`, `docs/package-lock.json`, `.github/workflows/tests.yml`, `.github/dependabot.yml`, ticket, packet, and evidence records.
- `docs/docusaurus.config.js` now uses `require(...)`, `require.resolve(...)`, and `module.exports` consistently.
- `docs/package.json` aligns React and React-DOM to `^18.3.1` while keeping Docusaurus packages at `3.7.0`.
- `.github/workflows/tests.yml` includes a docs job under the existing push/pull_request/schedule workflow with Node `18` and `24`, `npm --prefix docs ci`, `npm --prefix docs ls`, and `npm --prefix docs run build`.
- `.github/dependabot.yml` includes npm `/docs` coverage.
- `ticket:c10docs09`, `packet:ralph-ticket-c10docs09-20260504T025259Z`, and `evidence:c10docs09-local-docs-ci-validation`.
- Reviewer `git diff --check` - passed.
- Parent `npm --prefix docs explain typescript` - TypeScript is peer-only via Docusaurus dependency chain.

# Residual Risks

- GitHub Actions behavior with `actions/setup-node@v4.0.4` resolving Node 24 is unproven until CI runs.
- Existing 30 npm audit findings remain out of scope for this ticket.
- The TypeScript peer-only lockfile drift is documented but not minimized.

# Required Follow-up

- Commit and push to trigger CI.
- Verify the docs job passes on Node 18 and Node 24.
- Persist final GitHub Actions evidence before ticket acceptance.

# Acceptance Recommendation

Proceed to commit/push. Do not accept or close `ticket:c10docs09` until Node 18/24 CI passes and final evidence is reconciled.
