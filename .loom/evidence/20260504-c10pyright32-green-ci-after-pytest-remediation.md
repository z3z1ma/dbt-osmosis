---
id: evidence:c10pyright32-green-ci-after-pytest-remediation
kind: evidence
status: recorded
created_at: 2026-05-04T18:15:28Z
updated_at: 2026-05-04T18:15:28Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10pyright32
external_refs:
  github_actions:
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25334910408
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25334910369
    - https://github.com/z3z1ma/dbt-osmosis/actions/runs/25334910419
---

# Summary

Observed the pushed `origin/main` CI confirmation after the `--profiles-dir` pytest remediation. The `Tests`, `lint`, and `Labeler` workflows all completed successfully on head `fada0d68500a811335004c7b705436b35d35b59c`.

# Procedure

Observed at: 2026-05-04T18:15:28Z

Source state: `origin/main` head `fada0d68500a811335004c7b705436b35d35b59c`, which includes remediation commit `e151e760cce2bdeda8dcb9e4c269b1786be9a676` and Loom record commit `fada0d68500a811335004c7b705436b35d35b59c`.

Procedure: pushed with the guarded command `git fetch origin main && git merge-base --is-ancestor origin/main HEAD && git push origin HEAD:main`, watched GitHub Actions `Tests` run `25334910408` with `gh run watch --exit-status`, and checked `lint` run `25334910369` plus `Labeler` run `25334910419`.

Expected result when applicable: remote workflows should complete successfully, and the prior `tests/core/test_cli.py` failures caused by missing `/home/runner/.dbt` should not recur.

Actual observed result: `Tests` run `25334910408` completed with conclusion `success`; summary query reported `0` non-success jobs out of `20`. `lint` run `25334910369` and `Labeler` run `25334910419` also completed with conclusion `success`.

Procedure verdict / exit code: pass. `gh run watch 25334910408 --exit-status --interval 30` exited successfully after all `Tests` jobs passed.

# Artifacts

- `Tests` workflow: `https://github.com/z3z1ma/dbt-osmosis/actions/runs/25334910408`, status `completed`, conclusion `success`, head `fada0d68500a811335004c7b705436b35d35b59c`, `20` jobs, `0` non-success jobs.
- `lint` workflow: `https://github.com/z3z1ma/dbt-osmosis/actions/runs/25334910369`, status `completed`, conclusion `success`, head `fada0d68500a811335004c7b705436b35d35b59c`.
- `Labeler` workflow: `https://github.com/z3z1ma/dbt-osmosis/actions/runs/25334910419`, status `completed`, conclusion `success`, head `fada0d68500a811335004c7b705436b35d35b59c`.
- The successful `Tests` workflow included green latest dbt compatibility jobs for `1.10.0` and `1.11.0`, docs builds for Node 18 and Node 24, pip install smoke, lockfile check, and pytest matrix jobs.

# Supports Claims

- Supports `ticket:c10pyright32#ACC-004` by confirming the pushed commit passed the CI basedpyright gate inside the successful `Tests` workflow and the `lint` pre-commit workflow.
- Supports the ticket acceptance claim that the post-basedpyright pytest blocker from run `25333721046` was remediated on `origin/main`.
- Supports closing `ticket:c10pyright32` after the reopened `complete_pending_acceptance` state.

# Challenges Claims

None - no observed failure challenged the remediated acceptance claims in the checked workflows.

# Environment

Commit: `fada0d68500a811335004c7b705436b35d35b59c`

Branch: `main` on `origin`

Runtime: GitHub Actions workflows for Tests, lint, and Labeler

OS: GitHub Actions Ubuntu runners for Python workflows; Node workflows used the configured docs runners

Relevant config: default GitHub Actions workflow configuration at the pushed head

External service / harness / data source when applicable: GitHub Actions

# Validity

Valid for: `origin/main` at `fada0d68500a811335004c7b705436b35d35b59c` and the checked workflow runs.

Fresh enough for: closing `ticket:c10pyright32` and resuming the remaining backlog.

Recheck when: new commits are pushed to `origin/main`, workflow definitions change, dependency constraints change, or the CI matrix is rerun against a different source state.

Invalidated by: a later required workflow failure on the same head or a new commit that changes the tested source state.

Supersedes / superseded by: supersedes the pending-remote-confirmation limitation in `evidence:c10pyright32-ci-pytest-profiles-remediation`.

# Limitations

This evidence does not prove future commits will pass CI. The `Tests` workflow emitted a non-blocking GitHub Actions annotation about Node.js 20 action deprecation, but the workflow concluded successfully.

# Result

The pushed remediation was accepted by the current remote CI surface: `Tests`, `lint`, and `Labeler` completed successfully on `origin/main` head `fada0d68500a811335004c7b705436b35d35b59c`.

# Interpretation

The previously observed missing default profiles directory pytest blocker is resolved for the current pushed source state. This evidence is sufficient for the ticket acceptance gate to close the reopened ticket, while leaving the Node.js 20 deprecation annotation as a non-blocking workflow maintenance observation.

# Related Records

- ticket:c10pyright32
- evidence:c10pyright32-ci-pytest-profiles-remediation
- critique:c10pyright32-ci-pytest-profiles-remediation-review
