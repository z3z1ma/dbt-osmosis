---
id: evidence:c10rel08-local-release-workflow-validation
kind: evidence
status: recorded
created_at: 2026-05-04T01:35:08Z
updated_at: 2026-05-04T01:50:21Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10rel08
  packets:
    - packet:ralph-ticket-c10rel08-20260504T012824Z
  critique:
    - critique:c10rel08-release-workflow-hardening
external_refs:
  github_issue: https://github.com/z3z1ma/dbt-osmosis/issues/359
---

# Summary

Observed local validation for the `ticket:c10rel08` release workflow hardening. The final workflow gates version tagging, PyPI publish, and GitHub release-note publishing behind a successful same-repository push run of `Tests` on `main` plus a read-only release-candidate validation job; local package build plus wheel smoke passed without creating tags or publishing artifacts, and mandatory critique passed after follow-up workflow guard fixes.

# Procedure

Observed at: 2026-05-04T01:35:08Z, with follow-up workflow guard validation at 2026-05-04T01:50:21Z.

Source state: local worktree at `a32f3a7008fb5f70c4c5d6c5fc078540b951384d` plus uncommitted `ticket:c10rel08` changes to `.github/workflows/release.yml`, the Ralph packet, ticket, this evidence record, and `critique:c10rel08-release-workflow-hardening`.

Procedure:

- Ralph child inspected the pre-change workflow ordering and confirmed `Detect and tag new version` ran before package build/validation.
- Ralph child changed only `.github/workflows/release.yml` within packet scope.
- Parent inspected the resulting workflow diff and reran `pre-commit run --files .github/workflows/release.yml`.
- Parent ran a local package build and wheel smoke sequence mirroring the workflow package validation steps without pushing tags or publishing.
- Mandatory critique found pre-final blockers in validation credentials and trigger gating; parent split validation and release jobs, made validation read-only, disabled persisted checkout credentials in validation, and restricted `workflow_run` to successful same-repository push runs of `Tests` on `main`.
- Parent reran `pre-commit run --files .github/workflows/release.yml` after the final guard fix.

Expected result when applicable: release validation should happen before any tag/publish/release-note publish step, and package metadata plus clean wheel install smoke should pass locally.

Actual observed result: workflow ordering now puts successful `Tests` plus read-only release-candidate validation before `Detect and tag new version`; local workflow hook and package smoke checks passed, and mandatory critique passed with no open findings.

Procedure verdict / exit code: pass for local workflow hook, package metadata, wheel install smoke, and mandatory critique; final GitHub Actions evidence remains pending after commit/push.

# Artifacts

Ralph child reported:

- `pre-commit run --files .github/workflows/release.yml` - passed.
- Temp worktree package smoke passed: hatchling build, `twine check --strict`, clean Python 3.10 wheel install, CLI/module help, import smoke, and `pip check`.
- `git diff --check -- .github/workflows/release.yml` - passed.

Parent verification reran:

- `pre-commit run --files .github/workflows/release.yml` - passed, including YAML checks and actionlint.
- `uvx --from=hatchling==1.26.3 hatchling build` - passed; created `dist/dbt_osmosis-1.3.0.tar.gz` and `dist/dbt_osmosis-1.3.0-py3-none-any.whl`.
- `uvx --from=twine==6.1.0 twine check --strict dist/*` - passed for both sdist and wheel.
- Clean Python 3.10 wheel install smoke with `uv --no-config pip install --python "$SMOKE_ENV/bin/python" dist/dbt_osmosis-1.3.0-py3-none-any.whl` - passed.
- `"$SMOKE_ENV/bin/dbt-osmosis" --help` - passed.
- `"$SMOKE_ENV/bin/python" -m dbt_osmosis --help` - passed.
- `"$SMOKE_ENV/bin/python" -c "import dbt_osmosis.cli.main"` - passed.
- `uv --no-config pip check --python "$SMOKE_ENV/bin/python"` - passed and reported all installed packages compatible.
- After final `workflow_run` guard fix, `pre-commit run --files .github/workflows/release.yml` - passed, including YAML checks and actionlint.
- Mandatory critique read-only checks: `actionlint .github/workflows/release.yml` and `git diff --check -- .github/workflows/release.yml` - no errors.

# Supports Claims

- `ticket:c10rel08#ACC-001` - local workflow inspection shows build and metadata validation are in the `validate` job before the `release` job can detect/create tags; local package build and metadata validation passed.
- `ticket:c10rel08#ACC-002` - clean wheel install smoke verified CLI help, module help, import, and `pip check`.
- `ticket:c10rel08#ACC-003` - workflow now runs after successful `Tests` and gates tag/publish behind lock freshness, Python checks, dbt parse, pytest, docs build, package build, metadata validation, and wheel smoke in `validate`; local package checks passed.
- `ticket:c10rel08#ACC-004` - workflow ordering now places tag creation after successful `Tests` plus the `validate` job, and final critique found no hidden pre-validation tag/release path.
- `ticket:c10rel08#ACC-005` - workflow now has top-level `contents: read`, read-only validation permissions with `persist-credentials: false`, post-validation release permissions of `contents: write` and `pull-requests: read`, and explicit token-based PyPI publishing.
- `ticket:c10rel08#ACC-006` - workflow now declares Release Drafter as the GitHub release notes source before tag/publish steps.

# Challenges Claims

- `ticket:c10rel08#ACC-003` remains partially supported until GitHub Actions runs the full `Tests` and release validation workflow on `main`.

# Environment

Commit: `a32f3a7008fb5f70c4c5d6c5fc078540b951384d` plus uncommitted c10rel08 changes.

Branch: local worktree branch `loom/dbt-110-111-hardening`; delivery target remains `main`.

Runtime: local macOS/darwin, Python 3.10.15 for wheel smoke.

OS: macOS/darwin.

Relevant config: `.github/workflows/release.yml`, `.github/workflows/constraints.txt`, `pyproject.toml`, `docs/package-lock.json`, `uv.lock`.

External service / harness / data source when applicable: none for local evidence; GitHub Actions evidence is still pending.

# Validity

Valid for: supporting mandatory critique and commit/push trial for `ticket:c10rel08`.

Fresh enough for: review of the current uncommitted release workflow diff.

Recheck when: release workflow ordering, package metadata, docs build, test commands, PyPI publishing method, or release-note ownership changes.

Invalidated by: failing GitHub Actions release workflow, package metadata changes without new smoke, or switching PyPI auth mode.

Supersedes / superseded by: should be superseded by final `main` GitHub Actions evidence after this change is pushed.

# Limitations

- Parent did not run full pytest or docs build locally because those are encoded into the release workflow and will be validated by GitHub Actions after push.
- No real tag, PyPI publish, or GitHub release publish was attempted; this evidence validates ordering and smoke behavior without exercising external side effects.
- Token-based PyPI publishing was preserved; trusted publishing setup remains an external decision if desired later.

# Result

The local release workflow validation, package smoke checks, and mandatory critique passed.

# Interpretation

The current diff satisfies the release workflow hardening shape locally and passed mandatory critique, but final GitHub Actions evidence from `main` is still required before acceptance.

# Related Records

- `ticket:c10rel08`
- `packet:ralph-ticket-c10rel08-20260504T012824Z`
- `critique:c10rel08-release-workflow-hardening`
- `ticket:c10pkg10`
- `ticket:c10docs09`
