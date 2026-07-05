Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-04-ratify-dependency-cooldown-policy.md, .10x/tickets/done/2026-07-04-ratify-github-actions-pinning-policy.md

# Semgrep policy zero evidence

## What was observed

Semgrep `p/default` findings were reduced from 32 blocking findings to zero after dependency cooldowns, GitHub Actions SHA pinning, release workflow checkout hardening, and literal optional import checks.

The pre-fix Semgrep default findings were:

- 26 `github-actions-mutable-action-tag`
- 2 `dependabot-missing-cooldown`
- 2 `workflow-run-target-code-checkout`
- 1 `uv-missing-dependency-cooldown`
- 1 `non-literal-import`

## Procedure

- `uv run --no-sync --with semgrep semgrep scan --config p/default --error --json --output /tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/final-post-vulture/semgrep-default-after-policy.json .` → exit 0; 0 findings.
- `uv run --no-sync --with semgrep semgrep scan --config p/security-audit --error --json --output /tmp/dbt-osmosis-ai-quality/exhaustive-2026-07-05/final-post-vulture/semgrep-security-after-policy.json .` → exit 0; 0 findings.
- `uv run python - <<'PY' ... yaml.safe_load(...) ... PY` → exit 0; workflow and Dependabot YAML parse.
- `uvx ruff==0.15.17 check .` → exit 0.
- `uvx ruff==0.15.17 format --check .` → exit 0.
- `uv run basedpyright --level error` → exit 0; 0 errors, 0 warnings, 0 notes.

## What this supports or challenges

This supports closing the dependency cooldown and GitHub Actions pinning policy tickets. It also verifies that the optional workbench dependency import check no longer trips Semgrep's non-literal import rule.

## Limits

This evidence validates Semgrep, YAML parseability, formatting, and the configured type gate. It does not execute GitHub Actions on GitHub-hosted runners.
