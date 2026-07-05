Status: recorded
Created: 2026-07-04
Updated: 2026-07-04
Relates-To: .10x/tickets/done/2026-07-04-triage-semgrep-default-findings.md, .10x/tickets/2026-07-04-ratify-dependency-cooldown-policy.md, .10x/tickets/2026-07-04-ratify-github-actions-pinning-policy.md

# Semgrep Triage Evidence

## What Was Observed

Current Semgrep default scan:

```bash
uvx semgrep scan --config p/default --json --output /tmp/dbt-osmosis-ai-quality/semgrep-default-current.json --error
```

Output summary:

```text
Scan completed successfully.
Findings: 33 (33 blocking)
Rules run: 485
Targets scanned: 227
```

Rule counts from the JSON report:

- `package_managers.dependabot.dependabot-missing-cooldown`: 2
- `package_managers.uv.uv-missing-dependency-cooldown`: 1
- `python.lang.security.audit.dynamic-urllib-use-detected`: 1
- `python.lang.security.audit.non-literal-import`: 1
- `yaml.github-actions.security.github-actions-mutable-action-tag`: 26
- `yaml.github-actions.security.workflow-run-target-code-checkout`: 2

## Classification

### Dependency Cooldown Policy

Classification: policy-required, blocked.

Findings:

- `.github/dependabot.yml:8`
- `.github/dependabot.yml:12`
- `pyproject.toml:71`

Rationale:

Semgrep recommends 7-day cooldowns for Dependabot and uv. That duration changes dependency update behavior and may delay urgent release or security updates, so it is not a mechanical cleanup. Owner: `.10x/tickets/2026-07-04-ratify-dependency-cooldown-policy.md`.

### GitHub Actions Mutable Tags

Classification: policy-required, blocked.

Findings:

- `.github/workflows/labeler.yml:13`
- `.github/workflows/labeler.yml:16`
- `.github/workflows/lint.yml:13`
- `.github/workflows/lint.yml:17`
- `.github/workflows/release.yml:29`
- `.github/workflows/release.yml:36`
- `.github/workflows/release.yml:41`
- `.github/workflows/release.yml:122`
- `.github/workflows/release.yml:142`
- `.github/workflows/release.yml:152`
- `.github/workflows/release.yml:167`
- `.github/workflows/release.yml:181`
- `.github/workflows/release.yml:199`
- `.github/workflows/release.yml:207`
- `.github/workflows/tests.yml:23`
- `.github/workflows/tests.yml:26`
- `.github/workflows/tests.yml:73`
- `.github/workflows/tests.yml:76`
- `.github/workflows/tests.yml:187`
- `.github/workflows/tests.yml:190`
- `.github/workflows/tests.yml:225`
- `.github/workflows/tests.yml:228`
- `.github/workflows/tests.yml:311`
- `.github/workflows/tests.yml:314`
- `.github/workflows/tests.yml:385`
- `.github/workflows/tests.yml:388`

Rationale:

Pinning action refs to full SHAs would reduce supply-chain risk but changes workflow maintenance and update behavior, especially in release publishing paths. Owner: `.10x/tickets/2026-07-04-ratify-github-actions-pinning-policy.md`.

### Release `workflow_run` Checkout

Classification: false positive / already mitigated by workflow guards.

Findings:

- `.github/workflows/release.yml:28`
- `.github/workflows/release.yml:141`

Rationale:

Both jobs guard execution with `github.event.workflow_run.event == 'push'`, `head_branch == 'main'`, and `head_repository.full_name == github.repository`. The workflow is therefore checking out the tested commit from same-repository `main`, not untrusted pull request code. The validation job also uses `contents: read`; the publish job has write/release authority but shares the same same-repository-main guard.

No-action rationale:

No workflow behavior should change unless release authority is explicitly superseded. The scanner warning is useful to preserve as context, but the current source guards address the specific pwn-request risk described by the rule.

### Dynamic urllib

Classification: false positive / already guarded.

Finding:

- `src/dbt_osmosis/workbench/app.py:156`

Rationale:

`_fetch_feed_bytes()` calls `_is_http_url()` before `urllib.request.urlopen()`. `_is_http_url()` rejects non-HTTP(S) schemes and missing netlocs, so `file://` is not accepted. The workbench only enables the feed after `--enable-external-feed`, the default URL is the constant `https://news.ycombinator.com/rss`, the CLI does not accept an arbitrary feed URL, and response reads are capped by `FEED_RESPONSE_MAX_BYTES`.

No-action rationale:

Changing to `requests` would introduce a new direct dependency decision solely for a false positive. Adding a suppression would not improve runtime safety.

### Non-literal Import

Classification: false positive / hard-coded allowlist.

Finding:

- `src/dbt_osmosis/cli/main.py:90`

Rationale:

`_check_workbench_app_dependencies()` imports only names from the module-level constant `_WORKBENCH_APP_MODULES`. The tuple is hard-coded and not derived from user input.

No-action rationale:

The current code already implements the whitelist Semgrep recommends. Rewriting the import check would not reduce executable risk.

## What This Supports

- ACC-001: All 33 Semgrep findings are classified with rationale.
- ACC-002: No code/workflow fix was implemented under this ticket because the mechanical findings are false positives and the true positives are policy choices.
- ACC-003: Unimplemented true positives have durable blocked owners; false positives have explicit no-action rationale.

## Limits

This evidence does not resolve the policy-required findings. Semgrep remains at 33 findings until dependency cooldown and GitHub Actions pinning policies are ratified and implemented, or until narrowly justified suppressions are adopted.
