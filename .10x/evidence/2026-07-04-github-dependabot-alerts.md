Status: recorded
Created: 2026-07-04
Updated: 2026-07-04
Relates-To: .10x/tickets/done/2026-07-04-remediate-docs-npm-audit-vulnerabilities.md, .10x/tickets/done/2026-07-04-quality-optimizer-hill-climb.md

# GitHub Dependabot Alerts Evidence

## What Was Observed

After pushing `codex/quality-optimizer-hill-climb`, GitHub reported 20 open Dependabot alerts on the default branch.

The alerts were inspected with:

```bash
GH_TOKEN="$(gh auth token --user z3z1ma)" gh api repos/z3z1ma/dbt-osmosis/dependabot/alerts --paginate
```

Open default-branch alerts included:

- Python alerts for `msgpack`, `GitPython`, and `Pygments`, which are already fixed on `codex/quality-optimizer-hill-climb` by `uv.lock` updates.
- npm/docs alerts for packages including `@babel/core`, `@babel/plugin-transform-modules-systemjs`, `http-proxy-middleware`, `joi`, `js-yaml`, `qs`, `serialize-javascript`, `uuid`, and `webpack-dev-server`.

Local docs audit reproduced the npm side:

```bash
npm --prefix docs audit --json
```

Summary:

```text
low: 1
moderate: 23
high: 2
critical: 0
total: 26
```

The audit report indicates the non-major available Docusaurus remediation path is `@docusaurus/preset-classic@3.10.1`, with related transitive updates.

## What This Supports

The Python lock remediation in `.10x/tickets/done/2026-07-04-remediate-uv-audit-vulnerabilities.md` addresses 7 of the currently open default-branch Python alerts once merged. The docs npm alerts are a distinct non-trivial dependency remediation and have their own terminal ticket: `.10x/tickets/done/2026-07-04-remediate-docs-npm-audit-vulnerabilities.md`.

## Limits

GitHub Dependabot alert state is default-branch state and may change after this branch merges. The local npm audit was run against the current working tree before any docs dependency remediation.
