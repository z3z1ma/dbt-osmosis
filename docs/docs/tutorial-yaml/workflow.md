---
sidebar_position: 5
---

# Workflow

This guide shows common ways to run dbt-osmosis in day-to-day development.

## On-demand runs

Use this when you want manual control over when YAML updates happen.

```bash
dbt-osmosis yaml refactor --dry-run
```

When the output looks correct, apply:

```bash
dbt-osmosis yaml refactor --auto-apply
```

## Pre-commit hook

Run dbt-osmosis automatically for model changes.

```yaml title=".pre-commit-config.yaml"
repos:
  - repo: https://github.com/z3z1ma/dbt-osmosis
    rev: v1.1.5
    hooks:
      - id: dbt-osmosis
        files: ^models/
        args: ["yaml", "refactor", "--check"]
        additional_dependencies: ["dbt-<adapter>"]
```

Tip: add `--auto-apply` if you want file moves without prompts.

## CI/CD automation

A typical CI workflow:

```bash
dbt-osmosis yaml refactor --check
```

If it exits non-zero, fail the job and ask contributors to run dbt-osmosis locally. For PR automation, run it in a dedicated branch and open a PR with the generated YAML changes.

## Recommended safety flags

- `--dry-run` to preview changes
- `--check` to enforce zero-diff in CI
- `--auto-apply` for non-interactive runs
