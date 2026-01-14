---
sidebar_position: 3
---

# Review changes safely

Use dry runs and checks to validate dbt-osmosis output before writing files.

## Preview without writing

```bash
dbt-osmosis yaml refactor --dry-run
```

## Enforce no-diff mode

```bash
dbt-osmosis yaml refactor --check
```

`--check` exits non-zero if changes would be made, which is ideal for CI.

## Combine with selection

```bash
dbt-osmosis yaml refactor models/staging --dry-run --check
```

## Review file moves

When dbt-osmosis detects a file move, it prompts for confirmation. Use `--auto-apply` to skip prompts in non-interactive runs:

```bash
dbt-osmosis yaml refactor --auto-apply
```

## Audit the diff

```bash
git diff
```

Review the YAML diffs before committing.
