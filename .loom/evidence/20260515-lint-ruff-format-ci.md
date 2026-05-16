# Lint CI Ruff Format Failure

ID: evidence:20260515-lint-ruff-format-ci
Type: Evidence Observation
Status: recorded
Created: 2026-05-15
Updated: 2026-05-15
Observed: 2026-05-15 20:56 MST

## Summary

The latest failing GitHub Actions run on `main` failed in the `lint` workflow because
`ruff-format` reformatted two files. After applying the formatter locally,
`pre-commit run --all-files` passed.

## Observation

Source state observed:

- Repository: `z3z1ma/dbt-osmosis`
- Branch: `main`
- Commit: `856b323edde1ac614a6781ed2247e0a8a5486eba`
- GitHub Actions run: `https://github.com/z3z1ma/dbt-osmosis/actions/runs/25926468566`
- Failed job: `lint`, job id `76208944573`

GitHub Actions log excerpt from the failed job:

```text
ruff-format..............................................................Failed
- hook id: ruff-format
- files were modified by this hook

2 files reformatted, 94 files left unchanged
```

Local reproduction and validation:

```text
$ uv run ruff format --preview src/dbt_osmosis/core/diff.py tests/core/test_transforms.py
2 files reformatted

$ uv run ruff format --preview --check src/dbt_osmosis/core/diff.py tests/core/test_transforms.py
2 files already formatted

$ pre-commit run --all-files
check python ast.........................................................Passed
check json...............................................................Passed
check yaml...............................................................Passed
check toml...............................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
detect private key.......................................................Passed
debug statements (python)................................................Passed
ruff-format..............................................................Passed
ruff.....................................................................Passed
basedpyright.............................................................Passed
Detect hardcoded secrets.................................................Passed
Lint GitHub Actions workflow files.......................................Passed
```

## Artifacts

- `src/dbt_osmosis/core/diff.py` - Ruff preview formatting changed one dictionary
  comprehension from two lines to one.
- `tests/core/test_transforms.py` - Ruff preview formatting changed one generator
  expression from two lines to one.

## What This Shows

- Standalone observation - supports that the observed GitHub Actions failure was
  a formatting-only pre-commit failure and that the same local pre-commit hook set
  passes after applying Ruff preview formatting.

## What This Does Not Show

This evidence does not show a post-push GitHub Actions rerun for the fix. It also
does not add test coverage beyond the pre-commit lint/type-check workflow because
the code change is formatter-only.
