Status: recorded
Created: 2026-07-04
Updated: 2026-07-04
Relates-To: .10x/tickets/done/2026-07-04-remediate-uv-audit-vulnerabilities.md, .10x/tickets/2026-07-04-quality-optimizer-hill-climb.md

# uv Audit Remediation Evidence

## What Was Observed

The dependency lockfile was refreshed with the smallest targeted upgrade command needed to clear the baseline `uv audit --frozen` vulnerabilities:

```bash
uv lock --upgrade-package gitpython --upgrade-package msgpack --upgrade-package pygments
```

The resolver changed the vulnerable packages only:

- `gitpython 3.1.46 -> 3.1.50`
- `msgpack 1.1.2 -> 1.2.1`
- `pygments 2.19.2 -> 2.20.0`

After manually restoring unrelated resolver marker normalization, `git diff -- uv.lock` showed only the three targeted package version/hash/wheel metadata updates.

## Procedure

Verification commands were run after the lock update.

```bash
uv lock --check
```

Output:

```text
Using CPython 3.11.15
Resolved 178 packages in 4ms
```

```bash
uv audit --frozen
```

Output:

```text
warning: `uv audit` is experimental and may change without warning. Pass `--preview-features audit-command` to disable this warning.
Found no known vulnerabilities and no adverse project statuses in 177 packages
```

Additional verification already run for this ticket:

- `RUFF_CACHE_DIR=/tmp/dbt-osmosis-ai-quality/ruff-cache uvx ruff==0.15.17 check .` exited 0.
- `RUFF_CACHE_DIR=/tmp/dbt-osmosis-ai-quality/ruff-cache uvx ruff==0.15.17 format --check .` exited 0 with `196 files already formatted`.
- `UV_PROJECT_ENVIRONMENT=/tmp/dbt-osmosis-ai-quality/project-env-remediation uv run --frozen --extra openai --extra duckdb --group dev basedpyright --outputjson` completed with 0 errors and 1861 warnings.
- `UV_PROJECT_ENVIRONMENT=/tmp/dbt-osmosis-ai-quality/project-env-remediation-tests uv run --frozen --extra openai --extra duckdb --group dev pytest -q tests/test_package_metadata.py tests/core/test_workbench_app.py tests/workbench/test_ai_assistant.py` reported `22 passed, 2 warnings in 9.39s`.
- `UV_PROJECT_ENVIRONMENT=/tmp/dbt-osmosis-ai-quality/project-env-remediation-tests uv run --frozen --extra openai --extra duckdb --group dev pytest -q` reported `962 passed, 11 skipped, 2 warnings in 236.72s (0:03:56)`.

## What This Supports

- ACC-001: `uv audit --frozen` exits 0 after the lockfile change.
- ACC-002: `uv lock --check` exits 0.
- ACC-003: Ruff check and format-check exit 0.
- ACC-004: basedpyright has 0 errors in a dependency-complete temp environment.
- ACC-005: Focused optional-surface tests and full pytest pass.
- ACC-006: The effective lockfile diff is limited to the three audited package upgrades and their required artifact metadata.

## Limits

The full local verification used CPython 3.11.15. It does not replace the repository's broader CI matrix. `uv audit` is currently experimental, and the evidence is bounded to the advisory database results available at runtime.
