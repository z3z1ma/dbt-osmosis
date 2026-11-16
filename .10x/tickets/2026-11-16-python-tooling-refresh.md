Status: active
Created: 2026-11-16
Updated: 2026-11-16

# Coherent Python tooling refresh (#433, #447)
## Scope and acceptance
Refresh Hatchling 1.31.0 and Ruff 0.16.2 on 7279fec, all active Ruff pins and lock, preserve runtime/support contracts. Build and metadata checks, lint/format, CLI regression, independent review and exact-head CI required before merge. No release/version bump.
## Journal and evidence
- Inspected original PR patches: #433 only build requirement; #447 only two pyproject pins, missing lock, Taskfile, pre-commit and consistency test.
- Ruff 0.16.2 initially fails implicit-string-concatenation-in-collection-literal at CLI line 2726. Parentheses preserve existing concatenation semantics; existing CLI assertion checks exact documented columns output.
- `uv lock`: 178 packages, only Ruff version changes 0.15.17 -> 0.16.2. `uv lock --check`: pass.
- `uvx --from ruff==0.16.2 ruff check`: pass; `ruff format --check --preview`: 373 files already formatted.
- `uv build` with pinned Hatchling 1.31.0: sdist and wheel succeed. `twine check --strict dist/*`: both pass.
- `uv run --extra dev --extra duckdb --extra openai pytest -q tests/test_package_metadata.py tests/core/test_cli.py`: 56 passed, 2 warnings, 22.10s on Python 3.13.14. This is focused testing, not the full supported dbt matrix.
## Sources
https://hatch.pypa.io/latest/history/hatchling/ (1.31.0 fixes binary shared-script shebang handling); https://github.com/astral-sh/ruff/releases/tag/0.16.2 (upstream bugfix release). No newest-version chase: adopts requested versions with bounded migration.
## Blockers and review
Independent review and remote exact-head CI pending. No merge authorized to executor.
## Retrospective
Dependency PR labels are insufficient: Ruff has five active version surfaces including lock and tests. Search all pins before validation; build backend is separate from release workflow's deliberately pinned invocation, which is not changed here.
