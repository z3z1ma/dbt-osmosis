Status: done
Created: 2026-07-04
Updated: 2026-07-04

# Quality Optimizer Baseline

## Question

What objective quality signals identify the safest highest-impact next changes for the repository under the attached Production Python Quality Optimizer procedure?

## Sources and Methods

- Read `/Users/alexanderbut/.codex/attachments/d4d33cd1-b052-4dae-bffd-6cdcee129104/pasted-text.txt`.
- Read `AGENTS.md`, `src/dbt_osmosis/core/AGENTS.md`, `pyproject.toml`, `Taskfile.yml`, `.github/workflows/*`, and deleted `.loom` tickets from `HEAD`.
- Ran non-mutating or externally isolated checks where possible:
  - `uv lock --check`
  - `ruff format --check .`
  - `ruff check .`
  - `basedpyright --outputjson` in `/tmp/dbt-osmosis-ai-quality/project-env`
  - `ty check ...` in the temp environment
  - `radon cc src tests -s -a -j`
  - `complexipy src tests`
  - `vulture src tests --min-confidence 80`
  - `deptry .`
  - `pydoclint src tests`
  - `semgrep scan --config p/default --error`
  - `uv audit --frozen`
- Tool caches and reports were kept under `/tmp/dbt-osmosis-ai-quality` where possible. A transient `.venv` created by `uv run --no-sync` was removed.

## Findings

- `uv lock --check` completed successfully.
- Ruff format and lint are clean: 190 files already formatted and `ruff check .` passed.
- `basedpyright` is clean when run with declared dev/openai/duckdb dependencies in a temp environment: 0 errors, 1861 warnings, 35 files analyzed.
- `uv audit --frozen` found 7 known vulnerabilities:
  - `gitpython 3.1.46`: 5 advisories, fixed by at least `3.1.50`.
  - `msgpack 1.1.2`: 1 advisory, fixed by `1.2.1`.
  - `pygments 2.19.2`: 1 advisory, fixed by `2.20.0`.
- Dependency provenance from `uv tree --invert`:
  - `gitpython` arrives through `streamlit` in the `workbench` extra.
  - `msgpack` arrives through `mashumaro[msgpack]` via dbt packages.
  - `pygments` arrives through `rich`, `pytest`, `ipython`, and `textual`.
- Semgrep default rules found 33 blocking findings:
  - Dependabot and uv cooldown policy findings.
  - Mutable GitHub Action tag findings in workflows.
  - `workflow_run` checkout findings in release workflow.
  - Dynamic import and dynamic urllib findings in Python code.
- Radon reports `src/dbt_osmosis/core/sync_operations.py:_sync_doc_section` as the largest production complexity hotspot: CC 78, rank F, lines 21-313.
- Complexipy agrees that `_sync_doc_section` is a failed cognitive-complexity hotspot.
- Vulture findings are mostly expected signature, fixture, and callback noise. They are not safe deletion instructions without separate semantic confirmation.
- Deptry is not configured for this project and misclassifies internal `dbt_osmosis` imports as transitive dependency usage; its output is an adoption-policy signal, not an executable cleanup target.
- Pydoclint reports mostly `DOC301` constructor-docstring policy findings. There is no existing pydoclint config or adoption decision, so this is not an executable cleanup target yet.
- `ty` is not a repository gate and reports broader protocol/test diagnostics, but it also flags two redundant casts in `sync_operations.py`.

## Conclusions

Priority order under the attached optimizer is:

1. Remediate `uv audit` vulnerabilities because security and supply-chain findings outrank maintainability metrics.
2. Triage Semgrep default findings because they are security/policy findings, but several require workflow/policy ratification rather than mechanical code edits.
3. Refactor `_sync_doc_section` because independent complexity tools identify it as the clearest maintainability hotspot after security work.

No dependency, CI, or implementation files were changed during this baseline.

## Limits

- Full pytest, coverage, OSV-Scanner, Gitleaks, CodeQL, and jscpd were not run in this baseline turn.
- `mypy` was not run because this repository uses pyright/basedpyright as its configured type gate and has no mypy configuration.
- Some tools created or used external cache/environment state under `/tmp` or user-level tool caches.
- The local workspace date command returned `2026-07-04`; records in this baseline use that date.
