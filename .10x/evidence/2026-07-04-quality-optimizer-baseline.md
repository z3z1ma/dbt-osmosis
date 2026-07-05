Status: recorded
Created: 2026-07-04
Updated: 2026-07-04
Relates-To: .10x/tickets/2026-07-04-quality-optimizer-hill-climb.md, .10x/tickets/done/2026-07-04-remediate-uv-audit-vulnerabilities.md, .10x/tickets/done/2026-07-04-triage-semgrep-default-findings.md, .10x/tickets/done/2026-07-04-refactor-sync-doc-section-complexity.md

# Quality Optimizer Baseline Evidence

## What Was Observed

- Worktree started dirty with user-owned changes:
  - `AGENTS.md` modified from `MANDATORY: Use loom.` to `MANDATORY: Use 10x.`
  - `.loom/tickets/2026-06-17-gh392-fix-folded-block-scalar.md` deleted.
  - `.loom/tickets/done/2026-06-17-close-dependabot-prs.md` deleted.
  - `CLAUDE.md` deleted.
- No `.10x/` directory existed before this record set.
- `uv lock --check` exited 0.
- `ruff format --check .` exited 0 and reported 190 files already formatted.
- `ruff check .` exited 0 and reported all checks passed.
- `basedpyright --outputjson` in a temp project environment exited 0 with 0 errors and 1861 warnings.
- `uv audit --frozen` exited 1 with 7 known vulnerabilities:
  - 5 on `gitpython 3.1.46`.
  - 1 on `msgpack 1.1.2`.
  - 1 on `pygments 2.19.2`.
- `radon cc src tests -s -a -j` identified `src/dbt_osmosis/core/sync_operations.py:_sync_doc_section` at CC 78, rank F, lines 21-313.
- `complexipy src tests` marked `src/dbt_osmosis/core/sync_operations.py:_sync_doc_section` as failed.
- `semgrep scan --config p/default --error` scanned 219 git-tracked files, ran 485 rules, and reported 33 blocking findings.
- `.venv` and `demo_duckdb/target` were absent after baseline cleanup.

## Procedure

Commands were run from `/Users/alexanderbut/code_projects/personal/dbt-osmosis`. Tool outputs and JSON reports were stored under `/tmp/dbt-osmosis-ai-quality` where possible.

## What This Supports Or Challenges

This supports prioritizing a security/supply-chain ticket before complexity refactoring. It also supports a later focused refactor ticket for `_sync_doc_section` because two independent complexity tools identify the same hotspot.

## Limits

This evidence does not prove behavior is correct. Full pytest, coverage, OSV-Scanner, Gitleaks, CodeQL, jscpd, and a post-change metric vector still need to run for closure of any implementation ticket.
