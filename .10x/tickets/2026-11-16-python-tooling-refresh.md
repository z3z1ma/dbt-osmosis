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
Dependency PR labels are insufficient: Ruff has five active version surfaces including lock and tests. Search all pins before validation; release workflow invokes Hatchling directly, so its two pins must agree with the build backend. Both pins aligned; release gates/version remain unchanged.

## Independent review (dependency_review, 2026-11-16)
Target: 85c6ffeb840eafa164c6d98cbe31cc8475aa968c against 7279fec9bebeee6bf2fb6b1fa6d4d0838883ee9f.
Verdict: concerns (release-build pin rationale outstanding; no new source correctness defect found).
- Minor: `.github/workflows/release.yml` invokes Hatchling 1.26.3 twice, bypassing the updated 1.31.0 backend. This mismatch predates this patch, so it is not introduced by the change; however, calling it deliberately separate needs evidence or an explicit follow-up owner. Executor asked to align within tooling scope or substantiate separation.
- Ruff pins inspected across pyproject optional/dev group, Taskfile, pre-commit, lock and consistency test: coherent. Metadata tests retain Python/runtime extra constraints. The only CLI change groups existing adjacent f-strings in parentheses, preserving value and output. Lock changes are limited to Ruff artifact/version and root metadata.
- Recorded executor evidence supports lint, formatting, lock freshness, backend build, distribution metadata, and 56 focused CLI/metadata tests. Tests were not rerun by reviewer. Dependabot attribution is retained in commit trailer.
Residual risk: full remote matrix and exact-head CI pending; local focused Python 3.13 evidence does not prove all supported environments. No release or merge performed by reviewer.

- Review response: aligned both release workflow direct Hatchling invocations to 1.31.0; no gates, version or publishing behavior changed. `uvx --from=hatchling==1.31.0 hatchling build` passes locally (sdist and wheel). No release dispatched.

### Independent review resolution
Target: 7c06f06bd8af5ead9bdb333f748249c3bcc4572e.
Verdict: pass. Both direct release-build invocations now match Hatchling 1.31.0; executor recorded successful exact-command sdist/wheel builds. Follow-up diff contains only these two pins and ticket updates; publishing gates, version, permissions and credentials are unchanged. Previous minor concern resolved. Remote exact-head CI remains the merge gate; this verdict is source/evidence review, not a claim of matrix completion.
