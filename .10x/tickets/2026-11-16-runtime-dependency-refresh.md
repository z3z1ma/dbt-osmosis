Status: active
Created: 2026-11-16
Updated: 2026-11-16

# Runtime compatibility refresh (#448, #449)
## Scope / acceptance
Refresh packaging upper bound <27 and optional OpenAI upper bound <2.54.0 with lock to tested 26.3/2.53.0. Preserve floors, Python support, setuptools cap and source behavior. Separate from tooling and Pillow; independent review and exact-head supported CI required. No release or version bump.
## Sources / assessment
- https://packaging.pypa.io/en/stable/changelog.html: 26.3 adds version ranges, changes requirement/marker validation; Python>=3.9 fits project >=3.10. Project production import only parses dbt versions (config.py); dbt-core 1.10.20 permits packaging>20.9.
- https://raw.githubusercontent.com/openai/openai-python/v2.53.0/CHANGELOG.md: 2.49 raises Python floor to 3.10 (matching ours), 2.52 fixes Retry-After handling; 2.53 adds API types. Runtime uses OpenAI/Azure constructors and chat.completions.create, not removed beta aliases. This coherent compatibility batch updates only two runtime packages, retaining existing lower bounds.
## Journal / evidence
- `uv lock --upgrade-package packaging==26.3 --upgrade-package openai==2.53.0`: 178 packages, four resolution environments, only these two versions change (from 24.2 and 2.36.0). `uv lock --check`: pass.
- Installed versions confirmed via importlib.metadata: Python 3.13.14, OpenAI 2.53.0, packaging 26.3. Focused `uv run --extra dev --extra duckdb --extra openai pytest -q tests/core/test_llm.py tests/core/test_real_config_shapes.py tests/core/test_config_resolution.py`: 95 passed, 5 skipped (azure-identity absent), 2 warnings, 18.66s.
- Real SDK OpenAI and AzureOpenAI clients with httpx.MockTransport: chat completion response parsing passes for both without network/API credentials. packaging Version compares 1.9.6 < 1.10.0 < 1.11.0 correctly.
- `uv pip check`: fails three existing protobuf incompatibilities. Current main's override forces protobuf>=5,<6 while installed dbt-core 1.10.20, dbt-common 1.37.3, dbt-adapters 1.22.4 metadata require >=6,<7. Same failures reproduced in tooling env with original runtime versions: not introduced here. Owner: 2026-11-16-protobuf-override-audit.md. Focused tests do not prove supported matrix or Azure identity support.
## Blockers / review
Independent review and exact-head remote CI pending. No merge by executor.
## Retrospective
Constraint widening alone leaves old lock selections, so explicitly upgrade targeted packages and record actual imported versions. Mocked client constructors in unit tests are supplemented with real SDK transport-level response parsing. Resolver overrides can conceal incompatible metadata; never equate uv lock success with pip check success.

## Independent review (dependency_review, 2026-11-16)
Target: b93d0d73494d9b57795a13a22229d93cf22f24a6 against 7279fec.
Verdict: pass, subject to exact-head supported CI before merge.
Findings: no blocking defect. Only packaging/OpenAI upper bounds and corresponding lock package artifacts/root requirements change; floors, Python support and setuptools workbench cap remain unchanged. No source/tests modified or assertions weakened. Recorded imported versions prove new releases were exercised rather than only permitted; 95 focused tests and real SDK MockTransport smoke complement fake-client unit tests. Dependabot attribution retained.
Residual risk: five Azure identity tests skipped; no live API service test claimed. Baseline protobuf override mismatch is explicitly reproduced and owned separately, not concealed as a passing dependency check. Exact-head matrix remains needed. Reviewer inspected evidence and import/call surfaces without repeating executor validation.
