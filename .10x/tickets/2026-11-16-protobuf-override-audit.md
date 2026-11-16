Status: open
Created: 2026-11-16
Updated: 2026-11-16

# Audit existing protobuf override
Scope: investigate pyproject override protobuf>=5,<6 versus current locked dbt-core/common/adapters requiring >=6,<7. Do not remove override without understanding its original compatibility purpose and supported matrix.
Evidence: runtime and unchanged-runtime tooling environments both report same three uv pip check errors; focused runtime tests pass. Observed 2026-11-16, Python 3.13.14/dbt-core 1.10.20/common 1.37.3/adapters 1.22.4/protobuf 5.29.6.
Acceptance: explain override purpose, choose tested compatible resolution across supported dbt versions, require pip check and matrix CI before any change. No implementation here; separate from #448/#449 compatibility update.
Blockers: original override purpose and cross-version resolution unverified.
