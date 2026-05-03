---
id: constitution:main
kind: constitution
status: active
created_at: 2026-05-03T20:46:40Z
updated_at: 2026-05-03T20:51:03Z
scope:
  kind: workspace
links:
  wiki:
    - wiki:repository-atlas
---

# Vision

`dbt-osmosis` is a Python CLI and package that makes dbt development workflows safer and more maintainable, especially around schema YAML management, column-level documentation inheritance, ad-hoc SQL helpers, and the optional Streamlit workbench.

# Principles

- Preserve dbt project intent and schema YAML content instead of replacing files with lossy rewrites.
- Prefer the shared dbt project/bootstrap spine over parallel runtimes for individual CLI families.
- Keep configuration resolution centralized so version-specific dbt configuration shapes do not leak across the codebase.
- Treat observed structure, validation output, and scan results as evidence, not as policy or acceptance by themselves.
- Assume `dbt-osmosis` has millions of downstream consumers. Avoid breaking public CLI behavior, package APIs, configuration semantics, schema YAML output contracts, and documented workflows unless the change has explicit justification and a migration path.
- Keep the codebase lean. Compatibility work should prevent real user harm without accumulating indefinite cruft that obscures behavior, slows maintenance, or makes the project harder to reason about.

# Constraints

- Python support is `>=3.10,<3.14`.
- Ruff is the active formatter, linter, and import sorter for repository-managed Python code.
- YAML schema mutation must preserve formatting and non-osmosis-owned sections through the round-trip schema helpers.
- `demo_duckdb/` is the canonical dbt fixture for tests and examples.
- Optional OpenAI and workbench paths must fail clearly when optional dependencies are unavailable.
- Breaking changes require deliberate deprecation, migration guidance, and evidence that the non-breaking path is worse for the project and its users.
- Legacy shims, aliases, and transitional paths should be isolated, documented at the point of use, and removed once their migration purpose expires.

# Strategic Direction

Maintain dbt-osmosis as a single coherent CLI/package whose core YAML, inheritance, SQL, generation, lint, diff, and workbench surfaces share project context, configuration resolution, and validation patterns.

# Current Focus

Keep repository structure recoverable for future agents through a Loom atlas grounded in current scan evidence.

# Open Constitutional Questions

None recorded during the initial Loom workspace bootstrap.

# Change History

- 2026-05-03: Created initial Loom constitution from repository instructions, README, and structure scan context.
- 2026-05-03: Added compatibility and lean-code policy for a large downstream consumer base.
