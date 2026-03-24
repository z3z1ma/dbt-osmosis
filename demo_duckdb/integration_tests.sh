#!/bin/bash
set -euo pipefail

# Ensure we're in the demo_duckdb directory
cd "$(dirname "$0")"

# Common options for all commands
common_options=(
  --project-dir .
  --profiles-dir .
  --target test
)

# Run the tests
uv run dbt-osmosis yaml organize --auto-apply "${common_options[@]}"
uv run dbt-osmosis yaml document "${common_options[@]}"
uv run dbt-osmosis yaml refactor --auto-apply "${common_options[@]}"
uv run dbt-osmosis yaml --help >/dev/null

# Restore YAML fixtures that may have been overwritten by the commands above.
# git checkout restores tracked files; git clean removes untracked files created by organize.
git checkout -- models/ seeds/
git clean -fd models/ seeds/

echo "All dbt-osmosis yaml integration tests passed!"
