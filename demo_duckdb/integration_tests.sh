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

echo "All dbt-osmosis yaml integration tests passed!"
