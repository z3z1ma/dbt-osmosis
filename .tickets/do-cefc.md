---
"id": "do-cefc"
"status": "closed"
"deps": []
"links": []
"created": "2026-01-27T02:05:12Z"
"type": "feature"
"priority": 1
"assignee": "z3z1ma"
"tags": []
"external": {}
---
# Expose test_suggestions module via CLI command

## Notes

**2026-01-27T02:07:30Z**

Starting implementation: Adding test suggestions CLI command to main.py. Plan: Add 'test' command group with 'suggest' subcommand.

**2026-01-27T02:11:40Z**

Implementation complete. Added 'dbt-osmosis test suggest' command with:
- Pattern-based and AI-powered test suggestions
- Output formats: table, json, yaml
- FQN filtering and model selection
- All tests passing

Verification steps:
1. Run: uv run dbt-osmosis test suggest --help
2. Run: uv run dbt-osmosis test suggest --project-dir demo_duckdb --profiles-dir demo_duckdb -t test --pattern-only
3. Test JSON output: uv run dbt-osmosis test suggest --project-dir demo_duckdb --profiles-dir demo_duckdb -t test --pattern-only --format json
