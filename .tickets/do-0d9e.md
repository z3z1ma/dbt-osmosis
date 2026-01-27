---
"id": "do-0d9e"
"status": "closed"
"deps": []
"links": []
"created": "2026-01-27T02:05:13Z"
"type": "feature"
"priority": 1
"assignee": "z3z1ma"
"tags": []
"external": {}
---
# Expose sql_lint module via CLI command

## Notes

**2026-01-27T02:14:01Z**

Implemented sql_lint CLI exposure

Changes made:
1. Added sql_lint imports to osmosis.py
   - Imported LintLevel, LintViolation, LintResult, LintRule, SQLLinter
   - Imported all rule classes: KeywordCapitalizationRule, LineLengthRule, SelectStarRule, TableAliasRule, QuotedIdentifierRule
   - Added lint_sql_code convenience function
   - Added exports to __all__

2. Added new 'lint' CLI group to main.py with three subcommands:
   - lint file: Lint SQL strings or files
   - lint model: Lint dbt model SQL
   - lint project: Lint all models in a dbt project

3. Features implemented:
   - Rule filtering via --rules and --disable-rules options
   - SQL dialect selection via --dialect option
   - Color-coded severity levels in output
   - Exit code 1 if errors or warnings found
   - Detailed violation reporting with line numbers

Testing:
- Tested with demo_duckdb project
- Verified all commands work correctly
- Confirmed linting detects issues (e.g., SELECT * warning)

Committed changes to branch murmur/do-0d9e (sha: 9fa20bb)
