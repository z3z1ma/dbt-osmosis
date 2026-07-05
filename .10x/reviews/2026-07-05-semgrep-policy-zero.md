Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .github/workflows/, .github/dependabot.yml, pyproject.toml, src/dbt_osmosis/cli/main.py
Verdict: pass

# Semgrep policy zero review

## Target

Security and policy changes made to clear Semgrep default and security-audit findings.

## Findings

None.

## Assumptions tested

- GitHub Actions are pinned to full commit SHAs resolved from the source tags reported before the change.
- Dependabot and uv both use the Semgrep-recommended 7-day dependency cooldown.
- Release workflow no longer checks out workflow-run event code directly; it checks out the default branch and asserts the checkout commit matches the completed test workflow SHA.
- Workbench optional dependency checks use literal import targets rather than variable-controlled dynamic imports.

## Verdict

Pass. The changes remove the current Semgrep findings without adding suppressions or baselines.

## Residual risk

Pinned action SHAs require intentional future refreshes. The release workflow safety check should be validated in GitHub Actions after push because local YAML parsing cannot prove hosted-runner behavior.
