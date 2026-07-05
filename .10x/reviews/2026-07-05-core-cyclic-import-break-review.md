Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: .10x/tickets/done/2026-07-05-break-core-codeql-cyclic-imports.md
Verdict: pass

# Core cyclic import break review

## Target

Review of the helper ownership extraction that clears CodeQL cyclic-import findings in core modules.

## Assumptions Tested

- Moving catalog helpers to `catalog_operations.py` does not change catalog load/generate behavior.
- Moving model-version helpers to `model_versions.py` preserves private imports through `inheritance.py`.
- Moving `_get_node_yaml()` to `node_yaml.py` preserves private imports through `inheritance.py` while letting `PropertyAccessor` avoid importing `inheritance.py`.
- Updating test monkeypatch targets reflects the new helper owner rather than weakening assertions.
- The import-cycle fix should clear CodeQL without introducing other Python security-and-quality findings.

## Findings

No blocking findings.

## Verdict

Pass. The extraction keeps behavior covered by focused and full pytest, passes Ruff and basedpyright, and reduces CodeQL Python security-and-quality findings from 7 to 0.

## Residual Risk

Private monkeypatch users that patch `dbt_osmosis.core.inheritance._get_node_yaml` will not intercept `PropertyAccessor`, which now calls the canonical `node_yaml` owner. The private symbol remains importable for existing direct callers; tests were updated to patch the actual owner for the path under test.
