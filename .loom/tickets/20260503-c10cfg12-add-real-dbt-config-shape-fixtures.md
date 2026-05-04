---
id: ticket:c10cfg12
kind: ticket
status: ready
change_class: code-behavior
risk_class: high
created_at: 2026-05-03T21:10:43Z
updated_at: 2026-05-04T03:29:15Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  initiative:
    - initiative:dbt-110-111-hardening
  research:
    - research:dbt-110-111-api-surfaces
  evidence:
    - evidence:oracle-backlog-scan
  related_tickets:
    - ticket:c10col01
    - ticket:c10meta02
depends_on: []
---

# Summary

Add real dbt 1.10/1.11 parsed fixture coverage for `config.meta`, `config.extra`, `unrendered_config`, and column `config.meta`/`config.tags` instead of relying mainly on mocks.

# Context

The tests/fixtures oracle found that compatibility-sensitive config behavior is mostly mocked in `tests/core/test_config_resolution.py`, while demo fixtures mostly exercise legacy YAML `meta` and SQL `config(dbt_osmosis_...)`. This makes CI weak against actual dbt manifest shape changes. `ticket:c10col01#ACC-005` and `ticket:c10meta02#ACC-005` were accepted with exact parsed-fixture evidence gaps converted here.

# Why Now

Tickets c10meta02 and c10res14 need real parsed dbt nodes to prove dbt 1.10/1.11 compatibility. Mock-only tests can pass while the actual manifest places config differently.

# Scope

- Add a demo or temp fixture model/source/seed that uses YAML `config: meta:`, SQL `{{ config(meta={...}) }}`, legacy `config(dbt_osmosis_...)`, and column `config.meta`/`config.tags`.
- Include or explicitly split the converted follow-up shapes from ticket:c10col01 and ticket:c10meta02, especially a real parsed column `config.meta` / `config.tags` shape and a missing-YAML-column injection path under dbt 1.10/1.11.
- Parse under dbt 1.10.x and dbt 1.11.x.
- Assert actual manifest fields used by `SettingsResolver`, `PropertyAccessor`, inheritance, and sync logic.
- Include version-gated assertions only where dbt legitimately differs.
- Log/assert installed dbt version for the tests.

# Out Of Scope

- Implementing the resolver changes themselves; ticket:c10meta02 and ticket:c10res14 own behavior changes.
- Broad golden-file YAML rewrites unless needed for the fixture.

# Acceptance Criteria

- ACC-001: A real parsed fixture exposes dbt-osmosis options under node `config.meta` and column `config.meta`.
- ACC-002: Tests assert actual `node.config.meta`, `node.config.extra`, and `node.unrendered_config` fields under dbt 1.10.x and 1.11.x.
- ACC-003: Tests cover column `config.tags` and `config.meta` without relying only on `Mock` objects.
- ACC-004: The fixture remains compatible with the repository's documented supported dbt floor or is isolated/version-gated.
- ACC-005: Failing behavior before resolver fixes is captured if this ticket is implemented test-first with related fixes.
- ACC-006: Converted follow-up gaps from ticket:c10col01#ACC-005 and ticket:c10meta02#ACC-005 are either covered by this fixture work or split into explicit follow-up tickets before ticket:c10cfg12 closes.

# Coverage

Covers:

- ticket:c10cfg12#ACC-001
- ticket:c10cfg12#ACC-002
- ticket:c10cfg12#ACC-003
- ticket:c10cfg12#ACC-004
- ticket:c10cfg12#ACC-005
- ticket:c10cfg12#ACC-006
- ticket:c10col01#ACC-005
- ticket:c10meta02#ACC-005
- initiative:dbt-110-111-hardening#OBJ-001
- initiative:dbt-110-111-hardening#OBJ-002
- initiative:dbt-110-111-hardening#OBJ-004

# Claim Matrix

| Claim | Evidence | Critique | Status |
| --- | --- | --- | --- |
| ticket:c10cfg12#ACC-001 | research:dbt-110-111-api-surfaces | None | open |
| ticket:c10cfg12#ACC-002 | None - fixture tests not written yet | None | open |
| ticket:c10cfg12#ACC-006 | ticket:c10col01; ticket:c10meta02 | None | open; converted follow-up intake acknowledged |

# Execution Notes

This ticket may be implemented alongside ticket:c10meta02 or c10res14 if test-first execution is practical. Keep fixture additions small and explicitly named for dbt 1.10+ config shapes.

# Blockers

Potential blocker: if the current supported dbt floor cannot parse the desired fixture shape, isolate it under version-specific tests.

# Evidence

Existing evidence: research:dbt-110-111-api-surfaces and evidence:oracle-backlog-scan. Missing evidence: failing/passing pytest output under dbt 1.10/1.11, including disposition of the converted c10col01/c10meta02 fixture gaps.

# Critique Disposition

Risk class: high

Critique policy: mandatory

Policy rationale: Compatibility tests define the support claim and can easily become brittle or misleading.

Required critique profiles: test-coverage, dbt-compatibility

Findings: None - no critique yet.

Disposition status: pending

Deferral / not-required rationale: None.

# Retrospective / Promotion Disposition

Disposition status: pending

Promoted: None - implementation not complete.

Deferred / not-required rationale: Consider wiki/testing note if this becomes the canonical compatibility fixture.

# Wiki Disposition

N/A - no wiki promotion selected yet.

# Acceptance Decision

Accepted by: Not accepted yet.
Accepted at: N/A.
Basis: Pending test evidence.
Residual risks: dbt patch releases may alter manifest serialization.

# Dependencies

Coordinate with ticket:c10fix11, ticket:c10col01, ticket:c10meta02, and ticket:c10res14.

# Journal

- 2026-05-03T21:10:43Z: Created from tests/fixtures and dbt compatibility oracle findings.
- 2026-05-04T03:29:15Z: Recorded ownership of converted fixture evidence gaps from ticket:c10col01#ACC-005 and ticket:c10meta02#ACC-005 so those accepted tickets do not leave their exact dbt 1.11 parsed-fixture gaps only in closure prose.
