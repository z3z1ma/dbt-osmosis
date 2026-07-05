Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: docs/package.json, docs/package-lock.json, docs/docusaurus.config.js, .github/workflows/tests.yml, AGENTS.md
Verdict: pass

# Docs Node 20 Docusaurus upgrade review

## Target

Review of the docs Node 20 migration, Docusaurus `3.10.1` upgrade, docs npm security overrides, CI matrix update, and Docusaurus config deprecation cleanup.

## Findings

No blocking findings.

The transitive npm overrides are the main risk. They intentionally exceed some upstream transitive ranges for `serialize-javascript` and `uuid`, but the risk is bounded to the docs workspace and supported by clean install, npm audit, OSV scan, production build, and dev-server smoke evidence in `.10x/evidence/2026-07-05-docs-node20-docusaurus-upgrade.md`.

The local verification ran on Node `v22.23.1`; Node 20 compatibility is delegated to the CI matrix now that docs CI includes Node 20. Docusaurus `3.10.1` package metadata declares Node `>=20.0`.

## Verdict

Pass. The change satisfies the ratified platform upgrade, clears the docs audit vector, keeps CI coverage for the new Node floor, and records the override tradeoff in `.10x/decisions/docs-npm-security-overrides.md`.

## Residual Risk

Future upstream Docusaurus or webpack-dev-server releases may make one or more overrides unnecessary. The active decision requires revisiting overrides on future Docusaurus upgrades.
