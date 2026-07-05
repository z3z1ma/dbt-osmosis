Status: active
Created: 2026-07-05
Updated: 2026-07-05

# Docs npm security overrides

## Context

The docs site uses Docusaurus 3 under `docs/`. The project previously supported Node `>=18.0` for the docs toolchain, while Docusaurus `3.9+` requires Node `>=20.0`.

After the user ratified the docs Node upgrade on 2026-07-05, the docs package was moved to Docusaurus `3.10.1`, the current stable Docusaurus release observed locally on 2026-07-05. That upgrade alone did not clear `npm audit` because Docusaurus `3.10.1` still resolves vulnerable transitive packages through these chains:

- `@docusaurus/bundler@3.10.1` -> `copy-webpack-plugin@11.0.0` / `css-minimizer-webpack-plugin@5.0.1` -> `serialize-javascript@6.0.2`
- `@docusaurus/core@3.10.1` -> `webpack-dev-server@5.x` -> `sockjs@0.3.24` -> `uuid@8.3.2`

No fixed `webpack-dev-server` 5.x release newer than `5.2.6` existed in npm metadata observed locally. `webpack-dev-server@6.0.0` exists but requires Node `>=22.15.0`, which is a larger platform support change than the ratified Node 20 upgrade.

## Decision

For the docs npm workspace only, use exact `overrides` for security remediation when all of the following are true:

- the vulnerable package is transitive to the Docusaurus docs toolchain;
- the Docusaurus current stable line has no direct dependency update that clears the advisory;
- the override uses a concrete fixed version observed in npm metadata;
- `npm --prefix docs ci`, `npm --prefix docs audit`, `osv-scanner scan source -r .`, `npm --prefix docs run build`, and a Docusaurus dev-server smoke pass.

The current exact overrides are:

```json
{
  "qs": "6.15.3",
  "serialize-javascript": "7.0.7",
  "uuid": "11.1.1",
  "webpack-dev-server": "5.2.6"
}
```

## Alternatives Considered

- Keep only Docusaurus `3.10.1`: rejected because npm audit and OSV still reported docs vulnerabilities.
- Wait for an upstream Docusaurus or webpack-dev-server 5.x remediation: rejected for the current quality pass because it leaves known advisories unresolved with no available fixed 5.x `webpack-dev-server` release.
- Override `webpack-dev-server` to `6.0.0`: rejected because it requires Node `>=22.15.0`, expanding the support change beyond the ratified Node 20 floor.
- Suppress npm/OSV findings: rejected because fixed package versions are available and the docs build/start path can be verified locally.

## Consequences

The docs workspace now carries explicit transitive overrides. Future Docusaurus upgrades should revisit and remove any override that is no longer necessary.

The override choice is narrower than adopting Node 22 because the docs CI matrix can remain on Node 20 and Node 24.

The override choice is wider than pure Docusaurus semver resolution because `serialize-javascript` and `uuid` fixed versions exceed vulnerable packages' declared transitive ranges. This is accepted only for the docs workspace and only with clean install, audit, OSV, build, and start-smoke evidence.
