Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-04-remediate-docs-npm-audit-vulnerabilities.md, .10x/tickets/2026-07-05-ratify-docs-node20-docusaurus-upgrade.md

# Docs npm audit remediation evidence

## What was observed

The docs npm audit count was reduced from the previously recorded baseline of 26 vulnerabilities to 20 vulnerabilities without changing the docs Node 18 support contract.

Before this ticket closed, `.10x/tickets/done/2026-07-04-remediate-docs-npm-audit-vulnerabilities.md` recorded `npm --prefix docs audit --json` at 26 vulnerabilities: 2 high, 23 moderate, 1 low.

After updating the existing `qs` override from `6.14.1` to `6.15.3` and regenerating the lockfile, `npm --prefix docs audit --json` reported:

```text
info: 0
low: 0
moderate: 19
high: 1
critical: 0
total: 20
```

The remaining advisories are in the Docusaurus/webpack toolchain chain:

- `@docusaurus/*` packages at ranges `<=3.8.1`
- `copy-webpack-plugin` and `css-minimizer-webpack-plugin` via `serialize-javascript <=7.0.4`
- `webpack-dev-server <=5.2.6` via its own advisories and `sockjs`
- `uuid <11.1.1` via `sockjs`

`npm --prefix docs audit fix --package-lock-only` did not clear the remaining advisories and reported that the fix path requires `@docusaurus/preset-classic@3.10.1`.

## Procedure

Commands run:

```text
npm --prefix docs install --package-lock-only
npm --prefix docs audit fix --package-lock-only
npm --prefix docs audit --json
npm --prefix docs ci
npm --prefix docs run build
```

Compatibility checks:

```text
npm view @docusaurus/core@3.7.0 version engines dependencies.webpack-dev-server --json
npm view @docusaurus/core@3.8.1 version engines dependencies.webpack-dev-server --json
npm view @docusaurus/core@3.9.0 version engines dependencies.webpack-dev-server --json
npm view @docusaurus/core@3.10.1 version engines dependencies.webpack-dev-server --json
```

Observed metadata:

- `@docusaurus/core@3.7.0` and `3.8.1` require Node `>=18.0`.
- `@docusaurus/core@3.9.0`, `3.9.2`, and `3.10.1` require Node `>=20.0`.
- `.github/workflows/tests.yml` builds docs on Node 18 and Node 24.
- `webpack-dev-server@6.0.0` requires Node `>=22.15.0`.
- `serialize-javascript@7.0.5` and `7.0.7` require Node `>=20.0.0`.

Verification results:

- `npm --prefix docs ci` exited 0. It emitted the pre-existing `react-json-view-lite` React peer warning and reported 20 vulnerabilities.
- `npm --prefix docs run build` exited 0 and generated static files in `docs/build`.
- Python/root gates were ruled unaffected because this change touched only docs npm dependency metadata and 10x records.

## What this supports or challenges

This supports closing the safe remediation ticket because the lockfile change reduced advisories and passed docs install/build verification while preserving the documented Node 18 support line.

This challenges any claim that the remaining docs audit advisories can be fully remediated as a routine lockfile update. The remaining npm-proposed path changes the docs Node minimum from 18 to at least 20 and must be ratified separately.

## Limits

The verification ran on the local Node runtime, not a local Node 18 runtime. Node 18 compatibility is inferred from the unchanged `docs/package.json` engine, unchanged docs CI matrix, Docusaurus package metadata for the retained 3.7 line, and avoiding Node-20-only dependency upgrades.

The docs audit still exits non-zero because 20 advisories remain. Those advisories are not dismissed; they are owned by `.10x/tickets/2026-07-05-ratify-docs-node20-docusaurus-upgrade.md`.
