Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-ratify-docs-node20-docusaurus-upgrade.md, .10x/decisions/docs-npm-security-overrides.md

# Docs Node 20 Docusaurus upgrade evidence

## What Was Observed

The docs toolchain was upgraded from Docusaurus `3.7.0` on Node `>=18.0` to Docusaurus `3.10.1` on Node `>=20.0`.

Local metadata checks observed:

- local runtime: Node `v22.23.1`, npm `10.9.8`;
- `@docusaurus/core@3.10.1` and `@docusaurus/preset-classic@3.10.1` declare `engines.node >=20.0`;
- current npm metadata reported Docusaurus `3.10.1` as the current stable version.

The docs CI matrix changed from Node `18` and `24` to Node `20` and `24`.

The docs dependency graph after the final lockfile solve included:

- `@docusaurus/core@3.10.1`;
- `@docusaurus/preset-classic@3.10.1`;
- `@docusaurus/module-type-aliases@3.10.1`;
- `@docusaurus/types@3.10.1`;
- `webpack-dev-server@5.2.6 overridden`;
- `serialize-javascript@7.0.7 overridden`;
- `uuid@11.1.1 overridden`;
- `qs@6.15.3 overridden`.

## Procedure

Commands run:

```text
node --version
npm --version
npm view @docusaurus/core@3.10.1 engines --json
npm view @docusaurus/preset-classic@3.10.1 engines --json
npm --prefix docs install
npm --prefix docs audit --json
npm --prefix docs ls @docusaurus/core webpack-dev-server serialize-javascript uuid qs --depth=4
osv-scanner scan source -r . --format json --output /tmp/dbt-osmosis-ai-quality/node-upgrade/osv-after-overrides.json
npm --prefix docs ci
npm --prefix docs run build
npm --prefix docs run start -- --host 127.0.0.1 --port 3218 --no-open
curl -fsS http://127.0.0.1:3218/dbt-osmosis/
git diff --check
```

## Results

`npm --prefix docs audit --json` exited 0 and reported:

```text
info: 0
low: 0
moderate: 0
high: 0
critical: 0
total: 0
```

`osv-scanner scan source -r .` exited 0. The generated JSON contained 0 vulnerabilities across scanned package manifests.

`npm --prefix docs ci` exited 0 and reported `found 0 vulnerabilities`.

`npm --prefix docs run build` exited 0 and generated static files in `docs/build`. After migrating `onBrokenMarkdownLinks` to `markdown.hooks.onBrokenMarkdownLinks`, the Docusaurus v4 deprecation warning no longer appeared.

The Docusaurus dev server smoke exited 0. The local server compiled successfully at `http://127.0.0.1:3218/dbt-osmosis/`, and `curl` fetched a 1706-byte response.

`git diff --check` exited 0.

## What This Supports Or Challenges

This supports closing the docs Node 20 Docusaurus upgrade ticket and replacing the prior docs npm/OSV vulnerability residual with a clean audit result.

This also supports the scoped docs npm override decision in `.10x/decisions/docs-npm-security-overrides.md`: the override set clears npm/OSV advisories while preserving docs build and dev-server behavior.

## Limits

Local execution used Node `v22.23.1`, not a local Node 20 runtime. The Node 20 floor is supported by Docusaurus package metadata and by updating CI to build docs on Node 20 and Node 24.

The dev-server smoke fetched the docs home route and proves startup, webpack client compilation, and a successful HTTP response. It does not exhaustively exercise every docs page.
