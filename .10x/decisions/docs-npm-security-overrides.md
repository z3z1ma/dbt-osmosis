Status: active
Created: 2026-11-16
Updated: 2026-11-16

# Docs npm security overrides

## Context
Supersedes [the July decision](superseded/docs-npm-security-overrides.md) with freshly verified fixed versions; preserves its docs-only, Node20-compatible scope. Docusaurus 3.10.2 still constrains vulnerable transitive packages. Existing overrides themselves became stale.

## Decision
Pin qs 6.17.1, serialize-javascript 7.1.4, uuid 11.1.1, webpack-dev-server 5.2.6, and tinypool 2.1.2 in docs only. The first two refresh existing overrides; tinypool removes critical prototype-pollution worker execution gadgets. npm metadata declares Node20 support; tinypool v2 drops Node18, which this repository no longer supports for docs.

Docusaurus uses Tinypool constructor options, run, destroy and worker state. A forced two-thread SSG production build validates that boundary; ordinary small builds bypass the pool. Retain clean installation, audit/OSV, production build and dev-server smoke as validation requirements, and Node20/24 CI before merge. Revisit overrides on future Docusaurus changes.

## Alternatives and consequences
Leaving tinypool 1.1.1 retains two critical advisories; no fixed 1.x exists. Downgrading Docusaurus via npm audit force would abandon current fixes. Adopting webpack-dev-server 6 raises the Node floor and is excluded. This explicit compatible override is narrower but carries maintenance responsibility. Deepmerge 4.3.1 remains low severity with no fixed release; track .10x/tickets/2026-11-16-docs-deepmerge-advisory.md rather than claim a clean audit.

Evidence and primary sources: .10x/tickets/2026-11-16-dependency-docs.md.
