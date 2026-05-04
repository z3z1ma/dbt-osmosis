---
id: "wiki:sql-proxy-boundary"
kind: wiki
page_type: reference
status: accepted
created_at: 2026-05-04T16:50:00Z
updated_at: 2026-05-04T16:50:00Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  tickets:
    - ticket:c10proxy25
  evidence:
    - evidence:c10proxy25-sql-proxy-boundary-validation
  critique:
    - critique:c10proxy25-sql-proxy-boundary-review
---

# Summary

The MySQL SQL proxy is an experimental opt-in runtime. It is not part of the supported base CLI surface, and it is not a production-hardened database proxy.

# Contract

- `dbt-osmosis[proxy]` installs dependencies for the experimental proxy runtime only.
- `mysql-mimic` stays out of base dependencies and belongs to the `proxy` optional extra.
- `sqlglot` remains a base dependency because SQL linting uses it outside the proxy.
- Installing the extra does not start a server or configure authentication, TLS, listen, or bind settings.
- The module entrypoint in `src/dbt_osmosis/sql/proxy.py` is a local-only experiment that relies on `mysql-mimic` defaults. Do not expose it to untrusted networks.
- `DbtSession.query()` should pass the original client SQL text to `execute_sql_code()` instead of reserializing through SQLGlot before execution.
- Proxy ALTER COMMENT middleware mutates only the in-memory dbt manifest for the current proxy session. It does not write schema YAML or other durable files.

# Examples

Use precise wording when documenting or reviewing proxy behavior:

```text
supported claim: experimental opt-in local proxy runtime
unsupported claim: production-supported proxy server
unsupported claim: auth/TLS/listen hardening provided by dbt-osmosis
unsupported claim: proxy comments are written back to schema YAML
```

# Notes

- If future work promotes the proxy to a supported feature, it needs a new ticket for auth/listen/TLS posture, runtime tests, and user-facing launch/configuration docs.
- If future work removes the proxy, reconcile this page, the optional extra, package metadata tests, and README/Docusaurus docs together.
- If future work makes comment updates durable, it must use the canonical schema YAML helpers rather than ad hoc string editing.

# Sources

- `ticket:c10proxy25`
- `evidence:c10proxy25-sql-proxy-boundary-validation`
- `critique:c10proxy25-sql-proxy-boundary-review`
- `packet:ralph-ticket-c10proxy25-20260504T163103Z`

# Related Pages

- `wiki:repository-atlas`
- `wiki:ci-compatibility-matrix`
