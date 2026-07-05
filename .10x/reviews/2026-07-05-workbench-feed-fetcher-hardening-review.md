Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Target: src/dbt_osmosis/workbench/app.py, tests/core/test_workbench_app.py
Verdict: pass

# Workbench feed fetcher hardening review

## Target

Review of the workbench RSS feed fetcher change that replaces dynamic `urllib.request.urlopen` with explicit `http.client` HTTP(S) connections.

## Findings

No blocking findings.

The change preserves the existing opt-in model and safe fallback behavior. It keeps timeout handling, the response-size cap, and entry URL sanitization. New tests cover non-http scheme rejection, oversized responses, HTTP error closure, query paths, and root-path requests.

One behavior change is intentional: redirects are no longer followed by `urllib` automatically. For the fixed Hacker News RSS URL this is acceptable; any external feed failure returns the existing unavailable fallback.

## Verdict

Pass. The Semgrep security-audit finding is removed without adding a dependency or broad suppression, focused tests pass, and CodeQL remains clean.

## Residual Risk

The workbench feed remains an optional external network feature. Future feed-source expansion should keep the same fail-closed model and avoid user-controlled arbitrary fetches.
