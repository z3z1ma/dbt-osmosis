---
id: evidence:c10feed26-workbench-feed-hardening-validation
kind: evidence
status: recorded
created_at: 2026-05-04T22:22:20Z
updated_at: 2026-05-04T22:35:45Z
scope:
  kind: repository
  repositories:
    - repo:root
links:
  ticket:
    - ticket:c10feed26
  packet:
    - packet:ralph-ticket-c10feed26-20260504T221607Z
external_refs: {}
---

# Summary

Observed red/green validation for making the workbench Hacker News RSS feed disabled by default, explicitly opt-in, timeout-bound when enabled, failure-tolerant, and escaped before `InnerHTML` rendering.

# Procedure

Observed at: 2026-05-04T22:22:20Z

Source state: commit `4bc9bdbc99563806709486b27f00e1b842a6d210` on branch `loom/dbt-110-111-hardening` with uncommitted c10feed26 changes in `src/dbt_osmosis/workbench/app.py`, `src/dbt_osmosis/cli/main.py`, `tests/core/test_workbench_app.py`, `tests/core/test_cli.py`, `docs/docs/reference/cli.md`, `ticket:c10feed26`, and `packet:ralph-ticket-c10feed26-20260504T221607Z`.

Procedure: Ralph child first added feed hardening and CLI opt-in tests and ran the focused test command before implementation. Parent then reviewed the source diff and reran focused tests, Ruff, and basedpyright after implementation. After mandatory critique found malformed URL and uncapped response-size risks, Ralph iteration 02 added focused red tests and fixed both findings; parent reviewed the focused diff and reran validation. After Ruff formatting adjusted source/test files, parent reran full pre-commit and focused tests.

Expected result when applicable: pre-fix tests fail because the app lacks `build_feed_html`, `--enable-external-feed` parsing, CLI help/wiring, disabled default behavior, safe parser failure handling, and HTML/URL escaping. Post-fix tests pass, Ruff passes, and basedpyright reports zero errors.

Actual observed result: child reported expected red with `6 failed, 37 passed` in iteration 01 and expected red with 2 failures in iteration 02. Parent observed green focused tests, green Ruff, and basedpyright with zero errors after both implementation iterations. Parent later observed full pre-commit pass and focused tests still green after formatting.

Procedure verdict / exit code: mixed expected red then green. Parent-observed green commands exited successfully; basedpyright reported warnings but `0 errors`.

# Artifacts

Child-reported red command:

```bash
uv run pytest tests/core/test_workbench_app.py tests/core/test_cli.py -q
```

Child-reported red result:

```text
6 failed, 37 passed
```

Parent-observed green commands:

```bash
uv run pytest tests/core/test_workbench_app.py tests/core/test_cli.py -q
```

```text
43 passed in 9.99s
```

```bash
uv run ruff check src/dbt_osmosis/workbench/app.py src/dbt_osmosis/workbench/components/feed.py src/dbt_osmosis/cli/main.py tests/core/test_workbench_app.py tests/core/test_cli.py
```

```text
All checks passed!
```

```bash
uv run basedpyright src/dbt_osmosis/workbench/app.py src/dbt_osmosis/cli/main.py
```

```text
0 errors, 354 warnings, 0 notes
```

Parent-observed final local gate after formatting:

```bash
uv run pre-commit run --all-files
```

```text
check python ast.........................................................Passed
check json...............................................................Passed
check yaml...............................................................Passed
check toml...............................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
detect private key.......................................................Passed
debug statements (python)................................................Passed
ruff-format..............................................................Passed
ruff.....................................................................Passed
basedpyright.............................................................Passed
Detect hardcoded secrets.................................................Passed
Lint GitHub Actions workflow files.......................................Passed
```

```bash
uv run pytest tests/core/test_workbench_app.py tests/core/test_cli.py -q
```

```text
45 passed in 9.80s
```

Iteration 02 child-reported red command:

```bash
uv run pytest tests/core/test_workbench_app.py -q
```

Child-reported red result: expected red, 2 failures. Malformed entry URL raised `ValueError`; oversized RSS response did not raise.

Iteration 02 parent-observed green commands:

```bash
uv run pytest tests/core/test_workbench_app.py tests/core/test_cli.py -q
```

```text
45 passed in 9.78s
```

```bash
uv run ruff check src/dbt_osmosis/workbench/app.py tests/core/test_workbench_app.py src/dbt_osmosis/cli/main.py tests/core/test_cli.py
```

```text
All checks passed!
```

```bash
uv run basedpyright src/dbt_osmosis/workbench/app.py src/dbt_osmosis/cli/main.py
```

```text
0 errors, 354 warnings, 0 notes
```

Parent diff inspection observed:

- `dbt-osmosis workbench --enable-external-feed` now passes `--enable-external-feed` after the Streamlit app script path.
- Workbench app args now parse `--enable-external-feed`.
- `build_feed_html(enable_external_feed=False, ...)` returns safe disabled HTML without calling fetcher/parser.
- Enabled feed fetch uses `urllib.request.urlopen(..., timeout=3.0)`.
- Feed/network/parser failures return safe fallback HTML without exception details.
- Feed entry titles/published values are escaped and non-HTTP(S) entry URLs are omitted.
- URL validation now fails closed when URL parsing raises.
- Entry rendering now happens inside `build_feed_html()` fallback handling.
- `_fetch_feed_bytes()` now caps reads at `FEED_RESPONSE_MAX_BYTES + 1` and rejects oversized responses.
- CLI docs mention the new opt-in flag.

# Supports Claims

- ticket:c10feed26#ACC-001: disabled default path avoids outbound RSS fetch at startup unless the app receives explicit opt-in.
- ticket:c10feed26#ACC-002: tests and diff show entry text and attributes are escaped and unsafe URLs are rejected before HTML rendering.
- ticket:c10feed26#ACC-003: CLI and app expose `--enable-external-feed`; the feed is disabled by default.
- ticket:c10feed26#ACC-004: enabled feed fetch uses a 3 second timeout, catches network/parser/rendering failures, fails closed on malformed URLs, and caps response reads.
- ticket:c10feed26#ACC-005: parser failure test returns fallback HTML and does not raise.
- ticket:c10feed26#ACC-006: malicious feed entry test verifies script/HTML content and `javascript:`/`data:` URLs are not injected raw.
- initiative:dbt-110-111-hardening#OBJ-006: hardens external content and network behavior in the workbench.

# Challenges Claims

None observed. The expected red failure challenged the pre-fix implementation, not the post-fix claims.

# Environment

Commit: `4bc9bdbc99563806709486b27f00e1b842a6d210` plus uncommitted c10feed26 implementation and Loom record changes.

Branch: `loom/dbt-110-111-hardening`

Runtime: `uv run python --version` -> `Python 3.10.15`

OS: macOS 15.7.5 build 24G624

Relevant config: base `uv` environment; `VIRTUAL_ENV` warning observed and ignored by `uv`.

External service / harness / data source when applicable: none; tests used stubs and did not require outbound RSS access.

# Validity

Valid for: local source state containing the post-format c10feed26 diff and the installed local dependency set.

Fresh enough for: required security/code-change/test-coverage critique and ticket review state.

Recheck when: workbench feed rendering, CLI workbench argument forwarding, Streamlit component rendering, feedparser behavior, or RSS fetching code changes.

Invalidated by: failing focused tests, Ruff failure, basedpyright errors, or evidence that feed HTML can inject raw untrusted HTML/unsafe URLs in the post-fix source state.

Supersedes / superseded by: none.

# Limitations

This evidence does not run a browser-based Streamlit session and does not fetch the real Hacker News RSS feed. It validates the hardened helper, CLI/app opt-in wiring, response-size cap, and fallback/escaping behavior under unit tests.

# Result

The observed post-fix source state passes focused local validation and makes the workbench RSS feed disabled by default, opt-in, timeout-bound and size-capped when enabled, failure-tolerant, and escaped before rendering.

# Interpretation

The evidence supports moving `ticket:c10feed26` to required critique because implementation and local validation are complete. It does not by itself satisfy the security critique gate, remote CI expectations, or final acceptance decision.

# Related Records

- ticket:c10feed26
- packet:ralph-ticket-c10feed26-20260504T221607Z
- initiative:dbt-110-111-hardening
