Status: recorded
Created: 2026-07-05
Updated: 2026-07-05
Relates-To: .10x/tickets/done/2026-07-05-remediate-workbench-requirements-osv-vulnerabilities.md

# Workbench requirements OSV remediation evidence

## What was observed

OSV-Scanner findings sourced from `src/dbt_osmosis/workbench/requirements.txt` dropped from 33 vulnerability records to 0 after adding fixed lower-bound constraints for the vulnerable transitive packages.

Before:

```text
total OSV records: 38
workbench requirements records: 33
gitpython@3.1.9: 14
h11@0.9.0: 2
idna@3.9.0: 2
ipython@8.9.0: 2
pillow@9.5.0: 10
pyarrow@9.0.0: 3
```

After:

```text
total OSV records: 5
workbench requirements records: 0
```

The 5 remaining OSV records are from `docs/package-lock.json` and match the docs Docusaurus chain already owned by `.10x/tickets/done/2026-07-05-ratify-docs-node20-docusaurus-upgrade.md`.

## Procedure

Commands run:

```text
osv-scanner scan source -r . --format json --output /tmp/dbt-osmosis-ai-quality/continuation/osv-source.json
uv pip compile src/dbt_osmosis/workbench/requirements.txt --python-version 3.10 -o /tmp/dbt-osmosis-ai-quality/continuation/workbench-requirements-resolved-py310.txt
uv pip compile src/dbt_osmosis/workbench/requirements.txt --python-version 3.10 -o /tmp/dbt-osmosis-ai-quality/continuation/workbench-requirements-resolved-after-py310.txt
osv-scanner scan source -r . --format json --output /tmp/dbt-osmosis-ai-quality/continuation/osv-source-after-workbench.json
uv run pytest tests/test_package_metadata.py -o cache_dir=/tmp/dbt-osmosis-ai-quality/continuation/pytest-cache
uv audit --frozen
uvx ruff==0.15.17 check tests/test_package_metadata.py
```

Fixed lower-bound constraints added:

- `gitpython>=3.1.50`
- `h11>=0.16.0`
- `idna>=3.15`
- `ipython>=8.10.0,<9`
- `pillow>=12.2.0`
- `pyarrow>=17.0.0`

Python 3.10 resolution after the change selected safe current versions:

- `gitpython==3.1.50`
- `h11==0.16.0`
- `idna==3.18`
- `ipython==8.39.0`
- `pillow==12.3.0`
- `pyarrow==24.0.0`

Verification results:

- `uv pip compile ... --python-version 3.10` exited 0.
- OSV-Scanner still exited 1 because docs advisories remain, but the parsed after-report has 0 records sourced from `src/dbt_osmosis/workbench/requirements.txt`.
- `uv run pytest tests/test_package_metadata.py` exited 0: 9 passed, 2 warnings.
- `uv audit --frozen` exited 0 with no known vulnerabilities in the root lock.
- `uvx ruff==0.15.17 check tests/test_package_metadata.py` exited 0.

## What this supports or challenges

This supports closing the workbench requirements ticket: the deployment requirements file no longer contributes OSV findings and still resolves on Python 3.10.

This also supports keeping the docs npm advisories separate. OSV still reports docs package-lock vulnerabilities, but those require the Node 20 Docusaurus decision already recorded separately.

## Limits

The Streamlit app was not launched. Verification covered dependency resolution, package metadata tests, and the scanner finding source that motivated the change.

Ruff was not run against `src/dbt_osmosis/workbench/requirements.txt` because requirements files are not Python syntax; the touched Python test was linted instead.
