# Author: Alex B
# Description: Makefile for setting up the development environment. Includes convenient aliases for common tasks.

PY_VERSION := $(shell cat .python-version)

.PHONY: uv venv pre-commit dev all test lint format scan scan-new-baseline \
        scan-without-baseline clean

check: format lint test

.uv-installed-$(PY_VERSION): .python-version
	@if [ ! -f .python-version ]; then echo 'Please create a .python-version file with the desired Python version'; exit 1; fi
	@if command -v uv > /dev/null; then echo 'Verified uv is installed'; else echo 'Please install uv by running `curl -LsSf https://astral.sh/uv/install.sh | sh` or visit https://docs.astral.sh/uv/ for more information'; exit 1; fi
	@uv tool update-shell
	@uv python install
	@rm -f .uv-installed-*
	@touch .uv-installed-$(PY_VERSION)


uv: .uv-installed-$(PY_VERSION)

.venv: .uv-installed-$(PY_VERSION)
	@uv venv .venv

venv: .venv

.git/hooks/pre-commit: .uv-installed-$(PY_VERSION)
	@uv tool install pre-commit
	@uv tool run pre-commit install

pre-commit: .git/hooks/pre-commit

dev: .venv .git/hooks/pre-commit
	@uv sync --extra=dev --extra=duckdb

clean:
	@rm -rf .venv target demo_duckdb/target demo_sqlite/target

lint: .uv-installed-$(PY_VERSION)
	@uvx ruff check

format: .uv-installed-$(PY_VERSION)
	@uvx ruff check --fix --select I
	@uvx ruff format --preview

test: .uv-installed-$(PY_VERSION)
	@uv run pytest tests/

scan: .uv-installed-$(PY_VERSION)
	@uvx bandit -r src -b tests/bandit_baseline.json

scan-new-baseline: .uv-installed-$(PY_VERSION)
	@uvx bandit -r src -f json -o tests/bandit_baseline.json

scan-without-baseline: .uv-installed-$(PY_VERSION)
	@uvx bandit -r src

requirements.txt: .uv-installed-$(PY_VERSION)
	@uv export -o requirements.txt --no-hashes --frozen
