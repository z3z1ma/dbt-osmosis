from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
RUFF_VERSION = "0.8.6"
SUPPORTED_EXTRA_INSTALLS = (".[openai]", ".[azure]", ".[workbench]", ".[duckdb]", ".[proxy]")


def _pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _package_names(requirements: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].lower() for requirement in requirements
    }


def test_base_dependencies_and_optional_extras_are_intentional() -> None:
    pyproject = _pyproject()
    base_dependencies = _package_names(pyproject["project"]["dependencies"])
    extras = pyproject["project"]["optional-dependencies"]

    assert "sqlglot" in base_dependencies
    assert "pyyaml" in base_dependencies
    assert "mysql-mimic" not in base_dependencies

    assert {"openai", "azure", "workbench", "duckdb", "proxy", "dev"} <= set(extras)
    assert "dbt-duckdb" not in _package_names(extras["workbench"])
    assert "streamlit>=1.45.0,<2.0" in extras["workbench"]
    assert "setuptools>=70,<81" in extras["workbench"]
    assert "dbt-duckdb" in _package_names(extras["duckdb"])
    assert "mysql-mimic" in _package_names(extras["proxy"])
    assert "azure-identity" in _package_names(extras["azure"])


def test_workbench_requirements_reference_current_supported_extras() -> None:
    requirements = (ROOT / "src/dbt_osmosis/workbench/requirements.txt").read_text()

    assert "==1.1.5" not in requirements
    assert f"dbt-osmosis[workbench,duckdb]=={_pyproject()['project']['version']}" in requirements


def test_dev_dependency_surfaces_are_canonicalized() -> None:
    pyproject = _pyproject()
    project_dev = pyproject["project"]["optional-dependencies"]["dev"]
    dependency_group_dev = pyproject["dependency-groups"]["dev"]
    taskfile = (ROOT / "Taskfile.yml").read_text()
    pre_commit = (ROOT / ".pre-commit-config.yaml").read_text()

    assert dependency_group_dev == project_dev
    assert 'tomli>=2; python_version < "3.11"' in project_dev
    assert f"ruff=={RUFF_VERSION}" in project_dev
    assert "uv tool install 'pre-commit>3.0.0,<5'" in taskfile
    assert f"uvx ruff=={RUFF_VERSION} check" in taskfile
    assert f"uvx ruff=={RUFF_VERSION} format --preview" in taskfile
    assert f"rev: v{RUFF_VERSION}" in pre_commit


def test_fresh_clone_tasks_and_ci_install_smoke_cover_supported_extras() -> None:
    taskfile = (ROOT / "Taskfile.yml").read_text()
    workflow = (ROOT / ".github/workflows/tests.yml").read_text()

    assert "sources:\n      - .python-version" not in taskfile
    for install_target in (".", *SUPPORTED_EXTRA_INSTALLS):
        assert f'"{install_target}"' in taskfile
        assert f'"{install_target}"' in workflow
    assert 'run_smoke duckdb ".[duckdb]" "dbt-duckdb' not in taskfile
    assert 'run_smoke duckdb ".[duckdb]" "dbt-duckdb' not in workflow
    assert "streamlit_elements_fluence" in taskfile
    assert "streamlit_elements_fluence" in workflow


def test_proxy_extra_docs_are_dependency_only() -> None:
    docs = "\n".join([
        (ROOT / "README.md").read_text(),
        (ROOT / "docs/docs/intro.md").read_text(),
        (ROOT / "docs/docs/tutorial-basics/installation.md").read_text(),
    ])

    assert "dbt-osmosis[proxy]` only installs dependencies" in docs
    assert "ticket:c10proxy25" in docs
