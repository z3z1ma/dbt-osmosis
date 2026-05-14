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
SUPPORT_POLICY_DOCS = (
    ROOT / "README.md",
    ROOT / "docs/docs/intro.md",
    ROOT / "docs/docs/tutorial-basics/installation.md",
)


def _pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _package_names(requirements: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].lower() for requirement in requirements
    }


def _workflow_job(workflow: str, job_id: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(job_id)}:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", workflow
    )
    assert match is not None, f"missing workflow job: {job_id}"
    return match.group("body")


def test_dbt_core_dependency_remains_open_for_future_canaries() -> None:
    dependencies = _pyproject()["project"]["dependencies"]
    dbt_core_requirements = [
        dep
        for dep in dependencies
        if re.split(r"[<>=!~;\[]", dep, maxsplit=1)[0].lower() == "dbt-core"
    ]

    assert dbt_core_requirements == ["dbt-core>=1.8"]
    assert all("<" not in requirement for requirement in dbt_core_requirements)
    assert all("~=" not in requirement for requirement in dbt_core_requirements)


def test_docs_state_audited_support_and_future_canary_policy() -> None:
    required_policy_phrases = (
        "Audited blocking support covers dbt Core 1.8.x through 1.11.x",
        "package metadata intentionally remains `dbt-core>=1.8` without an upper bound",
        "Future dbt Core minors are canary-only until explicitly audited",
        "Install a dbt adapter version that is compatible with the dbt Core runtime",
    )

    for path in SUPPORT_POLICY_DOCS:
        docs = path.read_text()
        for phrase in required_policy_phrases:
            assert phrase in docs, f"{path.relative_to(ROOT)} does not state: {phrase}"


def test_workflow_has_non_blocking_unpinned_future_dbt_canary() -> None:
    workflow = (ROOT / ".github/workflows/tests.yml").read_text()
    canary = _workflow_job(workflow, "future-dbt-canary")

    assert "workflow_dispatch:" in workflow
    assert "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'" in canary
    assert "continue-on-error: true" in canary
    assert "Canary latest dbt Core compatibility" in canary
    assert "dbt-core dbt-duckdb" in canary
    assert "dbt-core~=" not in canary
    assert "dbt-duckdb~=" not in canary
    assert "uv --no-config pip check" in canary
    assert "dbt parse" in canary
    assert "pytest -q" in canary
    assert "dbt-osmosis --help" in canary


def test_pytest_and_ci_surface_dbt_deprecation_warnings() -> None:
    pyproject = _pyproject()
    filterwarnings = pyproject["tool"]["pytest"]["ini_options"]["filterwarnings"]
    workflow = (ROOT / ".github/workflows/tests.yml").read_text()

    assert "ignore::DeprecationWarning" not in filterwarnings
    assert "default::DeprecationWarning:dbt" in filterwarnings
    assert "default::DeprecationWarning:dbt_osmosis" in filterwarnings
    assert "PYTHONWARNINGS" in workflow
    assert "default::DeprecationWarning:dbt" in workflow
    assert "default::DeprecationWarning:dbt_osmosis" in workflow


def test_base_dependencies_and_optional_extras_are_intentional() -> None:
    pyproject = _pyproject()
    base_dependencies = _package_names(pyproject["project"]["dependencies"])
    extras = pyproject["project"]["optional-dependencies"]

    assert "sqlglot" in base_dependencies
    assert "pyyaml" in base_dependencies
    assert "mysql-mimic" not in base_dependencies

    assert {"openai", "azure", "workbench", "duckdb", "proxy", "dev"} <= set(extras)
    workbench_packages = _package_names(extras["workbench"])
    assert "dbt-duckdb" not in _package_names(extras["workbench"])
    assert "openai" not in workbench_packages
    assert "ydata-profiling" in workbench_packages
    assert "ipython" in workbench_packages
    assert "streamlit>=1.45.0,<2.0" in extras["workbench"]
    assert "setuptools>=70,<83" in extras["workbench"]
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
    assert "experimental opt-in SQL proxy runtime" in docs
    assert "does not start a proxy server" in docs
    assert "configure authentication, TLS, or listen/bind settings" in docs
    assert "local-only experiment with `mysql-mimic` defaults" in docs
    assert "do not expose it to untrusted networks" in docs
    assert "proxy comment middleware is in-memory only" in docs
    assert "ticket:c10proxy25" in docs
