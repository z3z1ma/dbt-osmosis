#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dbt-osmosis-demo.XXXXXX")"
DBT_OSMOSIS_BIN="${DBT_OSMOSIS_BIN:-dbt-osmosis}"
PYTHON_BIN="${PYTHON:-python}"

cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

repo_root_test_db_existed=0
source_target_existed=0
if [[ -e "${REPO_ROOT}/test.db" ]]; then
  repo_root_test_db_existed=1
fi
if [[ -e "${SCRIPT_DIR}/target" ]]; then
  source_target_existed=1
fi

PROJECT_DIR="$(${PYTHON_BIN} - "${REPO_ROOT}" "${SCRIPT_DIR}" "${TEMP_DIR}" <<'PY'
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
source_dir = Path(sys.argv[2])
temp_dir = Path(sys.argv[3])
support_path = repo_root / "tests" / "support.py"

spec = importlib.util.spec_from_file_location("dbt_osmosis_test_support", support_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load fixture support from {support_path}")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)

print(support.create_temp_project_copy(source_dir, temp_dir))
PY
)"

common_options=(
  --project-dir "${PROJECT_DIR}"
  --profiles-dir "${PROJECT_DIR}"
  --target test
)

command -v "${DBT_OSMOSIS_BIN}"
"${DBT_OSMOSIS_BIN}" --version
"${DBT_OSMOSIS_BIN}" yaml organize --auto-apply "${common_options[@]}"
"${DBT_OSMOSIS_BIN}" yaml document "${common_options[@]}"
"${DBT_OSMOSIS_BIN}" yaml refactor --auto-apply "${common_options[@]}"
"${DBT_OSMOSIS_BIN}" yaml --help >/dev/null

if [[ ${repo_root_test_db_existed} -eq 0 && -e "${REPO_ROOT}/test.db" ]]; then
  echo "Integration smoke created repo-root test.db" >&2
  exit 1
fi
if [[ ${source_target_existed} -eq 0 && -e "${SCRIPT_DIR}/target" ]]; then
  echo "Integration smoke created source demo_duckdb/target" >&2
  exit 1
fi

echo "All dbt-osmosis yaml integration tests passed against ${PROJECT_DIR}!"
