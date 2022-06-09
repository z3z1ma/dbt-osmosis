import shutil
from pathlib import Path

from dbt.task.deps import DepsTask

from dbt_osmosis.core.logging import logger
from dbt_osmosis.core.osmosis import DbtOsmosis

CUSTOM_MACROS = """
{% macro dbt_osmosis_compare(a_query, b_query, primary_key=none) %}
with a as (
    {{ a_query }}
),
b as (
    {{ b_query }}
),
a_except_b as (
    select * from a
    {{ dbt_utils.except() }}
    select * from b
),
b_except_a as (
    select * from b
    {{ dbt_utils.except() }}
    select * from a
),
all_records as (
    select
        'REMOVED' as _diff,
        *,
        true as in_a,
        false as in_b
    from a_except_b
    union all
    select
        'ADDED' as _diff,
        *,
        false as in_a,
        true as in_b
    from b_except_a
)
select * 
from all_records
order by {{ primary_key ~ ", " if primary_key is not none }} in_a desc, in_b desc
{% endmacro %}
"""


def inject_audit_helper_via_deps(dbt: DbtOsmosis) -> None:
    packages_path = Path(dbt.project_root) / "packages.yml"
    data = dbt.yaml.load(packages_path)
    if not next((p for p in data["packages"] if p["package"] == "dbt-labs/audit_helper"), None):
        data["packages"].append(
            {"package": "dbt-labs/audit_helper", "version": [">=0.5.0", "<0.6.0"]}
        )
        dbt.yaml.dump(data, packages_path)
        DepsTask.run(dbt)


def inject_audit_helper_via_copy(dbt: DbtOsmosis) -> None:
    logger().info("Injecting macros, please wait...")
    macro_path = Path(dbt.project_root) / dbt.config.macro_paths[0] / "utilities"
    audit_copy = Path(__file__).parent / "macro_reqs"

    audit_helper_container = macro_path / "audit_helper"

    macro_path.mkdir(parents=True, exist_ok=True)

    for spec in dbt.config.packages:
        if spec.package.split("/")[-1] == "audit_helper":
            break
    else:
        shutil.copytree(audit_copy, audit_helper_container)


def inject_macros(dbt: DbtOsmosis) -> None:
    logger().info("Injecting macros, please wait...")
    macro_path = Path(dbt.project_root) / dbt.config.macro_paths[0] / "utilities"

    custom_macro_container = macro_path / "dbt_osmosis_macros.sql"

    macro_path.mkdir(parents=True, exist_ok=True)

    if not custom_macro_container.exists():
        custom_macro_container.write_text(CUSTOM_MACROS)
