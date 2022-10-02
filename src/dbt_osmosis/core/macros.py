from pathlib import Path

from dbt_osmosis.core.log_controller import logger
from dbt_osmosis.core.osmosis import DbtProject


def inject_macros(dbt: DbtProject) -> None:
    logger().info("Injecting macros, please wait...")
    macro_overrides = {}
    for node in dbt.macro_parser.parse_remote(
        (Path(__file__).parent / "audit_macros.jinja2").read_text()
    ):
        macro_overrides[node.unique_id] = node
    dbt.dbt.macros.update(macro_overrides)
