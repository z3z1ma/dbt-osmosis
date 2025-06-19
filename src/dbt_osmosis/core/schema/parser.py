import re
import typing as t

import ruamel.yaml

import dbt_osmosis.core.logger as logger

__all__ = [
    "create_yaml_instance",
]


def create_yaml_instance(
    indent_mapping: int = 2,
    indent_sequence: int = 4,
    indent_offset: int = 2,
    width: int = 100,
    preserve_quotes: bool = False,
    default_flow_style: bool = False,
    encoding: str = "utf-8",
) -> ruamel.yaml.YAML:
    """Returns a ruamel.yaml.YAML instance configured with the provided settings."""
    logger.debug(":notebook: Creating ruamel.yaml.YAML instance with custom formatting.")
    y = ruamel.yaml.YAML()
    y.indent(mapping=indent_mapping, sequence=indent_sequence, offset=indent_offset)
    y.width = width
    y.preserve_quotes = preserve_quotes
    y.default_flow_style = default_flow_style
    y.encoding = encoding

    def str_representer(dumper: ruamel.yaml.RoundTripDumper, data: str) -> t.Any:
        # https://github.com/commx/ruamel-yaml/blob/280677cf647912c599d8886000020d6ffbbb4216/resolver.py#L32
        if re.match(r"^(y|Y|yes|Yes|YES|n|N|no|No|NO|on|On|ON|off|Off|OFF)$", data):
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
        newlines = len(data.splitlines())
        if newlines == 1 and len(data) > width - len(f"description{y.prefix_colon}: "):
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")
        if newlines > 1:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    y.representer.add_representer(str, str_representer)

    logger.debug(":notebook: YAML instance created => %s", y)
    return y
