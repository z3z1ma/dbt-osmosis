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
        """Custom string representer for ruamel.yaml with intelligent multi-line formatting.

        This representer applies different YAML scalar styles based on string content:
        - Quoted style (double quotes) for YAML boolean-like values (yes/no/on/off)
        - Folded style (>) for long single-line strings exceeding configured width
        - Literal style (|) for multi-line strings to preserve newlines
        - Plain style for all other strings

        The function prevents ambiguous YAML values from being misinterpreted as booleans
        and applies sensible formatting rules for dbt model descriptions and documentation.

        Args:
            dumper: The ruamel.yaml RoundTripDumper instance
            data: The string value to represent

        Returns:
            A YAML scalar node with appropriate style
        """
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
