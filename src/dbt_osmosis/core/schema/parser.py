import re
import typing as t
from types import MappingProxyType

import ruamel.yaml

import dbt_osmosis.core.logger as logger

__all__ = [
    "create_yaml_instance",
    "OsmosisYAML",
]


def _filter_yaml_content(data: dict) -> dict:
    """Filters a parsed YAML dictionary to only include keys relevant to dbt-osmosis.

    This prevents the tool from processing or being aware of semantic_models, macros, etc.
    """
    allowed_keys = {"version", "models", "sources", "seeds", "unit_tests"}

    # Create a new dictionary containing only the allowed keys from the parsed file
    filtered_data = {key: value for key, value in data.items() if key in allowed_keys}

    # Log which keys were ignored for debugging and transparency
    ignored_keys = set(data.keys()) - allowed_keys
    if ignored_keys:
        logger.debug(
            ":magnifying_glass_left: Parser ignoring irrelevant top-level keys in YAML: %s",
            ignored_keys,
        )

    return filtered_data


class OsmosisYAML(ruamel.yaml.YAML):
    """A custom ruamel.yaml.YAML subclass that filters loaded data.

    This class extends ruamel.yaml to automatically filter out YAML keys that
    are not relevant to dbt-osmosis, such as semantic_models, macros, etc.
    This prevents the tool from accidentally processing or modifying content
    it shouldn't touch.

    The following keys are preserved: version, models, sources, seeds, unit_tests.
    """

    def load(self, stream: t.Any) -> t.Any:
        """Loads a YAML stream and filters it for relevant dbt-osmosis content.

        Args:
            stream: The YAML stream to load (file-like object or string)

        Returns:
            The filtered YAML content, with only relevant keys preserved
        """
        # First, parse the YAML file into a standard dictionary using the parent method
        raw_data = super().load(stream)

        # If the parsed data is a dictionary, pass it through our content filter
        if isinstance(raw_data, dict):
            return _filter_yaml_content(raw_data)

        # If it's not a dictionary (e.g., an empty file parsing to None), return it as-is
        return raw_data


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
    y = OsmosisYAML()
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

    def mapping_proxy_representer(
        dumper: ruamel.yaml.RoundTripDumper, data: MappingProxyType
    ) -> t.Any:
        """Representer for MappingProxyType to allow dumping read-only dicts.

        MappingProxyType is used internally by dbt-osmosis to provide read-only
        views of YAML data (e.g., from _get_node_yaml()). This representer converts
        it to a regular dict for YAML serialization.
        """
        return dumper.represent_mapping(
            "tag:yaml.org,2002:map",
            dict(data),
        )

    y.representer.add_representer(str, str_representer)
    y.representer.add_representer(MappingProxyType, mapping_proxy_representer)

    logger.debug(":notebook: YAML instance created => %s", y)
    return y
