import re
import typing as t

import ruamel.yaml

import dbt_osmosis.core.logger as logger

__all__ = [
    "create_yaml_instance",
]

def _filter_yaml_content(data: dict) -> dict:
    """
    Filters a parsed YAML dictionary to only include keys relevant to dbt-osmosis.
    This prevents the tool from processing or being aware of semantic_models, macros, etc.
    """
    allowed_keys = {"version", "models", "sources", "seeds"}
    
    # Create a new dictionary containing only the allowed keys from the parsed file
    filtered_data = {key: value for key, value in data.items() if key in allowed_keys}

    # Log which keys were ignored for debugging and transparency
    ignored_keys = set(data.keys()) - allowed_keys
    if ignored_keys:
        logger.debug(
            f":magnifying_glass_left: Parser ignoring irrelevant top-level keys in YAML: {ignored_keys}"
        )
        
    return filtered_data


class OsmosisYAML(ruamel.yaml.YAML):
    """A custom ruamel.yaml.YAML subclass that filters loaded data."""
    def load(self, stream: t.Any) -> t.Any:
        """Loads a YAML stream and filters it for relevant dbt-osmosis content."""
        # First, parse the YAML file into a standard dictionary using the parent method
        raw_data = super().load(stream)
        
        # If the parsed data is a dictionary, pass it through our content filter
        if isinstance(raw_data, dict):
            return _filter_yaml_content(raw_data)
        
        # If it's not a dictionary (e.g., an empty file parsing to None), return it as-is
        return raw_data


def create_yaml_instance() -> ruamel.yaml.YAML:
    """Creates a ruamel.yaml.YAML instance with project-consistent settings."""
    # Use the custom OsmosisYAML class, initialized in round-trip mode.
    yaml = OsmosisYAML(typ="rt")

    # Apply our desired formatting
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    
    # These are reasonable defaults that were likely intended in the original code
    yaml.width = 800
    yaml.default_flow_style = False
    
    return yaml