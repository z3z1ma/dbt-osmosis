from __future__ import annotations

import typing as t
from decimal import Decimal, InvalidOperation

from dbt.contracts.graph.nodes import ResultNode

__all__ = [
    "_raw_model_version_value",
    "_version_values_match",
    "_versioned_model_yaml_view",
]


def _raw_model_version_value(value: t.Any) -> str | None:
    """Normalize only representation noise while preserving string version identity."""
    if value is None or isinstance(value, bool):
        return None

    raw_value = str(value).strip()
    if not raw_value:
        return None
    return raw_value


def _normalize_model_version_value(value: t.Any) -> str | None:
    """Normalize dbt model version values for numeric YAML/manifest equivalence checks."""
    raw_value = _raw_model_version_value(value)
    if raw_value is None:
        return None

    try:
        decimal_value = Decimal(raw_value)
    except (InvalidOperation, ValueError):
        return raw_value

    if decimal_value == decimal_value.to_integral_value():
        return str(int(decimal_value))
    return format(decimal_value.normalize(), "f")


def _can_use_numeric_version_fallback(left: t.Any, right: t.Any) -> bool:
    """Return whether non-exact version values can use numeric equivalence."""
    left_is_str = isinstance(left, str)
    right_is_str = isinstance(right, str)
    if left_is_str and right_is_str:
        return False

    for value in (left, right):
        if not isinstance(value, str):
            continue
        raw_value = _raw_model_version_value(value)
        normalized_value = _normalize_model_version_value(value)
        if raw_value is None or normalized_value is None or raw_value != normalized_value:
            return False
    return True


def _version_values_match(left: t.Any, right: t.Any) -> bool:
    """Return whether two dbt model version values identify the same version."""
    left_raw = _raw_model_version_value(left)
    right_raw = _raw_model_version_value(right)
    if left_raw is not None and left_raw == right_raw:
        return True
    if not _can_use_numeric_version_fallback(left, right):
        return False

    left_normalized = _normalize_model_version_value(left)
    right_normalized = _normalize_model_version_value(right)
    return left_normalized is not None and left_normalized == right_normalized


def _get_member_model_version(member: ResultNode) -> t.Any | None:
    """Read a dbt ModelNode version with a unique_id fallback for older shapes."""
    version = getattr(member, "version", None)
    if version is not None:
        return version

    unique_id = getattr(member, "unique_id", "")
    if not isinstance(unique_id, str):
        return None
    _, separator, suffix = unique_id.rpartition(".v")
    if separator and suffix:
        return suffix
    return None


def _versioned_model_yaml_view(
    model: t.Mapping[str, t.Any],
    member: ResultNode,
) -> dict[str, t.Any] | None:
    """Build a YAML view for the selected versioned model block.

    Version-level columns own the selected node's columns. Missing node-level
    metadata falls back explicitly to the parent model entry so properties such
    as description, meta, and tags remain visible when authored once at the
    top-level model block.
    """
    member_version = _get_member_model_version(member)
    if member_version is None:
        return None

    versions = model.get("versions", [])
    if not isinstance(versions, list):
        return None

    member_version_raw = _raw_model_version_value(member_version)
    selected_version = next(
        (
            version
            for version in versions
            if isinstance(version, t.Mapping)
            and _raw_model_version_value(version.get("v")) == member_version_raw
        ),
        None,
    )
    if selected_version is None:
        selected_version = next(
            (
                version
                for version in versions
                if isinstance(version, t.Mapping)
                and _version_values_match(version.get("v"), member_version)
            ),
            None,
        )
    if selected_version is None:
        return None

    selected = dict(selected_version)
    selected.setdefault("name", model.get("name"))
    for fallback_key in ("description", "meta", "tags"):
        if fallback_key not in selected and fallback_key in model:
            selected[fallback_key] = model[fallback_key]
    if not selected.get("description") and "description" in model:
        selected["description"] = model["description"]
    return selected
