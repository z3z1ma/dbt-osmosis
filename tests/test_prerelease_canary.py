"""Guard the version-selection boundary against real PyPI prerelease traps."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/select_dbt_prerelease.py"
spec = importlib.util.spec_from_file_location("select_dbt_prerelease", SCRIPT)
assert spec is not None and spec.loader is not None
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


def metadata(*versions: str) -> dict:
    return {"releases": {version: [{"yanked": False}] for version in versions}}


def test_selects_beta_instead_of_yanked_rc_or_rust_preview() -> None:
    payload = metadata("1.12.7", "1.13.0b1", "2.0.0rc10", "1.12.0rc3")
    payload["releases"]["1.13.0rc1"] = [{"yanked": True}]
    payload["releases"]["1.14.0a1"] = []
    assert selector.select_prerelease(payload) == "1.13.0b1"


def test_orders_versions_semantically_and_accepts_one_non_yanked_file() -> None:
    payload = metadata("1.9.9", "1.10.0b2", "1.10.0b10")
    payload["releases"]["1.10.0b10"].append({"yanked": True})
    assert selector.select_prerelease(payload) == "1.10.0b10"


@pytest.mark.parametrize("versions", [("1.13.0", "1.13.0rc1"), ("1.12.7",), ("1.12.7", "2.0.0rc1")])
def test_no_upcoming_preview_is_an_explicit_skip(versions: tuple[str, ...]) -> None:
    assert selector.select_prerelease(metadata(*versions)) is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"releases": []},
        {"releases": {}},
        metadata("2.0.0rc1"),
        {"releases": {"1.12.7": [{}]}},
    ],
)
def test_invalid_or_incomplete_metadata_is_not_a_skip(payload: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        selector.select_prerelease(payload)


def test_future_major_epoch_and_local_versions_do_not_select_wrong_runtime() -> None:
    assert selector.select_prerelease(metadata("1.12.7", "2!1.14.0a1", "1.14.0a1+local")) is None
