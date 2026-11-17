"""Select an upcoming Python Core release without opting all dependencies into previews."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen

from packaging.version import InvalidVersion, Version

PYPI_URL = "https://pypi.org/pypi/dbt-core/json"


def select_prerelease(metadata: object) -> str | None:
    """Return the newest non-yanked Core 1.x preview newer than every stable 1.x release."""
    if not isinstance(metadata, dict) or not isinstance(metadata.get("releases"), dict):
        raise TypeError("PyPI response must contain a releases object")
    versions: list[Version] = []
    for name, files in metadata["releases"].items():
        if not isinstance(files, list) or any(
            not isinstance(file, dict) or not isinstance(file.get("yanked"), bool) for file in files
        ):
            raise ValueError(f"Invalid PyPI file metadata for {name}")
        if not files or all(file["yanked"] for file in files):
            continue
        try:
            version = Version(name)
        except InvalidVersion:
            continue
        if version.major == 1 and version.epoch == 0 and version.local is None:
            versions.append(version)
    stable = [version for version in versions if not version.is_prerelease]
    if not stable:
        raise ValueError("No usable stable Python Core 1.x release found in PyPI metadata")
    previews = [version for version in versions if version.is_prerelease and version > max(stable)]
    return str(max(previews)) if previews else None


def main() -> None:
    # HTTPS failures and malformed metadata must fail CI, not masquerade as no preview.
    with urlopen(PYPI_URL, timeout=30) as response:
        candidate = select_prerelease(json.load(response))
    if candidate:
        print(candidate)
    else:
        print("No upcoming non-yanked Python Core 1.x prerelease; audit skipped.", file=sys.stderr)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write(f"version={candidate or ''}\n")


if __name__ == "__main__":
    main()
