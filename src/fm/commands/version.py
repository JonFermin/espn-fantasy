"""``fm version``: print the installed espn-fantasy version. Also the reference command module."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

import typer

DISTRIBUTION = "espn-fantasy"


def installed_version() -> str:
    """Version of the installed ``espn-fantasy`` distribution, or ``0+unknown`` when it is not installed."""
    try:
        return distribution_version(DISTRIBUTION)
    except PackageNotFoundError:
        return "0+unknown"


def version() -> None:
    """Print the installed version."""
    typer.echo(f"fm {installed_version()}")


def register(root: typer.Typer) -> None:
    root.command("version")(version)
