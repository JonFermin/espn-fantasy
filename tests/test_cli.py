"""``fm --help``, the console-script entry point, and command discovery."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType

import pytest
import typer
from typer.testing import CliRunner

import fm.cli
import fm.commands
from fm.cli import app, build_app, register_commands

runner = CliRunner()


def test_help_exits_zero_and_lists_discovered_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "version" in result.output


def test_short_help_flag() -> None:
    result = runner.invoke(app, ["-h"])
    assert result.exit_code == 0, result.output


def test_version_command_runs() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("fm ")
    assert "unknown" not in result.output  # the editable install has metadata


def test_console_script_points_at_the_app() -> None:
    (entry,) = entry_points(group="console_scripts", name="fm")
    assert entry.value == "fm.cli:app"
    assert entry.load() is fm.cli.app


def test_every_public_command_module_is_registered() -> None:
    public = sorted(m.name for m in pkgutil.iter_modules(fm.commands.__path__) if not m.name.startswith("_"))
    assert "version" in public
    assert register_commands(typer.Typer(), fm.commands) == public


def _make_package(root: Path, name: str, modules: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    package = root / name
    package.mkdir()
    (package / "__init__.py").write_text('"""Synthetic command package."""\n', encoding="utf-8")
    for module, source in modules.items():
        (package / f"{module}.py").write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    for key in [k for k in sys.modules if k == name or k.startswith(f"{name}.")]:
        monkeypatch.delitem(sys.modules, key)
    return importlib.import_module(name)


GOOD = """
import typer


def hello() -> None:
    typer.echo("hello")


def register(root: typer.Typer) -> None:
    root.command("hello")(hello)
"""

GROUP = """
import typer

app = typer.Typer(help="A group.", no_args_is_help=True)


@app.command("ping")
def ping() -> None:
    typer.echo("pong")


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="grp")
"""

PRIVATE = """
raise AssertionError("private helper modules must not be imported by discovery")
"""

NO_HOOK = """
import typer

def hello() -> None:
    typer.echo("hello")
"""


def test_discovery_registers_commands_and_groups_and_skips_private_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, "fake_cmds_ok", {"_shared": PRIVATE, "grp": GROUP, "zeta": GOOD}, monkeypatch)
    root = typer.Typer()
    assert register_commands(root, package) == ["grp", "zeta"]
    assert "fake_cmds_ok._shared" not in sys.modules

    test_app = build_app(package)
    assert runner.invoke(test_app, ["hello"]).output.strip() == "hello"
    assert runner.invoke(test_app, ["grp", "ping"]).output.strip() == "pong"
    help_out = runner.invoke(test_app, ["--help"]).output
    assert "hello" in help_out and "grp" in help_out


def test_public_module_without_register_hook_is_a_startup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, "fake_cmds_bad", {"broken": NO_HOOK}, monkeypatch)
    with pytest.raises(RuntimeError, match=r"fake_cmds_bad\.broken has no register\(root\) hook"):
        build_app(package)
