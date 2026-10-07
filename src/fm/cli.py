"""``fm`` command line: a typer root that auto-discovers command modules from ``fm.commands``.

Each public module in ``fm.commands`` exposes ``register(root)`` and attaches its commands there (the convention is
documented in ``fm.commands``). This file is never edited to add a command. Two modules claiming one top-level
command or group name is a startup error naming both; typer alone would let the later module shadow the earlier
one silently. ``build_app`` takes the package as a parameter so discovery can be tested against a synthetic package.
"""

from __future__ import annotations

import contextlib
import importlib
import pkgutil
import sys
from collections.abc import Iterable, Iterator
from types import ModuleType

import typer
from typer.main import get_command_name, solve_typer_info_defaults
from typer.models import CommandInfo, TyperInfo

from fm import commands as default_commands

REGISTER_HOOK = "register"
HELP = "Personal assistant GM for ESPN fantasy football (NFL) and basketball (NBA)."


def iter_command_modules(package: ModuleType = default_commands) -> Iterator[ModuleType]:
    """Import and yield every public module of ``package`` in sorted name order; ``_private`` modules are skipped."""
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        yield importlib.import_module(f"{package.__name__}.{info.name}")


def register_commands(root: typer.Typer, package: ModuleType = default_commands) -> list[str]:
    """Call ``register(root)`` on every public module of ``package``. Returns the registered module names.

    Raises ``RuntimeError`` when a module lacks the hook or registers a top-level command or group name that
    another module (or the module itself) already registered.
    """
    registered: list[str] = []
    owners: dict[str, str] = {}  # top-level command or group name -> module that registered it
    for module in iter_command_modules(package):
        hook = getattr(module, REGISTER_HOOK, None)
        if not callable(hook):
            raise RuntimeError(
                f"{module.__name__} has no {REGISTER_HOOK}(root) hook; every public module in {package.__name__} "
                "must define one (see the fm.commands docstring)"
            )
        commands_before, groups_before = len(root.registered_commands), len(root.registered_groups)
        hook(root)
        for name in _command_names(root.registered_commands[commands_before:], root.registered_groups[groups_before:]):
            owner = owners.get(name)
            if owner is not None:
                raise RuntimeError(
                    f"command {name!r} is registered twice, by {owner} and by {module.__name__}; top-level command "
                    f"and group names must be unique across {package.__name__}"
                )
            owners[name] = module.__name__
        registered.append(module.__name__.rsplit(".", 1)[-1])
    return registered


def _command_names(commands: Iterable[CommandInfo], groups: Iterable[TyperInfo]) -> list[str]:
    """The top-level names these registrations add, resolved the way typer does when it builds the click group."""
    names = [info.name or get_command_name(getattr(info.callback, "__name__", "")) for info in commands]
    for info in groups:
        solved = solve_typer_info_defaults(info)
        if solved.name:
            names.append(solved.name)
        elif solved.typer_instance is not None:  # typer merges an unnamed sub-app's commands into the parent
            sub = solved.typer_instance
            names.extend(_command_names(sub.registered_commands, sub.registered_groups))
    return names


def _utf8_console() -> None:
    """Windows consoles default to cp1252 (and a POSIX ``LANG=C`` to ASCII); switch to UTF-8 with replacement so
    player names never raise."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(ValueError, OSError, AttributeError):
            reconfigure(encoding="utf-8", errors="replace")


def _root() -> None:
    _utf8_console()


def build_app(package: ModuleType = default_commands) -> typer.Typer:
    """Create the root app and register every command module of ``package`` on it."""
    root = typer.Typer(
        name="fm",
        help=HELP,
        no_args_is_help=True,
        add_completion=False,
        pretty_exceptions_enable=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )
    # A callback keeps the root a command group even while it has a single command.
    root.callback(help=HELP)(_root)
    register_commands(root, package)
    return root


app = build_app()
