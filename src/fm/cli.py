"""``fm`` command line: a typer root that auto-discovers command modules from ``fm.commands``.

Each public module in ``fm.commands`` exposes ``register(root)`` and attaches its commands there (the convention is
documented in ``fm.commands``). This file is never edited to add a command. ``build_app`` takes the package as a
parameter so discovery can be tested against a synthetic package.
"""

from __future__ import annotations

import contextlib
import importlib
import pkgutil
import sys
from collections.abc import Iterator
from types import ModuleType

import typer

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
    """Call ``register(root)`` on every public module of ``package``. Returns the registered module names."""
    registered: list[str] = []
    for module in iter_command_modules(package):
        hook = getattr(module, REGISTER_HOOK, None)
        if not callable(hook):
            raise RuntimeError(
                f"{module.__name__} has no {REGISTER_HOOK}(root) hook; every public module in {package.__name__} "
                "must define one (see the fm.commands docstring)"
            )
        hook(root)
        registered.append(module.__name__.rsplit(".", 1)[-1])
    return registered


def _utf8_console() -> None:
    """Windows consoles default to cp1252; switch to UTF-8 with replacement so player names never raise."""
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
