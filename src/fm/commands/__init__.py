"""CLI command modules, auto-discovered by ``fm.cli``. Nobody edits ``cli.py`` to add a command.

Convention
----------
- One module per command group, named after what it exposes: ``sync.py``, ``proposals.py``, ``config_cmd.py``.
- Every public module defines ``register(root: typer.Typer) -> None`` and attaches its commands to ``root`` there.
- Modules whose name starts with ``_`` are private helpers (shared options, loaders) and are skipped by discovery.
- Discovery imports public modules in sorted name order and calls ``register`` on each. A public module without
  ``register`` is a startup error, so a typo surfaces in ``fm --help`` and in ``tests/test_cli.py``.
- A module may register several top-level commands (``advise.py`` adds ``status``, ``lineup`` and ``waivers``).
  Keep modules thin: parse options, call into ``fm.*``, render with ``fm.render``.

Top-level command (``fm sync``)::

    from typing import Annotated

    import typer


    def sync(league: Annotated[str | None, typer.Option(help="League key from config.toml.")] = None) -> None:
        \"\"\"Pull league state and sources into the store.\"\"\"
        ...


    def register(root: typer.Typer) -> None:
        root.command("sync")(sync)

Command group (``fm proposals list|approve|reject``)::

    import typer

    app = typer.Typer(help="Review and decide proposals.", no_args_is_help=True)


    @app.command("list")
    def list_() -> None:
        \"\"\"Show open proposals.\"\"\"
        ...


    def register(root: typer.Typer) -> None:
        root.add_typer(app, name="proposals")

``fm.commands.version`` is the smallest complete example.
"""
