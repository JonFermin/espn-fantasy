"""``fm config check``: validate ``config.toml`` and ``.env`` and show what was loaded."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from fm.config import OPTIONAL_SECRET_ENV_VARS, Config, ConfigError, config_paths, load_config

app = typer.Typer(help="Inspect and validate the configuration.", no_args_is_help=True)

PathOption = Annotated[
    Path | None,
    typer.Option(
        "--path",
        "-p",
        help="config.toml to check; defaults to $FM_CONFIG_DIR/config.toml. Its .env is read from the same directory.",
    ),
]


@app.command("check")
def check(path: PathOption = None) -> None:
    """Validate config.toml and .env, print a summary, and exit 1 when they are invalid.

    Missing secrets are reported but do not fail the check: the read-only commands work without them.
    """
    config_path, env_path = config_paths(path)
    try:
        config = load_config(config_path, env_path=env_path)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    for line in _summary(config, config_path, env_path):
        typer.echo(line)


def _summary(config: Config, config_path: Path, env_path: Path) -> list[str]:
    lines = [f"config: {config_path}", f"env:    {env_path}{'' if env_path.is_file() else ' (not found)'}"]
    lines.append(f"leagues ({len(config.leagues)}):")
    for league in config.leagues:
        policy = league.policy
        lines += [
            f"  {league.key}: {league.sport} ({league.game}) league {league.espn_league_id}, season {league.season}, "
            f"team {league.team_id}",
            f"    bench_inactive={policy.bench_inactive} lineup={policy.lineup} add={policy.add}"
            + (f" (min gain {policy.auto_add_min_gain:g})" if policy.add == "auto" else "")
            + f" add_drop={policy.add_drop} waiver={policy.waiver}",
            f"    max {policy.max_transactions_per_week} transactions/week, bids <= {policy.max_faab_pct_per_bid:.0%} "
            f"of FAAB, {len(policy.untouchables)} untouchable(s)",
        ]
    lines.append(f"llm:    {config.llm.model}, ${config.llm.daily_budget_usd:.2f}/day budget")
    lines.append(f"notify: {config.notify.channel}")
    lines.append("secrets:")
    for name in (*config.required_secrets(), *OPTIONAL_SECRET_ENV_VARS):
        if config.secrets.is_set(name):
            status = "set"
        else:
            status = "unset (optional)" if name in OPTIONAL_SECRET_ENV_VARS else "missing"
        lines.append(f"  {name}: {status}")
    missing = config.missing_secrets()
    verdict = f"{len(missing)} required secret(s) missing from .env" if missing else "all required secrets set"
    lines.append(f"ok: config.toml is valid; {verdict}")
    return lines


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="config")
