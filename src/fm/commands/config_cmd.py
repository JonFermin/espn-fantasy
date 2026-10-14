"""``fm config check``: validate ``config.toml`` and ``.env`` and show what was loaded; ``fm config set-preset``: the
CLI front of ``fm.config_edit.set_value`` for a league's policy preset."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from fm.config import OPTIONAL_SECRET_ENV_VARS, Config, ConfigError, League, config_paths, load_config
from fm.config_edit import preset_overrides, set_value, set_values
from fm.presets import PRESET_NAMES, preset_for

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


@app.command("set-preset")
def set_preset(
    league: Annotated[str, typer.Argument(help="League key from config.toml.", show_default=False)],
    preset: Annotated[str, typer.Argument(help=f"One of {', '.join(PRESET_NAMES)}.", show_default=False)],
    path: PathOption = None,
    reset_overrides: Annotated[
        bool,
        typer.Option(
            "--reset-overrides",
            help="Also remove the \\[league.policy] / \\[league.trade_finder] keys written in config.toml that would "
            "override the preset, so the preset decides them.",
        ),
    ] = False,
) -> None:
    """Set a league's policy preset in config.toml and show the policy that results.

    The preset fills every \\[league.policy] and \\[league.trade_finder] key the league leaves unset; a key written in the
    file still wins, and the command names any such key (or removes them with --reset-overrides). The edit is
    validated before the file changes, the replace is atomic, and the previous file is kept as config.toml.bak.
    """
    if preset not in PRESET_NAMES:
        raise typer.BadParameter(f"unknown preset {preset!r}; choose one of {', '.join(PRESET_NAMES)}")
    config_path, env_path = config_paths(path)
    removed: list[str] = []
    try:
        if reset_overrides:
            removed = preset_overrides(config_path, league, preset=preset)
            config = set_values(config_path, league, {"preset": preset, **dict.fromkeys(removed)}, env_path=env_path)
            overrides: list[str] = []
        else:
            config = set_value(config_path, league, "preset", preset, env_path=env_path)
            overrides = preset_overrides(config_path, league)
    except (ConfigError, ValueError) as exc:  # ValueError: a sport no preset knows, in a file that does not validate
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    chosen = config.league(league)
    if overrides:
        typer.echo(f"{config_path}: {league} uses the {preset} preset, with keys written in config.toml overriding it")
        typer.echo(
            f"note: these keys written in config.toml override the preset: {', '.join(overrides)}; remove them or run "
            "with --reset-overrides"
        )
    else:
        typer.echo(f"{config_path}: {league} uses the {preset} preset: {preset_for(preset, chosen.sport).description}")
    if removed:
        typer.echo(f"removed from config.toml so the preset decides them: {', '.join(removed)}")
    for line in _league_lines(chosen):
        typer.echo(line)


def _summary(config: Config, config_path: Path, env_path: Path) -> list[str]:
    lines = [f"config: {config_path}", f"env:    {env_path}{'' if env_path.is_file() else ' (not found)'}"]
    lines.append(f"leagues ({len(config.leagues)}):")
    for league in config.leagues:
        lines += _league_lines(league)
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


def _league_lines(league: League) -> list[str]:
    """The league's identity, its preset when it has one, and the policy that results."""
    policy = league.policy
    lines = [
        f"  {league.key}: {league.sport} ({league.game}) league {league.espn_league_id}, season {league.season}, "
        f"team {league.team_id}" + (f", preset {league.preset}" if league.preset else ""),
        f"    bench_inactive={policy.bench_inactive} lineup={policy.lineup} add={policy.add}"
        + (f" (min gain {policy.auto_add_min_gain:g})" if policy.add == "auto" else "")
        + f" add_drop={policy.add_drop} waiver={policy.waiver}",
        f"    max {policy.max_transactions_per_week} transactions/week, bids <= {policy.max_faab_pct_per_bid:.0%} "
        f"of FAAB, {len(policy.untouchables)} untouchable(s)",
    ]
    finder = league.trade_finder
    if finder.enabled:
        lines.append(
            f"    trade finder: drafts up to {finder.max_offers} offer(s) a period for approval when P(accept) >= "
            f"{finder.min_accept:.0%} and title odds +{finder.min_title_gain * 100:g} pts "
            f"(or +{finder.min_ros_gain:g} starter seasons)"
        )
    return lines


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="config")
