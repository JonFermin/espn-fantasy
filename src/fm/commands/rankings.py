"""``fm rankings [--league KEY] [--out DIR] [--fixtures DIR]``: the rest-of-season rankings sheet (ROADMAP #35).

One CSV per league, every rostered player and free agent valued for the rest of the season in that league's own
scoring (:mod:`fm.decide.rankings`): the sanity check for a waiver or trade call. It reads what the last ``fm sync``
stored, plus the pro schedule the way ``fm lineup`` and ``fm waivers`` get it (``--schedule FILE``, else the newest
capture under the cache dir, else a live read through the ESPN session; without one byes and off days are not counted
and the NBA cannot be valued), and proposes nothing and writes nothing to ESPN. The sheets go to ``--out`` (default
``<config dir>/rankings``, ``fm.paths.config_dir``), named ``rankings_<league key>.csv``. ``--as-of`` pins the clock the
availability model reads. Players the last sync's pool put on waivers are marked ``waivers``, not ``FA``.

Fixture home: ``FM_CONFIG_DIR=tests/fixtures/home FM_CACHE_DIR=tests/fixtures/home/cache uv run fm rankings --as-of
2026-10-04T15:00Z --out DIR`` (``tests/fixtures/home/README.md``).

``--fixtures DIR`` works offline and without ``fm login``, ``config.toml`` or a synced store: it serves recorded views
through the same ``httpx.MockTransport`` ``fm execute --dry-run --fixtures`` uses (:class:`fm.commands.execute.
RecordedViews`), runs the real sync job over them into a throwaway in-memory store, and ranks from that. ``DIR`` is a
folder of ``<view>.json`` files (``mSettings``, ``mTeam+mStandings``, ``mRoster``, ``kona_player_info``,
``kona_playercard``, ``proTeamSchedules_wl``) or holds one per game: ``DIR/ffl`` and ``DIR/fba``, or ``DIR/real/ffl``
and ``DIR/real/fba`` (``tests/fixtures/espn`` is the latter). The league key is the sport (``nfl``, ``nba``), our team
is ``--team`` or else the lowest team id with a recorded roster (the scrubbed real fixtures' own team), and the NBA's
projection blend uses ESPN's line alone, since DARKO needs the network. A game the folder has no views for is skipped
with a message, as is a league whose views cannot be valued; the command exits 1 only when no sheet was written.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn

import httpx
import typer

from fm import paths
from fm.commands.advise import _config, _parse_as_of, _schedule, _select
from fm.commands.execute import RecordedViews
from fm.config import ESPN_GAME, League, Sport
from fm.decide.rankings import Rankings, RankingsError, rank_league, write_csv
from fm.decide.waivers import WireStatus, load_wire
from fm.espn.client import EspnClient, EspnClientError
from fm.espn.ids import Game
from fm.espn.models import ProSchedule
from fm.espn.settings import LeagueSettings, load_league_settings
from fm.jobs.sync import SyncError, sync_league
from fm.model.ids_nba import NBA_SPORT
from fm.model.projections import ESPN, ProjectionSourceRegistry
from fm.render import columns, points, stamp, warning_lines
from fm.store import LeagueRow, Store

LeagueOption = Annotated[
    list[str] | None,
    typer.Option("--league", "-l", help="Only these league keys (repeatable). Default: every league."),
]
OutOption = Annotated[
    Path | None,
    typer.Option(
        "--out",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Folder for the CSV sheets (created if missing). Default: <config dir>/rankings.",
    ),
]
FixturesOption = Annotated[
    Path | None,
    typer.Option(
        "--fixtures",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Rank from recorded views in this folder (<view>.json files, or ffl/ and fba/ folders of them, such as "
        "tests/fixtures/espn) instead of the synced store. Offline; needs no fm login, config or sync.",
    ),
]
TeamOption = Annotated[
    int | None,
    typer.Option(
        "--team", min=1, help="With --fixtures: our team's id (default: the lowest id with a recorded roster)."
    ),
]

AsOfOption = Annotated[
    str | None,
    typer.Option(
        "--as-of",
        help="Pin the clock (ISO 8601, e.g. 2026-10-04T15:00Z; a naive time is UTC) for players' availability. "
        "Default: now.",
    ),
]
ScheduleOption = Annotated[
    Path | None,
    typer.Option(
        "--schedule",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="A recorded proTeamSchedules_wl view to use as the pro schedule instead of the cache or ESPN (synced "
        "store only; --fixtures reads its own).",
    ),
]

SETTINGS_VIEW = "mSettings.json"
PREVIEW = 5
"""Players printed per sheet, best first."""
GAME_SPORT: dict[Game, Sport] = {Game(game): sport for sport, game in ESPN_GAME.items()}


@dataclass(frozen=True, slots=True)
class Outcome:
    """One league's result: its sheet, or why there is none, and notes from getting there."""

    key: str
    rankings: Rankings | None = None
    skipped: str = ""
    notes: tuple[str, ...] = ()


def rankings(
    league: LeagueOption = None,
    out: OutOption = None,
    fixtures: FixturesOption = None,
    team: TeamOption = None,
    as_of: AsOfOption = None,
    schedule: ScheduleOption = None,
) -> None:
    """Write a rest-of-season rankings CSV per league: rostered players and free agents, in the league's own scoring."""
    if fixtures is None and team is not None:
        _fail("--team applies only with --fixtures")
    if fixtures is not None and schedule is not None:
        _fail("--schedule applies to the synced store; --fixtures reads proTeamSchedules_wl from its own folder")
    now = _parse_as_of(as_of)
    if fixtures is not None:
        typer.echo(f"note: reading recorded views from {fixtures}; offline, nothing is sent or stored")
        outcomes = fixture_outcomes(fixtures, keys=league, team=team, now=now)
    else:
        outcomes = store_outcomes(keys=league, now=now, path=schedule)
    directory = out if out is not None else paths.config_dir() / "rankings"
    written = 0
    for outcome in outcomes:
        for note in outcome.notes:
            typer.echo(f"note: {outcome.key}: {note}")
        if outcome.rankings is None:
            typer.echo(f"skipped {outcome.key}: {outcome.skipped}")
            continue
        path = write_csv(outcome.rankings, directory)
        written += 1
        for line in sheet_lines(outcome.rankings, path, as_of=now):
            typer.echo(line)
    if not written:
        _fail("no rankings sheet was written")


def sheet_lines(sheet: Rankings, path: Path, *, as_of: datetime) -> list[str]:
    """What the command prints for one sheet: the summary, the file, the best few, the replacement level, warnings."""
    lines = [f"{sheet.describe()}, as of {stamp(as_of)}", f"  wrote {path}"]
    rows = [
        (
            f"{player.rank}.",
            player.name,
            "/".join(player.positions) or "?",
            player.pro_team or "-",
            player.owner,
            points(player.value),
            points(player.vor),
        )
        for player in sheet.ranked[:PREVIEW]
    ]
    if rows:
        header = ("", "player", "pos", "team", "owner", sheet.metric.unit, "vor")
        lines.extend(columns([header, *rows], align=("<", "<", "<", "<", "<", ">", ">")))
    lines.extend(f"  replacement {line}" for line in sheet.replacement)
    lines.extend(warning_lines(sheet.warnings))
    return lines


# --- the synced store -------------------------------------------------------------------------------------------------


def store_outcomes(*, keys: list[str] | None, now: datetime, path: Path | None = None) -> list[Outcome]:
    """Rank the configured leagues (or ``keys``) from the state database. The pro schedule is ``path``, else the
    newest capture under the cache dir, else a live read, as ``fm lineup`` and ``fm waivers`` get it."""
    leagues = _select(_config(), keys)
    outcomes: list[Outcome] = []
    with Store.open() as store:
        for league in leagues:
            row = store.leagues.by_key(league.key)
            if row is None:
                outcomes.append(Outcome(league.key, skipped="not synced; run fm sync"))
                continue
            loaded = _schedule(store, league, path=path)
            notes = (*loaded.warnings, *(() if loaded.schedule is not None else (loaded.note,)))
            try:
                sheet = rank_league(store, row, now=now, schedule=loaded.schedule, waivers=_on_waivers(store, row))
            except RankingsError as exc:
                outcomes.append(Outcome(league.key, skipped=str(exc), notes=notes))
            else:
                outcomes.append(Outcome(league.key, sheet, notes=notes))
    return outcomes


def _on_waivers(store: Store, row: LeagueRow) -> frozenset[int]:
    """ESPN ids the last sync's pool says are on waivers; none when the pool pages are not in the cache."""
    wire = load_wire(store, row)
    return frozenset(espn_id for espn_id, entry in wire.entries.items() if entry.status is WireStatus.WAIVERS)


# --- recorded views (--fixtures) --------------------------------------------------------------------------------------


def recorded_leagues(directory: Path) -> dict[Game, Path]:
    """The folders of recorded views under ``directory`` by game: ``directory`` itself when it holds ``mSettings.json``
    (its game read from the settings), and ``ffl`` and ``fba`` folders in it or in its ``real`` folder. The first folder
    found for a game wins."""
    found: dict[Game, Path] = {}
    candidates = [
        directory,
        *(directory / game.value for game in Game),
        *(directory / "real" / game.value for game in Game),
    ]
    for folder in candidates:
        if not (folder / SETTINGS_VIEW).is_file():
            continue
        try:
            game = load_league_settings(folder / SETTINGS_VIEW).game
        except (OSError, ValueError):
            continue
        found.setdefault(game, folder)
    return found


def fixture_outcomes(directory: Path, *, keys: list[str] | None, team: int | None, now: datetime) -> list[Outcome]:
    """Rank each league the recorded views under ``directory`` describe, through a throwaway in-memory store."""
    folders = recorded_leagues(directory)
    wanted = None if keys is None else set(keys)
    outcomes: list[Outcome] = []
    with Store.open(":memory:") as store:
        for game, sport in GAME_SPORT.items():
            if wanted is not None and sport not in wanted:
                continue
            folder = folders.get(game)
            if folder is None:
                outcomes.append(
                    Outcome(sport, skipped=f"no recorded {game.value} views ({SETTINGS_VIEW}) under {directory}")
                )
                continue
            outcomes.append(_fixture_outcome(store, sport, folder, team, now))
        for key in sorted((wanted or set()) - set(GAME_SPORT.values())):
            outcomes.append(
                Outcome(key, skipped=f"recorded leagues are keyed by sport ({', '.join(GAME_SPORT.values())})")
            )
    return outcomes


def _fixture_outcome(store: Store, sport: Sport, folder: Path, team: int | None, now: datetime) -> Outcome:
    try:
        settings = load_league_settings(folder / SETTINGS_VIEW)
        with _fixture_client(folder, settings) as client:
            period = settings.current_scoring_period or client.teams().data.scoring_period_id
            if period is None:
                return Outcome(sport, skipped=f"the recorded views in {folder} name no current scoring period")
            rostered = [roster.team_id for roster in client.rosters(period).data.teams]
            team_id = team if team is not None else min(rostered, default=None)
            if team_id is None:
                return Outcome(sport, skipped=f"the recorded views in {folder} hold no roster")
            league = League(
                key=sport,
                sport=sport,
                espn_league_id=settings.league_id,
                season=settings.season,
                team_id=team_id,
            )
            sync_league(store, league, client)
            schedule, notes = _recorded_schedule(client)
        row = store.leagues.by_key(sport)
        if row is None:
            return Outcome(sport, skipped="the recorded league was not stored")
        sheet = rank_league(store, row, now=now, schedule=schedule, sources=_espn_only(sport))
    except (EspnClientError, SyncError, RankingsError, OSError, ValueError) as exc:
        return Outcome(sport, skipped=f"{type(exc).__name__}: {exc}")
    return Outcome(sport, sheet, notes=notes)


@contextmanager
def _fixture_client(directory: Path, settings: LeagueSettings) -> Iterator[EspnClient]:
    """An ESPN client whose reads come from ``directory`` and cannot reach the network (nothing is captured)."""
    views = RecordedViews(directory)
    with httpx.Client(transport=httpx.MockTransport(views.handle)) as http:
        yield EspnClient(
            settings.game,
            settings.league_id,
            settings.season,
            None,
            client=http,
            capture=False,
            min_interval_s=0.0,
            max_attempts=1,
            sleep=lambda _seconds: None,
        )


def _recorded_schedule(client: EspnClient) -> tuple[ProSchedule | None, tuple[str, ...]]:
    try:
        return client.pro_schedule().data, ()
    except EspnClientError as exc:
        return None, (f"no recorded pro schedule ({exc})",)


def _espn_only(sport: Sport) -> ProjectionSourceRegistry | None:
    """The NBA's projection sources for a recorded run: ESPN's stored line alone, since DARKO is fetched over the
    network. ``None`` (every registered source) for the NFL, whose other source has nothing stored either way."""
    if sport != NBA_SPORT:
        return None
    sources = ProjectionSourceRegistry()
    sources.register(NBA_SPORT, ESPN, label="ESPN season projection (recorded)", stored=True)
    return sources


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("rankings")(rankings)
