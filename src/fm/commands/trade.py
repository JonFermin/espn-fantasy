"""``fm trade eval "give A, B get C"`` and ``fm trade find``: the trade evaluator and finder (DESIGN section 9.4,
ROADMAP #38).

Both read the store ``fm sync`` filled and render through :mod:`fm.render`; neither touches ESPN's write path
(CLAUDE.md: trades are approval-only, workers propose, the executor acts).

- ``fm trade eval`` judges one deal named by players (names or ESPN ids: ``"give Josh Jacobs, Jaylen Warren get Trey
  McBride"``; every player we get is on one other team): both sides' change in rest-of-season lineup value and title
  odds, legality against the league's settings, P(accept), and a recommendation to accept, decline or counter.
- ``fm trade find`` searches every opponent (or ``--with TEAM``) for 1-for-1, 2-for-1, 1-for-2 and 2-for-2 deals,
  screens them with greedy lineups, re-scores the best with a season simulation and ranks them by our change in title
  odds times P(accept). ``--propose`` drafts the best as ``trade_propose`` proposals (approve-only; one open offer per
  team, at most ``--max-offers`` new ones) and shows the policy verdict; with ``--dry-run`` nothing is stored.

Title odds need the league's ``mMatchup`` schedule: ``--matchups FILE`` (a recorded view), else the newest capture
indexed in ``raw_snapshots`` under the cache dir, else a live read through the ESPN session (``fm login``), which is
captured for the next run; without any, deals are judged on rest-of-season value and the output says so. The pro
schedule comes the way ``fm lineup`` and ``fm waivers`` get it (``--schedule FILE``, the cache, or a live read).
Market values come from FantasyCalc and ESPN's public pool (both degrade to our own ranks; ``--no-market`` skips
them). ``--as-of`` pins the clock (trade deadline, locks, availability); ``--runs`` and ``--seed`` set the
simulation, whose draws are common to the before and after runs, so a deal's numbers reproduce.

Fixture home: ``FM_CONFIG_DIR=tests/fixtures/home FM_CACHE_DIR=tests/fixtures/home/cache uv run fm trade find
--no-market --as-of 2026-10-04T15:00Z`` (``tests/fixtures/home/README.md``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from fm import paths
from fm.browser.session import BrowserError
from fm.commands.advise import _config, _fail, _parse_as_of, _schedule, _select
from fm.config import Config, League
from fm.decide.trades import (
    DEFAULT_SEED,
    FINDER_RUNS,
    MarketLike,
    ProposedTrade,
    SearchOptions,
    TradeContext,
    TradeError,
    TradeEvaluation,
    TradeSearch,
    evaluate_trade,
    find_trades,
    load_trade_context,
    parse_trade_text,
    propose_trades,
)
from fm.espn.auth import AuthError, load_session
from fm.espn.client import EspnClient, EspnClientError, EspnRead, View
from fm.espn.models import MatchupsView
from fm.jobs.sync import ESPN_SOURCE
from fm.model.valuation import DEFAULT_PLAYOFF_WEIGHT
from fm.render import INDENT, columns, percent, points, signed, stamp, warning_lines
from fm.sources.market import MarketSource
from fm.store import LeagueRow, RawSnapshotRow, Store

app = typer.Typer(help="Evaluate and find trades (approval-only; nothing is sent to ESPN).", no_args_is_help=True)

LeagueOption = Annotated[
    str | None,
    typer.Option(
        "--league", "-l", help="League key from config.toml. Default: the only league, or every league (find)."
    ),
]
AsOfOption = Annotated[
    str | None,
    typer.Option(
        "--as-of",
        help="Pin the clock (ISO 8601, e.g. 2026-10-04T15:00Z; a naive time is UTC): the trade deadline, locks and "
        "availability are judged at this instant. Default: now.",
    ),
]
ScheduleOption = Annotated[
    Path | None,
    typer.Option(
        "--schedule",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="A recorded proTeamSchedules_wl view to use as the pro schedule instead of the cache or ESPN.",
    ),
]
MatchupsOption = Annotated[
    Path | None,
    typer.Option(
        "--matchups",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="A recorded mMatchup view (the league's schedule) instead of the cache or ESPN.",
    ),
]
RunsOption = Annotated[
    int | None,
    typer.Option("--runs", min=100, help=f"Simulated seasons per deal (eval: 10000, find: {FINDER_RUNS})."),
]
SeedOption = Annotated[int, typer.Option("--seed", help="The simulation's seed; the same seed gives the same numbers.")]
NoMarketOption = Annotated[
    bool,
    typer.Option("--no-market", help="Do not ask FantasyCalc or ESPN for market values; P(accept) uses our ranks."),
]


@contextmanager
def market_source() -> Iterator[MarketLike]:
    """The market source the commands use: FantasyCalc and ESPN's public pool (replaced in tests)."""
    with MarketSource() as source:
        yield source


# --- eval -------------------------------------------------------------------------------------------------------------


@app.command("eval")
def eval_(
    deal: Annotated[str, typer.Argument(help='The deal, as in "give A, B get C": players by name or ESPN id.')],
    league: LeagueOption = None,
    as_of: AsOfOption = None,
    schedule: ScheduleOption = None,
    matchups: MatchupsOption = None,
    runs: RunsOption = None,
    seed: SeedOption = DEFAULT_SEED,
    no_market: NoMarketOption = False,
) -> None:
    """Judge one deal: both sides' rest-of-season value and title odds, legality, P(accept) and a recommendation."""
    config = _config()
    at = _parse_as_of(as_of)
    chosen = _select(config, [league] if league else None)
    if len(chosen) != 1:
        _fail("several leagues are configured; name one with --league KEY")
    configured = chosen[0]
    with Store.open() as store, _market(no_market) as market:
        row = _row(store, configured)
        ctx, notes = _context(
            store, config, configured, row, at=at, schedule=schedule, matchups=matchups, market=market
        )
        try:
            spec = parse_trade_text(ctx, deal)
            evaluation = evaluate_trade(ctx, spec, runs=runs if runs is not None else 10_000, seed=seed)
        except TradeError as exc:
            _fail(str(exc))
    for line in evaluation_lines(ctx, evaluation, as_of=stamp(at), notes=notes):
        typer.echo(line)


# --- find -------------------------------------------------------------------------------------------------------------


@app.command("find")
def find(
    league: LeagueOption = None,
    with_team: Annotated[
        list[int] | None,
        typer.Option("--with", min=1, help="Only deals with this team id (repeatable). Default: every other team."),
    ] = None,
    top: Annotated[int, typer.Option("--top", min=1, help="Deals to show.")] = 10,
    propose: Annotated[
        bool, typer.Option("--propose", help="Draft the best deals as approve-only trade proposals.")
    ] = False,
    max_offers: Annotated[int, typer.Option("--max-offers", min=1, help="With --propose: at most this many.")] = 3,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="With --propose: show what policy says and store nothing.")
    ] = False,
    allow_drops: Annotated[
        bool, typer.Option("--allow-drops", help="Keep deals that make us drop a player (they cannot be proposed).")
    ] = False,
    include_ir: Annotated[bool, typer.Option("--include-ir", help="Keep players in IR slots in play.")] = False,
    as_of: AsOfOption = None,
    schedule: ScheduleOption = None,
    matchups: MatchupsOption = None,
    runs: RunsOption = None,
    seed: SeedOption = DEFAULT_SEED,
    no_market: NoMarketOption = False,
) -> None:
    """Search every opponent for deals that raise our title odds, ranked by that times P(accept)."""
    if dry_run and not propose:
        _fail("--dry-run applies to --propose")
    config = _config()
    at = _parse_as_of(as_of)
    chosen = _select(config, [league] if league else None)
    if with_team and len(chosen) != 1:
        _fail("--with names one league's teams; add --league KEY")
    options = SearchOptions(
        runs=runs if runs is not None else FINDER_RUNS,
        seed=seed,
        limit=top,
        allow_drops=allow_drops,
        include_ir=include_ir,
    )
    with Store.open() as store, _market(no_market) as market:
        for index, configured in enumerate(chosen):
            if index:
                typer.echo("")
            row = store.leagues.by_key(configured.key)
            if row is None:
                typer.echo(f"{configured.key}: not synced; run fm sync")
                continue
            ctx, notes = _context(
                store, config, configured, row, at=at, schedule=schedule, matchups=matchups, market=market
            )
            try:
                search = find_trades(ctx, opponents=with_team, options=options)
                proposed: list[ProposedTrade] = []
                if propose:
                    proposed = propose_trades(
                        store, config, ctx, search.results, max_offers=max_offers, dry_run=dry_run, now=at
                    )
            except TradeError as exc:
                _fail(str(exc))
            for line in search_lines(ctx, search, proposed, as_of=stamp(at), notes=notes):
                typer.echo(line)


# --- loading the league -----------------------------------------------------------------------------------------------


@contextmanager
def _market(skip: bool) -> Iterator[MarketLike | None]:
    if skip:
        yield None
        return
    with market_source() as source:
        yield source


def _row(store: Store, configured: League) -> LeagueRow:
    row = store.leagues.by_key(configured.key)
    if row is None:
        _fail(f"{configured.key}: not synced; run fm sync")
    return row


def _context(
    store: Store,
    config: Config,
    configured: League,
    row: LeagueRow,
    *,
    at: datetime,
    schedule: Path | None,
    matchups: Path | None,
    market: MarketLike | None,
) -> tuple[TradeContext, list[str]]:
    """The league's trade context and the notes from loading its inputs."""
    loaded = _schedule(store, configured, path=schedule)
    notes = [*loaded.warnings, *(() if loaded.schedule is not None else (loaded.note,))]
    view, note = _matchups(store, configured, row, path=matchups)
    if note:
        notes.append(note)
    try:
        ctx = load_trade_context(
            store,
            row,
            now=at,
            config=config,
            schedule=loaded.schedule,
            matchups=view,
            market=market,
            playoff_weight=DEFAULT_PLAYOFF_WEIGHT,
        )
    except TradeError as exc:
        _fail(f"{configured.key}: {exc}")
    return ctx, notes


def _matchups(
    store: Store, configured: League, row: LeagueRow, *, path: Path | None
) -> tuple[MatchupsView | None, str]:
    """The league's ``mMatchup`` read: ``--matchups FILE``, else the newest capture indexed for the league, else a live
    read (captured for next time); ``None`` and the reason when there is none."""
    if path is not None:
        view, problem = _read_matchups(path)
        if view is None:
            _fail(f"--matchups {path}: {problem}")
        return view, ""
    snapshots = store.raw_snapshots.find(ESPN_SOURCE, View.MATCHUP.value, league_id=row.row_id)
    if snapshots:
        view, problem = _read_matchups(paths.cache_dir() / snapshots[-1].path)
        if view is not None:
            return view, ""
        note = f"the captured mMatchup {snapshots[-1].path} {problem}; reading it again"
    else:
        note = ""
    try:
        session = load_session()
    except (AuthError, BrowserError) as exc:
        return (
            None,
            f"{configured.key}: no mMatchup schedule is captured and there is no ESPN session to read one ({exc})",
        )
    try:
        with EspnClient.for_league(configured, session) as client:
            read = client.matchups()
    except EspnClientError as exc:
        return None, f"{configured.key}: the mMatchup schedule could not be read from ESPN: {exc}"
    _remember(store, read, row.row_id)
    return read.data, note


def _read_matchups(path: Path) -> tuple[MatchupsView | None, str]:
    try:
        return MatchupsView.model_validate_json(path.read_bytes()), ""
    except FileNotFoundError:
        return None, "is gone from the cache"
    except (OSError, ValidationError, ValueError) as exc:
        return None, f"could not be read ({type(exc).__name__})"


def _remember(store: Store, read: EspnRead[MatchupsView], league_id: int) -> None:
    """Index a live read's capture in ``raw_snapshots`` so the next run reads it from the cache."""
    capture = read.capture
    if capture is None:
        return
    store.raw_snapshots.insert(
        RawSnapshotRow(
            source=ESPN_SOURCE,
            kind=capture.kind,
            league_id=league_id,
            scoring_period_id=capture.scoring_period_id,
            url=capture.url,
            params=dict[str, Any](capture.params),
            path=capture.relative_path,
            sha256=capture.sha256,
            size_bytes=capture.size_bytes,
            status_code=capture.status_code,
            fetched_at=capture.fetched_at,
        )
    )


# --- rendering --------------------------------------------------------------------------------------------------------


def _label(ctx: TradeContext, ids: Sequence[int]) -> str:
    parts = []
    for espn_id in ids:
        player = ctx.players.get(espn_id)
        position = f" ({player.position})" if player is not None and player.position else ""
        parts.append(f"{ctx.name(espn_id)}{position}")
    return ", ".join(parts) or "nobody"


def evaluation_lines(ctx: TradeContext, found: TradeEvaluation, *, as_of: str, notes: Sequence[str] = ()) -> list[str]:
    """One deal as text: the verdict, both sides' numbers, legality, P(accept), fit notes and warnings."""
    spec = found.spec
    league = ctx.league
    lines = [
        f"{league.key}: trade with {ctx.team_name(spec.other_team_id)} (as of {as_of})",
        f"{INDENT}give: {_label(ctx, spec.give)}",
        f"{INDENT}get:  {_label(ctx, spec.get)}",
        f"{INDENT}{found.recommendation.upper()}: {found.reason}",
    ]
    legality = found.legality
    lines.append(f"{INDENT}legal: " + ("yes" if legality.legal else "NO: " + "; ".join(legality.problems)))
    lines.extend(f"{INDENT * 2}{note}" for note in legality.notes)
    header = ("", f"lineup {found.unit}", "", "title", "playoffs", "bye")
    rows: list[Sequence[str]] = [header]
    for side in (found.ours, found.theirs):
        rows.append(
            (
                side.name,
                f"{points(side.ros_before)} -> {points(side.ros_after)}",
                signed(side.delta_ros),
                _delta(side.odds_before and side.odds_before.title, side.delta_title),
                _delta(side.odds_before and side.odds_before.playoffs, side.delta_playoffs),
                _delta(side.odds_before and side.odds_before.bye, side.delta_bye),
            )
        )
    lines.extend(columns(rows, align=("<", ">", ">", ">", ">", ">")))
    lines.append(f"{INDENT}{found.acceptance.describe()}")
    if found.simulated:
        lines.append(f"{INDENT}simulated {found.runs} seasons (seed {found.seed})")
    else:
        lines.append(f"{INDENT}no season simulation: judged on rest-of-season value ({found.unit})")
    lines.extend(f"{INDENT}fit: {note}" for note in found.fit)
    lines.extend(warning_lines((*found.warnings, *ctx.warnings, *notes)))
    return lines


def _delta(base: float | None, change: float | None) -> str:
    if base is None or change is None:
        return "-"
    return f"{percent(base)} {change * 100:+.1f}"


def search_lines(
    ctx: TradeContext,
    search: TradeSearch,
    proposed: Sequence[ProposedTrade],
    *,
    as_of: str,
    notes: Sequence[str] = (),
) -> list[str]:
    """A search as text: the ranked deals (our change in title odds or lineup value, theirs, P(accept), score), then
    what proposing them did, and the warnings."""
    league = ctx.league
    lines = [
        f"{league.key}: trades for {ctx.team_name(ctx.team_id)} (as of {as_of}); {search.enumerated} deals enumerated, "
        f"{search.screened} passed the screen, {search.rescored} re-scored"
    ]
    if not search.results:
        lines.append(f"{INDENT}no deal worth proposing")
    else:
        simulated = any(found.simulated for found in search.results)
        unit = ctx.model.unit
        header = (
            "#",
            "with",
            "give",
            "get",
            "ours",
            "theirs",
            "title" if simulated else "",
            "P(accept)",
            "score",
        )
        rows: list[Sequence[str]] = [header]
        for index, found in enumerate(search.results, start=1):
            spec = found.spec
            rows.append(
                (
                    str(index),
                    ctx.team_name(spec.other_team_id),
                    ", ".join(ctx.name(espn_id) for espn_id in spec.give),
                    ", ".join(ctx.name(espn_id) for espn_id in spec.get),
                    signed(found.ours.delta_ros),
                    signed(found.theirs.delta_ros),
                    "-" if found.ours.delta_title is None else f"{found.ours.delta_title * 100:+.1f}",
                    percent(found.acceptance.p_accept),
                    "-" if found.score is None else f"{found.score * 100:+.2f}",
                )
            )
        lines.extend(columns(rows, align=("<", "<", "<", "<", ">", ">", ">", ">", ">")))
        basis = "title odds (points of probability) times P(accept)" if simulated else "starter seasons times P(accept)"
        lines.append(f"{INDENT}ours, theirs: change in lineup {unit}; title: our change in title odds, in points")
        lines.append(f"{INDENT}score: our change in {basis}")
    for outcome in proposed:
        lines.append(f"{INDENT}{_proposed_line(ctx, outcome)}")
    lines.extend(warning_lines((*search.warnings, *ctx.warnings, *notes)))
    return lines


def _proposed_line(ctx: TradeContext, outcome: ProposedTrade) -> str:
    spec = outcome.evaluation.spec
    deal = f"{', '.join(ctx.name(i) for i in spec.give)} for {', '.join(ctx.name(i) for i in spec.get)}"
    if outcome.blocked is not None:
        return f"not proposed ({deal}): {outcome.blocked}"
    if outcome.dry:
        return f"would be proposed ({deal}; policy approve); --dry-run stored nothing"
    row = outcome.proposal
    assert row is not None
    if outcome.existing:
        return f"already open as #{row.row_id} ({deal}; {row.status})"
    return f"proposed as #{row.row_id} ({deal}; policy {row.policy}); fm proposals approve {row.row_id}"


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="trade")
