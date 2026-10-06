"""The weekly report (DESIGN section 9.5, ROADMAP #46): matchup outlook, playoff odds, moves made, upcoming deadlines.

Read-only: it reads what ``fm sync`` stored plus the optional ``mMatchup`` schedule and pro schedule it is handed, and
writes nothing to ESPN. :func:`build_league_report` gathers one league's sections, each degrading to a note rather
than failing when its input is missing (``fm sync`` does not store ``mMatchup``, so without ``--matchups`` or a
capture the matchup and odds sections say so). :func:`render_markdown` turns the reports into the markdown that
:func:`write_report` saves under ``paths.cache_dir()/reports`` and ``fm.notify.send_report`` pushes to the phone.

- **Matchup outlook:** the current matchup period's opponent and :func:`fm.model.simulate.simulate_matchup` between the
  two rosters' outlooks (the same ones the trade evaluator values), with each category's chance in a category league.
- **Playoff odds:** :func:`fm.model.simulate.simulate_season` over the league's schedule; the trend is the change from
  the previous report's odds, kept in a small sidecar next to the markdown (:func:`previous_odds`).
- **Moves made:** executions that finished in the window, one line per proposal with its verified or failed outcome.
- **Upcoming deadlines:** :func:`fm.jobs.deadlines.upcoming` over the league's settings, never a hardcoded time.

Claude is not called: the lines are templated from engine numbers and the rationale already stored with each proposal.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from fm import paths
from fm.config import Config
from fm.decide.trades import TradeContext, TradeError, load_trade_context
from fm.espn.models import Matchup, MatchupsView
from fm.espn.settings import LeagueSettings
from fm.jobs.deadlines import upcoming
from fm.model.simulate import (
    CategoryOutlook,
    SimulationError,
    TeamOdds,
    TeamOutlook,
    simulate_matchup,
    simulate_season,
)
from fm.notify.messages import describe_payload
from fm.proposals.policy import parse_payload, stored_settings
from fm.render import percent
from fm.render import stamp as _stamp
from fm.sports.base import ScheduleLike
from fm.store import ExecutionRow, LeagueRow, ProposalRow, Store

DEFAULT_WINDOW_DAYS: Final = 7
"""How far back moves made are listed, and how far ahead deadlines are."""
REPORT_RUNS: Final = 5_000
SEED: Final = 46
"""The simulations' seed, so the same inputs give the same report."""
REPORTS_DIRNAME: Final = "reports"
FINISHED: Final = ("verified", "failed")


@dataclass(frozen=True, slots=True)
class MatchupOutlookLine:
    """This matchup period's opponent and our chances: ``win``, ``tie`` and the expected points of each side in a
    points league, ``categories`` (P(we win it) per category) in a category league."""

    period: int
    opponent: str
    win: float
    tie: float
    ours: float | None = None
    theirs: float | None = None
    categories: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OddsSnapshot:
    """Our season odds as of a report; the next report's trend is measured against it."""

    as_of: datetime
    playoffs: float
    bye: float
    title: float

    def to_json(self) -> dict[str, Any]:
        return {"as_of": self.as_of.isoformat(), "playoffs": self.playoffs, "bye": self.bye, "title": self.title}

    @classmethod
    def from_odds(cls, odds: TeamOdds, as_of: datetime) -> OddsSnapshot:
        return cls(as_of, odds.playoffs, odds.bye, odds.title)


@dataclass(frozen=True, slots=True)
class MoveLine:
    """A proposal that was carried out (or tried) in the window."""

    proposal_id: int
    at: datetime
    kind: str
    outcome: str
    text: str


@dataclass(frozen=True, slots=True)
class LeagueReport:
    """One league's report. ``None`` (or empty) sections are explained in ``notes``."""

    key: str
    name: str
    team: str
    record: str
    matchup: MatchupOutlookLine | None = None
    odds: OddsSnapshot | None = None
    previous: OddsSnapshot | None = None
    moves: tuple[MoveLine, ...] = ()
    deadlines: tuple[str, ...] = ()
    pending: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


# --- building ---------------------------------------------------------------------------------------------------------


def build_league_report(
    store: Store,
    config: Config,
    league: LeagueRow,
    *,
    now: datetime,
    schedule: ScheduleLike | None = None,
    matchups: MatchupsView | None = None,
    previous: OddsSnapshot | None = None,
    window: timedelta = timedelta(days=DEFAULT_WINDOW_DAYS),
    runs: int = REPORT_RUNS,
) -> LeagueReport:
    """Gather the four sections for ``league`` (``now`` aware). Never raises for missing inputs: each one becomes a
    note, and the sections that do not need it are still filled."""
    notes: list[str] = []
    team = store.teams.get(league.row_id, league.team_id)
    team_name = team.name if team is not None else f"team {league.team_id}"
    record = "-" if team is None else _record(team.wins, team.losses, team.ties, team.playoff_seed)
    matchup: MatchupOutlookLine | None = None
    odds: OddsSnapshot | None = None
    ctx = _context(store, config, league, now, schedule, matchups, notes)
    if ctx is not None:
        matchup = _matchup_outlook(ctx, notes, runs)
        odds = _season_odds(ctx, notes, runs, now)
    return LeagueReport(
        key=league.key,
        name=league.name or league.key,
        team=team_name,
        record=record,
        matchup=matchup,
        odds=odds,
        previous=previous if previous is not None and previous.as_of < now else None,
        moves=moves_made(store, league, since=now - window, until=now),
        deadlines=_deadlines(store, league, schedule, now, window, notes),
        pending=tuple(_pending(store, league)),
        notes=tuple(dict.fromkeys(notes)),
    )


def _record(wins: int, losses: int, ties: int, seed: int | None) -> str:
    text = f"{wins}-{losses}" + (f"-{ties}" if ties else "")
    return text if seed is None else f"{text}, seed {seed}"


def _context(
    store: Store,
    config: Config,
    league: LeagueRow,
    now: datetime,
    schedule: ScheduleLike | None,
    matchups: MatchupsView | None,
    notes: list[str],
) -> TradeContext | None:
    try:
        ctx = load_trade_context(store, league, now=now, config=config, schedule=schedule, matchups=matchups)
    except TradeError as exc:
        notes.append(f"no outlook or odds: {exc}")
        return None
    if matchups is None:
        notes.append(
            "no mMatchup schedule (fm sync does not store it; pass --matchups FILE or run fm login): "
            "matchup outlook and playoff odds are skipped"
        )
    return ctx


def _outlook(ctx: TradeContext, team_id: int) -> TeamOutlook | CategoryOutlook | None:
    return ctx.model.outlook(team_id, ctx.playing(team_id))


def _our_matchup(ctx: TradeContext) -> Matchup | None:
    if ctx.matchups is None or ctx.matchup_period is None:
        return None
    return next((m for m in ctx.matchups.for_team(ctx.team_id, ctx.matchup_period) if not m.is_bye), None)


def _matchup_outlook(ctx: TradeContext, notes: list[str], runs: int) -> MatchupOutlookLine | None:
    if ctx.matchups is None:
        return None
    period = ctx.matchup_period
    matchup = _our_matchup(ctx)
    if period is None or matchup is None:
        notes.append(f"{ctx.league.key}: no matchup of ours in matchup period {period}; no matchup outlook")
        return None
    other = matchup.opponent(ctx.team_id)
    assert other is not None
    if matchup.is_decided:
        notes.append(f"{ctx.league.key}: matchup period {period} is already decided ({matchup.winner}); no outlook")
        return None
    ours, theirs = _outlook(ctx, ctx.team_id), _outlook(ctx, other.team_id)
    if ours is None or theirs is None:
        notes.append(f"{ctx.league.key}: no outlook could be built for this matchup; no matchup outlook")
        return None
    try:
        odds = simulate_matchup(ctx.settings, ours, theirs, period, seed=SEED, runs=runs)
    except (SimulationError, KeyError, ValueError) as exc:
        notes.append(f"{ctx.league.key}: the matchup cannot be simulated: {exc}")
        return None
    mean_ours = mean_theirs = None
    if isinstance(ours, TeamOutlook) and isinstance(theirs, TeamOutlook):
        mean_ours, mean_theirs = ours.score(period)[0], theirs.score(period)[0]
    return MatchupOutlookLine(
        period=period,
        opponent=ctx.team_name(other.team_id),
        win=odds.home + 0.5 * odds.tie,
        tie=odds.tie,
        ours=mean_ours,
        theirs=mean_theirs,
        categories=dict(odds.categories),
    )


def _season_odds(ctx: TradeContext, notes: list[str], runs: int, now: datetime) -> OddsSnapshot | None:
    if ctx.matchups is None:
        return None
    if not ctx.model.simulates:
        notes.append(f"{ctx.league.key}: the season cannot be simulated for this league's scoring; no playoff odds")
        return None
    outlooks: dict[int, TeamOutlook | CategoryOutlook] = {}
    for team in ctx.rosters:
        built = _outlook(ctx, team)
        if built is None:
            notes.append(f"{ctx.league.key}: no outlook for {ctx.team_name(team)}; no playoff odds")
            return None
        outlooks[team] = built
    try:
        season = simulate_season(
            ctx.settings, ctx.matchups, outlooks, seed=SEED, runs=runs, current_matchup_period=ctx.matchup_period
        )
        mine = season.team(ctx.team_id)
    except (SimulationError, KeyError) as exc:
        notes.append(f"{ctx.league.key}: the season cannot be simulated: {exc}")
        return None
    notes.extend(f"{ctx.league.key}: {warning}" for warning in season.warnings)
    return OddsSnapshot.from_odds(mine, now)


def moves_made(store: Store, league: LeagueRow, *, since: datetime, until: datetime) -> tuple[MoveLine, ...]:
    """The league's proposals whose last execution finished in ``[since, until]``, oldest first. Dry runs and
    attempts that never finished are not moves made."""
    lines: list[MoveLine] = []
    for proposal in store.proposals.find(league_id=league.row_id, statuses=FINISHED):
        assert proposal.row_id is not None
        finished = _last_finish(store.executions.for_proposal(proposal.row_id))
        if finished is None or not since <= finished <= until:
            continue
        lines.append(
            MoveLine(proposal.row_id, finished, proposal.kind, proposal.status, _describe(store, league, proposal))
        )
    return tuple(sorted(lines, key=lambda line: (line.at, line.proposal_id)))


def _last_finish(executions: Iterable[ExecutionRow]) -> datetime | None:
    done = [e.finished_at for e in executions if e.finished_at is not None and e.status != "dry_run"]
    return max(done) if done else None


def _describe(store: Store, league: LeagueRow, proposal: ProposalRow) -> str:
    try:
        return describe_payload(store, league, parse_payload(proposal))
    except (ValueError, KeyError):
        return proposal.kind


def _pending(store: Store, league: LeagueRow) -> list[str]:
    lines = []
    for proposal in store.proposals.open(league.row_id):
        due = "" if proposal.deadline is None else f", due {_stamp(proposal.deadline)}"
        lines.append(
            f"#{proposal.row_id} {proposal.kind} ({proposal.status}{due}): {_describe(store, league, proposal)}"
        )
    return lines


def _deadlines(
    store: Store,
    league: LeagueRow,
    schedule: ScheduleLike | None,
    now: datetime,
    window: timedelta,
    notes: list[str],
) -> tuple[str, ...]:
    settings: LeagueSettings | None = stored_settings(store, league)
    if settings is None:
        notes.append(f"{league.key}: league settings are not synced; no deadlines")
        return ()
    if schedule is None:
        notes.append(f"{league.key}: no pro schedule; no lock deadlines (waiver and trade deadlines need none)")
    found: list[str] = []
    if schedule is not None:
        deadlines, warnings = upcoming(league.key, league.sport, settings, schedule, now=now, horizon=window)
        notes.extend(warnings)
        found.extend(deadline.describe() for deadline in deadlines)
    trade = settings.trade.deadline
    if trade is not None and now <= trade <= now + window:
        found.append(f"{_stamp(trade)} {league.key}: trade deadline")
    return tuple(found)


# --- rendering --------------------------------------------------------------------------------------------------------


def render_league(report: LeagueReport) -> list[str]:
    """One league's markdown section, ``## `` level."""
    lines = [f"## {report.key}: {report.name}", "", f"{report.team} ({report.record})", ""]
    lines += ["### Matchup outlook", ""]
    matchup = report.matchup
    if matchup is None:
        lines.append("Not available (see notes).")
    else:
        lines.append(
            f"Matchup period {matchup.period} vs {matchup.opponent}: {percent(matchup.win)} to win"
            + (f", {percent(matchup.tie)} to tie" if matchup.tie >= 0.005 else "")
            + "."
        )
        if matchup.ours is not None and matchup.theirs is not None:
            lines.append(f"Expected points {matchup.ours:.1f} to {matchup.theirs:.1f}.")
        if matchup.categories:
            lines += ["", "| Category | P(win) |", "|---|---|"]
            lines += [f"| {name} | {percent(chance)} |" for name, chance in matchup.categories.items()]
    lines += ["", "### Playoff odds", ""]
    odds = report.odds
    if odds is None:
        lines.append("Not available (see notes).")
    else:
        lines.append(
            f"Playoffs {percent(odds.playoffs)}{_trend(odds.playoffs, report.previous and report.previous.playoffs)}, "
            f"bye {percent(odds.bye)}{_trend(odds.bye, report.previous and report.previous.bye)}, "
            f"title {percent(odds.title)}{_trend(odds.title, report.previous and report.previous.title)}."
        )
        if report.previous is None:
            lines.append("No earlier report to compare with.")
        else:
            lines.append(f"Trend is against the report of {_stamp(report.previous.as_of)}.")
    lines += ["", "### Moves made", ""]
    lines += [f"- {_stamp(m.at)} #{m.proposal_id} {m.kind} {m.outcome}: {m.text}" for m in report.moves] or ["None."]
    if report.pending:
        lines += ["", "Still open:", ""] + [f"- {text}" for text in report.pending]
    lines += ["", "### Upcoming deadlines", ""]
    lines += [f"- {text}" for text in report.deadlines] or ["None in the window."]
    return lines


def _trend(now: float, before: float | None) -> str:
    if before is None:
        return ""
    delta = round((now - before) * 100)
    return f" ({delta:+d} pts)" if delta else " (unchanged)"


def render_markdown(reports: Sequence[LeagueReport], *, as_of: datetime, notes: Sequence[str] = ()) -> str:
    """The whole report: a title, one section per league, then the notes of every section that degraded."""
    lines = [f"# Weekly report {as_of.astimezone(UTC):%Y-%m-%d}", "", f"As of {_stamp(as_of)}.", ""]
    for report in reports:
        lines += [*render_league(report), ""]
    every = [*notes, *(f"{r.key}: {note}" if not note.startswith(r.key) else note for r in reports for note in r.notes)]
    if every:
        lines += ["## Notes", "", *[f"- {note}" for note in dict.fromkeys(every)], ""]
    return "\n".join(lines)


def report_title(reports: Sequence[LeagueReport], as_of: datetime) -> str:
    keys = ", ".join(r.key for r in reports) or "no leagues"
    return f"Weekly report {as_of.astimezone(UTC):%Y-%m-%d} ({keys})"


# --- files ------------------------------------------------------------------------------------------------------------


def reports_dir() -> Path:
    """``paths.cache_dir()/reports``: the markdown reports and the odds sidecars (deletable; the next report just
    has no trend)."""
    path = paths.cache_dir() / REPORTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_path(as_of: datetime, directory: Path | None = None) -> Path:
    return (directory or reports_dir()) / f"weekly-{as_of.astimezone(UTC):%Y-%m-%d}.md"


def write_report(markdown: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8", newline="\n")
    return path


def _odds_file(directory: Path, key: str) -> Path:
    return directory / f"odds-{key}.json"


def previous_odds(directory: Path, key: str) -> OddsSnapshot | None:
    """The odds the last report stored for the league, or ``None`` (first report, or an unreadable sidecar)."""
    try:
        data = json.loads(_odds_file(directory, key).read_text(encoding="utf-8"))
        return OddsSnapshot(
            datetime.fromisoformat(data["as_of"]), float(data["playoffs"]), float(data["bye"]), float(data["title"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def remember_odds(directory: Path, key: str, odds: OddsSnapshot) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _odds_file(directory, key).write_text(json.dumps(odds.to_json()), encoding="utf-8")
