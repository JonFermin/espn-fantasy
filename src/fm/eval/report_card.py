"""The live report card (DESIGN section 15, ROADMAP #47): the backtest's metrics on the decisions actually made.

:mod:`fm.eval.backtest` replays a league's history to score projection sources; this module replays it to score *us*:

1. **Lineup efficiency.** What the lineup we actually started scored against the hindsight-optimal lineup from the same
   roster, per played week and over the season so far. It is the backtest's baseline
   (:func:`fm.eval.backtest.run_backtest`
   over :func:`fm.eval.backtest.load_store_data`), so slot eligibility, IR and the league's points come from the same
   machinery, never from a second implementation.
2. **Points left on the bench.** Per week, what the benched players scored (:attr:`WeekCard.bench`) and how much of the
   hindsight optimum the lineup missed (:attr:`WeekCard.left`: the start/sit regret; it is at most the bench's points
   and smaller when the bench could not have filled the slots the starters held).
3. **Pickup value.** For each add or claim we executed and verified, the add's actual points over the played weeks he
   was on our roster against what the player dropped for him scored in those same weeks (:class:`PickupLine`).

Everything is read from the state database: roster snapshots, stored actual stat lines, and executions. Points are
computed per league from its scoring items (:class:`fm.model.scoring.Scorer`), never from a stored total. Read-only:
nothing here can write to ESPN. A league without a points format (category leagues are not supported yet), without
synced settings, without a played week, or without actuals is a note in :attr:`ReportCard.notes`, never an error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Final

from fm.espn.ids import ids_for
from fm.espn.settings import LeagueSettings
from fm.eval.backtest import (
    EPSILON,
    BacktestData,
    BacktestError,
    LineupWeek,
    load_store_data,
    run_backtest,
)
from fm.jobs.report import LeagueReport, moves_made
from fm.model.projections import ESPN, position_for
from fm.model.scoring import Scorer
from fm.proposals.payloads import AddDropPayload, WaiverPayload
from fm.proposals.policy import parse_payload, stored_settings
from fm.render import percent
from fm.render import points as _points
from fm.sports.base import ScheduleLike
from fm.store import LeagueRow, Store

REPORT_CARD_WEEKS: Final = 6
"""How many of the latest played weeks the rendered card lists (the totals always cover every played week)."""
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class WeekCard:
    """One played week: ``points`` is what our starters scored, ``optimal`` the hindsight optimum from the same
    roster, ``bench`` what the players we benched scored (IR excluded) and ``swaps`` how many starters the optimum
    would have replaced."""

    period: int
    points: float
    optimal: float
    bench: float
    swaps: int

    @property
    def left(self) -> float:
        """Points left on the bench against the hindsight optimum (``optimal - points``, never negative)."""
        return max(self.optimal - self.points, 0.0)

    @property
    def efficiency(self) -> float | None:
        return self.points / self.optimal if self.optimal > EPSILON else None


@dataclass(frozen=True, slots=True)
class PickupLine:
    """An add or claim we executed. ``periods`` are the played weeks from the move on that he was on our roster;
    ``gained`` is what he scored in them, ``given_up`` what the player dropped for him scored in the same weeks
    (``0`` when nothing was dropped), so ``net`` is the points the move added over keeping the dropped player."""

    proposal_id: int
    at: datetime
    kind: str
    added: str
    dropped: str | None
    periods: tuple[int, ...]
    gained: float
    given_up: float

    @property
    def net(self) -> float:
        return self.gained - self.given_up

    @property
    def settled(self) -> bool:
        """Whether any played week counts toward it yet."""
        return bool(self.periods)


@dataclass(frozen=True, slots=True)
class ReportCard:
    """One league's card; ``weeks`` and ``pickups`` are empty (and ``notes`` says why) when there is no data."""

    key: str
    weeks: tuple[WeekCard, ...] = ()
    pickups: tuple[PickupLine, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def points(self) -> float:
        return math.fsum(week.points for week in self.weeks)

    @property
    def optimal(self) -> float:
        return math.fsum(week.optimal for week in self.weeks)

    @property
    def efficiency(self) -> float | None:
        return self.points / self.optimal if self.optimal > EPSILON else None

    @property
    def left(self) -> float:
        return math.fsum(week.left for week in self.weeks)

    @property
    def bench(self) -> float:
        return math.fsum(week.bench for week in self.weeks)

    @property
    def swaps(self) -> int:
        return sum(week.swaps for week in self.weeks)

    @property
    def pickup_net(self) -> float:
        return math.fsum(pickup.net for pickup in self.pickups if pickup.settled)


# --- building ---------------------------------------------------------------------------------------------------------


def build_report_card(
    store: Store,
    league: LeagueRow,
    *,
    now: datetime,
    settings: LeagueSettings | None = None,
    schedule: ScheduleLike | None = None,
) -> ReportCard:
    """The card for ``league`` as of ``now`` (aware). ``settings`` default to the synced ones. Never raises for
    missing or unsupported inputs: each becomes a note."""
    resolved = settings if settings is not None else stored_settings(store, league)
    if resolved is None:
        return ReportCard(league.key, notes=("league settings are not synced; no report card",))
    try:
        data = load_store_data(store, league, resolved, schedule=schedule)
    except BacktestError as exc:
        return ReportCard(league.key, notes=(f"no report card: {exc}",))
    scorer = Scorer(resolved)
    report = run_backtest(data, sources=(), scorer=scorer)
    bench = _bench_points(data, scorer)
    weeks = tuple(_week_card(week, bench[week.period]) for week in report.baseline.weeks)
    pickups, notes = _pickups(store, league, data, scorer, now)
    return ReportCard(league.key, weeks, pickups, (*data.warnings, *report.warnings, *notes))


def _week_card(week: LineupWeek, bench: float) -> WeekCard:
    return WeekCard(week.period, week.points, week.optimal, bench, week.swaps)


def _bench_points(data: BacktestData, scorer: Scorer) -> dict[int, float]:
    """Each replayed week's points scored by the players who sat on our bench (a player on IR is not benched)."""
    bench_slot = ids_for(data.settings.game).bench_slot
    positions = data.positions
    result: dict[int, float] = {}
    for week in data.weeks:
        result[week.period] = math.fsum(
            scorer.points(week.actuals[espn_id], position=position_for(data.sport, espn_id, positions))
            for espn_id, slot_id in week.lineup.items()
            if slot_id == bench_slot and espn_id in week.actuals
        )
    return result


def _pickups(
    store: Store, league: LeagueRow, data: BacktestData, scorer: Scorer, now: datetime
) -> tuple[tuple[PickupLine, ...], list[str]]:
    notes: list[str] = []
    actual: dict[int, dict[int, float]] = {}

    def scored(period: int, espn_id: int) -> float:
        if period not in actual:
            positions = data.positions
            actual[period] = {
                row.espn_id: scorer.points(row.stats, position=position_for(data.sport, row.espn_id, positions))
                for row in store.projections.for_period(league.sport, league.season, period, kind="actual")
                if row.source == ESPN
            }
        return actual[period].get(espn_id, 0.0)

    rosters = {week.period: week.lineup for week in data.weeks}
    lines: list[PickupLine] = []
    for move in moves_made(store, league, since=_EPOCH, until=now):
        proposal = store.proposals.get(move.proposal_id)
        if proposal is None or proposal.status != "verified":
            continue
        try:
            payload = parse_payload(proposal)
        except (ValueError, KeyError):
            continue
        if not isinstance(payload, AddDropPayload | WaiverPayload) or payload.add_espn_id is None:
            continue
        first = proposal.scoring_period_id or 0
        played = tuple(period for period in data.periods if period >= first and payload.add_espn_id in rosters[period])
        dropped = payload.drop_espn_id
        lines.append(
            PickupLine(
                proposal_id=move.proposal_id,
                at=move.at,
                kind=proposal.kind,
                added=_name(data, store, league, payload.add_espn_id),
                dropped=None if dropped is None else _name(data, store, league, dropped),
                periods=played,
                gained=math.fsum(scored(period, payload.add_espn_id) for period in played),
                given_up=0.0 if dropped is None else math.fsum(scored(period, dropped) for period in played),
            )
        )
    if not lines:
        notes.append("no verified add or claim to value yet")
    return tuple(lines), notes


def _name(data: BacktestData, store: Store, league: LeagueRow, espn_id: int) -> str:
    known = data.players.get(espn_id)
    if known is None:
        found = store.players.many(league.sport, [espn_id])
        known = found[0] if found else None
    return known.full_name if known is not None else f"player {espn_id}"


# --- rendering --------------------------------------------------------------------------------------------------------


def render_report_card(card: ReportCard, *, weeks: int = REPORT_CARD_WEEKS) -> list[str]:
    """The card as markdown lines (no heading level above ``###``)."""
    lines = ["### Report card", ""]
    if not card.weeks:
        return [*lines, "Not available (see notes)."]
    recent = card.weeks[-weeks:]
    lines.append(
        f"Lineup efficiency {percent(card.efficiency)} over {len(card.weeks)} played "
        f"week{'s' if len(card.weeks) != 1 else ''} ({_points(card.points)} of {_points(card.optimal)} points). "
        f"{_points(card.left)} points left on the bench against the hindsight lineup "
        f"({_points(card.left / len(card.weeks))} a week, {card.swaps} start/sit swaps); benched players scored "
        f"{_points(card.bench)}."
    )
    lines += [
        "",
        "| Period | Points | Optimal | Efficiency | Bench scored | Left on bench | Swaps |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {w.period} | {_points(w.points)} | {_points(w.optimal)} | {percent(w.efficiency)} | {_points(w.bench)} "
        f"| {_points(w.left)} | {w.swaps} |"
        for w in recent
    ]
    if len(card.weeks) > len(recent):
        lines.append(f"\nLatest {len(recent)} of {len(card.weeks)} weeks shown.")
    lines += ["", "Pickups (points the add scored while on our roster, less what the player dropped scored then):", ""]
    if not card.pickups:
        lines.append("None executed yet.")
    for pickup in card.pickups:
        move = f"{pickup.added}" + (f" for {pickup.dropped}" if pickup.dropped else "")
        if pickup.settled:
            span = f"period{'s' if len(pickup.periods) != 1 else ''} {_span(pickup.periods)}"
            lines.append(
                f"- #{pickup.proposal_id} {pickup.kind} {move}: {_points(pickup.gained)} vs "
                f"{_points(pickup.given_up)} over {span}, net {_points(pickup.net)}"
            )
        else:
            lines.append(f"- #{pickup.proposal_id} {pickup.kind} {move}: no played week yet")
    if any(pickup.settled for pickup in card.pickups):
        lines += ["", f"Pickups net {_points(card.pickup_net)} points."]
    return lines


def _span(periods: tuple[int, ...]) -> str:
    return str(periods[0]) if len(periods) == 1 else f"{periods[0]}-{periods[-1]}"


# --- the weekly report hook -------------------------------------------------------------------------------------------


def attach_report_card(
    report: LeagueReport,
    store: Store,
    league: LeagueRow,
    *,
    now: datetime,
    schedule: ScheduleLike | None = None,
) -> LeagueReport:
    """``report`` with the league's report card added as a section and its notes added to the report's notes."""
    card = build_report_card(store, league, now=now, schedule=schedule)
    block = "\n".join(render_report_card(card))
    return replace(
        report,
        sections=(*report.sections, block),
        notes=tuple(dict.fromkeys([*report.notes, *(f"{report.key}: report card: {note}" for note in card.notes)])),
    )
