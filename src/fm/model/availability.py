"""Availability: the chance a player plays his game in a scoring period, ``p_active`` (DESIGN section 8.2).

``p_active`` is built in four steps, and each step records what went into it in the row's ``inputs`` so a decision
replays (DESIGN principle 5):

1. **Designation.** ESPN's ``player.injuryStatus``, normalised to :class:`fm.espn.ids.InjuryStatus` by the game's id
   maps, gives a base rate per sport (:data:`BASE_RATES`, ROADMAP #15). In NBA, the league's official injury report
   (``fm.sources.nba_injuries``) replaces ESPN's designation when it lists the player for this period's game
   (:class:`OfficialReport`). The report is the team's own statement for that game, refreshed about every 15 minutes,
   and uses the probable / questionable / doubtful / out ladder where ESPN mostly says day-to-day. ``Available``
   counts as healthy.
2. **Hard zeros.** A player gets 0, and nothing below lifts it, when he is not on an active pro roster, has no pro
   team, his team has no game in the period (an NFL bye, an NBA off day), or his designation rules him out (out,
   injured reserve, suspended).
3. **Practice trend (NFL).** The week's practice reports (:class:`PracticeReport`) shift the base rate
   (:func:`practice_trend`). The latest practice day's participation sets the shift (:data:`PRACTICE_SHIFTS`: a full
   practice lifts a questionable player, a missed one lowers anyone). The direction since the week's first report adds
   :data:`TREND_STEP` per level gained or lost, so DNP, LP, FP beats LP, LP, LP, which beats FP, LP, DNP. Veteran rest
   days say nothing about an injury and are left out. nflverse keeps only each player's latest practice status per
   week (:func:`practice_from_nflverse`), so :func:`assess_stored` keeps the week's log in the period's own
   ``availability`` row and adds each day's observation to it.
4. **News (Claude).** The advisor's news signals (``news_signals``, DESIGN section 10) move ``p_active`` by at most
   :data:`NEWS_BOUND` either way (:func:`weigh_news`). Each proposal is clamped to the bound and weighted by its
   confidence. The latest signal of a kind supersedes earlier ones, a signal that cites no source is ignored, and the
   total is clamped to the bound again. A signal counts for the period's game when it was published after the team's
   previous game started and no later than ``as_of`` or this game's start (:func:`news_window`). Every signal
   considered is listed in ``inputs["news"]`` with what became of it and logged (``logging``, this module's logger),
   and ``inputs["news"]["before"]`` keeps the engine's number without Claude, so a decision can tell when a
   Claude-only signal is what moved it (CLAUDE.md: such a signal never triggers a drop or a trade).

Expected value is ``p_active`` times the projection (:func:`expected_points`).

**Late-game pivots.** Starting a player who may sit costs less when a bench player who is eligible for the same slot
can still be swapped in once the starter's status is known. That is the case when the bench player locks later than
the moment the starter's status settles. :func:`resolves_at` puts that moment :data:`RESOLUTION_LEAD` before the
game: NFL inactives come out 90 minutes before kickoff, and NBA late scratches are re-checked 30 minutes before the
tip (DESIGN sections 1 and 9.1). :func:`can_pivot` and :func:`late_pivots` find a starter's pivots, and
:func:`plan_pivots` values a lineup counting the swaps. It holds a distinct pivot for each uncertain starter through
an optimal assignment, so a lineup that puts a late questionable player where a later pivot can cover him (the
FLEX/UTIL trick, DESIGN section 8.2) scores higher than one that does not. Lock times are passed in from the sport
plugin under the league's lock type (``SportPlugin.lock_time``); nothing here assumes one.

The rates, shifts and bounds are model parameters, not league settings: per sport where it matters, overridable per
call (``rates``, :class:`AvailabilityParams`), and starting points for the backtest to fit once outcomes are stored.
A player with no designation (ESPN sends ``ACTIVE``, ``NORMAL`` or nothing, as for every D/ST) counts as active. A
designation the id maps do not know is not treated as healthy. It gets the questionable rate and is flagged in
``inputs`` so the new value gets mapped.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import polars as pl
from nba_api.stats.static import teams as nba_static_teams
from pydantic import BaseModel, ConfigDict, ValidationError
from scipy.optimize import linear_sum_assignment

from fm.config import Sport
from fm.espn.ids import Game, InjuryStatus, ids_for
from fm.model.ids import GSIS, Crosswalk
from fm.model.ids_nba import espn_tricode, normalize_name
from fm.sports.base import FREE_AGENT_TEAM, ScheduleLike, fantasy_day, game_for, is_provisional
from fm.sports.nba import eastern_day
from fm.store import AvailabilityRow, NewsSignalRow, PlayerRow, Store, format_timestamp

if TYPE_CHECKING:
    from fm.sources.nba_injuries import OfficialInjuryEntry, OfficialInjuryReport

logger = logging.getLogger(__name__)

MODEL: Final = "full"
"""The ``inputs["model"]`` tag of rows this module writes. Rows from ROADMAP #15's basic model say ``designation``."""

NFL_RATES: Mapping[InjuryStatus, float] = MappingProxyType(
    {
        InjuryStatus.ACTIVE: 1.0,
        InjuryStatus.PROBABLE: 0.95,
        InjuryStatus.QUESTIONABLE: 0.70,
        InjuryStatus.DOUBTFUL: 0.05,
        InjuryStatus.OUT: 0.0,
        InjuryStatus.DAY_TO_DAY: 0.75,
        InjuryStatus.INJURY_RESERVE: 0.0,
        InjuryStatus.SUSPENSION: 0.0,
        InjuryStatus.UNKNOWN: 1.0,
    }
)

NBA_RATES: Mapping[InjuryStatus, float] = MappingProxyType(
    {
        InjuryStatus.ACTIVE: 1.0,
        InjuryStatus.PROBABLE: 0.95,
        InjuryStatus.QUESTIONABLE: 0.50,
        InjuryStatus.DOUBTFUL: 0.10,
        InjuryStatus.OUT: 0.0,
        InjuryStatus.DAY_TO_DAY: 0.60,
        InjuryStatus.INJURY_RESERVE: 0.0,
        InjuryStatus.SUSPENSION: 0.0,
        InjuryStatus.UNKNOWN: 1.0,
    }
)

BASE_RATES: Mapping[Sport, Mapping[InjuryStatus, float]] = MappingProxyType({"nfl": NFL_RATES, "nba": NBA_RATES})
"""``p_active`` by designation per sport (``nfl`` / ``nba``), before the schedule and roster status apply."""

UNRECOGNIZED_AS: Final = InjuryStatus.QUESTIONABLE
"""Whose rate a designation the id maps do not know gets: something was reported, so not the healthy rate."""

INACTIVE_DESIGNATIONS: frozenset[InjuryStatus] = frozenset(
    {InjuryStatus.OUT, InjuryStatus.INJURY_RESERVE, InjuryStatus.SUSPENSION}
)
"""Designations that mean the player will not play, whatever else is known."""

# ``inputs["reason"]`` when p_active is forced to 0 or comes from a non-playing designation.
REASON_INACTIVE: Final = "inactive"
REASON_NO_TEAM: Final = "no_team"
REASON_NO_GAME: Final = "no_game"
REASON_DESIGNATION: Final = "designation"

# ``inputs["designation_source"]``: whose designation set the base rate.
SOURCE_ESPN: Final = "espn"
SOURCE_OFFICIAL: Final = "official"


class Participation(StrEnum):
    """A day's participation on the NFL's practice report."""

    DNP = "DNP"
    """Did not participate."""
    LIMITED = "LP"
    FULL = "FP"


PARTICIPATION_RANK: Mapping[Participation, int] = MappingProxyType(
    {Participation.DNP: 0, Participation.LIMITED: 1, Participation.FULL: 2}
)
"""Participation levels in order; the trend counts the steps between the week's first and latest report."""

PRACTICE_SHIFTS: Mapping[Participation, float] = MappingProxyType(
    {Participation.FULL: 0.15, Participation.LIMITED: 0.0, Participation.DNP: -0.25}
)
"""How the latest practice day's participation shifts the base rate. With the questionable rate of 0.70 these give
0.85 after a full practice and 0.45 after a missed one, and a healthy player who misses practice drops to 0.75 until
his designation comes out. Starting points, like the base rates, for the backtest to fit."""

TREND_STEP: Final = 0.05
"""The further shift per participation level gained (or lost) between the week's first and latest practice day."""

NEWS_BOUND: Final = 0.3
"""The most Claude's news signals may move ``p_active`` either way, per signal and in total (DESIGN section 10)."""

NEWS_LOOKBACK: Mapping[Sport, timedelta] = MappingProxyType({"nfl": timedelta(days=7), "nba": timedelta(days=2)})
"""How far back news counts when the schedule shows no earlier game for the player's team to start from."""

RESOLUTION_LEAD: Mapping[Sport, timedelta] = MappingProxyType(
    {"nfl": timedelta(minutes=90), "nba": timedelta(minutes=30)}
)
"""How long before his game a player's status for it is known: NFL inactives come out 90 minutes before kickoff, and
NBA late scratches are re-checked 30 minutes before the tip (DESIGN sections 1 and 9.1). Pro-league practice, not a
league setting; lock times themselves come from the schedule and the league's lock type."""

OFFICIAL_STATUSES: Mapping[str, InjuryStatus] = MappingProxyType(
    {
        "OUT": InjuryStatus.OUT,
        "DOUBTFUL": InjuryStatus.DOUBTFUL,
        "QUESTIONABLE": InjuryStatus.QUESTIONABLE,
        "PROBABLE": InjuryStatus.PROBABLE,
        "AVAILABLE": InjuryStatus.ACTIVE,
    }
)
"""The official report's status words (upper case) as designations; ``Available`` means cleared to play."""

OFFICIAL_TEAM_CODES: Mapping[str, str] = MappingProxyType(
    {
        **{str(team["full_name"]): str(team["abbreviation"]) for team in nba_static_teams.get_teams()},
        "LA Clippers": "LAC",
    }
)
"""Team names as the official report prints them to nba.com tricodes (the codes ``fm.espn.ids.FBA`` uses), from
``nba_api``'s bundled table plus the report's own spelling of the Clippers."""

type NewsStatus = Literal["counted", "superseded", "stale", "future", "uncited", "invalid"]
"""What became of a news signal: ``counted`` toward ``p_active``; ``superseded`` by a later signal of its kind;
published before (``stale``) or after (``future``) the window that bears on the game; ``uncited`` (no source URL);
``invalid`` (a non-finite proposal)."""


@dataclass(frozen=True, slots=True)
class AvailabilityParams:
    """The full model's parameters beyond the base rates, which ``rates`` overrides. Like the rates, these are
    starting points for the backtest to fit. Raises ``ValueError`` on construction when a value is out of range."""

    practice_shifts: Mapping[Participation, float] = PRACTICE_SHIFTS
    trend_step: float = TREND_STEP
    news_bound: float = NEWS_BOUND
    news_lookback: Mapping[Sport, timedelta] = NEWS_LOOKBACK

    def __post_init__(self) -> None:
        missing = [level.value for level in Participation if level not in self.practice_shifts]
        if missing:
            raise ValueError(f"practice_shifts has no shift for {', '.join(missing)}")
        for level, shift in self.practice_shifts.items():
            if not -1.0 <= shift <= 1.0:
                raise ValueError(f"the practice shift for {level.value} must be within [-1, 1], got {shift!r}")
        if not 0.0 <= self.trend_step <= 0.5:
            raise ValueError(f"trend_step must be within [0, 0.5], got {self.trend_step!r}")
        if not 0.0 <= self.news_bound <= 1.0:
            raise ValueError(f"news_bound must be within [0, 1], got {self.news_bound!r}")
        for sport in BASE_RATES:
            lookback = self.news_lookback.get(sport)
            if lookback is None or lookback <= timedelta(0):
                raise ValueError(f"news_lookback needs a positive span for {sport}, got {lookback!r}")


DEFAULT_PARAMS: Final = AvailabilityParams()


def _sport(value: Game | str) -> Sport:
    return "nfl" if Game.coerce(value) is Game.FFL else "nba"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _require_aware(at: datetime, what: str) -> None:
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError(f"{what} must be timezone-aware, got {at.isoformat()}")


# --- designations -----------------------------------------------------------------------------------------------------


def designation(raw: str | None, sport: Game | str) -> InjuryStatus:
    """ESPN's raw ``injuryStatus`` normalised through the game's id maps; missing or unrecognised is ``UNKNOWN``."""
    return ids_for(sport).injury_status(raw)


def is_unrecognized(raw: str | None, sport: Game | str) -> bool:
    """True when ESPN sent a designation the id maps do not know (as opposed to sending none)."""
    return bool(raw and raw.strip()) and designation(raw, sport) is InjuryStatus.UNKNOWN


def rates_for(sport: Game | str, overrides: Mapping[InjuryStatus, float] | None = None) -> dict[InjuryStatus, float]:
    """The sport's base rates with ``overrides`` applied. Raises ``ValueError`` for a rate outside ``[0, 1]``."""
    merged = {**BASE_RATES[_sport(sport)], **(overrides or {})}
    for status, rate in merged.items():
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"p_active for {status.value} must be within [0, 1], got {rate!r}")
    return merged


def base_rate(status: InjuryStatus, sport: Game | str, *, rates: Mapping[InjuryStatus, float] | None = None) -> float:
    """``p_active`` for a normalised designation before the schedule applies."""
    return rates_for(sport, rates)[status]


def p_active_for(raw: str | None, sport: Game | str, *, rates: Mapping[InjuryStatus, float] | None = None) -> float:
    """``p_active`` for ESPN's raw ``injuryStatus`` alone (no schedule, no roster status): the designation mapping.

    ``None``, ``ACTIVE`` and ``NORMAL`` are healthy; an unrecognised value gets the :data:`UNRECOGNIZED_AS` rate.
    """
    table = rates_for(sport, rates)
    if is_unrecognized(raw, sport):
        return table[UNRECOGNIZED_AS]
    return table[designation(raw, sport)]


# --- the NBA's official injury report ---------------------------------------------------------------------------------


def _matchup_teams(matchup: str | None) -> frozenset[str]:
    return frozenset(part.strip() for part in matchup.split("@")) if matchup else frozenset()


def _entry_on_team(entry: OfficialInjuryEntry, team: str) -> bool:
    """The entry's team is ``team``: by the team name it prints, or, for a name the table lacks, by its matchup."""
    code = OFFICIAL_TEAM_CODES.get(entry.team) if entry.team else None
    return code == team if code is not None else team in _matchup_teams(entry.matchup)


def _entry_on_day(entry: OfficialInjuryEntry, game_day: date | None) -> bool:
    return game_day is None or entry.game_day == game_day


def _shortens(first: str, other: str) -> bool:
    """One first name is a shortening of the other (``herb`` / ``herbert``), at least two letters long."""
    return first != other and min(len(first), len(other)) >= 2 and (first.startswith(other) or other.startswith(first))


class OfficialReport:
    """The NBA's official injury report (``fm.sources.nba_injuries.OfficialInjuryReport``), indexed to find a player.

    The report has no player ids, so a player's entry is the one with his name as ESPN spells it once normalised
    (``fm.model.ids_nba.normalize_name``: case, accents, punctuation and ``Jr.``/``III`` suffixes aside), on his team,
    for the day of his game. A name that matches no entry is tried once more on his team by surname and a first name
    one spelling shortens (``Herb Jones`` and ``Jones, Herbert``). Several entries for one player and day are a
    conflict: the player is left to ESPN's designation, with a warning.
    """

    def __init__(self, report: OfficialInjuryReport) -> None:
        self.report = report
        self._by_name: dict[str, list[OfficialInjuryEntry]] = {}
        for entry in report.entries:
            self._by_name.setdefault(normalize_name(entry.name), []).append(entry)

    def entry_for(self, player: PlayerRow, game_day: date | None = None) -> OfficialInjuryEntry | None:
        """The player's entry for the game on ``game_day`` (any day when ``None``), or ``None`` when not listed."""
        team = espn_tricode(player)
        if team is None:
            return None
        key = normalize_name(player.full_name)
        matches = [
            entry
            for entry in self._by_name.get(key, ())
            if _entry_on_team(entry, team) and _entry_on_day(entry, game_day)
        ]
        if not matches:
            matches = [entry for entry in self._shortened(key, team) if _entry_on_day(entry, game_day)]
        if len(matches) > 1:
            on_day = f" for {game_day.isoformat()}" if game_day is not None else ""
            logger.warning(
                "availability: the official injury report lists %s (%s) %d times%s; using ESPN's designation",
                player.full_name,
                team,
                len(matches),
                on_day,
            )
            return None
        return matches[0] if matches else None

    def not_submitted(self, player: PlayerRow) -> bool:
        """True when the player's team had not submitted its report yet (``NOT YET SUBMITTED``)."""
        team = espn_tricode(player)
        return team is not None and any(OFFICIAL_TEAM_CODES.get(name) == team for _, name in self.report.not_submitted)

    def _shortened(self, key: str, team: str) -> list[OfficialInjuryEntry]:
        first, _, surname = key.partition(" ")
        if not surname:
            return []
        found: list[OfficialInjuryEntry] = []
        for name, entries in self._by_name.items():
            other_first, _, other_surname = name.partition(" ")
            if other_surname == surname and _shortens(first, other_first):
                found.extend(entry for entry in entries if _entry_on_team(entry, team))
        return found


def _official_index(official: OfficialReport | OfficialInjuryReport | None) -> OfficialReport | None:
    if official is None or isinstance(official, OfficialReport):
        return official
    return OfficialReport(official)


def _official_inputs(
    entry: OfficialInjuryEntry | None, status: InjuryStatus | None, *, not_submitted: bool
) -> dict[str, Any]:
    """``inputs["official"]``: the player's entry, or that he is not listed (and whether his team has reported).
    ``designation`` is ``None`` for a status word the model does not read, which leaves ESPN's designation."""
    if entry is None:
        return {"listed": False, "not_submitted": not_submitted}
    return {
        "listed": True,
        "status": entry.status,
        "designation": status.value if status is not None else None,
        "reason": entry.reason,
        "game_day": entry.game_day.isoformat() if entry.game_day is not None else None,
        "game_time": entry.game_time,
        "matchup": entry.matchup,
        "team": entry.team,
        "player": entry.player,
    }


# --- NFL practice reports ---------------------------------------------------------------------------------------------

_WORD_BREAKS = re.compile(r"[^a-z]+")
_REST = re.compile(r"\brest(?:ing|ed)?\b", re.IGNORECASE)

PRACTICE_COLUMNS: tuple[str, ...] = ("season", "week", "gsis_id", "practice_status")
"""What :func:`practice_from_nflverse` needs from nflverse's injuries frame; the practice injury columns are read when
present."""
PRACTICE_NOTE_COLUMNS: tuple[str, ...] = ("practice_primary_injury", "practice_secondary_injury")


class PracticeReport(BaseModel):
    """One practice day of the NFL's practice report for a player.

    ``day`` orders the week: there is one report per day, and a later report for a day replaces an earlier one
    (:func:`merge_practice`). ``rest`` marks a veteran rest day (nflverse's ``Not injury related - resting player``),
    which says nothing about an injury and is left out of the trend.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    day: date
    participation: Participation
    rest: bool = False
    injury: str | None = None
    source: str | None = None


def participation(raw: str | None) -> Participation | None:
    """A practice status as :class:`Participation`: nflverse's ``Did Not Participate In Practice``, ``Limited
    Participation in Practice`` and ``Full Participation in Practice``, or the short forms (``DNP``, ``LP`` or
    ``Limited``, ``FP`` or ``Full``). ``None`` for a blank or for anything else."""
    words = " ".join(word for word in _WORD_BREAKS.split((raw or "").lower()) if word)
    if words in ("dnp", "did not participate") or words.startswith("did not participate "):
        return Participation.DNP
    if words in ("lp", "limited") or words.startswith("limited participation"):
        return Participation.LIMITED
    if words in ("fp", "full") or words.startswith("full participation"):
        return Participation.FULL
    return None


def is_rest_day(*notes: str | None) -> bool:
    """True when a practice report's injury notes give rest as the reason (``Not injury related - resting player``)."""
    return any(note is not None and _REST.search(note) is not None for note in notes)


def merge_practice(*groups: Iterable[PracticeReport]) -> tuple[PracticeReport, ...]:
    """One report per day, oldest day first. A later report for a day, in a later group or later in the same group,
    replaces an earlier one."""
    by_day: dict[date, PracticeReport] = {}
    for group in groups:
        for report in group:
            by_day[report.day] = report
    return tuple(by_day[day] for day in sorted(by_day))


@dataclass(frozen=True, slots=True)
class PracticeTrend:
    """The week's practice reports read as a shift of ``p_active``.

    ``reports`` are the reports considered, one per day, oldest first. ``first`` and ``latest`` are the participation
    on the week's first and latest days that were not rest days, ``steps`` the levels gained (negative: lost) between
    them, and ``shift`` what the trend adds to the base rate. Without a day that counts, the shift is 0.
    """

    reports: tuple[PracticeReport, ...]
    first: Participation | None
    latest: Participation | None
    steps: int
    shift: float

    def as_inputs(self, *, applied: bool) -> dict[str, Any]:
        """The trend as ``inputs["practice"]``; ``applied`` is false when ``p_active`` was a hard zero."""
        return {
            "reports": [report.model_dump(mode="json") for report in self.reports],
            "first": self.first.value if self.first is not None else None,
            "latest": self.latest.value if self.latest is not None else None,
            "steps": self.steps,
            "shift": self.shift,
            "applied": applied,
        }


def practice_trend(reports: Iterable[PracticeReport], *, params: AvailabilityParams = DEFAULT_PARAMS) -> PracticeTrend:
    """The shift the week's practice reports make: the latest counted day's :data:`PRACTICE_SHIFTS` entry plus
    :data:`TREND_STEP` per participation level gained since the week's first counted day (or minus, per level lost).
    Rest days do not count. Reports for the same day are merged first, the later one winning."""
    merged = merge_practice(reports)
    counted = [report.participation for report in merged if not report.rest]
    if not counted:
        return PracticeTrend(merged, None, None, 0, 0.0)
    first, latest = counted[0], counted[-1]
    steps = PARTICIPATION_RANK[latest] - PARTICIPATION_RANK[first]
    return PracticeTrend(merged, first, latest, steps, params.practice_shifts[latest] + params.trend_step * steps)


def recorded_practice(row: AvailabilityRow | None) -> tuple[PracticeReport, ...]:
    """The practice reports an earlier assessment recorded in ``row.inputs["practice"]``, one per day. A report that
    no longer reads is skipped with a warning; no row, or a row without a log, gives none."""
    practice = row.inputs.get("practice") if row is not None else None
    items = practice.get("reports") if isinstance(practice, dict) else None
    if row is None or not isinstance(items, list):
        return ()
    reports: list[PracticeReport] = []
    for item in items:
        try:
            reports.append(PracticeReport.model_validate(item))
        except ValidationError as exc:
            logger.warning(
                "availability: %s %d period %d: skipped a recorded practice report that does not read (%s)",
                row.sport,
                row.espn_id,
                row.scoring_period_id,
                exc.errors()[0]["msg"] if exc.errors() else exc,
            )
    return merge_practice(reports)


@dataclass(frozen=True, slots=True)
class PracticeReports:
    """Practice reports keyed by ESPN id, plus what could not be read: ``unmapped`` lists the GSIS ids the crosswalk
    lacks (fix them in ``data/id_overrides.csv``), and ``warnings`` summarise every skip for the output."""

    by_player: Mapping[int, tuple[PracticeReport, ...]]
    unmapped: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def practice_from_nflverse(
    frame: pl.DataFrame, crosswalk: Crosswalk, *, season: int, week: int, observed: datetime
) -> PracticeReports:
    """nflverse's injuries dataset (``NflverseSource.injuries``) as each player's practice report for ``week``.

    nflverse keeps one row per player-week with his latest practice status and no practice date. A row is therefore
    the report for the fantasy day it was observed on: ``observed`` is the fetch's ``as_of``, and the day is
    :func:`fm.sports.base.fantasy_day`. Fetched through the practice week, the observations make the week's trend
    once :func:`assess_stored` adds them to the log in each period's row. Rows of other seasons and weeks, and rows
    without a practice status, are skipped. A status the model does not read is skipped with a warning, and a GSIS id
    the crosswalk cannot map is listed in ``unmapped``. Raises ``ValueError`` when a needed column is missing.
    """
    missing = [column for column in PRACTICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"the nflverse injuries frame has no {', '.join(missing)} column")
    _require_aware(observed, "observed")
    day = fantasy_day(observed)
    notes = [column for column in PRACTICE_NOTE_COLUMNS if column in frame.columns]
    rows = frame.filter((pl.col("season") == season) & (pl.col("week") == week)).select([*PRACTICE_COLUMNS, *notes])
    found: dict[int, list[PracticeReport]] = {}
    unmapped: list[str] = []
    unread: dict[str, None] = {}
    for record in rows.iter_rows(named=True):
        status = _text(record["practice_status"])
        if status is None:
            continue
        level = participation(status)
        if level is None:
            unread[status] = None
            continue
        gsis_id = _text(record["gsis_id"])
        espn_id = crosswalk.espn_id(GSIS, gsis_id) if gsis_id is not None else None
        if espn_id is None:
            unmapped.append(gsis_id or "(no GSIS id)")
            continue
        injuries = [_text(record[column]) for column in notes]
        injury = next((note for note in injuries if note is not None and not is_rest_day(note)), None)
        found.setdefault(espn_id, []).append(
            PracticeReport(day=day, participation=level, rest=is_rest_day(*injuries), injury=injury, source="nflverse")
        )
    warnings: list[str] = []
    if unmapped:
        warnings.append(
            f"nflverse injuries: {len(unmapped)} practice reports for GSIS ids the crosswalk does not map "
            f"({', '.join(unmapped[:5])}{' and more' if len(unmapped) > 5 else ''})"
        )
    if unread:
        warnings.append(f"nflverse injuries: practice statuses the model does not read: {', '.join(unread)}")
    by_player = {espn_id: merge_practice(reports) for espn_id, reports in found.items()}
    return PracticeReports(MappingProxyType(by_player), tuple(unmapped), tuple(warnings))


# --- news signals -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NewsEntry:
    """One news signal as the model read it: its ``status``, the proposal clamped to the bound (``bounded``, 0 for an
    invalid one) and that times the signal's confidence (``weighted``), which is what a counted signal adds."""

    signal: NewsSignalRow
    status: NewsStatus
    bounded: float
    weighted: float

    @property
    def clamped(self) -> bool:
        """True when the proposal lay outside the bound."""
        return self.status != "invalid" and self.bounded != self.signal.p_active_delta

    def as_inputs(self) -> dict[str, Any]:
        signal = self.signal
        proposed = signal.p_active_delta
        return {
            "signal_id": signal.id,
            "news_item_id": signal.news_item_id,
            "kind": signal.kind,
            "severity": signal.severity,
            "games_out": signal.games_out,
            "published_at": format_timestamp(signal.published_at),
            "source_url": signal.source_url,
            "proposed": proposed if math.isfinite(proposed) else None,
            "confidence": signal.confidence,
            "bounded": self.bounded,
            "weighted": self.weighted,
            "clamped": self.clamped,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class NewsEffect:
    """What the news signals do to ``p_active``: ``requested`` is the sum of the counted signals' weighted changes and
    ``delta`` that sum clamped to the bound, the change applied. ``entries`` lists every signal considered, in
    publication order."""

    entries: tuple[NewsEntry, ...]
    requested: float
    delta: float
    since: datetime
    until: datetime

    @property
    def clamped(self) -> bool:
        """True when the counted signals together asked for more than the bound."""
        return self.delta != self.requested

    @property
    def counted(self) -> tuple[NewsEntry, ...]:
        return tuple(entry for entry in self.entries if entry.status == "counted")

    def as_inputs(self, *, before: float, applied: bool) -> dict[str, Any]:
        """The effect as ``inputs["news"]``. ``before`` is ``p_active`` without the news, ``applied`` false when it
        was a hard zero that news cannot move."""
        return {
            "before": before,
            "applied": applied,
            "requested": self.requested,
            "delta": self.delta,
            "clamped": self.clamped,
            "since": format_timestamp(self.since),
            "until": format_timestamp(self.until),
            "signals": [entry.as_inputs() for entry in self.entries],
        }


def _team_id(player: PlayerRow) -> int | None:
    return player.pro_team_id if player.pro_team_id != FREE_AGENT_TEAM else None


def _previous_start(schedule: ScheduleLike, team_id: int, scoring_period: int) -> datetime | None:
    """The start of the team's last game in an earlier period of the schedule, or ``None`` when it shows none."""
    for period in sorted((p for p in schedule.scoring_periods if p < scoring_period), reverse=True):
        game = game_for(schedule, team_id, period)
        if game is not None:
            return game.date
    return None


def news_window(
    player: PlayerRow,
    scoring_period: int,
    *,
    as_of: datetime,
    schedule: ScheduleLike | None = None,
    lookback: timedelta | None = None,
) -> tuple[datetime, datetime]:
    """The publication times whose news bears on the player's game in ``scoring_period``, as ``(since, until)``.

    News published since his team's previous game started is about this game: an injury in that game, the week's
    practice, a pregame decision. When the schedule shows no earlier game (or the window would be empty, as for a
    period assessed before the previous game is played), the window opens ``lookback`` before its end instead (the
    sport's :data:`NEWS_LOOKBACK`). It closes at ``as_of``, or at the game's start when that is earlier and not a
    placeholder: news after the kickoff is about the game being played, not whether he plays it.
    """
    _require_aware(as_of, "as_of")
    span = lookback if lookback is not None else NEWS_LOOKBACK[player.sport]
    until = as_of
    since: datetime | None = None
    team_id = _team_id(player)
    if schedule is not None and team_id is not None:
        game = game_for(schedule, team_id, scoring_period)
        if game is not None and not is_provisional(game):
            until = min(until, game.date)
        since = _previous_start(schedule, team_id, scoring_period)
    if since is None or since >= until:
        since = until - span
    return since, until


def weigh_news(
    signals: Iterable[NewsSignalRow], *, since: datetime, until: datetime, bound: float = NEWS_BOUND
) -> NewsEffect:
    """How much the news signals move ``p_active``: a pure reading, for :func:`assess` to apply and log.

    Signals are read in publication order. A non-finite proposal is ``invalid`` and a signal without a source URL is
    ``uncited``; neither counts (DESIGN section 10: an adjustment must cite a source and a time). A signal published
    before ``since`` is ``stale``, after ``until`` ``future``. Of the rest, the latest of each kind (``injury``,
    ``rest``, ...) counts and supersedes the earlier ones, since each reading is the advisor's current view and a
    second report of the same news must not count twice. A counted signal adds its proposal clamped to ``±bound``
    times its confidence, and the sum is clamped to ``±bound`` again.
    """
    entries: list[NewsEntry] = []
    latest: dict[str, int] = {}
    for signal in sorted(signals, key=lambda item: (item.published_at, item.created_at, item.id or 0)):
        proposed = signal.p_active_delta
        if not math.isfinite(proposed):
            entries.append(NewsEntry(signal, "invalid", 0.0, 0.0))
            continue
        bounded = _clamp(proposed, -bound, bound)
        status: NewsStatus
        if not (signal.source_url or "").strip():
            status = "uncited"
        elif signal.published_at < since:
            status = "stale"
        elif signal.published_at > until:
            status = "future"
        else:
            status = "counted"
            earlier = latest.get(signal.kind)
            if earlier is not None:
                entries[earlier] = dataclasses.replace(entries[earlier], status="superseded")
            latest[signal.kind] = len(entries)
        entries.append(NewsEntry(signal, status, bounded, bounded * signal.confidence))
    requested = math.fsum(entry.weighted for entry in entries if entry.status == "counted")
    return NewsEffect(tuple(entries), requested, _clamp(requested, -bound, bound), since, until)


def _log_news(player: PlayerRow, scoring_period: int, effect: NewsEffect, *, applied: bool, bound: float) -> None:
    """One line per signal that did something, or should have: counted ones at INFO, clamps and rejections at WARNING,
    the rest at DEBUG."""
    who = f"availability: {player.sport} {player.espn_id} ({player.full_name}) period {scoring_period}"
    for entry in effect.entries:
        signal = entry.signal
        label = f"news signal {signal.id if signal.id is not None else '(unsaved)'} ({signal.kind})"
        published = format_timestamp(signal.published_at)
        if entry.status == "invalid":
            logger.warning("%s: %s proposes a non-finite change (%r); ignored", who, label, signal.p_active_delta)
            continue
        if entry.status == "uncited":
            logger.warning("%s: %s, published %s, cites no source; ignored", who, label, published)
            continue
        if entry.status != "counted":
            logger.debug("%s: %s, published %s, is %s", who, label, published, entry.status)
            continue
        if entry.clamped:
            logger.warning(
                "%s: %s proposed %+.2f, outside the +/-%.2f bound; clamped to %+.2f",
                who,
                label,
                signal.p_active_delta,
                bound,
                entry.bounded,
            )
        if applied:
            logger.info(
                "%s: %s, published %s, moves p_active %+.2f (%+.2f x confidence %.2f), source %s",
                who,
                label,
                published,
                entry.weighted,
                entry.bounded,
                signal.confidence,
                signal.source_url,
            )
        else:
            logger.info("%s: %s, published %s, not applied: p_active is a hard zero", who, label, published)
    if effect.clamped:
        logger.warning(
            "%s: news signals together asked for %+.2f; clamped to %+.2f", who, effect.requested, effect.delta
        )


# --- assessing players ------------------------------------------------------------------------------------------------


def _require_covered(schedule: ScheduleLike, scoring_period: int) -> None:
    """A schedule with no games in the period answers "no game" for everyone only if the period lies inside it (an
    All-Star break day); outside it the schedule simply does not cover the period, which is the caller's mistake."""
    if schedule.games(scoring_period):
        return
    periods = schedule.scoring_periods
    if periods and min(periods) <= scoring_period <= max(periods):
        return
    covered = f"periods {min(periods)}-{max(periods)}" if periods else "no periods"
    raise ValueError(
        f"the pro schedule covers {covered}, not scoring period {scoring_period}; pass the season's schedule "
        "(or none, to assume every player has a game)"
    )


def _check_inputs(
    player: PlayerRow,
    reports: Sequence[PracticeReport],
    official: OfficialReport | OfficialInjuryReport | None,
    signals: Sequence[NewsSignalRow],
) -> None:
    who = f"{player.full_name} ({player.sport} {player.espn_id})"
    if reports and player.sport != "nfl":
        raise ValueError(f"practice reports are an NFL input; {who} is not an NFL player")
    if official is not None and player.sport != "nba":
        raise ValueError(f"the official injury report is the NBA's; {who} is not an NBA player")
    for signal in signals:
        if (signal.sport, signal.espn_id) != (player.sport, player.espn_id):
            raise ValueError(
                f"news signal {signal.id} is about {signal.sport} player {signal.espn_id}, not {who}; "
                "pass each player his own signals"
            )


def assess(
    player: PlayerRow,
    *,
    season: int,
    scoring_period: int,
    as_of: datetime,
    schedule: ScheduleLike | None = None,
    rates: Mapping[InjuryStatus, float] | None = None,
    practice: Iterable[PracticeReport] = (),
    official: OfficialReport | OfficialInjuryReport | None = None,
    signals: Iterable[NewsSignalRow] = (),
    params: AvailabilityParams | None = None,
) -> AvailabilityRow:
    """``p_active`` for one player in one scoring period, as a store row (see the module docs for the four steps).

    With a ``schedule`` (the sport's pro schedule, ``fm.espn.models.ProSchedule``), a player whose pro team has no game
    in the period, or who has no pro team (none, or ESPN's free-agent team 0), gets ``has_game=False`` and
    ``p_active=0``. Otherwise ``game_time`` is the game's start, and ``inputs`` names the game, whether its start is
    still provisional, and when his status settles (``resolves_at``). A schedule that does not cover the period raises
    ``ValueError``. Without a schedule the game is assumed (``has_game=True``, no ``game_time``), ``inputs["schedule"]``
    says so, and the official report's entry is taken for the game whatever its day. A player not on an active pro
    roster (``PlayerRow.active`` false) gets 0 regardless of designation.

    ``practice`` (NFL) are the week's practice reports, ``official`` (NBA) the latest official injury report, and
    ``signals`` this player's news signals; each raises ``ValueError`` when given for the wrong sport or player.
    """
    sport = player.sport
    tuning = params if params is not None else DEFAULT_PARAMS
    table = rates_for(sport, rates)
    reports = tuple(practice)
    news = tuple(signals)
    _check_inputs(player, reports, official, news)

    raw = player.injury_status
    status = designation(raw, sport)
    unrecognized = is_unrecognized(raw, sport)
    inputs: dict[str, Any] = {
        "model": MODEL,
        "designation_raw": raw,
        "designation": status.value,
        "active": player.active,
        "pro_team_id": player.pro_team_id,
        "schedule": schedule is not None,
    }
    if unrecognized:
        inputs["designation_unrecognized"] = True

    team_id = _team_id(player)
    has_game = team_id is not None
    game_time: datetime | None = None
    if schedule is not None:
        _require_covered(schedule, scoring_period)
        game = game_for(schedule, team_id, scoring_period) if team_id is not None else None
        has_game = game is not None
        if game is not None:
            game_time = game.date
            inputs["game_id"] = game.id
            inputs["provisional"] = is_provisional(game)
    inputs["has_game"] = has_game
    if game_time is not None:
        inputs["resolves_at"] = format_timestamp(game_time - RESOLUTION_LEAD[sport])

    effective = status
    rate = table[UNRECOGNIZED_AS] if unrecognized else table[status]
    source = SOURCE_ESPN
    index = _official_index(official)
    if index is not None and has_game:
        entry = index.entry_for(player, eastern_day(game_time) if game_time is not None else None)
        listed = OFFICIAL_STATUSES.get(entry.status.upper()) if entry is not None else None
        if entry is not None and listed is None:
            logger.warning("availability: official report status %r is not one the model reads", entry.status)
        if listed is not None:
            effective, rate, source = listed, table[listed], SOURCE_OFFICIAL
        inputs["official"] = _official_inputs(entry, listed, not_submitted=index.not_submitted(player))
    inputs["designation_source"] = source
    inputs["base_rate"] = rate

    reason: str | None = None
    if not player.active:
        reason = REASON_INACTIVE
    elif team_id is None:
        reason = REASON_NO_TEAM
    elif not has_game:
        reason = REASON_NO_GAME
    elif effective in INACTIVE_DESIGNATIONS:
        reason = REASON_DESIGNATION
    hard_zero = reason is not None
    p_active = rate if reason in (None, REASON_DESIGNATION) else 0.0
    if reason is not None:
        inputs["reason"] = reason

    if reports:
        trend = practice_trend(reports, params=tuning)
        if not hard_zero:
            p_active = _clamp(p_active + trend.shift)
        inputs["practice"] = trend.as_inputs(applied=not hard_zero)
    if news:
        since, until = news_window(
            player, scoring_period, as_of=as_of, schedule=schedule, lookback=tuning.news_lookback[sport]
        )
        effect = weigh_news(news, since=since, until=until, bound=tuning.news_bound)
        before = p_active
        if not hard_zero:
            p_active = _clamp(p_active + effect.delta)
        inputs["news"] = effect.as_inputs(before=before, applied=not hard_zero)
        _log_news(player, scoring_period, effect, applied=not hard_zero, bound=tuning.news_bound)

    shown = raw or source == SOURCE_OFFICIAL
    return AvailabilityRow(
        sport=sport,
        espn_id=player.espn_id,
        season=season,
        scoring_period_id=scoring_period,
        designation=effective.value if shown else None,
        p_active=p_active,
        has_game=has_game,
        game_time=game_time,
        inputs=inputs,
        as_of=as_of,
    )


def assess_many(
    players: Iterable[PlayerRow],
    *,
    season: int,
    scoring_period: int,
    as_of: datetime,
    schedule: ScheduleLike | None = None,
    rates: Mapping[InjuryStatus, float] | None = None,
    practice: Mapping[int, Iterable[PracticeReport]] | None = None,
    official: OfficialReport | OfficialInjuryReport | None = None,
    signals: Iterable[NewsSignalRow] = (),
    params: AvailabilityParams | None = None,
) -> list[AvailabilityRow]:
    """:func:`assess` for many players, in the order given. ``practice`` is keyed by ESPN id and goes to NFL players,
    ``official`` to NBA players, and each player gets the ``signals`` about him (signals about anyone else are
    ignored)."""
    index = _official_index(official)
    by_player: dict[tuple[Sport, int], list[NewsSignalRow]] = {}
    for signal in signals:
        by_player.setdefault((signal.sport, signal.espn_id), []).append(signal)
    reports = practice or {}
    return [
        assess(
            player,
            season=season,
            scoring_period=scoring_period,
            as_of=as_of,
            schedule=schedule,
            rates=rates,
            practice=reports.get(player.espn_id, ()) if player.sport == "nfl" else (),
            official=index if player.sport == "nba" else None,
            signals=by_player.get((player.sport, player.espn_id), ()),
            params=params,
        )
        for player in players
    ]


def assess_stored(
    store: Store,
    sport: Game | str,
    espn_ids: Iterable[int],
    *,
    season: int,
    scoring_period: int,
    as_of: datetime,
    schedule: ScheduleLike | None = None,
    rates: Mapping[InjuryStatus, float] | None = None,
    practice: Mapping[int, Iterable[PracticeReport]] | None = None,
    official: OfficialReport | OfficialInjuryReport | None = None,
    signals: Iterable[NewsSignalRow] | None = None,
    params: AvailabilityParams | None = None,
    save: bool = True,
) -> list[AvailabilityRow]:
    """Assess the stored players with the given ESPN ids (ids the ``players`` table lacks are skipped), by ESPN id,
    and unless ``save`` is false upsert the rows into ``availability``, replacing each player's row for the period.

    NFL practice reports accumulate: each player's new ``practice`` reports join the ones his stored row for the period
    recorded (a new report for a day replaces the old one), so daily observations build the week's trend. ``signals``
    defaults to each player's signals in the store (``news_signals``) from the start of his :func:`news_window`.
    """
    sport_key = _sport(sport)
    tuning = params if params is not None else DEFAULT_PARAMS
    index = _official_index(official) if sport_key == "nba" else None
    given = practice or {}
    by_player: dict[int, list[NewsSignalRow]] = {}
    for signal in signals or ():
        if signal.sport == sport_key:
            by_player.setdefault(signal.espn_id, []).append(signal)
    rows: list[AvailabilityRow] = []
    for player in store.players.many(sport_key, espn_ids):
        reports: tuple[PracticeReport, ...] = ()
        if sport_key == "nfl":
            stored = store.availability.get(sport_key, player.espn_id, season, scoring_period)
            reports = merge_practice(recorded_practice(stored), given.get(player.espn_id, ()))
        if signals is None:
            since, _ = news_window(
                player, scoring_period, as_of=as_of, schedule=schedule, lookback=tuning.news_lookback[sport_key]
            )
            player_signals: Sequence[NewsSignalRow] = store.news_signals.for_player(
                sport_key, player.espn_id, since=since
            )
        else:
            player_signals = by_player.get(player.espn_id, [])
        rows.append(
            assess(
                player,
                season=season,
                scoring_period=scoring_period,
                as_of=as_of,
                schedule=schedule,
                rates=rates,
                practice=reports,
                official=index,
                signals=player_signals,
                params=tuning,
            )
        )
    if save and rows:
        store.availability.upsert_many(rows)
    return rows


def expected_points(points: float, availability: AvailabilityRow | float) -> float:
    """Expected value of a projection: ``p_active`` times its points (DESIGN section 8.2)."""
    p_active = availability.p_active if isinstance(availability, AvailabilityRow) else float(availability)
    if not 0.0 <= p_active <= 1.0:
        raise ValueError(f"p_active must be within [0, 1], got {p_active!r}")
    return p_active * points


# --- late-game pivots -------------------------------------------------------------------------------------------------


def resolves_at(availability: AvailabilityRow, *, lead: timedelta | None = None) -> datetime | None:
    """When the player's status for the period's game will be known: ``lead`` (the sport's :data:`RESOLUTION_LEAD`)
    before the game starts. ``None`` without a game time."""
    if availability.game_time is None:
        return None
    return availability.game_time - (lead if lead is not None else RESOLUTION_LEAD[availability.sport])


def is_uncertain(availability: AvailabilityRow | float) -> bool:
    """True when the player may or may not play (``0 < p_active < 1``): the starters a late pivot can cover."""
    p_active = availability.p_active if isinstance(availability, AvailabilityRow) else float(availability)
    return 0.0 < p_active < 1.0


@dataclass(frozen=True, slots=True)
class LineupPlayer:
    """A rostered player as the late-swap logic sees him: his availability row, his projected points under the
    league's scoring, the slot ids he may fill (``PlayerRow.eligible_slot_ids``) and when he locks (the sport
    plugin's ``lock_time`` for his team under the league's lineup lock type; ``None`` when he never locks)."""

    availability: AvailabilityRow
    points: float
    eligible_slot_ids: frozenset[int]
    lock: datetime | None

    @property
    def espn_id(self) -> int:
        return self.availability.espn_id

    @property
    def expected(self) -> float:
        return expected_points(self.points, self.availability)


def can_pivot(starter: LineupPlayer, slot_id: int, pivot: LineupPlayer, *, lead: timedelta | None = None) -> bool:
    """True when ``pivot`` could take ``slot_id`` from ``starter`` once the starter's status is known.

    The starter must be uncertain (:func:`is_uncertain`) with a settled start time (not a provisional placeholder,
    which would make his status look known too early) and still unlocked when his status settles
    (:func:`resolves_at`). The pivot must be someone else who may fill the slot, has expected points, and locks
    strictly after that moment, so the swap can still be made.
    """
    if pivot.espn_id == starter.espn_id or slot_id not in pivot.eligible_slot_ids:
        return False
    if not is_uncertain(starter.availability) or starter.availability.inputs.get("provisional"):
        return False
    settles = resolves_at(starter.availability, lead=lead)
    if settles is None or (starter.lock is not None and starter.lock <= settles):
        return False
    return pivot.lock is not None and pivot.lock > settles and pivot.expected > 0.0


def late_pivots(
    starter: LineupPlayer, slot_id: int, bench: Iterable[LineupPlayer], *, lead: timedelta | None = None
) -> list[LineupPlayer]:
    """The bench players who could replace ``starter`` in ``slot_id`` once his status is known (:func:`can_pivot`),
    best first: most expected points, then the latest lock, then ESPN id."""
    feasible = [pivot for pivot in bench if can_pivot(starter, slot_id, pivot, lead=lead)]
    return sorted(
        feasible,
        key=lambda pivot: (-pivot.expected, -(pivot.lock.timestamp() if pivot.lock else 0.0), pivot.espn_id),
    )


@dataclass(frozen=True, slots=True)
class PivotPlan:
    """A lineup's expected points counting late swaps. ``base`` is the sum of the starters' expected points,
    ``expected`` adds each held pivot's expected points times the chance its starter sits, and ``pivots`` maps each
    covered starter's ESPN id to the bench player held for him."""

    base: float
    expected: float
    pivots: Mapping[int, int]

    @property
    def gain(self) -> float:
        return self.expected - self.base


def plan_pivots(
    lineup: Iterable[tuple[int, LineupPlayer]], bench: Iterable[LineupPlayer], *, lead: timedelta | None = None
) -> PivotPlan:
    """Value a lineup, given as ``(slot_id, player)`` pairs, counting the late swaps its bench allows.

    Each uncertain starter may hold one bench player as his pivot (:func:`can_pivot`), and no bench player covers two
    starters. The pairing maximises the total gain, ``(1 - p_active)`` of the starter times the pivot's expected
    points, through an optimal assignment rather than a greedy one: a late starter can only use a late pivot, so the
    best pivot overall may be worth more to him than to an early starter with other options. Comparing the plans of two
    lineups is how a lineup prefers to keep a late questionable player where a later pivot can cover him.
    """
    starters = list(lineup)
    starting = {player.espn_id for _, player in starters}
    reserves = [player for player in bench if player.espn_id not in starting]
    base = math.fsum(player.expected for _, player in starters)
    if not starters or not reserves:
        return PivotPlan(base, base, MappingProxyType({}))
    gains = np.zeros((len(starters), len(reserves)))
    for row, (slot_id, starter) in enumerate(starters):
        sits = 1.0 - starter.availability.p_active
        for column, pivot in enumerate(reserves):
            if can_pivot(starter, slot_id, pivot, lead=lead):
                gains[row, column] = sits * pivot.expected
    rows, columns = linear_sum_assignment(gains, maximize=True)
    pivots: dict[int, int] = {}
    gained: list[float] = []
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        gain = float(gains[row, column])
        if gain > 0.0:
            pivots[starters[row][1].espn_id] = reserves[column].espn_id
            gained.append(gain)
    return PivotPlan(base, base + math.fsum(gained), MappingProxyType(pivots))
