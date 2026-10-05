"""Parse ESPN's ``mSettings`` league view into :class:`LeagueSettings`.

League settings are data (CLAUDE.md): scoring items, roster slots, lock behavior, acquisition/FAAB/waiver timing, the
trade deadline and playoff weeks all come from here, never from code. One parser serves both games and every format
(PPR, half-PPR and custom scoring; NBA points, H2H categories and roto).

Input is the JSON object returned by ``.../leagues/{leagueId}?view=mSettings`` (DESIGN section 6.1): ``settings``
holds the league configuration, ``status`` the season position, and ``gameId`` / ``id`` / ``seasonId`` /
``scoringPeriodId`` sit at the top level. ESPN keys numeric ids as strings in JSON; they are integers here.

Parsing is strict about the blocks a decision depends on (``scoringSettings``, ``rosterSettings``,
``scheduleSettings``) and forgiving about enumerations: an unseen ``scoringType``, lock type or acquisition type maps
to ``UNKNOWN`` while the raw string is kept, so a new ESPN value surfaces in output instead of crashing a sync. The
exact lineup-lock key and value set is an open unknown that the real-league capture (ROADMAP #14) settles; this module
reads ``rosterSettings.lineupLocktimeType`` and keeps ``rosterLocktimeType`` raw. The shape of
``rosterSettings.lineupSlotStatLimits`` (NBA games-played caps) is another: the parser accepts the bare integer per
slot and stat that the fixtures use and raises on anything else, because a cap dropped silently makes lineups illegal.

Times: ``tradeSettings.deadlineDate`` is epoch milliseconds and becomes an aware UTC datetime. ``waiverProcessHour``
is an hour of the day in US Eastern time, as ESPN configures it; turning it into instants is the deadline job's work.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fm.espn.ids import Game, IdMaps, ids_for

VIEW = "mSettings"
SLOT_STAT_LIMITS = "rosterSettings.lineupSlotStatLimits"
FBA_GAMES_PLAYED_STAT = 42  # fba stat id ``GP``, the stat behind NBA games-played limits


class SettingsParseError(ValueError):
    """The payload is not a usable ``mSettings`` response."""


class _TolerantEnum(StrEnum):
    """A ``StrEnum`` whose unknown values resolve to the ``UNKNOWN`` member instead of raising."""

    @classmethod
    def _missing_(cls, value: object) -> Any:
        return cls.__members__.get("UNKNOWN")


class ScoringType(_TolerantEnum):
    """ESPN ``scoringSettings.scoringType``."""

    H2H_POINTS = "H2H_POINTS"
    H2H_CATEGORY = "H2H_CATEGORY"  # each category is its own win/loss
    H2H_MOST_CATEGORIES = "H2H_MOST_CATEGORIES"
    ROTO = "ROTO"
    TOTAL_SEASON_POINTS = "TOTAL_SEASON_POINTS"
    UNKNOWN = "UNKNOWN"


class ScoringKind(StrEnum):
    """Points leagues compute fantasy points from stat lines; category leagues compare the stats themselves."""

    POINTS = "points"
    CATEGORIES = "categories"


_SCORING_KINDS: Mapping[ScoringType, ScoringKind] = {
    ScoringType.H2H_POINTS: ScoringKind.POINTS,
    ScoringType.TOTAL_SEASON_POINTS: ScoringKind.POINTS,
    ScoringType.H2H_CATEGORY: ScoringKind.CATEGORIES,
    ScoringType.H2H_MOST_CATEGORIES: ScoringKind.CATEGORIES,
    ScoringType.ROTO: ScoringKind.CATEGORIES,
}


class LockType(_TolerantEnum):
    """When a lineup slot locks: at each player's own game, or for everyone at the week's first game."""

    INDIVIDUAL_GAME = "INDIVIDUAL_GAME"
    FIRST_GAME_OF_WEEK = "FIRST_GAME_OF_WEEK"
    UNKNOWN = "UNKNOWN"


class AcquisitionType(_TolerantEnum):
    """ESPN ``acquisitionSettings.acquisitionType``: how dropped and unowned players are claimed."""

    WAIVERS_TRADITIONAL = "WAIVERS_TRADITIONAL"
    WAIVERS_CONTINUOUS = "WAIVERS_CONTINUOUS"
    FREEAGENT = "FREEAGENT"
    UNKNOWN = "UNKNOWN"


class SlotKind(StrEnum):
    ACTIVE = "active"
    BENCH = "bench"
    IR = "ir"


class ScoringItem(BaseModel):
    """One ``scoringItems`` entry: a stat and what it is worth (points) or whether lower is better (categories)."""

    model_config = ConfigDict(frozen=True)

    stat_id: int
    stat: str
    label: str
    points: float = 0.0
    points_overrides: dict[int, float] = Field(default_factory=dict)  # position id -> points (e.g. D/ST = 16 in ffl)
    is_reverse: bool = False  # category leagues: lower wins (TO)

    def points_for(self, position_id: int | None = None) -> float:
        """Points per unit of this stat for a player at ``position_id``, honoring per-position overrides."""
        if position_id is not None and position_id in self.points_overrides:
            return self.points_overrides[position_id]
        return self.points


class LineupSlot(BaseModel):
    """A roster slot the league uses, with how many of it each team has."""

    model_config = ConfigDict(frozen=True)

    slot_id: int
    label: str
    count: int
    kind: SlotKind


class AcquisitionSettings(BaseModel):
    """Adds, drops, waivers and FAAB (``acquisitionSettings``). ``None`` limits mean unlimited."""

    model_config = ConfigDict(frozen=True)

    type: AcquisitionType
    type_raw: str | None
    uses_faab: bool
    budget: int | None  # FAAB dollars per season; None when the league does not use FAAB
    minimum_bid: int
    season_limit: int | None  # acquisitionLimit
    matchup_limit: int | None  # matchupAcquisitionLimit as a fixed per-matchup cap; None when unlimited or per-period
    matchup_limit_per_scoring_period: bool
    waiver_hours: int | None  # how long a dropped player sits on waivers
    waiver_process_days: tuple[str, ...]  # e.g. ("WEDNESDAY",)
    waiver_process_hour: int | None  # hour of day, US Eastern
    waiver_order_reset: bool | None
    transaction_locking_enabled: bool | None
    # With matchupLimitPerScoringPeriod, ESPN sends a per-period rate (NBA: 3 per weekly matchup arrives as 3/7).
    matchup_limit_rate: float | None = None

    def matchup_limit_for(self, scoring_periods: int) -> int | None:
        """Acquisitions allowed in a matchup spanning ``scoring_periods`` scoring periods; ``None`` means unlimited."""
        if self.matchup_limit_rate is not None:
            return math.floor(round(self.matchup_limit_rate * scoring_periods, 6))
        return self.matchup_limit


class TradeSettings(BaseModel):
    """``tradeSettings``. ``deadline`` is an aware UTC datetime, or ``None`` when the league has no deadline."""

    model_config = ConfigDict(frozen=True)

    deadline: datetime | None
    max_trades: int | None
    revision_hours: int | None
    veto_votes_required: int | None

    def is_open(self, at: datetime) -> bool:
        """True when trades are still allowed at ``at`` (an aware datetime)."""
        return self.deadline is None or at < self.deadline


class ScheduleSettings(BaseModel):
    """``scheduleSettings``: matchup periods, the scoring periods inside them, and the playoff structure.

    A scoring period is a week in ``ffl`` and a day in ``fba``. Matchup periods above ``regular_season_matchups`` are
    playoff rounds.
    """

    model_config = ConfigDict(frozen=True)

    regular_season_matchups: int
    matchup_periods: dict[int, tuple[int, ...]]  # matchup period -> scoring periods
    playoff_team_count: int
    playoff_matchup_period_length: int | None
    variable_playoff_matchup_period_length: bool
    playoff_seeding_rule: str | None

    @property
    def playoff_matchup_periods(self) -> tuple[int, ...]:
        return tuple(sorted(mp for mp in self.matchup_periods if mp > self.regular_season_matchups))

    @property
    def playoff_scoring_periods(self) -> tuple[int, ...]:
        return tuple(sp for mp in self.playoff_matchup_periods for sp in self.matchup_periods[mp])

    def scoring_periods(self, matchup_period: int) -> tuple[int, ...]:
        """Scoring periods in a matchup period; empty when ESPN does not list it."""
        return self.matchup_periods.get(matchup_period, ())

    def matchup_period_for(self, scoring_period: int) -> int | None:
        """The matchup period containing a scoring period, or ``None`` outside the schedule."""
        for matchup_period, periods in self.matchup_periods.items():
            if scoring_period in periods:
                return matchup_period
        return None

    def is_playoff(self, matchup_period: int) -> bool:
        return matchup_period > self.regular_season_matchups


class LeagueSettings(BaseModel):
    """Everything a decision needs to know about a league's rules, parsed from ``mSettings``."""

    model_config = ConfigDict(frozen=True)

    game: Game
    league_id: int
    season: int
    name: str
    team_count: int
    current_scoring_period: int | None
    current_matchup_period: int | None
    first_scoring_period: int | None
    final_scoring_period: int | None
    scoring_type: ScoringType
    scoring_type_raw: str | None
    scoring_kind: ScoringKind
    scoring_items: tuple[ScoringItem, ...]
    lineup_slots: tuple[LineupSlot, ...]  # slots with count > 0, by slot id
    position_limits: dict[int, int | None]  # position id -> max rostered; None = unlimited, 0 = not allowed
    slot_stat_limits: dict[int, dict[int, int]]  # slot id -> stat id -> season cap (NBA games-played limits)
    lineup_lock_type: LockType
    lineup_lock_type_raw: str | None
    roster_lock_type_raw: str | None
    acquisition: AcquisitionSettings
    trade: TradeSettings
    schedule: ScheduleSettings

    @property
    def ids(self) -> IdMaps:
        return ids_for(self.game)

    @property
    def is_points(self) -> bool:
        return self.scoring_kind is ScoringKind.POINTS

    @property
    def is_categories(self) -> bool:
        return self.scoring_kind is ScoringKind.CATEGORIES

    @property
    def categories(self) -> tuple[str, ...]:
        """Stat abbreviations the league competes on; empty for points leagues."""
        if not self.is_categories:
            return ()
        return tuple(item.stat for item in self.scoring_items)

    def scoring_item(self, stat: int | str) -> ScoringItem | None:
        """Find a scoring item by stat id or abbreviation."""
        for item in self.scoring_items:
            if item.stat_id == stat or item.stat == stat:
                return item
        return None

    def points_for(self, stat: int | str, position_id: int | None = None) -> float:
        """Points per unit of a stat (0.0 when the league does not score it)."""
        item = self.scoring_item(stat)
        return item.points_for(position_id) if item else 0.0

    @property
    def slot_counts(self) -> dict[int, int]:
        return {slot.slot_id: slot.count for slot in self.lineup_slots}

    def slot_count(self, slot: int | str) -> int:
        """Slots of a kind per team, by id, label or alias (``FLEX``, ``UTIL``); 0 when the league does not use it.

        An unknown label raises ``KeyError`` (from the id map); an unknown id is simply unused and returns 0.
        """
        slot_id = slot if isinstance(slot, int) else self.ids.slot_id(slot)
        return self.slot_counts.get(slot_id, 0)

    @property
    def active_slots(self) -> tuple[LineupSlot, ...]:
        return tuple(slot for slot in self.lineup_slots if slot.kind is SlotKind.ACTIVE)

    @property
    def active_slot_count(self) -> int:
        return sum(slot.count for slot in self.active_slots)

    @property
    def bench_count(self) -> int:
        return sum(slot.count for slot in self.lineup_slots if slot.kind is SlotKind.BENCH)

    @property
    def ir_count(self) -> int:
        return sum(slot.count for slot in self.lineup_slots if slot.kind is SlotKind.IR)

    @property
    def roster_size(self) -> int:
        """Active plus bench slots; IR is extra."""
        return self.active_slot_count + self.bench_count

    def position_limit(self, position_id: int) -> int | None:
        """Max rostered players at a position; ``None`` means unlimited (also when ESPN lists no limit)."""
        return self.position_limits.get(position_id)

    def games_played_limit(self, slot_id: int) -> int | None:
        """NBA season games-played cap for a slot, or ``None`` when the league has none."""
        return self.slot_stat_limits.get(slot_id, {}).get(FBA_GAMES_PLAYED_STAT)

    @property
    def playoff_matchup_periods(self) -> tuple[int, ...]:
        return self.schedule.playoff_matchup_periods


# --- parsing ----------------------------------------------------------------------------------------------------------


def load_league_settings(path: str | Path, *, game: Game | str | None = None) -> LeagueSettings:
    """Parse a saved ``mSettings`` response (a fixture or a raw-capture file)."""
    with Path(path).open(encoding="utf-8") as handle:
        return parse_league_settings(json.load(handle), game=game)


def parse_league_settings(view: Mapping[str, Any], *, game: Game | str | None = None) -> LeagueSettings:
    """Parse the ``mSettings`` response for a league.

    ``game`` may be given when the response lacks ``gameId``; when both are present they must agree.
    """
    settings = _mapping(view, "settings")
    raw_status = view.get("status")
    status: Mapping[str, Any] = raw_status if isinstance(raw_status, Mapping) else {}
    resolved_game = _resolve_game(view, game)
    ids = ids_for(resolved_game)

    scoring = _mapping(settings, "scoringSettings")
    roster = _mapping(settings, "rosterSettings")
    schedule = _mapping(settings, "scheduleSettings")
    acquisition = settings.get("acquisitionSettings") or {}
    trade = settings.get("tradeSettings") or {}

    scoring_type_raw = _optional_str(scoring.get("scoringType"))
    scoring_type = ScoringType(scoring_type_raw) if scoring_type_raw else ScoringType.UNKNOWN
    scoring_items = _parse_scoring_items(scoring.get("scoringItems") or (), ids)
    scoring_kind = _SCORING_KINDS.get(scoring_type) or _infer_scoring_kind(scoring_items)

    lineup_lock_raw = _optional_str(roster.get("lineupLocktimeType"))

    return LeagueSettings(
        game=resolved_game,
        league_id=_required_int(view, "id", what="league id"),
        season=_required_int(view, "seasonId", what="season"),
        name=str(settings.get("name") or ""),
        team_count=_required_int(settings, "size", what="league size"),
        current_scoring_period=_optional_int(view.get("scoringPeriodId")),
        current_matchup_period=_optional_int(status.get("currentMatchupPeriod")),
        first_scoring_period=_optional_int(status.get("firstScoringPeriod")),
        final_scoring_period=_optional_int(status.get("finalScoringPeriod")),
        scoring_type=scoring_type,
        scoring_type_raw=scoring_type_raw,
        scoring_kind=scoring_kind,
        scoring_items=scoring_items,
        lineup_slots=_parse_lineup_slots(roster.get("lineupSlotCounts") or {}, ids),
        position_limits={pos: _limit(value) for pos, value in _int_keyed(roster.get("positionLimits")).items()},
        slot_stat_limits=_parse_slot_stat_limits(roster.get("lineupSlotStatLimits")),
        lineup_lock_type=LockType(lineup_lock_raw) if lineup_lock_raw else LockType.UNKNOWN,
        lineup_lock_type_raw=lineup_lock_raw,
        roster_lock_type_raw=_optional_str(roster.get("rosterLocktimeType")),
        acquisition=_parse_acquisition(acquisition),
        trade=_parse_trade(trade),
        schedule=_parse_schedule(schedule),
    )


def _resolve_game(view: Mapping[str, Any], game: Game | str | None) -> Game:
    explicit = Game.coerce(game) if game is not None else None
    game_id = _optional_int(view.get("gameId"))
    if game_id is None:
        if explicit is None:
            raise SettingsParseError("response has no gameId; pass game='ffl' or game='fba'")
        return explicit
    try:
        from_payload = Game.from_game_id(game_id)
    except ValueError as exc:
        if explicit is None:
            raise SettingsParseError(str(exc)) from exc
        return explicit
    if explicit is not None and explicit is not from_payload:
        raise SettingsParseError(f"response is for {from_payload.value} (gameId {game_id}), not {explicit.value}")
    return from_payload


def _parse_scoring_items(raw_items: Iterable[Any], ids: IdMaps) -> tuple[ScoringItem, ...]:
    items: list[ScoringItem] = []
    for raw in raw_items:
        if not isinstance(raw, Mapping) or "statId" not in raw:
            raise SettingsParseError(f"malformed scoring item: {raw!r}")
        stat_id = int(raw["statId"])
        items.append(
            ScoringItem(
                stat_id=stat_id,
                stat=ids.stat_abbr(stat_id),
                label=ids.stat_label(stat_id),
                points=float(raw.get("points") or 0.0),
                points_overrides={pos: float(pts) for pos, pts in _int_keyed(raw.get("pointsOverrides")).items()},
                is_reverse=bool(raw.get("isReverseItem", False)),
            )
        )
    return tuple(sorted(items, key=lambda item: item.stat_id))


def _infer_scoring_kind(items: tuple[ScoringItem, ...]) -> ScoringKind:
    """For an unrecognized ``scoringType``: a league that awards points anywhere is a points league."""
    awards_points = any(item.points != 0 or any(item.points_overrides.values()) for item in items)
    return ScoringKind.POINTS if awards_points else ScoringKind.CATEGORIES


def _parse_lineup_slots(raw_counts: Mapping[str, Any], ids: IdMaps) -> tuple[LineupSlot, ...]:
    slots: list[LineupSlot] = []
    for slot_id, count in sorted(_int_keyed(raw_counts).items()):
        count = int(count)
        if count <= 0:
            continue
        if slot_id == ids.bench_slot:
            kind = SlotKind.BENCH
        elif slot_id == ids.ir_slot:
            kind = SlotKind.IR
        else:
            kind = SlotKind.ACTIVE
        slots.append(LineupSlot(slot_id=slot_id, label=ids.slot_label(slot_id), count=count, kind=kind))
    return tuple(slots)


def _parse_slot_stat_limits(raw: Any) -> dict[int, dict[int, int]]:
    """``lineupSlotStatLimits``: slot id -> stat id -> season cap (the NBA games-played limits).

    The only shape seen so far, in the fixtures (``espn-api`` never reads this field), is a bare integer per stat with
    ESPN's negative-means-unlimited convention; ``null`` reads as no cap. Anything else raises instead of being
    skipped, since a cap dropped silently would let the lineup optimizer break the league's rules. ROADMAP #14
    confirms the shape against a real capture.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise SettingsParseError(f"{SLOT_STAT_LIMITS} should be an object keyed by slot id, got {raw!r}")
    limits: dict[int, dict[int, int]] = {}
    for slot_key, per_stat in raw.items():
        slot_id = _optional_int(slot_key)
        if slot_id is None:
            raise SettingsParseError(f"{SLOT_STAT_LIMITS} has a non-numeric slot id {slot_key!r}")
        if not isinstance(per_stat, Mapping):
            raise SettingsParseError(
                f"{SLOT_STAT_LIMITS}[{slot_key}] should be an object keyed by stat id, got {per_stat!r}"
            )
        caps: dict[int, int] = {}
        for stat_key, cap in per_stat.items():
            stat_id = _optional_int(stat_key)
            if stat_id is None:
                raise SettingsParseError(f"{SLOT_STAT_LIMITS}[{slot_key}] has a non-numeric stat id {stat_key!r}")
            if cap is None:
                continue
            limit = _optional_int(cap)
            if limit is None:
                raise SettingsParseError(
                    f"{SLOT_STAT_LIMITS}[{slot_key}][{stat_key}] should be an integer cap (negative for unlimited), "
                    f"got {cap!r}"
                )
            if limit >= 0:
                caps[stat_id] = limit
        if caps:
            limits[slot_id] = caps
    return limits


def _parse_acquisition(raw: Mapping[str, Any]) -> AcquisitionSettings:
    type_raw = _optional_str(raw.get("acquisitionType"))
    uses_faab = bool(raw.get("isUsingAcquisitionBudget", False))
    per_period = bool(raw.get("matchupLimitPerScoringPeriod", False))
    return AcquisitionSettings(
        type=AcquisitionType(type_raw) if type_raw else AcquisitionType.UNKNOWN,
        type_raw=type_raw,
        uses_faab=uses_faab,
        budget=_optional_int(raw.get("acquisitionBudget")) if uses_faab else None,
        minimum_bid=_optional_int(raw.get("minimumBid")) or 0,
        season_limit=_limit(raw.get("acquisitionLimit")),
        matchup_limit=None if per_period else _limit(raw.get("matchupAcquisitionLimit")),
        matchup_limit_per_scoring_period=per_period,
        matchup_limit_rate=_rate(raw.get("matchupAcquisitionLimit")) if per_period else None,
        waiver_hours=_optional_int(raw.get("waiverHours")),
        waiver_process_days=tuple(str(day).upper() for day in raw.get("waiverProcessDays") or ()),
        waiver_process_hour=_optional_int(raw.get("waiverProcessHour")),
        waiver_order_reset=_optional_bool(raw.get("waiverOrderReset")),
        transaction_locking_enabled=_optional_bool(raw.get("transactionLockingEnabled")),
    )


def _parse_trade(raw: Mapping[str, Any]) -> TradeSettings:
    deadline_ms = _optional_int(raw.get("deadlineDate"))
    return TradeSettings(
        deadline=datetime.fromtimestamp(deadline_ms / 1000, tz=UTC) if deadline_ms else None,
        max_trades=_limit(raw.get("max")),
        revision_hours=_optional_int(raw.get("revisionHours")),
        veto_votes_required=_optional_int(raw.get("vetoVotesRequired")),
    )


def _parse_schedule(raw: Mapping[str, Any]) -> ScheduleSettings:
    matchup_periods = {
        matchup: tuple(sorted(int(period) for period in periods))
        for matchup, periods in _int_keyed(raw.get("matchupPeriods")).items()
    }
    return ScheduleSettings(
        regular_season_matchups=_required_int(raw, "matchupPeriodCount", what="matchupPeriodCount"),
        matchup_periods=dict(sorted(matchup_periods.items())),
        playoff_team_count=_optional_int(raw.get("playoffTeamCount")) or 0,
        playoff_matchup_period_length=_optional_int(raw.get("playoffMatchupPeriodLength")),
        variable_playoff_matchup_period_length=bool(raw.get("variablePlayoffMatchupPeriodLength", False)),
        playoff_seeding_rule=_optional_str(raw.get("playoffSeedingRule")),
    )


# --- coercion helpers -------------------------------------------------------------------------------------------------


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise SettingsParseError(f"mSettings response is missing {key!r}")
    return value


def _int_keyed(raw: Any) -> dict[int, Any]:
    """ESPN serializes integer-keyed maps with string keys; convert them, dropping anything non-numeric."""
    if not isinstance(raw, Mapping):
        return {}
    result: dict[int, Any] = {}
    for key, value in raw.items():
        try:
            result[int(key)] = value
        except (TypeError, ValueError):
            continue
    return result


def _required_int(parent: Mapping[str, Any], key: str, *, what: str) -> int:
    value = _optional_int(parent.get(key))
    if value is None:
        raise SettingsParseError(f"mSettings response is missing {what} ({key!r})")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _limit(value: Any) -> int | None:
    """ESPN encodes "unlimited" as a negative number; return ``None`` for it and for a missing value."""
    limit = _optional_int(value)
    return None if limit is None or limit < 0 else limit


def _rate(value: Any) -> float | None:
    """A per-scoring-period acquisition rate, kept fractional; negative or missing means unlimited."""
    if value is None or isinstance(value, bool):
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise SettingsParseError(f"acquisitionSettings.matchupAcquisitionLimit is not a number: {value!r}") from exc
    return None if rate < 0 else rate
