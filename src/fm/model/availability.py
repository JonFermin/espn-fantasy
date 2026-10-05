"""Availability: a player's official designation and the pro schedule become ``p_active`` (DESIGN section 8.2).

This is the basic model (ROADMAP #15). ``p_active`` is the base rate for the designation ESPN reports
(``player.injuryStatus``, normalised to :class:`fm.espn.ids.InjuryStatus` by the game's id maps), forced to 0 when the
player is not on an active pro roster, has no pro team, or his team has no game in the scoring period (an NFL bye, an
NBA off day). Expected value is ``p_active`` times the projection (:func:`expected_points`). The full model
(ROADMAP #22) layers practice-participation trends, the NBA injury report and bounded, logged Claude news signals on
top of the same :class:`fm.store.AvailabilityRow`; its ``inputs`` column records what went into each number so a
decision replays (DESIGN principle 5).

The base rates are model parameters, not league settings: per sport, by designation, and a caller may override any of
them (``rates``). They encode the usual reading of each designation and are starting points for #22 to refine (and
for the backtest to fit once outcomes are stored): a questionable NFL player usually plays, a doubtful one rarely
does; the NBA ladder runs from probable (nearly always plays) through ESPN's catch-all day-to-day to doubtful. A player
with no designation (ESPN sends ``ACTIVE``, ``NORMAL`` or nothing, as for every D/ST) counts as active. A designation
the id maps do not know is not treated as healthy: it gets the questionable rate and is flagged in ``inputs`` so the
new value gets mapped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any, Final

from fm.config import Sport
from fm.espn.ids import Game, InjuryStatus, ids_for
from fm.sports.base import FREE_AGENT_TEAM, ScheduleLike, game_for, is_provisional
from fm.store import AvailabilityRow, PlayerRow, Store

MODEL: Final = "designation"
"""The ``inputs["model"]`` tag of rows this module writes; the full model (ROADMAP #22) tags its own."""

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


def _sport(value: Game | str) -> Sport:
    return "nfl" if Game.coerce(value) is Game.FFL else "nba"


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


def assess(
    player: PlayerRow,
    *,
    season: int,
    scoring_period: int,
    as_of: datetime,
    schedule: ScheduleLike | None = None,
    rates: Mapping[InjuryStatus, float] | None = None,
) -> AvailabilityRow:
    """``p_active`` for one player in one scoring period, as a store row.

    With a ``schedule`` (the sport's pro schedule, ``fm.espn.models.ProSchedule``), a player whose pro team has no game
    in the period, or who has no pro team (none, or ESPN's free-agent team 0), gets ``has_game=False`` and
    ``p_active=0``; otherwise ``game_time`` is the game's start and ``inputs`` names the game and whether its start
    is still provisional. A schedule that does not cover the period raises ``ValueError``. Without a schedule the game
    is assumed (``has_game=True``, no ``game_time``) and ``inputs["schedule"]`` says so. A player not on an active pro
    roster (``PlayerRow.active`` false) gets 0 regardless of designation.
    """
    sport = player.sport
    table = rates_for(sport, rates)
    raw = player.injury_status
    status = designation(raw, sport)
    unrecognized = is_unrecognized(raw, sport)
    rate = table[UNRECOGNIZED_AS] if unrecognized else table[status]
    inputs: dict[str, Any] = {
        "model": MODEL,
        "designation_raw": raw,
        "designation": status.value,
        "base_rate": rate,
        "active": player.active,
        "pro_team_id": player.pro_team_id,
        "schedule": schedule is not None,
    }
    if unrecognized:
        inputs["designation_unrecognized"] = True

    team_id = player.pro_team_id if player.pro_team_id != FREE_AGENT_TEAM else None
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

    if not player.active:
        p_active = 0.0
        inputs["reason"] = REASON_INACTIVE
    elif team_id is None:
        p_active = 0.0
        inputs["reason"] = REASON_NO_TEAM
    elif not has_game:
        p_active = 0.0
        inputs["reason"] = REASON_NO_GAME
    else:
        p_active = rate
        if status in INACTIVE_DESIGNATIONS:
            inputs["reason"] = REASON_DESIGNATION
    return AvailabilityRow(
        sport=sport,
        espn_id=player.espn_id,
        season=season,
        scoring_period_id=scoring_period,
        designation=status.value if raw else None,
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
) -> list[AvailabilityRow]:
    """:func:`assess` for many players, in the order given."""
    return [
        assess(player, season=season, scoring_period=scoring_period, as_of=as_of, schedule=schedule, rates=rates)
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
    save: bool = True,
) -> list[AvailabilityRow]:
    """Assess the stored players with the given ESPN ids (ids the ``players`` table lacks are skipped), by ESPN id,
    and unless ``save`` is false upsert the rows into ``availability``, replacing each player's row for the period."""
    players = store.players.many(_sport(sport), espn_ids)
    rows = assess_many(
        players, season=season, scoring_period=scoring_period, as_of=as_of, schedule=schedule, rates=rates
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
