"""Typed models for ESPN's fantasy read views (DESIGN section 6.1): teams, rosters, matchups, the player pool, player
cards, transactions and the pro schedule. ``mSettings`` has its own parser in :mod:`fm.espn.settings`.

The models mirror ESPN's JSON closely so a field can be looked up in a captured response: names are the snake_case form
of ESPN's camelCase keys (``scoringPeriodId`` -> ``scoring_period_id``), ids stay numeric, and the few renames keep an
explicit alias. Every model is frozen and ignores unknown keys, because ESPN adds fields without notice and a sync must
not break on one. Enumerations that ESPN extends (transaction ``type``/``status``, ``injuryStatus``, pool ``status``)
are kept as strings with the known values as module constants; :mod:`fm.espn.ids` normalizes injury status.

Conventions shared by every model:

- Dates are epoch milliseconds in ESPN's JSON and aware UTC datetimes here; ``0``/``null`` mean "no date".
- Maps keyed by stat, slot, position or scoring-period id are serialized with string keys by ESPN and are ``dict[int,
  ...]`` here; non-numeric keys are dropped.
- A player's ``stats`` entries are stat lines, never points: ``stat_source_id`` 0 is actual and 1 projected,
  ``stat_split_type_id`` 0 is the season and 1 a single scoring period, and ``applied_total`` is ESPN's own points
  under the league's scoring (kept for reference only; league points are computed from the scoring items).
- ``*View`` models are whole responses: a :class:`LeagueEnvelope` (league id, season, current scoring period,
  ``status``) plus the requested view's block.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

STAT_SOURCE_ACTUAL = 0
STAT_SOURCE_PROJECTED = 1
STAT_SPLIT_SEASON = 0
STAT_SPLIT_SCORING_PERIOD = 1

# ``playerPoolEntry.status`` values.
POOL_FREE_AGENT = "FREEAGENT"
POOL_WAIVERS = "WAIVERS"
POOL_ON_TEAM = "ONTEAM"

# ``transactions[].type`` values ESPN is known to use; the models accept any string.
TRANSACTION_TYPES: frozenset[str] = frozenset(
    {
        "FREEAGENT",
        "WAIVER",
        "WAIVER_ERROR",
        "TRADE_PROPOSAL",
        "TRADE_ACCEPT",
        "TRADE_DECLINE",
        "TRADE_VETO",
        "TRADE_UPHOLD",
        "ROSTER",
        "FUTURE_ROSTER",
        "RETRO_ROSTER",
        "DRAFT",
    }
)
TRADE_TRANSACTION_TYPES: frozenset[str] = frozenset(t for t in TRANSACTION_TYPES if t.startswith("TRADE"))
TRANSACTION_PENDING = "PENDING"
TRANSACTION_EXECUTED = "EXECUTED"
TRANSACTION_CANCELED = "CANCELED"
FAILED_STATUS_PREFIX = "FAILED"
"""Failed waiver claims keep their ``bidAmount`` (``FAILED_INVALIDPLAYERSOURCE`` is an outbid claim), which is what
the FAAB bid model learns from."""

# ``items[].type`` values.
ITEM_ADD = "ADD"
ITEM_DROP = "DROP"
ITEM_LINEUP = "LINEUP"
ITEM_TRADE = "TRADE"

MATCHUP_UNDECIDED = "UNDECIDED"
NO_PLAYOFF_TIER = "NONE"


# --- coercion ---------------------------------------------------------------------------------------------------------


def _from_epoch_ms(value: Any) -> Any:
    """ESPN dates are integer milliseconds since the epoch; ``0``, negative and ``null`` mean "no date"."""
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, bool):
        raise ValueError("a boolean is not a timestamp")
    if isinstance(value, int | float):
        if value <= 0:
            return None
        try:
            return datetime.fromtimestamp(value / 1000, tz=UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError(f"timestamp {value!r} is out of range") from exc
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return _from_epoch_ms(int(value))
    return value  # anything else fails pydantic's own datetime validation


def _int_keyed(value: Any) -> Any:
    """ESPN serializes integer-keyed maps with string keys; convert them, dropping anything non-numeric."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        result: dict[int, Any] = {}
        for key, item in value.items():
            try:
                result[int(key)] = item
            except (TypeError, ValueError):
                continue
        return result
    return value


def _none_as_empty(value: Any) -> Any:
    return () if value is None else value


def _str_tuple(value: Any) -> Any:
    if value is None:
        return ()
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return value


def _as_str(value: Any) -> Any:
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else value


type EpochMs = Annotated[datetime, BeforeValidator(_from_epoch_ms)]
type EpochMsOrNone = Annotated[datetime | None, BeforeValidator(_from_epoch_ms)]
type IntKeyed[V] = Annotated[dict[int, V], BeforeValidator(_int_keyed)]
type TupleOf[V] = Annotated[tuple[V, ...], BeforeValidator(_none_as_empty)]
"""A JSON list ESPN may send as ``null`` (``items``, ``stats``, ``entries``)."""
type IntTuple = TupleOf[int]
type StrTuple = Annotated[tuple[str, ...], BeforeValidator(_str_tuple)]
type IdStr = Annotated[str, BeforeValidator(_as_str)]


class EspnModel(BaseModel):
    """Base for every parsed view: frozen, camelCase aliases, unknown keys ignored."""

    model_config = ConfigDict(
        frozen=True, extra="ignore", alias_generator=to_camel, validate_by_name=True, validate_by_alias=True
    )


# --- league envelope --------------------------------------------------------------------------------------------------


class LeagueStatus(EspnModel):
    """The top-level ``status`` block: where the season stands and when waivers run next."""

    current_matchup_period: int | None = None
    latest_scoring_period: int | None = None
    first_scoring_period: int | None = None
    final_scoring_period: int | None = None
    transaction_scoring_period: int | None = None
    is_active: bool | None = None
    is_full: bool | None = None
    teams_joined: int | None = None
    activated_date: EpochMsOrNone = None
    standings_update_date: EpochMsOrNone = None
    waiver_last_execution_date: EpochMsOrNone = None
    waiver_next_execution_date: EpochMsOrNone = None


class LeagueEnvelope(EspnModel):
    """Fields every league-scoped response carries beside the requested view."""

    game_id: int | None = None
    league_id: int | None = Field(default=None, alias="id")
    season_id: int | None = None
    scoring_period_id: int | None = None
    segment_id: int | None = None
    status: LeagueStatus | None = None


# --- teams (mTeam, mStandings) ----------------------------------------------------------------------------------------


class Member(EspnModel):
    """A league member (``members[]``). ``id`` is the account's SWID; fixtures carry placeholders."""

    id: IdStr
    display_name: str | None = None
    is_league_manager: bool = False


class TeamRecord(EspnModel):
    wins: int = 0
    losses: int = 0
    ties: int = 0
    percentage: float | None = None
    points_for: float = 0.0
    points_against: float = 0.0
    games_back: float | None = None
    streak_length: int | None = None
    streak_type: str | None = None


class TeamRecords(EspnModel):
    """``record``: the overall record plus the home/away/division splits."""

    overall: TeamRecord = Field(default_factory=TeamRecord)
    home: TeamRecord | None = None
    away: TeamRecord | None = None
    division: TeamRecord | None = None


class TransactionCounter(EspnModel):
    """``transactionCounter``: season-to-date moves and FAAB spent."""

    acquisitions: int = 0
    acquisition_budget_spent: int = 0
    drops: int = 0
    trades: int = 0
    move_to_ir: int = Field(default=0, alias="moveToIR")
    paid: float | None = None
    matchup_acquisition_totals: IntKeyed[int] = Field(default_factory=dict)


class Team(EspnModel):
    """A fantasy team (``teams[]``) with its standing, record, waiver position and transaction counters."""

    id: int
    abbrev: str | None = None
    name: str | None = None
    location: str | None = None
    nickname: str | None = None
    logo: str | None = None
    division_id: int | None = None
    is_active: bool | None = None
    owners: StrTuple = ()
    primary_owner: str | None = None
    playoff_seed: int | None = None
    waiver_rank: int | None = None
    rank_calculated_final: int | None = None
    rank_final: int | None = None
    current_projected_rank: int | None = None
    draft_day_projected_rank: int | None = None
    record: TeamRecords = Field(default_factory=TeamRecords)
    transaction_counter: TransactionCounter = Field(default_factory=TransactionCounter)
    values_by_stat: IntKeyed[float] = Field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """``name`` (current ESPN payloads) or ``location nickname`` (older ones), falling back to the abbreviation."""
        if self.name:
            return self.name
        joined = " ".join(part for part in (self.location, self.nickname) if part)
        return joined or self.abbrev or f"Team {self.id}"


class TeamsView(LeagueEnvelope):
    """``mTeam`` (+ ``mStandings``): every team and member in the league."""

    teams: TupleOf[Team] = ()
    members: TupleOf[Member] = ()

    def team(self, team_id: int) -> Team:
        for team in self.teams:
            if team.id == team_id:
                return team
        raise KeyError(f"no team {team_id} in the league; known: {sorted(team.id for team in self.teams)}")

    def member(self, member_id: str) -> Member | None:
        return next((member for member in self.members if member.id == member_id), None)


# --- players ----------------------------------------------------------------------------------------------------------


class Ownership(EspnModel):
    """``player.ownership``: league-wide ownership trends across ESPN."""

    percent_owned: float | None = None
    percent_started: float | None = None
    percent_change: float | None = None
    average_draft_position: float | None = None
    auction_value_average: float | None = None


class Rating(EspnModel):
    """One ``ratings`` entry, keyed by scoring period (``0`` = season): ESPN's rank at the position and overall."""

    positional_ranking: int | None = None
    total_ranking: int | None = None
    total_rating: float | None = None


class PlayerStats(EspnModel):
    """One ``player.stats[]`` entry: a stat line for one scope.

    ``stats`` is keyed by ESPN stat id (:mod:`fm.espn.ids`) and is the input to league scoring; ``applied_stats`` /
    ``applied_total`` are ESPN's points for it under this league's rules, kept for cross-checks only. ``id`` is ESPN's
    composite key ``{source}{split}{season}[{period}]`` (``"1120264"`` = projected, single period, 2026, week 4).
    """

    id: IdStr | None = None
    season_id: int
    scoring_period_id: int = 0
    stat_source_id: int = STAT_SOURCE_ACTUAL
    stat_split_type_id: int = STAT_SPLIT_SEASON
    pro_team_id: int | None = None
    external_id: IdStr | None = None
    stats: IntKeyed[float] = Field(default_factory=dict)
    applied_stats: IntKeyed[float] = Field(default_factory=dict)
    applied_total: float | None = None
    applied_average: float | None = None

    @property
    def is_projection(self) -> bool:
        return self.stat_source_id == STAT_SOURCE_PROJECTED

    @property
    def is_actual(self) -> bool:
        return self.stat_source_id == STAT_SOURCE_ACTUAL

    @property
    def is_season(self) -> bool:
        return self.stat_split_type_id == STAT_SPLIT_SEASON

    @property
    def is_single_period(self) -> bool:
        return self.stat_split_type_id == STAT_SPLIT_SCORING_PERIOD


class Player(EspnModel):
    """A player (``player``): identity, position, pro team, injury state, ownership and stat lines."""

    id: int
    full_name: str = ""
    first_name: str | None = None
    last_name: str | None = None
    default_position_id: int | None = None
    eligible_slots: IntTuple = ()
    pro_team_id: int | None = None
    injury_status: str | None = None
    injured: bool = False
    active: bool = True
    droppable: bool | None = None
    jersey: str | None = None
    universe_id: int | None = None
    last_news_date: EpochMsOrNone = None
    ownership: Ownership | None = None
    stats: TupleOf[PlayerStats] = ()

    def stat_entry(self, *, season: int, scoring_period: int = 0, projected: bool = False) -> PlayerStats | None:
        """The season (``scoring_period=0``) or single-period stat line of one source, or ``None`` when absent.

        Rolling-window splits (last 7/15/30 days in ``fba``) are never returned.
        """
        source = STAT_SOURCE_PROJECTED if projected else STAT_SOURCE_ACTUAL
        split = STAT_SPLIT_SEASON if scoring_period == 0 else STAT_SPLIT_SCORING_PERIOD
        for entry in self.stats:
            if (
                entry.season_id == season
                and entry.scoring_period_id == scoring_period
                and entry.stat_source_id == source
                and entry.stat_split_type_id == split
            ):
                return entry
        return None

    def projection(self, season: int, scoring_period: int) -> PlayerStats | None:
        """ESPN's projected stat line for one scoring period (``0`` for the full season)."""
        return self.stat_entry(season=season, scoring_period=scoring_period, projected=True)

    def actual(self, season: int, scoring_period: int) -> PlayerStats | None:
        """The actual stat line for one scoring period (``0`` for the season to date)."""
        return self.stat_entry(season=season, scoring_period=scoring_period, projected=False)


class PoolEntry(EspnModel):
    """A ``playerPoolEntry``: a player plus his standing in this league (free agent, on waivers, or on a team) and
    the lock flags the executor checks before a move. Returned as-is by ``kona_player_info`` / ``kona_playercard`` and
    nested in roster entries."""

    id: int
    player: Player
    on_team_id: int | None = None
    status: str | None = None
    lineup_locked: bool = False
    roster_locked: bool = False
    trade_locked: bool = False
    waiver_process_date: EpochMsOrNone = None
    keeper_value: int | None = None
    keeper_value_future: int | None = None
    applied_stat_total: float | None = None
    ratings: IntKeyed[Rating] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _id_from_player(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "id" not in data and isinstance(data.get("player"), Mapping):
            data = {**data, "id": data["player"].get("id")}
        return data

    @property
    def is_free_agent(self) -> bool:
        return self.status == POOL_FREE_AGENT

    @property
    def is_on_waivers(self) -> bool:
        return self.status == POOL_WAIVERS

    @property
    def rostered_team_id(self) -> int | None:
        """The fantasy team holding the player; ``None`` when unowned (ESPN sends ``onTeamId: 0``)."""
        return self.on_team_id or None


class PlayersView(LeagueEnvelope):
    """``kona_player_info`` (the filtered player pool) or ``kona_playercard`` (specific players)."""

    players: TupleOf[PoolEntry] = ()

    def entry(self, player_id: int) -> PoolEntry:
        for entry in self.players:
            if entry.id == player_id:
                return entry
        raise KeyError(f"player {player_id} is not in the response")

    @property
    def player_ids(self) -> tuple[int, ...]:
        return tuple(entry.id for entry in self.players)


# --- rosters (mRoster) ------------------------------------------------------------------------------------------------


class RosterEntry(EspnModel):
    """One player on a roster for one scoring period: the slot he sits in and how he was acquired."""

    player_id: int
    lineup_slot_id: int
    acquisition_type: str | None = None
    acquisition_date: EpochMsOrNone = None
    injury_status: str | None = None
    status: str | None = None
    pending_transaction_ids: StrTuple = ()
    player_pool_entry: PoolEntry

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        """Accept a bare ``player`` (box-score entries) in place of ``playerPoolEntry`` and default ``playerId``."""
        if not isinstance(data, Mapping):
            return data
        normalized = dict(data)
        pool = normalized.get("playerPoolEntry")
        if not isinstance(pool, Mapping) and isinstance(normalized.get("player"), Mapping):
            pool = {"id": normalized["player"].get("id"), "player": normalized["player"]}
            normalized["playerPoolEntry"] = pool
        if "playerId" not in normalized and isinstance(pool, Mapping):
            normalized["playerId"] = pool.get("id")
        return normalized

    @property
    def player(self) -> Player:
        return self.player_pool_entry.player

    @property
    def lineup_locked(self) -> bool:
        return self.player_pool_entry.lineup_locked


class Roster(EspnModel):
    """A set of roster entries (``roster`` / ``rosterForCurrentScoringPeriod``) and ESPN's point total for them."""

    entries: TupleOf[RosterEntry] = ()
    applied_stat_total: float | None = None

    def entry(self, player_id: int) -> RosterEntry:
        for entry in self.entries:
            if entry.player_id == player_id:
                return entry
        raise KeyError(f"player {player_id} is not on this roster")

    @property
    def player_ids(self) -> tuple[int, ...]:
        return tuple(entry.player_id for entry in self.entries)

    def in_slot(self, slot_id: int) -> tuple[RosterEntry, ...]:
        return tuple(entry for entry in self.entries if entry.lineup_slot_id == slot_id)


class TeamRoster(Roster):
    """``mRoster`` ``teams[]``: one team's roster for the requested scoring period."""

    team_id: int = Field(alias="id")

    @model_validator(mode="before")
    @classmethod
    def _lift_roster(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and isinstance(data.get("roster"), Mapping):
            data = {**data, **data["roster"]}
        return data


class RostersView(LeagueEnvelope):
    """``mRoster``: every team's roster for the requested scoring period."""

    teams: TupleOf[TeamRoster] = ()

    def roster(self, team_id: int) -> TeamRoster:
        for roster in self.teams:
            if roster.team_id == team_id:
                return roster
        raise KeyError(f"no roster for team {team_id}; known: {sorted(roster.team_id for roster in self.teams)}")

    def team_of(self, player_id: int) -> int | None:
        """The team rostering a player, or ``None`` when nobody does."""
        for roster in self.teams:
            if player_id in roster.player_ids:
                return roster.team_id
        return None


# --- matchups (mMatchup, mMatchupScore, mScoreboard) ------------------------------------------------------------------


class CategoryScore(EspnModel):
    """One category in a category league's running score: the team's value and whether it leads."""

    score: float = 0.0
    result: str | None = None  # WIN / LOSS / TIE, or None while undecided
    ineligible: bool = False


class CumulativeScore(EspnModel):
    """``cumulativeScore``: category wins/losses/ties so far (points leagues send zeros and no ``scoreByStat``)."""

    wins: int = 0
    losses: int = 0
    ties: int = 0
    score_by_stat: IntKeyed[CategoryScore] = Field(default_factory=dict)


class MatchupSide(EspnModel):
    """One team's side of a matchup. Rosters are present only in ``mMatchupScore``/``mScoreboard`` responses."""

    team_id: int
    total_points: float = 0.0
    total_points_live: float | None = None
    total_projected_points_live: float | None = None
    points_by_scoring_period: IntKeyed[float] = Field(default_factory=dict)
    cumulative_score: CumulativeScore | None = None
    roster_for_current_scoring_period: Roster | None = None
    roster_for_matchup_period: Roster | None = None
    games_played: int | None = None
    adjustment: float | None = None
    tiebreak: float | None = None

    @property
    def roster(self) -> Roster | None:
        """The lineup for the requested scoring period, when the view carried one."""
        return self.roster_for_current_scoring_period or self.roster_for_matchup_period


class Matchup(EspnModel):
    """One ``schedule[]`` entry. A bye has only a ``home`` side."""

    id: int
    matchup_period_id: int
    playoff_tier_type: str | None = None
    winner: str | None = None
    home: MatchupSide | None = None
    away: MatchupSide | None = None

    @property
    def is_bye(self) -> bool:
        return self.home is None or self.away is None

    @property
    def is_playoff(self) -> bool:
        return self.playoff_tier_type is not None and self.playoff_tier_type != NO_PLAYOFF_TIER

    @property
    def is_decided(self) -> bool:
        return self.winner is not None and self.winner != MATCHUP_UNDECIDED

    @property
    def sides(self) -> tuple[MatchupSide, ...]:
        return tuple(side for side in (self.home, self.away) if side is not None)

    @property
    def team_ids(self) -> tuple[int, ...]:
        return tuple(side.team_id for side in self.sides)

    def side(self, team_id: int) -> MatchupSide | None:
        return next((side for side in self.sides if side.team_id == team_id), None)

    def opponent(self, team_id: int) -> MatchupSide | None:
        """The other side, or ``None`` when ``team_id`` is not in this matchup or is on a bye."""
        if self.side(team_id) is None:
            return None
        return next((side for side in self.sides if side.team_id != team_id), None)


class MatchupsView(LeagueEnvelope):
    """``mMatchup`` (the whole schedule with scores) or ``mMatchupScore``/``mScoreboard`` (one period with lineups)."""

    schedule: TupleOf[Matchup] = ()

    def for_period(self, matchup_period: int) -> tuple[Matchup, ...]:
        return tuple(matchup for matchup in self.schedule if matchup.matchup_period_id == matchup_period)

    def for_team(self, team_id: int, matchup_period: int | None = None) -> tuple[Matchup, ...]:
        return tuple(
            matchup
            for matchup in self.schedule
            if team_id in matchup.team_ids and (matchup_period is None or matchup.matchup_period_id == matchup_period)
        )

    @property
    def matchup_periods(self) -> tuple[int, ...]:
        return tuple(sorted({matchup.matchup_period_id for matchup in self.schedule}))


# --- transactions (mTransactions2, mPendingTransactions) --------------------------------------------------------------


class TransactionItem(EspnModel):
    """One leg of a transaction: an add, drop, lineup move or trade piece."""

    type: str
    player_id: int
    from_team_id: int | None = None
    to_team_id: int | None = None
    from_lineup_slot_id: int | None = None
    to_lineup_slot_id: int | None = None
    is_keeper: bool | None = None


class Transaction(EspnModel):
    """A transaction: a free-agent move, waiver claim (with its bid), trade step or lineup change.

    ``bid_amount`` is present on executed and failed claims alike. ``date`` is when it happened (processed, else
    accepted, else proposed).
    """

    id: IdStr
    type: str
    status: str | None = None
    team_id: int | None = None
    member_id: IdStr | None = None
    scoring_period_id: int | None = None
    bid_amount: int | None = None
    proposed_date: EpochMsOrNone = None
    process_date: EpochMsOrNone = None
    accepted_date: EpochMsOrNone = None
    expiration_date: EpochMsOrNone = None
    execution_type: str | None = None
    is_pending: bool | None = None
    is_league_manager: bool | None = None
    related_transaction_id: IdStr | None = None
    rating: int | None = None
    items: TupleOf[TransactionItem] = ()

    @property
    def date(self) -> datetime | None:
        return self.process_date or self.accepted_date or self.proposed_date

    @property
    def pending(self) -> bool:
        return self.status == TRANSACTION_PENDING or bool(self.is_pending)

    @property
    def executed(self) -> bool:
        return self.status == TRANSACTION_EXECUTED

    @property
    def failed(self) -> bool:
        return self.status is not None and self.status.startswith(FAILED_STATUS_PREFIX)

    @property
    def is_trade(self) -> bool:
        return self.type in TRADE_TRANSACTION_TYPES

    @property
    def is_waiver(self) -> bool:
        return self.type in ("WAIVER", "WAIVER_ERROR")

    def items_of(self, item_type: str) -> tuple[TransactionItem, ...]:
        return tuple(item for item in self.items if item.type == item_type)

    @property
    def adds(self) -> tuple[int, ...]:
        return tuple(item.player_id for item in self.items_of(ITEM_ADD))

    @property
    def drops(self) -> tuple[int, ...]:
        return tuple(item.player_id for item in self.items_of(ITEM_DROP))

    @property
    def player_ids(self) -> tuple[int, ...]:
        return tuple(item.player_id for item in self.items)

    @property
    def team_ids(self) -> tuple[int, ...]:
        """Every team touched: the acting team plus each item's from/to team."""
        ids: list[int] = [] if self.team_id is None else [self.team_id]
        for item in self.items:
            for team in (item.from_team_id, item.to_team_id):
                if team is not None and team not in ids:
                    ids.append(team)
        return tuple(ids)


class TransactionsView(LeagueEnvelope):
    """``mTransactions2`` or ``mPendingTransactions``. The pending view's list key is unconfirmed (ROADMAP #14), so
    both ``transactions`` and ``pendingTransactions`` are read."""

    transactions: TupleOf[Transaction] = ()

    @model_validator(mode="before")
    @classmethod
    def _accept_pending_key(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "transactions" not in data and "pendingTransactions" in data:
            data = {**data, "transactions": data["pendingTransactions"]}
        return data

    def pending(self) -> tuple[Transaction, ...]:
        return tuple(transaction for transaction in self.transactions if transaction.pending)

    def of_type(self, *types: str) -> tuple[Transaction, ...]:
        return tuple(transaction for transaction in self.transactions if transaction.type in types)

    def with_bids(self) -> tuple[Transaction, ...]:
        """Transactions that report a ``bidAmount``: waiver claims won or lost (free-agent adds report ``0``)."""
        return tuple(transaction for transaction in self.transactions if transaction.bid_amount is not None)


# --- pro schedule (proTeamSchedules_wl) -------------------------------------------------------------------------------


class ProGame(EspnModel):
    """One pro game. ``date`` is the kickoff/tip (UTC); ``valid_for_locking`` says ESPN locks lineups on it."""

    id: int | None = None
    date: EpochMs
    scoring_period_id: int
    home_pro_team_id: int
    away_pro_team_id: int
    start_time_tbd: bool = Field(default=False, alias="startTimeTBD")
    stats_official: bool = False
    valid_for_locking: bool = True

    @property
    def pro_team_ids(self) -> tuple[int, int]:
        return (self.home_pro_team_id, self.away_pro_team_id)

    def involves(self, pro_team_id: int) -> bool:
        return pro_team_id in self.pro_team_ids

    def opponent_of(self, pro_team_id: int) -> int | None:
        if pro_team_id == self.home_pro_team_id:
            return self.away_pro_team_id
        if pro_team_id == self.away_pro_team_id:
            return self.home_pro_team_id
        return None


class ProTeam(EspnModel):
    """A pro team (``settings.proTeams[]``) and its games keyed by scoring period. Id ``0`` is the free-agent pseudo
    team and has no games."""

    id: int
    abbrev: str | None = None
    location: str | None = None
    name: str | None = None
    bye_week: int | None = None
    universe_id: int | None = None
    pro_games_by_scoring_period: IntKeyed[TupleOf[ProGame]] = Field(default_factory=dict)

    def games(self, scoring_period: int) -> tuple[ProGame, ...]:
        return self.pro_games_by_scoring_period.get(scoring_period, ())


class ProSchedule(EspnModel):
    """``proTeamSchedules_wl``: every pro team's schedule for the season, the source of lock times and game counts."""

    pro_teams: TupleOf[ProTeam] = ()

    @model_validator(mode="before")
    @classmethod
    def _lift_settings(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "proTeams" not in data and isinstance(data.get("settings"), Mapping):
            data = {**data, "proTeams": data["settings"].get("proTeams")}
        return data

    def team(self, pro_team_id: int) -> ProTeam | None:
        return next((team for team in self.pro_teams if team.id == pro_team_id), None)

    @property
    def scoring_periods(self) -> tuple[int, ...]:
        return tuple(sorted({period for team in self.pro_teams for period in team.pro_games_by_scoring_period}))

    def games(self, scoring_period: int) -> tuple[ProGame, ...]:
        """Every game in a scoring period, once each (both teams list it), in start order."""
        seen: dict[int | tuple[int, int], ProGame] = {}
        for team in self.pro_teams:
            for game in team.games(scoring_period):
                key = game.id if game.id is not None else game.pro_team_ids
                seen.setdefault(key, game)
        return tuple(sorted(seen.values(), key=lambda game: (game.date, game.pro_team_ids)))

    def games_for(self, pro_team_id: int, scoring_period: int) -> tuple[ProGame, ...]:
        team = self.team(pro_team_id)
        return () if team is None else team.games(scoring_period)

    def has_game(self, pro_team_id: int, scoring_period: int) -> bool:
        return bool(self.games_for(pro_team_id, scoring_period))

    def first_game(self, scoring_period: int) -> ProGame | None:
        """The earliest start in the period: the NBA add/drop cutoff for the day, the week's first lock in the NFL."""
        games = self.games(scoring_period)
        return games[0] if games else None

    def idle_teams(self, scoring_period: int) -> tuple[int, ...]:
        """Pro teams without a game in the period (NFL byes, NBA off days), excluding the free-agent pseudo team."""
        return tuple(sorted(team.id for team in self.pro_teams if team.id != 0 and not team.games(scoring_period)))
