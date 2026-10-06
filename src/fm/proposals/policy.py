"""Proposal kinds, per-kind policy, and the guardrails a proposal clears before it is stored (DESIGN section 11).

Kinds and settings
------------------
Every move the executor can make is a :class:`ProposalKind`. :data:`KINDS` says, per kind, which ``[league.policy]``
field in ``config.toml`` governs it, which settings that field may take, which payload model it carries, whether it
spends one of the week's transactions and whether it can send players away. The lineup kinds allow ``auto`` (which
fires only at T-15 when the proposal is unanswered, see :mod:`fm.proposals.queue`); adds, drops and claims are
``approve`` or ``off``; **trade kinds have no policy field and are approve-only**. ``fm.config.Policy`` already refuses
``trade*`` keys, and :func:`effective_setting` returns ``approve`` for a trade kind whatever the config says, so the
rule holds even if the config layer were bypassed (CLAUDE.md).

Guardrails
----------
:func:`evaluate` returns a :class:`Verdict` listing every reason a proposal is blocked, so a decision module sees all
of them at once:

- league allowlist: the store's league row must be one of the leagues in ``config.toml``, pointing at the same ESPN
  league and season;
- the kind's setting is not ``off``; ``auto`` proposals carry a deadline;
- the deadline has not passed; the ``fm pause`` switch is not thrown;
- untouchables: no player on the policy's list is dropped or given away;
- weekly cap: ``max_transactions_per_week`` counts the league's add/drop and waiver proposals that are pending or went
  through: those in the same matchup period as the new one when a scoring period is known and the matchup's scoring
  periods can be told, plus any stored without a scoring period in the trailing seven days (a producer that omits the
  period cannot free the week for one that supplies it); otherwise every one in the trailing seven days. The
  matchup's scoring periods are the synced settings' own where they list scoring periods, and ESPN's season calendar
  (:mod:`fm.espn.calendar`) resolves the weeks the NBA league lists them as; a season without a calendar file uses the
  trailing seven days;
- FAAB cap: a bid is at most ``max_faab_pct_per_bid`` of the league's season budget (from the synced settings);
- trade deadline: once settings are synced, proposing or accepting a trade must happen before the league's trade
  deadline; declining or withdrawing an offer stays allowed, and the executor re-checks the live league either way.

The budget, matchup periods and trade deadline are league settings, read from the store, never constants.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from fractions import Fraction
from typing import cast

from fm.config import Approval, Config, League, Policy, Sport
from fm.espn.calendar import matchup_scoring_periods
from fm.espn.settings import LeagueSettings
from fm.proposals.pause import pause_state
from fm.proposals.payloads import (
    AddDropPayload,
    LineupPayload,
    Payload,
    TradePayload,
    TradeResponsePayload,
    TransactionCancelPayload,
    WaiverPayload,
)
from fm.store import LeagueRow, ProposalRow, ProposalStatus, Store

ROLLING_WEEK = timedelta(days=7)
"""The transaction-cap window when the matchup period cannot be determined."""


class ProposalError(Exception):
    """Base of the errors raised by ``fm.proposals``."""


class PolicyError(ProposalError):
    """A proposal is blocked by policy or a guardrail, or a setting is not allowed. The message lists every reason."""


class ProposalKind(StrEnum):
    """Every move the executor can make, as stored in ``proposals.kind``."""

    BENCH_INACTIVE = "bench_inactive"  # bench an OUT, bye or no-game starter
    LINEUP = "lineup"  # any other lineup optimization
    ADD_DROP = "add_drop"  # free-agent add and/or drop, including streaming
    WAIVER = "waiver"  # waiver claim, with a FAAB bid where the league bids
    WAIVER_CANCEL = "waiver_cancel"  # withdraw our pending claim
    TRADE_PROPOSE = "trade_propose"
    TRADE_ACCEPT = "trade_accept"
    TRADE_DECLINE = "trade_decline"
    TRADE_CANCEL = "trade_cancel"  # withdraw our pending offer

    @property
    def is_trade(self) -> bool:
        return self.value.startswith("trade")


@dataclass(frozen=True, slots=True)
class KindSpec:
    """How one kind is governed."""

    kind: ProposalKind
    label: str
    policy_field: str | None
    """The ``fm.config.Policy`` field holding the setting; ``None`` for trade kinds, which are approve-only."""
    allowed: tuple[Approval, ...]
    payload_type: type[Payload]
    acquisition: bool = False
    """Counts toward ``max_transactions_per_week``."""
    may_lose_players: bool = False
    """The payload's ``outgoing_players`` really leave our roster, so the untouchables check applies."""
    trade_window: bool = False
    """The move would make a trade happen, so it must land before the league's trade deadline."""

    @property
    def is_trade(self) -> bool:
        return self.kind.is_trade

    @property
    def default(self) -> Approval:
        """The setting a league gets when ``config.toml`` says nothing."""
        if self.policy_field is None:
            return "approve"
        return cast(Approval, Policy.model_fields[self.policy_field].default)


_LINEUP: tuple[Approval, ...] = ("off", "approve", "auto")
_TRANSACTION: tuple[Approval, ...] = ("off", "approve")
_TRADE: tuple[Approval, ...] = ("approve",)

KINDS: Mapping[ProposalKind, KindSpec] = {
    spec.kind: spec
    for spec in (
        KindSpec(
            ProposalKind.BENCH_INACTIVE, "bench an OUT/bye/no-game starter", "bench_inactive", _LINEUP, LineupPayload
        ),
        KindSpec(ProposalKind.LINEUP, "lineup change", "lineup", _LINEUP, LineupPayload),
        KindSpec(
            ProposalKind.ADD_DROP,
            "free-agent add/drop",
            "add_drop",
            _TRANSACTION,
            AddDropPayload,
            acquisition=True,
            may_lose_players=True,
        ),
        KindSpec(
            ProposalKind.WAIVER,
            "waiver claim",
            "waiver",
            _TRANSACTION,
            WaiverPayload,
            acquisition=True,
            may_lose_players=True,
        ),
        KindSpec(ProposalKind.WAIVER_CANCEL, "cancel waiver claim", "waiver", _TRANSACTION, TransactionCancelPayload),
        KindSpec(
            ProposalKind.TRADE_PROPOSE,
            "propose trade",
            None,
            _TRADE,
            TradePayload,
            may_lose_players=True,
            trade_window=True,
        ),
        KindSpec(
            ProposalKind.TRADE_ACCEPT,
            "accept trade",
            None,
            _TRADE,
            TradeResponsePayload,
            may_lose_players=True,
            trade_window=True,
        ),
        KindSpec(ProposalKind.TRADE_DECLINE, "decline trade", None, _TRADE, TradeResponsePayload),
        KindSpec(ProposalKind.TRADE_CANCEL, "cancel trade offer", None, _TRADE, TransactionCancelPayload),
    )
}

TRADE_KINDS: tuple[ProposalKind, ...] = tuple(kind for kind in ProposalKind if kind.is_trade)
ACQUISITION_KINDS: tuple[ProposalKind, ...] = tuple(spec.kind for spec in KINDS.values() if spec.acquisition)
COUNTED_STATUSES: tuple[ProposalStatus, ...] = ("proposed", "approved", "executing", "verified")
"""Proposals that hold or used a transaction slot; rejected, expired and failed ones give it back."""


def kind_spec(kind: ProposalKind | str) -> KindSpec:
    """The spec for a kind given as the enum or its stored string; unknown kinds raise ``PolicyError``."""
    try:
        return KINDS[ProposalKind(kind)]
    except ValueError:
        known = ", ".join(member.value for member in ProposalKind)
        raise PolicyError(f"unknown proposal kind {kind!r}; known kinds: {known}") from None


def default_setting(kind: ProposalKind | str) -> Approval:
    return kind_spec(kind).default


def validate_setting(kind: ProposalKind | str, setting: str) -> Approval:
    """Check that ``setting`` is allowed for ``kind``; trade kinds accept only ``approve``."""
    spec = kind_spec(kind)
    if setting not in spec.allowed:
        detail = (
            "trades are approval-only and not configurable (CLAUDE.md)"
            if spec.is_trade
            else f"allowed: {', '.join(spec.allowed)}"
        )
        raise PolicyError(f"{spec.kind.value}: policy {setting!r} is not allowed; {detail}")
    return cast(Approval, setting)


def effective_setting(kind: ProposalKind | str, policy: Policy) -> Approval:
    """The setting a league's policy gives a kind: its policy field's value, or ``approve`` for every trade kind."""
    spec = kind_spec(kind)
    if spec.policy_field is None:
        return "approve"
    return validate_setting(spec.kind, getattr(policy, spec.policy_field))


def parse_payload(proposal: ProposalRow) -> Payload:
    """The stored payload as the model for the proposal's kind."""
    return kind_spec(proposal.kind).payload_type.model_validate(proposal.payload)


def as_utc(now: datetime | None = None) -> datetime:
    """``now`` in UTC, or the current time; naive datetimes are refused so deadlines always compare."""
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return now.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class Verdict:
    """The outcome of :func:`evaluate`: the setting the proposal would run under and why it is blocked, if it is."""

    kind: ProposalKind
    setting: Approval
    reasons: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return not self.reasons

    def raise_if_blocked(self) -> None:
        if self.reasons:
            raise PolicyError(f"{self.kind.value} blocked: " + "; ".join(self.reasons))


def evaluate(
    store: Store,
    config: Config,
    league: LeagueRow,
    kind: ProposalKind | str,
    payload: Payload,
    *,
    scoring_period_id: int | None = None,
    deadline: datetime | None = None,
    now: datetime | None = None,
) -> Verdict:
    """Run every policy check and guardrail for a would-be proposal without storing anything.

    ``league`` is the store's row for the league the move is in; ``deadline`` is when the move stops making sense
    (lock, waiver run, first tip). A payload of the wrong type for the kind is a programming error (``TypeError``).
    """
    spec = kind_spec(kind)
    if not isinstance(payload, spec.payload_type):
        raise TypeError(f"{spec.kind.value} takes a {spec.payload_type.__name__}, not {type(payload).__name__}")
    at = as_utc(now)
    reasons: list[str] = []

    configured = _allowlisted(config, league, reasons)
    policy = configured.policy if configured is not None else Policy()
    setting = effective_setting(spec.kind, policy)
    if setting == "off":
        reasons.append(f"{spec.kind.value} is off for league {league.key!r} ([league.policy] in config.toml)")
    if setting == "auto" and deadline is None:
        reasons.append("an auto proposal needs a deadline: auto fires at T-15 when the proposal is unanswered")
    if deadline is not None and as_utc(deadline) <= at:
        reasons.append(f"deadline {as_utc(deadline):%Y-%m-%d %H:%M} UTC has passed")
    paused = pause_state()
    if paused is not None:
        reasons.append(f"{paused.describe()}; run fm resume")

    if configured is not None:
        settings = stored_settings(store, league)
        if spec.trade_window:
            reasons.extend(_trade_window_reasons(settings, at))
        if spec.may_lose_players and payload.outgoing_players:
            for espn_id, name in find_untouchables(store, league.sport, policy, payload.outgoing_players).items():
                reasons.append(f"{name} ({espn_id}) is untouchable")
        if spec.acquisition:
            used = acquisitions_this_week(store, league, settings, scoring_period_id=scoring_period_id, now=at)
            cap = policy.max_transactions_per_week
            if used >= cap:
                reasons.append(f"{used} of {cap} transactions already used this week (max_transactions_per_week)")
        if payload.bid is not None:
            reasons.extend(_bid_reasons(payload.bid, policy, settings))
    return Verdict(spec.kind, setting, tuple(reasons))


def _allowlisted(config: Config, league: LeagueRow, reasons: list[str]) -> League | None:
    """The config entry for the store's league row, or ``None`` (with the reason) when it is not allowlisted."""
    try:
        configured = config.league(league.key)
    except KeyError:
        reasons.append(f"league {league.key!r} is not in config.toml (league allowlist)")
        return None
    stored = (league.sport, league.espn_league_id, league.season)
    wanted = (configured.sport, configured.espn_league_id, configured.season)
    if stored != wanted:
        reasons.append(
            f"league {league.key!r} in the store is ESPN {stored[0]} league {stored[1]} season {stored[2]}, but "
            f"config.toml says {wanted[0]} league {wanted[1]} season {wanted[2]}; run fm sync"
        )
        return None
    return configured


def stored_settings(store: Store, league: LeagueRow) -> LeagueSettings | None:
    """The league's synced ESPN settings, or ``None`` before the first ``fm sync``."""
    row = store.settings.get(league.row_id)
    return None if row is None else LeagueSettings.model_validate(row.settings)


def faab_bid_cap(policy: Policy, settings: LeagueSettings) -> int | None:
    """The largest single bid policy allows: ``max_faab_pct_per_bid`` of the season budget, rounded down. ``None``
    when the league does not bid (waiver priority)."""
    budget = settings.acquisition.budget
    if not settings.acquisition.uses_faab or budget is None:
        return None
    return math.floor(Fraction(budget) * Fraction(repr(policy.max_faab_pct_per_bid)))


def _bid_reasons(bid: int, policy: Policy, settings: LeagueSettings | None) -> list[str]:
    if settings is None:
        return ["FAAB budget unknown: league settings are not synced (run fm sync)"]
    cap = faab_bid_cap(policy, settings)
    if cap is None:
        return [f"a bid of ${bid} was given but the league does not use FAAB"]
    if bid > cap:
        pct = policy.max_faab_pct_per_bid
        return [f"bid ${bid} exceeds {pct:.0%} of the ${settings.acquisition.budget} FAAB budget (${cap})"]
    return []


def _trade_window_reasons(settings: LeagueSettings | None, at: datetime) -> list[str]:
    """Blocked once the synced settings say the league's trade deadline has passed. Before the first ``fm sync`` the
    deadline is unknown and policy does not block; the executor's precondition check reads the live league anyway."""
    if settings is None or settings.trade.deadline is None or settings.trade.is_open(at):
        return []
    return [f"the league's trade deadline {settings.trade.deadline:%Y-%m-%d %H:%M} UTC has passed"]


def find_untouchables(store: Store, sport: Sport, policy: Policy, espn_ids: Iterable[int]) -> dict[int, str]:
    """The untouchable players among ``espn_ids``: ESPN id -> display name.

    List entries that are ints (or digit strings) match ids directly; other strings match the stored ``full_name``
    case-insensitively, ignoring spaces, periods, apostrophes and hyphens (``"A.J. Brown"`` is ``"aj brown"``), since a
    match that is too loose only blocks a move. A player the store has not seen yet can only match by id.
    """
    wanted = set(espn_ids)
    if not wanted or not policy.untouchables:
        return {}
    ids: set[int] = set()
    names: set[str] = set()
    for entry in policy.untouchables:
        if isinstance(entry, int):
            ids.add(entry)
        elif entry.strip().isdigit():
            ids.add(int(entry))
        else:
            names.add(_normalize_name(entry))
    players = {player.espn_id: player for player in store.players.many(sport, wanted)}
    found: dict[int, str] = {}
    for espn_id in sorted(wanted):
        player = players.get(espn_id)
        if espn_id in ids or (player is not None and _normalize_name(player.full_name) in names):
            found[espn_id] = player.full_name if player is not None else str(espn_id)
    return found


def _normalize_name(name: str) -> str:
    return re.sub(r"[\s.'\-]", "", name).casefold()


def acquisitions_this_week(
    store: Store,
    league: LeagueRow,
    settings: LeagueSettings | None,
    *,
    scoring_period_id: int | None,
    now: datetime,
) -> int:
    """How many transaction slots the league's add/drop and waiver proposals hold this week (``COUNTED_STATUSES``).

    The week is the matchup period containing ``scoring_period_id`` when its scoring periods can be told, plus the rows
    stored without a scoring period in the seven days before ``now``: the synced settings list them
    (``ScheduleSettings.lists_scoring_periods``: NFL weeks) or ESPN's season calendar resolves them
    (:func:`fm.espn.calendar.matchup_scoring_periods`: the real NBA league's weekly matchups, days 1-6, 7-13, ...).
    Otherwise it is every row created in the seven days before ``now``: without settings, a period or a calendar for the
    season; reading a week id as a day would count one day's moves against a seven-day cap.
    """
    rows = store.proposals.find(
        league_id=league.row_id, statuses=COUNTED_STATUSES, kinds=[kind.value for kind in ACQUISITION_KINDS]
    )
    in_week = week_filter(settings, scoring_period_id, now)
    return sum(1 for row in rows if in_week(row))


def week_filter(
    settings: LeagueSettings | None, scoring_period_id: int | None, now: datetime
) -> Callable[[ProposalRow], bool]:
    """Whether a proposal falls in the same week as one for ``scoring_period_id`` (see ``acquisitions_this_week``).

    A row stored without a scoring period can only be placed by its creation time, so it counts when that is inside
    the trailing seven days however the week is defined; otherwise a producer that omits the period (an MCP
    ``create_proposal``, say) would free the week for one that supplies it.
    """
    since = as_utc(now) - ROLLING_WEEK
    if settings is not None and scoring_period_id is not None:
        listed = matchup_scoring_periods(settings, scoring_period_id)
        if listed is not None:
            periods = frozenset(listed)
            return lambda row: (
                row.scoring_period_id in periods or (row.scoring_period_id is None and row.created_at >= since)
            )
    return lambda row: row.created_at >= since
