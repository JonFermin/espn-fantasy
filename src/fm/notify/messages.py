"""What the phone shows (DESIGN sections 4 and 11): proposal pushes, decision confirmations, alerts and reports.

:func:`proposal_message` describes a proposal so it can be decided from a lock screen: the move in player names and
slot labels (an ESPN id only for a player the store has not seen), the engine's numbers, the rationale, the deadline
in local time and the policy, since an ``auto`` proposal fires at T-15 when nobody answers. :func:`alert` is for
something that needs attention now and can carry a deep link to the manual fix (DESIGN principle 6); :func:`report`
is the quiet, longer push the weekly report sends.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, tzinfo
from typing import Any

from fm.espn.ids import ids_for
from fm.notify.base import DecisionResult, Message
from fm.proposals import (
    AddDropPayload,
    LineupPayload,
    Payload,
    TradePayload,
    TradeResponsePayload,
    WaiverPayload,
    kind_spec,
    parse_payload,
)
from fm.proposals.policy import as_utc
from fm.store import LeagueRow, ProposalRow, Sport, Store

MAX_ENGINE_NUMBERS = 6
"""How many ``engine_numbers`` entries a proposal push shows."""


def proposal_message(
    store: Store, league: LeagueRow, row: ProposalRow, *, now: datetime | None = None, tz: tzinfo | None = None
) -> Message:
    """The push for a proposal waiting on a decision. ``tz`` is the zone the deadline is shown in (the machine's own
    zone by default)."""
    at = as_utc(now)
    lines = [describe_payload(store, league, parse_payload(row))]
    numbers = _engine_numbers(row.engine_numbers)
    if numbers:
        lines.append(numbers)
    if row.rationale and row.rationale.strip():
        lines.append(row.rationale.strip())
    lines.append(_timing(row, at, tz))
    title = f"{league.key}: {kind_spec(row.kind).label} #{row.row_id}"
    return Message(title=title, body="\n".join(lines), priority="high")


def describe_payload(store: Store, league: LeagueRow, payload: Payload) -> str:
    """The move in words: player names from the store, slot labels from the league's game, the other team's name."""
    names = player_names(store, league.sport, payload)

    def who(espn_id: int) -> str:
        return names.get(espn_id, f"player {espn_id}")

    if isinstance(payload, LineupPayload):
        slots = ids_for(league.sport)
        return "; ".join(
            f"{who(move.espn_id)}: {slots.slot_label(move.from_slot_id)} -> {slots.slot_label(move.to_slot_id)}"
            for move in payload.moves
        )
    if isinstance(payload, AddDropPayload):
        parts: list[str] = []
        if payload.add_espn_id is not None:
            parts.append(f"add {who(payload.add_espn_id)}")
        if payload.drop_espn_id is not None:
            parts.append(f"drop {who(payload.drop_espn_id)}")
        return ", ".join(parts)
    if isinstance(payload, WaiverPayload):
        text = f"claim {who(payload.add_espn_id)}"
        if payload.drop_espn_id is not None:
            text += f", drop {who(payload.drop_espn_id)}"
        if payload.bid_amount is not None:
            text += f", bid ${payload.bid_amount}"
        return text
    if isinstance(payload, TradePayload):  # TradeResponsePayload too
        team = store.teams.get(league.row_id, payload.other_team_id)
        other = team.name if team is not None else f"team {payload.other_team_id}"
        give = ", ".join(who(espn_id) for espn_id in payload.give_espn_ids) or "nothing"
        get = ", ".join(who(espn_id) for espn_id in payload.get_espn_ids) or "nothing"
        text = f"with {other}: give {give}; get {get}"
        if isinstance(payload, TradeResponsePayload):
            text = f"offer {payload.espn_transaction_id} {text}"
        return text
    return payload.summary()  # TransactionCancelPayload names no player


def player_names(store: Store, sport: Sport, payload: Payload) -> dict[int, str]:
    """ESPN id -> full name for every player ``payload`` names that the store has seen."""
    espn_ids = _player_ids(payload)
    if not espn_ids:
        return {}
    return {player.espn_id: player.full_name for player in store.players.many(sport, espn_ids)}


def decision_message(result: DecisionResult) -> Message:
    """The confirmation push after a press: what was recorded, or why it was refused."""
    tags = ("white_check_mark",) if result.ok else ("x",)
    return Message(title=result.headline, body=result.detail, priority="default", tags=tags)


def alert(title: str, body: str = "", *, link: str | None = None) -> Message:
    """Something needs attention now (a missed window, an expired session, a failed execution); ``link`` opens the
    page with the manual fix."""
    return Message(title=title, body=body, priority="high", tags=("warning",), link=link)


def report(title: str, body: str) -> Message:
    """A quiet, longer push (the weekly report), sent silently where the channel allows and split when long."""
    return Message(title=title, body=body, priority="low", tags=("page_facing_up",))


def relative(delta: timedelta) -> str:
    """``in 3d 4h``, ``in 2h 05m``, ``in 12m``, or ``passed``."""
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "passed"
    hours, minutes = divmod(seconds // 60, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"in {days}d {hours}h"
    if hours:
        return f"in {hours}h {minutes:02d}m"
    return f"in {minutes}m"


def _player_ids(payload: Payload) -> list[int]:
    if isinstance(payload, LineupPayload):
        return [move.espn_id for move in payload.moves]
    if isinstance(payload, AddDropPayload | WaiverPayload):
        return [espn_id for espn_id in (payload.add_espn_id, payload.drop_espn_id) if espn_id is not None]
    if isinstance(payload, TradePayload):
        return [*payload.give_espn_ids, *payload.get_espn_ids]
    return []


def _engine_numbers(numbers: Mapping[str, Any]) -> str:
    shown = [f"{key} {_number(value)}" for key, value in list(numbers.items())[:MAX_ENGINE_NUMBERS]]
    if len(numbers) > MAX_ENGINE_NUMBERS:
        shown.append(f"+{len(numbers) - MAX_ENGINE_NUMBERS} more")
    return ", ".join(shown)


def _number(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) < 1 else f"{value:.1f}"
    return str(value)


def _timing(row: ProposalRow, at: datetime, tz: tzinfo | None) -> str:
    policy = f"policy {row.policy}"
    if row.policy == "auto":
        policy += " (fires at T-15 if unanswered)"
    if row.deadline is None:
        return f"No deadline; {policy}"
    local = row.deadline.astimezone(tz)
    return f"Due {local:%a %d %b %H:%M} {_zone(local)} ({relative(row.deadline - at)}); {policy}"


def _zone(value: datetime) -> str:
    """A short zone name (``EDT``, ``UTC``), or the UTC offset when the platform only knows a long one."""
    name = value.tzname() or ""
    if 0 < len(name) <= 5:
        return name
    offset = value.utcoffset() or timedelta()
    sign = "-" if offset < timedelta() else "+"
    minutes = abs(int(offset.total_seconds())) // 60
    return f"UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"
