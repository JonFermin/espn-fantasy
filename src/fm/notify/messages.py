"""What the phone shows (DESIGN sections 4 and 11): proposal pushes, decision confirmations, alerts and reports.

:func:`proposal_message` describes a proposal so it can be decided from a lock screen, and nothing more: the move as
one line per action in player names and slot labels (an ESPN id only for a player the store has not seen), the
rationale (which already states the numbers that matter), and the deadline in local time with what happens if nobody
answers. The full ``engine_numbers`` stay in the store for ``fm`` and the audit. :func:`alert` is for something that
needs attention now and can carry a deep link to the manual fix (DESIGN principle 6); :func:`report` is the quiet,
longer push the weekly report sends.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, tzinfo

from fm.espn.ids import ids_for
from fm.notify.base import DecisionResult, Message
from fm.proposals import (
    AUTO_LEAD,
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

SPORT_ICONS: Mapping[str, str] = {"nfl": "🏈", "nba": "🏀"}


def proposal_message(
    store: Store, league: LeagueRow, row: ProposalRow, *, now: datetime | None = None, tz: tzinfo | None = None
) -> Message:
    """The push for a proposal waiting on a decision. ``tz`` is the zone the deadline is shown in (the machine's own
    zone by default)."""
    at = as_utc(now)
    lines = _move_lines(store, league, parse_payload(row))
    if row.rationale and row.rationale.strip():
        lines.append(row.rationale.strip())
    lines.append(_timing(row, at, tz))
    icon = SPORT_ICONS.get(league.sport, "")
    label = kind_spec(row.kind).label
    title = f"{icon} {league.key.upper()}: {label[:1].upper()}{label[1:]} (#{row.row_id})".strip()
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


def _move_lines(store: Store, league: LeagueRow, payload: Payload) -> list[str]:
    """One line per action, verb first: ``Bench: A, B`` / ``Start: C (WR)`` for a lineup, ``+ Add`` / ``- Drop`` for a
    transaction, ``Give`` / ``Get`` for a trade."""
    names = player_names(store, league.sport, payload)

    def who(espn_id: int) -> str:
        return names.get(espn_id, f"player {espn_id}")

    if isinstance(payload, LineupPayload):
        slots = ids_for(league.sport)
        benched: list[str] = []
        started: list[str] = []
        moved: list[str] = []
        for move in payload.moves:
            to_label = slots.slot_label(move.to_slot_id)
            if move.to_slot_id == slots.bench_slot:
                benched.append(who(move.espn_id))
            elif not slots.is_active_slot(move.from_slot_id) and slots.is_active_slot(move.to_slot_id):
                started.append(f"{who(move.espn_id)} ({to_label})")
            else:
                moved.append(f"{who(move.espn_id)} ({slots.slot_label(move.from_slot_id)} → {to_label})")
        groups = (("Bench", benched), ("Start", started), ("Move", moved))
        return [f"{verb}: {', '.join(group)}" for verb, group in groups if group]
    if isinstance(payload, AddDropPayload | WaiverPayload):
        verb = "Claim" if isinstance(payload, WaiverPayload) else "Add"
        lines = []
        if payload.add_espn_id is not None:
            add = f"➕ {verb} {who(payload.add_espn_id)}"
            if isinstance(payload, WaiverPayload) and payload.bid_amount is not None:
                add += f" (bid ${payload.bid_amount})"
            lines.append(add)
        if payload.drop_espn_id is not None:
            lines.append(f"➖ Drop {who(payload.drop_espn_id)}")
        return lines
    if isinstance(payload, TradePayload):  # TradeResponsePayload too
        team = store.teams.get(league.row_id, payload.other_team_id)
        other = team.name if team is not None else f"team {payload.other_team_id}"
        give = ", ".join(who(espn_id) for espn_id in payload.give_espn_ids) or "nothing"
        get = ", ".join(who(espn_id) for espn_id in payload.get_espn_ids) or "nothing"
        return [f"With {other}", f"Give: {give}", f"Get: {get}"]
    return [payload.summary()]


def _player_ids(payload: Payload) -> list[int]:
    if isinstance(payload, LineupPayload):
        return [move.espn_id for move in payload.moves]
    if isinstance(payload, AddDropPayload | WaiverPayload):
        return [espn_id for espn_id in (payload.add_espn_id, payload.drop_espn_id) if espn_id is not None]
    if isinstance(payload, TradePayload):
        return [*payload.give_espn_ids, *payload.get_espn_ids]
    return []


def _timing(row: ProposalRow, at: datetime, tz: tzinfo | None) -> str:
    """When it is due and what silence means: ``auto`` goes ahead at T-15, anything else lapses at the deadline."""
    if row.deadline is None:
        return "⏰ No deadline"
    local = row.deadline.astimezone(tz)
    when = f"⏰ Decide by {local:%a %d %b %H:%M} {_zone(local)} ({relative(row.deadline - at)})"
    if row.policy == "auto":
        lead = int(AUTO_LEAD.total_seconds()) // 60
        return f"{when}\nNo answer: it goes ahead automatically {lead} min before"
    return f"{when}\nNo answer: nothing happens"


def _zone(value: datetime) -> str:
    """A short zone name (``EDT``, ``UTC``): Windows' long names (``Mountain Daylight Time``) as their initials, else
    the UTC offset."""
    name = value.tzname() or ""
    if 0 < len(name) <= 5:
        return name
    words = name.split()
    if len(words) > 1 and all(word[:1].isalpha() for word in words):
        return "".join(word[0].upper() for word in words)
    offset = value.utcoffset() or timedelta()
    sign = "-" if offset < timedelta() else "+"
    minutes = abs(int(offset.total_seconds())) // 60
    return f"UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"
