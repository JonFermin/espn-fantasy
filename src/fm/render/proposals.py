"""Proposal text shared by the phone push, the dashboard and ``fm proposals`` (ROADMAP #61): the trade pitch.

A ``trade_propose`` row carries, in ``engine_numbers["pitch"]``, the message the owner pastes to the other manager
(:mod:`fm.advisor.pitch`), stored by the tick when the offer is new: ``{"text", "source", "detail"}`` with ``source``
``claude`` when Claude wrote it or ``template`` for the engine's wording (also when Claude could not be used, with
``detail`` saying why). :func:`stored_pitch` reads it back defensively, since a row from before pitches existed, or one
another writer touched, may carry none. Nothing here sends or writes anything: the pitch is text for the owner.

:func:`plain_pitch` is the pitch without Claude, word for word what :func:`fm.advisor.pitch.template_pitch` writes
(a test holds the two together). It lives here too so a tick with no advisor can store it without importing
``fm.advisor``, whose client loads the Anthropic SDK. :func:`pitch_facts` turns a trade offer's ``engine_numbers``
into the facts about the other side that :func:`fm.advisor.pitch.draft_pitch` may use.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from fm.proposals.payloads import TradePayload
from fm.store import ProposalRow

PITCH_KEY: Final = "pitch"
"""The ``engine_numbers`` key the pitch lives under."""


@dataclass(frozen=True, slots=True)
class StoredPitch:
    """A pitch as a proposal row keeps it (see the module docs)."""

    text: str
    source: str
    detail: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {"text": self.text, "source": self.source, "detail": self.detail}


def stored_pitch(row: ProposalRow) -> StoredPitch | None:
    """The row's pitch, or ``None`` when it has none (not a trade offer, from before pitches existed, or malformed)."""
    raw = row.engine_numbers.get(PITCH_KEY)
    if not isinstance(raw, Mapping):
        return None
    text, source, detail = raw.get("text"), raw.get("source"), raw.get("detail")
    if not isinstance(text, str) or not text.strip() or not isinstance(source, str) or not source:
        return None
    return StoredPitch(text.strip(), source, detail if isinstance(detail, str) and detail else None)


def with_pitch(numbers: Mapping[str, Any], pitch: StoredPitch) -> dict[str, Any]:
    """``numbers`` with ``pitch`` set under :data:`PITCH_KEY`: what the tick stores back as the row's
    ``engine_numbers``, everything else untouched."""
    return {**numbers, PITCH_KEY: pitch.as_json()}


def _player(espn_id: int, names: Mapping[int, str] | None) -> str:
    return names[espn_id] if names and espn_id in names else f"player {espn_id}"


def _list(ids: tuple[int, ...], names: Mapping[int, str] | None) -> str:
    items = [_player(i, names) for i in ids]
    if not items:
        return "nothing"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def plain_pitch(
    payload: TradePayload,
    *,
    names: Mapping[int, str] | None = None,
    their_name: str | None = None,
    our_name: str | None = None,
) -> str:
    """The pitch without Claude: the offer, the terms exactly, and an invitation to counter (the same text as
    :func:`fm.advisor.pitch.template_pitch`, without loading the advisor)."""
    greeting = f"Hey {their_name}," if their_name else "Hey,"
    text = (
        f"{greeting} would you be open to a trade? I'd send you {_list(payload.give_espn_ids, names)} "
        f"for {_list(payload.get_espn_ids, names)}. If it is close but not quite right, let me know what would work "
        "for you and I am happy to talk it through."
    )
    return f"{text} - {our_name}" if our_name else text


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def pitch_facts(numbers: Mapping[str, Any]) -> tuple[str, ...]:
    """What a trade offer's ``engine_numbers`` (:func:`fm.decide.trades.engine_numbers`) say for the other side, as the
    facts the pitch may use: only what favors them, since the message is for them. Their rest-of-season gain, the
    lineup need the players they receive fill, and the market value they receive against what they send. Empty for
    numbers without them (a hand-made offer), so the pitch rests on the terms alone."""
    facts: list[str] = []
    unit = numbers.get("unit")
    theirs = numbers.get("theirs")
    if isinstance(theirs, Mapping):
        gain = _number(theirs.get("delta_ros"))
        if gain is not None and gain > 0:
            suffix = f" {unit}" if isinstance(unit, str) and unit else ""
            facts.append(f"Their rest-of-season lineup value improves by {gain:.1f}{suffix}.")
    acceptance = numbers.get("acceptance")
    if isinstance(acceptance, Mapping):
        need = _number(acceptance.get("need_gain"))
        if need is not None and need > 0:
            facts.append(f"The players they receive fill a need in their lineup (+{need:.2f} starter seasons).")
        receive, send = _number(acceptance.get("market_receive")), _number(acceptance.get("market_send"))
        surplus = _number(acceptance.get("surplus"))
        if receive is not None and send is not None and surplus is not None and surplus >= 0:
            rank_type = acceptance.get("rank_type")
            label = f"{rank_type} " if isinstance(rank_type, str) and rank_type else ""
            facts.append(f"By {label}market value they receive {receive:,.0f} for {send:,.0f}.")
    return tuple(facts)


__all__ = ["PITCH_KEY", "StoredPitch", "pitch_facts", "plain_pitch", "stored_pitch", "with_pitch"]
