"""Proposal payloads: what a proposal asks the executor to do, one model per kind (DESIGN sections 6.3 and 11).

The payload is the contract between the decision modules that create proposals, the policy guardrails that screen
them, and the executor flows that carry them out. Each model holds the ESPN ids the transaction envelope needs
(``LINEUP {playerId, fromLineupSlotId, toLineupSlotId}``, ``ADD`` / ``DROP``, ``bidAmount``, ``TRADE`` with the other
team, ``relatedTransactionId`` for responses and cancels) and tells the guardrails which players would leave our roster
(``outgoing_players``) and what is being bid (``bid``). Everything is an ESPN id; names are looked up for display.

Payloads are frozen and reject unknown fields. ``model_dump(mode="json")`` is what the ``proposals.payload`` column
stores; :func:`fm.proposals.policy.parse_payload` reads a stored row back into the model for its kind.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_PAYLOAD = ConfigDict(frozen=True, extra="forbid")


class Payload(BaseModel):
    """Base of every payload. Subclasses override the two guardrail hooks when they apply."""

    model_config = _PAYLOAD

    @property
    def outgoing_players(self) -> tuple[int, ...]:
        """ESPN ids that would leave our roster if the move went through; the untouchables check reads this."""
        return ()

    @property
    def bid(self) -> int | None:
        """The FAAB bid in dollars, for the per-bid cap; ``None`` when the move is not a bid."""
        return None

    def summary(self) -> str:
        """One line for ``fm proposals list`` and notifications."""
        return self.model_dump_json()


class LineupMove(BaseModel):
    """One slot change: ``playerId`` from ``fromLineupSlotId`` to ``toLineupSlotId``."""

    model_config = _PAYLOAD

    espn_id: int
    from_slot_id: int
    to_slot_id: int

    @model_validator(mode="after")
    def _slots_differ(self) -> Self:
        if self.from_slot_id == self.to_slot_id:
            raise ValueError(f"player {self.espn_id} is already in slot {self.to_slot_id} (TRAN_ROSTER_SAME_SLOT)")
        return self


class LineupPayload(Payload):
    """``bench_inactive`` and ``lineup``: slot moves for one scoring period, with both sides of every swap."""

    moves: tuple[LineupMove, ...] = Field(min_length=1)

    def summary(self) -> str:
        return "; ".join(f"{move.espn_id}: slot {move.from_slot_id} -> {move.to_slot_id}" for move in self.moves)


class AddDropPayload(Payload):
    """``add_drop``: a free-agent add, a drop, or both in one transaction."""

    add_espn_id: int | None = None
    drop_espn_id: int | None = None

    @model_validator(mode="after")
    def _adds_or_drops(self) -> Self:
        if self.add_espn_id is None and self.drop_espn_id is None:
            raise ValueError("an add/drop needs add_espn_id, drop_espn_id or both")
        if self.add_espn_id is not None and self.add_espn_id == self.drop_espn_id:
            raise ValueError(f"player {self.add_espn_id} cannot be both added and dropped")
        return self

    @property
    def outgoing_players(self) -> tuple[int, ...]:
        return () if self.drop_espn_id is None else (self.drop_espn_id,)

    def summary(self) -> str:
        parts = []
        if self.add_espn_id is not None:
            parts.append(f"add {self.add_espn_id}")
        if self.drop_espn_id is not None:
            parts.append(f"drop {self.drop_espn_id}")
        return ", ".join(parts)


class WaiverPayload(Payload):
    """``waiver``: claim ``add_espn_id`` off waivers, dropping ``drop_espn_id`` if given, with a FAAB ``bid_amount``
    in leagues that bid (``None`` in priority leagues)."""

    add_espn_id: int
    drop_espn_id: int | None = None
    bid_amount: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if self.add_espn_id == self.drop_espn_id:
            raise ValueError(f"player {self.add_espn_id} cannot be both claimed and dropped")
        return self

    @property
    def outgoing_players(self) -> tuple[int, ...]:
        return () if self.drop_espn_id is None else (self.drop_espn_id,)

    @property
    def bid(self) -> int | None:
        return self.bid_amount

    def summary(self) -> str:
        text = f"claim {self.add_espn_id}"
        if self.drop_espn_id is not None:
            text += f", drop {self.drop_espn_id}"
        if self.bid_amount is not None:
            text += f", bid ${self.bid_amount}"
        return text


class TradePayload(Payload):
    """``trade_propose``: offer ``give_espn_ids`` (ours) to ``other_team_id`` for ``get_espn_ids`` (theirs)."""

    other_team_id: int
    give_espn_ids: tuple[int, ...] = ()
    get_espn_ids: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _exchanges_something(self) -> Self:
        if not self.give_espn_ids and not self.get_espn_ids:
            raise ValueError("a trade needs players on at least one side")
        both = set(self.give_espn_ids) & set(self.get_espn_ids)
        if both:
            raise ValueError(f"players {sorted(both)} appear on both sides of the trade")
        return self

    @property
    def outgoing_players(self) -> tuple[int, ...]:
        return self.give_espn_ids

    def summary(self) -> str:
        give = ", ".join(map(str, self.give_espn_ids)) or "nothing"
        get = ", ".join(map(str, self.get_espn_ids)) or "nothing"
        return f"with team {self.other_team_id}: give {give}; get {get}"


class TradeResponsePayload(TradePayload):
    """``trade_accept`` / ``trade_decline``: the pending offer (``espn_transaction_id``) and what it exchanges, with
    ``give_espn_ids`` still meaning the players we would send."""

    espn_transaction_id: str = Field(min_length=1)

    def summary(self) -> str:
        return f"offer {self.espn_transaction_id} {super().summary()}"


class TransactionCancelPayload(Payload):
    """``waiver_cancel`` / ``trade_cancel``: withdraw our own pending claim or offer by ESPN transaction id."""

    espn_transaction_id: str = Field(min_length=1)

    def summary(self) -> str:
        return f"cancel transaction {self.espn_transaction_id}"
