"""The ``trade_pitch`` worker: a draft message to the other manager for one approved trade idea (DESIGN §10).

:func:`draft_pitch` takes a stored ``trade_propose`` proposal (what :func:`fm.decide.trades.propose_trades` writes)
and returns a :class:`Pitch`: text the owner reads, edits and sends themselves. It is advisory text only. It never
changes the proposal, never sends anything and has no write path to ESPN; the trade itself stays approval-only and
nothing here can change that.

**Terms are the engine's.** Claude is given the two sides of the deal and writes around them. An answer that does not
name every player given (when names are given) is not used, so a pitch can never misstate the terms. Our own odds and
P(accept) are not sent: the pitch is for the other manager and has no reason to know them.

**Fallback.** When Claude cannot be used the template is returned, never an error and never partial text: a spent
daily budget, an unreachable API, a refusal, a cut-off or unparseable answer, an empty one, or one that drops a
player. :attr:`Pitch.source` says which path (``claude`` or ``fallback``) and :attr:`Pitch.detail` why. A message
longer than :data:`PITCH_MAX_CHARS` is cut at a sentence end.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from fm.advisor.client import AdvisorClient, AdvisorError, BudgetExceededError, Prompt, WorkerReply, effort_for
from fm.advisor.prompts import prompt_text
from fm.proposals.payloads import TradePayload
from fm.proposals.policy import PolicyError, ProposalKind, parse_payload
from fm.store import ProposalRow, utc_now

logger = logging.getLogger(__name__)

PITCH_WORKER: Final = "trade_pitch"
PITCH_MAX_TOKENS: Final = 1024
PITCH_MAX_CHARS: Final = 700
"""A message to another manager is short; a longer answer is cut at a sentence end."""
FACTS_LIMIT: Final = 6
"""Facts about the other side sent at most."""

type PitchSource = Literal["claude", "fallback"]


class PitchError(ValueError):
    """The proposal is not a trade offer a pitch can be written for."""


class PitchOutput(BaseModel):
    """Claude's answer (the schema the API is held to; the length is applied in code)."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(description="The message to the other manager, two to five sentences.")


@dataclass(frozen=True, slots=True)
class Pitch:
    """The draft message for one trade and where it came from. ``reply`` is the client's reply when a request was
    answered (it has the ``llm_usage`` row)."""

    proposal_id: int | None
    text: str
    source: PitchSource
    detail: str | None = None
    reply: WorkerReply[PitchOutput] | None = field(default=None, compare=False)

    @property
    def used_claude(self) -> bool:
        return self.source == "claude"


def trade_payload(proposal: ProposalRow) -> TradePayload:
    """The offer ``proposal`` carries; :class:`PitchError` for any other kind or an unreadable payload."""
    if proposal.kind != ProposalKind.TRADE_PROPOSE.value:
        raise PitchError(f"proposal {proposal.id} is a {proposal.kind!r}, not a trade_propose: no pitch to write")
    try:
        payload = parse_payload(proposal)
    except (PolicyError, ValueError) as exc:
        raise PitchError(f"proposal {proposal.id}: unreadable trade payload: {exc}") from exc
    if not isinstance(payload, TradePayload):
        raise PitchError(f"proposal {proposal.id}: not a trade offer")
    return payload


def _player(espn_id: int, names: Mapping[int, str] | None) -> str:
    return names[espn_id] if names and espn_id in names else f"player {espn_id}"


def _list(ids: tuple[int, ...], names: Mapping[int, str] | None) -> str:
    items = [_player(i, names) for i in ids]
    if not items:
        return "nothing"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def template_pitch(
    payload: TradePayload,
    *,
    names: Mapping[int, str] | None = None,
    their_name: str | None = None,
    our_name: str | None = None,
) -> str:
    """The pitch without Claude: the offer, the terms exactly, and an invitation to counter."""
    greeting = f"Hey {their_name}," if their_name else "Hey,"
    text = (
        f"{greeting} would you be open to a trade? I'd send you {_list(payload.give_espn_ids, names)} "
        f"for {_list(payload.get_espn_ids, names)}. If it is close but not quite right, let me know what would work "
        "for you and I am happy to talk it through."
    )
    return f"{text} - {our_name}" if our_name else text


def _clip(text: str, limit: int = PITCH_MAX_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > 0 else cut.rstrip()


def pitch_text(
    payload: TradePayload,
    *,
    names: Mapping[int, str] | None = None,
    their_name: str | None = None,
    our_name: str | None = None,
    league_label: str | None = None,
    facts: tuple[str, ...] = (),
) -> str:
    """The user block: the two sides by name, the managers, and the facts about why it works for them."""
    lines = []
    if league_label:
        lines.append(f"League: {league_label}")
    lines.append(f"From: {our_name or 'me'}")
    lines.append(f"To: {their_name or f'team {payload.other_team_id}'}")
    lines.append(f"We send them: {_list(payload.give_espn_ids, names)}")
    lines.append(f"They send us: {_list(payload.get_espn_ids, names)}")
    for fact in facts[:FACTS_LIMIT]:
        lines.append(f"Fact about their side: {fact}")
    return "\n".join(lines)


def _fallback(proposal: ProposalRow, text: str, detail: str, reply: WorkerReply[PitchOutput] | None = None) -> Pitch:
    logger.warning("advisor: trade_pitch: proposal %s: %s; template used", proposal.id, detail)
    return Pitch(proposal.id, text, "fallback", detail, reply)


def draft_pitch(
    client: AdvisorClient,
    proposal: ProposalRow,
    *,
    names: Mapping[int, str] | None = None,
    their_name: str | None = None,
    our_name: str | None = None,
    league_label: str | None = None,
    facts: tuple[str, ...] = (),
    now: datetime | None = None,
) -> Pitch:
    """A draft message for the ``trade_propose`` ``proposal``: Claude's, or the template when Claude cannot be used
    (see the module docs). Never raises for a spent budget or an unreachable API; :class:`PitchError` for a proposal
    that is not a trade offer."""
    payload = trade_payload(proposal)
    template = template_pitch(payload, names=names, their_name=their_name, our_name=our_name)
    prompt = Prompt(
        system=prompt_text(PITCH_WORKER),
        user=pitch_text(
            payload, names=names, their_name=their_name, our_name=our_name, league_label=league_label, facts=facts
        ),
        max_tokens=PITCH_MAX_TOKENS,
        effort=effort_for(PITCH_WORKER),
        league_id=proposal.league_id,
    )
    when = now if now is not None else utc_now()
    try:
        reply = client.ask(PITCH_WORKER, prompt, PitchOutput, now=when)
    except BudgetExceededError as exc:
        return _fallback(proposal, template, f"blocked: {exc}")
    except AdvisorError as exc:
        return _fallback(proposal, template, f"failed: {exc}")
    if reply.output is None:
        return _fallback(proposal, template, reply.detail or reply.status, reply)
    text = _clip(reply.output.message)
    if not text:
        return _fallback(proposal, template, "empty message", reply)
    if names:
        missing = [
            names[i] for i in (*payload.give_espn_ids, *payload.get_espn_ids) if i in names and names[i] not in text
        ]
        if missing:
            return _fallback(proposal, template, f"the message leaves out {', '.join(missing)}", reply)
    return Pitch(proposal.id, text, "claude", reply=reply)
