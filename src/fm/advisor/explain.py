"""The ``explain`` worker: a short rationale for each non-trivial proposal (DESIGN §10).

:func:`explain_proposal` takes a stored proposal (:class:`fm.store.ProposalRow`) and returns an
:class:`Explanation`: 2 to 4 sentences for the person who approves it, from the engine's own numbers. It never
changes the proposal and has no write path; the caller stores or shows the text (the weekly report and the proposal
pushes).

**Trivial moves are templated, with no API call.** :func:`is_trivial` is the definition, in code:

- ``bench_inactive`` (bench an OUT, bye or no-game starter: a hard rule, the reason is the designation or the missing
  game);
- ``waiver_cancel`` and ``trade_cancel`` (withdrawing our own pending claim or offer);
- a ``lineup`` change that is a single swap, exactly :data:`TRIVIAL_SWAP_MOVES` slot moves (one player in, one out).

Everything else (a lineup that rearranges more than one swap, free-agent adds and drops, waiver claims, trade
offers and responses, and any kind this module does not know) gets Claude. :func:`template_rationale` is the
template: the engine's own rationale when the proposal has one, else a line built from the kind and the payload.

**Fallback.** When Claude cannot be used the template is returned instead, never an error and never partial text:
a spent daily budget (``blocked``), an unreachable API, a refusal, a cut-off answer (``max_tokens``), an unparseable
one, or an answer with no text. :attr:`Explanation.source` says which path produced the text (``template``,
``claude`` or ``fallback``) and :attr:`Explanation.detail` why. An answer longer than :data:`EXPLAIN_MAX_SENTENCES`
sentences is cut to that many.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fm.advisor.client import AdvisorClient, AdvisorError, BudgetExceededError, Prompt, WorkerReply, effort_for
from fm.advisor.prompts import prompt_text
from fm.proposals.policy import PolicyError, ProposalKind, kind_spec, parse_payload
from fm.store import ProposalRow, utc_now

logger = logging.getLogger(__name__)

EXPLAIN_WORKER: Final = "explain"
EXPLAIN_MAX_TOKENS: Final = 1024
"""Room for four sentences with headroom; effort is low."""
EXPLAIN_MAX_SENTENCES: Final = 4
EXPLAIN_NUMBERS_LIMIT: Final = 4000
"""Characters of the engine's numbers sent; a proposal's numbers are a few hundred, a lineup's a few thousand."""
TRIVIAL_SWAP_MOVES: Final = 2
"""A single swap moves two players: one into the slot the other leaves."""
TRIVIAL_KINDS: Final[frozenset[ProposalKind]] = frozenset(
    {ProposalKind.BENCH_INACTIVE, ProposalKind.WAIVER_CANCEL, ProposalKind.TRADE_CANCEL}
)
"""Kinds that are templated whatever their payload (see the module docs)."""

type ExplainSource = Literal["template", "claude", "fallback"]
"""``template``: trivial, no call; ``claude``: Claude's rationale; ``fallback``: non-trivial, Claude could not be
used, so the template stands in."""

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])")


class ExplainOutput(BaseModel):
    """Claude's answer (the schema the API is held to; the length is applied in code)."""

    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(description="Two to four sentences.")


@dataclass(frozen=True, slots=True)
class Explanation:
    """The rationale for one proposal and where it came from. ``reply`` is the client's reply when a request was
    answered (it has the ``llm_usage`` row)."""

    proposal_id: int | None
    text: str
    source: ExplainSource
    detail: str | None = None
    reply: WorkerReply[ExplainOutput] | None = field(default=None, compare=False)

    @property
    def used_claude(self) -> bool:
        return self.source == "claude"


# --- trivial or not ---------------------------------------------------------------------------------------------------


def is_trivial(proposal: ProposalRow) -> bool:
    """True for a move that is templated and needs no call (see the module docs); a kind this module does not know
    is not trivial."""
    try:
        kind = ProposalKind(proposal.kind)
    except ValueError:
        return False
    if kind in TRIVIAL_KINDS:
        return True
    if kind is ProposalKind.LINEUP:
        moves = proposal.payload.get("moves")
        return isinstance(moves, list) and len(moves) == TRIVIAL_SWAP_MOVES
    return False


def _label(proposal: ProposalRow) -> str:
    try:
        return kind_spec(proposal.kind).label
    except PolicyError:
        return str(proposal.kind)


def _named(text: str, names: Mapping[int, str] | None) -> str:
    """``text`` with each ESPN id that has a name written as ``Name (id)``."""
    if not names:
        return text
    return re.sub(r"\b\d{4,}\b", lambda m: f"{names[int(m[0])]} ({m[0]})" if int(m[0]) in names else m[0], text)


def _payload_summary(proposal: ProposalRow) -> str:
    try:
        return parse_payload(proposal).summary()
    except (PolicyError, ValidationError):
        return json.dumps(proposal.payload, sort_keys=True, separators=(",", ":"), default=str)


def template_rationale(proposal: ProposalRow, *, names: Mapping[int, str] | None = None) -> str:
    """The rationale without Claude: the engine's own when the proposal has one (the decision modules write it from
    their numbers), else the kind and what the payload does, with player names where given."""
    if proposal.rationale and proposal.rationale.strip():
        return proposal.rationale.strip()
    label = _label(proposal)
    return f"{label[:1].upper()}{label[1:]}: {_named(_payload_summary(proposal), names)}."


# --- the call ---------------------------------------------------------------------------------------------------------


def proposal_text(
    proposal: ProposalRow, *, names: Mapping[int, str] | None = None, league_label: str | None = None
) -> str:
    """The user block: the proposal's kind, deadline, payload, the engine's rationale and numbers."""
    lines = [f"Proposal: {_label(proposal)} ({proposal.kind})"]
    if league_label:
        lines.append(f"League: {league_label}")
    if proposal.scoring_period_id is not None:
        lines.append(f"Scoring period: {proposal.scoring_period_id}")
    if proposal.deadline is not None:
        lines.append(f"Deadline: {proposal.deadline:%Y-%m-%dT%H:%M:%SZ}")
    lines.append(f"What it does: {_named(_payload_summary(proposal), names)}")
    if names:
        lines.append("Names: " + ", ".join(f"{espn_id} = {name}" for espn_id, name in sorted(names.items())))
    if proposal.rationale and proposal.rationale.strip():
        lines.append(f"Engine rationale: {proposal.rationale.strip()}")
    numbers = json.dumps(proposal.engine_numbers, sort_keys=True, separators=(",", ":"), default=str)
    if len(numbers) > EXPLAIN_NUMBERS_LIMIT:
        numbers = numbers[:EXPLAIN_NUMBERS_LIMIT].rstrip() + " [...]"
    if proposal.engine_numbers:
        lines.append(f"Engine numbers: {numbers}")
    return "\n".join(lines)


def clip_sentences(text: str, limit: int = EXPLAIN_MAX_SENTENCES) -> str:
    """``text`` cut to its first ``limit`` sentences, whitespace collapsed."""
    sentences = _SENTENCE_END.split(" ".join(text.split()))
    return " ".join(sentences[:limit])


def _fallback(proposal: ProposalRow, names: Mapping[int, str] | None, detail: str, **kwargs: Any) -> Explanation:
    logger.warning("advisor: explain: proposal %s: %s; template used", proposal.id, detail)
    return Explanation(proposal.id, template_rationale(proposal, names=names), "fallback", detail, **kwargs)


def explain_proposal(
    client: AdvisorClient,
    proposal: ProposalRow,
    *,
    names: Mapping[int, str] | None = None,
    league_label: str | None = None,
    now: datetime | None = None,
) -> Explanation:
    """The rationale for ``proposal``: the template for a trivial move (no call), Claude's for the rest, falling
    back to the template when Claude cannot be used (see the module docs). Never raises for a spent budget or an
    unreachable API."""
    if is_trivial(proposal):
        return Explanation(proposal.id, template_rationale(proposal, names=names), "template")
    when = now if now is not None else utc_now()
    prompt = Prompt(
        system=prompt_text(EXPLAIN_WORKER),
        user=proposal_text(proposal, names=names, league_label=league_label),
        max_tokens=EXPLAIN_MAX_TOKENS,
        effort=effort_for(EXPLAIN_WORKER),
        league_id=proposal.league_id,
    )
    try:
        reply = client.ask(EXPLAIN_WORKER, prompt, ExplainOutput, now=when)
    except BudgetExceededError as exc:
        return _fallback(proposal, names, f"blocked: {exc}")
    except AdvisorError as exc:
        return _fallback(proposal, names, f"failed: {exc}")
    if reply.output is None:
        return _fallback(proposal, names, reply.detail or reply.status, reply=reply)
    text = clip_sentences(reply.output.rationale)
    if not text:
        return _fallback(proposal, names, "empty rationale", reply=reply)
    if text != " ".join(reply.output.rationale.split()):
        logger.info("advisor: explain: proposal %s: rationale cut to %d sentences", proposal.id, EXPLAIN_MAX_SENTENCES)
    return Explanation(proposal.id, text, "claude", reply=reply)


def explain_proposals(
    client: AdvisorClient,
    proposals: Iterable[ProposalRow],
    *,
    names: Mapping[int, str] | None = None,
    league_label: str | None = None,
    now: datetime | None = None,
) -> list[Explanation]:
    """One :class:`Explanation` per proposal, in order. Once the daily budget is spent the rest get the template
    without trying again."""
    explanations: list[Explanation] = []
    blocked = False
    for proposal in proposals:
        if blocked and not is_trivial(proposal):
            explanations.append(_fallback(proposal, names, "blocked: the daily budget is spent"))
            continue
        explanation = explain_proposal(client, proposal, names=names, league_label=league_label, now=now)
        blocked = blocked or (explanation.detail or "").startswith("blocked")
        explanations.append(explanation)
    return explanations
