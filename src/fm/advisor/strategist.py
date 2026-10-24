"""The ``weekly_strategist`` worker: the weekly priorities, punts and targets of one league (DESIGN §9.5, §10).

:func:`run_strategist` produces a :class:`StrategyReport` (the note the weekly report carries) and drafts the report's
trade targets as proposals. The numbers are the engine's, and so are the decisions (:mod:`fm.decide.strategy`, whose
names this module re-exports so a caller reads them from here):

- **Plan.** :func:`engine_plan` is :func:`fm.decide.weekly.plan_weekly` with the simulation's category odds. A league
  without category matchups has no plan (the warning says why) and the report is built from the odds and the trade
  ideas alone.
- **Punts** are the plan's: Claude is never asked for them and cannot add one.
- **Targets** are :class:`StrategyIdea` values, the legal, positive-score deals :func:`fm.decide.trades.evaluate_trade`
  found. Claude may add one sentence to an idea it is shown (``target_notes``), and nothing else: it cannot add,
  remove or reorder a target, and a note about an idea it was not shown is dropped and logged. A Claude-only signal
  never triggers a trade.
- **Proposals** are drafted through :func:`propose_targets` as ``trade_propose`` with ``max_setting="approve"``:
  trades stay approval-only, under the trade finder's etiquette (one open offer a team, a weekly cap, ``max_offers``
  a run) and dedupe key. A policy refusal is reported in :attr:`StrategyProposal.blocked`, never raised.

**Fallback.** The report always has a template (:func:`template_report`, from the plan, the odds and the ideas);
Claude's text replaces its summary and priorities when it answers. A spent budget, an unreachable API, a refusal, a
cut-off or unparseable answer or an empty one leaves the template, with :attr:`StrategyReport.source` ``fallback`` and
the reason in :attr:`StrategyReport.detail`. With nothing to say (no plan, no odds, no ideas) there is no call at all.
The tick (:mod:`fm.jobs.tick`) uses the template and :func:`propose_targets` straight from :mod:`fm.decide.strategy`
when it has no advisor, so a keyless tick never loads this module or the SDK.

This module has no write path to ESPN: it stores proposals through the queue and nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from fm.advisor.client import AdvisorClient, AdvisorError, BudgetExceededError, Prompt, WorkerReply, effort_for
from fm.advisor.prompts import prompt_text
from fm.config import Config
from fm.decide.strategy import (
    MAX_IDEAS,
    MAX_LINE_CHARS,
    MAX_PRIORITIES,
    MAX_SUMMARY_CHARS,
    STRATEGIST_CREATED_BY,
    EnginePlan,
    StrategyIdea,
    StrategyInputs,
    StrategyProposal,
    StrategyReport,
    StrategyResult,
    StrategySource,
    engine_plan,
    idea_from,
    idea_line,
    propose_targets,
    strategy_text,
    template_report,
)
from fm.store import LeagueRow, Store, utc_now

logger = logging.getLogger(__name__)

STRATEGIST_WORKER: Final = "weekly_strategist"
STRATEGIST_MAX_TOKENS: Final = 4096


class TargetNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idea: int = Field(description="The number of a trade idea, as listed.")
    note: str = Field(description="One sentence on why it matters this week.")


class StrategyOutput(BaseModel):
    """Claude's answer (the schema the API is held to; the bounds are applied in code)."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="Two to four sentences on where the team stands.")
    priorities: list[str] = Field(default_factory=list, description="Up to five short lines, most important first.")
    target_notes: list[TargetNote] = Field(default_factory=list)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def write_strategy(client: AdvisorClient, inputs: StrategyInputs, *, now: datetime | None = None) -> StrategyReport:
    """The report: Claude's summary, priorities and target notes laid over the template, or the template when Claude
    cannot be used (see the module docs). Never raises for a spent budget or an unreachable API."""
    base = template_report(inputs)
    if inputs.empty:
        return base
    prompt = Prompt(
        system=prompt_text(STRATEGIST_WORKER),
        user=strategy_text(inputs),
        max_tokens=STRATEGIST_MAX_TOKENS,
        effort=effort_for(STRATEGIST_WORKER),
        league_id=inputs.league_id,
    )
    when = now if now is not None else utc_now()

    def fallback(detail: str, reply: WorkerReply[StrategyOutput] | None = None) -> StrategyReport:
        logger.warning("advisor: weekly_strategist: %s: %s; template used", inputs.league_label, detail)
        return replace(base, source="fallback", detail=detail, reply=reply)

    try:
        reply = client.ask(STRATEGIST_WORKER, prompt, StrategyOutput, now=when)
    except BudgetExceededError as exc:
        return fallback(f"blocked: {exc}")
    except AdvisorError as exc:
        return fallback(f"failed: {exc}")
    if reply.output is None:
        return fallback(reply.detail or reply.status, reply)
    out = reply.output
    summary = _clip(out.summary, MAX_SUMMARY_CHARS)
    if not summary:
        return fallback("empty summary", reply)
    priorities = tuple(line for line in (_clip(p, MAX_LINE_CHARS) for p in out.priorities) if line)[:MAX_PRIORITIES]
    shown = inputs.targets[:MAX_IDEAS]
    notes: dict[str, str] = {}
    for item in out.target_notes:
        if not 1 <= item.idea <= len(shown):
            logger.info("advisor: weekly_strategist: note for idea %d, which was not shown; dropped", item.idea)
            continue
        key = shown[item.idea - 1].key
        note = _clip(item.note, MAX_LINE_CHARS)
        if note and key not in notes:
            notes[key] = note
    return replace(
        base,
        summary=summary,
        priorities=priorities or base.priorities,
        target_notes=notes,
        source="claude",
        reply=reply,
    )


def run_strategist(
    client: AdvisorClient,
    store: Store,
    config: Config,
    league: LeagueRow,
    inputs: StrategyInputs,
    *,
    max_offers: int = 3,
    dry_run: bool = False,
    now: datetime | None = None,
    deadline: datetime | None = None,
) -> StrategyResult:
    """Write the weekly report and draft its targets as proposals (module docs). The report's ``targets`` are exactly
    the engine's eligible ideas: Claude's text never changes which are proposed. ``deadline`` is the drafts'
    (:func:`propose_targets`)."""
    at = now if now is not None else utc_now()
    report = write_strategy(client, inputs, now=at)
    proposals = propose_targets(
        store, config, league, report.targets, max_offers=max_offers, dry_run=dry_run, now=at, deadline=deadline
    )
    return StrategyResult(report, tuple(proposals))


__all__ = [
    "MAX_IDEAS",
    "MAX_LINE_CHARS",
    "MAX_PRIORITIES",
    "MAX_SUMMARY_CHARS",
    "STRATEGIST_CREATED_BY",
    "STRATEGIST_MAX_TOKENS",
    "STRATEGIST_WORKER",
    "EnginePlan",
    "StrategyIdea",
    "StrategyInputs",
    "StrategyOutput",
    "StrategyProposal",
    "StrategyReport",
    "StrategyResult",
    "StrategySource",
    "TargetNote",
    "engine_plan",
    "idea_from",
    "idea_line",
    "propose_targets",
    "run_strategist",
    "strategy_text",
    "template_report",
    "write_strategy",
]
