"""The ``weekly_strategist`` worker: the weekly priorities, punts and targets of one league (DESIGN §9.5, §10).

:func:`run_strategist` produces a :class:`StrategyReport` (the note the weekly report carries) and drafts the report's
trade targets as proposals. The numbers are the engine's, and so are the decisions:

- **Plan.** For an NBA head-to-head category league :func:`engine_plan` is :func:`fm.decide.weekly.plan_weekly`: per
  category P(win), ``contest`` / ``punt`` / ``safe``, the stat gap and the swing weight. When the caller has a
  :class:`fm.model.simulate.SeasonOdds` its per-category probabilities for our team are passed in as ``simulated``,
  so the plan and the simulation agree. A league without category matchups has no plan (the warning says why) and the
  report is built from the odds and the trade ideas alone.
- **Punts** are the plan's: Claude is never asked for them and cannot add one.
- **Targets** are :class:`StrategyIdea` values, the legal, positive-score deals :func:`fm.decide.trades.evaluate_trade`
  found. Claude may add one sentence to an idea it is shown (``target_notes``), and nothing else: it cannot add,
  remove or reorder a target, and a note about an idea it was not shown is dropped and logged. A Claude-only signal
  never triggers a trade.
- **Proposals** are drafted through :func:`fm.proposals.propose` as ``trade_propose`` with ``max_setting="approve"``:
  trades stay approval-only. Etiquette is :func:`fm.decide.trades.propose_trades`'s (one open offer a team, a weekly
  cap, ``max_offers`` a run) and the dedupe key is the same, so a deal the trade finder already stored is returned,
  never stored twice. A policy refusal is reported in :attr:`StrategyProposal.blocked`, never raised.

**Fallback.** The report always has a template (from the plan, the odds and the ideas); Claude's text replaces its
summary and priorities when it answers. A spent budget, an unreachable API, a refusal, a cut-off or unparseable answer
or an empty one leaves the template, with :attr:`StrategyReport.source` ``fallback`` and the reason in
:attr:`StrategyReport.detail`. With nothing to say (no plan, no odds, no ideas) there is no call at all.

This module has no write path to ESPN: it stores proposals through the queue and nothing else.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from fm.advisor.client import AdvisorClient, AdvisorError, BudgetExceededError, Prompt, WorkerReply, effort_for
from fm.advisor.prompts import prompt_text
from fm.config import Config
from fm.decide.trades import WEEKLY_OFFERS, TradeContext, TradeEvaluation, trade_etiquette
from fm.decide.trades import engine_numbers as trade_engine_numbers
from fm.decide.trades import rationale as trade_rationale
from fm.decide.weekly import WeeklyPlan, WeeklyPlanError, plan_weekly
from fm.model.simulate import SeasonOdds, TeamOdds
from fm.proposals.policy import PolicyError, ProposalKind, evaluate
from fm.proposals.queue import propose
from fm.sports.base import ScheduleLike
from fm.store import LeagueRow, ProposalRow, Store, utc_now

logger = logging.getLogger(__name__)

STRATEGIST_WORKER: Final = "weekly_strategist"
STRATEGIST_CREATED_BY: Final = "advisor.strategist"
STRATEGIST_MAX_TOKENS: Final = 4096
MAX_PRIORITIES: Final = 5
MAX_LINE_CHARS: Final = 240
MAX_SUMMARY_CHARS: Final = 700
MAX_IDEAS: Final = 10
"""Trade ideas shown to Claude at most (the engine's best, by score)."""

type StrategySource = Literal["template", "claude", "fallback"]
"""``template``: nothing to ask (or no client); ``claude``: Claude's text; ``fallback``: Claude could not be used."""


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


# --- the engine's side ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyIdea:
    """One trade idea the engine found: the evaluation, the proposal's dedupe ``key``, the ``numbers`` and
    ``rationale`` a proposal would carry, and the scoring period it locks in. :func:`idea_from` builds one."""

    evaluation: TradeEvaluation
    key: str
    numbers: Mapping[str, Any] = field(default_factory=dict)
    rationale: str | None = None
    scoring_period_id: int | None = None

    @property
    def eligible(self) -> bool:
        """Legal, helps us, and needs no drop on our side (a trade payload cannot carry one)."""
        ev = self.evaluation
        return ev.legal and ev.score is not None and ev.score > 0 and not ev.legality.drops_ours

    @property
    def score(self) -> float:
        return self.evaluation.score if self.evaluation.score is not None else float("-inf")


def idea_from(ctx: TradeContext, evaluation: TradeEvaluation) -> StrategyIdea:
    """A :class:`StrategyIdea` for ``evaluation`` found in ``ctx``, with the numbers and rationale the trade finder's
    own proposals carry and the same dedupe key."""
    return StrategyIdea(
        evaluation,
        f"{ctx.league.key}:{evaluation.spec.key}",
        trade_engine_numbers(ctx, evaluation),
        trade_rationale(ctx, evaluation),
        ctx.lock_period,
    )


@dataclass(frozen=True, slots=True)
class EnginePlan:
    """What :func:`engine_plan` found: the plan (``None`` when there is nothing to plan) and why."""

    plan: WeeklyPlan | None
    warnings: tuple[str, ...] = ()


def engine_plan(
    store: Store,
    league: LeagueRow,
    *,
    schedule: ScheduleLike,
    now: datetime,
    odds: SeasonOdds | None = None,
    **options: Any,
) -> EnginePlan:
    """The week's category plan for ``league`` via :func:`fm.decide.weekly.plan_weekly`, with the simulation's
    per-category probabilities for our team passed as ``simulated`` when ``odds`` has them. A league that cannot be
    planned (no category matchups, not synced, season over) is a warning, never an error."""
    simulated = None
    if odds is not None and league.team_id is not None and league.team_id in odds.teams:
        simulated = dict(odds.team(league.team_id).categories) or None
    try:
        decision = plan_weekly(store, league, schedule=schedule, now=now, simulated=simulated, **options)
    except WeeklyPlanError as exc:
        return EnginePlan(None, (str(exc),))
    return EnginePlan(decision.plan, decision.warnings)


@dataclass(frozen=True, slots=True)
class StrategyInputs:
    """Everything the strategist reads, all of it the engine's: ``plan`` (``None`` without category matchups),
    ``our_odds`` (our :class:`~fm.model.simulate.TeamOdds` from the simulation), ``ideas`` (the trade targets) and the
    ``names`` of players and ``team_names`` of teams for the text."""

    league_label: str
    plan: WeeklyPlan | None = None
    our_odds: TeamOdds | None = None
    ideas: tuple[StrategyIdea, ...] = ()
    names: Mapping[int, str] = field(default_factory=dict)
    team_names: Mapping[int, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def targets(self) -> tuple[StrategyIdea, ...]:
        """The eligible ideas, best score first (the order Claude sees and proposals are drafted in)."""
        return tuple(sorted((i for i in self.ideas if i.eligible), key=lambda i: -i.score))

    @property
    def empty(self) -> bool:
        return self.plan is None and self.our_odds is None and not self.targets


# --- the report -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyReport:
    """The weekly note. ``punts`` and ``contest`` are the plan's (never Claude's); ``target_notes`` maps a target's
    ``key`` to Claude's sentence. ``reply`` is the client's reply when a request was answered."""

    league_label: str
    summary: str
    priorities: tuple[str, ...]
    punts: tuple[str, ...]
    contest: tuple[str, ...]
    targets: tuple[StrategyIdea, ...]
    target_notes: Mapping[str, str]
    source: StrategySource
    detail: str | None = None
    reply: WorkerReply[StrategyOutput] | None = field(default=None, compare=False)

    @property
    def used_claude(self) -> bool:
        return self.source == "claude"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _team(inputs: StrategyInputs, team_id: int) -> str:
    return inputs.team_names.get(team_id, f"team {team_id}")


def _players(inputs: StrategyInputs, ids: Sequence[int]) -> str:
    return ", ".join(inputs.names.get(i, str(i)) for i in ids) or "nothing"


def idea_line(inputs: StrategyInputs, idea: StrategyIdea) -> str:
    ev = idea.evaluation
    ours = ev.ours
    text = (
        f"give {_players(inputs, ev.spec.give)} to {_team(inputs, ev.spec.other_team_id)} for "
        f"{_players(inputs, ev.spec.get)}: our rest-of-season value {ours.delta_ros:+.1f} {ev.unit}"
    )
    if ours.delta_title is not None:
        text += f", title odds {ours.delta_title:+.1%}"
    return f"{text}; P(accept) {ev.acceptance.p_accept:.0%}"


def strategy_text(inputs: StrategyInputs) -> str:
    """The user block: the plan by category, the odds, and the numbered trade ideas."""
    lines = [f"League: {inputs.league_label}"]
    plan = inputs.plan
    if plan is None:
        lines.append("Matchup plan: none (no category matchup to plan)")
    else:
        lines.append(
            f"Matchup plan: expected category wins {plan.expected_wins:.1f} of {len(plan.categories)}"
            + (
                f", P(win matchup) {plan.matchup_win_probability:.0%}"
                if plan.matchup_win_probability is not None
                else ""
            )
        )
        for entry in plan.categories:
            lines.append(
                f"- {entry.category}: {entry.status.value}, P(win) {entry.win_probability:.0%}, "
                f"gap {entry.gap:+.1f}, swing weight {entry.relative_weight:.2f}"
            )
    odds = inputs.our_odds
    if odds is None:
        lines.append("Season odds: none")
    else:
        lines.append(
            f"Season odds: playoffs {_pct(odds.playoffs)}, bye {_pct(odds.bye)}, title {_pct(odds.title)}, "
            f"expected wins {odds.expected_wins:.1f}, P(win this week) {_pct(odds.win_week)}"
        )
    targets = inputs.targets[:MAX_IDEAS]
    if targets:
        lines.append("Trade ideas:")
        lines.extend(f"{n}. {idea_line(inputs, idea)}" for n, idea in enumerate(targets, start=1))
    else:
        lines.append("Trade ideas: none")
    lines.extend(f"Warning: {w}" for w in inputs.warnings)
    return "\n".join(lines)


def template_report(inputs: StrategyInputs) -> StrategyReport:
    """The report without Claude: the plan's statuses, the odds and the targets, stated as they are."""
    plan = inputs.plan
    contest = tuple(e.category for e in plan.contested) if plan is not None else ()
    punts = plan.punts if plan is not None else ()
    parts: list[str] = []
    priorities: list[str] = []
    if plan is not None:
        parts.append(f"Expected category wins this week: {plan.expected_wins:.1f} of {len(plan.categories)}.")
        if plan.matchup_win_probability is not None:
            parts[-1] = parts[-1][:-1] + f" (P(win matchup) {plan.matchup_win_probability:.0%})."
        for entry in plan.contested[:MAX_PRIORITIES]:
            priorities.append(
                f"Chase {entry.category}: P(win) {entry.win_probability:.0%}, gap {entry.gap:+.1f}, "
                f"swing weight {entry.relative_weight:.2f}."
            )
        if punts:
            priorities.append(f"Punt {', '.join(punts)}: win chance below {plan.punt_below:.0%} and not worth chasing.")
        if plan.safe:
            priorities.append(f"Safe: {', '.join(plan.safe)}; no need to spend moves there.")
    odds = inputs.our_odds
    if odds is not None:
        parts.append(f"Season odds: playoffs {_pct(odds.playoffs)}, bye {_pct(odds.bye)}, title {_pct(odds.title)}.")
    targets = inputs.targets
    if targets:
        priorities.append(f"Trade target: {idea_line(inputs, targets[0])}.")
    if not parts:
        parts.append("Nothing to plan this week: no category matchup and no season odds.")
    return StrategyReport(
        inputs.league_label,
        " ".join(parts),
        tuple(priorities[:MAX_PRIORITIES]),
        punts,
        contest,
        targets,
        {},
        "template",
    )


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


# --- proposals --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyProposal:
    """One target's fate: the stored (or already open) ``proposal``, or the ``blocked`` reason; with ``dry`` nothing
    was stored and ``blocked`` is what policy would say."""

    idea: StrategyIdea
    proposal: ProposalRow | None = None
    blocked: str | None = None
    existing: bool = False
    dry: bool = False


def propose_targets(
    store: Store,
    config: Config,
    league: LeagueRow,
    ideas: Sequence[StrategyIdea],
    *,
    max_offers: int = 3,
    weekly_cap: int = WEEKLY_OFFERS,
    dry_run: bool = False,
    now: datetime | None = None,
) -> list[StrategyProposal]:
    """Draft ``trade_propose`` proposals for the best of ``ideas`` through :func:`fm.proposals.propose`, capped at
    ``approve`` and under the trade finder's etiquette (module docs). A policy refusal is reported, never raised."""
    at = now if now is not None else utc_now()
    etiquette = trade_etiquette(store, league, at=at, weekly_cap=weekly_cap)
    open_teams, open_keys = etiquette.open_teams, etiquette.open_keys
    room = min(max_offers, max(0, weekly_cap - etiquette.recent))
    outcomes: list[StrategyProposal] = []
    fresh = 0
    for idea in sorted(ideas, key=lambda i: -i.score):
        spec = idea.evaluation.spec
        if not idea.eligible:
            ev = idea.evaluation
            reason = (
                "not legal: " + "; ".join(ev.legality.problems)
                if not ev.legal
                else "needs a drop on our side, which a trade payload cannot carry"
                if ev.legality.drops_ours
                else "it does not help us"
            )
            outcomes.append(StrategyProposal(idea, blocked=reason, dry=dry_run))
            continue
        if idea.key in open_keys:
            outcomes.append(StrategyProposal(idea, open_keys[idea.key], existing=True, dry=dry_run))
            continue
        if spec.other_team_id in open_teams:
            outcomes.append(
                StrategyProposal(
                    idea, blocked=f"one offer a team: team {spec.other_team_id} already has one", dry=dry_run
                )
            )
            continue
        if fresh >= room:
            outcomes.append(StrategyProposal(idea, blocked=f"already at {room} offers for now", dry=dry_run))
            continue
        payload = spec.payload()
        if dry_run:
            verdict = evaluate(
                store,
                config,
                league,
                ProposalKind.TRADE_PROPOSE,
                payload,
                scoring_period_id=idea.scoring_period_id,
                max_setting="approve",
                now=at,
            )
            blocked = None if verdict.allowed else "; ".join(verdict.reasons)
            outcomes.append(StrategyProposal(idea, blocked=blocked, dry=True))
            if verdict.allowed:
                fresh += 1
                open_teams.add(spec.other_team_id)
            continue
        try:
            row = propose(
                store,
                config,
                league,
                ProposalKind.TRADE_PROPOSE,
                payload,
                created_by=STRATEGIST_CREATED_BY,
                scoring_period_id=idea.scoring_period_id,
                engine_numbers=idea.numbers,
                rationale=idea.rationale,
                dedupe_key=idea.key,
                max_setting="approve",
                now=at,
            )
        except PolicyError as exc:
            outcomes.append(StrategyProposal(idea, blocked=str(exc)))
            continue
        outcomes.append(StrategyProposal(idea, row))
        fresh += 1
        open_teams.add(spec.other_team_id)
    return outcomes


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """The report and what became of its targets."""

    report: StrategyReport
    proposals: tuple[StrategyProposal, ...]


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
) -> StrategyResult:
    """Write the weekly report and draft its targets as proposals (module docs). The report's ``targets`` are exactly
    the engine's eligible ideas: Claude's text never changes which are proposed."""
    at = now if now is not None else utc_now()
    report = write_strategy(client, inputs, now=at)
    proposals = propose_targets(store, config, league, report.targets, max_offers=max_offers, dry_run=dry_run, now=at)
    return StrategyResult(report, tuple(proposals))
