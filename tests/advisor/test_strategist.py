"""The weekly_strategist worker over a fake transport: the engine's plan and ideas in, the report and approve-capped
trade proposals out. Claude only words the report; it cannot add a punt or a target. No key, no socket."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

import fm.advisor.strategist as strategist_module
from fm.advisor.client import AdvisorClient, AdvisorError, CallParams, RawReply, TokenUsage, Transport
from fm.advisor.prompts import prompt_text
from fm.advisor.strategist import (
    STRATEGIST_CREATED_BY,
    STRATEGIST_WORKER,
    StrategyIdea,
    StrategyInputs,
    StrategyOutput,
    TargetNote,
    engine_plan,
    propose_targets,
    run_strategist,
    strategy_text,
    template_report,
    write_strategy,
)
from fm.config import DEFAULT_MODEL, Config, Llm
from fm.decide.trades import Acceptance, SideImpact, TradeEvaluation, TradeLegality, TradeSpec
from fm.decide.weekly import CategoryPlan, CategoryStatus, WeeklyPlan, WeeklyPlanError
from fm.model.simulate import TeamOdds
from fm.store import LeagueRow, Store

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
NAMES = {10: "Our Guard", 20: "Their Wing", 11: "Our Big", 21: "Their Big"}
TEAMS = {2: "Sharks", 3: "Jets"}


class FakeTransport(Transport):
    def __init__(self, *replies: RawReply | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[CallParams] = []

    def parse(self, params: CallParams, output: type[BaseModel], *, fallback: bool = False) -> RawReply:
        assert output is StrategyOutput
        self.calls.append(params)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def raw(stop_reason: str | None = "end_turn", output: object = None, *, detail: str | None = None) -> RawReply:
    return RawReply("msg_1", DEFAULT_MODEL, stop_reason, TokenUsage(2000, 400, 0, 0), output, detail)


def said(
    summary: str = "A close week.", priorities: list[str] | None = None, notes: list[TargetNote] | None = None
) -> RawReply:
    return raw(output=StrategyOutput(summary=summary, priorities=priorities or [], target_notes=notes or []))


def category(name: str, status: CategoryStatus, p: float, gap: float = 1.0, weight: float = 1.0) -> CategoryPlan:
    return CategoryPlan(name, status, p, 0.0, 1.0, weight, weight, weight, gap, gap / 3)


def make_plan() -> WeeklyPlan:
    return WeeklyPlan(
        categories=(
            category("PTS", CategoryStatus.CONTEST, 0.55, 2.0, 1.4),
            category("REB", CategoryStatus.CONTEST, 0.45, 3.0, 1.1),
            category("BLK", CategoryStatus.PUNT, 0.08, 9.0, 0.0),
            category("STL", CategoryStatus.SAFE, 0.92, -5.0, 0.2),
        ),
        games=12.0,
        expected_wins=2.0,
        weights={"PTS": 1.4},
        relative_weights={"PTS": 1.4},
        punt_below=0.15,
        safe_above=0.85,
        max_punts=1,
        matchup_win_probability=0.51,
    )


ODDS = TeamOdds(1, 0.51, 0.62, 0.2, 0.08, (0.1, 0.1), 7.5, {"PTS": 0.6})


def evaluation(
    team: int = 2,
    give: int = 10,
    get: int = 20,
    score: float | None = 0.03,
    problems: tuple[str, ...] = (),
    drops: tuple[int, ...] = (),
) -> TradeEvaluation:
    spec = TradeSpec(team, (give,), (get,))
    ours = SideImpact(1, "us", spec.give, spec.get, 100.0, 104.0)
    theirs = SideImpact(team, TEAMS.get(team, "them"), spec.get, spec.give, 90.0, 92.0)
    chance = Acceptance(0.7, 120.0, 100.0, 0.1, 0.5)
    return TradeEvaluation(
        spec, ours, theirs, TradeLegality(problems, (), drops), chance, "accept", "good", score, "ros", "G-score"
    )


def idea(team: int = 2, give: int = 10, get: int = 20, score: float | None = 0.03, **kwargs: Any) -> StrategyIdea:
    ev = evaluation(team, give, get, score, **kwargs)
    return StrategyIdea(ev, f"fba:{ev.spec.key}", {"score": score}, f"Give {give} for {get}.", 5)


def inputs(**overrides: Any) -> StrategyInputs:
    fields: dict[str, Any] = {
        "league_label": "fba",
        "plan": make_plan(),
        "our_odds": ODDS,
        "ideas": (idea(2, 10, 20, 0.03), idea(3, 11, 21, 0.01)),
        "names": NAMES,
        "team_names": TEAMS,
    }
    return StrategyInputs(**(fields | overrides))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        league = opened.leagues.upsert(
            LeagueRow(key="fba", sport="nba", espn_league_id=777, season=2026, team_id=1, as_of=NOW)
        )
        assert league.id == 1
        yield opened


def league_row(store: Store) -> LeagueRow:
    row = store.leagues.by_key("fba")
    assert row is not None
    return row


CONFIG = Config.model_validate(
    {"league": [{"key": "fba", "sport": "nba", "espn_league_id": 777, "season": 2026, "team_id": 1}]}
)


def make_client(
    store: Store, *replies: RawReply | Exception, budget: float = 2.0
) -> tuple[AdvisorClient, FakeTransport]:
    transport = FakeTransport(*replies)
    return AdvisorClient(store, Llm(daily_budget_usd=budget), transport), transport


# --- the engine's plan ---


def test_the_template_is_the_plans_statuses_the_odds_and_the_best_target() -> None:
    report = template_report(inputs())
    assert report.source == "template" and report.punts == ("BLK",) and report.contest == ("PTS", "REB")
    assert "Expected category wins this week: 2.0 of 4 (P(win matchup) 51%)." in report.summary
    assert "playoffs 62%, bye 20%, title 8%" in report.summary
    assert report.priorities[0].startswith("Chase PTS") and any(p.startswith("Punt BLK") for p in report.priorities)
    assert any(p.startswith("Trade target: give Our Guard to Sharks for Their Wing") for p in report.priorities)
    assert [i.key for i in report.targets] == [i.key for i in inputs().targets]


def test_targets_are_the_eligible_ideas_best_first() -> None:
    bad = (
        idea(2, 12, 22, 0.5, problems=("roster full",)),
        idea(3, 13, 23, None),
        idea(3, 14, 24, -0.1),
        idea(3, 15, 25, 0.4, drops=(99,)),
    )
    ranked = inputs(ideas=(*bad, idea(2, 10, 20, 0.03), idea(3, 11, 21, 0.09))).targets
    assert [i.evaluation.spec.give for i in ranked] == [(11,), (10,)]


def test_engine_plan_hands_the_simulations_category_odds_to_plan_weekly(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    class Decision:
        plan = make_plan()
        warnings = ("w",)

    def fake(*args: Any, **kwargs: Any) -> Decision:
        seen.update(kwargs)
        return Decision()

    monkeypatch.setattr(strategist_module, "plan_weekly", fake)
    from fm.model.simulate import SeasonOdds

    odds = cast(Any, SeasonOdds(3, cast(Any, None), 100, 1, {1: ODDS}))
    found = engine_plan(store, league_row(store), schedule=cast(Any, object()), now=NOW, odds=odds)
    assert found.plan is not None and found.warnings == ("w",) and seen["simulated"] == {"PTS": 0.6}


def test_a_league_without_category_matchups_is_a_warning_not_an_error(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise WeeklyPlanError("scoring type has no weekly category matchups to plan")

    monkeypatch.setattr(strategist_module, "plan_weekly", refuse)
    found = engine_plan(store, league_row(store), schedule=cast(Any, object()), now=NOW)
    assert found.plan is None and "no weekly category matchups" in found.warnings[0]


# --- the report ---


def test_claude_words_the_report_and_cannot_change_the_punts_or_targets(store: Store) -> None:
    ours = inputs(league_id=league_row(store).id)
    first = ours.targets[0]
    reply = said(
        "We are even this week.",
        ["Win PTS", "Also punt STL", "  "],
        [TargetNote(idea=1, note="Helps PTS."), TargetNote(idea=9, note="Not shown."), TargetNote(idea=1, note="dup")],
    )
    client, transport = make_client(store, reply)
    report = write_strategy(client, ours, now=NOW)
    assert report.source == "claude" and report.summary == "We are even this week."
    assert report.priorities == ("Win PTS", "Also punt STL")  # Claude's words, but...
    assert report.punts == ("BLK",) and report.contest == ("PTS", "REB")  # ...the engine's punts
    assert report.targets == ours.targets and report.target_notes == {first.key: "Helps PTS."}
    (params,) = transport.calls
    assert params["output_config"] == {"effort": "high"} and "tools" not in params
    assert params["system"][0]["text"] == prompt_text(STRATEGIST_WORKER)
    sent = cast(Any, params["messages"][0]["content"])[0]["text"]
    assert "- BLK: punt, P(win) 8%" in sent and "- PTS: contest, P(win) 55%" in sent
    assert "Season odds: playoffs 62%, bye 20%, title 8%" in sent
    assert "1. give Our Guard to Sharks for Their Wing" in sent and "2. give Our Big to Jets for Their Big" in sent
    (usage,) = store.llm_usage.since(datetime(2026, 1, 1, tzinfo=UTC))
    assert usage.worker == "weekly_strategist" and usage.league_id == league_row(store).id


def test_the_bounds_on_claudes_text_are_applied() -> None:
    assert strategy_text(inputs(plan=None, our_odds=None, ideas=())).endswith("Trade ideas: none")


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens", "pause_turn", None])
def test_an_unusable_answer_keeps_the_template(store: Store, stop_reason: str | None) -> None:
    client, _ = make_client(store, raw(stop_reason, StrategyOutput(summary="not read")))
    report = write_strategy(client, inputs(), now=NOW)
    assert report.source == "fallback" and report.detail and report.summary == template_report(inputs()).summary
    assert report.punts == ("BLK",)


def test_an_empty_summary_an_unreachable_api_and_a_spent_budget_fall_back(store: Store) -> None:
    client, _ = make_client(store, said("  "), AdvisorError("Claude API call failed: down"))
    assert write_strategy(client, inputs(), now=NOW).detail == "empty summary"
    assert "Claude API call failed" in (write_strategy(client, inputs(), now=NOW).detail or "")
    broke, transport = make_client(store, said(), budget=0.0)
    report = write_strategy(broke, inputs(), now=NOW)
    assert report.source == "fallback" and (report.detail or "").startswith("blocked: ") and transport.calls == []


def test_with_nothing_to_say_there_is_no_call(store: Store) -> None:
    client, transport = make_client(store)
    report = write_strategy(client, inputs(plan=None, our_odds=None, ideas=()), now=NOW)
    assert report.source == "template" and "Nothing to plan" in report.summary and transport.calls == []


# --- proposals ---


def test_targets_become_approve_capped_trade_proposals(store: Store) -> None:
    league = league_row(store)
    outcomes = propose_targets(store, CONFIG, league, inputs().targets, now=NOW)
    assert [o.blocked for o in outcomes] == [None, None]
    first = outcomes[0].proposal
    assert first is not None and first.kind == "trade_propose" and first.policy == "approve"
    assert first.created_by == STRATEGIST_CREATED_BY and first.status == "proposed"
    assert first.payload == {"other_team_id": 2, "give_espn_ids": [10], "get_espn_ids": [20]}
    assert first.dedupe_key == inputs().targets[0].key and first.engine_numbers == {"score": 0.03}
    assert first.rationale == "Give 10 for 20." and first.scoring_period_id == 5
    assert len(store.proposals.open(league.row_id)) == 2


def test_running_twice_returns_the_open_proposals(store: Store) -> None:
    league = league_row(store)
    once = propose_targets(store, CONFIG, league, inputs().targets, now=NOW)
    again = propose_targets(store, CONFIG, league, inputs().targets, now=NOW)
    assert [o.existing for o in again] == [True, True]
    assert [o.proposal and o.proposal.id for o in again] == [o.proposal and o.proposal.id for o in once]
    assert len(store.proposals.open(league.row_id)) == 2


def test_etiquette_one_offer_a_team_and_a_cap_per_run(store: Store) -> None:
    league = league_row(store)
    same_team = (idea(2, 10, 20, 0.05), idea(2, 11, 21, 0.04), idea(3, 12, 22, 0.03), idea(4, 13, 23, 0.02))
    outcomes = propose_targets(store, CONFIG, league, same_team, max_offers=2, now=NOW)
    stored = [o for o in outcomes if o.proposal is not None]
    assert [o.proposal.payload["other_team_id"] for o in stored if o.proposal] == [2, 3]
    blocked = [o.blocked or "" for o in outcomes if o.proposal is None]
    assert blocked[0].startswith("one offer a team") and blocked[1].startswith("already at")


def test_an_ineligible_idea_is_reported_and_never_stored(store: Store) -> None:
    league = league_row(store)
    outcomes = propose_targets(
        store,
        CONFIG,
        league,
        (idea(2, 10, 20, 0.5, problems=("roster full",)), idea(3, 11, 21, 0.5, drops=(5,))),
        now=NOW,
    )
    assert [o.proposal for o in outcomes] == [None, None]
    assert all(o.blocked for o in outcomes) and store.proposals.open(league.row_id) == []


def test_a_policy_refusal_is_reported_never_raised(store: Store) -> None:
    league = league_row(store)
    unlisted = Config.model_validate(
        {"league": [{"key": "other", "sport": "nba", "espn_league_id": 1, "season": 2026, "team_id": 1}]}
    )
    outcomes = propose_targets(store, unlisted, league, inputs().targets, now=NOW)
    assert all(o.proposal is None and "not in config.toml" in (o.blocked or "") for o in outcomes)
    assert store.proposals.open(league.row_id) == []


def test_a_dry_run_stores_nothing(store: Store) -> None:
    league = league_row(store)
    outcomes = propose_targets(store, CONFIG, league, inputs().targets, dry_run=True, now=NOW)
    assert [(o.dry, o.blocked, o.proposal) for o in outcomes] == [(True, None, None)] * 2
    assert store.proposals.open(league.row_id) == []


def test_run_strategist_drafts_the_engines_targets_whatever_claude_says(store: Store) -> None:
    client, _ = make_client(store, said("Quiet.", [], [TargetNote(idea=2, note="Skip it.")]))
    result = run_strategist(client, store, CONFIG, league_row(store), inputs(), now=NOW)
    assert result.report.source == "claude" and len(result.proposals) == 2
    assert all(p.proposal is not None for p in result.proposals)  # a note cannot veto or add a target

    fallback_client, _ = make_client(store, AdvisorError("down"))
    again = run_strategist(fallback_client, store, CONFIG, league_row(store), inputs(), now=NOW)
    assert again.report.source == "fallback" and all(p.existing for p in again.proposals)


def test_the_module_has_no_write_path() -> None:
    tree = ast.parse(Path(strategist_module.__file__ or "").read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("fm.executor", "fm.browser", "playwright")
    assert [name for name in imported if name.startswith(forbidden)] == []
