"""The explain worker over a fake transport: which moves are trivial (templated, no API call), what Claude is asked
for the rest, and the fallback to the template on a refusal, a cut-off answer, a failed call or a spent budget. No
test holds a key or opens a socket."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from fm.advisor.client import AdvisorClient, AdvisorError, CallParams, RawReply, TokenUsage, Transport
from fm.advisor.explain import (
    EXPLAIN_MAX_SENTENCES,
    EXPLAIN_WORKER,
    TRIVIAL_SWAP_MOVES,
    ExplainOutput,
    clip_sentences,
    explain_proposal,
    explain_proposals,
    is_trivial,
    proposal_text,
    template_rationale,
)
from fm.advisor.prompts import prompt_text
from fm.config import DEFAULT_MODEL, Llm
from fm.store import LeagueRow, ProposalRow, Store

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
DEADLINE = datetime(2026, 10, 8, 0, 20, tzinfo=UTC)
MOVE_A = {"espn_id": 4426348, "from_slot_id": 20, "to_slot_id": 2}
MOVE_B = {"espn_id": 4373626, "from_slot_id": 2, "to_slot_id": 20}
MOVE_C = {"espn_id": 3121422, "from_slot_id": 4, "to_slot_id": 20}
MOVE_D = {"espn_id": 3916387, "from_slot_id": 20, "to_slot_id": 4}


class FakeTransport(Transport):
    def __init__(self, *replies: RawReply | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[CallParams] = []

    def parse(self, params: CallParams, output: type[BaseModel], *, fallback: bool = False) -> RawReply:
        assert output is ExplainOutput
        self.calls.append(params)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def raw(stop_reason: str | None = "end_turn", output: object = None, *, detail: str | None = None) -> RawReply:
    return RawReply("msg_1", DEFAULT_MODEL, stop_reason, TokenUsage(1000, 200, 0, 0), output, detail)


def said(text: str, **kwargs: Any) -> RawReply:
    return raw(output=ExplainOutput(rationale=text), **kwargs)


def proposal(kind: str, payload: dict[str, Any], **overrides: Any) -> ProposalRow:
    fields: dict[str, Any] = {
        "id": 7,
        "league_id": 1,
        "kind": kind,
        "policy": "approve",
        "payload": payload,
        "scoring_period_id": 5,
        "engine_numbers": {"gain": 2.4, "expected_points": 118.2},
        "rationale": None,
        "deadline": DEADLINE,
        "created_by": "test",
        "created_at": NOW,
    }
    return ProposalRow(**(fields | overrides))


def add_drop(**overrides: Any) -> ProposalRow:
    return proposal("add_drop", {"add_espn_id": 4567890, "drop_espn_id": 3916387}, **overrides)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        league = opened.leagues.upsert(
            LeagueRow(key="nfl", sport="nfl", espn_league_id=1234567, season=2026, team_id=1, as_of=NOW)
        )
        assert league.id == 1  # the proposals below belong to league 1
        yield opened


def make_client(
    store: Store, *replies: RawReply | Exception, budget: float = 2.0
) -> tuple[AdvisorClient, FakeTransport]:
    transport = FakeTransport(*replies)
    return AdvisorClient(store, Llm(daily_budget_usd=budget), transport), transport


# --- trivial or not ---


@pytest.mark.parametrize(
    ("row", "trivial"),
    [
        (proposal("bench_inactive", {"moves": [MOVE_A, MOVE_B]}), True),
        (proposal("bench_inactive", {"moves": [MOVE_A, MOVE_B, MOVE_C, MOVE_D]}), True),
        (proposal("waiver_cancel", {"espn_transaction_id": "abc"}), True),
        (proposal("trade_cancel", {"espn_transaction_id": "abc"}), True),
        (proposal("lineup", {"moves": [MOVE_A, MOVE_B]}), True),  # a single swap
        (proposal("lineup", {"moves": [MOVE_A, MOVE_B, MOVE_C, MOVE_D]}), False),
        (add_drop(), False),
        (proposal("waiver", {"add_espn_id": 1, "bid_amount": 7}), False),
        (proposal("trade_propose", {"other_team_id": 3, "give_espn_ids": [1], "get_espn_ids": [2]}), False),
        (proposal("trade_accept", {"other_team_id": 3, "espn_transaction_id": "x", "give_espn_ids": [1]}), False),
        (proposal("trade_decline", {"other_team_id": 3, "espn_transaction_id": "x", "give_espn_ids": [1]}), False),
        (proposal("something_new", {}), False),  # a kind it does not know is not trivial
    ],
)
def test_which_moves_are_trivial(row: ProposalRow, trivial: bool) -> None:
    assert is_trivial(row) is trivial


def test_a_single_swap_is_two_moves() -> None:
    assert TRIVIAL_SWAP_MOVES == 2


def test_a_trivial_move_is_templated_with_no_api_call(store: Store) -> None:
    client, transport = make_client(store)
    row = proposal(
        "bench_inactive",
        {"moves": [MOVE_A, MOVE_B]},
        rationale="Bench Player A (no game); start Player B: 11.0 expected points (+11.0).",
    )
    explanation = explain_proposal(client, row, now=NOW)
    assert explanation.source == "template" and not explanation.used_claude and explanation.reply is None
    assert explanation.text == "Bench Player A (no game); start Player B: 11.0 expected points (+11.0)."
    assert transport.calls == []
    assert store.llm_usage.since(datetime(2026, 1, 1, tzinfo=UTC)) == []


def test_the_template_without_an_engine_rationale_describes_the_payload() -> None:
    row = proposal("waiver_cancel", {"espn_transaction_id": "tx-9"})
    assert template_rationale(row) == "Cancel waiver claim: cancel transaction tx-9."
    named = proposal("bench_inactive", {"moves": [MOVE_A, MOVE_B]})
    assert template_rationale(named, names={4426348: "Player A"}).startswith(
        "Bench inactive starters: Player A (4426348): slot 20 -> 2; "
    )
    unknown = proposal("something_new", {"x": 1})
    assert template_rationale(unknown) == 'Something_new: {"x":1}.'


def test_the_template_names_players_where_given() -> None:
    row = add_drop()
    text = template_rationale(row, names={4567890: "Streamer"})
    assert text == "Free-agent add/drop: add Streamer (4567890), drop 3916387."


# --- the call for a non-trivial proposal ---


def test_a_non_trivial_proposal_asks_claude_with_its_numbers(store: Store) -> None:
    reply = said("Adds Streamer for Player D. The engine projects 2.4 more points. It locks Thursday.")
    client, transport = make_client(store, reply)
    row = add_drop(rationale="Add Streamer, drop Player D: +2.4 points.")
    explanation = explain_proposal(
        client, row, names={4567890: "Streamer", 3916387: "Player D"}, league_label="ffl", now=NOW
    )

    assert explanation.source == "claude" and explanation.used_claude and explanation.detail is None
    assert explanation.text == "Adds Streamer for Player D. The engine projects 2.4 more points. It locks Thursday."
    assert explanation.reply is not None and explanation.reply.usage.id is not None
    (params,) = transport.calls
    assert params["output_config"] == {"effort": "low"}
    assert "tools" not in params  # no web search for an explanation
    assert params["system"][0]["text"] == prompt_text(EXPLAIN_WORKER)
    text = cast(Any, params["messages"][0]["content"])[0]["text"]
    assert "Proposal: free-agent add/drop (add_drop)" in text and "League: ffl" in text
    assert "Deadline: 2026-10-08T00:20:00Z" in text and "Scoring period: 5" in text
    assert "add Streamer (4567890), drop Player D (3916387)" in text
    assert "Engine rationale: Add Streamer, drop Player D: +2.4 points." in text
    assert '"gain":2.4' in text
    (usage,) = store.llm_usage.since(datetime(2026, 1, 1, tzinfo=UTC))
    assert usage.worker == "explain" and usage.league_id == 1


def test_a_long_answer_is_cut_to_four_sentences(store: Store) -> None:
    long = "One is 12.5 points. Two. Three. Four. Five. Six."
    client, _ = make_client(store, said(long))
    explanation = explain_proposal(client, add_drop(), now=NOW)
    assert explanation.source == "claude" and explanation.text == "One is 12.5 points. Two. Three. Four."
    assert clip_sentences(long, 1) == "One is 12.5 points."
    assert EXPLAIN_MAX_SENTENCES == 4


def test_the_numbers_sent_are_bounded() -> None:
    big = add_drop(engine_numbers={"rows": ["x" * 100] * 200})
    assert len(proposal_text(big)) < 4500


# --- fallbacks ---


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens", "pause_turn", None])
def test_an_unusable_answer_falls_back_to_the_template(store: Store, stop_reason: str | None) -> None:
    client, _ = make_client(store, raw(stop_reason, ExplainOutput(rationale="not read")))
    row = add_drop(rationale="Add Streamer, drop Player D: +2.4 points.")
    explanation = explain_proposal(client, row, now=NOW)
    assert explanation.source == "fallback" and explanation.text == "Add Streamer, drop Player D: +2.4 points."
    assert explanation.detail and explanation.reply is not None and explanation.reply.output is None


def test_a_refusal_and_a_cut_off_answer_say_why(store: Store) -> None:
    client, _ = make_client(store, raw("refusal", detail="declined"), raw("max_tokens"))
    assert explain_proposal(client, add_drop(), now=NOW).detail == "declined"
    assert "cut off" in (explain_proposal(client, add_drop(), now=NOW).detail or "")


def test_an_empty_answer_falls_back(store: Store) -> None:
    client, _ = make_client(store, said("   "))
    explanation = explain_proposal(client, add_drop(), now=NOW)
    assert explanation.source == "fallback" and explanation.detail == "empty rationale"
    assert explanation.text.startswith("Free-agent add/drop: add 4567890")


def test_an_unreachable_api_falls_back(store: Store) -> None:
    client, _ = make_client(store, AdvisorError("Claude API call failed: down"))
    explanation = explain_proposal(client, add_drop(), now=NOW)
    assert explanation.source == "fallback" and "Claude API call failed" in (explanation.detail or "")


def test_the_budget_cap_blocks_the_call_and_the_template_stands_in(store: Store) -> None:
    client, transport = make_client(store, said("never asked"), budget=0.0)
    explanation = explain_proposal(client, add_drop(), now=NOW)
    assert explanation.source == "fallback" and (explanation.detail or "").startswith("blocked: ")
    assert transport.calls == []


def test_explaining_many_stops_asking_once_the_budget_is_spent(store: Store) -> None:
    client, transport = make_client(store, said("First."), said("never"), said("never"), budget=0.005)
    rows = [
        add_drop(id=1),
        add_drop(id=2),
        proposal("waiver_cancel", {"espn_transaction_id": "t"}, id=3),
        add_drop(id=4),
    ]
    results = explain_proposals(client, rows, now=NOW)

    assert [r.source for r in results] == ["claude", "fallback", "template", "fallback"]
    assert [r.proposal_id for r in results] == [1, 2, 3, 4]
    assert "daily Claude budget" in (results[1].detail or "")
    assert results[3].detail == "blocked: the daily budget is spent"
    assert len(transport.calls) == 1  # the first call spent the cap; no request after it
