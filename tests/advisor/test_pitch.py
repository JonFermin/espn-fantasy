"""The trade_pitch worker over a fake transport: the template, what Claude is sent (and not sent), the guard on the
terms, the fallbacks, and the structural check that the module has no write path. No key, no socket."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

import fm.advisor.pitch as pitch_module
from fm.advisor.client import AdvisorClient, AdvisorError, CallParams, RawReply, TokenUsage, Transport
from fm.advisor.pitch import (
    PITCH_MAX_CHARS,
    PITCH_WORKER,
    PitchError,
    PitchOutput,
    draft_pitch,
    pitch_text,
    template_pitch,
    trade_payload,
)
from fm.advisor.prompts import prompt_text
from fm.config import DEFAULT_MODEL, Llm
from fm.proposals.payloads import TradePayload
from fm.store import LeagueRow, ProposalRow, Store

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
NAMES = {111: "Alpha Back", 222: "Beta Back", 333: "Gamma Tight"}


class FakeTransport(Transport):
    def __init__(self, *replies: RawReply | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[CallParams] = []

    def parse(self, params: CallParams, output: type[BaseModel], *, fallback: bool = False) -> RawReply:
        assert output is PitchOutput
        self.calls.append(params)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def raw(stop_reason: str | None = "end_turn", output: object = None, *, detail: str | None = None) -> RawReply:
    return RawReply("msg_1", DEFAULT_MODEL, stop_reason, TokenUsage(800, 150, 0, 0), output, detail)


def said(text: str) -> RawReply:
    return raw(output=PitchOutput(message=text))


def trade(**overrides: Any) -> ProposalRow:
    fields: dict[str, Any] = {
        "id": 9,
        "league_id": 1,
        "kind": "trade_propose",
        "policy": "approve",
        "payload": {"other_team_id": 3, "give_espn_ids": [111, 222], "get_espn_ids": [333]},
        "scoring_period_id": 5,
        "engine_numbers": {"acceptance": {"p_accept": 0.7}, "ours": {"delta_title": 0.04}},
        "rationale": "Give two backs for a tight end.",
        "deadline": None,
        "created_by": "decide.trades",
        "created_at": NOW,
    }
    return ProposalRow(**(fields | overrides))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        league = opened.leagues.upsert(
            LeagueRow(key="nfl", sport="nfl", espn_league_id=1234567, season=2026, team_id=1, as_of=NOW)
        )
        assert league.id == 1
        yield opened


def make_client(
    store: Store, *replies: RawReply | Exception, budget: float = 2.0
) -> tuple[AdvisorClient, FakeTransport]:
    transport = FakeTransport(*replies)
    return AdvisorClient(store, Llm(daily_budget_usd=budget), transport), transport


GOOD = "Hey Sam, I'd send you Alpha Back and Beta Back for Gamma Tight. You are thin at RB. Open to a counter."


def test_the_template_states_the_terms_exactly() -> None:
    payload = trade_payload(trade())
    text = template_pitch(payload, names=NAMES, their_name="Sam", our_name="Jon")
    assert text.startswith(
        "Hey Sam, would you be open to a trade? I'd send you Alpha Back and Beta Back for Gamma Tight."
    )
    assert text.endswith("- Jon")
    bare = template_pitch(payload)
    assert "player 111 and player 222 for player 333" in bare


def test_claude_is_sent_the_terms_and_none_of_our_odds(store: Store) -> None:
    client, transport = make_client(store, said(GOOD))
    pitch = draft_pitch(
        client,
        trade(),
        names=NAMES,
        their_name="Sam",
        our_name="Jon",
        league_label="ffl",
        facts=("thin at running back",),
        now=NOW,
    )
    assert pitch.source == "claude" and pitch.used_claude and pitch.text == GOOD and pitch.detail is None
    assert pitch.reply is not None and pitch.reply.usage.id is not None
    (params,) = transport.calls
    assert params["output_config"] == {"effort": "medium"}
    assert "tools" not in params
    assert params["system"][0]["text"] == prompt_text(PITCH_WORKER)
    sent = cast(Any, params["messages"][0]["content"])[0]["text"]
    assert "We send them: Alpha Back and Beta Back" in sent and "They send us: Gamma Tight" in sent
    assert "Fact about their side: thin at running back" in sent and "To: Sam" in sent
    assert "0.7" not in sent and "0.04" not in sent and "p_accept" not in sent  # our numbers never reach the pitch
    (usage,) = store.llm_usage.since(datetime(2026, 1, 1, tzinfo=UTC))
    assert usage.worker == "trade_pitch" and usage.league_id == 1


def test_a_pitch_that_drops_a_player_is_not_used(store: Store) -> None:
    client, _ = make_client(store, said("Hey Sam, want Alpha Back for Gamma Tight?"))
    pitch = draft_pitch(client, trade(), names=NAMES, their_name="Sam", now=NOW)
    assert pitch.source == "fallback" and "leaves out Beta Back" in (pitch.detail or "")
    assert "Alpha Back and Beta Back for Gamma Tight" in pitch.text


def test_a_long_message_is_cut_at_a_sentence_end(store: Store) -> None:
    long = "Alpha Back, Beta Back, Gamma Tight. " + "More words here. " * 80
    client, _ = make_client(store, said(long))
    pitch = draft_pitch(client, trade(), names=NAMES, now=NOW)
    assert pitch.source == "claude" and len(pitch.text) <= PITCH_MAX_CHARS and pitch.text.endswith(".")


def test_pitch_text_without_names_uses_ids() -> None:
    text = pitch_text(trade_payload(trade()))
    assert "We send them: player 111 and player 222" in text and "To: team 3" in text


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens", "pause_turn", None])
def test_an_unusable_answer_falls_back_to_the_template(store: Store, stop_reason: str | None) -> None:
    client, _ = make_client(store, raw(stop_reason, PitchOutput(message="not read")))
    pitch = draft_pitch(client, trade(), names=NAMES, now=NOW)
    assert pitch.source == "fallback" and pitch.detail and pitch.text.startswith("Hey, would you be open")
    assert pitch.reply is not None and pitch.reply.output is None


def test_an_empty_answer_an_unreachable_api_and_a_spent_budget_fall_back(store: Store) -> None:
    client, _ = make_client(store, said("  "), AdvisorError("Claude API call failed: down"))
    assert draft_pitch(client, trade(), now=NOW).detail == "empty message"
    assert "Claude API call failed" in (draft_pitch(client, trade(), now=NOW).detail or "")
    broke, transport = make_client(store, said("never asked"), budget=0.0)
    pitch = draft_pitch(broke, trade(), now=NOW)
    assert pitch.source == "fallback" and (pitch.detail or "").startswith("blocked: ") and transport.calls == []


def test_only_a_trade_offer_has_a_pitch(store: Store) -> None:
    client, transport = make_client(store)
    with pytest.raises(PitchError):
        draft_pitch(client, trade(kind="add_drop", payload={"add_espn_id": 1, "drop_espn_id": 2}), now=NOW)
    assert transport.calls == []
    assert isinstance(trade_payload(trade()), TradePayload)


def test_the_module_has_no_write_path() -> None:
    tree = ast.parse(Path(pitch_module.__file__ or "").read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("fm.executor", "fm.browser", "playwright", "fm.proposals.queue")
    assert [name for name in imported if name.startswith(forbidden)] == []
