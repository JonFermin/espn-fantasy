"""The advisor client over a fake transport: statuses from ``stop_reason``, usage rows, the budget cap, batches, and
the SDK transport's conversion of real SDK messages. No test holds a key or opens a socket."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import anthropic
import httpx
import pytest
from anthropic.types import Message, ParsedMessage, ParsedTextBlock, TextBlock, Usage
from anthropic.types.messages import MessageBatchIndividualResponse
from pydantic import BaseModel, ValidationError

from fm.advisor.client import (
    EFFORT_BY_WORKER,
    AdvisorClient,
    AdvisorError,
    BatchRequest,
    BatchResult,
    BatchStatus,
    BudgetExceededError,
    CallParams,
    Pricing,
    Prompt,
    RawReply,
    SdkTransport,
    TokenUsage,
    Transport,
    day_start,
    effort_for,
    output_format,
    pricing_for,
    reply_from_message,
    spend_report,
)
from fm.config import DEFAULT_MODEL, Config, Llm
from fm.store import LeagueRow, LlmUsageRow, Store

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)


class Answer(BaseModel):
    verdict: str
    score: float


class FakeTransport(Transport):
    """Scripted replies, in order; records every call."""

    def __init__(self, *replies: RawReply) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[CallParams, type[BaseModel]]] = []
        self.batches: dict[str, list[BatchRequest]] = {}
        self.status: BatchStatus | None = None
        self.results: list[BatchResult] = []

    def parse(self, params: CallParams, output: type[BaseModel]) -> RawReply:
        self.calls.append((params, output))
        return self.replies.pop(0)

    def create_batch(self, requests: Sequence[BatchRequest]) -> str:
        batch_id = f"msgbatch_{len(self.batches) + 1}"
        self.batches[batch_id] = list(requests)
        return batch_id

    def batch_status(self, batch_id: str) -> BatchStatus:
        assert self.status is not None, "set FakeTransport.status first"
        return self.status

    def batch_results(self, batch_id: str, output: type[BaseModel]) -> list[BatchResult]:
        return list(self.results)


def raw(
    stop_reason: str | None = "end_turn",
    output: object = None,
    *,
    tokens: tuple[int, int, int, int] = (1000, 200, 0, 0),
    detail: str | None = None,
    request_id: str | None = "msg_1",
) -> RawReply:
    return RawReply(request_id, DEFAULT_MODEL, stop_reason, TokenUsage(*tokens), output, detail)


def prompt(**overrides: Any) -> Prompt:
    fields: dict[str, Any] = {
        "system": "You judge.",
        "user": "Which?",
        "max_tokens": 256,
        "effort": "low",
        "context": "League nfl.",
        "league_id": None,
    }
    return Prompt(**(fields | overrides))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def make_client(store: Store, *replies: RawReply, budget: float = 2.0) -> tuple[AdvisorClient, FakeTransport]:
    transport = FakeTransport(*replies)
    return AdvisorClient(store, Llm(daily_budget_usd=budget), transport), transport


# --- one call ---


def test_an_end_turn_answer_is_parsed_and_recorded(store: Store) -> None:
    answer = Answer(verdict="start", score=0.8)
    client, transport = make_client(store, raw(output=answer, tokens=(1000, 200, 100, 400)))

    reply = client.ask("close_call", prompt(league_id=None), Answer, now=NOW)

    assert reply.ok and reply.status == "ok"
    assert reply.output == answer
    assert reply.stop_reason == "end_turn"
    (row,) = store.llm_usage.since(NOW - timedelta(minutes=1))
    assert row == reply.usage
    assert (row.worker, row.model, row.batch, row.stop_reason, row.request_id) == (
        "close_call",
        DEFAULT_MODEL,
        False,
        "end_turn",
        "msg_1",
    )
    assert (row.input_tokens, row.output_tokens, row.cache_creation_input_tokens, row.cache_read_input_tokens) == (
        1000,
        200,
        100,
        400,
    )
    # (1000 x 5 + 200 x 25 + 100 x 5 x 1.25 + 400 x 5 x 0.1) per million tokens
    assert row.cost_usd == pytest.approx(0.010825)
    assert row.called_at == NOW
    assert "close_call: ok (end_turn), 1,000 in / 200 out, 400 cached, $0.0108" == reply.describe()


def test_the_request_carries_the_cached_prefix_the_effort_and_the_schema(store: Store) -> None:
    client, transport = make_client(store, raw(output=Answer(verdict="x", score=0.0)))
    client.ask("explain", prompt(effort="medium"), Answer, now=NOW)

    ((params, output),) = transport.calls
    assert output is Answer
    assert params["model"] == DEFAULT_MODEL
    assert params["max_tokens"] == 256
    assert params["output_config"] == {"effort": "medium"}
    assert params["system"] == [
        {"type": "text", "text": "You judge.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "League nfl.", "cache_control": {"type": "ephemeral"}},
    ]
    assert params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Which?"}]}]
    assert prompt(context=None).params(DEFAULT_MODEL)["system"] == [
        {"type": "text", "text": "You judge.", "cache_control": {"type": "ephemeral"}}
    ]


def test_a_refusal_has_no_output_and_is_logged_and_recorded(store: Store, caplog: pytest.LogCaptureFixture) -> None:
    client, _ = make_client(store, raw("refusal", Answer(verdict="x", score=1.0), detail="general_harms: no"))
    with caplog.at_level(logging.WARNING, logger="fm.advisor.client"):
        reply = client.ask("explain", prompt(), Answer, now=NOW)

    assert reply.status == "refusal" and not reply.ok
    assert reply.output is None  # whatever the SDK attached
    assert reply.detail == "general_harms: no"
    assert reply.usage.stop_reason == "refusal" and reply.usage.id is not None
    assert "explain: refusal (refusal)" in caplog.text and "general_harms: no" in caplog.text


def test_max_tokens_is_a_failure_not_partial_data(store: Store) -> None:
    client, _ = make_client(store, raw("max_tokens", Answer(verdict="partial", score=0.5)))
    reply = client.ask("explain", prompt(), Answer, now=NOW)
    assert reply.status == "max_tokens"
    assert reply.output is None
    assert reply.detail is not None and "cut off" in reply.detail
    assert reply.usage.stop_reason == "max_tokens"


@pytest.mark.parametrize(
    ("reply", "status"),
    [
        (raw("end_turn", None), "unparseable"),
        (raw("end_turn", "not an Answer"), "unparseable"),
        (raw(None, None, detail="SDK could not parse", request_id=None), "unparseable"),
        (raw("tool_use", Answer(verdict="x", score=0.0)), "stopped"),
        (raw("pause_turn", None), "stopped"),
        (raw("model_context_window_exceeded", None), "stopped"),
    ],
)
def test_other_outcomes_never_yield_an_output(store: Store, reply: RawReply, status: str) -> None:
    client, _ = make_client(store, reply)
    got = client.ask("explain", prompt(), Answer, now=NOW)
    assert got.status == status
    assert got.output is None
    assert got.detail
    assert store.llm_usage.since(NOW - timedelta(days=1))[0].stop_reason == reply.stop_reason


# --- budget ---


def spent(store: Store, cost: float, at: datetime, worker: str = "news_triage") -> LlmUsageRow:
    return store.llm_usage.insert(LlmUsageRow(called_at=at, worker=worker, model=DEFAULT_MODEL, cost_usd=cost))


def test_the_daily_budget_cap_blocks_calls_before_they_are_sent(store: Store) -> None:
    client, transport = make_client(store, raw(output=Answer(verdict="x", score=0.0)), budget=0.05)
    spent(store, 0.03, NOW - timedelta(hours=2))
    spent(store, 0.02, NOW - timedelta(hours=1), worker="explain")

    with pytest.raises(BudgetExceededError, match=r"\$0\.05 of \$0\.05"):
        client.ask("explain", prompt(), Answer, now=NOW)
    assert transport.calls == []
    assert client.spent_today(NOW) == pytest.approx(0.05)
    assert client.budget_left(NOW) == 0.0


def test_yesterdays_spend_does_not_count(store: Store) -> None:
    client, transport = make_client(store, raw(output=Answer(verdict="x", score=0.0)), budget=0.05)
    spent(store, 0.05, day_start(NOW) - timedelta(minutes=1))
    spent(store, 0.04, day_start(NOW))

    reply = client.ask("explain", prompt(), Answer, now=NOW)
    assert reply.ok and len(transport.calls) == 1
    assert client.budget_left(NOW) == pytest.approx(0.05 - 0.04 - reply.usage.cost_usd)


def test_a_zero_budget_blocks_everything(store: Store) -> None:
    client, transport = make_client(store, raw(output=Answer(verdict="x", score=0.0)), budget=0.0)
    with pytest.raises(BudgetExceededError):
        client.ask("explain", prompt(), Answer, now=NOW)
    with pytest.raises(BudgetExceededError):
        client.submit_batch("explain", {"a": prompt()}, Answer, now=NOW)
    assert transport.calls == [] and transport.batches == {}


def test_spend_report_sums_the_day_by_worker(store: Store) -> None:
    client, _ = make_client(store, budget=2.0)
    spent(store, 0.20, NOW - timedelta(hours=3))
    spent(store, 0.11, NOW - timedelta(hours=1))
    spent(store, 0.05, NOW - timedelta(minutes=5), worker="explain")
    spent(store, 9.0, day_start(NOW) - timedelta(seconds=1))

    report = spend_report(client, NOW)
    assert report.calls == 3
    assert report.spent_usd == pytest.approx(0.36)
    assert report.left_usd == pytest.approx(1.64)
    assert dict(report.by_worker) == pytest.approx({"news_triage": 0.31, "explain": 0.05})
    assert report.describe() == (
        "Claude since 2026-10-06 00:00 UTC: 3 calls, $0.36 of $2.00 (explain $0.05, news_triage $0.31)"
    )


# --- pricing, effort, prompts ---


def test_pricing_and_batch_discount(caplog: pytest.LogCaptureFixture) -> None:
    pricing = Pricing(input_per_mtok=5.0, output_per_mtok=25.0)
    usage = TokenUsage(1_000_000, 100_000, 0, 0)
    assert pricing.cost(usage) == pytest.approx(7.5)
    assert pricing.cost(usage, batch=True) == pytest.approx(3.75)
    assert pricing_for(DEFAULT_MODEL) == pricing
    with caplog.at_level(logging.WARNING, logger="fm.advisor.client"):
        assert pricing_for("claude-unlisted") == pricing
    assert "no pricing listed for model 'claude-unlisted'" in caplog.text


def test_effort_per_worker_matches_the_design_table() -> None:
    assert dict(EFFORT_BY_WORKER) == {
        "news_triage": "low",
        "close_call": "medium",
        "explain": "low",
        "trade_pitch": "medium",
        "weekly_strategist": "high",
    }
    assert effort_for("weekly_strategist") == "high"
    with pytest.raises(KeyError, match="no effort set for worker 'lineup'; known: news_triage"):
        effort_for("lineup")


def test_prompts_are_checked() -> None:
    with pytest.raises(ValueError, match="non-empty system and user"):
        prompt(system=" ")
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        prompt(max_tokens=0)


def test_output_format_is_the_transformed_json_schema() -> None:
    schema = output_format(Answer)
    assert schema["type"] == "json_schema"
    assert schema["schema"]["required"] == ["verdict", "score"]
    assert schema["schema"]["additionalProperties"] is False


# --- batches ---


def test_a_batch_is_submitted_with_the_schema_and_collected_at_half_price(store: Store) -> None:
    client, transport = make_client(store, budget=2.0)
    prompts = {"a": prompt(user="A?"), "b": prompt(user="B?"), "c": prompt(user="C?")}

    league = store.leagues.upsert(
        LeagueRow(key="nfl", sport="nfl", espn_league_id=1, season=2026, team_id=1, as_of=NOW)
    )
    submitted = client.submit_batch("news_triage", prompts, Answer, league_id=league.row_id, now=NOW)
    assert submitted.batch_id == "msgbatch_1"
    assert submitted.custom_ids == ("a", "b", "c") and submitted.worker == "news_triage"
    requests = transport.batches["msgbatch_1"]
    assert [r.custom_id for r in requests] == ["a", "b", "c"]
    assert requests[0].params.get("output_config") == {"effort": "low", "format": output_format(Answer)}
    assert requests[1].params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "B?"}]}]
    assert store.llm_usage.since(NOW - timedelta(days=1)) == []  # nothing recorded until collected

    transport.status = BatchStatus("msgbatch_1", "in_progress", processing=3)
    assert client.collect_batch(submitted, Answer, now=NOW) is None

    transport.status = BatchStatus("msgbatch_1", "ended", succeeded=1, errored=1)
    transport.results = [
        BatchResult("a", raw(output=Answer(verdict="a", score=1.0), tokens=(1000, 200, 0, 0), request_id="msg_a")),
        BatchResult("b", None, "errored: invalid request"),
        BatchResult("zzz", raw(output=Answer(verdict="?", score=0.0))),  # not ours
    ]
    collected = client.collect_batch(submitted, Answer, now=NOW)
    assert collected is not None
    assert set(collected.replies) == {"a"}
    assert collected.replies["a"].ok and collected.replies["a"].output == Answer(verdict="a", score=1.0)
    assert dict(collected.failed) == {"b": "errored: invalid request", "c": "missing from the results"}
    (row,) = store.llm_usage.since(NOW - timedelta(days=1))
    assert row.batch is True and row.league_id == league.row_id and row.request_id == "msg_a"
    assert row.cost_usd == pytest.approx(0.01 / 2)
    assert collected.describe() == "batch msgbatch_1: 1 ok, 0 not usable, 2 unanswered, $0.0050"


def test_an_empty_batch_is_refused(store: Store) -> None:
    client, _ = make_client(store)
    with pytest.raises(ValueError, match="at least one prompt"):
        client.submit_batch("news_triage", {}, Answer, now=NOW)


# --- the SDK transport over a fake SDK ---


def sdk_message(stop_reason: str, text: str | None, *, parsed: object = None) -> Message:
    fields: dict[str, Any] = {
        "id": "msg_sdk",
        "type": "message",
        "role": "assistant",
        "model": DEFAULT_MODEL,
        "stop_reason": stop_reason,
        "usage": Usage(input_tokens=12, output_tokens=3, cache_read_input_tokens=8),
    }
    if parsed is None:
        content: list[Any] = [TextBlock(type="text", text=text)] if text is not None else []
        return Message(**fields, content=content)
    block = ParsedTextBlock[Answer](type="text", text=text or "", parsed_output=cast(Answer, parsed))
    return ParsedMessage[Answer](**fields, content=[block])


class FakeMessages:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.kwargs: dict[str, Any] = {}

    def parse(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeSdk:
    def __init__(self, outcome: object) -> None:
        self.messages = FakeMessages(outcome)


def test_sdk_transport_sends_parse_and_reads_the_parsed_message() -> None:
    answer = Answer(verdict="sit", score=0.2)
    sdk = FakeSdk(sdk_message("end_turn", answer.model_dump_json(), parsed=answer))
    transport = SdkTransport(cast(anthropic.Anthropic, sdk))

    reply = transport.parse(prompt().params(DEFAULT_MODEL), Answer)

    assert sdk.messages.kwargs["output_format"] is Answer
    assert sdk.messages.kwargs["output_config"] == {"effort": "low"}
    assert sdk.messages.kwargs["model"] == DEFAULT_MODEL
    assert reply == RawReply("msg_sdk", DEFAULT_MODEL, "end_turn", TokenUsage(12, 3, 0, 8), answer, None)


def test_sdk_transport_turns_a_validation_error_into_an_unparseable_reply() -> None:
    try:
        Answer.model_validate_json('{"verdict": "cut')
    except ValidationError as exc:
        error = exc
    else:  # pragma: no cover
        raise AssertionError("expected a ValidationError")
    transport = SdkTransport(cast(anthropic.Anthropic, FakeSdk(error)))
    reply = transport.parse(prompt().params(DEFAULT_MODEL), Answer)
    assert reply.stop_reason is None and reply.output is None and reply.request_id is None
    assert reply.detail is not None and reply.detail.startswith("SDK could not parse the answer as Answer")


def test_sdk_transport_wraps_api_errors() -> None:
    request = cast(Any, httpx.Request("POST", "https://api.anthropic.com/v1/messages"))  # the SDK vendors httpx
    transport = SdkTransport(cast(anthropic.Anthropic, FakeSdk(anthropic.APIConnectionError(request=request))))
    with pytest.raises(AdvisorError, match="Claude API call failed"):
        transport.parse(prompt().params(DEFAULT_MODEL), Answer)


def test_reply_from_a_plain_message_parses_the_text_only_on_end_turn() -> None:
    answer = Answer(verdict="start", score=0.9)
    assert reply_from_message(sdk_message("end_turn", answer.model_dump_json()), Answer).output == answer
    bad = reply_from_message(sdk_message("end_turn", '{"verdict": 1}'), Answer)
    assert bad.output is None and bad.detail == "output does not match Answer (2 errors)"
    cut = reply_from_message(sdk_message("max_tokens", '{"verdict": "sta'), Answer)
    assert cut.output is None and cut.stop_reason == "max_tokens" and cut.detail is None
    refused = Message.model_validate(
        sdk_message("refusal", None).model_dump()
        | {"stop_details": {"type": "refusal", "category": "general_harms", "explanation": "no"}}
    )
    assert reply_from_message(refused, Answer).detail == "general_harms: no"


def test_batch_results_lines_become_batch_results() -> None:
    from fm.advisor.client import _batch_result

    answer = Answer(verdict="a", score=1.0)
    ok = MessageBatchIndividualResponse.model_validate(
        {
            "custom_id": "a",
            "result": {"type": "succeeded", "message": sdk_message("end_turn", answer.model_dump_json()).model_dump()},
        }
    )
    assert _batch_result(ok, Answer) == BatchResult(
        "a", RawReply("msg_sdk", DEFAULT_MODEL, "end_turn", TokenUsage(12, 3, 0, 8), answer)
    )
    expired = MessageBatchIndividualResponse.model_validate({"custom_id": "b", "result": {"type": "expired"}})
    assert _batch_result(expired, Answer) == BatchResult("b", None, "expired")
    errored = MessageBatchIndividualResponse.model_validate(
        {
            "custom_id": "c",
            "result": {
                "type": "errored",
                "error": {"type": "error", "error": {"type": "invalid_request_error", "message": "too long"}},
            },
        }
    )
    assert _batch_result(errored, Answer) == BatchResult("c", None, "errored: too long")


# --- configuration ---


def config(**secrets: str) -> Config:
    return Config.model_validate(
        {
            "league": [{"key": "nfl", "sport": "nfl", "espn_league_id": 1, "season": 2026, "team_id": 1}],
            "llm": {"daily_budget_usd": 1.5},
            "secrets": secrets,
        }
    )


def test_from_config_needs_a_key_unless_a_transport_is_given(store: Store) -> None:
    with pytest.raises(AdvisorError, match="ANTHROPIC_API_KEY is not set"):
        AdvisorClient.from_config(store, config())
    client = AdvisorClient.from_config(store, config(), transport=FakeTransport())
    assert client.model == DEFAULT_MODEL and client.daily_budget_usd == 1.5
    with pytest.raises(AdvisorError, match="blank"):
        SdkTransport.with_key(" ")
    real = AdvisorClient.from_config(store, config(ANTHROPIC_API_KEY="sk-ant-test"))
    assert isinstance(real._transport, SdkTransport)  # built, never called
