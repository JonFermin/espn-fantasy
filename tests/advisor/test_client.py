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
from anthropic.types.beta import BetaMessage
from anthropic.types.messages import MessageBatchIndividualResponse
from pydantic import BaseModel, ValidationError

from fm.advisor.client import (
    EFFORT_BY_WORKER,
    FALLBACK_BETA,
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
    served_by_fallback,
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
        self.fallbacks: list[bool] = []
        self.batches: dict[str, list[BatchRequest]] = {}
        self.status: BatchStatus | None = None
        self.results: list[BatchResult] = []

    def parse(self, params: CallParams, output: type[BaseModel], *, fallback: bool = False) -> RawReply:
        self.calls.append((params, output))
        self.fallbacks.append(fallback)
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
    # (1000 x 4 + 200 x 20 + 100 x 4 x 1.25 + 400 x 4 x 0.05) per million tokens
    assert row.cost_usd == pytest.approx(0.00858)
    assert row.called_at == NOW
    assert "close_call: ok (end_turn), 1,000 in / 200 out, 400 cached, $0.0086" == reply.describe()
    assert transport.fallbacks == [True]  # the refusal fallback is on by default


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
    pricing = Pricing(input_per_mtok=4.0, output_per_mtok=20.0, cache_read_multiplier=0.05)
    usage = TokenUsage(1_000_000, 100_000, 0, 0)
    assert pricing.cost(usage) == pytest.approx(6.0)
    assert pricing.cost(usage, batch=True) == pytest.approx(3.0)
    assert pricing.cost(TokenUsage(0, 0, 1_000_000, 1_000_000)) == pytest.approx(5.0 + 0.2)  # write 1.25x, read $0.20
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
    assert row.cost_usd == pytest.approx(0.008 / 2)
    assert collected.describe() == "batch msgbatch_1: 1 ok, 0 not usable, 2 unanswered, $0.0040"
    assert all("fallbacks" not in r.params and "betas" not in r.params for r in requests)  # Batches reject it


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

    create = parse  # a call with server tools goes through ``create``; the fake answers both the same way


class FakeBeta:
    def __init__(self, outcome: object) -> None:
        self.messages = FakeMessages(outcome)


class FakeSdk:
    """``messages.parse`` and ``beta.messages.parse``, both answering ``outcome``."""

    def __init__(self, outcome: object) -> None:
        self.messages = FakeMessages(outcome)
        self.beta = FakeBeta(outcome)


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
    assert real.refusal_fallback is True


# --- the refusal fallback ---


def test_a_fallback_served_answer_is_recorded_under_the_served_model(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    answer = Answer(verdict="start", score=0.7)
    served = RawReply("msg_fb", "claude-sonnet-5", "end_turn", TokenUsage(1000, 200, 0, 0), answer, fallback=True)
    client, transport = make_client(store, served)
    with caplog.at_level(logging.INFO, logger="fm.advisor.client"):
        reply = client.ask("explain", prompt(), Answer, now=NOW)

    assert reply.ok and reply.output == answer
    assert transport.fallbacks == [True]
    assert reply.usage.model == "claude-sonnet-5" and reply.usage.request_id == "msg_fb"
    assert reply.usage.cost_usd == pytest.approx(pricing_for("claude-sonnet-5").cost(TokenUsage(1000, 200, 0, 0)))
    assert (
        f"explain: {DEFAULT_MODEL} declined; served by the fallback model claude-sonnet-5 (request msg_fb)"
        in caplog.text
    )


def test_a_whole_chain_refusal_is_still_a_refusal(store: Store) -> None:
    client, transport = make_client(store, raw("refusal", detail="general_harms: every hop declined"))
    reply = client.ask("explain", prompt(), Answer, now=NOW)
    assert reply.status == "refusal" and reply.output is None
    assert reply.detail == "general_harms: every hop declined"
    assert transport.fallbacks == [True]


def test_the_fallback_is_a_client_setting(store: Store) -> None:
    transport = FakeTransport(raw(output=Answer(verdict="x", score=0.0)))
    client = AdvisorClient(store, Llm(), transport, refusal_fallback=False)
    assert client.ask("explain", prompt(), Answer, now=NOW).ok
    assert transport.fallbacks == [False]


def test_sdk_transport_asks_the_beta_surface_for_the_fallback() -> None:
    answer = Answer(verdict="sit", score=0.2)
    sdk = FakeSdk(sdk_message("end_turn", answer.model_dump_json(), parsed=answer))
    transport = SdkTransport(cast(anthropic.Anthropic, sdk))

    reply = transport.parse(prompt().params(DEFAULT_MODEL), Answer, fallback=True)

    assert sdk.messages.kwargs == {}  # the plain surface is not used
    kwargs = sdk.beta.messages.kwargs
    assert kwargs["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert kwargs["fallbacks"] == "default"
    assert kwargs["output_format"] is Answer and kwargs["output_config"] == {"effort": "low"}
    assert kwargs["model"] == DEFAULT_MODEL and kwargs["max_tokens"] == 256
    assert reply.output == answer and reply.fallback is False


def test_a_fallback_message_iteration_marks_the_served_model() -> None:
    answer = Answer(verdict="start", score=0.9)
    message = BetaMessage.model_validate(
        {
            "id": "msg_beta",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": answer.model_dump_json()}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 40,
                "output_tokens": 9,
                "iterations": [
                    {
                        "type": "fallback_message",
                        "model": "claude-sonnet-5",
                        "input_tokens": 40,
                        "output_tokens": 9,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    }
                ],
            },
        }
    )
    assert served_by_fallback(message) is True
    reply = reply_from_message(message, Answer)
    assert reply == RawReply("msg_beta", "claude-sonnet-5", "end_turn", TokenUsage(40, 9, 0, 0), answer, fallback=True)
    assert served_by_fallback(sdk_message("end_turn", answer.model_dump_json())) is False


# --- server tools ---

SEARCH_TOOL: dict[str, Any] = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}
PAUSED_BLOCK: dict[str, Any] = {
    "type": "server_tool_use",
    "id": "srvtoolu_1",
    "name": "web_search",
    "input": {"q": "x"},
}


def test_tools_are_in_the_params_only_when_the_prompt_has_them() -> None:
    plain = prompt().params(DEFAULT_MODEL)
    assert "tools" not in plain  # a call without tools is the call it always was
    assert plain == prompt(tools=()).params(DEFAULT_MODEL)
    with_tools = prompt(tools=(SEARCH_TOOL,)).params(DEFAULT_MODEL)
    assert with_tools.get("tools") == [SEARCH_TOOL]
    assert {k: v for k, v in with_tools.items() if k != "tools"} == plain  # the cached prefix is unchanged
    assert prompt(tools=(SEARCH_TOOL,)).params(DEFAULT_MODEL) == with_tools  # stable across calls


@pytest.mark.parametrize("fallback", [False, True])
def test_sdk_transport_sends_the_tools_and_the_schema_on_both_surfaces(fallback: bool) -> None:
    answer = Answer(verdict="sit", score=0.2)
    sdk = FakeSdk(sdk_message("end_turn", answer.model_dump_json()))
    transport = SdkTransport(cast(anthropic.Anthropic, sdk))

    reply = transport.parse(prompt(tools=(SEARCH_TOOL,)).params(DEFAULT_MODEL), Answer, fallback=fallback)

    sent = (sdk.beta if fallback else sdk).messages.kwargs
    other = (sdk if fallback else sdk.beta).messages.kwargs
    assert other == {}
    assert sent["tools"] == [SEARCH_TOOL]
    assert sent["output_config"] == {"effort": "low", "format": output_format(Answer)}
    assert "output_format" not in sent  # create, not parse: the answer is read from the final text block
    assert (sent.get("fallbacks") == "default") is fallback and (sent.get("betas") == [FALLBACK_BETA]) is fallback
    assert reply.output == answer and reply.stop_reason == "end_turn"


def test_a_call_without_tools_still_goes_through_parse() -> None:
    answer = Answer(verdict="sit", score=0.2)
    sdk = FakeSdk(sdk_message("end_turn", answer.model_dump_json(), parsed=answer))
    SdkTransport(cast(anthropic.Anthropic, sdk)).parse(prompt().params(DEFAULT_MODEL), Answer)
    assert "tools" not in sdk.messages.kwargs and sdk.messages.kwargs["output_format"] is Answer


def search_message(content: list[dict[str, Any]], stop_reason: str = "end_turn") -> Message:
    return Message.model_validate(
        {
            "id": "msg_s",
            "type": "message",
            "role": "assistant",
            "model": DEFAULT_MODEL,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 5, "output_tokens": 2},
            "content": content,
        }
    )


def test_search_urls_come_from_the_result_blocks_and_an_error_object_adds_none() -> None:
    answer = Answer(verdict="start", score=0.9)
    result = {"type": "web_search_result", "url": "https://a.test/1", "title": "t", "encrypted_content": "e"}
    ok = {
        "type": "web_search_tool_result",
        "tool_use_id": "s1",
        "content": [result, {**result, "url": "https://a.test/2"}],
    }
    failed = {
        "type": "web_search_tool_result",
        "tool_use_id": "s2",
        "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"},
    }
    text = {"type": "text", "text": answer.model_dump_json()}
    reply = reply_from_message(search_message([PAUSED_BLOCK, ok, failed, text]), Answer)
    assert reply.output == answer and reply.search_urls == ("https://a.test/1", "https://a.test/2")
    assert reply.resume_content == ()  # kept only for a pause_turn
    assert reply_from_message(search_message([failed, text]), Answer).search_urls == ()
    assert reply_from_message(sdk_message("end_turn", answer.model_dump_json()), Answer).search_urls == ()


def test_a_paused_reply_keeps_its_content_as_request_blocks() -> None:
    message = search_message([{"type": "text", "text": "Searching."}, PAUSED_BLOCK], "pause_turn")
    reply = reply_from_message(message, Answer)
    assert reply.stop_reason == "pause_turn" and reply.output is None
    assert reply.resume_content == ({"type": "text", "text": "Searching."}, PAUSED_BLOCK)


def paused(content: tuple[dict[str, Any], ...] = (PAUSED_BLOCK,), urls: tuple[str, ...] = ()) -> RawReply:
    return RawReply("msg_p", DEFAULT_MODEL, "pause_turn", TokenUsage(1000, 200, 0, 0), None, None, False, urls, content)


def test_pause_turn_is_resumed_with_the_assistant_content_and_every_round_is_recorded(store: Store) -> None:
    answer = Answer(verdict="start", score=0.9)
    client, transport = make_client(store, paused(urls=("https://a.test/1",)), raw(output=answer))

    reply = client.ask("close_call", prompt(tools=(SEARCH_TOOL,)), Answer, now=NOW)

    assert reply.ok and reply.output == answer and reply.rounds == 2
    first, second = (params for params, _ in transport.calls)
    assert second["messages"] == [*first["messages"], {"role": "assistant", "content": [PAUSED_BLOCK]}]
    assert second.get("tools") == first.get("tools") == [SEARCH_TOOL]
    assert len(first["messages"]) == 1  # the first request was not touched
    assert reply.search_urls == ("https://a.test/1",)
    rows = store.llm_usage.since(NOW - timedelta(days=1))
    assert [row.stop_reason for row in rows] == ["pause_turn", "end_turn"]
    assert reply.usages == tuple(rows) and reply.usage == rows[-1]
    assert reply.cost_usd == pytest.approx(0.016) and client.spent_today(NOW) == pytest.approx(0.016)
    assert "last of 2 rounds" in reply.describe()


def test_resumes_are_capped_and_each_round_still_counts(store: Store) -> None:
    client, transport = make_client(store, paused(), paused(), paused(), paused())
    reply = client.ask("close_call", prompt(tools=(SEARCH_TOOL,)), Answer, now=NOW)
    assert reply.status == "stopped" and reply.output is None and reply.rounds == 3
    assert reply.detail == "still paused (pause_turn) after 2 resumes; not used"
    assert len(transport.calls) == 3 and len(store.llm_usage.since(NOW - timedelta(days=1))) == 3

    single = AdvisorClient(store, Llm(), FakeTransport(paused(), paused()), max_resumes=0)
    assert single.ask("close_call", prompt(), Answer, now=NOW).detail == (
        "still paused (pause_turn) after 0 resumes; not used"
    )
    with pytest.raises(ValueError, match="max_resumes"):
        AdvisorClient(store, Llm(), FakeTransport(), max_resumes=-1)


def test_the_budget_is_checked_before_every_resume(store: Store) -> None:
    client, transport = make_client(store, paused(), raw(output=Answer(verdict="x", score=0.0)), budget=0.005)
    reply = client.ask("close_call", prompt(tools=(SEARCH_TOOL,)), Answer, now=NOW)  # round 1 costs $0.008
    assert reply.status == "stopped" and reply.output is None and reply.rounds == 1
    assert reply.detail is not None and "not resumed" in reply.detail and "daily Claude budget reached" in reply.detail
    assert len(transport.calls) == 1


def test_a_pause_with_nothing_to_resume_from_is_not_resumed(store: Store) -> None:
    client, transport = make_client(store, paused(content=()), raw(output=Answer(verdict="x", score=0.0)))
    reply = client.ask("close_call", prompt(), Answer, now=NOW)
    assert reply.status == "stopped" and len(transport.calls) == 1


def test_a_batch_request_carries_the_prompts_tools(store: Store) -> None:
    client, transport = make_client(store)
    client.submit_batch("close_call", {"a": prompt(tools=(SEARCH_TOOL,)), "b": prompt()}, Answer, now=NOW)
    a, b = transport.batches["msgbatch_1"]
    assert a.params.get("tools") == [SEARCH_TOOL] and "tools" not in b.params
