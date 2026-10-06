"""The close-call worker over a fake transport: the allowlisted web search tool in the call, the rules that bound the
answer (near-ties only, options only, cited and confident or ignored), and what happens on a refusal, a cut-off
answer, a failed call and the budget cap. No test holds a key or opens a socket."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import anthropic
import pytest
from anthropic.types import Message
from pydantic import BaseModel

from fm.advisor.client import (
    AdvisorClient,
    AdvisorError,
    CallParams,
    RawReply,
    SdkTransport,
    TokenUsage,
    Transport,
)
from fm.advisor.close_call import (
    CLOSE_CALL_MAX_MARGIN,
    CLOSE_CALL_MAX_MARGIN_FRACTION,
    CLOSE_CALL_MAX_SEARCHES,
    CLOSE_CALL_MIN_CONFIDENCE,
    CLOSE_CALL_WORKER,
    DEFAULT_ALLOWED_DOMAINS,
    WEB_SEARCH_TOOL,
    CloseCall,
    CloseCallFinding,
    CloseCallOption,
    CloseCallOutput,
    close_call_prompt,
    decide_close_call,
    domain_allowed,
    max_margin,
    near_ties,
    normalize_domains,
    normalize_url,
    option_from_candidate,
    result_numbers,
)
from fm.advisor.prompts import prompt_text
from fm.config import DEFAULT_MODEL, Llm
from fm.decide.lineup import LineupCandidate
from fm.store import Store

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
DEADLINE = NOW + timedelta(hours=20)
A, B, C = 4426348, 4373626, 3121422
FREE_AGENT = 3999999
ROSTER = frozenset({A, B, C})
"""Our current roster: the only players a close call may name."""
URL = "https://www.espn.com/nfl/story/_/id/46000001"
ROTO = "https://www.rotowire.com/football/news/46000002"


class FakeTransport(Transport):
    """Scripted replies, in order; records every call."""

    def __init__(self, *replies: RawReply | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[CallParams] = []
        self.fallbacks: list[bool] = []

    def parse(self, params: CallParams, output: type[BaseModel], *, fallback: bool = False) -> RawReply:
        assert output is CloseCallOutput
        self.calls.append(params)
        self.fallbacks.append(fallback)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


RETRIEVED = (URL, ROTO)
"""What the fake search "returned": the pages the default findings cite."""


def raw(
    stop_reason: str | None = "end_turn",
    output: object = None,
    *,
    detail: str | None = None,
    search_urls: tuple[str, ...] = RETRIEVED,
    resume_content: tuple[dict[str, Any], ...] = (),
) -> RawReply:
    return RawReply(
        "msg_1",
        DEFAULT_MODEL,
        stop_reason,
        TokenUsage(1000, 200, 0, 0),
        output,
        detail,
        search_urls=search_urls,
        resume_content=resume_content,
    )


def finding(espn_id: int = A, **overrides: Any) -> CloseCallFinding:
    fields: dict[str, Any] = {
        "espn_id": espn_id,
        "claim": "The coach said he will lead the backfield on Sunday.",
        "source_url": URL,
        "source_title": "Coach: he is the lead back",
        "published_at": "2026-10-06T09:30:00Z",
    }
    return CloseCallFinding(**(fields | overrides))


def output(
    pick: int | None = A, *, confidence: float = 0.8, findings: list[CloseCallFinding] | None = None, **kw: Any
) -> CloseCallOutput:
    fields: dict[str, Any] = {
        "pick_espn_id": pick,
        "confidence": confidence,
        "rationale": "Fresh coaching comments favor the pick.",
        "findings": [finding()] if findings is None else findings,
    }
    return CloseCallOutput(**(fields | kw))


def call(**overrides: Any) -> CloseCall:
    fields: dict[str, Any] = {
        "sport": "nfl",
        "league_label": "ffl",
        "question": "Who starts at FLEX in week 5?",
        "options": (
            CloseCallOption(A, "Player A", 11.0, position="RB", pro_team="SEA", opponent="ARI", p_active=0.95),
            CloseCallOption(
                B, "Player B", 10.8, position="WR", pro_team="DEN", opponent="NYJ", designation="QUESTIONABLE"
            ),
        ),
        "margin": 0.5,
        "roster_ids": ROSTER,
        "deadline": DEADLINE,
        "league_id": None,
    }
    return CloseCall(**(fields | overrides))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def make_client(
    store: Store, *replies: RawReply | Exception, budget: float = 2.0
) -> tuple[AdvisorClient, FakeTransport]:
    transport = FakeTransport(*replies)
    return AdvisorClient(store, Llm(daily_budget_usd=budget), transport), transport


# --- the web search tool and its allowlist ---


def test_the_search_tool_is_configured_with_the_allowlist(store: Store) -> None:
    client, transport = make_client(store, raw(output=output()))
    decide_close_call(client, call(), allowed_domains=("espn.com", "https://www.RotoWire.com/", "nfl.com"), now=NOW)

    (params,) = transport.calls
    tools = cast(Any, params)["tools"]
    assert tools == [
        {
            "type": WEB_SEARCH_TOOL,
            "name": "web_search",
            "max_uses": CLOSE_CALL_MAX_SEARCHES,
            "allowed_domains": ["espn.com", "rotowire.com", "nfl.com"],
        }
    ]
    assert WEB_SEARCH_TOOL == "web_search_20260209"
    assert params["output_config"] == {"effort": "medium"}  # close_call's effort (DESIGN section 10)
    assert params["model"] == DEFAULT_MODEL
    assert transport.fallbacks == [True]  # the refusal fallback stays on


def test_the_default_allowlist_and_the_search_cap_are_used_unless_given(store: Store) -> None:
    client, transport = make_client(store, raw(output=output()), raw(output=output()))
    decide_close_call(client, call(), now=NOW)
    decide_close_call(client, call(), max_searches=2, now=NOW)

    first, second = (cast(Any, params)["tools"][0] for params in transport.calls)
    assert first["allowed_domains"] == list(DEFAULT_ALLOWED_DOMAINS)
    assert first["max_uses"] == CLOSE_CALL_MAX_SEARCHES and second["max_uses"] == 2


def test_an_empty_allowlist_is_refused_because_it_would_mean_the_whole_web(store: Store) -> None:
    client, transport = make_client(store)
    with pytest.raises(ValueError, match="non-empty domain allowlist"):
        decide_close_call(client, call(), allowed_domains=(), now=NOW)
    with pytest.raises(ValueError, match="not a domain"):
        normalize_domains(["localhost"])
    assert transport.calls == []


def test_the_prompt_is_cached_system_text_and_the_question(store: Store) -> None:
    prompt = close_call_prompt(call(), allowed_domains=("espn.com",), now=NOW)
    params = prompt.params(DEFAULT_MODEL)
    instructions, league = params["system"]
    assert instructions["text"] == prompt_text("close_call")
    assert instructions.get("cache_control") == {"type": "ephemeral"}
    assert "ffl" in league["text"] and "espn.com" in league["text"]
    (message,) = params["messages"]
    text = cast(Any, message["content"])[0]["text"]
    assert "Who starts at FLEX in week 5?" in text
    assert f"{A}: Player A, RB, SEA, vs ARI, engine score 11.00, chance of playing 95%" in text
    assert f"{B}: Player B" in text and "designation QUESTIONABLE" in text
    assert "Deadline: 2026-10-07T11:00:00Z" in text and "Now: 2026-10-06T15:00:00Z" in text
    assert len(prompt.tools) == 1 and prompt.tools[0]["allowed_domains"] == ["espn.com"]


def test_domains_match_the_domain_and_its_subdomains_only() -> None:
    allowed = ("espn.com", "nfl.com")
    assert domain_allowed("https://www.espn.com/nfl/story", allowed)
    assert domain_allowed("http://espn.com/x", allowed)
    assert domain_allowed("https://static.nfl.com/x", allowed)
    assert not domain_allowed("https://notespn.com/x", allowed)
    assert not domain_allowed("https://espn.com.evil.io/x", allowed)
    assert not domain_allowed("https://evil.io/?u=https://espn.com", allowed)
    assert not domain_allowed("ftp://espn.com/x", allowed)
    assert not domain_allowed("espn.com/x", allowed)
    assert not domain_allowed("", allowed)


# --- the real client path: an SDK message, its search results, the judgement ---


def search_message(results: object, text: str, *, stop_reason: str = "end_turn") -> Message:
    """What the API returns for a searching call: a remark, the server tool use, its result, then the answer."""
    return Message.model_validate(
        {
            "id": "msg_sdk",
            "type": "message",
            "role": "assistant",
            "model": DEFAULT_MODEL,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 12, "output_tokens": 3},
            "content": [
                {"type": "text", "text": "Let me check the latest reports."},
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "A snaps"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": results},
                {"type": "text", "text": text},
            ],
        }
    )


def search_results(*urls: str) -> list[dict[str, Any]]:
    return [
        {"type": "web_search_result", "url": url, "title": "t", "encrypted_content": "enc", "page_age": None}
        for url in urls
    ]


class FakeMessages:
    def __init__(self, *messages: Message) -> None:
        self.messages = list(messages)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        return self.messages.pop(0)


class FakeBeta:
    def __init__(self, *messages: Message) -> None:
        self.messages = FakeMessages(*messages)


class FakeSdk:
    def __init__(self, *messages: Message) -> None:
        self.messages = FakeMessages(*messages)
        self.beta = FakeBeta(*messages)


def sdk_client(store: Store, sdk: FakeSdk, *, fallback: bool = False) -> AdvisorClient:
    transport = SdkTransport(cast(anthropic.Anthropic, sdk))
    return AdvisorClient(store, Llm(), transport, refusal_fallback=fallback)


def test_the_sdk_client_sends_the_search_tool_and_reads_the_answer_after_the_search(store: Store) -> None:
    answer = search_message(search_results(URL), output().model_dump_json())
    sdk = FakeSdk(answer)
    result = decide_close_call(sdk_client(store, sdk), call(), now=NOW)

    (sent,) = sdk.messages.calls
    assert sent["tools"] == [
        {
            "type": WEB_SEARCH_TOOL,
            "name": "web_search",
            "max_uses": CLOSE_CALL_MAX_SEARCHES,
            "allowed_domains": list(DEFAULT_ALLOWED_DOMAINS),
        }
    ]
    assert sent["output_config"]["effort"] == "medium" and sent["output_config"]["format"]["type"] == "json_schema"
    assert result.status == "pick" and result.pick == A  # the last text block is the answer, not the remark
    assert result.reply is not None and result.reply.search_urls == (URL,)


def test_the_sdk_client_sends_the_tool_on_the_fallback_surface_too(store: Store) -> None:
    sdk = FakeSdk(search_message(search_results(URL), output().model_dump_json()))
    result = decide_close_call(sdk_client(store, sdk, fallback=True), call(), now=NOW)

    assert result.status == "pick" and sdk.messages.calls == []
    (sent,) = sdk.beta.messages.calls
    assert sent["tools"][0]["name"] == "web_search" and sent["fallbacks"] == "default"


def test_a_web_search_error_object_is_handled_and_nothing_is_cited(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    error = {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}
    sdk = FakeSdk(search_message(error, output().model_dump_json()))
    with caplog.at_level(logging.WARNING):
        result = decide_close_call(sdk_client(store, sdk), call(), now=NOW)

    assert result.reply is not None and result.reply.search_urls == ()
    assert "web search failed: max_uses_exceeded" in caplog.text
    # The model answered anyway, but with no page retrieved nothing it cites can stand.
    assert result.status == "rejected" and result.choice == A and result.sources == ()


# --- citations are required and must have been retrieved ---


def test_a_finding_citing_a_page_the_search_did_not_return_is_dropped(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    # Allowlisted host, plausible path, but the search returned only the Rotowire page.
    client, _ = make_client(store, raw(output=output(A), search_urls=(ROTO,)))
    with caplog.at_level(logging.WARNING, logger="fm.advisor.close_call"):
        result = decide_close_call(client, call(), now=NOW)
    assert result.status == "rejected" and result.choice == A and result.sources == ()
    assert "was not among the pages the search returned" in caplog.text


def test_a_search_with_no_results_leaves_every_citation_unbacked(store: Store) -> None:
    client, _ = make_client(store, raw(output=output(A), search_urls=()))
    assert decide_close_call(client, call(), now=NOW).status == "rejected"


def test_a_retrieved_allowlisted_page_is_accepted_whatever_its_spelling(store: Store) -> None:
    returned = ("HTTPS://WWW.ESPN.COM/nfl/story/_/id/46000001/",)
    cited = finding(A, source_url=URL + "#comments")
    client, _ = make_client(store, raw(output=output(A, findings=[cited]), search_urls=returned))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "pick" and [s.url for s in result.sources] == [URL + "#comments"]


def test_a_retrieved_page_off_the_allowlist_is_still_dropped(store: Store) -> None:
    reddit = "https://www.reddit.com/r/fantasyfootball/x"
    client, _ = make_client(
        store, raw(output=output(B, findings=[finding(B, source_url=reddit)]), search_urls=(reddit,))
    )
    assert decide_close_call(client, call(), now=NOW).status == "rejected"  # both checks must hold


def test_the_query_distinguishes_pages_but_the_fragment_and_slash_do_not() -> None:
    assert normalize_url("HTTPS://Www.ESPN.com/nfl/Story/?id=1#top") == "https://www.espn.com/nfl/Story?id=1"
    assert normalize_url(" https://espn.com/ ") == normalize_url("https://espn.com")
    assert normalize_url("https://espn.com/a") != normalize_url("https://espn.com/A")
    assert normalize_url("https://espn.com/a?id=1") != normalize_url("https://espn.com/a?id=2")
    assert normalize_url("http://[bad") == "http://[bad"  # not a URL: kept, and it will not match a result
    assert not domain_allowed("http://[bad", ("espn.com",))


# --- a paused search loop is resumed ---


PAUSED = ({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "A"}},)


def test_a_paused_search_is_resumed_and_the_urls_of_every_round_count(store: Store) -> None:
    both = [finding(A), finding(B, source_url=ROTO)]
    client, transport = make_client(
        store,
        raw("pause_turn", search_urls=(URL,), resume_content=PAUSED),
        raw(output=output(A, findings=both), search_urls=(ROTO,)),
    )
    result = decide_close_call(client, call(), now=NOW)

    # Both pages were retrieved, in different rounds: both findings stand.
    assert result.status == "pick" and [s.url for s in result.sources] == [URL, ROTO]
    first, second = transport.calls
    assert second["messages"][0] == first["messages"][0] and len(first["messages"]) == 1
    assert second["messages"][1] == {"role": "assistant", "content": list(PAUSED)}  # no "continue" user turn
    assert len(second["messages"]) == 2
    assert first.get("tools") and second.get("tools") == first.get("tools")  # same tools, same cached prefix
    assert second["system"] == first["system"]
    assert result.reply is not None and result.reply.rounds == 2 and result.reply.search_urls == (URL, ROTO)
    rows = store.llm_usage.since(NOW - timedelta(days=1))
    assert [row.stop_reason for row in rows] == ["pause_turn", "end_turn"]  # every round is recorded
    assert result.reply.cost_usd == pytest.approx(sum(row.cost_usd for row in rows))


def test_a_search_still_paused_after_the_resume_cap_is_a_failure(store: Store) -> None:
    paused = raw("pause_turn", output=output(B, findings=[finding(B)]), resume_content=PAUSED)
    client, transport = make_client(store, paused, paused, paused, raw(output=output()))
    result = decide_close_call(client, call(), now=NOW)

    assert result.status == "failed" and result.choice == A and result.pick is None
    assert "still paused" in (result.detail or "") and "2 resumes" in (result.detail or "")
    assert len(transport.calls) == 3  # the first call and two resumes, no more
    assert len(store.llm_usage.since(NOW - timedelta(days=1))) == 3


def test_a_resume_counts_against_the_daily_budget(store: Store) -> None:
    # The first round alone ($0.008) spends the $0.005 cap: the paused answer is not resumed.
    client, transport = make_client(store, raw("pause_turn", resume_content=PAUSED), raw(output=output()), budget=0.005)
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "failed" and "daily Claude budget reached" in (result.detail or "")
    assert len(transport.calls) == 1


def test_a_pause_without_content_to_resume_from_is_a_failure(store: Store) -> None:
    client, transport = make_client(store, raw("pause_turn"), raw(output=output()))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "failed" and len(transport.calls) == 1


# --- citations are required ---


def test_a_cited_pick_is_used_with_its_sources(store: Store) -> None:
    both = [finding(A), finding(B, source_url=ROTO, source_title="B is limited", claim="B was limited in practice.")]
    client, _ = make_client(store, raw(output=output(A, findings=both)))
    result = decide_close_call(client, call(deadline=None), now=NOW)

    assert result.status == "pick" and result.pick == A and result.choice == A and not result.overrides_engine
    assert [(s.url, s.espn_id) for s in result.sources] == [(URL, A), (ROTO, B)]
    assert result.sources[0].published_at == "2026-10-06T09:30:00Z"
    assert result.rationale == "Fresh coaching comments favor the pick."
    assert result.reply is not None and result.reply.usage.id is not None  # recorded in llm_usage
    assert result.describe() == f"close_call: pick {A}, as the engine (0.80), 2 sources"


def test_a_pick_can_overrule_the_engine_between_near_tied_options(store: Store) -> None:
    client, _ = make_client(store, raw(output=output(B, findings=[finding(B, source_url=ROTO)])))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "pick" and result.choice == B and result.engine_pick == A and result.overrides_engine
    assert result.describe().startswith(f"close_call: pick {B} over the engine's {A}")


def test_an_uncited_pick_is_rejected_and_the_engine_choice_stands(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    client, _ = make_client(store, raw(output=output(B, findings=[])))
    with caplog.at_level(logging.WARNING, logger="fm.advisor.close_call"):
        result = decide_close_call(client, call(), now=NOW)
    assert result.status == "rejected" and result.pick is None and result.choice == A
    assert result.detail == "the pick has no valid cited finding; ignored"
    assert "no valid cited finding" in caplog.text


def test_findings_off_the_allowlist_or_off_the_options_do_not_count(store: Store) -> None:
    bad = [
        finding(B, source_url="https://www.reddit.com/r/fantasyfootball/x"),  # not allowlisted
        finding(B, source_url="https://espn.com.evil.io/story"),  # lookalike host
        finding(B, source_url="not a url"),
        finding(B, claim="   "),  # no claim
        finding(C),  # about a player who is not an option
    ]
    client, _ = make_client(store, raw(output=output(B, findings=bad)))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "rejected" and result.choice == A and result.sources == ()


def test_one_valid_finding_among_bad_ones_is_enough(store: Store) -> None:
    findings = [finding(B, source_url="https://www.reddit.com/x"), finding(B, source_url=ROTO)]
    client, _ = make_client(store, raw(output=output(B, findings=findings)))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "pick" and [s.url for s in result.sources] == [ROTO]


def test_duplicate_findings_collapse(store: Store) -> None:
    client, _ = make_client(store, raw(output=output(A, findings=[finding(A), finding(A)])))
    assert len(decide_close_call(client, call(), now=NOW).sources) == 1


# --- the answer is bounded ---


def test_a_pick_outside_the_options_is_rejected(store: Store) -> None:
    client, _ = make_client(store, raw(output=output(C, findings=[finding(A)])))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "rejected" and result.choice == A and result.choice in result.call.option_ids
    assert "not one of the near-tied options" in (result.detail or "")


def test_a_null_pick_keeps_the_engine_choice(store: Store) -> None:
    client, _ = make_client(store, raw(output=output(None, findings=[])))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "no_change" and result.choice == A and result.pick is None


def test_a_low_confidence_pick_keeps_the_engine_choice(store: Store) -> None:
    low = CLOSE_CALL_MIN_CONFIDENCE - 0.01
    client, _ = make_client(store, raw(output=output(B, confidence=low, findings=[finding(B)])))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "no_change" and result.choice == A and result.confidence == pytest.approx(low)


def test_confidence_is_clamped_and_a_nonfinite_one_is_rejected(store: Store) -> None:
    client, _ = make_client(
        store, raw(output=output(A, confidence=7.0)), raw(output=output(A, confidence=float("nan")))
    )
    clamped = decide_close_call(client, call(), now=NOW)
    assert clamped.status == "pick" and clamped.confidence == 1.0
    assert decide_close_call(client, call(), now=NOW).status == "rejected"


def test_a_pick_without_a_rationale_is_rejected(store: Store) -> None:
    client, _ = make_client(store, raw(output=output(B, rationale="  ", findings=[finding(B)])))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "rejected" and result.choice == A


def test_only_a_near_tie_can_be_asked() -> None:
    options = (CloseCallOption(A, "A", 11.0), CloseCallOption(B, "B", 9.0))
    with pytest.raises(ValueError, match="not a close call"):
        call(options=options, margin=0.5)
    with pytest.raises(ValueError, match="at least two"):
        call(options=options[:1])
    with pytest.raises(ValueError, match="distinct"):
        call(options=(options[0], options[0]))
    with pytest.raises(ValueError, match="aware"):
        call(deadline=datetime(2026, 10, 7))
    assert call(options=options, margin=2.0).best.espn_id == A  # an exactly-at-margin pair is a call


def test_the_margin_has_a_ceiling_so_near_does_not_mean_anything_goes() -> None:
    assert max_margin(11.0) == pytest.approx(11.0 * CLOSE_CALL_MAX_MARGIN_FRACTION)  # a fraction of a small score
    assert max_margin(100.0) == CLOSE_CALL_MAX_MARGIN  # a fixed ceiling for a large one
    assert max_margin(0.0) == 0.0 and max_margin(-4.0) == 0.0  # nothing wider than an exact tie
    with pytest.raises(ValueError, match="wider than a near-tie may be"):
        call(margin=1e6)  # the pair is 0.2 apart, but a margin this wide would make any pair a "close call"
    with pytest.raises(ValueError, match="at most 2.20"):
        call(margin=max_margin(11.0) + 0.01)
    assert call(margin=max_margin(11.0)).margin == pytest.approx(2.2)  # the ceiling itself is allowed
    big = (CloseCallOption(A, "A", 80.0), CloseCallOption(B, "B", 79.0))
    with pytest.raises(ValueError, match="at most 3.00"):
        call(options=big, margin=3.5)
    options = [CloseCallOption(A, "A", 11.0), CloseCallOption(B, "B", 10.0)]
    with pytest.raises(ValueError, match="wider than a near-tie may be"):
        near_ties(options, 1e6)


def test_every_option_must_be_on_our_roster() -> None:
    off = (CloseCallOption(A, "A", 11.0), CloseCallOption(FREE_AGENT, "Free Agent", 10.9))
    with pytest.raises(ValueError, match=f"not on our roster: {FREE_AGENT}"):
        call(options=off)
    with pytest.raises(ValueError, match="not on our roster"):
        call(roster_ids=())  # an empty roster has no one to ask about
    assert call(options=off, roster_ids={A, FREE_AGENT}).option_ids == (A, FREE_AGENT)  # once he is rostered
    assert call(roster_ids=[A, B]).roster_ids == frozenset({A, B})  # any collection of ids, kept as a set


def test_near_ties_selects_the_options_within_the_margin() -> None:
    options = [CloseCallOption(i, f"P{i}", score) for i, score in [(1, 8.0), (2, 10.0), (3, 9.7), (4, 9.4)]]
    assert [o.espn_id for o in near_ties(options, 0.6)] == [2, 3, 4]
    assert [o.espn_id for o in near_ties(options, 0.4)] == [2, 3]
    assert near_ties(options, 0.1) == ()  # a lone best is not a call
    assert near_ties([], 1.0) == ()
    with pytest.raises(ValueError, match="margin"):
        near_ties(options, -1.0)


def test_an_option_comes_from_a_lineup_candidate() -> None:
    candidate = LineupCandidate(
        espn_id=A, slot_id=20, points=14.0, p_active=0.5, position="RB", name="Player A", designation="QUESTIONABLE"
    )
    option = option_from_candidate(candidate, pro_team="SEA", opponent="ARI")
    assert (option.espn_id, option.name, option.score, option.p_active) == (A, "Player A", 7.0, 0.5)
    assert (
        option.describe()
        == f"{A}: Player A, RB, SEA, vs ARI, engine score 7.00, chance of playing 50%, designation QUESTIONABLE"
    )


# --- failures fall back to the engine ---


@pytest.mark.parametrize(
    ("stop_reason", "status"),
    [("refusal", "refusal"), ("max_tokens", "max_tokens"), ("pause_turn", "stopped"), (None, "unparseable")],
)
def test_an_unusable_answer_is_a_failure_and_the_engine_choice_stands(
    store: Store, stop_reason: str | None, status: str
) -> None:
    # The structured output is attached to every reply here: it must not be read unless stop_reason is end_turn.
    client, _ = make_client(store, raw(stop_reason, output=output(B, findings=[finding(B)])))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "failed" and result.pick is None and result.choice == A and not result.overrides_engine
    assert result.reply is not None and result.reply.status == status and result.reply.output is None
    assert result.detail


def test_a_refusal_carries_the_models_explanation(store: Store) -> None:
    client, _ = make_client(store, raw("refusal", detail="declined: cannot help"))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "failed" and result.detail == "declined: cannot help"


def test_a_cut_off_answer_is_not_read(store: Store) -> None:
    client, transport = make_client(store, raw("max_tokens"))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "failed" and "cut off" in (result.detail or "")
    assert len(transport.calls) == 1  # no retry: the engine decides


def test_an_unreachable_api_leaves_the_engine_choice(store: Store) -> None:
    client, _ = make_client(store, AdvisorError("Claude API call failed: down"))
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "failed" and result.reply is None and result.choice == A
    assert store.llm_usage.since(NOW - timedelta(days=1)) == []  # nothing recorded for a call with no answer


# --- the budget cap ---


def test_the_budget_cap_blocks_the_call_before_it_is_sent(store: Store) -> None:
    client, transport = make_client(store, raw(output=output()), budget=0.0)
    result = decide_close_call(client, call(), now=NOW)
    assert result.status == "blocked" and result.choice == A and result.reply is None
    assert "daily Claude budget reached" in (result.detail or "")
    assert transport.calls == []


def test_a_call_that_spends_the_budget_blocks_the_next(store: Store) -> None:
    client, transport = make_client(store, raw(output=output()), raw(output=output()), budget=0.005)
    assert decide_close_call(client, call(), now=NOW).status == "pick"  # $0.008 recorded: over the cap
    assert decide_close_call(client, call(), now=NOW).status == "blocked"
    assert len(transport.calls) == 1


# --- the numbers for a proposal ---


def test_the_result_is_json_for_a_proposals_engine_numbers(store: Store) -> None:
    client, _ = make_client(store, raw(output=output(B, findings=[finding(B, source_url=ROTO)])))
    numbers = result_numbers(decide_close_call(client, call(), now=NOW))
    decoded = json.loads(json.dumps(numbers))["close_call"]
    assert decoded["status"] == "pick" and decoded["choice"] == B and decoded["engine_pick"] == A
    assert decoded["margin"] == 0.5 and decoded["options"] == {str(A): 11.0, str(B): 10.8}
    assert decoded["sources"][0]["url"] == ROTO and decoded["sources"][0]["published_at"] == "2026-10-06T09:30:00Z"
    assert CLOSE_CALL_WORKER == "close_call"
