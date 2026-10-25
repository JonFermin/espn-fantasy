"""``fm.advisor.claude_cli``: the advisor over ``claude -p`` on the subscription. No test runs the real ``claude``
(CLAUDE.md): the stream parser reads recorded-shape events, and the transport runs a fake CLI script that checks what
it was given and writes the events a real run would."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from fm.advisor.claude_cli import (
    CLI_SEARCH_TOOL,
    NO_BATCHES,
    ClaudeCliTransport,
    cli_argv,
    cli_env,
    reply_from_stream,
    system_text,
    user_text,
)
from fm.advisor.client import AdvisorClient, AdvisorError, CallParams, Prompt
from fm.advisor.close_call import web_search_tool
from fm.config import DEFAULT_MODEL, Llm
from fm.store import Store


class Answer(BaseModel):
    playing: bool
    source_url: str


QUESTION = "Is the player active tonight?"
SEARCH_ID = "toolu_search_1"
URLS = ("https://www.espn.com/nba/injuries", "https://www.rotowire.com/basketball/news.php?id=1")


def prompt(*, search: bool = False) -> Prompt:
    tools = (web_search_tool(["espn.com", "rotowire.com"], 2),) if search else ()
    return Prompt(
        system="You are a terse analyst.",
        context="League context.",
        user=QUESTION,
        max_tokens=1000,
        effort="low",
        tools=tools,
    )


def params(*, search: bool = False) -> CallParams:
    return prompt(search=search).params(DEFAULT_MODEL)


def result_event(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "session_id": "sess-1",
        "stop_reason": "tool_use",
        "structured_output": {"playing": True, "source_url": URLS[0]},
        "modelUsage": {
            DEFAULT_MODEL: {
                "inputTokens": 120,
                "outputTokens": 40,
                "cacheReadInputTokens": 900,
                "cacheCreationInputTokens": 300,
            }
        },
    }
    event.update(overrides)
    return event


def search_events() -> list[dict[str, Any]]:
    links = json.dumps([{"title": "a", "url": URLS[0]}, {"title": "b", "url": URLS[1]}])
    return [
        {"type": "system", "subtype": "init"},
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "id": SEARCH_ID, "name": CLI_SEARCH_TOOL, "input": {}}]},
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": SEARCH_ID,
                        "content": f'Web search results for query: "x"\n\nLinks: {links}\n\nSummary ...',
                    }
                ]
            },
        },
    ]


def stream(*events: dict[str, Any]) -> str:
    return "\n".join(json.dumps(event) for event in events) + "\n"


# --- the command line -------------------------------------------------------------------------------------------------


def test_argv_shuts_the_run_off_from_tools_mcp_and_settings(tmp_path: Path) -> None:
    argv = cli_argv(["claude"], params(), Answer, tmp_path / "system.md")
    assert argv[:2] == ["claude", "-p"]
    flags = dict(zip(argv, [*argv[1:], None], strict=True))
    assert flags["--model"] == DEFAULT_MODEL
    assert flags["--effort"] == "low"
    assert flags["--tools"] == "" and flags["--setting-sources"] == ""
    assert flags["--system-prompt-file"] == str(tmp_path / "system.md")
    assert {"--strict-mcp-config", "--no-session-persistence", "--verbose"} <= set(argv)
    assert "--allowedTools" not in argv
    schema = json.loads(str(flags["--json-schema"]))
    assert set(schema["properties"]) == {"playing", "source_url"}


def test_a_search_prompt_gets_websearch_with_its_domains_and_cap_in_the_system_prompt(tmp_path: Path) -> None:
    argv = cli_argv(["claude"], params(search=True), Answer, tmp_path / "system.md")
    flags = dict(zip(argv, [*argv[1:], None], strict=True))
    assert flags["--tools"] == CLI_SEARCH_TOOL and flags["--allowedTools"] == CLI_SEARCH_TOOL
    text = system_text(params(search=True))
    assert text.startswith("You are a terse analyst.\n\nLeague context.")
    assert "espn.com, rotowire.com" in text and "at most 2 times" in text
    assert CLI_SEARCH_TOOL not in system_text(params())
    assert user_text(params()) == QUESTION


def test_the_child_environment_has_no_api_credentials() -> None:
    env = cli_env({"ANTHROPIC_API_KEY": "sk", "ANTHROPIC_AUTH_TOKEN": "t", "CLAUDECODE": "1", "PATH": "p"})
    assert env == {"PATH": "p"}


# --- reading a run ----------------------------------------------------------------------------------------------------


def test_a_searched_answer_parses_with_its_urls_and_summed_tokens() -> None:
    reply = reply_from_stream(stream(*search_events(), result_event()), Answer, model=DEFAULT_MODEL)
    assert reply.stop_reason == "end_turn"
    assert reply.output == Answer(playing=True, source_url=URLS[0])
    assert reply.search_urls == URLS
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (120, 40)
    assert (reply.usage.cache_read_input_tokens, reply.usage.cache_creation_input_tokens) == (900, 300)
    assert reply.request_id == "sess-1" and reply.model == DEFAULT_MODEL and not reply.fallback


def test_stderr_lines_and_unrelated_tool_results_are_ignored() -> None:
    other = {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "tool_use_id": "x", "content": "Links: []"}]},
    }
    text = "a warning on stderr\n" + stream(other, result_event())
    reply = reply_from_stream(text, Answer, model=DEFAULT_MODEL)
    assert reply.output is not None and reply.search_urls == ()


def test_an_off_schema_answer_is_end_turn_with_no_output() -> None:
    reply = reply_from_stream(stream(result_event(structured_output={"playing": "maybe"})), Answer, model=DEFAULT_MODEL)
    assert reply.stop_reason == "end_turn" and reply.output is None
    assert reply.detail is not None and "does not match Answer" in reply.detail
    missing = reply_from_stream(stream(result_event(structured_output=None)), Answer, model=DEFAULT_MODEL)
    assert missing.output is None and missing.detail == "the CLI returned no structured output"


def test_a_refusal_is_a_refusal_and_a_fallback_model_is_named() -> None:
    refused = reply_from_stream(
        stream(result_event(stop_reason="refusal", structured_output=None, result="declined")),
        Answer,
        model=DEFAULT_MODEL,
    )
    assert refused.stop_reason == "refusal" and refused.detail == "declined" and refused.output is None
    other = result_event(modelUsage={"claude-opus-5": {"inputTokens": 1, "outputTokens": 1}})
    served = reply_from_stream(stream(other), Answer, model=DEFAULT_MODEL)
    assert served.model == "claude-opus-5" and served.fallback


def test_an_error_result_or_no_result_raises() -> None:
    limit = result_event(subtype="error_during_execution", is_error=True, result="usage limit reached")
    with pytest.raises(AdvisorError, match="usage limit reached"):
        reply_from_stream(stream(limit), Answer, model=DEFAULT_MODEL)
    with pytest.raises(AdvisorError, match="no result .*not logged in"):
        reply_from_stream("not logged in\n", Answer, model=DEFAULT_MODEL)


# --- the transport over a fake CLI ------------------------------------------------------------------------------------

FAKE_CLI = """
import json, os, sys, time
argv = sys.argv[1:]
flags = dict(zip(argv, argv[1:] + [None]))
if os.environ.get("FAKE_SLEEP"):
    time.sleep(float(os.environ["FAKE_SLEEP"]))
problems = []
if "ANTHROPIC_API_KEY" in os.environ:
    problems.append("api key leaked")
if sys.stdin.read() != EXPECTED_QUESTION:
    problems.append("question not on stdin")
with open(flags["--system-prompt-file"], encoding="utf-8") as f:
    if not f.read().startswith("You are a terse analyst."):
        problems.append("system prompt missing")
if set(os.listdir(".")) - {"system.md", "question.md", "events.jsonl"}:
    problems.append("cwd holds more than the run's own files")
print("stderr noise", file=sys.stderr, flush=True)
for event in EVENTS:
    print(json.dumps(event), flush=True)
print(json.dumps(dict(RESULT, structured_output=None, result="; ".join(problems)) if problems else RESULT))
"""


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def fake_cli(tmp_path: Path, *, search: bool) -> list[str]:
    script = tmp_path / "fake_claude.py"
    script.write_text(
        f"EXPECTED_QUESTION = {QUESTION!r}\nEVENTS = {search_events() if search else []!r}\n"
        f"RESULT = {result_event()!r}\n" + FAKE_CLI,
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def test_a_worker_call_runs_the_cli_and_is_recorded(tmp_path: Path, store: Store) -> None:
    transport = ClaudeCliTransport(fake_cli(tmp_path, search=True), environ={**os.environ, "ANTHROPIC_API_KEY": "sk"})
    client = AdvisorClient(store, Llm(backend="claude_code"), transport, refusal_fallback=False)
    reply = client.ask("close_call", prompt(search=True), Answer)
    assert reply.ok, reply.describe()
    assert reply.output == Answer(playing=True, source_url=URLS[0])
    assert reply.search_urls == URLS
    assert reply.usage.id is not None and reply.usage.cost_usd > 0
    assert client.spent_today() == pytest.approx(reply.usage.cost_usd)


def test_a_hung_cli_is_killed_and_raises(tmp_path: Path) -> None:
    transport = ClaudeCliTransport(
        fake_cli(tmp_path, search=False), timeout=1.0, environ={**os.environ, "FAKE_SLEEP": "30"}
    )
    with pytest.raises(AdvisorError, match="timed out after 1s"):
        transport.parse(params(), Answer)


def test_a_missing_cli_and_batches_raise(tmp_path: Path) -> None:
    missing = ClaudeCliTransport([str(tmp_path / "no-such-claude")])
    with pytest.raises(AdvisorError, match="could not start"):
        missing.parse(params(), Answer)
    with pytest.raises(AdvisorError, match="no Message Batches"):
        missing.create_batch([])
    assert "backend" in NO_BATCHES
