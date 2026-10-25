"""``fm.jobs.advice``: the tick's advisor, or the reason there is none (ROADMAP #58). No test holds a real key or
opens a socket: the keyed configurations carry a made-up key that is never sent."""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from fm.advisor.claude_cli import ClaudeCliTransport
from fm.advisor.client import AdvisorClient, AdvisorError, CallParams, RawReply, Transport
from fm.config import DEFAULT_MODEL, Config, Llm, Secrets
from fm.jobs.advice import advisor_for, advisor_off, claude_cli_path
from fm.store import Store

LEAGUE: dict[str, Any] = {"key": "nfl", "sport": "nfl", "espn_league_id": 1234567, "season": 2026, "team_id": 1}
CONFIG = Config.model_validate({"league": [LEAGUE]})
KEYED = CONFIG.model_copy(update={"secrets": Secrets.model_validate({"ANTHROPIC_API_KEY": "sk-test-not-a-real-key"})})
NO_KEY_NOTE = "ANTHROPIC_API_KEY is not set in .env, so the advisor is off (fm config check)"
NO_BUDGET_NOTE = "[llm] daily_budget_usd is 0 in config.toml, so the advisor is off"


class NoTransport(Transport):
    def parse(self, params: CallParams, output: type[BaseModel], *, fallback: bool = False) -> RawReply:
        raise AssertionError("nothing is asked here")


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def test_advisor_off_names_the_missing_key_then_the_zero_budget() -> None:
    assert advisor_off(CONFIG) == NO_KEY_NOTE
    assert advisor_off(CONFIG.model_copy(update={"llm": Llm(daily_budget_usd=0)})) == NO_KEY_NOTE  # the key first
    assert advisor_off(KEYED.model_copy(update={"llm": Llm(daily_budget_usd=0)})) == NO_BUDGET_NOTE
    assert advisor_off(KEYED) is None


def test_advisor_for_builds_a_client_for_the_configured_model_and_cap(store: Store) -> None:
    config = KEYED.model_copy(update={"llm": Llm(model="claude-opus-5-5", daily_budget_usd=0.75)})
    client = advisor_for(store, config, transport=NoTransport())
    assert isinstance(client, AdvisorClient)
    assert (client.model, client.daily_budget_usd, client.store) == (DEFAULT_MODEL, 0.75, store)
    live = advisor_for(store, KEYED)  # the real SDK transport over the made-up key: built, nothing sent
    assert isinstance(live, AdvisorClient) and live.daily_budget_usd == 2.0


def test_advisor_for_is_none_with_the_reason_logged_and_never_raises(
    store: Store, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with caplog.at_level(logging.INFO, logger="fm.jobs.advice"):
        assert advisor_for(store, CONFIG) is None
        assert advisor_for(store, KEYED.model_copy(update={"llm": Llm(daily_budget_usd=0)})) is None
    assert NO_KEY_NOTE in caplog.text and NO_BUDGET_NOTE in caplog.text

    def broken(*_args: Any, **_kwargs: Any) -> AdvisorClient:
        raise AdvisorError("the SDK refused the key")

    monkeypatch.setattr(AdvisorClient, "from_config", broken)
    with caplog.at_level(logging.WARNING, logger="fm.jobs.advice"):
        assert advisor_for(store, KEYED) is None
    assert "could not build the Claude client; running without it" in caplog.text
    assert "the SDK refused the key" in caplog.text


def test_a_tick_without_a_key_never_loads_the_sdk() -> None:
    """Importing the tick and asking it for an advisor it cannot have must not pay for the Anthropic SDK."""
    code = (
        "import sys, fm.jobs.tick; from fm.config import Config; from fm.jobs.advice import advisor_for; "
        f"config = Config.model_validate({{'league': [{LEAGUE!r}]}}); "
        "assert advisor_for(None, config) is None; "
        "print([n for n in sorted(sys.modules) if n.startswith(('anthropic', 'fm.advisor'))])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]", done.stdout


SUBSCRIPTION = CONFIG.model_copy(update={"llm": Llm(backend="claude_code")})


def test_the_claude_code_backend_needs_the_cli_not_a_key() -> None:
    assert advisor_off(SUBSCRIPTION, cli_path=lambda: "/bin/claude") is None
    note = advisor_off(SUBSCRIPTION, cli_path=lambda: None)
    assert note is not None and "claude CLI was not found" in note
    spent = SUBSCRIPTION.model_copy(update={"llm": Llm(backend="claude_code", daily_budget_usd=0)})
    assert advisor_off(spent, cli_path=lambda: "/bin/claude") == NO_BUDGET_NOTE
    assert "ANTHROPIC_API_KEY" not in SUBSCRIPTION.required_secrets()
    assert "ANTHROPIC_API_KEY" in CONFIG.required_secrets()


def test_advisor_for_runs_the_subscription_over_the_cli(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fm.jobs.advice.claude_cli_path", lambda: "/opt/claude")
    client = advisor_for(store, SUBSCRIPTION)
    assert isinstance(client, AdvisorClient) and not client.refusal_fallback
    transport = client._transport  # pyright: ignore[reportPrivateUsage]
    assert isinstance(transport, ClaudeCliTransport) and transport.command == ("/opt/claude",)
    with pytest.raises(AdvisorError, match="ClaudeCliTransport"):
        AdvisorClient.from_config(store, SUBSCRIPTION)


def test_claude_cli_path_prefers_the_native_install(tmp_path: Path) -> None:
    local = tmp_path / ".local" / "bin" / "claude.exe"
    assert claude_cli_path(which=lambda _: "C:/on/path/claude", home=tmp_path, platform="win32") == "C:/on/path/claude"
    local.parent.mkdir(parents=True)
    local.write_text("")
    assert claude_cli_path(which=lambda _: None, home=tmp_path, platform="win32") == str(local)
    assert claude_cli_path(which=lambda _: None, home=tmp_path, platform="darwin") is None
