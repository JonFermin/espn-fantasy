"""The MCP server (ROADMAP #40): read tools and ``create_proposal``, with no way to act.

The tools are called through FastMCP's in-memory client against a private copy of ``tests/fixtures/home`` (config,
state DB and the captured views the sync left in the cache), so ``create_proposal`` may store rows without touching the
committed fixture. The clock is pinned with ``--as-of``-style arguments and ``fm.mcp_server._now``. No network, no
browser: the server never opens an ESPN session.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StdioTransport
from mcp.types import Tool
from typer.testing import CliRunner

import fm.mcp_server as server_module
from fm import paths
from fm.cli import app
from fm.mcp_server import MCP_CREATED_BY, TOOL_NAMES, build_server
from fm.proposals import approve, auto_approve_due, pause
from fm.store import Store

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HOME = FIXTURES / "home"
AS_OF = "2026-10-04T15:00Z"  # Sunday 11 a.m. ET of week 4, as in tests/test_advise_cli.py
NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
DEADLINE = "2026-10-04T17:00:00Z"  # the early games' kickoff, after AS_OF
ALLEGIER, SHAHEED, ALLEN, DOWDLE, BARKLEY = 4373626, 4249836, 3918298, 4038815, 3929630
OPEN_SLOT, BENCH_SLOT = 4, 20  # WR and the bench
READ_TOOLS = ("status", "lineup", "waivers", "trade_eval", "list_proposals")
WRITEY = ("approv", "execut", "writ", "send", "submit", "delet", "paus", "resum")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the fixture home (FM_CONFIG_DIR and FM_CACHE_DIR point into it), with the clock at AS_OF."""
    copy = tmp_path / "home"
    shutil.copytree(HOME, copy, ignore=shutil.ignore_patterns("snapshots", "build.py", "README.md", "__pycache__"))
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(copy))
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(copy / "cache"))
    monkeypatch.setattr(server_module, "_now", lambda: NOW)
    return copy


def call_raw(tool: str, **arguments: Any) -> CallToolResult:
    async def go() -> CallToolResult:
        async with Client(build_server()) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    return asyncio.run(go())


def call(tool: str, **arguments: Any) -> dict[str, Any]:
    result = call_raw(tool, **arguments)
    assert not result.is_error, result.content
    assert isinstance(result.structured_content, dict)
    json.dumps(result.structured_content)  # JSON all the way down
    return result.structured_content


def tools() -> list[Tool]:
    async def go() -> list[Tool]:
        async with Client(build_server()) as client:
            return await client.list_tools()

    return asyncio.run(go())


def stored(home: Path) -> list[tuple[str, str, str, str]]:
    with Store.open(home / "state.db") as store:
        return [(row.kind, row.status, row.policy, row.created_by) for row in store.proposals.find()]


# --- the tool list is the guarantee -------------------------------------------------------------------------------


def test_the_tool_list_is_exactly_the_allowlist() -> None:
    listed = tools()
    assert sorted(tool.name for tool in listed) == sorted(TOOL_NAMES)
    assert set(TOOL_NAMES) == {*READ_TOOLS, "create_proposal"}
    by_name = {tool.name: tool for tool in listed}
    for name in READ_TOOLS:
        annotations = by_name[name].annotations
        assert annotations is not None and annotations.readOnlyHint is True
    stores = by_name["create_proposal"].annotations
    assert stores is not None and stores.destructiveHint is False and stores.openWorldHint is False


def test_no_tool_name_description_or_parameter_implies_a_write() -> None:
    for tool in tools():
        text = json.dumps({"name": tool.name, "description": tool.description, "input": tool.inputSchema}).lower()
        hits = [word for word in WRITEY if word in text]
        assert hits == [], f"{tool.name} mentions {hits}"
    assert not any(word in server_module.INSTRUCTIONS.lower() for word in WRITEY)


FORBIDDEN_MODULES = ("fm.executor", "fm.browser", "playwright", "fm.commands", "fm.advisor")
FORBIDDEN_NAMES = frozenset(
    {"approve", "reject", "begin_execution", "finish_execution", "auto_approve_due", "pause", "resume", "expire_due"}
)


def test_the_server_module_does_not_import_the_executor_a_browser_or_a_lifecycle_function() -> None:
    tree = ast.parse(Path(server_module.__file__ or "").read_text(encoding="utf-8"))
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):  # every import, including ones inside functions
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            names.update(alias.name for alias in node.names)
    assert {m for m in modules if m.startswith(FORBIDDEN_MODULES)} == set()
    assert names & FORBIDDEN_NAMES == set()
    assert "propose" in names and "evaluate" in names  # the checks above would pass on an empty parse


def test_importing_and_building_the_server_loads_no_executor_or_write_flow() -> None:
    code = (
        "import sys, fm.mcp_server as m; m.build_server(); "
        "print([n for n in sorted(sys.modules) if n.startswith(('fm.executor', 'fm.browser.flows', "
        "'fm.browser.transactions', 'fm.advisor', 'fm.commands'))])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]", done.stdout


def test_the_server_cannot_reach_a_lifecycle_function_through_its_namespace() -> None:
    namespace = vars(server_module)
    assert FORBIDDEN_NAMES & set(namespace) == set()
    assert "executor" not in {name.lower() for name in namespace}


# --- the read tools ---------------------------------------------------------------------------------------------------


def test_status_reports_the_league_the_schedule_and_the_kill_switch(home: Path) -> None:
    out = call("status")
    assert out["kill_switch"] == {"on": False, "detail": None}
    (league,) = out["leagues"]
    assert league["league"] == "nfl" and league["synced"] is True
    assert league["team"]["name"] == "Fixture Team 1"
    assert league["faab"] == {"spent": 23, "budget": 100}
    assert league["pro_schedule"]["first_period"] == 4
    assert league["open_proposals"] == []
    pause(reason="testing", now=NOW)
    assert call("status")["kill_switch"]["on"] is True


def test_status_for_an_empty_store_says_to_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "empty"
    config.mkdir()
    shutil.copy(HOME / "config.toml", config / "config.toml")
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(config))
    (league,) = call("status")["leagues"]
    assert league["synced"] is False and "fm sync" in league["note"]


def test_a_missing_league_or_bad_clock_is_a_tool_error(home: Path) -> None:
    assert call_raw("status", league="nope").is_error
    assert call_raw("lineup", as_of="yesterday").is_error


def test_lineup_is_a_plan_that_stores_nothing(home: Path) -> None:
    out = call("lineup", league="nfl", as_of=AS_OF, opponent_team_id=2)
    (block,) = out["leagues"]
    assert block["ok"] is True and block["scoring_period"] == 4
    assert block["recommended"]["expected_points"] >= block["current"]["expected_points"]
    assert block["opponent"]["name"] == "Fixture Team 2"
    starters = {row["name"]: row for row in block["lineup"]}
    assert starters["Josh Allen"]["slot"] == "QB"
    assert block["moves"] and all({"name", "from_slot", "to_slot"} <= set(move) for move in block["moves"])
    for suggestion in block["suggested_proposals"]:
        assert suggestion["policy"]["allowed"] is True
        assert suggestion["payload"]["moves"]
    assert stored(home) == []  # a plan, not proposals
    assert call("list_proposals")["count"] == 0


def test_lineup_without_a_captured_schedule_says_why_instead_of_planning(home: Path) -> None:
    shutil.rmtree(home / "cache")
    (block,) = call("lineup", as_of=AS_OF)["leagues"]
    assert block["ok"] is False and "pro schedule" in block["error"]


def test_waivers_is_a_ranking_that_stores_nothing(home: Path) -> None:
    out = call("waivers", as_of=AS_OF, top=3)
    (block,) = out["leagues"]
    assert block["ok"] is True
    claim = block["moves"][0]
    assert claim["kind"] == "waiver" and claim["add"]["name"] == "Jaylen Warren"
    assert claim["policy"]["allowed"] is True and claim["payload"]["add_espn_id"] == claim["add"]["espn_id"]
    assert len(block["ranked"]) == 3 and block["ranked_total"] >= 3
    assert block["replacement_levels"]
    assert stored(home) == []


def test_trade_eval_judges_a_deal_and_returns_the_payload(home: Path) -> None:
    out = call("trade_eval", deal="give Tyler Allgeier get Saquon Barkley", as_of=AS_OF, runs=100)
    assert out["league"] == "nfl" and out["with_team"]["team_id"] == 2
    assert [p["name"] for p in out["give"]] == ["Tyler Allgeier"]
    assert [p["name"] for p in out["get"]] == ["Saquon Barkley"]
    assert out["recommendation"] in {"accept", "decline", "counter"}
    assert out["simulated"] is False  # the fixture home has no mMatchup capture; the output says so
    assert any("mMatchup" in warning for warning in out["warnings"])
    assert 0 < out["acceptance"]["p_accept"] < 1
    assert out["proposal_payload"] == {"other_team_id": 2, "give_espn_ids": [ALLEGIER], "get_espn_ids": [BARKLEY]}
    assert stored(home) == []


def test_trade_eval_reports_a_deal_it_cannot_read_as_a_tool_error(home: Path) -> None:
    for deal in (
        "trade Allgeier for Barkley",
        "give Nobody Atall get Saquon Barkley",
        "give Josh Allen get Josh Allen",
    ):
        result = call_raw("trade_eval", deal=deal, as_of=AS_OF)
        assert result.is_error, deal


def test_list_proposals_shows_the_queue_without_the_execution_token(home: Path) -> None:
    with Store.open(home / "state.db") as store:
        row = store.leagues.by_key("nfl")
        assert row is not None
    created = call("create_proposal", league="nfl", kind="add_drop", payload={"add_espn_id": DOWDLE}, rationale="r")
    assert created["status"] == "proposed"
    with Store.open(home / "state.db") as store:  # a decided proposal carries a token; it must not be shown
        approve(store, created["proposal"]["id"], decided_by="test", now=NOW)
        assert store.proposals.get(created["proposal"]["id"]).execution_token  # type: ignore[union-attr]
        token = store.proposals.get(created["proposal"]["id"]).execution_token  # type: ignore[union-attr]
    listed = call("list_proposals")
    (item,) = listed["proposals"]
    assert item["status"] == "approved" and token is not None and token not in json.dumps(listed)
    assert "execution_token" not in json.dumps(listed)
    assert call("list_proposals", include_closed=True)["count"] == 1
    assert call("status")["leagues"][0]["open_proposals"][0]["id"] == item["id"]


# --- create_proposal --------------------------------------------------------------------------------------------------


def test_create_proposal_stores_one_proposed_row_tagged_mcp(home: Path) -> None:
    out = call(
        "create_proposal",
        league="nfl",
        kind="add_drop",
        payload={"add_espn_id": DOWDLE, "drop_espn_id": ALLEGIER},
        rationale="Dowdle has the better rest-of-season outlook.",
        scoring_period_id=4,
    )
    assert out["status"] == "proposed" and out["reasons"] == []
    proposal = out["proposal"]
    assert proposal["created_by"] == MCP_CREATED_BY == "mcp"
    assert proposal["status"] == "proposed" and proposal["policy"] == "approve"
    assert proposal["rationale"].startswith("Dowdle")
    assert proposal["decided_by"] is None and "token" not in json.dumps(out)
    assert stored(home) == [("add_drop", "proposed", "approve", "mcp")]


def test_asking_twice_returns_the_open_proposal(home: Path) -> None:
    args = {"league": "nfl", "kind": "add_drop", "payload": {"add_espn_id": DOWDLE}, "rationale": "first"}
    first = call("create_proposal", **args)
    again = call("create_proposal", **{**args, "rationale": "second"})
    assert first["status"] == "proposed" and again["status"] == "existing"
    assert again["proposal"]["id"] == first["proposal"]["id"] and again["proposal"]["rationale"] == "first"
    assert len(stored(home)) == 1


def test_a_policy_refusal_is_a_result_with_every_reason(home: Path) -> None:
    out = call(
        "create_proposal",
        league="nfl",
        kind="add_drop",
        payload={"add_espn_id": DOWDLE, "drop_espn_id": ALLEN},
        rationale="drop the untouchable",
    )
    assert out["status"] == "blocked" and out["proposal"] is None
    assert any("Josh Allen" in reason and "untouchable" in reason for reason in out["reasons"])
    assert out["policy"]["allowed"] is False
    assert stored(home) == []


def test_a_waiver_bid_over_the_cap_and_a_passed_deadline_are_refused_together(home: Path) -> None:
    out = call(
        "create_proposal",
        league="nfl",
        kind="waiver",
        payload={"add_espn_id": DOWDLE, "bid_amount": 99},
        rationale="all in",
        deadline="2026-10-04T14:00:00Z",
    )
    assert out["status"] == "blocked"
    assert len(out["reasons"]) >= 2 and any("passed" in reason for reason in out["reasons"])
    assert stored(home) == []


def test_the_kill_switch_blocks_a_proposal(home: Path) -> None:
    pause(reason="testing", now=NOW)
    out = call("create_proposal", league="nfl", kind="add_drop", payload={"add_espn_id": DOWDLE}, rationale="r")
    assert out["status"] == "blocked" and any("paused" in reason for reason in out["reasons"])
    assert stored(home) == []


@pytest.mark.parametrize(
    ("arguments", "needle"),
    [
        ({"kind": "teleport", "payload": {}}, "unknown proposal kind"),
        ({"kind": "add_drop", "payload": {}}, "payload"),
        ({"kind": "add_drop", "payload": {"add_espn_id": 1, "extra": 2}}, "extra"),
        ({"kind": "waiver", "payload": {"add_espn_id": 1, "bid_amount": -3}}, "bid_amount"),
        ({"kind": "lineup", "payload": {"moves": []}}, "moves"),
        ({"kind": "add_drop", "payload": {"add_espn_id": 1}, "deadline": "soon"}, "deadline"),
        ({"kind": "add_drop", "payload": {"add_espn_id": 1}, "league": "nope"}, "nope"),
    ],
)
def test_an_invalid_request_is_a_result_and_stores_nothing(home: Path, arguments: dict[str, Any], needle: str) -> None:
    request = {"league": "nfl", "rationale": "r", **arguments}
    out = call("create_proposal", **request)
    assert out["status"] == "invalid" and out["proposal"] is None
    assert needle in " ".join(out["reasons"])
    assert stored(home) == []


def test_a_trade_is_stored_as_approve_only(home: Path) -> None:
    out = call(
        "create_proposal",
        league="nfl",
        kind="trade_propose",
        payload={"other_team_id": 2, "give_espn_ids": [ALLEGIER], "get_espn_ids": [BARKLEY]},
        rationale="Saquon for Allgeier.",
    )
    assert out["status"] == "proposed", out
    assert out["proposal"]["policy"] == "approve" and out["proposal"]["kind"] == "trade_propose"
    assert stored(home) == [("trade_propose", "proposed", "approve", "mcp")]


def test_an_mcp_proposal_is_never_left_on_auto(home: Path) -> None:
    """bench_inactive is ``auto`` in the fixture config: the engine's own drafts may be approved at T-15, a move Claude
    suggested may not."""
    out = call(
        "create_proposal",
        league="nfl",
        kind="bench_inactive",
        payload={"moves": [{"espn_id": SHAHEED, "from_slot_id": BENCH_SLOT, "to_slot_id": OPEN_SLOT}]},
        rationale="Start Shaheed.",
        scoring_period_id=4,
        deadline=DEADLINE,
    )
    assert out["status"] == "proposed", out
    assert out["proposal"]["policy"] == "approve" and "never approved on its own" in out["note"]
    assert stored(home) == [("bench_inactive", "proposed", "approve", "mcp")]
    with Store.open(home / "state.db") as store:  # and the tick's sweep finds nothing to approve on its own
        assert auto_approve_due(store, now=datetime(2026, 10, 4, 16, 50, tzinfo=UTC)) == []


# --- the real entry point ---------------------------------------------------------------------------------------------


def test_fm_mcp_is_discovered_and_has_help() -> None:
    result = CliRunner().invoke(app, ["mcp", "--help"], catch_exceptions=False)
    assert result.exit_code == 0 and "MCP" in result.output
    assert "mcp" in CliRunner().invoke(app, ["--help"], catch_exceptions=False).output


def test_fm_mcp_serves_the_tools_over_stdio(home: Path) -> None:
    """The real command in a subprocess: stdout carries only the protocol, and the tools are the allowlist."""
    transport = StdioTransport(
        command=sys.executable,
        args=["-c", "from fm.cli import app; app()", "mcp"],
        env={**os.environ, "FM_CONFIG_DIR": str(home), "FM_CACHE_DIR": str(home / "cache")},
        keep_alive=False,
    )

    async def go() -> tuple[list[str], dict[str, Any]]:
        async with Client(transport) as client:
            names = [tool.name for tool in await client.list_tools()]
            result = await client.call_tool("list_proposals", {})
            return names, dict(result.structured_content or {})

    names, listed = asyncio.run(go())
    assert sorted(names) == sorted(TOOL_NAMES)
    assert listed == {"count": 0, "proposals": []}
