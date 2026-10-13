"""The shared read views (ROADMAP #52): :mod:`fm.views` builds what the MCP tools and the dashboard show, and acts on
nothing.

Each public view is called directly against a private copy of ``tests/fixtures/home`` (``FM_CONFIG_DIR`` and
``FM_CACHE_DIR`` point into it, as in ``tests/test_mcp_server.py``), with the clock pinned through ``as_of`` or ``now``.
The import checks pin the no-write-path rule the way the MCP server's tests pin it, and add that the module loads no
part of the MCP stack, so a dashboard can serve the views without fastmcp. A last test checks that each MCP read tool
returns exactly what its view returns. No network, no browser.
"""

from __future__ import annotations

import ast
import asyncio
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import fm.views as views
from fm import paths
from fm.config import load_config
from fm.proposals import AddDropPayload, approve, evaluate, pause, propose
from fm.store import ProposalRow, Store
from fm.views import ViewError

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HOME = FIXTURES / "home"
PRO_SCHEDULE = FIXTURES / "sports" / "ffl_pro_schedule_2026.json"  # the capture the fixture home's cache holds
MATCHUPS = FIXTURES / "espn" / "ffl_matchups.json"  # the fixture league's mMatchup view, which the home lacks
AS_OF = "2026-10-04T15:00Z"  # Sunday 11 a.m. ET of week 4, as in tests/test_mcp_server.py
NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
DEAL = "give Tyler Allgeier get Saquon Barkley"
ALLEGIER, ALLEN, DOWDLE, BARKLEY = 4373626, 3918298, 4038815, 3929630


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the fixture home, with FM_CONFIG_DIR and FM_CACHE_DIR pointing into it."""
    copy = tmp_path / "home"
    shutil.copytree(HOME, copy, ignore=shutil.ignore_patterns("snapshots", "build.py", "README.md", "__pycache__"))
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(copy))
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(copy / "cache"))
    return copy


@pytest.fixture
def unsynced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The fixture config with an empty store: the league is configured but never synced."""
    config = tmp_path / "unsynced"
    config.mkdir()
    shutil.copy(HOME / "config.toml", config / "config.toml")
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(config))
    return config


def jsonable(out: dict[str, Any]) -> dict[str, Any]:
    """``out`` survives a JSON round trip unchanged: only dicts, lists, strings, numbers, booleans and ``None``."""
    assert json.loads(json.dumps(out)) == out
    return out


def stored(home: Path) -> list[tuple[str, str]]:
    with Store.open(home / "state.db") as store:
        return [(row.kind, row.status) for row in store.proposals.find()]


# --- no write path, no MCP stack --------------------------------------------------------------------------------------

FORBIDDEN_MODULES = (
    "fastmcp",
    "mcp",
    "fm.mcp_server",
    "fm.executor",
    "fm.browser",
    "playwright",
    "fm.commands",
    "fm.advisor",
)
FORBIDDEN_NAMES = frozenset(
    {
        "propose",
        "approve",
        "reject",
        "begin_execution",
        "finish_execution",
        "auto_approve_due",
        "pause",
        "resume",
        "expire_due",
    }
)


def _within(module: str, roots: tuple[str, ...]) -> bool:
    return any(module == root or module.startswith(f"{root}.") for root in roots)


def test_the_views_module_imports_no_mcp_stack_executor_browser_or_lifecycle_function() -> None:
    tree = ast.parse(Path(views.__file__ or "").read_text(encoding="utf-8"))
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):  # every import, including ones inside functions
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            names.update(alias.name for alias in node.names)
    assert {module for module in modules if _within(module, FORBIDDEN_MODULES)} == set()
    assert names & FORBIDDEN_NAMES == set()
    assert "evaluate" in names and "plan_lineup" in names  # the checks above would pass on an empty parse


def test_importing_the_views_loads_no_mcp_stack_executor_or_write_flow() -> None:
    """``fm.espn.auth`` reads the session cookies through ``fm.browser.session``, so that one module (no write flow)
    comes in with the ESPN models; nothing else under ``fm.browser`` may, and nothing of fastmcp or the executor."""
    code = (
        "import sys, fm.views; "
        "roots = ('fastmcp', 'mcp', 'fm.mcp_server', 'fm.executor', 'fm.advisor', 'fm.commands', 'fm.browser'); "
        "allowed = {'fm.browser', 'fm.browser.session'}; "
        "print([n for n in sorted(sys.modules) "
        "if n not in allowed and any(n == r or n.startswith(r + '.') for r in roots)])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]", done.stdout


def test_the_views_cannot_reach_a_lifecycle_function_through_their_namespace() -> None:
    namespace = vars(views)
    assert FORBIDDEN_NAMES & set(namespace) == set()
    assert not any("executor" in name.lower() or "mcp" in name.lower() for name in namespace)


# --- inputs and errors ------------------------------------------------------------------------------------------------


def test_parse_as_of_gives_an_aware_utc_instant() -> None:
    assert views.parse_as_of(AS_OF) == NOW
    assert views.parse_as_of("2026-10-04T15:00") == NOW  # naive is UTC
    assert views.parse_as_of("2026-10-04T11:00-04:00") == NOW and views.parse_as_of("2026-10-04T11:00-04:00").tzinfo
    assert views.parse_as_of(None, now=NOW) == NOW
    assert views.parse_as_of(None, now=datetime(2026, 10, 4, 15, 0)) == NOW
    eastern = datetime(2026, 10, 4, 11, 0, tzinfo=timezone(timedelta(hours=-4)))
    assert views.parse_as_of(None, now=eastern).tzinfo is UTC
    assert views.parse_as_of(AS_OF, now=datetime(2030, 1, 1, tzinfo=UTC)) == NOW  # as_of wins over now
    assert abs(views.parse_as_of(None) - datetime.now(UTC)) < timedelta(minutes=1)
    with pytest.raises(ViewError, match="as_of 'yesterday' is not an ISO 8601 timestamp"):
        views.parse_as_of("yesterday")


VIEW_CALLS: dict[str, Callable[[str | None], dict[str, Any]]] = {
    "status": lambda league: views.status(league),
    "lineup": lambda league: views.lineup(league, as_of=AS_OF),
    "waivers": lambda league: views.waivers(league, as_of=AS_OF),
    "trade_eval": lambda league: views.trade_eval(DEAL, league, as_of=AS_OF, runs=100),
    "list_proposals": lambda league: views.list_proposals(league),
}


@pytest.mark.parametrize("view", sorted(VIEW_CALLS))
def test_an_unknown_league_is_a_view_error_naming_the_known_ones(home: Path, view: str) -> None:
    assert issubclass(ViewError, ValueError)
    with pytest.raises(ViewError, match="no league 'nope' in config.toml; known: nfl"):
        VIEW_CALLS[view]("nope")


@pytest.mark.parametrize("view", sorted(VIEW_CALLS))
def test_a_config_that_does_not_load_is_a_view_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, view: str
) -> None:
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(tmp_path / "nowhere"))
    with pytest.raises(ViewError):
        VIEW_CALLS[view](None)


@pytest.mark.parametrize(
    ("call", "needle"),
    [
        (lambda: views.lineup(as_of="yesterday"), "as_of 'yesterday'"),
        (lambda: views.lineup(opponent_team_id=0), "opponent_team_id must be at least 1"),
        (lambda: views.waivers(top=-1), "top must be at least 0"),
        (lambda: views.waivers(max_moves=0), "max_moves must be at least 1"),
        (lambda: views.trade_eval(DEAL, runs=0), "runs must be at least 1"),
        (lambda: views.list_proposals(limit=0), "limit must be at least 1"),
        (lambda: views.lineup(schedule_path=FIXTURES / "missing.json"), "does not exist"),
        (lambda: views.waivers(schedule_path=MATCHUPS), "has no scoring periods"),
        (lambda: views.trade_eval(DEAL, matchups_path=PRO_SCHEDULE), "has no matchups"),
    ],
)
def test_a_bad_argument_is_a_view_error_that_stores_nothing(
    home: Path, call: Callable[[], dict[str, Any]], needle: str
) -> None:
    with pytest.raises(ViewError, match=needle):
        call()
    assert stored(home) == []


# --- status -----------------------------------------------------------------------------------------------------------


def test_status_reports_the_league_the_schedule_and_the_kill_switch(home: Path) -> None:
    out = jsonable(views.status())
    assert out["kill_switch"] == {"on": False, "detail": None}
    (league,) = out["leagues"]
    assert league["league"] == "nfl" and league["synced"] is True and league["sport"] == "nfl"
    assert league["team"]["name"] == "Fixture Team 1"
    assert league["faab"] == {"spent": 23, "budget": 100}
    assert league["pro_schedule"]["first_period"] == 4 and league["pro_schedule"]["captured_at"]
    assert league["roster_period"] == 4 and league["open_proposals"] == []
    assert views.status("nfl") == out
    pause(reason="testing", now=NOW)
    switch = views.status()["kill_switch"]
    assert switch["on"] is True and "testing" in switch["detail"]


def test_status_for_an_unsynced_league_says_to_sync(unsynced: Path) -> None:
    (league,) = jsonable(views.status())["leagues"]
    assert league["synced"] is False and "fm sync" in league["note"]


# --- lineup -----------------------------------------------------------------------------------------------------------


def test_lineup_is_a_plan_that_stores_nothing(home: Path) -> None:
    out = jsonable(views.lineup("nfl", as_of=AS_OF, opponent_team_id=2))
    assert out["as_of"] == NOW.isoformat()
    (block,) = out["leagues"]
    assert block["ok"] is True and block["scoring_period"] == 4
    assert block["recommended"]["expected_points"] >= block["current"]["expected_points"]
    assert block["opponent"]["name"] == "Fixture Team 2"
    starters = {row["name"]: row for row in block["lineup"]}
    assert starters["Josh Allen"]["slot"] == "QB"
    assert block["moves"] and all({"name", "from_slot", "to_slot"} <= set(move) for move in block["moves"])
    for suggestion in block["suggested_proposals"]:
        assert suggestion["policy"]["allowed"] is True and suggestion["payload"]["moves"]
    assert stored(home) == []


def test_lineup_takes_the_clock_as_text_or_as_an_instant(home: Path) -> None:
    assert views.lineup(now=NOW, opponent_team_id=2) == views.lineup(as_of=AS_OF, opponent_team_id=2)


def test_lineup_without_a_captured_schedule_says_why_or_plans_from_an_injected_one(home: Path) -> None:
    shutil.rmtree(home / "cache")
    (block,) = views.lineup(as_of=AS_OF)["leagues"]
    assert block["ok"] is False and "pro schedule" in block["error"]
    (planned,) = jsonable(views.lineup(as_of=AS_OF, schedule_path=PRO_SCHEDULE))["leagues"]
    assert planned["ok"] is True and planned["scoring_period"] == 4


def test_lineup_for_an_unsynced_league_is_a_block_not_an_error(unsynced: Path) -> None:
    (block,) = views.lineup(as_of=AS_OF)["leagues"]
    assert block == {"league": "nfl", "ok": False, "error": "not synced; run fm sync"}


# --- waivers ----------------------------------------------------------------------------------------------------------


def test_waivers_is_a_ranking_that_stores_nothing(home: Path) -> None:
    (block,) = jsonable(views.waivers(as_of=AS_OF, top=3))["leagues"]
    assert block["ok"] is True and block["wire"]["size"] > 0
    claim = block["moves"][0]
    assert claim["kind"] == "waiver" and claim["add"]["name"] == "Jaylen Warren"
    assert claim["policy"]["allowed"] is True and claim["payload"]["add_espn_id"] == claim["add"]["espn_id"]
    assert len(block["ranked"]) == 3 and block["ranked_total"] >= 3
    assert all("policy" not in move for move in block["ranked"])  # verdicts only for the planned moves
    assert block["replacement_levels"]
    (none_ranked,) = views.waivers(as_of=AS_OF, top=0, max_moves=1)["leagues"]
    assert none_ranked["ranked"] == [] and none_ranked["ranked_total"] == block["ranked_total"]
    assert len(none_ranked["moves"]) <= 1
    assert stored(home) == []


# --- trades -----------------------------------------------------------------------------------------------------------


def test_trade_eval_judges_a_deal_and_returns_the_payload(home: Path) -> None:
    out = jsonable(views.trade_eval(DEAL, as_of=AS_OF, runs=100))
    assert out["league"] == "nfl" and out["with_team"]["team_id"] == 2
    assert [p["name"] for p in out["give"]] == ["Tyler Allgeier"]
    assert [p["name"] for p in out["get"]] == ["Saquon Barkley"]
    assert out["recommendation"] in {"accept", "decline", "counter"}
    assert out["simulated"] is False and any("mMatchup" in warning for warning in out["warnings"])
    assert 0 < out["acceptance"]["p_accept"] < 1
    assert out["proposal_payload"] == {"other_team_id": 2, "give_espn_ids": [ALLEGIER], "get_espn_ids": [BARKLEY]}
    assert views.trade_eval(DEAL, "nfl", now=NOW, runs=100) == out
    assert stored(home) == []


def test_trade_eval_reads_an_injected_matchup_schedule_instead_of_the_cache(home: Path) -> None:
    out = jsonable(views.trade_eval(DEAL, as_of=AS_OF, runs=100, matchups_path=MATCHUPS, schedule_path=PRO_SCHEDULE))
    assert not any("no mMatchup schedule is captured" in warning for warning in out["warnings"])
    assert any("simulated" in warning for warning in out["warnings"])  # the schedule is read; the season is judged


@pytest.mark.parametrize(
    "deal", ["trade Allgeier for Barkley", "give Nobody Atall get Saquon Barkley", "give Josh Allen get Josh Allen"]
)
def test_trade_eval_reports_a_deal_it_cannot_read_as_a_view_error(home: Path, deal: str) -> None:
    with pytest.raises(ViewError, match="^nfl: "):
        views.trade_eval(deal, as_of=AS_OF)


def test_trade_eval_needs_a_synced_league(unsynced: Path) -> None:
    with pytest.raises(ViewError, match="nfl: not synced; run fm sync"):
        views.trade_eval(DEAL, as_of=AS_OF)


# --- the proposal queue -----------------------------------------------------------------------------------------------


def _store_add_drop(home: Path, *, add: int, rationale: str) -> int:
    with Store.open(home / "state.db") as store:
        row = store.leagues.by_key("nfl")
        assert row is not None
        made = propose(
            store,
            load_config(),
            row,
            "add_drop",
            AddDropPayload(add_espn_id=add),
            created_by="test",
            rationale=rationale,
            now=NOW,
        )
        assert made.row_id is not None
        return made.row_id


def test_list_proposals_shows_the_queue_newest_first_without_the_execution_token(home: Path) -> None:
    assert jsonable(views.list_proposals()) == {"count": 0, "proposals": []}
    first = _store_add_drop(home, add=DOWDLE, rationale="first")
    second = _store_add_drop(home, add=BARKLEY, rationale="second")
    with Store.open(home / "state.db") as store:  # a decided proposal carries a token; it must not be shown
        token = approve(store, first, decided_by="test", now=NOW).execution_token
    assert token is not None
    listed = jsonable(views.list_proposals("nfl"))
    assert listed["count"] == 2 and [item["id"] for item in listed["proposals"]] == [second, first]
    assert token not in json.dumps(listed) and "execution_token" not in json.dumps(listed)
    (newest,) = views.list_proposals(limit=1)["proposals"]
    assert newest["id"] == second and newest["league"] == "nfl" and newest["rationale"] == "second"
    assert views.list_proposals(limit=1)["count"] == 2  # the count is before the limit
    approved = next(item for item in listed["proposals"] if item["id"] == first)
    assert approved["status"] == "approved" and approved["decided_by"] == "test" and approved["decided_at"]
    assert views.list_proposals(include_closed=True)["count"] == 2
    assert {item["id"] for item in views.status()["leagues"][0]["open_proposals"]} == {first, second}


def test_list_proposals_skips_an_unsynced_league(unsynced: Path) -> None:
    assert views.list_proposals() == {"count": 0, "proposals": []}


def test_proposal_dict_summarizes_the_payload_and_falls_back_to_its_json(home: Path) -> None:
    with Store.open(home / "state.db") as store:
        row = store.leagues.by_key("nfl")
        assert row is not None and row.row_id is not None
        bad = store.proposals.insert(
            ProposalRow(
                league_id=row.row_id,
                kind="add_drop",
                policy="approve",
                payload={"zzz": 1, "aaa": 2},
                created_by="test",
                created_at=NOW,
                execution_token="secret-token",
            )
        )
    good_id = _store_add_drop(home, add=DOWDLE, rationale="r")
    with Store.open(home / "state.db") as store:
        good = store.proposals.get(good_id)
        assert good is not None
    out = jsonable(views.proposal_dict(bad))
    assert out["summary"] == '{"aaa": 2, "zzz": 1}' and out["league"] is None
    assert "secret-token" not in json.dumps(out)
    assert set(out) == {
        "id",
        "league",
        "kind",
        "status",
        "policy",
        "scoring_period_id",
        "payload",
        "summary",
        "rationale",
        "deadline",
        "created_by",
        "created_at",
        "decided_by",
        "decided_at",
    }
    assert str(DOWDLE) in views.proposal_dict(good, "nfl")["summary"]
    assert views.proposal_dict(good, "nfl")["created_at"] == NOW.isoformat()


def test_verdict_dict_lists_every_reason(home: Path) -> None:
    with Store.open(home / "state.db") as store:
        row = store.leagues.by_key("nfl")
        assert row is not None
        verdict = evaluate(
            store, load_config(), row, "add_drop", AddDropPayload(add_espn_id=DOWDLE, drop_espn_id=ALLEN), now=NOW
        )
    out = jsonable(views.verdict_dict(verdict))
    assert out["allowed"] is False and out["setting"] == "approve"
    assert any("Josh Allen" in reason and "untouchable" in reason for reason in out["reasons"])


# --- the MCP tools are thin wrappers ----------------------------------------------------------------------------------


def test_each_mcp_read_tool_returns_exactly_its_view(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastmcp import Client  # the test may load the MCP stack; fm.views may not

    import fm.mcp_server as server_module

    monkeypatch.setattr(server_module, "_now", lambda: NOW)
    calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        ("status", {}, views.status()),
        ("lineup", {"as_of": AS_OF, "opponent_team_id": 2}, views.lineup(as_of=AS_OF, opponent_team_id=2)),
        ("lineup", {}, views.lineup(now=NOW)),
        ("waivers", {"as_of": AS_OF, "top": 3}, views.waivers(as_of=AS_OF, top=3)),
        ("trade_eval", {"deal": DEAL, "as_of": AS_OF, "runs": 100}, views.trade_eval(DEAL, as_of=AS_OF, runs=100)),
        ("list_proposals", {"include_closed": True}, views.list_proposals(include_closed=True)),
    ]

    async def go() -> list[Any]:
        async with Client(server_module.build_server()) as client:
            return [(await client.call_tool(tool, arguments)).structured_content for tool, arguments, _ in calls]

    for (tool, _, expected), got in zip(calls, asyncio.run(go()), strict=True):
        assert got == expected, tool
