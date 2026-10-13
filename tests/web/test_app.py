"""The dashboard's JSON API (ROADMAP #54) through Starlette's ``TestClient``, on a private copy of the fixture home.

Every GET answers 200 with JSON. Approve and reject record the decision with ``decided_by="dashboard"`` and execute
nothing. The kill switch, the reports and the settings round-trip, and a settings edit goes through
:mod:`fm.config_edit` (validated first, the old file kept as ``config.toml.bak``). No response carries a value from
``.env``. The guard (Host, token, Origin) has its own tests in ``test_security.py``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pytest
from starlette.testclient import TestClient

import fm.web.app as web_app
from fm.config import load_config
from fm.proposals import AddDropPayload, pause_state, propose
from fm.store import Store

AS_OF = "2026-10-04T15:00Z"  # Sunday 11 a.m. ET of week 4, when the fixture home was synced (tests/test_views.py)
NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
DOWDLE, BARKLEY = 4038815, 3929630
REPORT_NAME = "weekly-2026-10-04.md"
REPORT = """# Weekly report 2026-10-04

As of Sun 04 Oct 15:00Z.

## nfl: Fixture League (PPR)

| Category | P(win) |
|---|---|
| PTS | 55% |

- one move
- another move
"""
SECRET_VALUES = ("sk-ant-test-SECRET-VALUE-1234", "123456:TELEGRAM-SECRET-TOKEN", "ntfy-secret-topic-name")


class Answer(Protocol):
    """What the ``TestClient`` returns (an ``httpx2.Response``, since starlette 1.x moved its test client to httpx2),
    narrowed to what these tests read."""

    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    @property
    def text(self) -> str: ...

    def json(self, **kwargs: Any) -> Any: ...


def ok(response: Answer) -> dict[str, Any]:
    """A 200 JSON object (and only JSON: the API never answers with HTML or text)."""
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert isinstance(body, dict)
    return body


def refused(response: Answer, status: int) -> str:
    assert response.status_code == status, response.text
    assert response.headers["content-type"].startswith("application/json")
    return str(response.json()["error"])


def post(client: TestClient, token: str, path: str, body: dict[str, Any] | None = None) -> Answer:
    return client.post(path, headers={"X-FM-Token": token}, json=body)


def store_add_drop(home: Path, *, add: int, rationale: str) -> int:
    """A proposed free-agent add in the fixture league, as the engine would store one."""
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


def stored(home: Path, proposal_id: int) -> tuple[str, str | None]:
    with Store.open(home / "state.db") as store:
        row = store.proposals.get(proposal_id)
        assert row is not None
        return row.status, row.decided_by


def write_report(home: Path, name: str = REPORT_NAME, text: str = REPORT) -> Path:
    directory = home / "cache" / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


# --- every GET --------------------------------------------------------------------------------------------------------

GETS = (
    "/api/overview",
    f"/api/overview?as_of={AS_OF}",
    "/api/proposals",
    "/api/proposals?all=1",
    "/api/proposals?all=0&league=nfl",
    f"/api/lineup/nfl?as_of={AS_OF}",
    "/api/lineup/nfl",
    f"/api/waivers/nfl?as_of={AS_OF}&top=3",
    "/api/waivers/nfl",
    "/api/reports",
    f"/api/reports/{REPORT_NAME}",
    "/api/health",
    "/api/settings",
)


@pytest.mark.parametrize("path", GETS)
def test_every_get_answers_200_with_json(client: TestClient, home: Path, path: str) -> None:
    write_report(home)
    store_add_drop(home, add=DOWDLE, rationale="a bench upgrade")
    body = ok(client.get(path))
    assert "error" not in body
    assert json.loads(json.dumps(body)) == body


def test_the_page_and_its_static_files_are_served(client: TestClient, token: str) -> None:
    page = client.get("/")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert '<section id="overview"' in page.text and token not in page.text  # the CSRF token never goes into HTML
    script = client.get("/static/app.js")
    assert script.status_code == 200 and script.headers["content-type"].startswith("text/javascript")
    assert "X-FM-Token" in script.text and "/api/session" in script.text
    style = client.get("/static/app.css")
    assert style.status_code == 200 and style.headers["content-type"].startswith("text/css")
    assert "prefers-color-scheme: dark" in style.text
    assert "no static file" in refused(client.get("/static/index.html"), 404)
    assert refused(client.get("/api/nothing-here"), 404)


# --- overview ---------------------------------------------------------------------------------------------------------


def test_the_overview_shows_each_league_its_deadlines_and_what_needs_attention(client: TestClient, home: Path) -> None:
    proposal = store_add_drop(home, add=DOWDLE, rationale="a bench upgrade")
    body = ok(client.get(f"/api/overview?as_of={AS_OF}"))
    assert body["as_of"] == NOW.isoformat() and body["config_error"] is None
    assert body["paused"] == {"on": False, "since": None, "reason": None, "detail": None}
    assert body["session"]["name"] == "session" and body["session"]["status"] in {"ok", "warn", "fail"}
    assert set(body["health"]) == {"fail", "warn"}
    (league,) = body["leagues"]
    assert league["league"] == "nfl" and league["synced"] is True and league["preset"] is None
    assert league["team"]["name"] == "Fixture Team 1" and (league["team"]["wins"], league["team"]["losses"]) == (2, 1)
    assert league["standing"] == {"playoff_seed": 4, "teams": 10}
    assert league["sync_age_s"] == 0 and league["synced_at"] == NOW.isoformat()
    kinds = [deadline["kind"] for deadline in league["deadlines"]]
    assert {"lineup_lock", "waiver_run", "period_turn"} <= set(kinds) and len(kinds) == len(set(kinds))
    times = [deadline["at"] for deadline in league["deadlines"]]
    assert times == sorted(times) and all(at >= NOW.isoformat() for at in times)
    (open_proposal,) = league["open_proposals"]
    assert open_proposal["id"] == proposal and open_proposal["status"] == "proposed"
    assert open_proposal["can_approve"] and "Dowdle" in open_proposal["moves"]


def test_the_overview_of_a_broken_config_still_answers_with_the_problem(client: TestClient, home: Path) -> None:
    (home / "config.toml").write_text("[[league]]\nkey = 'nfl'\n", encoding="utf-8")
    body = ok(client.get("/api/overview"))
    assert body["config_error"] and "config.toml" in body["config_error"] and body["leagues"] == []
    assert ok(client.get("/api/settings"))["config_error"]
    assert refused(client.get("/api/proposals"), 400)
    assert ok(client.get("/api/health"))["has_failure"] is True


# --- proposals --------------------------------------------------------------------------------------------------------


def test_approve_records_the_dashboards_decision_and_executes_nothing(
    client: TestClient, home: Path, token: str
) -> None:
    proposal = store_add_drop(home, add=DOWDLE, rationale="a bench upgrade")
    body = ok(post(client, token, f"/api/proposals/{proposal}/approve"))
    decided = body["proposal"]
    assert decided["id"] == proposal and decided["status"] == "approved" and decided["decided_by"] == "dashboard"
    assert decided["can_approve"] is False and decided["can_reject"] is True  # an approval can still be withdrawn
    assert "execution_token" not in json.dumps(body)
    assert stored(home, proposal) == ("approved", "dashboard")  # recorded, not executing: fm run's tick acts on it
    with Store.open(home / "state.db") as store:
        row = store.proposals.get(proposal)
        assert row is not None and row.execution_token  # issued, and kept in the store


def test_reject_records_the_dashboards_decision_and_withdraws_an_approval(
    client: TestClient, home: Path, token: str
) -> None:
    first = store_add_drop(home, add=DOWDLE, rationale="first")
    second = store_add_drop(home, add=BARKLEY, rationale="second")
    rejected = ok(post(client, token, f"/api/proposals/{first}/reject"))["proposal"]
    assert rejected["status"] == "rejected" and rejected["decided_by"] == "dashboard"
    assert stored(home, first) == ("rejected", "dashboard")
    ok(post(client, token, f"/api/proposals/{second}/approve"))
    assert ok(post(client, token, f"/api/proposals/{second}/reject"))["proposal"]["status"] == "rejected"
    assert stored(home, second) == ("rejected", "dashboard")


def test_a_decision_the_status_does_not_allow_is_409_and_an_unknown_proposal_404(
    client: TestClient, home: Path, token: str
) -> None:
    proposal = store_add_drop(home, add=DOWDLE, rationale="a bench upgrade")
    ok(post(client, token, f"/api/proposals/{proposal}/reject"))
    assert "rejected" in refused(post(client, token, f"/api/proposals/{proposal}/approve"), 409)
    assert stored(home, proposal) == ("rejected", "dashboard")
    assert refused(post(client, token, "/api/proposals/9999/approve"), 404) == "no proposal #9999"
    wrong_method = client.get(f"/api/proposals/{proposal}/approve")
    assert refused(wrong_method, 405) and wrong_method.headers["allow"] == "POST"


def test_the_proposal_list_says_what_each_proposal_still_allows(client: TestClient, home: Path, token: str) -> None:
    first = store_add_drop(home, add=DOWDLE, rationale="first")
    second = store_add_drop(home, add=BARKLEY, rationale="second")
    ok(post(client, token, f"/api/proposals/{first}/reject"))
    listed = ok(client.get("/api/proposals"))
    assert [item["id"] for item in listed["proposals"]] == [second]  # open only by default
    everything = ok(client.get("/api/proposals?all=1"))
    by_id = {item["id"]: item for item in everything["proposals"]}
    assert set(by_id) == {first, second} and everything["count"] == 2
    assert (by_id[first]["can_approve"], by_id[first]["can_reject"]) == (False, False)
    assert (by_id[second]["can_approve"], by_id[second]["can_reject"]) == (True, True)
    assert by_id[second]["label"] and "Barkley" in by_id[second]["moves"] and by_id[second]["rationale"] == "second"
    assert refused(client.get("/api/proposals?all=maybe"), 400)
    assert "no league" in refused(client.get("/api/proposals?league=nba"), 400)


# --- the kill switch --------------------------------------------------------------------------------------------------


def test_pause_and_resume_throw_and_lift_the_kill_switch(client: TestClient, home: Path, token: str) -> None:
    paused = ok(post(client, token, "/api/pause", {"reason": "on vacation"}))["paused"]
    assert paused["on"] is True and paused["reason"] == "on vacation" and "on vacation" in paused["detail"]
    state = pause_state()
    assert state is not None and state.reason == "on vacation"
    assert ok(client.get("/api/overview"))["paused"]["on"] is True
    assert ok(post(client, token, "/api/resume"))["paused"]["on"] is False
    assert pause_state() is None
    assert ok(post(client, token, "/api/pause"))["paused"]["reason"] is None  # no body, no reason
    ok(post(client, token, "/api/resume"))


def test_a_pause_body_that_is_not_a_json_object_with_a_text_reason_is_400(client: TestClient, token: str) -> None:
    assert "text" in refused(post(client, token, "/api/pause", {"reason": 3}), 400)
    headers = {"X-FM-Token": token, "Content-Type": "application/json"}
    assert "not JSON" in refused(client.post("/api/pause", headers=headers, content=b"{nope"), 400)
    assert "object" in refused(client.post("/api/pause", headers=headers, content=b"[1, 2]"), 400)
    too_big = json.dumps({"reason": "x" * (web_app.MAX_BODY_BYTES + 1)}).encode()
    assert refused(client.post("/api/pause", headers=headers, content=too_big), 413)
    assert pause_state() is None


# --- lineup and waivers -----------------------------------------------------------------------------------------------


def test_lineup_and_waivers_are_the_views_plans_and_store_nothing(client: TestClient, home: Path) -> None:
    lineup = ok(client.get(f"/api/lineup/nfl?as_of={AS_OF}"))
    (block,) = lineup["leagues"]
    assert block["ok"] is True and block["scoring_period"] == 4 and block["lineup"]
    waivers = ok(client.get(f"/api/waivers/nfl?as_of={AS_OF}&top=3"))
    (claims,) = waivers["leagues"]
    assert claims["ok"] is True and claims["moves"][0]["add"]["name"] == "Jaylen Warren" and len(claims["ranked"]) <= 3
    with Store.open(home / "state.db") as store:
        assert store.proposals.find() == []


def test_lineup_and_waivers_refuse_an_unknown_league_or_a_bad_argument(client: TestClient) -> None:
    assert "known: nfl" in refused(client.get("/api/lineup/nba"), 404)
    assert refused(client.get("/api/waivers/nba"), 404)
    assert "ISO 8601" in refused(client.get("/api/lineup/nfl?as_of=yesterday"), 400)
    assert "whole number" in refused(client.get("/api/waivers/nfl?top=lots"), 400)
    assert "top" in refused(client.get("/api/waivers/nfl?top=-1"), 400)


# --- reports ----------------------------------------------------------------------------------------------------------


def test_reports_are_listed_newest_first_and_served_as_json_markdown(client: TestClient, home: Path) -> None:
    assert ok(client.get("/api/reports"))["reports"] == []  # no reports dir yet: an empty list, nothing created
    assert not (home / "cache" / "reports").exists()
    older = write_report(home, "weekly-2026-09-27.md", "# Weekly report 2026-09-27\n")
    newer = write_report(home)
    os.utime(older, (1_790_000_000, 1_790_000_000))
    os.utime(newer, (1_791_000_000, 1_791_000_000))
    (home / "cache" / "reports" / "odds-nfl.json").write_text("[]", encoding="utf-8")  # the odds sidecar
    listed = ok(client.get("/api/reports"))
    assert [item["name"] for item in listed["reports"]] == [REPORT_NAME, "weekly-2026-09-27.md"]
    assert listed["reports"][0]["size"] == newer.stat().st_size
    assert ok(client.get(f"/api/reports/{REPORT_NAME}")) == {"name": REPORT_NAME, "markdown": REPORT}


@pytest.mark.parametrize(
    "name", ["odds-nfl.json", "missing.md", "..%2Fconfig.toml", "..%5Cconfig.toml", ".hidden.md", "%2e%2e.md"]
)
def test_a_report_name_that_is_not_a_listed_markdown_file_is_404(client: TestClient, home: Path, name: str) -> None:
    write_report(home)
    (home / "cache" / "reports" / "odds-nfl.json").write_text("[]", encoding="utf-8")
    response = client.get(f"/api/reports/{name}")
    assert response.status_code == 404 and "markdown" not in response.json()


# --- health -----------------------------------------------------------------------------------------------------------


def test_health_is_run_checks_offline_without_launching_a_browser(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    real: Callable[..., Any] = web_app.run_checks

    def spy(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(web_app, "run_checks", spy)
    body = ok(client.get("/api/health"))
    names = [check["name"] for check in body["checks"]]
    assert names[:5] == ["config", "session", "browser", "notify", "tick"] and "sync:nfl" in names
    assert all(set(check) == {"name", "status", "detail", "fix"} for check in body["checks"])
    assert isinstance(body["has_failure"], bool)
    ok(client.get("/api/overview"))
    assert len(calls) == 2
    assert all(not call.get("launch_browser", False) and not call.get("online", False) for call in calls)


# --- settings ---------------------------------------------------------------------------------------------------------


def test_settings_show_presets_overrides_the_channel_and_only_whether_keys_are_set(client: TestClient) -> None:
    body = ok(client.get("/api/settings"))
    assert [preset["name"] for preset in body["presets"]] == ["hands_off", "balanced", "ask_first"]
    assert all(preset["description"] for preset in body["presets"])
    assert body["channels"] == ["telegram", "ntfy", "none"] and body["notify"] == {"channel": "telegram"}
    (league,) = body["leagues"]
    assert league["league"] == "nfl" and league["preset"] is None and league["overrides"] == []
    assert league["trade_finder_enabled"] is False and league["policy"]["lineup"] == "approve"
    assert set(body["secrets"].values()) == {False}
    assert body["missing_secrets"] == ["ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]


def test_a_settings_post_changes_the_preset_through_config_edit(
    client: TestClient, home: Path, token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = home / "config.toml"
    original = config.read_bytes()
    edits: list[tuple[str, dict[str, object]]] = []
    real = web_app.set_values

    def spy(path: Path, league: str, values: dict[str, object], **kwargs: Any) -> Any:
        edits.append((league, dict(values)))
        return real(path, league, values, **kwargs)

    monkeypatch.setattr(web_app, "set_values", spy)
    body = ok(post(client, token, "/api/settings", {"league": "nfl", "preset": "hands_off"}))
    assert edits == [("nfl", {"preset": "hands_off"})]
    (league,) = body["leagues"]
    assert league["preset"] == "hands_off" and league["overrides"] == ["policy.lineup"]
    assert league["trade_finder_enabled"] is True  # the preset fills what the file leaves unset
    text = config.read_text(encoding="utf-8")
    assert 'preset = "hands_off"' in text and text.startswith("# The fixture home's config.toml")  # comments kept
    assert (home / "config.toml.bak").read_bytes() == original  # config_edit's backup of the file it replaced
    assert load_config().leagues[0].preset == "hands_off"

    reset = ok(post(client, token, "/api/settings", {"league": "nfl", "reset_overrides": True}))
    assert edits[-1] == ("nfl", {"policy.lineup": None})
    assert reset["leagues"][0]["overrides"] == [] and load_config().leagues[0].policy.lineup == "auto"

    ok(post(client, token, "/api/settings", {"league": "nfl", "preset": "ask_first", "reset_overrides": True}))
    assert load_config().leagues[0].preset == "ask_first"


def test_a_settings_post_toggles_the_trade_finder_and_sets_the_channel(
    client: TestClient, home: Path, token: str
) -> None:
    body = ok(post(client, token, "/api/settings", {"league": "nfl", "trade_finder_enabled": True}))
    assert body["leagues"][0]["trade_finder_enabled"] is True
    assert load_config().leagues[0].trade_finder.enabled is True
    body = ok(post(client, token, "/api/settings", {"notify_channel": "none"}))
    assert body["notify"] == {"channel": "none"} and body["missing_secrets"] == ["ANTHROPIC_API_KEY"]
    assert load_config().notify.channel == "none"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"league": "nfl", "preset": "yolo"}, "preset must be one of"),
        ({"league": "nfl", "trade_finder_enabled": "yes"}, "true or false"),
        ({"league": "nfl", "reset_overrides": "please"}, "true or false"),
        ({"league": "nfl", "reset_overrides": True}, "has no preset"),
        ({"league": "nba", "preset": "balanced"}, "no league 'nba'"),
        ({"preset": "balanced"}, "league must name"),
        ({"notify_channel": "sms"}, "notify_channel must be one of"),
        ({"league": "nfl"}, "nothing to change"),
        ({}, "nothing to change"),
        ({"league": "nfl", "policy.lineup": "auto"}, "unknown settings"),
    ],
)
def test_a_settings_post_the_config_would_not_take_is_400_and_leaves_the_file_alone(
    client: TestClient, home: Path, token: str, change: dict[str, Any], message: str
) -> None:
    config = home / "config.toml"
    original = config.read_bytes()
    assert message in refused(post(client, token, "/api/settings", change), 400)
    assert config.read_bytes() == original and not (home / "config.toml.bak").exists()


# --- no secrets -------------------------------------------------------------------------------------------------------


def test_no_response_carries_a_value_from_env(client: TestClient, home: Path, token: str) -> None:
    anthropic, telegram, topic = SECRET_VALUES
    (home / ".env").write_text(
        f"ANTHROPIC_API_KEY={anthropic}\nTELEGRAM_BOT_TOKEN={telegram}\nTELEGRAM_CHAT_ID=42\nNTFY_TOPIC={topic}\n",
        encoding="utf-8",
    )
    write_report(home)
    store_add_drop(home, add=DOWDLE, rationale="a bench upgrade")
    texts = [client.get(path).text for path in (*GETS, "/")]
    texts.append(post(client, token, "/api/settings", {"league": "nfl", "preset": "balanced"}).text)
    assert not any(secret in text for secret in SECRET_VALUES for text in texts)
    secrets = ok(client.get("/api/settings"))["secrets"]
    assert secrets["ANTHROPIC_API_KEY"] is True and secrets["TELEGRAM_CHAT_ID"] is True
    assert secrets["NTFY_TOPIC"] is True and secrets["ODDS_API_KEY"] is False
