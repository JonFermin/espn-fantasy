"""``fm.health.run_checks`` and ``fm doctor`` (ROADMAP #50).

Every test runs against a private copy of ``tests/fixtures/home`` (its league is synced at ``SYNCED``) with the
clock pinned, the session loader, browser finder and ESPN client injected, and the online pings served by
``httpx.MockTransport``. No browser starts (the profile check refuses first, and ``conftest`` would fail a launch) and
nothing leaves the machine. The CLI is exercised on a private root as the other command tests do.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import typer
from typer.testing import CliRunner

import fm.health as health_module
from fm import paths
from fm.browser.session import BrowserLaunchError
from fm.commands import doctor
from fm.config import League, load_config
from fm.espn.auth import EspnSession, NotLoggedInError
from fm.espn.client import EspnClient
from fm.health import (
    AUTOSTART_FIX,
    DEADLINE_SOON,
    NEWS_LATE_FIX,
    NEWS_TRIAGE_WINDOW,
    NO_STATE,
    SYNC_FAIL_AFTER,
    SYNC_WARN_AFTER,
    TICK_FAIL_AFTER,
    TICK_LATE_FIX,
    TICK_STATE_FILE,
    TICK_WARN_AFTER,
    Check,
    CheckStatus,
    find_browsers,
    has_failure,
    job_installed,
    run_checks,
)
from fm.jobs.advice import advisor_off
from fm.jobs.runlock import RunLock
from fm.jobs.scheduler_base import Runner, RunResult, SchedulerError
from fm.jobs.scheduler_windows import WINDOWS_AUTOSTART
from fm.proposals import pause
from fm.store import LlmUsageRow, NewsItemRow, ProposalRow, Sport, Store

HOME = Path(__file__).resolve().parent / "fixtures" / "home"
SYNCED = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
"""When the fixture home's league was synced (tests/fixtures/home/build.py)."""
NOW = SYNCED + timedelta(minutes=20)
SESSION = EspnSession(espn_s2="s2-value", swid="{ABCD}", expires_at=NOW + timedelta(days=30))
EDGE = Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe")
TELEGRAM_ENV = "ANTHROPIC_API_KEY=sk-ant-test\nTELEGRAM_BOT_TOKEN=123456:ABCDEFghijkl\nTELEGRAM_CHAT_ID=42\n"
NTFY_CONFIG = """
[[league]]
key = "nfl"
sport = "nfl"
espn_league_id = 1234567
season = 2026
team_id = 1

[llm]
daily_budget_usd = 1.0

[notify]
channel = "ntfy"
"""
NTFY_ENV = "ANTHROPIC_API_KEY=sk-ant-test\nNTFY_TOPIC=espn-fantasy-abc\nNTFY_REPLY_TOPIC=espn-fantasy-def\n"
OFFLINE_NAMES = [
    "config",
    "session",
    "browser",
    "notify",
    "tick",
    "autostart",
    "sync:nfl",
    "pause",
    "proposals",
    "deadlines",
    "llm",
    "news",
]

TASK_INFO = "TaskName: espn-fantasy-run\nState: Running\nExecute: C:\\uv\\uvw.exe\nLastTaskResult: 267009"
"""What ``Get-ScheduledTask`` prints for the registered logon task (scheduler_windows.render_get_task)."""
NO_TASK = (
    "Get-ScheduledTask : No MSFT_ScheduledTask objects found with property 'TaskName' equal to 'espn-fantasy-run'."
)

runner = CliRunner()


def scheduler(*, installed: bool = True, calls: list[Sequence[str]] | None = None) -> Runner:
    """A fake scheduler runner for the Windows logon-job backend: the task is registered, or it is not."""

    def run(argv: Sequence[str]) -> RunResult:
        if calls is not None:
            calls.append(argv)
        if installed:
            return RunResult(tuple(argv), 0, stdout=TASK_INFO)
        return RunResult(tuple(argv), 1, stderr=NO_TASK)

    return run


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the fixture home, with FM_CONFIG_DIR and FM_CACHE_DIR pointing into it."""
    copy = tmp_path / "home"
    shutil.copytree(HOME, copy, ignore=shutil.ignore_patterns("snapshots", "build.py", "README.md", "__pycache__"))
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(copy))
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(copy / "cache"))
    return copy


def checks(**overrides: Any) -> dict[str, Check]:
    """Run the checks as ``fm doctor`` does (allowed to load the session when no fresh tick record says it) with a
    pinned clock, a saved OK session and Edge installed unless overridden."""
    kwargs: dict[str, Any] = {
        "now": NOW,
        "launch_browser": True,
        "session_loader": lambda: SESSION,
        "browsers": lambda: {"msedge": EDGE},
        # Online collaborators default to fakes, so no test reaches the network by leaving one out.
        "client_factory": espn_factory(_status_ok),
        "http_client": telegram_client(),
        # The logon job is asked of a fake scheduler: no test runs PowerShell or launchctl.
        "autostart_backend": WINDOWS_AUTOSTART,
        "scheduler_runner": scheduler(),
    }
    kwargs.update(overrides)
    result = run_checks(**kwargs)
    names = [check.name for check in result]
    assert len(names) == len(set(names)), names
    return {check.name: check for check in result}


def make_profile(home: Path) -> None:
    profile = home / "browser-profile"
    profile.mkdir()
    (profile / "Default").mkdir()  # any entry inside counts as "a browser wrote here"


def write_tick(
    home: Path,
    at: datetime | str,
    *,
    session: dict[str, str | None] | None = None,
    news: dict[str, datetime | str] | None = None,
) -> None:
    stamp = at if isinstance(at, str) else at.isoformat()
    state: dict[str, Any] = {"last_tick": stamp, "runs": {}}
    if session is not None:
        state["session"] = session
    if news is not None:  # TickState.news: league key -> when its news was last polled
        state["news"] = {key: value if isinstance(value, str) else value.isoformat() for key, value in news.items()}
    (home / TICK_STATE_FILE).write_text(json.dumps(state), encoding="utf-8")


def session_record(
    status: str, *, checked: timedelta = timedelta(minutes=5), expires_at: datetime | None = None, detail: str = ""
) -> dict[str, str | None]:
    """A tick's session record (``fm.jobs.tick.session_record``) checked ``checked`` before NOW."""
    return {
        "status": status,
        "expires_at": None if expires_at is None else expires_at.isoformat(),
        "checked_at": (NOW - checked).isoformat(),
        "detail": detail or f"recorded verdict {status}",
    }


def refuse_launch() -> EspnSession:
    raise AssertionError("the session loader starts a browser; this check must not call it")


def make_ready(home: Path) -> None:
    """Everything an installation needs for every offline check to pass."""
    (home / ".env").write_text(TELEGRAM_ENV, encoding="utf-8")
    make_profile(home)
    write_tick(home, NOW - timedelta(minutes=5))


def add_proposal(home: Path, *, status: str, deadline: datetime | None, kind: str = "lineup") -> int:
    with Store.open(home / "state.db") as store:
        league = store.leagues.by_key("nfl")
        assert league is not None
        row = store.proposals.insert(
            ProposalRow(
                league_id=league.row_id,
                kind=kind,
                status=status,  # type: ignore[arg-type]  (the literal is checked by the model)
                policy="approve",
                payload={"kind": kind, "moves": []},
                deadline=deadline,
                created_by="test",
                created_at=NOW - timedelta(hours=1),
            )
        )
        return row.row_id


def add_spend(home: Path, cost: float, at: datetime) -> None:
    with Store.open(home / "state.db") as store:
        store.llm_usage.insert(LlmUsageRow(called_at=at, worker="explain", model="claude-opus-5-5", cost_usd=cost))


def add_news(home: Path, *, fetched: datetime, triaged: datetime | None = None, sport: Sport = "nfl") -> int:
    """A news item the tick's poll stored at ``fetched`` (its ``published_at`` too), triaged at ``triaged`` or not."""
    with Store.open(home / "state.db") as store:
        row = store.news.ingest(
            NewsItemRow(
                source="espn",
                external_id=f"{sport}-{fetched.isoformat()}-{triaged}",
                sport=sport,
                title="Fixture news item",
                published_at=fetched,
                fetched_at=fetched,
                triaged_at=triaged,
            )
        )
        assert row is not None
        return row.row_id


# --- the fixture home, offline ----------------------------------------------------------------------------------------


def test_a_fresh_fixture_home_says_what_is_missing_and_how_to_fix_it(home: Path) -> None:
    loads: list[str] = []
    factories: list[str] = []

    def loader() -> EspnSession:
        loads.append("called")
        return SESSION

    def factory(league: League, session: EspnSession) -> EspnClient:
        factories.append(league.key)
        raise AssertionError("offline checks must not build an ESPN client")

    found = checks(session_loader=loader, client_factory=factory)
    assert list(found) == OFFLINE_NAMES

    assert found["config"].status is CheckStatus.WARN  # valid, but no ANTHROPIC_API_KEY
    assert "ANTHROPIC_API_KEY" in found["config"].detail and found["config"].fix == "fm setup"
    assert found["session"] == Check(
        "session", CheckStatus.FAIL, "no browser profile yet: nothing is signed in to ESPN", fix="fm login"
    )
    assert loads == [] and factories == []  # no profile, so the loader (which would start a browser) never runs
    assert found["browser"].status is CheckStatus.OK and "Microsoft Edge" in found["browser"].detail
    assert found["notify"].status is CheckStatus.FAIL and found["notify"].fix == "fm notify setup"
    assert "TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID" in found["notify"].detail
    assert found["tick"].status is CheckStatus.FAIL and found["tick"].fix == "fm run"
    assert found["autostart"].status is CheckStatus.WARN and found["autostart"].fix == AUTOSTART_FIX
    assert "installed but fm run is not running" in found["autostart"].detail
    assert found["sync:nfl"].status is CheckStatus.OK and "20 min ago" in found["sync:nfl"].detail
    assert found["pause"] == Check("pause", CheckStatus.OK, "not paused")
    assert found["proposals"] == Check("proposals", CheckStatus.OK, "no open proposals")
    assert found["deadlines"].status is CheckStatus.OK
    assert found["llm"].status is CheckStatus.OK and "$0.00" in found["llm"].detail  # the home's budget is 0
    assert found["news"] == Check(
        "news",
        CheckStatus.OK,
        f"nfl news never polled; no item triaged in the last {NEWS_TRIAGE_WINDOW.days} d; 0 items waiting",
    )
    assert has_failure(found.values())


def test_a_ready_installation_passes_every_offline_check(home: Path) -> None:
    make_ready(home)
    # fm run is up, as autostart keeps it, so the session verdict comes from its last tick's record.
    write_tick(home, NOW - timedelta(minutes=5), session=session_record("ok", expires_at=SESSION.expires_at))
    with RunLock(command="fm run"):
        found = checks(session_loader=refuse_launch)
    assert {name: check.status for name, check in found.items()} == dict.fromkeys(OFFLINE_NAMES, CheckStatus.OK)
    assert all(check.fix is None for check in found.values())
    assert not has_failure(found.values())
    assert "1 league(s) (nfl)" in found["config"].detail
    assert found["notify"].detail == "telegram configured"
    assert "ESPN session OK until" in found["session"].detail


def test_checks_are_plain_json_data(home: Path) -> None:
    found = checks()
    encoded = json.loads(json.dumps([check.as_dict() for check in found.values()]))
    assert [item["name"] for item in encoded] == OFFLINE_NAMES
    assert all(set(item) == {"name", "status", "detail", "fix"} for item in encoded)
    assert {item["status"] for item in encoded} <= {"ok", "warn", "fail", "skip"}


# --- config -----------------------------------------------------------------------------------------------------------


def test_a_missing_config_fails_with_fm_setup_and_skips_what_needs_it(home: Path) -> None:
    (home / "config.toml").unlink()
    found = checks()
    assert found["config"].status is CheckStatus.FAIL and found["config"].fix == "fm setup"
    assert "not found" in found["config"].detail
    assert found["notify"].status is CheckStatus.SKIP
    assert found["sync"].status is CheckStatus.SKIP and "sync:nfl" not in found
    assert found["llm"].status is CheckStatus.SKIP
    assert found["session"].status is CheckStatus.FAIL  # independent of the config
    assert found["pause"].status is CheckStatus.OK


def test_an_invalid_config_fails_with_fm_config_check(home: Path) -> None:
    (home / "config.toml").write_text("[[league]]\nkey = 'nfl'\n", encoding="utf-8")
    found = checks()
    assert found["config"].status is CheckStatus.FAIL and found["config"].fix == "fm config check"


def test_online_checks_skip_without_a_config(home: Path) -> None:
    (home / "config.toml").unlink()
    found = checks(online=True)
    assert found["espn"].status is CheckStatus.SKIP and found["channel"].status is CheckStatus.SKIP


# --- session ----------------------------------------------------------------------------------------------------------


def expiring() -> EspnSession:
    return EspnSession(espn_s2="s2", swid="{ABCD}", expires_at=NOW + timedelta(days=3))


def expired() -> EspnSession:
    return EspnSession(espn_s2="s2", swid="{ABCD}", expires_at=NOW - timedelta(days=1))


def not_logged_in() -> EspnSession:
    raise NotLoggedInError("no ESPN session in the profile; run `fm login`")


def browser_held() -> EspnSession:
    raise BrowserLaunchError("msedge failed to launch with profile X: already in use")


@pytest.mark.parametrize(
    ("loader", "status", "fix", "fragment"),
    [
        (lambda: SESSION, CheckStatus.OK, None, "ESPN session OK until"),
        (expiring, CheckStatus.WARN, "fm login", "expires soon"),
        (expired, CheckStatus.FAIL, "fm login", "expired on"),
        (not_logged_in, CheckStatus.FAIL, "fm login", "no ESPN session in the profile"),
        (browser_held, CheckStatus.WARN, None, "another fm process may hold it"),
    ],
)
def test_the_session_verdict_follows_the_cookie_and_the_browser(
    home: Path, loader: Callable[[], EspnSession], status: CheckStatus, fix: str | None, fragment: str
) -> None:
    make_profile(home)
    found = checks(session_loader=loader)
    assert found["session"].status is status
    assert found["session"].fix == fix
    assert fragment in found["session"].detail


def test_the_profile_dir_is_injectable(home: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "profile"
    elsewhere.mkdir()
    (elsewhere / "Cookies").write_text("", encoding="utf-8")
    assert checks(profile_dir=elsewhere)["session"].status is CheckStatus.OK
    assert checks(profile_dir=tmp_path / "nowhere")["session"].status is CheckStatus.FAIL


# --- session: the tick's record first, a browser only for fm doctor with the lock free --------------------------------


@pytest.mark.parametrize("launch_browser", [False, True])
def test_a_fresh_tick_record_is_the_verdict_and_no_browser_starts(home: Path, launch_browser: bool) -> None:
    make_profile(home)
    write_tick(home, NOW - timedelta(minutes=5), session=session_record("ok", expires_at=NOW + timedelta(days=30)))
    check = checks(session_loader=refuse_launch, launch_browser=launch_browser)["session"]
    assert check.status is CheckStatus.OK and check.fix is None
    assert check.detail.startswith("ESPN session OK until") and "(the tick checked it 5 min ago)" in check.detail
    assert not (home / "run.lock").exists()  # no launch, so the run lock was not even taken


@pytest.mark.parametrize(
    ("record", "status", "fix", "fragment"),
    [
        (session_record("expiring", expires_at=NOW + timedelta(days=3)), CheckStatus.WARN, "fm login", "expires soon"),
        (session_record("expired", expires_at=NOW - timedelta(days=1)), CheckStatus.FAIL, "fm login", "expired on"),
        (session_record("missing", detail="no ESPN session in the profile"), CheckStatus.FAIL, "fm login", "no ESPN"),
        (session_record("unavailable", detail="could not read the ESPN session"), CheckStatus.WARN, None, "held"),
        # The tick judged it OK a few minutes ago, but the cookie has expired since: judged again against now.
        (session_record("ok", expires_at=NOW - timedelta(minutes=1)), CheckStatus.FAIL, "fm login", "expired on"),
    ],
)
def test_a_fresh_tick_record_gives_each_verdict(
    home: Path, record: dict[str, str | None], status: CheckStatus, fix: str | None, fragment: str
) -> None:
    make_profile(home)
    write_tick(home, NOW - timedelta(minutes=5), session=record)
    check = checks(session_loader=refuse_launch)["session"]
    assert (check.status, check.fix) == (status, fix)
    assert fragment in check.detail


def test_the_dashboard_never_launches_and_points_at_fm_doctor(home: Path) -> None:
    make_profile(home)
    found = checks(session_loader=refuse_launch, launch_browser=False)
    assert found["session"] == Check(
        "session",
        CheckStatus.WARN,
        "ESPN session not checked recently: no tick has recorded it yet; fm doctor checks it",
        fix="fm doctor",
    )
    write_tick(home, NOW - timedelta(minutes=5), session=session_record("ok", checked=timedelta(hours=2)))
    check = checks(session_loader=refuse_launch, launch_browser=False)["session"]
    assert check.status is CheckStatus.WARN and check.fix == "fm doctor"
    assert "the last tick checked it 2 h 0 min ago (ok)" in check.detail
    assert not (home / "run.lock").exists()


def test_fm_doctor_does_not_launch_beside_a_run_lock_holder(home: Path) -> None:
    make_profile(home)
    with RunLock(command="fm run"):
        check = checks(session_loader=refuse_launch)["session"]
        assert check.status is CheckStatus.WARN and check.fix == "fm login --check"
        assert f"fm run (pid {os.getpid()})" in check.detail and "no browser is started beside it" in check.detail
        # A fresh record still answers while the lock is held.
        write_tick(home, NOW - timedelta(minutes=5), session=session_record("ok"))
        assert checks(session_loader=refuse_launch)["session"].status is CheckStatus.OK


def test_fm_doctor_holds_the_run_lock_while_it_launches_and_releases_it(home: Path) -> None:
    make_profile(home)
    write_tick(home, NOW - timedelta(minutes=5), session=session_record("ok", checked=timedelta(hours=1)))
    seen: list[str | None] = []

    def loader() -> EspnSession:
        holder = RunLock().holder()
        seen.append(holder.command if holder is not None and RunLock().is_locked() else None)
        return SESSION

    check = checks(session_loader=loader)["session"]
    assert check.status is CheckStatus.OK and "ESPN session OK until" in check.detail
    assert seen == ["fm doctor"]  # a tick that fires now skips instead of launching beside it
    assert not RunLock().is_locked() and RunLock().holder() is None


def test_an_online_doctor_loads_the_cookies_even_with_a_fresh_record(home: Path) -> None:
    make_ready(home)
    write_tick(home, NOW - timedelta(minutes=5), session=session_record("ok"))
    loads: list[str] = []

    def loader() -> EspnSession:
        loads.append("loaded")
        return SESSION

    found = checks(online=True, session_loader=loader, client_factory=espn_factory(_status_ok))
    assert loads == ["loaded"] and found["espn:nfl"].status is CheckStatus.OK
    found = checks(online=True, session_loader=refuse_launch, launch_browser=False)
    assert found["espn:nfl"].status is CheckStatus.SKIP  # the dashboard never launches, so it has no cookies


# --- browser ----------------------------------------------------------------------------------------------------------


def test_find_browsers_looks_where_playwright_does_without_launching() -> None:
    environ = {"PROGRAMFILES": "C:/Program Files", "PROGRAMFILES(X86)": "C:/Program Files (x86)", "LOCALAPPDATA": ""}
    present = {
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    }
    found = find_browsers(platform="win32", environ=environ, is_file=lambda path: path in present)
    assert list(found) == ["msedge", "chrome"]  # Edge first, as fm.browser.session tries them
    assert found["msedge"] == Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
    assert find_browsers(platform="win32", environ=environ, is_file=lambda path: False) == {}

    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    assert find_browsers(platform="darwin", environ={}, is_file=lambda path: path == mac) == {"chrome": mac}
    linux = Path("/opt/microsoft/msedge/msedge")
    assert find_browsers(platform="linux", environ={}, is_file=lambda path: path == linux) == {"msedge": linux}


def test_no_browser_is_a_failure_naming_what_to_install(home: Path) -> None:
    found = checks(browsers=dict)
    assert found["browser"].status is CheckStatus.FAIL
    assert "Microsoft Edge or Google Chrome" in found["browser"].detail


# --- notify -----------------------------------------------------------------------------------------------------------


def test_the_ntfy_channel_needs_both_topics(home: Path) -> None:
    (home / "config.toml").write_text(NTFY_CONFIG, encoding="utf-8")
    (home / ".env").write_text("NTFY_TOPIC=espn-fantasy-abc\n", encoding="utf-8")
    found = checks()
    assert found["notify"].status is CheckStatus.FAIL and "NTFY_REPLY_TOPIC" in found["notify"].detail
    assert found["notify"].fix == "fm notify setup"
    (home / ".env").write_text(NTFY_ENV, encoding="utf-8")
    assert checks()["notify"] == Check("notify", CheckStatus.OK, "ntfy configured")


def test_no_phone_channel_is_a_warning_not_a_failure(home: Path) -> None:
    (home / "config.toml").write_text(NTFY_CONFIG.replace('channel = "ntfy"', 'channel = "none"'), encoding="utf-8")
    (home / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-test\n", encoding="utf-8")
    found = checks(online=True, http_client=httpx.Client(transport=httpx.MockTransport(_refuse_request)))
    assert found["notify"].status is CheckStatus.WARN and found["notify"].fix == "fm notify setup"
    assert 'notify.channel = "none"' in found["notify"].detail
    assert found["config"].status is CheckStatus.OK  # no channel secrets are missing
    assert found["channel"].status is CheckStatus.SKIP
    assert not has_failure([found[name] for name in ("config", "notify", "channel")])


def _refuse_request(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected request to {request.url}")


# --- tick -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("age", "status"),
    [
        (timedelta(minutes=5), CheckStatus.OK),
        (TICK_WARN_AFTER - timedelta(seconds=1), CheckStatus.OK),
        (TICK_WARN_AFTER, CheckStatus.WARN),
        (TICK_FAIL_AFTER - timedelta(seconds=1), CheckStatus.WARN),
        (TICK_FAIL_AFTER, CheckStatus.FAIL),
    ],
)
def test_the_last_tick_warns_after_30_min_and_fails_after_6_h(home: Path, age: timedelta, status: CheckStatus) -> None:
    write_tick(home, NOW - age)
    check = checks()["tick"]
    assert check.status is status
    assert check.fix == (None if status is CheckStatus.OK else "fm run")
    assert "last tick" in check.detail and "ago" in check.detail


def test_the_tick_fix_is_fm_run_only_when_nothing_holds_the_run_lock(home: Path) -> None:
    write_tick(home, NOW - TICK_FAIL_AFTER)
    assert checks()["tick"].fix == "fm run"
    assert not (home / "run.lock").exists()  # the probe never creates the lock file
    with RunLock(command="fm run"):
        check = checks()["tick"]
        assert check.status is CheckStatus.FAIL and check.fix == TICK_LATE_FIX
        assert TICK_LATE_FIX == "fm run is running but ticks are late: check logs/run.log"
        (home / TICK_STATE_FILE).unlink()
        assert checks()["tick"].fix == TICK_LATE_FIX  # fm run just started and has not ticked yet
    assert checks()["tick"].fix == "fm run"


def test_a_never_run_or_unreadable_tick_state_fails(home: Path) -> None:
    assert checks()["tick"].status is CheckStatus.FAIL
    (home / TICK_STATE_FILE).write_text("{not json", encoding="utf-8")
    assert checks()["tick"].status is CheckStatus.FAIL
    write_tick(home, "2026-10-04T15:00:00")  # naive: the tick itself ignores such a file
    assert checks()["tick"].status is CheckStatus.FAIL


def test_the_tick_state_file_name_matches_the_tick() -> None:
    from fm.jobs.tick import STATE_FILE

    assert TICK_STATE_FILE == STATE_FILE


# --- sync -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("age", "status"),
    [
        (timedelta(minutes=20), CheckStatus.OK),
        (SYNC_WARN_AFTER, CheckStatus.WARN),
        (SYNC_FAIL_AFTER - timedelta(minutes=1), CheckStatus.WARN),
        (SYNC_FAIL_AFTER, CheckStatus.FAIL),
        (timedelta(days=8), CheckStatus.FAIL),
    ],
)
def test_the_sync_age_per_league(home: Path, age: timedelta, status: CheckStatus) -> None:
    check = checks(now=SYNCED + age)["sync:nfl"]
    assert check.status is status
    assert check.fix == (None if status is CheckStatus.OK else "fm sync --league nfl")
    assert "Fixture League (PPR) synced" in check.detail and "2026-10-04 15:00Z" in check.detail


def test_a_league_that_was_never_synced_fails(home: Path) -> None:
    extra = '\n[[league]]\nkey = "nba"\nsport = "nba"\nespn_league_id = 99\nseason = 2027\nteam_id = 3\n'
    with (home / "config.toml").open("a", encoding="utf-8") as handle:
        handle.write(extra)
    found = checks()
    assert found["sync:nfl"].status is CheckStatus.OK
    assert found["sync:nba"] == Check(
        "sync:nba", CheckStatus.FAIL, "NBA league 99 has never been synced", fix="fm sync --league nba"
    )


def test_an_unopenable_store_fails_once_and_skips_its_checks(home: Path) -> None:
    def opener() -> Store:
        raise OSError("state.db is locked")

    found = checks(store_opener=opener)
    assert found["store"].status is CheckStatus.FAIL and "locked" in found["store"].detail
    assert {found[name].status for name in ("sync", "proposals", "deadlines", "llm")} == {CheckStatus.SKIP}
    assert found["pause"].status is CheckStatus.OK  # the pause marker is a file, not a table


def test_a_fresh_install_without_state_db_is_warned_and_never_creates_it(home: Path) -> None:
    (home / "state.db").unlink()

    def opener() -> Store:
        raise AssertionError("opening state.db would create and migrate it")

    found = checks(store_opener=opener)
    assert list(found) == OFFLINE_NAMES
    for name in ("sync:nfl", "proposals", "deadlines", "llm", "news"):
        assert found[name] == Check(name, CheckStatus.WARN, NO_STATE, fix="fm sync")
    assert "no state yet" in NO_STATE and found["pause"].status is CheckStatus.OK
    assert not (home / "state.db").exists()
    (home / "config.toml").unlink()
    found = checks(store_opener=opener)
    assert found["sync"] == Check("sync", CheckStatus.WARN, NO_STATE, fix="fm setup")
    assert not (home / "state.db").exists()


def test_fm_doctor_on_a_fresh_install_writes_nothing(home: Path) -> None:
    (home / "state.db").unlink()
    before = listing(home)
    result = runner.invoke(cli(), ["doctor", "--json"], catch_exceptions=False)
    by_name = {item["name"]: item for item in json.loads(result.output)}
    assert by_name["llm"]["status"] == "warn" and by_name["llm"]["detail"] == NO_STATE
    assert listing(home) == before  # no state.db, no run.lock


# --- pause, proposals, deadlines --------------------------------------------------------------------------------------


def test_the_kill_switch_is_a_warning_with_fm_resume(home: Path) -> None:
    pause("roster check", now=NOW - timedelta(hours=1))
    check = checks()["pause"]
    assert check.status is CheckStatus.WARN and check.fix == "fm resume"
    assert "paused since 2026-10-04 14:20 UTC (roster check)" in check.detail


def test_open_proposals_are_counted_and_near_deadlines_listed(home: Path) -> None:
    soon = add_proposal(home, status="proposed", deadline=NOW + timedelta(hours=1))
    later = add_proposal(home, status="approved", deadline=NOW + DEADLINE_SOON + timedelta(minutes=1), kind="waivers")
    add_proposal(home, status="rejected", deadline=NOW + timedelta(minutes=10))  # decided: not open
    found = checks()
    assert found["proposals"] == Check("proposals", CheckStatus.OK, "2 open: 1 proposed, 1 approved")
    assert found["deadlines"].status is CheckStatus.WARN and found["deadlines"].fix == "fm proposals list"
    assert found["deadlines"].detail == f"nfl #{soon} lineup (proposed) due in 1 h 0 min"
    assert f"#{later}" not in found["deadlines"].detail


def test_overdue_open_proposals_are_listed_first(home: Path) -> None:
    late = add_proposal(home, status="approved", deadline=NOW - timedelta(minutes=10), kind="waivers")
    soon = add_proposal(home, status="proposed", deadline=NOW + timedelta(minutes=30))
    detail = checks()["deadlines"].detail
    assert detail == f"nfl #{late} waivers (approved) overdue by 10 min; nfl #{soon} lineup (proposed) due in 30 min"


def test_no_deadline_within_two_hours_is_fine(home: Path) -> None:
    add_proposal(home, status="proposed", deadline=None)
    add_proposal(home, status="proposed", deadline=NOW + timedelta(days=1))
    found = checks()
    assert found["proposals"].detail == "2 open: 2 proposed"
    assert found["deadlines"] == Check("deadlines", CheckStatus.OK, "no open proposal is due within 2 h 0 min")


# --- llm --------------------------------------------------------------------------------------------------------------


def advisor_on(home: Path) -> None:
    """A $1.00 daily budget and the key in ``.env``: the tick would build the advisor."""
    (home / "config.toml").write_text(NTFY_CONFIG, encoding="utf-8")
    (home / ".env").write_text(NTFY_ENV, encoding="utf-8")


def test_todays_spend_counts_against_the_budget(home: Path) -> None:
    advisor_on(home)
    add_spend(home, 0.40, NOW - timedelta(hours=2))
    add_spend(home, 0.35, NOW - timedelta(days=1))  # yesterday: outside the window
    check = checks()["llm"]
    assert check == Check("llm", CheckStatus.OK, "advisor on: Claude spend since 00:00 UTC: $0.40 of $1.00")
    add_spend(home, 0.60, NOW - timedelta(minutes=1))
    check = checks()["llm"]
    assert check.status is CheckStatus.WARN and check.fix is None
    assert check.detail.startswith("advisor off until 00:00 UTC: today's Claude budget is spent ($1.00 of $1.00)")
    assert "refused" in check.detail and "template rationales" in check.detail


def test_a_zero_budget_means_the_advisor_is_off_not_broken(home: Path) -> None:
    check = checks()["llm"]
    assert check.status is CheckStatus.OK and "advisor is off" in check.detail
    assert check.detail == f"{advisor_off(load_config())}; $0.00 used today"


def test_a_missing_key_is_reported_as_the_tick_would_note_it(home: Path) -> None:
    """``llm`` says what ``fm.jobs.advice.advisor_off`` tells the tick, word for word; the WARN and the fix for the
    key stay with the ``config`` check, so one cause is not flagged twice."""
    (home / "config.toml").write_text(NTFY_CONFIG, encoding="utf-8")  # budget $1.00, but no .env: no key
    found = checks()
    reason = advisor_off(load_config())
    assert reason is not None and "ANTHROPIC_API_KEY is not set" in reason
    assert found["llm"] == Check("llm", CheckStatus.OK, f"{reason}; $0.00 used today")
    assert found["config"].status is CheckStatus.WARN and found["config"].fix == "fm setup"
    (home / ".env").write_text(NTFY_ENV, encoding="utf-8")
    found = checks()
    assert advisor_off(load_config()) is None
    assert found["llm"] == Check("llm", CheckStatus.OK, "advisor on: Claude spend since 00:00 UTC: $0.00 of $1.00")
    assert found["config"].status is CheckStatus.OK


# --- news -------------------------------------------------------------------------------------------------------------


def test_news_reports_the_last_poll_the_last_triage_and_what_waits(home: Path) -> None:
    advisor_on(home)
    polled = NOW - timedelta(minutes=12)
    write_tick(home, NOW - timedelta(minutes=5), news={"nfl": polled})
    add_news(home, fetched=NOW - timedelta(hours=3), triaged=NOW - timedelta(hours=2))
    add_news(home, fetched=polled, triaged=NOW - timedelta(minutes=11))
    add_news(home, fetched=NOW - timedelta(hours=1), sport="nba")  # no NBA league is configured: not the tick's
    check = checks()["news"]
    assert check == Check("news", CheckStatus.OK, "nfl news polled 12 min ago; last triage 11 min ago; 0 items waiting")


def test_items_the_last_poll_left_untriaged_warn_only_with_the_advisor_on(home: Path) -> None:
    """The tick triages everything untriaged right after each poll, so an item fetched at that poll and still waiting
    means the triage failed: a WARN pointing at the log. Without an advisor the items wait by design."""
    advisor_on(home)
    polled = NOW - timedelta(minutes=12)
    write_tick(home, NOW - timedelta(minutes=5), news={"nfl": polled})
    add_news(home, fetched=NOW - timedelta(hours=2), triaged=NOW - timedelta(hours=1))
    add_news(home, fetched=polled)
    check = checks()["news"]
    assert check.status is CheckStatus.WARN and check.fix == NEWS_LATE_FIX
    assert check.detail == (
        "nfl news polled 12 min ago; last triage 1 h 0 min ago; 1 item waiting; all of them fetched by the last poll "
        "or earlier and still untriaged with the advisor on"
    )
    add_spend(home, 1.00, NOW - timedelta(minutes=1))  # the budget is spent: the triage stopped on purpose
    check = checks()["news"]
    assert check == Check(
        "news",
        CheckStatus.OK,
        "nfl news polled 12 min ago; last triage 1 h 0 min ago; 1 item waiting (the advisor is off, so they wait)",
    )
    (home / ".env").unlink()  # no key: same
    assert checks()["news"].status is CheckStatus.OK and "the advisor is off" in checks()["news"].detail


def test_news_fetched_after_the_last_poll_is_not_late_yet(home: Path) -> None:
    advisor_on(home)
    write_tick(home, NOW - timedelta(minutes=5), news={"nfl": NOW - timedelta(minutes=40)})
    add_news(home, fetched=NOW - timedelta(minutes=2))  # stored by a later hand poll; the next tick triages it
    check = checks()["news"]
    assert check == Check(
        "news",
        CheckStatus.OK,
        f"nfl news polled 40 min ago; no item triaged in the last {NEWS_TRIAGE_WINDOW.days} d; 1 item waiting",
    )


@pytest.mark.parametrize("record", [{"nfl": "2026-10-04T15:00:00"}, {"nfl": "soon"}, {"nba": NOW}, "polled"])
def test_a_malformed_or_missing_news_record_reads_as_never_polled(home: Path, record: Any) -> None:
    state: dict[str, Any] = {"last_tick": NOW.isoformat(), "runs": {}, "news": record}
    (home / TICK_STATE_FILE).write_text(json.dumps(state, default=str), encoding="utf-8")
    assert checks()["news"].detail.startswith("nfl news never polled;")


# --- online -----------------------------------------------------------------------------------------------------------


def espn_factory(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[League, EspnSession], EspnClient]:
    def factory(league: League, session: EspnSession) -> EspnClient:
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return EspnClient.for_league(
            league, session, client=client, capture=False, max_attempts=1, min_interval_s=0, sleep=lambda s: None
        )

    return factory


def test_the_online_espn_read_reports_the_leagues_status(home: Path) -> None:
    make_ready(home)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"status": {"currentMatchupPeriod": 4, "latestScoringPeriod": 4}})

    found = checks(online=True, client_factory=espn_factory(handler), http_client=telegram_client())
    assert list(found) == [*OFFLINE_NAMES, "espn:nfl", "channel"]
    assert found["espn:nfl"] == Check(
        "espn:nfl", CheckStatus.OK, "ESPN answered mStatus: scoring period 4, matchup period 4"
    )
    (request,) = seen
    assert request.url.params.get_list("view") == ["mStatus"] and "1234567" in str(request.url)
    assert request.headers["Cookie"] == "espn_s2=s2-value; SWID={ABCD}"
    assert not list((home / "cache").rglob("*mStatus*"))  # no capture written


def test_the_online_espn_read_maps_a_refused_session_to_fm_login(home: Path) -> None:
    make_ready(home)
    found = checks(online=True, client_factory=espn_factory(lambda request: httpx.Response(401, json={})))
    assert found["espn:nfl"].status is CheckStatus.FAIL and found["espn:nfl"].fix == "fm login"
    found = checks(online=True, client_factory=espn_factory(lambda request: httpx.Response(503, text="down")))
    assert found["espn:nfl"].status is CheckStatus.FAIL and found["espn:nfl"].fix is None


def test_the_online_espn_read_is_skipped_without_a_usable_session(home: Path) -> None:
    (home / ".env").write_text(TELEGRAM_ENV, encoding="utf-8")
    calls: list[str] = []

    def factory(league: League, session: EspnSession) -> EspnClient:
        calls.append(league.key)
        raise AssertionError("no session, no read")

    found = checks(online=True, client_factory=factory)  # no profile at all
    assert found["espn:nfl"].status is CheckStatus.SKIP and calls == []
    make_profile(home)
    found = checks(online=True, client_factory=factory, session_loader=expired)
    assert found["espn:nfl"].status is CheckStatus.SKIP and calls == []


def telegram_client(*, username: str | None = "fm_test_bot", status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.telegram.org" and request.url.path.endswith("/getMe")
        if status != 200:
            return httpx.Response(status, json={"ok": False, "error_code": status, "description": "Unauthorized"})
        return httpx.Response(200, json={"ok": True, "result": {"id": 1, "is_bot": True, "username": username}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_the_channel_ping_calls_telegram_getme(home: Path) -> None:
    make_ready(home)
    found = checks(online=True, http_client=telegram_client(), client_factory=espn_factory(_status_ok))
    assert found["channel"] == Check("channel", CheckStatus.OK, "telegram bot @fm_test_bot answered getMe")
    found = checks(online=True, http_client=telegram_client(status=401), client_factory=espn_factory(_status_ok))
    assert found["channel"].status is CheckStatus.FAIL and found["channel"].fix == "fm notify setup"
    assert "Unauthorized" in found["channel"].detail


def test_an_unconfigured_channel_fails_the_ping_before_any_request(home: Path) -> None:
    make_profile(home)
    write_tick(home, NOW)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    found = checks(online=True, http_client=httpx.Client(transport=httpx.MockTransport(refuse)))
    assert found["channel"].status is CheckStatus.FAIL and found["channel"].fix == "fm notify setup"
    assert "TELEGRAM_BOT_TOKEN" in found["channel"].detail


def ntfy_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_the_channel_ping_reads_the_ntfy_health_endpoint(home: Path) -> None:
    (home / "config.toml").write_text(NTFY_CONFIG, encoding="utf-8")
    (home / ".env").write_text(NTFY_ENV, encoding="utf-8")
    make_profile(home)
    seen: list[httpx.Request] = []

    def healthy(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"healthy": True})

    found = checks(online=True, http_client=ntfy_client(healthy), client_factory=espn_factory(_status_ok))
    assert found["channel"] == Check("channel", CheckStatus.OK, "ntfy server https://ntfy.sh is healthy (HTTP 200)")
    (request,) = seen
    assert request.method == "GET" and str(request.url) == "https://ntfy.sh/v1/health"

    found = checks(
        online=True,
        http_client=ntfy_client(lambda r: httpx.Response(200, json={"healthy": False})),
        client_factory=espn_factory(_status_ok),
    )
    assert found["channel"].status is CheckStatus.FAIL and "unhealthy" in found["channel"].detail

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    found = checks(online=True, http_client=ntfy_client(down), client_factory=espn_factory(_status_ok))
    assert found["channel"].status is CheckStatus.FAIL and "unreachable" in found["channel"].detail


def _status_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"status": {"currentMatchupPeriod": 4, "latestScoringPeriod": 4}})


# --- autostart --------------------------------------------------------------------------------------------------------


def test_autostart_not_installed_warns_with_the_install_fix(home: Path) -> None:
    calls: list[Sequence[str]] = []
    check = checks(scheduler_runner=scheduler(installed=False, calls=calls))["autostart"]
    assert check.status is CheckStatus.WARN and check.fix == AUTOSTART_FIX == "fm autostart install"
    assert "fm run at logon (Windows Task Scheduler) is not installed" in check.detail
    assert "the run lock is free" in check.detail
    (argv,) = calls
    assert argv[0] == "powershell" and "Get-ScheduledTask -TaskName 'espn-fantasy-run'" in argv[-1]
    assert not (home / "run.lock").exists()  # the lock probe never creates the file


def test_autostart_installed_is_ok_only_while_fm_run_holds_the_run_lock(home: Path) -> None:
    check = checks()["autostart"]
    assert check.status is CheckStatus.WARN and check.fix == AUTOSTART_FIX
    assert "is installed but fm run is not running (the run lock is free)" in check.detail
    assert "logs/run.log" in check.detail
    with RunLock(command="fm run"):
        check = checks()["autostart"]
        assert check.status is CheckStatus.OK and check.fix is None
        assert check.detail.startswith("fm run at logon (Windows Task Scheduler) is installed; fm run (pid ")
        assert "holds the run lock" in check.detail
    with RunLock(command="fm tick"):  # a scheduled tick holds the lock, but nothing listens between ticks
        check = checks()["autostart"]
        assert check.status is CheckStatus.WARN and "fm run is not running (fm tick (pid " in check.detail


def test_autostart_without_a_runner_judges_from_the_run_lock_alone(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(argv: Sequence[str]) -> RunResult:
        raise AssertionError(f"only fm doctor may run the scheduler's query: {argv}")

    monkeypatch.setattr(health_module, "default_runner", refuse)
    check = checks(launch_browser=False, scheduler_runner=None)["autostart"]
    assert check.status is CheckStatus.WARN and check.fix == AUTOSTART_FIX
    assert check.detail == (
        "fm run is not running (the run lock is free); whether fm run at logon (Windows Task Scheduler) is installed "
        "is checked by fm doctor"
    )
    with RunLock(command="fm run"):
        check = checks(launch_browser=False, scheduler_runner=None)["autostart"]
        assert check.status is CheckStatus.OK and check.fix is None


def test_autostart_reports_a_scheduler_it_cannot_ask(home: Path) -> None:
    def broken(argv: Sequence[str]) -> RunResult:
        raise SchedulerError("powershell not found on PATH")

    check = checks(scheduler_runner=broken)["autostart"]
    assert check.status is CheckStatus.WARN and check.fix == "fm autostart show"
    assert "could not ask Windows Task Scheduler" in check.detail

    def failing(argv: Sequence[str]) -> RunResult:
        return RunResult(tuple(argv), 5, stderr="Access is denied.")

    assert checks(scheduler_runner=failing)["autostart"].fix == "fm autostart show"


def test_autostart_skips_on_a_platform_without_a_backend(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health_module, "AUTOSTART_BACKENDS", {})
    check = checks(autostart_backend=None)["autostart"]
    assert check.status is CheckStatus.SKIP and "supports Windows and macOS" in check.detail


@pytest.mark.parametrize(
    ("shown", "installed"),
    [
        (TASK_INFO, True),
        ("gui/501/local.espn-fantasy-run = {\n\tstate = running\n}", True),
        ("espn-fantasy-run: not installed (fm autostart install)\nEnv file: x", False),
        ("local.espn-fantasy-run: not installed (fm autostart install)\nPlist: x", False),
        ("espn-fantasy-tick: not installed (fm schedule install)\nWrapper: x", False),
        ("espn-fantasy-run: cannot query the scheduler (powershell not found)\nEnv file: x", None),
        ("local.espn-fantasy-run: cannot query launchd (no uid)\nPlist: x", None),
        ("espn-fantasy-run: error (5): Access is denied.", None),
        ("", None),
    ],
)
def test_job_installed_reads_every_backends_show_text(shown: str, installed: bool | None) -> None:
    assert job_installed(shown) is installed


# --- import weight ----------------------------------------------------------------------------------------------------

FORBIDDEN_MODULES = ("fm.executor", "fm.browser.flows", "fm.browser.transactions", "fm.jobs.tick", "fm.advisor")
FORBIDDEN_PACKAGES = ("fastmcp", "anthropic", "fm.commands")


def test_the_health_module_imports_neither_the_executor_nor_the_flows() -> None:
    tree = ast.parse(Path(health_module.__file__ or "").read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    assert {m for m in modules if m.startswith(FORBIDDEN_MODULES + FORBIDDEN_PACKAGES)} == set()
    assert "fm.store" in modules  # the parse found the real imports
    assert "fm.jobs.advice" in modules  # the advisor's on/off rule comes from the tick's module, not a copy


def test_importing_health_loads_no_executor_flow_or_sdk() -> None:
    code = (
        "import sys, fm.health; "
        f"print([n for n in sorted(sys.modules) if n.startswith({FORBIDDEN_MODULES + FORBIDDEN_PACKAGES!r})])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]", done.stdout


# --- fm doctor --------------------------------------------------------------------------------------------------------


def _root() -> None:
    pass


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    doctor.register(root)
    return root


def listing(home: Path) -> list[str]:
    return sorted(path.relative_to(home).as_posix() for path in home.rglob("*"))


def test_fm_doctor_json_on_the_fixture_home_is_complete_and_writes_nothing(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Sequence[str]] = []
    monkeypatch.setattr(health_module, "AUTOSTART_BACKENDS", {sys.platform: WINDOWS_AUTOSTART})
    monkeypatch.setattr(health_module, "default_runner", scheduler(installed=False, calls=calls))
    before = listing(home)
    result = runner.invoke(cli(), ["doctor", "--json"], catch_exceptions=False)
    assert result.exit_code == 1, result.output  # no profile, no tick: FAIL checks
    payload = json.loads(result.output)
    assert [item["name"] for item in payload] == OFFLINE_NAMES
    assert all(set(item) == {"name", "status", "detail", "fix"} for item in payload)
    by_name = {item["name"]: item for item in payload}
    assert by_name["session"] == {
        "name": "session",
        "status": "fail",
        "detail": "no browser profile yet: nothing is signed in to ESPN",
        "fix": "fm login",
    }
    assert by_name["tick"]["status"] == "fail" and by_name["tick"]["fix"] == "fm run"
    assert by_name["autostart"]["status"] == "warn" and by_name["autostart"]["fix"] == "fm autostart install"
    assert len(calls) == 1  # fm doctor asks the scheduler whether the logon job is registered
    assert listing(home) == before  # no profile, state, capture, cache or run lock created
    assert not (home / "browser-profile").exists()


def fake_checks(*checks: Check) -> Callable[..., list[Check]]:
    def run_checks(*, online: bool = False, launch_browser: bool = False) -> list[Check]:
        run_checks.online = online  # type: ignore[attr-defined]
        assert launch_browser, "fm doctor is the one caller allowed to load the session"
        return list(checks)

    return run_checks


def test_fm_doctor_prints_one_line_per_check_and_exits_1_on_a_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = fake_checks(
        Check("config", CheckStatus.OK, "config.toml valid:\n  1 league"),
        Check("session", CheckStatus.FAIL, "no browser profile yet", fix="fm login"),
        Check("tick", CheckStatus.WARN, "last tick 40 min ago", fix="fm run"),
        Check("espn:nfl", CheckStatus.SKIP, "no usable ESPN session"),
    )
    monkeypatch.setattr(doctor, "run_checks", fake)
    result = runner.invoke(cli(), ["doctor", "--online"], catch_exceptions=False)
    assert result.exit_code == 1, result.output
    assert result.output.splitlines() == [
        "OK    config    config.toml valid: 1 league",
        "FAIL  session   no browser profile yet  → fix: fm login",
        "WARN  tick      last tick 40 min ago  → fix: fm run",
        "SKIP  espn:nfl  no usable ESPN session",
    ]
    assert fake.online is True  # type: ignore[attr-defined]


def test_fm_doctor_exits_0_when_nothing_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = fake_checks(
        Check("config", CheckStatus.OK, "fine"), Check("pause", CheckStatus.WARN, "paused", fix="fm resume")
    )
    monkeypatch.setattr(doctor, "run_checks", fake)
    result = runner.invoke(cli(), ["doctor"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "WARN  pause   paused  → fix: fm resume" in result.output
    assert fake.online is False  # type: ignore[attr-defined]
    result = runner.invoke(cli(), ["doctor", "--json"], catch_exceptions=False)
    assert result.exit_code == 0 and json.loads(result.output)[1]["fix"] == "fm resume"


def test_the_subscription_backend_needs_no_key_and_says_so(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = NTFY_CONFIG.replace("daily_budget_usd = 1.0", 'daily_budget_usd = 1.0\nbackend = "claude_code"')
    (home / "config.toml").write_text(config, encoding="utf-8")
    (home / ".env").write_text(NTFY_ENV.replace("ANTHROPIC_API_KEY=sk-ant-test\n", ""), encoding="utf-8")
    monkeypatch.setattr("fm.jobs.advice.claude_cli_path", lambda: "/opt/claude")
    found = checks()
    assert found["config"].status is CheckStatus.OK
    assert found["llm"] == Check(
        "llm",
        CheckStatus.OK,
        "advisor on over the Claude subscription (claude CLI): API-price equivalent since 00:00 UTC: $0.00 of $1.00",
    )
    monkeypatch.setattr("fm.jobs.advice.claude_cli_path", lambda: None)
    assert "claude CLI was not found" in checks()["llm"].detail
