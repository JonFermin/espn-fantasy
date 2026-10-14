"""The plug-and-play journey (ROADMAP #56), offline, end to end: the commands a new user runs, in order.

1. ``fm setup --yes --channel none`` finds the league, picks the ``balanced`` preset and writes ``config.toml``.
2. ``fm config check`` accepts it.
3. ``fm doctor --json`` prints its checks as JSON.
4. One ``fm run`` iteration (:func:`fm.jobs.runner.run_forever` with the tick in process, ``--no-execute``) runs the
   real lineup decision and pushes its proposals to the phone.
5. The dashboard (:func:`fm.web.app.create_app`) signs in through its one-time ``/auth?t=`` link and approves one.
6. The next iteration executes it through :func:`fm.executor.execute` and the real ``set_lineup`` flow, with a fake
   runtime: the write lands in a fake transport and the re-read verifies it.
7. ``fm doctor --json`` again: with a tick recorded, nothing fails.

The world is the fixture home (``tests/fixtures/home``: the synthetic PPR league 1234567, team 1, synced on the Sunday
of week 4 at 15:00Z), copied into this test's private ``FM_CONFIG_DIR``/``FM_CACHE_DIR`` by the fake first sync and
never written to. Every external effect is a fake: the ESPN session and discovery, the scheduler jobs, ESPN's read
API (an ``httpx.MockTransport`` over the home's captures), the write transport, the phone and the clock (pinned to
15:35Z, inside the 1 p.m. kickoffs' lineup window; the dashboard's approval reads the wall clock through
``fm.proposals.queue.as_utc``, pinned too). The conftest network guard is on, and any real httpx request fails the
test, so nothing can reach ESPN; the only ESPN write is the one the fake transport records.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from fm import paths
from fm.browser.fakes import FakeEspnApi, FakeOpener, FakeTransport, fake_runtime, view_key
from fm.browser.flows import FlowRegistry, WriteRequest
from fm.browser.transactions import WEB_CLIENT_HEADERS
from fm.cli import app as fm_app
from fm.commands import doctor as doctor_cmd
from fm.commands import setup as setup_cmd
from fm.config import Config, League, load_config
from fm.espn.auth import EspnSession
from fm.espn.client import EspnClient
from fm.espn.discover import DiscoveredLeague, LeagueRef
from fm.executor import ExecutorOptions, RuntimeOpener
from fm.health import Check, run_checks
from fm.jobs.runner import RunOptions, RunStats, in_process_tick, run_forever
from fm.jobs.scheduler_base import RunResult
from fm.jobs.scheduler_windows import WINDOWS_AUTOSTART
from fm.jobs.sync import SyncReport
from fm.jobs.tick import TickOptions, TickReport, tick
from fm.notify import DecisionResult, Message, NotifyChannel, ProposalNotice, Reply
from fm.proposals import queue as proposals_queue
from fm.proposals.policy import as_utc
from fm.setup import Job, SetupDeps
from fm.store import ProposalRow, Store
from fm.web.app import create_app
from fm.web.login import read_url_file, url_file_path

TESTS = Path(__file__).resolve().parent
HOME = TESTS / "fixtures" / "home"
MATCHUPS = TESTS / "fixtures" / "espn" / "ffl_matchups.json"
LEAGUE_ID, SEASON, TEAM, WEEK = 1234567, 2026, 1, 4
SUNDAY = datetime(2026, 10, 4, 15, 35, tzinfo=UTC)
"""Sunday of week 4, 15:35Z: the fixture home synced 35 minutes ago and the 1 p.m. ET kickoffs (17:00Z) lock soon."""
SESSION = EspnSession(espn_s2="s2-journey", swid="{0000-JOURNEY-SWID}", expires_at=datetime(2099, 1, 1, tzinfo=UTC))
LEAGUE = DiscoveredLeague(
    sport="nfl",
    espn_league_id=LEAGUE_ID,
    season=SEASON,
    team_id=TEAM,
    league_name="Fixture League (PPR)",
    team_name="Fixture Team 1",
)
EDGE = Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe")
PORT = 8711
FAST = ExecutorOptions(
    write_timeout_s=5.0, ui_timeout_s=5.0, verify_attempts=2, verify_interval_s=0.0, sleep=lambda _s: None
)
COMMANDS = ("setup", "doctor", "run", "autostart", "ui", "claude")
READS_HOST = "lm-api-reads.fantasy.espn.com"
WRITES_HOST = "lm-api-writes.fantasy.espn.com"

runner = CliRunner()


# --- the clock and the phone ------------------------------------------------------------------------------------------


@dataclass
class Clock:
    """The journey's wall clock: set by the test, never advancing on its own."""

    now: datetime = SUNDAY

    def __call__(self) -> datetime:
        return self.now


class FakePhone:
    """The phone channel ``fm run`` opens: records what it is sent; nobody presses anything on it."""

    def __init__(self) -> None:
        self.notices: list[ProposalNotice] = []
        self.messages: list[Message] = []

    @property
    def name(self) -> str:
        return "fake"

    def send(self, message: Message) -> None:
        self.messages.append(message)

    def send_proposal(self, notice: ProposalNotice) -> None:
        self.notices.append(notice)

    def poll(self, *, wait: bool = True) -> list[Reply]:
        return []

    def confirm(self, reply: Reply, result: DecisionResult) -> None:
        pass

    def dismiss(self, reply: Reply, reason: str) -> None:
        pass

    def close(self) -> None:
        pass


# --- ESPN, faked ------------------------------------------------------------------------------------------------------


class FakeEspn:
    """ESPN as the journey sees it after the first sync: the read API over the home's captures (the pro schedule, the
    week-4 roster and settings) plus the recorded matchups, and a write transport that applies a lineup transaction
    that lands to the roster it serves, so the executor's re-read can verify it. ``opener`` is the executor's runtime
    opener over both, standing in for the browser session ``fm.executor.live_opener`` opens. Nothing here opens a
    socket."""

    def __init__(self) -> None:
        captures = paths.cache_dir() / "espn" / "ffl" / str(SEASON)
        self.rosters: dict[str, Any] = _capture(captures / str(LEAGUE_ID) / "mRoster")
        self.api = FakeEspnApi()
        self.api.serve("proTeamSchedules_wl", _capture(captures / "game" / "proTeamSchedules_wl"))
        self.api.serve("mSettings", _capture(captures / str(LEAGUE_ID) / "mSettings"))
        self.api.serve("mRoster", self._serve_rosters)
        self.api.serve_file("mMatchup", MATCHUPS)
        self.transport = FakeTransport(on_send=self._apply)
        with Store.open() as store:
            league = store.leagues.by_key("nfl")
        assert league is not None
        self.opener = FakeOpener(fake_runtime(self.api, league, transport=self.transport))

    def client(self, league: League, session: EspnSession | None) -> EspnClient:
        """The tick's client factory."""
        return self.api.client(league.game, league.espn_league_id, league.season, session=session)

    def slot(self, espn_id: int) -> int:
        return next(entry["lineupSlotId"] for entry in self._entries() if entry["playerId"] == espn_id)

    def _entries(self) -> list[dict[str, Any]]:
        team = next(team for team in self.rosters["teams"] if team["id"] == TEAM)
        return team["roster"]["entries"]

    def _apply(self, request: WriteRequest) -> None:
        moves = {item["playerId"]: item["toLineupSlotId"] for item in request.body["items"]}
        for entry in self._entries():
            if entry["playerId"] in moves:
                entry["lineupSlotId"] = moves[entry["playerId"]]

    def _serve_rosters(self, request: httpx.Request) -> dict[str, Any]:
        period = request.url.params.get("scoringPeriodId")
        return {**self.rosters, "scoringPeriodId": int(period)} if period is not None else self.rosters


def _capture(directory: Path) -> Any:
    """The one capture the fixture home's sync left in ``directory`` (``<stamp>[_suffix].json`` beside its meta)."""
    (path,) = (found for found in directory.glob("*.json") if not found.name.endswith(".meta.json"))
    return json.loads(path.read_text(encoding="utf-8"))


# --- fm setup's collaborators, faked ----------------------------------------------------------------------------------


@dataclass
class SetupWorld:
    """What ``fm setup`` reaches outside the process: the session in the browser profile, ESPN's league discovery, the
    first sync (a copy of the fixture home's ``state.db`` and captures) and the scheduler jobs."""

    clock: Clock
    installed: list[str] = field(default_factory=list[str])
    synced: list[str] = field(default_factory=list[str])

    def deps(self) -> SetupDeps:
        return SetupDeps(
            load_session=lambda: SESSION,
            login=_never("fm login (the profile already holds a session)"),
            discover=lambda _session: [LEAGUE],
            confirm_league=self._confirm,
            setup_channel=_never("a phone channel setup (--channel none)"),
            sync=self._sync,
            autostart=self._job("fm run at logon", "fm autostart install"),
            wake_timer=self._job("the wake timer (fm tick)", "fm schedule install"),
            bot_pids=lambda: (),
            doctor=lambda: doctor_cmd.check_lines(pinned_checks(self.clock)()),
        )

    def _confirm(self, ref: LeagueRef, session: EspnSession) -> DiscoveredLeague:
        raise AssertionError(f"discovery named the team, so no league needs confirming: {ref}")

    def _sync(self, config: Config, keys: Sequence[str], session: EspnSession) -> list[str]:
        assert [league.key for league in config.leagues] == ["nfl"] and session is SESSION
        shutil.copyfile(HOME / "state.db", paths.state_db())
        shutil.copytree(HOME / "cache", paths.cache_dir(), dirs_exist_ok=True)
        self.synced.extend(keys)
        return [f"{key}: synced (the fixture home's state and captures)" for key in keys]

    def _job(self, name: str, command: str) -> Job:
        def install() -> list[str]:
            self.installed.append(name)
            return [f"installed: {name} (fake scheduler)"]

        return Job(name, command, lambda: name in self.installed, install, install)


def _never(what: str) -> Callable[..., Any]:
    def refuse(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError(f"the journey never runs {what}")

    return refuse


def pinned_checks(clock: Clock) -> Callable[..., list[Check]]:
    """``fm.health.run_checks`` as ``fm doctor`` calls it, at the journey's clock, with the browser launch faked (the
    session comes from the fake loader), Edge "installed" and the logon job asked of a fake scheduler."""

    def scheduler(argv: Sequence[str]) -> RunResult:
        return RunResult(tuple(argv), 1, stderr="No MSFT_ScheduledTask objects found (fake scheduler)")

    def checks(*, online: bool = False, launch_browser: bool = False) -> list[Check]:
        assert not online, "the journey stays offline"
        return run_checks(
            now=clock(),
            launch_browser=launch_browser,
            session_loader=lambda: SESSION,
            browsers=lambda: {"msedge": EDGE},
            autostart_backend=WINDOWS_AUTOSTART,
            scheduler_runner=scheduler,
        )

    return checks


# --- guards -----------------------------------------------------------------------------------------------------------


@pytest.fixture
def real_http(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every request a real httpx transport was asked to send (none may be): each one fails, so a client the journey
    forgot to fake cannot reach ESPN even before the socket-level network guard would stop it."""
    attempts: list[str] = []

    def refuse(self: object, request: httpx.Request) -> httpx.Response:
        attempts.append(f"{request.method} {request.url}")
        raise AssertionError(f"real HTTP in the offline journey: {request.method} {request.url}")

    async def refuse_async(self: object, request: httpx.Request) -> httpx.Response:
        return refuse(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse_async)
    return attempts


@pytest.fixture
def fixture_digest() -> Iterator[str]:
    """The fixture home's ``state.db`` must be byte-identical after the journey: it is copied, never written."""
    before = hashlib.sha256((HOME / "state.db").read_bytes()).hexdigest()
    yield before
    assert hashlib.sha256((HOME / "state.db").read_bytes()).hexdigest() == before


# --- the journey ------------------------------------------------------------------------------------------------------


def fm(*args: str, expect: int | None = 0) -> str:
    result = runner.invoke(fm_app, list(args), catch_exceptions=False)
    if expect is not None:
        assert result.exit_code == expect, result.output
    return result.output


class Syncs:
    """The tick's syncer. The fixture home synced at 15:00Z and the ticks run within the hour, so none is expected;
    each call is recorded."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self, store: Store, config: Config, *, session: EspnSession | None, leagues: Sequence[str]
    ) -> SyncReport:
        self.calls.append(tuple(leagues))
        return SyncReport((), ())


def iteration(clock: Clock, espn: FakeEspn, phone: FakePhone, syncs: Syncs, *, execute: bool) -> RunStats:
    """One ``fm run`` iteration: the live loop with its defaults for config, store and run lock, the tick in this
    process (:func:`fm.jobs.runner.in_process_tick`) over the fakes, ``--no-execute`` unless ``execute``."""

    def journey_tick(
        store: Store,
        config: Config,
        *,
        channel: NotifyChannel | None,
        opener: RuntimeOpener | None,
        flows: FlowRegistry | None,
        options: TickOptions,
        clock: Callable[[], datetime],
    ) -> TickReport:
        return tick(
            store,
            config,
            channel=channel,
            opener=opener,
            flows=flows,  # None: the process-wide flows, set_lineup among them
            options=options,
            clock=clock,
            session_loader=lambda: SESSION,
            clients=espn.client,
            syncer=syncs,
            refresh=None,  # the availability inputs come from nflverse and the NBA injury report: online only
        )

    return run_forever(
        options=RunOptions(execute=execute, max_ticks=1, executor=FAST),
        clock=clock,
        sleep=_never("a sleep (one iteration ticks at once)"),
        channel_opener=lambda _config: phone,
        tick_fn=in_process_tick(tick_fn=journey_tick, opener=espn.opener),
        tree_killer=_never("a process-tree kill"),
    )


def test_setup_check_doctor_run_approve_execute_offline(
    monkeypatch: pytest.MonkeyPatch, real_http: list[str], fixture_digest: str
) -> None:
    probe = socket.socket()
    try:  # the conftest network guard is on in this test
        with pytest.raises(RuntimeError, match="outbound network is disabled"):
            probe.connect(("192.0.2.1", 443))
    finally:
        probe.close()
    clock = Clock()
    help_text = fm("--help")
    assert all(f" {command} " in help_text for command in COMMANDS), help_text

    # 1. fm setup --yes: the session the profile holds, the league discovery finds, the balanced preset, no phone.
    paths.browser_profile_dir().mkdir(parents=True)
    (paths.browser_profile_dir() / "Local State").write_text("{}", encoding="utf-8")  # what fm login leaves behind
    world = SetupWorld(clock)
    monkeypatch.setattr(setup_cmd, "live_deps", world.deps)
    output = fm("setup", "--yes", "--channel", "none")
    assert 'NFL  Fixture League (PPR)  -  team 1 "Fixture Team 1"' in output
    assert "nfl: NFL  Fixture League (PPR)" in output and "preset balanced" in output
    assert world.synced == ["nfl"] and world.installed == ["fm run at logon", "the wake timer (fm tick)"]
    config_text = (paths.config_dir() / "config.toml").read_text(encoding="utf-8")
    assert 'preset = "balanced"' in config_text and 'channel = "none"' in config_text
    assert "\n[league.policy]" not in config_text  # the preset fills the policy; no policy key is written out

    # 2. fm config check: the written file loads, the preset filled the policy.
    checked = fm("config", "check")
    assert "nfl: nfl (ffl) league 1234567, season 2026, team 1, preset balanced" in checked
    assert "notify: none" in checked and "ok: config.toml is valid" in checked
    config = load_config()
    (league,) = config.leagues
    assert (league.policy.bench_inactive, league.policy.lineup, league.policy.add_drop) == ("auto", "auto", "approve")

    # 3. fm doctor --json: valid JSON, one entry per check. No tick has run yet, the one FAIL (exit 1); --yes skipped
    # the API key and there is no phone, two WARNs.
    monkeypatch.setattr(doctor_cmd, "run_checks", pinned_checks(clock))
    doctor = {item["name"]: item for item in json.loads(fm("doctor", "--json", expect=1))}
    assert all(set(item) == {"name", "status", "detail", "fix"} for item in doctor.values())
    status = {name: item["status"] for name, item in doctor.items()}
    assert [name for name, value in status.items() if value == "fail"] == ["tick"]
    assert doctor["tick"]["fix"] == "fm run" and status["session"] == status["browser"] == status["sync:nfl"] == "ok"
    assert status["config"] == "warn" and "ANTHROPIC_API_KEY missing" in doctor["config"]["detail"]
    assert status["notify"] == "warn" and 'notify.channel = "none"' in doctor["notify"]["detail"]

    # 4. One fm run iteration, --no-execute: the lineup window is open, so the real lineup decision drafts proposals
    # and the tick pushes them to the phone.
    espn, phone, syncs = FakeEspn(), FakePhone(), Syncs()
    stats = iteration(clock, espn, phone, syncs, execute=False)
    assert (stats.ticks, stats.tick_failures, stats.stopped_by) == (1, 0, "max_ticks")
    with Store.open() as store:
        drafted = {row.kind: row for row in store.proposals.find(statuses=("proposed",))}
    assert set(drafted) == {"bench_inactive", "lineup"}
    assert sorted(notice.proposal_id for notice in phone.notices) == sorted(row.row_id for row in drafted.values())
    lineup = drafted["lineup"]
    assert lineup.scoring_period_id == WEEK and lineup.policy == "auto"  # balanced: lineups are set on their own
    assert lineup.deadline is not None and lineup.deadline > SUNDAY  # auto fires at T-15 unless someone decides first
    assert espn.transport.sent == []  # --no-execute

    # 5. The dashboard: sign in with the link fm ui opens, take the CSRF token, approve the lineup proposal. Approving
    # checks the deadline against the wall clock, which the journey pins to its Sunday.
    monkeypatch.setattr(proposals_queue, "as_utc", lambda now=None: as_utc(now if now is not None else clock()))
    approved = approve_on_dashboard(lineup)
    assert (approved["status"], approved["decided_by"]) == ("approved", "dashboard")

    # 6. The next iteration ticks ten minutes later and executes the approval: set_lineup in API mode over the fake
    # runtime, one write into the fake transport, verified by re-reading the roster.
    clock.now = SUNDAY + timedelta(minutes=10)
    stats = iteration(clock, espn, phone, syncs, execute=True)
    assert (stats.ticks, stats.tick_failures) == (1, 0)
    with Store.open() as store:
        done = store.proposals.get(lineup.row_id)
        assert done is not None and (done.status, done.decided_by) == ("verified", "dashboard")
        (execution,) = store.executions.for_proposal(lineup.row_id)
        assert (execution.mode, execution.status) == ("api", "verified")
        alternative = store.proposals.get(drafted["bench_inactive"].row_id)
        assert alternative is not None and alternative.status == "rejected"  # the same roster read: superseded
    (write,) = espn.transport.sent
    assert httpx.URL(write.url).host == WRITES_HOST and dict(write.headers) == dict(WEB_CLIENT_HEADERS)
    assert write.body["type"] == "ROSTER" and write.body["isLeagueManager"] is False
    assert all(espn.slot(int(espn_id)) == slot for espn_id, slot in moves(lineup).items())
    assert any(message.title.startswith("✅ Done") for message in phone.messages)

    # Nothing reached ESPN: the reads were GETs the fake API answered, the one write stayed in the fake transport,
    # no real HTTP transport was asked for anything, and no league was synced.
    assert {(request.method, request.url.host) for request in espn.api.requests} == {("GET", READS_HOST)}
    assert {view_key(request) for request in espn.api.requests} >= {"proTeamSchedules_wl", "mRoster", "mSettings"}
    assert real_http == [] and syncs.calls == [] and espn.opener.calls == [(lineup.league_id, False)]

    # fm doctor once more: the tick has run, so nothing fails any longer (exit 0).
    after = {item["name"]: item["status"] for item in json.loads(fm("doctor", "--json"))}
    assert after["tick"] == "ok" and "fail" not in after.values()


def approve_on_dashboard(proposal: ProposalRow) -> dict[str, Any]:
    app = create_app(port=PORT, url_file=url_file_path())
    try:
        with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as client:
            assert client.get("/api/proposals").status_code == 401  # not signed in yet
            link = read_url_file()
            assert link is not None and link.startswith(f"http://127.0.0.1:{PORT}/auth?t=")
            signed_in = client.get(link, follow_redirects=False)
            assert signed_in.status_code == 303 and signed_in.headers["location"] == "/"
            csrf = client.get("/api/session").json()["csrf"]
            listed = client.get("/api/proposals?league=nfl").json()["proposals"]
            assert proposal.row_id in {item["id"] for item in listed}
            refused = client.post(f"/api/proposals/{proposal.row_id}/approve")  # no X-FM-Token
            assert refused.status_code == 403
            answer = client.post(f"/api/proposals/{proposal.row_id}/approve", headers={"X-FM-Token": csrf})
            assert answer.status_code == 200, answer.text
            return answer.json()["proposal"]
    finally:
        app.state.login.close()


def moves(proposal: ProposalRow) -> dict[str, int]:
    return {str(move["espn_id"]): move["to_slot_id"] for move in proposal.payload["moves"]}
