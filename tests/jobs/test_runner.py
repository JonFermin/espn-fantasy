"""``fm.jobs.runner``: the ``fm run`` loop in fake time, and the tick child for real.

The loop runs with a fake clock (time moves only when the loop sleeps, a poll waits, or a fake tick takes time), a
fake phone channel that delivers scripted presses, and a fake tick runner that records each call and answers with a
scripted :class:`TickResult` or exception. Presses go through the real :func:`fm.notify.handle_reply` and the real
:func:`fm.jobs.tick.execute_on_approval` into the executor, over the NFL fixture league with a fake read API, a fake
write transport and a minimal lineup flow (as in ``test_tick.py``), so a near-deadline approval is executed and
verified without a browser or a socket. The run lock is the real one, in the per-test config dir.

The tick child itself is tested with real, tiny, offline processes: a child that hangs with a grandchild is stopped
at a one-second timeout and nothing of it survives, a child's output and exit code come back, and the real
``python -m fm.jobs.runner tick`` runs under the parent's lock (it gets as far as the missing ``config.toml`` of the
temp config dir, so nothing reaches ESPN).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from fm import paths
from fm.browser.fakes import FakeEspnApi, FakeOpener, FakeTransport, fake_runtime
from fm.browser.flows import Flow, FlowContext, FlowRegistry, Mode, Preconditions, Verification, WriteRequest
from fm.config import Config, ConfigError
from fm.espn.settings import load_league_settings
from fm.executor import ExecutorOptions
from fm.jobs import proctree
from fm.jobs.runlock import HELD_BY_ENV, RunLock
from fm.jobs.runner import (
    DEFAULT_TICK_TIMEOUT,
    TICK_CHILD_MODULE,
    TICK_CRASHED_EXIT,
    RunOptions,
    RunStats,
    TickCall,
    TickResult,
    child_tick,
    in_process_tick,
    run_forever,
    run_log_path,
    run_logging,
    tick_child_argv,
    tick_child_main,
)
from fm.jobs.tick import SessionCheck, SessionVerdict, TickOptions, TickReport
from fm.notify import DecisionResult, Message, NotifyChannel, NotifyError, ProposalNotice, Reply, nonces
from fm.proposals import LineupMove, LineupPayload, ProposalKind, get_proposal, propose
from fm.store import LeagueRow, LeagueSettingsRow, Store, utc_now

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "espn"
LEAGUE_ID, SEASON, TEAM, WEEK = 1234567, 2026, 1, 4
T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
HUBBARD, ALLGEIER = 4241416, 4373626
BENCH, FLEX = 20, 23
SWAP = LineupPayload(
    moves=(
        LineupMove(espn_id=HUBBARD, from_slot_id=FLEX, to_slot_id=BENCH),
        LineupMove(espn_id=ALLGEIER, from_slot_id=BENCH, to_slot_id=FLEX),
    )
)
FAST = ExecutorOptions(
    write_timeout_s=5.0, ui_timeout_s=5.0, verify_attempts=2, verify_interval_s=0.0, sleep=lambda _s: None
)
CONFIG = Config.model_validate(
    {
        "league": [
            {
                "key": "nfl",
                "sport": "nfl",
                "espn_league_id": LEAGUE_ID,
                "season": SEASON,
                "team_id": TEAM,
                "policy": {"bench_inactive": "auto", "lineup": "approve"},
            }
        ]
    }
)
TEN_MINUTES = timedelta(minutes=10)


# --- fakes ------------------------------------------------------------------------------------------------------------


class FakeTime:
    """The loop's clock and sleep: time moves only when something sleeps or advances it."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start
        self.sleeps: list[float] = []
        self.on_sleep: Callable[[], None] | None = None

    def clock(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)
        if self.on_sleep is not None:
            self.on_sleep()


class FakeTick:
    """A tick runner that records each call, takes ``took`` of fake time and answers from a script (then exit 0)."""

    def __init__(self, time_: FakeTime, *script: TickResult | BaseException, took: timedelta = timedelta(0)) -> None:
        self.time = time_
        self.script = list(script)
        self.took = took
        self.calls: list[tuple[datetime, TickCall]] = []
        self.during: Callable[[TickCall], None] | None = None

    def answer(self, *script: TickResult | BaseException) -> None:
        """What the next ticks answer, in order (exit 0 after that)."""
        self.script = list(script)

    def __call__(self, call: TickCall, /) -> TickResult:
        self.calls.append((self.time.now, call))
        if self.during is not None:
            self.during(call)
        self.time.now += self.took
        outcome = self.script.pop(0) if self.script else TickResult(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def times(self) -> list[datetime]:
        return [at for at, _ in self.calls]


class FakeChannel:
    """A phone that delivers scripted presses when their time comes. A waiting poll holds for ``wait_s`` of fake
    time, or less when a press arrives in between (Telegram's long poll); ``errors`` make the next polls fail."""

    name = "fake"

    def __init__(self, time_: FakeTime, *, wait_s: float = 25.0) -> None:
        self.time = time_
        self.wait_s = wait_s
        self.inbox: list[tuple[datetime, Reply]] = []
        self.errors: list[Exception] = []
        self.polls: list[tuple[datetime, bool]] = []
        self.sent: list[Message] = []
        self.confirmed: list[DecisionResult] = []
        self.closed = 0

    def press(self, at: datetime, reply: Reply) -> None:
        self.inbox.append((at, reply))

    def poll(self, *, wait: bool = True) -> list[Reply]:
        self.polls.append((self.time.now, wait))
        if self.errors:
            raise self.errors.pop(0)
        if wait and not self._ready():
            later = [at for at, _ in self.inbox if at > self.time.now]
            limit = self.time.now + timedelta(seconds=self.wait_s)
            self.time.now = min([*later, limit]) if later and min(later) < limit else limit
        ready = self._ready()
        self.inbox = [(at, reply) for at, reply in self.inbox if at > self.time.now]
        return ready

    def _ready(self) -> list[Reply]:
        return [reply for at, reply in self.inbox if at <= self.time.now]

    def send(self, message: Message) -> None:
        self.sent.append(message)

    def send_proposal(self, notice: ProposalNotice) -> None:
        raise AssertionError("fm run itself never pushes a proposal; the tick does")

    def confirm(self, reply: Reply, result: DecisionResult) -> None:
        self.confirmed.append(result)

    def dismiss(self, reply: Reply, reason: str) -> None:
        pass

    def close(self) -> None:
        self.closed += 1

    def titles(self) -> list[str]:
        return [message.title for message in self.sent]


def no_channel(config: Config) -> NotifyChannel:
    raise NotifyError("TELEGRAM_BOT_TOKEN is not set")


def refuse_opener(league: LeagueRow, *, dry_run: bool) -> Any:
    raise AssertionError("nothing should execute in this test")


class SwapFlow(Flow[LineupPayload]):
    """Lineup moves over ``mRoster``: preconditions, one API write, a re-read (``test_tick.py``'s flow)."""

    name = "test_lineup"
    kinds = (ProposalKind.LINEUP, ProposalKind.BENCH_INACTIVE)
    payload_type = LineupPayload
    modes = (Mode.API,)

    def _slots(self, ctx: FlowContext[LineupPayload]) -> dict[int, int]:
        roster = ctx.reader.rosters(ctx.scoring_period_id).data.roster(ctx.team_id)
        return {entry.player_id: entry.lineup_slot_id for entry in roster.entries}

    def check(self, ctx: FlowContext[LineupPayload]) -> Preconditions:
        slots = self._slots(ctx)
        failures = tuple(
            f"player {move.espn_id} is in slot {slots.get(move.espn_id)}, not {move.from_slot_id}"
            for move in ctx.payload.moves
            if slots.get(move.espn_id) != move.from_slot_id
        )
        return Preconditions(failures, {})

    def build_request(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> WriteRequest:
        items = [
            {
                "playerId": m.espn_id,
                "type": "LINEUP",
                "fromLineupSlotId": m.from_slot_id,
                "toLineupSlotId": m.to_slot_id,
            }
            for m in ctx.payload.moves
        ]
        body = {
            "isLeagueManager": False,
            "teamId": ctx.team_id,
            "type": "ROSTER",
            "memberId": ctx.member_id,
            "scoringPeriodId": ctx.scoring_period_id,
            "executionType": "EXECUTE",
            "items": items,
        }
        return WriteRequest(url=ctx.transactions_url, body=body)

    def verify(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> Verification:
        slots = self._slots(ctx)
        expected = {str(m.espn_id): m.to_slot_id for m in ctx.payload.moves}
        observed = {str(m.espn_id): slots.get(m.espn_id) for m in ctx.payload.moves}
        detail = "slots match" if observed == expected else "mismatch"
        return Verification(observed == expected, detail, expected, observed)


class World:
    """The loop's collaborators over the NFL fixture league, all faked; ``run`` drives ``run_forever``."""

    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self.time = FakeTime()
        self.channel = FakeChannel(self.time)
        self.ticks = FakeTick(self.time)
        self.configs: list[Config | Exception] = []
        self.config_loads = 0
        with Store.open(store_path) as store:
            self.league = store.leagues.upsert(
                LeagueRow(key="nfl", sport="nfl", espn_league_id=LEAGUE_ID, season=SEASON, team_id=TEAM, as_of=T0)
            )
            settings = load_league_settings(FIXTURES / "ffl_settings_ppr.json", game="ffl")
            store.settings.upsert(
                LeagueSettingsRow(league_id=self.league.row_id, settings=settings.model_dump(mode="json"), as_of=T0)
            )
        self.rosters: dict[str, Any] = json.loads((FIXTURES / "ffl_rosters_week4.json").read_text(encoding="utf-8"))
        self.api = FakeEspnApi()
        self.api.serve("mRoster", lambda _request: self.rosters)
        self.transport = FakeTransport(on_send=self.apply_request)
        self.opener = FakeOpener(fake_runtime(self.api, self.league, transport=self.transport))
        self.flows = FlowRegistry()
        self.flows.register(SwapFlow())

    def load_config(self) -> Config:
        self.config_loads += 1
        if self.configs:
            loaded = self.configs.pop(0)
            if isinstance(loaded, Exception):
                raise loaded
            return loaded
        return CONFIG

    def open_store(self) -> Store:
        return Store.open(self.store_path)

    def apply_request(self, request: WriteRequest) -> None:
        wanted = {item["playerId"]: item["toLineupSlotId"] for item in request.body["items"]}
        team = next(team for team in self.rosters["teams"] if team["id"] == TEAM)
        for entry in team["roster"]["entries"]:
            if entry["playerId"] in wanted:
                entry["lineupSlotId"] = wanted[entry["playerId"]]

    def propose(self, *, deadline: datetime) -> int:
        with Store.open(self.store_path) as store:
            row = propose(
                store,
                CONFIG,
                self.league,
                ProposalKind.LINEUP,
                SWAP,
                created_by="test",
                scoring_period_id=WEEK,
                deadline=deadline,
                now=T0,
            )
        return row.row_id

    def status(self, proposal_id: int) -> str:
        with Store.open(self.store_path) as store:
            return get_proposal(store, proposal_id).status

    def run(self, *, channel: bool = True, **options: Any) -> RunStats:
        opts = {"interval": TEN_MINUTES, "executor": FAST, **options}
        return run_forever(
            options=RunOptions(**opts),
            clock=self.time.clock,
            sleep=self.time.sleep,
            config_loader=self.load_config,
            store_opener=self.open_store,
            channel_opener=(lambda _config: self.channel) if channel else no_channel,
            tick_fn=self.ticks,
            opener=self.opener,
            flows=self.flows,
        )


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path / "state.db")


def approve_press(proposal_id: int, *, deadline: datetime) -> Reply:
    issued = nonces.issue(proposal_id, deadline=deadline, now=T0)
    return Reply(channel="fake", proposal_id=proposal_id, decision="approve", nonce=issued.nonce, sender="test")


# --- ticks ------------------------------------------------------------------------------------------------------------


def test_ticks_fire_on_the_wall_clock_schedule_with_a_fresh_config_each_time(world: World) -> None:
    stats = world.run(max_ticks=3, channel=False)
    assert world.ticks.times == [T0, T0 + TEN_MINUTES, T0 + 2 * TEN_MINUTES]
    assert stats.ticks == 3 and stats.stopped_by == "max_ticks"
    assert world.config_loads == 3
    assert max(world.time.sleeps) <= 1.0  # no channel: short slices, so stop is answered quickly
    (_, call), *_ = world.ticks.calls
    assert call.config is CONFIG and call.channel is None and call.timeout == DEFAULT_TICK_TIMEOUT
    assert call.options == TickOptions(execute=True, sync=True, executor=FAST)


def test_the_next_tick_is_measured_from_the_end_of_a_slow_one(world: World) -> None:
    world.ticks.took = timedelta(minutes=3)
    world.run(max_ticks=2, channel=False)
    assert world.ticks.times == [T0, T0 + timedelta(minutes=13)]


def test_a_three_hour_clock_jump_ticks_at_once_and_only_once(world: World) -> None:
    jumped: list[datetime] = []

    def jump() -> None:
        if not jumped and len(world.ticks.calls) == 1:
            jumped.append(world.time.now)
            world.time.now += timedelta(hours=3)  # the laptop slept through 18 ticks

    world.time.on_sleep = jump
    world.run(max_ticks=3)
    first, second, third = world.ticks.times
    assert first == T0
    assert second == jumped[0] + timedelta(hours=3)  # due at once on waking, not 10 minutes later
    assert third == second + TEN_MINUTES  # and the 18 missed ones are not replayed


def test_a_raising_tick_is_alerted_and_the_loop_goes_on(world: World) -> None:
    world.ticks.answer(RuntimeError("boom"), RuntimeError("boom again"), TickResult(0))
    stats = world.run(max_ticks=3)
    assert stats.tick_failures == 2 and stats.ticks == 1 and len(world.ticks.calls) == 3
    assert world.channel.titles() == ["fm run: tick failed"]  # once an hour, not once a tick
    assert stats.alerts == ["fm run: tick failed"]


def test_a_tick_that_crashed_or_reported_a_failure_is_counted_apart(world: World) -> None:
    world.ticks.answer(TickResult(TICK_CRASHED_EXIT), TickResult(1), TickResult(0))
    stats = world.run(max_ticks=3)
    assert stats.tick_failures == 1 and stats.ticks == 2
    assert world.channel.titles() == ["fm run: tick failed"]  # exit 1 was alerted by the tick itself


def test_a_hung_tick_stopped_at_the_timeout_is_alerted_once_an_hour_and_the_loop_goes_on(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    timeout = timedelta(minutes=20)
    hung = TickResult(None, elapsed=timeout, pid=4242, notes=("pid 4242 still running 10s after the kill",))
    world.ticks.answer(hung, hung, TickResult(0))
    world.ticks.took = timeout
    with caplog.at_level(logging.INFO, logger="fm"):
        stats = world.run(max_ticks=3, tick_timeout=timeout)
    assert stats.tick_timeouts == 2 and stats.ticks == 1 and len(world.ticks.calls) == 3
    assert all(call.timeout == timeout for _, call in world.ticks.calls)
    assert world.channel.titles() == ["fm run: tick timed out"]  # the second, 30 minutes later, is not re-alerted
    (alert,) = world.channel.sent
    assert "tick timed out after 20 min and was stopped" in alert.body
    assert "tick timed out after 20 min and was stopped (pid 4242" in caplog.text
    assert "pid 4242 still running 10s after the kill" in caplog.text
    first, second, third = world.ticks.times
    assert second - first == timeout + TEN_MINUTES and third - second == timeout + TEN_MINUTES


def test_a_config_that_will_not_load_skips_the_tick_and_the_next_one_reloads(world: World) -> None:
    world.configs = [CONFIG, ConfigError("config.toml: invalid TOML"), CONFIG]
    stats = world.run(max_ticks=3)
    assert stats.config_failures == 1 and stats.ticks == 2 and len(world.ticks.calls) == 2
    assert world.config_loads == 3
    assert world.channel.titles() == ["fm run: config.toml not loaded"]


def test_without_a_channel_the_loop_sleeps_to_the_tick_and_alerts_nowhere(world: World) -> None:
    world.ticks.answer(RuntimeError("boom"))
    stats = world.run(max_ticks=2, channel=False)
    assert stats.alerts == ["fm run: tick failed"] and world.channel.sent == []
    assert world.channel.polls == []
    assert sum(world.time.sleeps) == pytest.approx(TEN_MINUTES.total_seconds())


# --- approvals --------------------------------------------------------------------------------------------------------


def test_a_press_between_ticks_is_recorded_and_a_near_deadline_approval_executes_at_once(world: World) -> None:
    deadline = T0 + timedelta(minutes=20)
    proposal_id = world.propose(deadline=deadline)
    world.channel.press(T0 + timedelta(minutes=2), approve_press(proposal_id, deadline=deadline))
    stats = world.run(max_ticks=2)
    assert stats.decisions == 1 and stats.executions == 1
    assert world.status(proposal_id) == "verified"
    assert world.opener.calls == [(world.league.row_id, False)] and len(world.transport.sent) == 1
    assert [result.headline for result in world.channel.confirmed] == [f"#{proposal_id} approved"]
    assert [title for title in world.channel.titles() if title.startswith("✅ Done")]
    assert len(world.ticks.calls) == 2  # the approval did not wait for the next tick
    polls = [at for at, _ in world.channel.polls]
    assert polls and all(T0 <= at < T0 + TEN_MINUTES for at in polls)  # all between the two ticks


def test_a_press_far_from_its_deadline_is_recorded_and_left_to_the_tick(world: World) -> None:
    deadline = T0 + timedelta(hours=3)
    proposal_id = world.propose(deadline=deadline)
    world.channel.press(T0 + timedelta(minutes=1), approve_press(proposal_id, deadline=deadline))
    stats = world.run(max_ticks=2)
    assert stats.decisions == 1 and stats.executions == 0
    assert world.status(proposal_id) == "approved" and world.opener.calls == []


def test_with_no_execute_a_press_is_recorded_but_nothing_runs(world: World) -> None:
    deadline = T0 + timedelta(minutes=20)
    proposal_id = world.propose(deadline=deadline)
    world.channel.press(T0 + timedelta(minutes=1), approve_press(proposal_id, deadline=deadline))
    stats = world.run(max_ticks=2, execute=False)
    assert stats.decisions == 1 and stats.executions == 0 and world.opener.calls == []
    assert world.status(proposal_id) == "approved"
    assert all(not call.options.execute for _, call in world.ticks.calls)


def test_an_unknown_button_is_ignored(world: World) -> None:
    world.channel.press(T0 + timedelta(minutes=1), Reply("fake", 99, "approve", "x" * 24, "test"))
    stats = world.run(max_ticks=2)
    assert stats.decisions == 0 and stats.executions == 0


def test_poll_errors_back_off_doubling_and_are_alerted_once(world: World) -> None:
    world.channel.errors = [NotifyError("502"), NotifyError("502"), NotifyError("502"), RuntimeError("socket")]
    stats = world.run(max_ticks=2)
    assert stats.poll_failures == 4 and stats.ticks == 2
    assert world.channel.titles() == ["fm run: phone channel failing"]
    times = [at for at, _ in world.channel.polls]
    backoffs = [(later - earlier).total_seconds() for earlier, later in zip(times, times[1:], strict=False)]
    assert backoffs[:4] == [2.0, 4.0, 8.0, 16.0]


def test_a_press_that_blows_up_is_alerted_and_the_loop_goes_on(world: World) -> None:
    def broken(store: Store, channel: NotifyChannel, reply: Reply, *, now: datetime | None) -> DecisionResult | None:
        raise OSError("disk full")

    world.channel.press(T0 + timedelta(minutes=1), Reply("fake", 1, "approve", "x" * 24, "test"))
    stats = run_forever(
        options=RunOptions(interval=TEN_MINUTES, max_ticks=2),
        clock=world.time.clock,
        sleep=world.time.sleep,
        config_loader=world.load_config,
        store_opener=world.open_store,
        channel_opener=lambda _config: world.channel,
        tick_fn=world.ticks,
        reply_handler=broken,
        opener=refuse_opener,
    )
    assert stats.ticks == 2 and stats.decisions == 0
    assert world.channel.titles() == ["fm run: press on #1 not recorded"]


# --- stop, interrupt, lock --------------------------------------------------------------------------------------------


def test_stop_ends_the_loop_cleanly_and_releases_the_lock(world: World) -> None:
    import threading

    stop = threading.Event()
    held: list[bool] = []
    world.ticks.during = lambda _call: held.append(RunLock().is_locked())
    world.time.on_sleep = lambda: stop.set() if world.time.now >= T0 + timedelta(minutes=15) else None
    stats = run_forever(
        options=RunOptions(interval=TEN_MINUTES),
        stop=stop,
        clock=world.time.clock,
        sleep=world.time.sleep,
        config_loader=world.load_config,
        store_opener=world.open_store,
        channel_opener=lambda _config: world.channel,
        tick_fn=world.ticks,
        opener=refuse_opener,
    )
    assert stats.stopped_by == "stop" and stats.ticks == 2
    assert held == [True, True]
    assert not RunLock().is_locked() and RunLock().holder() is None
    assert world.channel.closed == 1


def test_ctrl_c_in_a_tick_is_a_clean_stop_that_releases_the_lock(world: World) -> None:
    world.ticks.answer(TickResult(0), KeyboardInterrupt())
    stats = world.run()
    assert stats.stopped_by == "interrupt" and stats.ticks == 1
    assert not RunLock().is_locked()


def test_the_loop_waits_for_a_tick_that_holds_the_lock_and_says_so(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    scheduled = RunLock(command="fm tick").acquire()
    world.time.on_sleep = lambda: scheduled.release() if len(world.time.sleeps) == 4 else None
    with caplog.at_level(logging.INFO, logger="fm"):
        stats = world.run(max_ticks=1)
    assert stats.ticks == 1 and world.time.sleeps[:4] == [0.5, 0.5, 0.5, 0.5]
    assert f"waiting for the run lock held by fm tick (pid {os.getpid()})" in caplog.text
    assert world.ticks.times == [T0 + timedelta(seconds=2)]


# --- the tick runners -------------------------------------------------------------------------------------------------


def test_tick_child_argv_runs_fm_tick_with_the_options_through_this_interpreter() -> None:
    assert tick_child_argv(TickOptions()) == [sys.executable, "-m", TICK_CHILD_MODULE, "tick"]
    assert tick_child_argv(TickOptions(execute=False, sync=False, leagues=("nfl", "nba")), python="py") == [
        "py",
        "-m",
        "fm.jobs.runner",
        "tick",
        "--no-execute",
        "--no-sync",
        "--league",
        "nfl",
        "--league",
        "nba",
    ]


def test_tick_result_says_what_happened() -> None:
    assert TickResult(0).ok and not TickResult(0).crashed and not TickResult(0).timed_out
    assert not TickResult(1).ok and not TickResult(1).crashed
    assert TickResult(TICK_CRASHED_EXIT).crashed and TickResult(-9).crashed
    assert TickResult(None).timed_out and not TickResult(None).crashed


def _report(*, ok: bool) -> TickReport:
    from fm.jobs.tick import DecisionRun, LeagueTick

    runs = () if ok else (DecisionRun(name="lineup", window="w", error="RuntimeError: boom"),)
    league = LeagueTick(key="nfl", sport="nfl", period=4, decisions=runs)
    return TickReport(
        at=T0,
        session=SessionCheck(SessionVerdict.OK, "session ok"),
        paused=None,
        reconciled=(),
        expired=(),
        leagues=(league,),
        auto_approved=(),
        executions=(),
        would_execute=(),
        alerts=(),
        warnings=(),
    )


def test_in_process_tick_runs_the_tick_with_the_loops_store_config_and_channel(world: World) -> None:
    seen: dict[str, Any] = {}

    def fake_tick(store: Store, config: Config, **kwargs: Any) -> TickReport:
        seen.update(store=store, config=config, **kwargs)
        return _report(ok=bool(seen.get("ok", True)))

    runner = in_process_tick(tick_fn=fake_tick, opener=world.opener, flows=world.flows)
    with Store.open(world.store_path) as store:
        call = TickCall(store, CONFIG, None, TickOptions(execute=False), DEFAULT_TICK_TIMEOUT, world.time.clock)
        result = runner(call)
        assert result.ok and "tick at 2026-10-04 15:00 UTC" in result.lines
        assert seen["store"] is store and seen["config"] is CONFIG and seen["opener"] is world.opener
        assert seen["options"] == TickOptions(execute=False) and seen["channel"] is None
        seen["ok"] = False
        assert runner(call).returncode == 1


def test_a_tick_function_report_reaches_the_loop_log(world: World, caplog: pytest.LogCaptureFixture) -> None:
    def fake_tick(store: Store, config: Config, **kwargs: Any) -> TickReport:
        return _report(ok=True)

    with caplog.at_level(logging.INFO, logger="fm"):
        stats = run_forever(
            options=RunOptions(max_ticks=1),
            clock=world.time.clock,
            sleep=world.time.sleep,
            config_loader=world.load_config,
            store_opener=world.open_store,
            channel_opener=no_channel,
            tick_fn=in_process_tick(tick_fn=fake_tick),
            opener=refuse_opener,
        )
    assert stats.ticks == 1 and "nfl (nfl): period 4" in caplog.text


# --- the tick child, for real -----------------------------------------------------------------------------------------

PYTHON = [sys.executable, "-I", "-S"]
HANG = (
    "import os, subprocess, sys, time\n"
    "grandchild = subprocess.Popen([sys.executable, '-I', '-S', '-c', "
    "'import os, time; print(\"grandchild\", os.getpid(), flush=True); time.sleep(60)'])\n"
    f"print('child', os.getpid(), grandchild.pid, os.environ.get('{HELD_BY_ENV}', '-'), flush=True)\n"
    "time.sleep(60)\n"
)
"""A tick that hangs, with a grandchild that hangs too (the browser): both report their pids to the log."""


def call_with(store: Store, timeout: timedelta) -> TickCall:
    return TickCall(store, CONFIG, None, TickOptions(), timeout, utc_now)


def log_words(path: Path, name: str, *, timeout: float = 10.0) -> list[str]:
    deadline = time.monotonic() + timeout
    while True:
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        for line in text.splitlines():
            if line.split()[:1] == [name]:
                return line.split()
        if time.monotonic() >= deadline:
            raise AssertionError(f"no {name!r} line in {path}:\n{text}")
        time.sleep(0.05)


def still_running(pids: set[int], timeout: float = 10.0) -> set[int]:
    deadline = time.monotonic() + timeout
    while True:
        alive = {pid for pid in pids if proctree.is_running(pid)}
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(0.05)


def test_a_real_hung_tick_is_stopped_at_the_timeout_with_its_whole_tree(world: World, tmp_path: Path) -> None:
    log = tmp_path / "logs" / "run.log"
    started = time.monotonic()
    with Store.open(world.store_path) as store:
        result = child_tick(call_with(store, timedelta(seconds=1)), argv=[*PYTHON, "-c", HANG], log_path=log)
    elapsed = time.monotonic() - started
    assert result.timed_out and result.notes == ()
    assert 1.0 <= elapsed < 30.0 and result.elapsed >= timedelta(seconds=1)
    child = log_words(log, "child", timeout=0)
    grandchild = log_words(log, "grandchild", timeout=0)
    assert child[3] == str(os.getpid())  # the lock marker names this process
    pids = {result.pid or 0, int(child[1]), int(child[2]), int(grandchild[1])}
    assert still_running(pids) == set(), "the tick child's tree survived the timeout"


def test_ctrl_c_while_a_real_tick_child_runs_stops_its_tree_before_the_interrupt_goes_on(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "logs" / "run.log"

    def interrupted(self: proctree.ProcessTree, timeout: float | None = None) -> int:
        log_words(log, "grandchild")  # both are up: now the user presses Ctrl+C
        raise KeyboardInterrupt

    monkeypatch.setattr(proctree.ProcessTree, "wait", interrupted)
    with Store.open(world.store_path) as store, pytest.raises(KeyboardInterrupt):
        child_tick(call_with(store, timedelta(seconds=60)), argv=[*PYTHON, "-c", HANG], log_path=log)
    child, grandchild = log_words(log, "child", timeout=0), log_words(log, "grandchild", timeout=0)
    assert still_running({int(child[1]), int(child[2]), int(grandchild[1])}) == set()


def test_a_real_tick_child_appends_its_output_and_returns_its_exit_code(world: World, tmp_path: Path) -> None:
    log = tmp_path / "logs" / "run.log"
    with run_logging(log):
        logging.getLogger("fm.test").info("the parent wrote first")
        with Store.open(world.store_path) as store:
            ok = child_tick(
                call_with(store, timedelta(seconds=60)), argv=[*PYTHON, "-c", "print('tick ok')"], log_path=log
            )
            crashed = child_tick(
                call_with(store, timedelta(seconds=60)),
                argv=[*PYTHON, "-c", "import sys; print('tick crashed', file=sys.stderr); sys.exit(70)"],
                log_path=log,
            )
        logging.getLogger("fm.test").info("the parent wrote last")
    assert ok.ok and ok.pid is not None and crashed.crashed and crashed.returncode == TICK_CRASHED_EXIT
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith("the parent wrote first")
    assert "tick ok" in lines and "tick crashed" in lines
    assert lines[-1].endswith("the parent wrote last")
    assert log == tmp_path / "logs" / "run.log" and run_log_path() == paths.config_dir() / "logs" / "run.log"


def test_the_real_tick_child_runs_fm_tick_under_the_parents_lock(world: World, tmp_path: Path) -> None:
    # The temp config dir has no config.toml, so the real `fm tick` stops at loading it: past the lock, before ESPN.
    environ = dict(os.environ)
    assert Path(environ[paths.CONFIG_DIR_ENV]).is_relative_to(tmp_path)  # never the real ~/.config/espn-fantasy
    assert not (paths.config_dir() / "config.toml").exists()
    log = tmp_path / "logs" / "run.log"
    with RunLock(command="fm run"), Store.open(world.store_path) as store:
        result = child_tick(call_with(store, timedelta(seconds=120)), log_path=log, environ=environ)
    text = log.read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 1, text
    assert "config.toml not found" in text and "skipping" not in text


def test_tick_child_main_runs_the_cli_and_maps_a_crash_to_its_own_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    from fm.commands import schedule as schedule_cmd

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # typer installs its own when an app is called
    assert tick_child_main(["tick", "--help"]) == 0

    def boom(*_args: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(schedule_cmd, "_tick_once", boom)
    assert tick_child_main(["tick", "--no-sync"]) == TICK_CRASHED_EXIT
    assert not RunLock().is_locked()  # the tick's own lock was released on the way out
    assert tick_child_main(["no-such-command"]) == 2


@pytest.fixture(autouse=True)
def _no_held_marker(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(HELD_BY_ENV, raising=False)
    yield
