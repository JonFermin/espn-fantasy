"""The tick (ROADMAP #29) against a fixture store with fakes: what it does, in what order, and only once.

The world is the NFL fixture league (``tests/fixtures/espn``: league 1234567, week 4 of 2026, our team 1) on the
Sunday of week 4. ESPN reads (the pro schedule, matchups, the roster) come from a fake read API, writes land on the
roster through a fake transport, decisions are recording fakes on a private registry, the lineup flow is a minimal
``mRoster`` swap on a private flow registry, and the phone is a fake channel. The session is a saved cookie or a
``NotLoggedInError``. The news feeds are the recorded captures in ``tests/fixtures/sources/news`` (ROADMAP #58:
Tyler Allgeier, on team 1's bench, is tagged in ESPN's pickups story) handed over by a fake poller, and Claude is a
fake transport that answers every triage with an out-of-bounds signal per candidate. Nothing opens a browser or a
socket; nothing is scheduled.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import polars as pl
import pytest
import typer
from pydantic import BaseModel
from typer.testing import CliRunner

from fm.advisor.client import AdvisorClient, AdvisorError, CallParams, RawReply, TokenUsage, Transport
from fm.advisor.news_triage import TriagedItem, TriageOutput, TriageSignal
from fm.browser.fakes import FakeEspnApi, FakeOpener, FakeTransport, fake_runtime
from fm.browser.flows import Flow, FlowContext, FlowRegistry, Mode, Preconditions, Verification, WriteRequest
from fm.browser.session import BrowserError
from fm.commands import schedule as schedule_cmd
from fm.config import DEFAULT_MODEL, Config, League, Llm, Secrets, Sport
from fm.decide.registry import DecisionRegistry
from fm.espn.auth import EspnSession, NotLoggedInError
from fm.espn.calendar import CalendarError
from fm.espn.client import EspnClient
from fm.espn.settings import load_league_settings
from fm.executor import STALE_EXECUTION, ExecutorOptions
from fm.jobs import tick as tick_module
from fm.jobs.deadlines import WindowKind
from fm.jobs.sync import SyncReport
from fm.jobs.tick import (
    ALERT_REPEAT,
    DECIDED_BY_TICK,
    INPUTS_EVERY,
    NEWS_EVERY,
    AdvisorFactory,
    LeagueContext,
    SessionVerdict,
    TickOptions,
    TickReport,
    TickState,
    check_session,
    execute_on_approval,
    refresh_nfl_inputs,
    tick,
)
from fm.model.availability import NEWS_BOUND
from fm.model.ids import GSIS
from fm.notify import DecisionResult, Message, NotifyError, ProposalNotice, Reply
from fm.proposals import (
    LineupMove,
    LineupPayload,
    ProposalKind,
    approve,
    begin_execution,
    get_proposal,
    pause,
    propose,
)
from fm.sources.base import Fetched
from fm.sources.news import ESPN_NEWS, ROTOWIRE, NewsItem, NewsPoll, dedupe, parse_espn_news, parse_rotowire_rss
from fm.sports.base import plugin_for
from fm.store import (
    ExecutionRow,
    LeagueRow,
    LeagueSettingsRow,
    PlayerIdRow,
    PlayerRow,
    RosterEntryRow,
    Store,
    TeamRow,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "espn"
NEWS = FIXTURES.parent / "sources" / "news"
LEAGUE_ID, SEASON, TEAM, WEEK = 1234567, 2026, 1, 4
# Sunday of week 4: the 1 p.m. ET kickoffs lock at 17:00 UTC, so their decision window opens at 15:30 UTC.
LOCK = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
EARLY = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
NOW = datetime(2026, 10, 4, 15, 35, tzinfo=UTC)
THURSDAY = datetime(2026, 10, 1, 20, 0, tzinfo=UTC)
HUBBARD, ALLGEIER, GIBBS, DANIELS = 4241416, 4373626, 4429795, 4426348
RB, BENCH, FLEX = 2, 20, 23
# The recorded NFL feeds (2026-10-05): five ESPN articles, one tagging Allgeier, and five RotoWire blurbs, one titled
# "Jayden Daniels: ...", which the store matches by name once Daniels is a known player.
NEWS_ITEMS: tuple[NewsItem, ...] = (
    *parse_espn_news((NEWS / "espn_news_nfl.json").read_bytes(), "nfl"),
    *parse_rotowire_rss((NEWS / "rotowire_nfl.xml").read_bytes(), "nfl"),
)
PICKUPS_STORY = "50108832"
"""ESPN's free-agent pickups article, the one tagging Allgeier."""
SWAP = LineupPayload(
    moves=(
        LineupMove(espn_id=HUBBARD, from_slot_id=FLEX, to_slot_id=BENCH),
        LineupMove(espn_id=ALLGEIER, from_slot_id=BENCH, to_slot_id=FLEX),
    )
)
OTHER = LineupPayload(moves=(LineupMove(espn_id=ALLGEIER, from_slot_id=BENCH, to_slot_id=RB),))
FAST = ExecutorOptions(
    write_timeout_s=5.0, ui_timeout_s=5.0, verify_attempts=2, verify_interval_s=0.0, sleep=lambda _s: None
)
SESSION = EspnSession(espn_s2="s2-secret", swid="{TEST-SWID}", expires_at=NOW + timedelta(days=200))
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

runner = CliRunner()


# --- fakes ------------------------------------------------------------------------------------------------------------


class SwapFlow(Flow[LineupPayload]):
    """Lineup moves over ``mRoster``, cut down to what the tick needs: preconditions, one API write, a re-read."""

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
        body = {
            "isLeagueManager": False,
            "teamId": ctx.team_id,
            "type": "ROSTER",
            "memberId": ctx.member_id,
            "scoringPeriodId": ctx.scoring_period_id,
            "executionType": "EXECUTE",
            "items": [
                {
                    "playerId": m.espn_id,
                    "type": "LINEUP",
                    "fromLineupSlotId": m.from_slot_id,
                    "toLineupSlotId": m.to_slot_id,
                }
                for m in ctx.payload.moves
            ],
        }
        return WriteRequest(url=ctx.transactions_url, body=body)

    def verify(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> Verification:
        slots = self._slots(ctx)
        expected = {str(m.espn_id): m.to_slot_id for m in ctx.payload.moves}
        observed = {str(m.espn_id): slots.get(m.espn_id) for m in ctx.payload.moves}
        detail = "slots match" if observed == expected else "mismatch"
        return Verification(observed == expected, detail, expected, observed)


class FakeChannel:
    name = "fake"

    def __init__(self) -> None:
        self.sent: list[Message] = []
        self.notices: list[ProposalNotice] = []
        self.fail_push = False

    def send(self, message: Message) -> None:
        self.sent.append(message)

    def send_proposal(self, notice: ProposalNotice) -> None:
        if self.fail_push:
            raise NotifyError("push failed")
        self.notices.append(notice)

    def poll(self, *, wait: bool = True) -> list[Reply]:
        return []

    def confirm(self, reply: Reply, result: DecisionResult) -> None:
        pass

    def dismiss(self, reply: Reply, reason: str) -> None:
        pass

    def close(self) -> None:
        pass

    def alerts(self) -> list[str]:
        return [message.title for message in self.sent if message.priority == "high"]


class FakeClaude(Transport):
    """Answers every triage chunk with one injury signal per (item, candidate) it was asked about, proposed outside
    the bounds (delta -0.9, confidence 1.5) so what the tick stores shows the clamping; ``error`` fails the next call
    instead. Nothing else is implemented: the tick never batches."""

    def __init__(self) -> None:
        self.calls: list[CallParams] = []
        self.error: Exception | None = None

    def parse(self, params: CallParams, output: type[BaseModel], *, fallback: bool = False) -> RawReply:
        assert output is TriageOutput
        self.calls.append(params)
        if self.error is not None:
            raise self.error
        text = params["messages"][0]["content"][0]["text"]  # type: ignore[index]
        items = [
            TriagedItem(
                news_item_id=int(match[1]),
                signals=[
                    TriageSignal(
                        espn_id=int(espn_id),
                        kind="injury",
                        severity="moderate",
                        games_out=1,
                        p_active_delta=-0.9,
                        confidence=1.5,
                        summary="Listed as doubtful for Sunday.",
                    )
                    for espn_id in re.findall(r"\b(\d+) \(", match[2])
                ],
            )
            for match in re.finditer(r"^Item (\d+) \(.*\ncandidates: (.*)$", str(text), re.MULTILINE)
        ]
        usage = TokenUsage(input_tokens=1200, output_tokens=150)
        return RawReply(f"msg_{len(self.calls)}", DEFAULT_MODEL, "end_turn", usage, TriageOutput(items=items))


@dataclass
class DecisionCall:
    kind: str
    league: LeagueRow
    kwargs: dict[str, Any]


class World:
    """The fixture league with every tick collaborator faked, plus knobs the tests turn."""

    def __init__(self, store: Store, state_file: Path) -> None:
        self.store = store
        self.config = CONFIG
        self.state_file = state_file
        self.league = store.leagues.upsert(
            LeagueRow(key="nfl", sport="nfl", espn_league_id=LEAGUE_ID, season=SEASON, team_id=TEAM, as_of=NOW)
        )
        settings = load_league_settings(FIXTURES / "ffl_settings_ppr.json", game="ffl")
        store.settings.upsert(
            LeagueSettingsRow(league_id=self.league.row_id, settings=settings.model_dump(mode="json"), as_of=NOW)
        )
        self.rosters: dict[str, Any] = json.loads((FIXTURES / "ffl_rosters_week4.json").read_text(encoding="utf-8"))
        self.api = FakeEspnApi()
        self.api.serve("mRoster", lambda _request: self.rosters)
        self.api.serve_file("proTeamSchedules_wl", FIXTURES / "ffl_pro_schedule_2026.json")
        self.api.serve_file("mMatchup", FIXTURES / "ffl_matchups.json")
        self.transport = FakeTransport(on_send=self.apply_request)
        self.opener = FakeOpener(fake_runtime(self.api, self.league, transport=self.transport))
        self.flows = FlowRegistry()
        self.flows.register(SwapFlow())
        self.channel = FakeChannel()
        self.calls: list[DecisionCall] = []
        self.decision_payload: LineupPayload | None = SWAP
        self.decision_error: Exception | None = None
        self.registry = DecisionRegistry()
        self.registry.register("nfl", "lineup", self.lineup_decision)
        self.registry.register("nfl", "waivers", self.waivers_decision)
        self.session: EspnSession | Exception = SESSION
        self.syncs: list[tuple[str, ...]] = []
        self.refreshes: list[LeagueContext] = []
        self.refresh_error: Exception | None = None
        self.news_items: list[NewsItem] = list(NEWS_ITEMS)
        self.news_polls: list[Sport] = []
        self.news_error: Exception | None = None
        self.claude = FakeClaude()
        self.advisor_factory: AdvisorFactory = tick_module.advisor_for  # the real one: CONFIG has no key

    # collaborators

    def load_session(self) -> EspnSession:
        if isinstance(self.session, Exception):
            raise self.session
        return self.session

    def clients(self, league: League, session: EspnSession | None) -> EspnClient:
        return self.api.client(league.game, league.espn_league_id, league.season, session=session)

    def syncer(self, store: Store, config: Config, *, session: EspnSession | None, leagues: Any) -> SyncReport:
        self.syncs.append(tuple(leagues))
        self.store.leagues.upsert(self.league.model_copy(update={"as_of": NOW}))
        return SyncReport((), ())

    def refresh(self, ctx: LeagueContext) -> tuple[str, ...]:
        if self.refresh_error is not None:
            raise self.refresh_error
        self.refreshes.append(ctx)
        return (f"{ctx.key}: inputs refreshed for period {ctx.period}",)

    def poll_news(self, sport: Sport) -> NewsPoll:
        """The recorded feeds (plus whatever a test appended to ``news_items``), as :func:`poll_news` shapes them."""
        if self.news_error is not None:
            raise self.news_error
        self.news_polls.append(sport)
        items = tuple(item for item in self.news_items if item.sport == sport)
        feeds = tuple(
            Fetched(tuple(item for item in items if item.source == source), NOW, source, dataset, sport)
            for source, dataset in ((ESPN_NEWS, "news"), (ROTOWIRE, "rss"))
        )
        return NewsPoll(sport, feeds, dedupe(items))

    def use_advisor(self, budget: float = 2.0) -> None:
        """Hand the tick an advisor over the fake Claude instead of the keyless default."""
        self.advisor_factory = lambda store, config: AdvisorClient(store, Llm(daily_budget_usd=budget), self.claude)

    def roster(self, *espn_ids: int) -> None:
        """Put these players on team 1's roster for the week, as a sync would, so the league's news is about them."""
        names = {ALLGEIER: ("Tyler Allgeier", "RB", "ATL"), DANIELS: ("Jayden Daniels", "QB", "WSH")}
        self.store.teams.upsert(TeamRow(league_id=self.league.row_id, team_id=TEAM, name="Fixture Team 1", as_of=NOW))
        for espn_id in espn_ids:
            name, position, pro_team = names[espn_id]
            self.store.players.upsert(
                PlayerRow(sport="nfl", espn_id=espn_id, full_name=name, position=position, pro_team=pro_team, as_of=NOW)
            )
        self.store.rosters.replace(
            self.league.row_id,
            WEEK,
            TEAM,
            [
                RosterEntryRow(
                    league_id=self.league.row_id,
                    scoring_period_id=WEEK,
                    team_id=TEAM,
                    espn_id=espn_id,
                    lineup_slot_id=BENCH,
                    as_of=NOW,
                )
                for espn_id in espn_ids
            ],
        )

    def lineup_decision(self, store: Store, config: Config, league: LeagueRow, **kwargs: Any) -> Any:
        self.calls.append(DecisionCall("lineup", league, kwargs))
        if self.decision_error is not None:
            raise self.decision_error
        if self.decision_payload is None:
            return SimpleNamespace(proposals=(), blocked=(), warnings=())
        row = propose(
            store,
            config,
            league,
            ProposalKind.LINEUP,
            self.decision_payload,
            created_by="decide.fake",
            scoring_period_id=WEEK,
            engine_numbers={"basis": "roster-read-1"},
            deadline=LOCK,
            dedupe_key="lineup:fake",
            now=kwargs["now"],
        )
        return SimpleNamespace(proposals=(row,), blocked=(), warnings=("fake lineup warning",))

    def waivers_decision(self, store: Store, config: Config, league: LeagueRow, *, now: datetime, **kwargs: Any) -> Any:
        self.calls.append(DecisionCall("waivers", league, {"now": now, **kwargs}))
        return SimpleNamespace(proposals=(), blocked=((object(), "cap reached"),), warnings=())

    # the fake league

    def _entries(self) -> list[dict[str, Any]]:
        team = next(team for team in self.rosters["teams"] if team["id"] == TEAM)
        return team["roster"]["entries"]

    def slot(self, espn_id: int) -> int:
        return next(entry["lineupSlotId"] for entry in self._entries() if entry["playerId"] == espn_id)

    def apply_request(self, request: WriteRequest) -> None:
        wanted = {item["playerId"]: item["toLineupSlotId"] for item in request.body["items"]}
        for entry in self._entries():
            if entry["playerId"] in wanted:
                entry["lineupSlotId"] = wanted[entry["playerId"]]

    def propose(
        self,
        payload: LineupPayload = SWAP,
        *,
        kind: ProposalKind = ProposalKind.LINEUP,
        deadline: datetime | None = LOCK,
        approved: bool = False,
        basis: str | None = None,
        now: datetime = EARLY,
        dedupe_key: str | None = None,
    ) -> int:
        row = propose(
            self.store,
            self.config,
            self.league,
            kind,
            payload,
            created_by="test",
            scoring_period_id=WEEK,
            engine_numbers={"basis": basis} if basis else {},
            deadline=deadline,
            dedupe_key=dedupe_key,
            now=now,
        )
        if approved:
            approve(self.store, row.row_id, decided_by="test", now=now)
        return row.row_id

    def status(self, proposal_id: int) -> str:
        return get_proposal(self.store, proposal_id).status

    def tick(self, now: datetime = NOW, **overrides: Any) -> TickReport:
        kwargs: dict[str, Any] = {
            "now": now,
            "session_loader": self.load_session,
            "clients": self.clients,
            "syncer": self.syncer,
            "refresh": self.refresh,
            "news": self.poll_news,
            "advisor_factory": self.advisor_factory,
            "opener": self.opener,
            "channel": self.channel,
            "registry": self.registry,
            "flows": self.flows,
            "options": TickOptions(executor=FAST),
            "state_file": self.state_file,
        }
        kwargs.update(overrides)
        return tick(self.store, self.config, **kwargs)

    def state(self) -> TickState:
        return TickState.load(self.state_file)


@pytest.fixture
def world(tmp_path: Path) -> Iterator[World]:
    with Store.open(tmp_path / "state.db") as store:
        yield World(store, tmp_path / "tick-state.json")


def decisions(report: TickReport, key: str = "nfl") -> list[str]:
    league = next(lg for lg in report.leagues if lg.key == key)
    return [run.name for run in league.decisions]


def keyed(config: Config, *, budget: float = 2.0) -> Config:
    """``config`` with an Anthropic key in its secrets (never sent: the advisor is off or faked) and ``budget``."""
    secrets = Secrets.model_validate({"ANTHROPIC_API_KEY": "sk-test-not-a-real-key"})
    return config.model_copy(update={"secrets": secrets, "llm": Llm(daily_budget_usd=budget)})


def stored_news(world: World) -> dict[str, int]:
    """Every stored NFL item's external id -> row id."""
    since = datetime(2026, 1, 1, tzinfo=UTC)
    return {row.external_id: row.row_id for row in world.store.news.published_since(since, sport="nfl")}


# --- decisions run once per window, proposals are pushed --------------------------------------------------------------


def test_nothing_is_due_before_a_window_opens(world: World) -> None:
    report = world.tick(now=EARLY)
    (league,) = report.leagues
    assert league.period == WEEK and league.skipped is None
    assert league.due == () and league.missed == () and league.decisions == ()
    assert world.calls == [] and world.channel.notices == []
    assert any(d.kind.value == "lineup_lock" and d.at == LOCK for d in league.deadlines)
    assert any(d.kind.value == "waiver_run" for d in league.deadlines)  # Wednesday 3 a.m. ET from the settings
    assert world.state().last_tick == EARLY


def test_a_due_window_runs_its_decisions_once_and_pushes_the_proposal(world: World) -> None:
    report = world.tick()
    (league,) = report.leagues
    (window,) = league.due
    assert window.kind is WindowKind.PRE_LOCK and window.closes_at == LOCK
    assert decisions(report) == ["nfl:lineup"]  # a lock window runs the lineup kinds, not waivers
    (call,) = world.calls
    assert call.kwargs["now"] == NOW and call.kwargs["schedule"] is not None
    assert call.kwargs["opponent_team_id"] == 2  # from mMatchup: team 1 plays team 2 in matchup period 4
    (run,) = league.decisions
    assert run.error is None and len(run.proposals) == 1 and run.warnings == ("fake lineup warning",)
    (proposal_id,) = run.proposals
    assert league.notified == (proposal_id,)
    assert [notice.proposal_id for notice in world.channel.notices] == [proposal_id]
    assert world.state().runs == {window.key: NOW}
    assert any("decision nfl:lineup" in line for line in report.lines())

    again = world.tick(now=NOW + timedelta(minutes=10))
    (league_again,) = again.leagues
    assert league_again.due == () and league_again.decisions == () and len(world.calls) == 1
    assert len(world.channel.notices) == 1 and league_again.notified == ()


def test_the_period_opening_runs_every_registered_decision(world: World) -> None:
    opening = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)  # Wednesday before week 4's Thursday game
    world.decision_payload = None
    report = world.tick(now=opening)
    (league,) = report.leagues
    assert [w.kind for w in league.due] == [WindowKind.PERIOD_OPEN]
    assert decisions(report) == ["nfl:lineup", "nfl:waivers"]
    waivers = next(run for run in league.decisions if run.name == "nfl:waivers")
    assert waivers.blocked == ("cap reached",)
    assert [call.kind for call in world.calls] == ["lineup", "waivers"]
    assert "opponent_team_id" not in world.calls[1].kwargs  # only the keywords the decision accepts


def test_a_failed_decision_is_alerted_and_retried_next_tick(world: World) -> None:
    world.decision_error = RuntimeError("no projections")
    report = world.tick()
    (league,) = report.leagues
    (run,) = league.decisions
    assert run.error == "RuntimeError: no projections" and not report.ok
    assert world.channel.alerts() == ["Decision nfl:lineup failed"]
    assert world.state().runs == {}

    world.decision_error = None
    again = world.tick(now=NOW + timedelta(minutes=10))
    assert decisions(again) == ["nfl:lineup"] and len(world.calls) == 2 and again.ok


def test_a_failed_push_is_a_warning_and_the_proposal_is_pushed_next_tick(world: World) -> None:
    world.channel.fail_push = True
    report = world.tick()
    (league,) = report.leagues
    assert league.notified == () and any("not pushed" in w for w in league.warnings)
    world.channel.fail_push = False
    again = world.tick(now=NOW + timedelta(minutes=10))
    (league,) = again.leagues
    assert len(league.notified) == 1 and len(world.channel.notices) == 1


# --- order: reconcile, expire, session, auto-approve, execute ---------------------------------------------------------


def test_reconcile_and_expire_come_before_execution(world: World) -> None:
    world.decision_payload = None
    stale = world.propose(SWAP, approved=True, dedupe_key="stale")
    begin_execution(world.store, stale, get_proposal(world.store, stale).execution_token or "", now=EARLY)
    world.store.executions.insert(
        ExecutionRow(proposal_id=stale, mode="api", status="running", started_at=NOW - STALE_EXECUTION)
    )
    missed = world.propose(OTHER, approved=True, deadline=NOW - timedelta(minutes=1), dedupe_key="late")
    live = world.propose(SWAP, approved=True, dedupe_key="live")

    report = world.tick()
    assert report.reconciled == (stale,) and world.status(stale) == "failed"
    (execution,) = world.store.executions.for_proposal(stale)
    assert execution.status == "unknown"
    assert report.expired == (missed,) and world.status(missed) == "expired"
    (outcome,) = report.executions
    assert outcome.proposal_id == live and outcome.ok and world.status(live) == "verified"
    assert world.slot(HUBBARD) == BENCH and world.slot(ALLGEIER) == FLEX
    assert len(world.transport.sent) == 1
    assert world.channel.alerts() == [
        f"Check ESPN: NFL lineup change #{stale} may be half done",
        f"Missed: NFL lineup change #{missed} didn't run",
    ]
    assert any(message.title == f"✅ Done: NFL lineup change #{live}" for message in world.channel.sent)
    assert report.ok


def test_auto_bench_fires_only_inside_t_minus_15_and_rejects_its_alternative(world: World) -> None:
    world.decision_payload = None
    bench = world.propose(SWAP, kind=ProposalKind.BENCH_INACTIVE, basis="read-1", dedupe_key="bench")
    alternative = world.propose(OTHER, kind=ProposalKind.LINEUP, basis="read-1", dedupe_key="full")
    unrelated = world.propose(OTHER, kind=ProposalKind.LINEUP, basis="read-2", dedupe_key="other")
    assert get_proposal(world.store, bench).policy == "auto"

    early = world.tick(now=LOCK - timedelta(minutes=20))
    assert early.auto_approved == () and early.executions == ()
    assert world.status(bench) == "proposed"

    due = world.tick(now=LOCK - timedelta(minutes=14))
    assert due.auto_approved == (bench,)
    (outcome,) = due.executions
    assert outcome.proposal_id == bench and outcome.ok and outcome.rejected == (alternative,)
    assert world.status(bench) == "verified"
    rejected = get_proposal(world.store, alternative)
    assert rejected.status == "rejected" and rejected.decided_by == DECIDED_BY_TICK
    assert world.status(unrelated) == "proposed"
    assert get_proposal(world.store, bench).decided_by == "auto"


def test_paused_blocks_auto_approval_and_execution(world: World) -> None:
    world.decision_payload = None
    bench = world.propose(SWAP, kind=ProposalKind.BENCH_INACTIVE, dedupe_key="bench")
    live = world.propose(OTHER, approved=True, dedupe_key="live")
    pause("testing", now=EARLY)
    report = world.tick(now=LOCK - timedelta(minutes=10))
    assert report.paused is not None and "testing" in report.paused
    assert report.auto_approved == () and report.executions == () and report.would_execute == (live,)
    assert world.status(bench) == "proposed" and world.status(live) == "approved"
    assert world.transport.sent == [] and world.opener.calls == []
    assert any(line.startswith("PAUSED") for line in report.lines())


def test_no_execute_option_lists_what_would_run_and_sends_nothing(world: World) -> None:
    world.decision_payload = None
    live = world.propose(SWAP, approved=True)
    report = world.tick(options=TickOptions(execute=False, executor=FAST))
    assert report.executions == () and report.would_execute == (live,) and world.transport.sent == []
    assert world.status(live) == "approved"


def test_executions_run_soonest_deadline_first_and_failures_are_alerted(world: World) -> None:
    world.decision_payload = None
    later = world.propose(OTHER, approved=True, deadline=LOCK + timedelta(hours=3), dedupe_key="later")
    sooner = world.propose(SWAP, approved=True, deadline=LOCK, dedupe_key="sooner")
    report = world.tick()
    assert [o.proposal_id for o in report.executions] == [sooner, later]
    assert report.executions[0].ok
    failed = report.executions[1]
    assert not failed.ok and "preconditions failed" in failed.detail  # Allgeier is at FLEX now, not on the bench
    assert world.status(later) == "failed" and not report.ok
    assert world.channel.alerts() == [f"❌ Failed: NFL lineup change #{later}"]
    assert len(world.transport.sent) == 1


# --- session ----------------------------------------------------------------------------------------------------------


def test_check_session_maps_the_exceptions_to_verdicts() -> None:
    def missing() -> EspnSession:
        raise NotLoggedInError("no browser profile yet; run `fm login`")

    def broken() -> EspnSession:
        raise BrowserError("no Edge or Chrome")

    assert check_session(missing, now=NOW).verdict is SessionVerdict.MISSING
    assert check_session(broken, now=NOW).verdict is SessionVerdict.UNAVAILABLE
    ok = check_session(lambda: SESSION, now=NOW)
    assert ok.verdict is SessionVerdict.OK and ok.usable and "200 days left" in ok.detail
    expiring = check_session(lambda: SESSION.__class__("s", "{w}", NOW + timedelta(days=2)), now=NOW)
    assert expiring.verdict is SessionVerdict.EXPIRING and expiring.usable
    expired = check_session(lambda: SESSION.__class__("s", "{w}", NOW - timedelta(days=1)), now=NOW)
    assert expired.verdict is SessionVerdict.EXPIRED and not expired.usable
    assert not SessionVerdict.MISSING.usable and not SessionVerdict.UNAVAILABLE.usable


def test_a_missing_session_is_alerted_once_and_blocks_sync_and_execution_but_not_decisions(world: World) -> None:
    world.session = NotLoggedInError("no browser profile yet; run `fm login`")
    world.store.leagues.upsert(world.league.model_copy(update={"as_of": NOW - timedelta(days=2)}))
    live = world.propose(SWAP, approved=True, dedupe_key="live")
    report = world.tick()
    assert report.session.verdict is SessionVerdict.MISSING
    assert world.channel.alerts() == ["ESPN session missing"]
    assert world.syncs == [] and report.executions == () and report.would_execute == (live,)
    assert decisions(report) == ["nfl:lineup"]  # the pro schedule is public; decisions still run on stored data
    assert world.calls[0].kwargs["opponent_team_id"] is None  # matchups need the session

    again = world.tick(now=NOW + timedelta(minutes=10))
    assert world.channel.alerts() == ["ESPN session missing"]  # not repeated within ALERT_REPEAT
    assert again.session.verdict is SessionVerdict.MISSING
    world.tick(now=NOW + ALERT_REPEAT + timedelta(minutes=1))  # past the lock: other alerts fire, the session's again
    assert world.channel.alerts().count("ESPN session missing") == 2


def test_an_expiring_session_warns_but_still_executes(world: World) -> None:
    world.decision_payload = None
    world.session = EspnSession(espn_s2="s", swid="{w}", expires_at=NOW + timedelta(days=3))
    live = world.propose(SWAP, approved=True)
    report = world.tick()
    assert report.session.verdict is SessionVerdict.EXPIRING
    assert world.channel.alerts() == ["ESPN session expiring"]
    assert [o.proposal_id for o in report.executions] == [live] and world.status(live) == "verified"


def test_the_session_verdict_is_recorded_for_fm_health_without_any_cookie(world: World) -> None:
    """fm.health reads this record instead of launching a second browser on the profile the tick uses."""
    world.decision_payload = None
    world.session = EspnSession(espn_s2="s2-cookie-value", swid="{swid-value}", expires_at=NOW + timedelta(days=3))
    world.tick()
    record = world.state().session
    assert record is not None and set(record) == {"status", "expires_at", "checked_at", "detail"}
    assert (record["status"], record["expires_at"], record["checked_at"]) == (
        "expiring",
        (NOW + timedelta(days=3)).isoformat(),
        NOW.isoformat(),
    )
    assert "expires soon" in (record["detail"] or "")
    raw = world.state_file.read_text(encoding="utf-8")
    assert "s2-cookie-value" not in raw and "swid-value" not in raw
    world.session = NotLoggedInError("no ESPN session in the profile; run `fm login`")
    world.tick(now=NOW + timedelta(minutes=10))
    record = world.state().session
    assert record is not None and record["status"] == "missing" and record["expires_at"] is None
    assert record["checked_at"] == (NOW + timedelta(minutes=10)).isoformat()
    assert record["detail"] == "no ESPN session in the profile; run `fm login`"


# --- sync, inputs, missed windows -------------------------------------------------------------------------------------


def test_sync_runs_only_when_the_stored_state_is_stale(world: World) -> None:
    world.decision_payload = None
    world.tick()
    assert world.syncs == []
    world.store.leagues.upsert(world.league.model_copy(update={"as_of": NOW - timedelta(hours=2)}))
    report = world.tick(now=NOW + timedelta(minutes=10))
    assert world.syncs == [("nfl",)] and report.leagues[0].synced
    world.store.leagues.upsert(world.league.model_copy(update={"as_of": NOW - timedelta(hours=2)}))
    world.tick(now=NOW + timedelta(minutes=20), options=TickOptions(sync=False, executor=FAST))
    assert world.syncs == [("nfl",)]


def test_a_league_never_synced_is_skipped_without_a_session(world: World) -> None:
    world.session = NotLoggedInError("no session")
    world.store.db.execute("DELETE FROM league_settings")
    report = world.tick()
    (league,) = report.leagues
    assert league.skipped is not None and "fm sync" in league.skipped and league.decisions == ()


def test_inputs_are_refreshed_on_a_schedule_and_a_failure_does_not_stop_the_tick(world: World) -> None:
    world.decision_payload = None
    report = world.tick(now=EARLY)
    assert len(world.refreshes) == 1 and world.refreshes[0].period == WEEK
    assert "nfl: inputs refreshed for period 4" in report.leagues[0].notes
    world.tick(now=EARLY + timedelta(minutes=10))
    assert len(world.refreshes) == 1
    world.refresh_error = RuntimeError("nflverse down")
    report = world.tick(now=EARLY + INPUTS_EVERY["nfl"] + timedelta(minutes=1))
    (league,) = report.leagues
    assert len(world.refreshes) == 1 and any("not refreshed: nflverse down" in w for w in league.warnings)
    world.refresh_error = None
    world.tick(now=EARLY + INPUTS_EVERY["nfl"] + timedelta(minutes=11))
    assert len(world.refreshes) == 2


# --- news: polled, stored, triaged (ROADMAP #58) ----------------------------------------------------------------------


def test_news_is_polled_stored_and_triaged_on_its_own_cadence(world: World) -> None:
    world.decision_payload = None
    world.roster(ALLGEIER, DANIELS)
    world.use_advisor()

    report = world.tick(now=EARLY)
    (league,) = report.leagues
    assert league.skipped is None and world.news_polls == ["nfl"]
    items = stored_news(world)
    assert len(items) == 10 and world.store.news.untriaged() == []  # every item read is marked, relevant or not
    assert len(world.claude.calls) == 1  # the two relevant items fit one chunk
    notes = "\n".join(league.notes)
    assert "nfl: nfl news: 10 items (espn_news 5 fresh, rotowire 5 fresh); news: 10 stored (" in notes
    assert "news triage (nfl): 10 read, 2 relevant, 10 triaged, 2 signals, 0 refused, 1 call" in notes
    assert any("team 2 has no roster in scoring period 4" in w for w in league.warnings)  # the opponent from mMatchup
    (allgeier,) = world.store.news_signals.for_player("nfl", ALLGEIER)
    (daniels,) = world.store.news_signals.for_player("nfl", DANIELS)
    assert allgeier.news_item_id == items[PICKUPS_STORY]
    daniels_item = world.store.news.get(daniels.news_item_id)
    assert daniels_item is not None and daniels_item.source == ROTOWIRE  # matched by the name in its title
    assert daniels_item.title.startswith("Jayden Daniels:")
    for signal in (allgeier, daniels):  # bounded: the proposal -0.9 x 1.5 is stored as -0.3 x 1.0, cited by the item
        assert signal.p_active_delta == -NEWS_BOUND and signal.confidence == 1.0
        assert signal.source_url and signal.created_at == EARLY and signal.kind == "injury"
    assert world.state().news == {"nfl": EARLY}
    (usage,) = world.store.llm_usage.since(EARLY - timedelta(days=1))
    assert usage.worker == "news_triage" and usage.league_id == world.league.row_id
    assert len(world.refreshes) == 1  # the first tick refreshes the inputs anyway
    assert any("news triage (nfl)" in line for line in report.lines())

    world.tick(now=EARLY + timedelta(minutes=10))
    assert world.news_polls == ["nfl"] and len(world.claude.calls) == 1  # not due: NEWS_EVERY is an hour for the NFL

    world.news_items.append(
        NewsItem(
            source=ESPN_NEWS,
            external_id="60000001",
            sport="nfl",
            title="Falcons' Tyler Allgeier (ankle) doubtful for Sunday",
            published_at=datetime(2026, 10, 5, 23, 30, tzinfo=UTC),
            body="Allgeier did not practice Friday.",
            url="https://www.espn.com/nfl/story/_/id/60000001/falcons-tyler-allgeier-ankle-doubtful-sunday",
            espn_ids=(ALLGEIER,),
            player_names=("Tyler Allgeier",),
        )
    )
    again = world.tick(now=EARLY + NEWS_EVERY["nfl"] + timedelta(minutes=1))
    (league,) = again.leagues
    assert world.news_polls == ["nfl", "nfl"] and len(stored_news(world)) == 11
    assert any("news: 1 stored (0 about no known player), 10 already stored, 0 repeats" in n for n in league.notes)
    assert len(world.claude.calls) == 2 and len(world.store.news_signals.for_player("nfl", ALLGEIER)) == 2
    assert len(world.refreshes) == 2  # new signals: the availability rows are rewritten now, not at INPUTS_EVERY
    assert world.state().news == {"nfl": EARLY + NEWS_EVERY["nfl"] + timedelta(minutes=1)}


def test_a_tick_without_a_key_stores_the_news_untriaged_and_still_runs_every_decision(world: World) -> None:
    world.decision_payload = None
    world.roster(ALLGEIER, DANIELS)
    opening = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)  # the period opening runs every registered decision

    report = world.tick(now=opening)
    (league,) = report.leagues
    assert league.skipped is None and decisions(report) == ["nfl:lineup", "nfl:waivers"]
    assert len(stored_news(world)) == 10 and len(world.store.news.untriaged()) == 10  # stored, waiting for a key
    assert world.store.news_signals.for_player("nfl", ALLGEIER) == [] and world.claude.calls == []
    (note,) = [note for note in league.notes if "not triaged" in note]
    assert note == (
        "nfl: 10 news items not triaged: ANTHROPIC_API_KEY is not set in .env, so the advisor is off (fm config check)"
    )
    assert world.state().news == {"nfl": opening}
    assert any("not triaged: ANTHROPIC_API_KEY" in line for line in report.lines())

    world.config = keyed(CONFIG, budget=0)  # a key, but a cap of 0 refuses every call: off, and says so
    again = world.tick(now=opening + NEWS_EVERY["nfl"] + timedelta(minutes=1))
    (note,) = [note for note in again.leagues[0].notes if "not triaged" in note]
    assert note == "nfl: 10 news items not triaged: [llm] daily_budget_usd is 0 in config.toml, so the advisor is off"
    assert world.claude.calls == [] and len(world.store.news.untriaged()) == 10


def test_news_failures_are_warnings_and_never_skip_the_league(world: World) -> None:
    world.decision_payload = None
    world.roster(ALLGEIER)

    world.news_error = RuntimeError("feeds down")
    report = world.tick()
    (league,) = report.leagues
    assert league.skipped is None and decisions(report) == ["nfl:lineup"]
    assert any(w == "nfl: news not polled: feeds down" for w in league.warnings)
    assert "nfl" not in world.state().news and stored_news(world) == {}  # polled again next tick

    world.news_error = None
    world.use_advisor()
    world.claude.error = AdvisorError("Claude API call failed: 503")
    again = world.tick(now=NOW + timedelta(minutes=10))
    (league,) = again.leagues
    assert league.skipped is None and world.state().news == {"nfl": NOW + timedelta(minutes=10)}
    assert any(w == "nfl: news triage stopped: API failure: Claude API call failed: 503" for w in league.warnings)
    assert world.store.news_signals.for_player("nfl", ALLGEIER) == []
    assert [item.external_id for item in world.store.news.untriaged()] == [PICKUPS_STORY]  # the relevant one waits
    assert len(world.refreshes) == 1  # no new signal, no forced refresh

    def broken(store: Store, config: Config) -> AdvisorClient | None:
        raise RuntimeError("no SDK here")

    world.advisor_factory = broken
    third = world.tick(now=NOW + NEWS_EVERY["nfl"] + timedelta(minutes=11))
    (league,) = third.leagues
    assert league.skipped is None and decisions(third) == []  # the lock has passed; nothing else is due
    assert any(n == "nfl: 1 news item not triaged: the advisor could not be built (no SDK here)" for n in league.notes)
    assert third.ok


def test_one_leagues_triage_reads_for_every_league_of_the_sport(world: World) -> None:
    """The triage marks every item it reads, so league nfl's run asks about nfl2's players too: McConkey, on nfl2's
    roster only, gets his signal although only nfl ticks."""
    mcconkey = 4612826
    world.config = Config.model_validate(
        {
            "league": [
                {"key": "nfl", "sport": "nfl", "espn_league_id": LEAGUE_ID, "season": SEASON, "team_id": TEAM},
                {"key": "nfl2", "sport": "nfl", "espn_league_id": 7654321, "season": SEASON, "team_id": 3},
            ]
        }
    )
    other = world.store.leagues.upsert(
        LeagueRow(key="nfl2", sport="nfl", espn_league_id=7654321, season=SEASON, team_id=3, as_of=NOW)
    )
    world.store.teams.upsert(TeamRow(league_id=other.row_id, team_id=3, name="Other Team", as_of=NOW))
    world.store.players.upsert(
        PlayerRow(sport="nfl", espn_id=mcconkey, full_name="Ladd McConkey", position="WR", pro_team="LAC", as_of=NOW)
    )
    entry = RosterEntryRow(
        league_id=other.row_id, scoring_period_id=WEEK, team_id=3, espn_id=mcconkey, lineup_slot_id=BENCH, as_of=NOW
    )
    world.store.rosters.replace(other.row_id, WEEK, 3, [entry])
    world.roster(ALLGEIER)
    world.use_advisor()
    world.decision_payload = None

    report = world.tick(now=EARLY, options=TickOptions(executor=FAST, leagues=("nfl",)))
    assert [league.key for league in report.leagues] == ["nfl"]
    (params,) = world.claude.calls
    context = params["system"][1]["text"]
    assert "- nfl2 (nfl): our team 3, no opponent given" in context and "Ladd McConkey, WR, LAC; roster" in context
    assert "news triage (nfl): 10 read, 2 relevant, 10 triaged, 2 signals" in "\n".join(report.leagues[0].notes)
    (signal,) = world.store.news_signals.for_player("nfl", mcconkey)
    assert signal.p_active_delta == -NEWS_BOUND and world.store.news.untriaged() == []


def test_the_news_step_is_skipped_without_a_poller(world: World) -> None:
    world.decision_payload = None
    report = world.tick(news=None)
    (league,) = report.leagues
    assert world.news_polls == [] and stored_news(world) == {} and world.state().news == {}
    assert not any("news" in note for note in league.notes)


def test_the_first_tick_after_a_gap_reports_the_windows_it_missed(world: World) -> None:
    world.decision_payload = None
    state = TickState(last_tick=THURSDAY)
    state.save(world.state_file)
    report = world.tick(now=EARLY)
    (league,) = report.leagues
    assert [w.kind for w in league.missed] == [WindowKind.PERIOD_OPEN, WindowKind.PRE_LOCK]
    assert league.missed[1].closes_at == datetime(2026, 10, 2, 0, 15, tzinfo=UTC)  # Thursday night's kickoff
    assert world.channel.alerts() == ["Missed: NFL period open check", "Missed: NFL pre lock check"]
    assert all(message.link and "fantasy.espn.com" in message.link for message in world.channel.sent)
    again = world.tick(now=EARLY + timedelta(minutes=10))
    assert again.leagues[0].missed == () and len(world.channel.sent) == 2


def test_refresh_nfl_inputs_writes_availability_rows_with_the_practice_trend(world: World) -> None:
    store = world.store
    ids = plugin_for("nfl").ids
    detroit = next(team for team, code in ids.pro_teams.items() if code == "DET")
    store.players.upsert(
        PlayerRow(sport="nfl", espn_id=GIBBS, full_name="Jahmyr Gibbs", position="RB", pro_team_id=detroit, as_of=NOW)
    )
    store.player_ids.upsert(
        PlayerIdRow(sport="nfl", espn_id=GIBBS, source=GSIS, source_id="00-0039165", origin="test", as_of=NOW)
    )
    store.teams.upsert(TeamRow(league_id=world.league.row_id, team_id=TEAM, name="Fixture Team 1", as_of=NOW))
    store.rosters.replace(
        world.league.row_id,
        WEEK,
        TEAM,
        [
            RosterEntryRow(
                league_id=world.league.row_id,
                scoring_period_id=WEEK,
                team_id=TEAM,
                espn_id=GIBBS,
                lineup_slot_id=RB,
                as_of=NOW,
            )
        ],
    )
    frame = pl.DataFrame(
        {
            "season": [SEASON, SEASON],
            "week": [WEEK, 3],
            "gsis_id": ["00-0039165", "00-0039165"],
            "practice_status": ["Limited Participation in Practice", "Did Not Participate In Practice"],
        }
    )
    observed = datetime(2026, 10, 2, 20, 0, tzinfo=UTC)

    class FakeNflverse:
        def injuries(self, season: int, **options: Any) -> Fetched[pl.DataFrame]:
            assert season == SEASON
            return Fetched(frame, observed, "nflverse", "injuries", str(season))

    schedule = world.api.client_for(world.league).pro_schedule().data
    ctx = LeagueContext(
        store=store,
        config=CONFIG,
        league=CONFIG.league("nfl"),
        row=world.league,
        settings=load_league_settings(FIXTURES / "ffl_settings_ppr.json", game="ffl"),
        plugin=plugin_for("nfl"),
        schedule=schedule,
        period=WEEK,
        now=NOW,
    )
    notes = refresh_nfl_inputs(ctx, source=FakeNflverse())  # type: ignore[arg-type]
    assert notes[0].startswith("nfl: availability assessed for 1 players with 1 practice reports")
    row = store.availability.get("nfl", GIBBS, SEASON, WEEK)
    assert row is not None and row.has_game and row.game_time is not None
    practice = row.inputs["practice"]
    assert len(practice["reports"]) == 1 and practice["latest"] == "LP"  # week 3's row was skipped
    assert 0.0 < row.p_active <= 1.0


# --- the bot hook and the state file ----------------------------------------------------------------------------------


def test_execute_on_approval_runs_an_approval_near_its_deadline_at_once(world: World) -> None:
    near = world.propose(SWAP, approved=True, deadline=NOW + timedelta(minutes=20), dedupe_key="near")
    far = world.propose(OTHER, approved=True, deadline=NOW + timedelta(hours=3), dedupe_key="far")
    outcomes: list[Any] = []
    hook = execute_on_approval(
        world.store, opener=world.opener, flows=world.flows, options=FAST, clock=lambda: NOW, on_outcome=outcomes.append
    )
    hook(DecisionResult(far, "approve", ok=True, detail=""))
    assert outcomes == [] and world.status(far) == "approved"
    hook(DecisionResult(near, "reject", ok=True, detail=""))
    hook(DecisionResult(near, "approve", ok=False, detail="refused"))
    assert outcomes == []
    hook(DecisionResult(near, "approve", ok=True, detail=""))
    (outcome,) = outcomes
    assert outcome.proposal_id == near and outcome.ok and world.status(near) == "verified"


def test_state_round_trips_and_tolerates_a_broken_file(tmp_path: Path) -> None:
    path = tmp_path / "tick-state.json"
    assert TickState.load(path) == TickState()
    state = TickState(last_tick=NOW, runs={"nfl:pre_lock:x": NOW}, notified={3}, alerts={"session:missing": NOW})
    state.inputs["nfl"] = NOW
    state.news["nfl"] = NOW - timedelta(minutes=5)
    state.session = {"status": "ok", "expires_at": None, "checked_at": NOW.isoformat(), "detail": "fine"}
    state.save(path)
    assert TickState.load(path) == state
    state.prune(NOW + timedelta(days=30))
    assert state.runs == {} and state.alerts == {} and state.notified == {3}
    path.write_text("{not json", encoding="utf-8")
    assert TickState.load(path) == TickState()
    path.write_text(json.dumps({"last_tick": "2026-10-04T15:00:00"}), encoding="utf-8")  # naive: refused
    assert TickState.load(path) == TickState()


# --- fm tick ----------------------------------------------------------------------------------------------------------


def _root() -> None:
    pass


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    schedule_cmd.register(root)
    return root


def test_fm_tick_prints_the_report_and_passes_its_options(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_tick(store: Store, config: Config, **kwargs: Any) -> TickReport:
        seen.update(kwargs)
        return world.tick(now=EARLY)

    def no_channel(config: Config, *, client: httpx.Client | None = None) -> Any:
        raise NotifyError("TELEGRAM_BOT_TOKEN is not set")

    monkeypatch.setattr(schedule_cmd, "load_config", lambda: CONFIG)
    monkeypatch.setattr(schedule_cmd, "open_channel", no_channel)
    monkeypatch.setattr(schedule_cmd, "tick", fake_tick)
    args = ["tick", "--in-process", "--no-execute", "--no-sync", "--league", "nfl"]
    result = runner.invoke(cli(), args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "note: no phone channel" in result.output and "tick at 2026-10-04 15:00 UTC" in result.output
    assert "nfl (nfl): period 4" in result.output
    assert seen["options"] == TickOptions(execute=False, sync=False, leagues=("nfl",)) and seen["channel"] is None

    result = runner.invoke(cli(), ["tick", "--in-process", "--league", "nope"], catch_exceptions=False)
    assert result.exit_code == 1 and "no league 'nope'" in result.output


def test_fm_tick_exits_1_when_a_decision_failed(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    world.decision_error = RuntimeError("boom")
    monkeypatch.setattr(schedule_cmd, "load_config", lambda: CONFIG)
    monkeypatch.setattr(schedule_cmd, "open_channel", lambda config: world.channel)
    monkeypatch.setattr(schedule_cmd, "tick", lambda store, config, **kwargs: world.tick())
    result = runner.invoke(cli(), ["tick", "--in-process"], catch_exceptions=False)
    assert result.exit_code == 1 and "FAILED: RuntimeError: boom" in result.output


def test_the_real_tick_imports_register_the_nfl_decisions() -> None:
    from fm.decide import registry

    assert {entry.kind for entry in registry.registered("nfl")} >= {"lineup", "waivers"}
    assert tick_module.DECIDED_BY_TICK == "tick"


def test_the_real_tick_imports_register_the_nba_decisions_and_streaming_sees_the_opponent() -> None:
    from fm.decide import registry

    assert {entry.kind for entry in registry.registered("nba")} >= {"lineup_daily", "streaming"}
    assert {"lineup", "lineup_daily", "streaming"} <= tick_module.OPPONENT_DECISIONS
    assert "waivers" not in tick_module.OPPONENT_DECISIONS


def test_the_opponent_is_read_once_per_league_and_tick_for_every_decision_that_needs_it(world: World) -> None:
    seen: list[int | None] = []

    def streaming(store: Store, config: Config, league: LeagueRow, *, opponent_team_id: int | None, **_: Any) -> Any:
        seen.append(opponent_team_id)
        return SimpleNamespace(proposals=(), blocked=(), warnings=())

    world.registry.register("nfl", "streaming", streaming)
    world.decision_payload = None
    report = world.tick(now=datetime(2026, 9, 30, 12, 0, tzinfo=UTC))  # the period opening runs every decision
    assert {"nfl:lineup", "nfl:streaming"} <= set(decisions(report))
    assert seen == [2] and world.calls[0].kwargs["opponent_team_id"] == 2
    assert world.api.reads("mMatchup") == 1


def test_a_calendar_error_resolving_the_matchup_week_leaves_the_decision_without_an_opponent(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> int:
        raise CalendarError("/data/calendars/fba_2027.json: cannot be read (JSONDecodeError)")

    monkeypatch.setattr(tick_module, "matchup_period_of", broken)
    report = world.tick()
    (run,) = next(lg for lg in report.leagues if lg.key == "nfl").decisions
    assert run.error is None  # the decision still ran
    assert world.calls[0].kwargs["opponent_team_id"] is None
    assert any("no opponent (matchup week unknown" in warning for warning in report.warnings)
