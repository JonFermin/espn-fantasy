"""The tick: one cheap, idempotent run every few minutes that works out what is due (DESIGN section 13, ROADMAP #29).

``fm tick`` (scheduled by :mod:`fm.jobs.scheduler_windows` or :mod:`fm.jobs.scheduler_macos`) calls :func:`tick`,
which does, in this order:

1. **Reconcile** what a killed process left ``executing`` (:func:`fm.executor.reconcile_executions`, once a run is
   :data:`fm.executor.STALE_EXECUTION` old; a run going in ``fm bot`` is never touched) and **expire** every proposal
   whose deadline has passed (:func:`fm.proposals.expire_due`). An approved or ``auto`` proposal that expired
   unexecuted is a missed window and is alerted.
2. **Check the session**: :func:`fm.espn.auth.load_session` raises ``NotLoggedInError`` when no session is saved,
   and ``EspnSession.status()`` is only OK / EXPIRING / EXPIRED, so the tick maps the exception to its own
   :attr:`SessionVerdict.MISSING` (and a browser that will not start to ``UNAVAILABLE``) and alerts, at most once per
   :data:`ALERT_REPEAT` per verdict. Without a usable session nothing syncs and nothing executes.
3. **Per league** (every ``[[league]]`` in ``config.toml``): sync when the stored state is older than
   :data:`SYNC_MAX_AGE` (:func:`fm.jobs.sync.sync`); read the sport's pro schedule once per tick; find the current
   period between syncs with ``plugin.scoring_period_at(now, schedule)`` (periods turn at 03:00 ET,
   :data:`fm.sports.base.PERIOD_TURN`); compute the deadlines and decision windows (:mod:`fm.jobs.deadlines`: every
   lock from the pro schedule, international and holiday slots included, the NBA first-tip add cutoff, the waiver
   run from the league settings); alert the windows missed since the previous tick; refresh the availability inputs
   (:func:`refresh_inputs`: nflverse practice reports for the NFL, the official injury report and the DARKO blend for
   the NBA, a few times a day); run each registered decision whose window is due, **once** per window (the run is
   recorded in the tick's state file); push every new proposal to the phone with Approve / Reject buttons
   (:func:`fm.notify.notify_proposal`).
4. **Auto-approve** at T-15 (:func:`fm.proposals.auto_approve_due`, which fires only inside :data:`AUTO_LEAD` and
   never while paused), then **execute** every approved proposal whose deadline has not passed, soonest deadline
   first, through :func:`fm.executor.execute`, the one write path. Nothing executes while paused
   (:func:`fm.proposals.is_paused`). A ``bench_inactive`` and a ``lineup`` proposal from one lineup run are
   alternatives against one roster read (the same ``engine_numbers["basis"]``), so once one is verified the other is
   rejected (:func:`reject_alternatives`) rather than left to auto-approve at T-15 and fail. Failures are alerted
   with a link to the team page.
5. **Record** the tick (``tick-state.json`` under the config dir: last run, windows run, proposals notified, alerts
   sent, inputs refreshed), so the next tick knows what it already did and what closed in between.

Decision modules register at import (:mod:`fm.decide.registry` has no discovery helper), so this module imports
:mod:`fm.decide.lineup` and :mod:`fm.decide.waivers`. A decision is called as ``fn(store, config, league_row,
schedule=..., now=..., opponent_team_id=...)`` with only the keywords its signature accepts; its result's
``proposals``, ``blocked`` and ``warnings`` are read when present. The lineup decision's win-probability switch needs
``opponent_team_id``, read from ``EspnClient.matchups`` for the current matchup period.

``fm bot`` records decisions and nothing more; :func:`execute_on_approval` is the ``on_result`` hook for
:func:`fm.notify.listen` that executes an approval at once when its deadline is within :data:`NEAR_DEADLINE`, instead
of waiting for the next tick. Every collaborator (session loader, ESPN client factory, sync, inputs refresh, runtime
opener, phone channel, decision and flow registries, clock) is injectable, so tests run the whole tick offline.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import fm.decide.lineup  # noqa: F401  (registers the NFL lineup decision)
import fm.decide.lineup_daily  # noqa: F401  (registers the NBA daily lineup decision)
import fm.decide.offers  # noqa: F401  (registers the incoming trade offer decision, both sports)
import fm.decide.streaming  # noqa: F401  (registers the NBA streaming decision)
import fm.decide.trades  # noqa: F401  (registers the outgoing trade finder, both sports; off unless configured)
import fm.decide.waivers  # noqa: F401  (registers the NFL waiver decision)
from fm import paths
from fm.browser.flows import FlowRegistry
from fm.browser.selectors import team_page_url
from fm.browser.session import BrowserError
from fm.config import Config, League, Sport
from fm.decide import registry as decide_registry
from fm.decide.registry import DecisionRegistry, Registration
from fm.espn.auth import AuthError, EspnSession, NotLoggedInError, SessionStatus, load_session
from fm.espn.calendar import CalendarError, matchup_period_of
from fm.espn.client import EspnClient, EspnClientError
from fm.espn.settings import LeagueSettings
from fm.executor import (
    ExecutionResult,
    ExecutorError,
    ExecutorOptions,
    RuntimeOpener,
    execute,
    live_opener,
    reconcile_executions,
)
from fm.jobs.deadlines import (
    DEFAULT_HORIZON,
    LINEUP_DECISIONS,
    Deadline,
    RunWindow,
    due_windows,
    missed_windows,
    run_windows,
    upcoming,
)
from fm.jobs.sync import SyncError, SyncReport, sync
from fm.model.availability import assess_stored, practice_from_nflverse
from fm.model.ids import Crosswalk, CrosswalkError
from fm.model.projections import BlendWeights
from fm.model.relevance import opponent_from
from fm.notify import DecisionResult, Message, NotifyChannel, NotifyError, notify_proposal, send_alert
from fm.proposals import (
    ProposalError,
    auto_approve_due,
    expire_due,
    parse_payload,
    pause_state,
    reject,
)
from fm.proposals.policy import as_utc, stored_settings
from fm.sources.base import SourceError
from fm.sources.nba_injuries import NbaInjuriesSource
from fm.sources.nflverse import NflverseSource
from fm.sports.base import ScheduleLike, SportPlugin, fantasy_day, plugin_for
from fm.store import LeagueRow, ProposalRow, Store, utc_now

logger = logging.getLogger(__name__)

STATE_FILE = "tick-state.json"
"""The tick's memory under the config dir: last run, windows run, proposals notified, alerts sent."""
DECIDED_BY_TICK = "tick"
"""``decided_by`` of the proposals the tick rejects as executed alternatives."""
SYNC_MAX_AGE = timedelta(hours=1)
"""A league whose stored state is older than this is synced before its decisions run."""
INPUTS_EVERY: Mapping[Sport, timedelta] = {"nfl": timedelta(hours=8), "nba": timedelta(hours=1)}
"""How often the availability inputs are refreshed: nflverse practice reports a few times a day, the NBA official
injury report (published through the afternoon of a game day) hourly."""
ALERT_REPEAT = timedelta(hours=12)
"""A standing problem (a missing session, a decision that keeps failing) is alerted again after this long."""
NEAR_DEADLINE = timedelta(minutes=30)
"""An approval with its deadline inside this runs at once from ``fm bot`` (:func:`execute_on_approval`)."""
RUNS_KEPT = timedelta(days=14)
"""How long the state file remembers a window it ran."""
_CRITICAL_ERRORS = (KeyboardInterrupt, SystemExit, MemoryError)
OPPONENT_DECISIONS = LINEUP_DECISIONS | {"streaming"}
"""Decisions that weigh this week's opponent (lineups, and category-league streamers' swing)."""


class SessionVerdict(StrEnum):
    """The tick's reading of the ESPN session: the cookie statuses plus the two cases ``status()`` cannot express."""

    OK = "ok"
    EXPIRING = "expiring"
    EXPIRED = "expired"
    MISSING = "missing"
    UNAVAILABLE = "unavailable"

    @property
    def usable(self) -> bool:
        """The session can back reads and writes (an expiring cookie still works)."""
        return self in (SessionVerdict.OK, SessionVerdict.EXPIRING)


@dataclass(frozen=True, slots=True)
class SessionCheck:
    verdict: SessionVerdict
    detail: str
    session: EspnSession | None = None

    @property
    def usable(self) -> bool:
        return self.verdict.usable and self.session is not None


def check_session(loader: Callable[[], EspnSession] = load_session, *, now: datetime) -> SessionCheck:
    """Load the saved session and judge it. ``NotLoggedInError`` is ``MISSING``, any other ``AuthError`` or a browser
    that will not start is ``UNAVAILABLE``; otherwise the cookie's own status."""
    try:
        session = loader()
    except NotLoggedInError as exc:
        return SessionCheck(SessionVerdict.MISSING, str(exc))
    except (AuthError, BrowserError) as exc:
        return SessionCheck(SessionVerdict.UNAVAILABLE, f"could not read the ESPN session: {exc}")
    status = session.status(now)
    detail = "ESPN session saved; the cookie carries no expiry date."
    if session.expires_at is not None:
        day = session.expires_at.date().isoformat()
        days_left = max((session.expires_at - now).days, 0)
        detail = {
            SessionStatus.OK: f"ESPN session OK until {day} ({days_left} days left).",
            SessionStatus.EXPIRING: f"ESPN session expires soon: {day} ({days_left} days left); run `fm login`.",
            SessionStatus.EXPIRED: f"ESPN session expired on {day}; run `fm login`.",
        }[status]
    return SessionCheck(SessionVerdict(status.value), detail, session)


# --- state ------------------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class TickState:
    """What earlier ticks did, kept in ``tick-state.json`` (:func:`state_path`)."""

    last_tick: datetime | None = None
    runs: dict[str, datetime] = field(default_factory=dict[str, datetime])
    """Window key -> when it ran (:attr:`fm.jobs.deadlines.RunWindow.key`)."""
    notified: set[int] = field(default_factory=set[int])
    """Proposal ids pushed to the phone."""
    alerts: dict[str, datetime] = field(default_factory=dict[str, datetime])
    """Alert key -> when it was last sent, for the ones that repeat."""
    inputs: dict[str, datetime] = field(default_factory=dict[str, datetime])
    """League key -> when its availability inputs were last refreshed."""

    @classmethod
    def load(cls, path: Path) -> TickState:
        """Read the file; a missing or unreadable file is an empty state (the tick then runs what is due now)."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError) as exc:
            logger.warning("tick: ignoring unreadable state file %s: %s", path, exc)
            return cls()
        try:
            return cls(
                last_tick=_stamp(data.get("last_tick")),
                runs=_stamps(data.get("runs", {})),
                notified={int(v) for v in data.get("notified", [])},
                alerts=_stamps(data.get("alerts", {})),
                inputs=_stamps(data.get("inputs", {})),
            )
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("tick: ignoring malformed state file %s: %s", path, exc)
            return cls()

    def save(self, path: Path) -> None:
        data = {
            "last_tick": None if self.last_tick is None else self.last_tick.isoformat(),
            "runs": {k: v.isoformat() for k, v in sorted(self.runs.items())},
            "notified": sorted(self.notified),
            "alerts": {k: v.isoformat() for k, v in sorted(self.alerts.items())},
            "inputs": {k: v.isoformat() for k, v in sorted(self.inputs.items())},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temp, path)

    def prune(self, now: datetime, *, keep: timedelta = RUNS_KEPT) -> None:
        cutoff = now - keep
        self.runs = {k: v for k, v in self.runs.items() if v >= cutoff}
        self.alerts = {k: v for k, v in self.alerts.items() if v >= cutoff}


def _stamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"naive timestamp {value!r}")
    return parsed.astimezone(UTC)


def _stamps(value: object) -> dict[str, datetime]:
    if not isinstance(value, Mapping):
        return {}
    found: dict[str, datetime] = {}
    for key, raw in value.items():
        parsed = _stamp(raw)
        if parsed is not None:
            found[str(key)] = parsed
    return found


def state_path() -> Path:
    """``$FM_CONFIG_DIR/tick-state.json``."""
    return paths.config_dir() / STATE_FILE


# --- collaborators ----------------------------------------------------------------------------------------------------


class ClientFactory(Protocol):
    """An :class:`fm.espn.client.EspnClient` for a configured league (``EspnClient.for_league`` live; a fake-backed
    client in tests). The tick closes it."""

    def __call__(self, league: League, session: EspnSession | None) -> EspnClient: ...


class Syncer(Protocol):
    """:func:`fm.jobs.sync.sync` narrowed to how the tick calls it."""

    def __call__(
        self, store: Store, config: Config, *, session: EspnSession | None, leagues: Sequence[str]
    ) -> SyncReport: ...


@dataclass(slots=True)
class LeagueContext:
    """One league as the tick works on it: the config, the stored row and settings, the sport's plugin and pro
    schedule, the current period, and the client (open for the league's reads while the tick works on it)."""

    store: Store
    config: Config
    league: League
    row: LeagueRow
    settings: LeagueSettings
    plugin: SportPlugin
    schedule: ScheduleLike
    period: int
    now: datetime
    client: EspnClient | None = None

    @property
    def key(self) -> str:
        return self.league.key

    @property
    def sport(self) -> Sport:
        return self.row.sport

    def team_url(self) -> str:
        return team_page_url(self.league.game, self.row.espn_league_id, self.row.team_id, self.row.season, self.period)


class InputsRefresher(Protocol):
    """Refreshes a league's availability inputs (and projections) for the current period; returns notes."""

    def __call__(self, ctx: LeagueContext) -> Sequence[str]: ...


def _default_clients(league: League, session: EspnSession | None) -> EspnClient:
    return EspnClient.for_league(league, session)


# --- inputs -----------------------------------------------------------------------------------------------------------


def tracked_players(store: Store, row: LeagueRow, period: int) -> tuple[int, ...]:
    """Every player the league's decisions can see this period: everyone rostered in the latest snapshot plus everyone
    with an ESPN projection line for the period (the pool the sync wrote cards for)."""
    ids = set(store.rosters.rostered_ids(row.row_id, period))
    ids.update(line.espn_id for line in store.projections.for_period(row.sport, row.season, period))
    return tuple(sorted(ids))


def refresh_nfl_inputs(ctx: LeagueContext, *, source: NflverseSource) -> tuple[str, ...]:
    """nflverse's injuries dataset -> this week's practice reports -> the stored availability rows
    (:func:`fm.model.availability.assess_stored`, which keeps the week's practice log in each period's row)."""
    fetched = source.injuries(ctx.row.season)
    reports = practice_from_nflverse(
        fetched.data, Crosswalk.from_store(ctx.store), season=ctx.row.season, week=ctx.period, observed=fetched.as_of
    )
    rows = assess_stored(
        ctx.store,
        "nfl",
        tracked_players(ctx.store, ctx.row, ctx.period),
        season=ctx.row.season,
        scoring_period=ctx.period,
        as_of=ctx.now,
        schedule=ctx.schedule,
        practice=reports.by_player,
    )
    count = sum(len(found) for found in reports.by_player.values())
    state = "stale" if fetched.stale else "degraded" if fetched.degraded else "cached" if fetched.cached else "fresh"
    note = (
        f"{ctx.key}: availability assessed for {len(rows)} players with {count} practice reports "
        f"(nflverse injuries as of {fetched.as_of.astimezone(UTC):%Y-%m-%d %H:%M} UTC, {state})"
    )
    return (note, *fetched.warnings, *reports.warnings)


def refresh_nba_inputs(
    ctx: LeagueContext, *, source: NbaInjuriesSource, weights: BlendWeights | None = None
) -> tuple[str, ...]:
    """The official injury report for the fantasy day -> the stored availability rows, then the day's projection
    blend (:func:`fm.model.value_nba.blend_day`; importing the module attaches the DARKO loader)."""
    from fm.model import value_nba  # the DARKO loader registers on import; kept lazy so NFL-only ticks never pay

    day = fantasy_day(ctx.now)
    fetched = source.official_report(day)
    rows = assess_stored(
        ctx.store,
        "nba",
        tracked_players(ctx.store, ctx.row, ctx.period),
        season=ctx.row.season,
        scoring_period=ctx.period,
        as_of=ctx.now,
        schedule=ctx.schedule,
        official=fetched.data,
    )
    notes = [
        f"{ctx.key}: availability assessed for {len(rows)} players against the official injury report for {day} "
        f"({len(fetched.data.entries)} entries{', degraded' if fetched.degraded else ''})",
        *fetched.warnings,
    ]
    blend = value_nba.blend_day(
        ctx.store, ctx.row.season, ctx.period, weights=weights if weights is not None else BlendWeights.load()
    )
    notes.append(f"{ctx.key}: day {ctx.period} projections blended for {len(blend.rows)} players ({blend.saved} saved)")
    notes.extend(blend.warnings)
    return tuple(notes)


def refresh_inputs(ctx: LeagueContext) -> tuple[str, ...]:
    """The production refresh: fresh adapters over the cache dir, dispatched on the league's sport."""
    if ctx.sport == "nfl":
        return refresh_nfl_inputs(ctx, source=NflverseSource())
    with NbaInjuriesSource() as source:
        return refresh_nba_inputs(ctx, source=source)


# --- results ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionRun:
    """One registered decision run in one window."""

    name: str
    window: str
    proposals: tuple[int, ...] = ()
    blocked: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None

    def describe(self) -> str:
        if self.error is not None:
            return f"{self.name} ({self.window}): FAILED: {self.error}"
        ids = ", ".join(f"#{pid}" for pid in self.proposals) or "no proposals"
        blocked = f", {len(self.blocked)} blocked by policy" if self.blocked else ""
        return f"{self.name} ({self.window}): {ids}{blocked}"


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """What executing one approved proposal came to."""

    proposal_id: int
    league_key: str
    kind: str
    summary: str
    ok: bool
    detail: str
    result: ExecutionResult | None = None
    rejected: tuple[int, ...] = ()
    """Alternatives rejected because this one executed."""

    def describe(self) -> str:
        head = f"#{self.proposal_id} {self.kind} in {self.league_key} ({self.summary}): "
        head += "verified" if self.ok else f"FAILED: {self.detail}"
        if self.rejected:
            head += "; rejected alternative " + ", ".join(f"#{pid}" for pid in self.rejected)
        return head


@dataclass(frozen=True, slots=True)
class LeagueTick:
    """What the tick did for one league."""

    key: str
    sport: Sport
    period: int | None
    synced: bool = False
    deadlines: tuple[Deadline, ...] = ()
    due: tuple[RunWindow, ...] = ()
    missed: tuple[RunWindow, ...] = ()
    decisions: tuple[DecisionRun, ...] = ()
    notified: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    skipped: str | None = None
    """Why the league's decisions did not run (not synced, season over, no schedule)."""


@dataclass(frozen=True, slots=True)
class TickReport:
    at: datetime
    session: SessionCheck
    paused: str | None
    reconciled: tuple[int, ...]
    expired: tuple[int, ...]
    leagues: tuple[LeagueTick, ...]
    auto_approved: tuple[int, ...]
    executions: tuple[ExecutionOutcome, ...]
    would_execute: tuple[int, ...]
    """Approved proposals the tick would have executed (``TickOptions.execute`` off, paused, or no session)."""
    alerts: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """No decision failed and every execution verified."""
        return all(run.error is None for lg in self.leagues for run in lg.decisions) and all(
            outcome.ok for outcome in self.executions
        )

    def lines(self) -> list[str]:
        """The tick as log lines."""
        lines = [f"tick at {self.at.astimezone(UTC):%Y-%m-%d %H:%M} UTC", f"session: {self.session.detail}"]
        if self.paused:
            lines.append(f"PAUSED: {self.paused}; nothing executes until fm resume")
        if self.reconciled:
            lines.append("reconciled stale executions: " + ", ".join(f"#{pid}" for pid in self.reconciled))
        if self.expired:
            lines.append("expired: " + ", ".join(f"#{pid}" for pid in self.expired))
        for league in self.leagues:
            lines.extend(_league_lines(league))
        if self.auto_approved:
            lines.append("auto-approved at T-15: " + ", ".join(f"#{pid}" for pid in self.auto_approved))
        lines.extend(f"executed {outcome.describe()}" for outcome in self.executions)
        if self.would_execute:
            lines.append("not executed: " + ", ".join(f"#{pid}" for pid in self.would_execute))
        lines.extend(f"alert: {alert}" for alert in self.alerts)
        lines.extend(f"warning: {warning}" for warning in self.warnings)
        return lines


def _league_lines(league: LeagueTick) -> list[str]:
    head = f"{league.key} ({league.sport})"
    if league.skipped is not None:
        return [f"{head}: skipped: {league.skipped}"]
    lines = [f"{head}: period {league.period}" + (", synced" if league.synced else "")]
    lines.extend(f"  deadline {deadline.describe()}" for deadline in league.deadlines[:12])
    if len(league.deadlines) > 12:
        lines.append(f"  ... {len(league.deadlines) - 12} more deadlines")
    lines.extend(f"  missed {window.describe()}" for window in league.missed)
    lines.extend(f"  due {window.describe()}" for window in league.due)
    lines.extend(f"  decision {run.describe()}" for run in league.decisions)
    if league.notified:
        lines.append("  notified " + ", ".join(f"#{pid}" for pid in league.notified))
    lines.extend(f"  {note}" for note in league.notes)
    lines.extend(f"  warning: {warning}" for warning in league.warnings)
    return lines


@dataclass(frozen=True, slots=True)
class TickOptions:
    execute: bool = True
    """Execute approved proposals; off, the tick reports what it would have executed and sends nothing."""
    sync: bool = True
    """Sync a league whose stored state is older than ``sync_max_age``."""
    sync_max_age: timedelta = SYNC_MAX_AGE
    horizon: timedelta = DEFAULT_HORIZON
    executor: ExecutorOptions | None = None
    leagues: tuple[str, ...] | None = None
    """Only these league keys; every configured league when ``None``."""


# --- the tick ---------------------------------------------------------------------------------------------------------


def tick(
    store: Store,
    config: Config,
    *,
    now: datetime | None = None,
    clock: Callable[[], datetime] = utc_now,
    session_loader: Callable[[], EspnSession] = load_session,
    clients: ClientFactory = _default_clients,
    syncer: Syncer = sync,
    refresh: InputsRefresher | None = refresh_inputs,
    opener: RuntimeOpener | None = None,
    channel: NotifyChannel | None = None,
    registry: DecisionRegistry | None = None,
    flows: FlowRegistry | None = None,
    options: TickOptions | None = None,
    state_file: Path | None = None,
) -> TickReport:
    """Run one tick (the module docs give the steps). Every collaborator defaults to the live one: the browser
    profile's session, ``EspnClient.for_league``, :func:`fm.jobs.sync.sync`, :func:`refresh_inputs`,
    :func:`fm.executor.live_opener`, the process-wide decision and flow registries. ``channel`` ``None`` sends nothing
    to the phone. ``now`` pins the clock (tests); otherwise ``clock`` is read at the start and for each execution."""
    at = as_utc(now) if now is not None else clock()
    run = _Tick(
        store,
        config,
        now=at,
        clock=clock if now is None else (lambda: at),
        session_loader=session_loader,
        clients=clients,
        syncer=syncer,
        refresh=refresh,
        opener=opener,
        channel=channel,
        registry=registry if registry is not None else decide_registry.registry,
        flows=flows,
        options=options if options is not None else TickOptions(),
        state_file=state_file if state_file is not None else state_path(),
    )
    return run.run()


class _Tick:
    def __init__(
        self,
        store: Store,
        config: Config,
        *,
        now: datetime,
        clock: Callable[[], datetime],
        session_loader: Callable[[], EspnSession],
        clients: ClientFactory,
        syncer: Syncer,
        refresh: InputsRefresher | None,
        opener: RuntimeOpener | None,
        channel: NotifyChannel | None,
        registry: DecisionRegistry,
        flows: FlowRegistry | None,
        options: TickOptions,
        state_file: Path,
    ) -> None:
        self.store = store
        self.config = config
        self.now = now
        self.clock = clock
        self.session_loader = session_loader
        self.clients = clients
        self.syncer = syncer
        self.refresh = refresh
        self.opener = opener
        self.channel = channel
        self.registry = registry
        self.flows = flows
        self.options = options
        self.state_file = state_file
        self.state = TickState.load(state_file)
        self.alerts: list[str] = []
        self.warnings: list[str] = []
        self.schedules: dict[Sport, ScheduleLike] = {}
        self.opponents: dict[str, int | None] = {}
        self.session = SessionCheck(SessionVerdict.MISSING, "not checked")

    # --- steps

    def run(self) -> TickReport:
        reconciled = reconcile_executions(self.store, now=self.now)
        for row in reconciled:
            self._alert(
                f"Execution of #{row.row_id} was cut short",
                f"{row.kind} {_summary(row)}: the run never recorded an outcome; check the league before acting again",
                league_id=row.league_id,
            )
        expired = expire_due(self.store, now=self.now)
        for row in expired:
            if row.execution_token is not None or row.policy == "auto":
                when = f" at {row.deadline.astimezone(UTC):%H:%M} UTC" if row.deadline is not None else ""
                self._alert(
                    f"Missed: #{row.row_id} expired unexecuted",
                    f"{row.kind} {_summary(row)} reached its deadline{when} without executing",
                    league_id=row.league_id,
                )
        self.session = check_session(self.session_loader, now=self.now)
        if self.session.verdict is not SessionVerdict.OK:
            self._alert(
                f"ESPN session {self.session.verdict.value}", self.session.detail, key=f"session:{self.session.verdict}"
            )
        paused = pause_state()

        leagues: list[LeagueTick] = []
        selected = [
            league
            for league in self.config.leagues
            if self.options.leagues is None or league.key in self.options.leagues
        ]
        for league in selected:
            try:
                leagues.append(self._league(league))
            except _CRITICAL_ERRORS:
                raise
            except Exception as exc:  # one league's failure must not stop the others
                logger.exception("tick: league %s failed", league.key)
                self.warnings.append(f"{league.key}: {exc}")
                leagues.append(LeagueTick(key=league.key, sport=league.sport, period=None, skipped=f"error: {exc}"))

        auto_approved = auto_approve_due(self.store, now=self.now)
        executions, would_execute = self._execute(paused=paused is not None)

        self.state.notified &= {row.row_id for row in self.store.proposals.find(statuses=("proposed",))}
        self.state.last_tick = self.now
        self.state.prune(self.now)
        self.state.save(self.state_file)
        return TickReport(
            at=self.now,
            session=self.session,
            paused=paused.describe() if paused is not None else None,
            reconciled=tuple(row.row_id for row in reconciled),
            expired=tuple(row.row_id for row in expired),
            leagues=tuple(leagues),
            auto_approved=tuple(row.row_id for row in auto_approved),
            executions=executions,
            would_execute=would_execute,
            alerts=tuple(self.alerts),
            warnings=tuple(self.warnings),
        )

    # --- one league

    def _league(self, league: League) -> LeagueTick:
        warnings: list[str] = []
        notes: list[str] = []
        key = league.key
        row = self.store.leagues.by_key(key)
        synced = False
        if self.options.sync and self.session.usable:
            stale = row is None or row.as_of <= self.now - self.options.sync_max_age
            if stale:
                synced = self._sync(league, warnings)
                row = self.store.leagues.by_key(key)
        if row is None:
            return LeagueTick(key=key, sport=league.sport, period=None, skipped="not synced yet (fm sync)")
        settings = stored_settings(self.store, row)
        if settings is None:
            return LeagueTick(key=key, sport=row.sport, period=None, skipped="no synced settings (fm sync)")
        plugin = plugin_for(row.sport)
        session = self.session.session
        with self.clients(league, session) as client:
            schedule = self._schedule(row.sport, client, warnings)
            if schedule is None:
                return LeagueTick(
                    key=key,
                    sport=row.sport,
                    period=None,
                    synced=synced,
                    warnings=tuple(warnings),
                    skipped="no pro schedule",
                )
            period = plugin.scoring_period_at(self.now, schedule)
            final = settings.final_scoring_period
            if period is None or (final is not None and period > final):
                return LeagueTick(
                    key=key,
                    sport=row.sport,
                    period=period,
                    synced=synced,
                    warnings=tuple(warnings),
                    skipped="the season is over on the pro schedule",
                )
            ctx = LeagueContext(
                store=self.store,
                config=self.config,
                league=league,
                row=row,
                settings=settings,
                plugin=plugin,
                schedule=schedule,
                period=period,
                now=self.now,
                client=client if self.session.usable else None,
            )
            deadlines, found_warnings = upcoming(
                key,
                row.sport,
                settings,
                schedule,
                now=self.now,
                since=self.state.last_tick,
                horizon=self.options.horizon,
            )
            warnings.extend(found_warnings)
            windows = run_windows(key, row.sport, deadlines, schedule=schedule, now=self.now)
            missed = missed_windows(windows, now=self.now, last_tick=self.state.last_tick, ran=self.state.runs)
            for window in missed:
                self._alert(
                    f"Missed window: {key} {window.kind.value}",
                    f"{window.describe()} closed at {window.closes_at.astimezone(UTC):%H:%M} UTC with no tick "
                    "running (PC asleep?); check the lineup",
                    link=ctx.team_url(),
                )
            due = due_windows(windows, now=self.now, ran=self.state.runs)
            notes.extend(self._refresh_inputs(ctx, warnings))
            decisions = [run for window in due for run in self._run_window(ctx, window, warnings)]
            notified = self._notify_new(ctx, warnings)
        return LeagueTick(
            key=key,
            sport=row.sport,
            period=period,
            synced=synced,
            deadlines=deadlines,
            due=due,
            missed=missed,
            decisions=tuple(decisions),
            notified=notified,
            notes=tuple(notes),
            warnings=tuple(warnings),
        )

    def _sync(self, league: League, warnings: list[str]) -> bool:
        try:
            report = self.syncer(self.store, self.config, session=self.session.session, leagues=[league.key])
        except (SyncError, EspnClientError, SourceError, CrosswalkError, AuthError, OSError) as exc:
            warnings.append(f"{league.key}: sync failed, deciding on the stored state: {exc}")
            return False
        warnings.extend(f"{league.key}: sync: {warning}" for warning in report.warnings)
        for failed in report.failed:
            if failed.gate is not None and failed.gate.error is not None:
                warnings.append(f"{league.key}: sync gate: {failed.gate.error}")
        return True

    def _schedule(self, sport: Sport, client: EspnClient, warnings: list[str]) -> ScheduleLike | None:
        cached = self.schedules.get(sport)
        if cached is not None:
            return cached
        try:
            schedule = client.pro_schedule().data
        except EspnClientError as exc:
            warnings.append(f"could not read the {sport} pro schedule: {exc}")
            return None
        self.schedules[sport] = schedule
        return schedule

    def _refresh_inputs(self, ctx: LeagueContext, warnings: list[str]) -> list[str]:
        if self.refresh is None:
            return []
        last = self.state.inputs.get(ctx.key)
        if last is not None and self.now - last < INPUTS_EVERY[ctx.sport]:
            return []
        try:
            notes = list(self.refresh(ctx))
        except _CRITICAL_ERRORS:
            raise
        except Exception as exc:  # inputs are an improvement, never a blocker: the decision has ESPN's status
            logger.exception("tick: refreshing %s inputs failed", ctx.key)
            warnings.append(f"{ctx.key}: availability inputs not refreshed: {exc}")
            return []
        self.state.inputs[ctx.key] = self.now
        return notes

    def _run_window(self, ctx: LeagueContext, window: RunWindow, warnings: list[str]) -> list[DecisionRun]:
        runs: list[DecisionRun] = []
        failed = False
        for registration in self.registry.registered(ctx.sport):
            if not window.runs(registration.kind):
                continue
            run = self._run_decision(ctx, registration, window)
            runs.append(run)
            warnings.extend(f"{registration.name}: {warning}" for warning in run.warnings)
            if run.error is not None:
                failed = True
                self._alert(
                    f"Decision {registration.name} failed",
                    f"{window.describe()}: {run.error}",
                    key=f"decision:{registration.name}",
                    link=ctx.team_url(),
                )
        if not failed:  # a failed decision is retried on the next tick, until the window closes
            self.state.runs[window.key] = self.now
        return runs

    def _run_decision(self, ctx: LeagueContext, registration: Registration, window: RunWindow) -> DecisionRun:
        kwargs: dict[str, Any] = {"schedule": ctx.schedule, "now": self.now}
        if registration.kind in OPPONENT_DECISIONS:
            kwargs["opponent_team_id"] = self._opponent(ctx)
        if ctx.client is not None:
            kwargs["client"] = ctx.client
        try:
            result = registration.fn(ctx.store, ctx.config, ctx.row, **_accepted(registration.fn, kwargs))
        except _CRITICAL_ERRORS:
            raise
        except Exception as exc:
            logger.exception("tick: decision %s failed", registration.name)
            return DecisionRun(registration.name, window.key, error=f"{type(exc).__name__}: {exc}")
        proposals = tuple(row.row_id for row in _rows(getattr(result, "proposals", ())))
        blocked = tuple(_strings(getattr(result, "blocked", ())))
        warnings = tuple(_strings(getattr(result, "warnings", ())))
        return DecisionRun(registration.name, window.key, proposals=proposals, blocked=blocked, warnings=warnings)

    def _opponent(self, ctx: LeagueContext) -> int | None:
        """Our opponent this matchup period, from ``mMatchup``; ``None`` without a session, on a bye, or when the read
        fails or the matchup week cannot be placed on the calendar (the lineup then plans for expected points). Read
        once per league per tick: the lineup and streaming decisions share it."""
        if ctx.key in self.opponents:
            return self.opponents[ctx.key]
        opponent = self._read_opponent(ctx)
        if ctx.client is not None:  # without a session nothing was read, and a later read might work
            self.opponents[ctx.key] = opponent
        return opponent

    def _read_opponent(self, ctx: LeagueContext) -> int | None:
        if ctx.client is None:
            return None
        try:
            matchups = ctx.client.matchups().data
        except EspnClientError as exc:
            self.warnings.append(f"{ctx.key}: no opponent (matchups unreadable: {exc}); planning for expected points")
            return None
        try:
            matchup_period = matchup_period_of(ctx.settings, ctx.period)  # the calendar resolves NBA weekly matchups
        except CalendarError as exc:
            self.warnings.append(f"{ctx.key}: no opponent (matchup week unknown: {exc}); planning for expected points")
            return None
        return opponent_from(matchups, ctx.row.team_id, matchup_period)

    def _notify_new(self, ctx: LeagueContext, warnings: list[str]) -> tuple[int, ...]:
        if self.channel is None:
            return ()
        sent: list[int] = []
        for row in self.store.proposals.find(league_id=ctx.row.row_id, statuses=("proposed",)):
            if row.row_id in self.state.notified:
                continue
            try:
                notify_proposal(self.channel, self.store, row.row_id, now=self.now)
            except NotifyError as exc:
                warnings.append(f"{ctx.key}: proposal #{row.row_id} not pushed: {exc}")
                continue
            self.state.notified.add(row.row_id)
            sent.append(row.row_id)
        return tuple(sent)

    # --- execution

    def _execute(self, *, paused: bool) -> tuple[tuple[ExecutionOutcome, ...], tuple[int, ...]]:
        approved = sorted(
            (
                row
                for row in self.store.proposals.find(statuses=("approved",))
                if row.execution_token is not None and (row.deadline is None or row.deadline > self.now)
            ),
            key=lambda row: (row.deadline is None, row.deadline or self.now, row.row_id),
        )
        if not approved:
            return (), ()
        if paused or not self.options.execute or not self.session.usable:
            return (), tuple(row.row_id for row in approved)
        opener = self.opener if self.opener is not None else live_opener()
        outcomes: list[ExecutionOutcome] = []
        done: set[int] = set()
        for row in approved:
            if row.row_id in done:
                continue
            current = self.store.proposals.get(row.row_id)
            if current is None or current.status != "approved":
                continue  # rejected as an alternative, or decided elsewhere, since the list was read
            outcome = execute_proposal(
                self.store, current, opener=opener, flows=self.flows, options=self.options.executor, clock=self.clock
            )
            done.update(outcome.rejected)
            outcomes.append(outcome)
            self._report_execution(current, outcome)
        return tuple(outcomes), ()

    def _report_execution(self, row: ProposalRow, outcome: ExecutionOutcome) -> None:
        if self.channel is None:
            return
        league = self.store.leagues.get(row.league_id)
        link = None
        if league is not None:
            link = team_page_url(
                league.sport, league.espn_league_id, league.team_id, league.season, row.scoring_period_id
            )
        if outcome.ok:
            message = Message(
                title=f"#{row.row_id} executed: {outcome.kind}",
                body=f"{outcome.summary} in {outcome.league_key}; verified on ESPN",
                tags=("white_check_mark",),
                link=link,
            )
            try:
                self.channel.send(message)
            except NotifyError as exc:
                self.warnings.append(f"execution of #{row.row_id} not reported: {exc}")
        else:
            self._alert(
                f"#{row.row_id} failed: {outcome.kind}",
                f"{outcome.summary} in {outcome.league_key}: {outcome.detail}",
                link=link,
            )

    # --- alerts

    def _alert(
        self, title: str, body: str, *, key: str | None = None, link: str | None = None, league_id: int | None = None
    ) -> None:
        """Push an alert (once per :data:`ALERT_REPEAT` when ``key`` names a standing problem) and record it."""
        if key is not None:
            last = self.state.alerts.get(key)
            if last is not None and self.now - last < ALERT_REPEAT:
                return
            self.state.alerts[key] = self.now
        if link is None and league_id is not None:
            league = self.store.leagues.get(league_id)
            if league is not None:
                link = team_page_url(league.sport, league.espn_league_id, league.team_id, league.season)
        self.alerts.append(f"{title}: {body}")
        if self.channel is None:
            return
        try:
            send_alert(self.channel, title, body, link=link)
        except NotifyError as exc:
            self.warnings.append(f"alert not pushed ({title}): {exc}")


# --- execution helpers (shared with the bot hook) ---------------------------------------------------------------------


def execute_proposal(
    store: Store,
    row: ProposalRow,
    *,
    opener: RuntimeOpener,
    flows: FlowRegistry | None = None,
    options: ExecutorOptions | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> ExecutionOutcome:
    """Execute one approved proposal with its own token and, when it verifies, reject its alternatives. A refusal
    (``ProposalError``, :class:`fm.executor.ExecutorError`, ``AuthError``, ``BrowserError``) is a failed outcome
    with the proposal unchanged, never an exception."""
    league = store.leagues.get(row.league_id)
    key = league.key if league is not None else f"league {row.league_id}"
    summary = _summary(row)
    try:
        result = execute(
            store, row.row_id, token=row.execution_token, opener=opener, registry=flows, options=options, clock=clock
        )
    except (ProposalError, ExecutorError, AuthError, BrowserError) as exc:
        return ExecutionOutcome(row.row_id, key, row.kind, summary, ok=False, detail=f"refused: {exc}")
    detail = _result_detail(result)
    rejected = reject_alternatives(store, result.proposal, now=clock()) if result.ok else ()
    return ExecutionOutcome(
        row.row_id,
        key,
        row.kind,
        summary,
        ok=result.ok,
        detail=detail,
        result=result,
        rejected=tuple(other.row_id for other in rejected),
    )


def reject_alternatives(store: Store, executed: ProposalRow, *, now: datetime | None = None) -> tuple[ProposalRow, ...]:
    """Reject the league's open proposals drafted against the same roster read as ``executed`` (the same
    ``engine_numbers["basis"]``): alternatives to a move that has landed, which would otherwise be auto-approved at
    T-15 and fail their preconditions."""
    basis = executed.engine_numbers.get("basis")
    if not basis:
        return ()
    rejected: list[ProposalRow] = []
    for other in store.proposals.open(executed.league_id):
        if other.row_id == executed.row_id or other.status not in ("proposed", "approved"):
            continue
        if other.engine_numbers.get("basis") != basis:
            continue
        try:
            rejected.append(reject(store, other.row_id, decided_by=DECIDED_BY_TICK, now=now))
        except ProposalError as exc:  # decided or expired in the meantime
            logger.info("tick: alternative #%d not rejected: %s", other.row_id, exc)
    return tuple(rejected)


def execute_on_approval(
    store: Store,
    *,
    opener: RuntimeOpener,
    flows: FlowRegistry | None = None,
    options: ExecutorOptions | None = None,
    near: timedelta = NEAR_DEADLINE,
    clock: Callable[[], datetime] = utc_now,
    on_outcome: Callable[[ExecutionOutcome], object] | None = None,
) -> Callable[[DecisionResult], None]:
    """An ``on_result`` hook for :func:`fm.notify.listen`: an approval whose deadline is within ``near`` executes at
    once instead of waiting for the next tick (DESIGN section 13). Everything else, and anything while paused, is
    left to the tick. ``on_outcome`` receives what happened."""

    def on_result(result: DecisionResult) -> None:
        if not result.ok or result.decision != "approve":
            return
        row = store.proposals.get(result.proposal_id)
        if row is None or row.status != "approved" or row.execution_token is None or row.deadline is None:
            return
        now = clock()
        if row.deadline <= now or row.deadline - now > near or pause_state() is not None:
            return
        outcome = execute_proposal(store, row, opener=opener, flows=flows, options=options, clock=clock)
        if on_outcome is not None:
            on_outcome(outcome)

    return on_result


# --- small helpers ----------------------------------------------------------------------------------------------------


def _accepted(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The keywords of ``kwargs`` that ``fn`` accepts (all of them when it takes ``**kwargs``)."""
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(kwargs)
    return {name: value for name, value in kwargs.items() if name in parameters}


def _rows(value: object) -> list[ProposalRow]:
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return [item for item in value if isinstance(item, ProposalRow)]
    return []


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Iterable):
        return []
    found: list[str] = []
    for item in value:
        if isinstance(item, str):
            found.append(item)
        elif isinstance(item, tuple) and item and isinstance(item[-1], str):
            found.append(str(item[-1]))  # fm.decide.waivers reports (move, reason) pairs
        else:
            found.append(str(item))
    return found


def _summary(row: ProposalRow) -> str:
    try:
        return parse_payload(row).summary()
    except (ProposalError, ValueError):
        return row.kind


def _result_detail(result: ExecutionResult) -> str:
    if result.ok:
        return "verified"
    if not result.preconditions.ok:
        return "preconditions failed: " + "; ".join(result.preconditions.failures)
    last = result.last
    if last is None:
        return "nothing was attempted"
    detail = f"{last.mode} attempt {last.status}"
    if last.error:
        detail += f": {last.error}"
    verification = last.verification or {}
    if verification.get("detail"):
        detail += f" (re-read: {verification['detail']})"
    return detail


__all__ = [
    "ALERT_REPEAT",
    "DECIDED_BY_TICK",
    "INPUTS_EVERY",
    "NEAR_DEADLINE",
    "RUNS_KEPT",
    "STATE_FILE",
    "SYNC_MAX_AGE",
    "ClientFactory",
    "DecisionRun",
    "ExecutionOutcome",
    "InputsRefresher",
    "LeagueContext",
    "LeagueTick",
    "SessionCheck",
    "SessionVerdict",
    "Syncer",
    "TickOptions",
    "TickReport",
    "TickState",
    "check_session",
    "execute_on_approval",
    "execute_proposal",
    "refresh_inputs",
    "refresh_nba_inputs",
    "refresh_nfl_inputs",
    "reject_alternatives",
    "state_path",
    "tick",
    "tracked_players",
]
