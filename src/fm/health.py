"""Health checks: what is broken and how to fix it (docs/PLUG_AND_PLAY.md section 3, ROADMAP #50).

:func:`run_checks` answers a fixed list of questions about the installation and returns plain data, one
:class:`Check` per question, so ``fm doctor``, the dashboard (#54) and the MCP ``health`` tool (#55) show one set of
verdicts. By default only local state is read. ``online=True`` adds an ``mStatus`` read per league and a ping of the
phone channel; those are the only network calls, and they go through injectable collaborators so the unit tests run
on fakes.

Offline checks, in order:

- ``config``: ``config.toml`` and ``.env`` load (``ConfigError`` is a FAIL). A missing ``ANTHROPIC_API_KEY`` is a WARN
  because every read-only command works without it and the advisor falls back to template rationales.
- ``session``: the browser profile exists and the ESPN login cookie's expiry gives ok (OK), expiring (WARN) or expired
  (FAIL). Reading the cookie means launching a headless browser on the profile, and a launch that coincides with a
  tick's makes the tick's own session check fail, so the verdict comes first from the record every tick leaves in
  ``tick-state.json`` (``session: {status, expires_at, checked_at, detail}``, :func:`fm.jobs.tick.session_record`).
  Only a caller that passes ``launch_browser=True`` (``fm doctor``) may launch, and only when the run lock is free
  (it is held for the launch and released right after, so no tick starts beside it) and the record is older than
  :data:`SESSION_RECORD_FRESH` (or the online ESPN reads need the cookies themselves). The dashboard and the MCP tool
  never launch: without a fresh record they report WARN "not checked recently" and point at ``fm doctor``. The loader
  is called only once the profile exists, so a fresh install never starts a browser; a profile that will not open
  (held by another ``fm`` process) is a WARN.
- ``browser``: Microsoft Edge or Google Chrome is installed, found where Playwright's registry looks for the channel
  executables (:func:`find_browsers`) without launching anything.
- ``notify``: the ``[notify]`` channel's secrets are in ``.env``; ``channel = "none"`` (no phone on purpose) is a WARN.
- ``tick``: the last tick in ``tick-state.json``: WARN after :data:`TICK_WARN_AFTER`, FAIL after
  :data:`TICK_FAIL_AFTER` or when no tick ever ran. The fix is ``fm run`` when nothing holds the run lock, and a
  pointer at ``logs/run.log`` when ``fm run`` holds it but its ticks are late.
- ``sync:<league>``: the store's ``as_of`` per configured league: WARN after :data:`SYNC_WARN_AFTER` (the tick syncs
  a league hourly, so an older sync means the ticks are not syncing), FAIL after :data:`SYNC_FAIL_AFTER` or before
  the first sync.
- ``pause``: the kill switch (a WARN while thrown).
- ``proposals``: what is open, by status. ``deadlines``: open proposals due within :data:`DEADLINE_SOON` or overdue.
- ``llm``: today's recorded Claude spend against ``[llm] daily_budget_usd``; the cap reached is a WARN.

The store checks need ``state.db``; on a fresh install it does not exist yet, and opening it would create and migrate
it, so they report WARN "no state yet" instead. ``--online`` adds ``espn:<league>`` (an ``mStatus`` read through
:class:`fm.espn.client.EspnClient`) and ``channel`` (Telegram ``getMe``, or the ntfy server's ``/v1/health``).

Verdicts: FAIL means something cannot do its job and ``fix`` names the command that repairs it; WARN means it works but
needs attention; SKIP means the check could not run because an earlier one failed (no config, no session). This
module reads only: nothing is created or changed, except that a session launch with ``launch_browser=True`` takes the
run lock for its duration (writing the holder record into ``run.lock``). It never touches ESPN's write path, imports
neither ``fm.executor`` nor the browser flows, and stays clear of the tick's and the advisor's import graphs so the
dashboard can load it cheaply.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import httpx

from fm import paths, render
from fm.browser.session import CHANNELS, BrowserError, profile_exists
from fm.config import Config, ConfigError, League, load_config
from fm.espn.auth import AuthError, EspnSession, NotLoggedInError, SessionStatus, describe, load_session
from fm.espn.client import EspnAuthError, EspnClient, EspnClientError
from fm.espn.models import LeagueStatus
from fm.jobs.runlock import RunLock, RunLockHeld
from fm.jobs.scheduler_base import LOGS_DIRNAME, RUN_LOG_FILENAME
from fm.notify import NotifyError, NtfyChannel, TelegramChannel, open_channel
from fm.notify.ntfy import DEFAULT_SERVER, REQUEST_TIMEOUT
from fm.proposals import PauseState, pause_state
from fm.store import OPEN_PROPOSAL_STATUSES, ProposalRow, SchemaError, Store

TICK_STATE_FILE = "tick-state.json"
"""``fm.jobs.tick.STATE_FILE``, read here directly: importing the tick would pull in the executor and the flows."""
SESSION_RECORD_FRESH = timedelta(minutes=30)
"""A tick's session record younger than this is the session verdict; an older one is "not checked recently"."""
TICK_LATE_FIX = f"fm run is running but ticks are late: check {LOGS_DIRNAME}/{RUN_LOG_FILENAME}"
"""The tick check's fix when ``fm run`` holds the run lock: starting another loop would not help."""
NO_STATE = "no state yet: state.db does not exist (run fm setup, or fm sync once config.toml is in place)"
TICK_WARN_AFTER = timedelta(minutes=30)
"""A tick older than this is overdue (the scheduler runs one every few minutes)."""
TICK_FAIL_AFTER = timedelta(hours=6)
"""A tick older than this means ticks have stopped: deadlines pass unexecuted."""
SYNC_WARN_AFTER = timedelta(hours=2)
"""The tick syncs a league whose state is an hour old (``fm.jobs.tick.SYNC_MAX_AGE``), so older means it is not."""
SYNC_FAIL_AFTER = timedelta(hours=24)
"""A day-old roster and standings mislead every recommendation."""
DEADLINE_SOON = timedelta(hours=2)
"""An open proposal due within this is listed by the ``deadlines`` check."""
ADVISOR_KEY = "ANTHROPIC_API_KEY"
NTFY_HEALTH_PATH = "/v1/health"
"""ntfy's health endpoint (``{"healthy": true}``), the channel ping for an ntfy setup."""

BROWSER_NAMES: Mapping[str, str] = {"msedge": "Microsoft Edge", "chrome": "Google Chrome"}
BROWSER_PATHS: Mapping[str, Mapping[str, tuple[tuple[str | None, str], ...]]] = {
    "win32": {
        "msedge": (
            ("PROGRAMFILES", "Microsoft/Edge/Application/msedge.exe"),
            ("PROGRAMFILES(X86)", "Microsoft/Edge/Application/msedge.exe"),
            ("LOCALAPPDATA", "Microsoft/Edge/Application/msedge.exe"),
        ),
        "chrome": (
            ("PROGRAMFILES", "Google/Chrome/Application/chrome.exe"),
            ("PROGRAMFILES(X86)", "Google/Chrome/Application/chrome.exe"),
            ("LOCALAPPDATA", "Google/Chrome/Application/chrome.exe"),
        ),
    },
    "darwin": {
        "msedge": ((None, "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),),
        "chrome": ((None, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),),
    },
    "linux": {
        "msedge": ((None, "/opt/microsoft/msedge/msedge"),),
        "chrome": ((None, "/opt/google/chrome/chrome"),),
    },
}
"""Where each channel's executable lives per platform (an environment variable root plus a relative path, or an
absolute path): the locations Playwright's own registry resolves ``channel="msedge"`` / ``"chrome"`` to."""


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class Check:
    """One verdict. ``fix`` is the exact command to run when there is one (``fm login``, ``fm setup``)."""

    name: str
    status: CheckStatus
    detail: str
    fix: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """The JSON shape ``fm doctor --json`` and the dashboard serve."""
        return {"name": self.name, "status": self.status.value, "detail": self.detail, "fix": self.fix}


def has_failure(checks: Iterable[Check]) -> bool:
    """True when any check is a FAIL: ``fm doctor``'s exit code."""
    return any(check.status is CheckStatus.FAIL for check in checks)


class EspnClientFactory(Protocol):
    """How the online check gets an ESPN client for a league. Tests pass one over an ``httpx.MockTransport``."""

    def __call__(self, league: League, session: EspnSession) -> EspnClient: ...


def _default_client(league: League, session: EspnSession) -> EspnClient:
    return EspnClient.for_league(league, session, capture=False)


def find_browsers(
    *,
    platform: str = sys.platform,
    environ: Mapping[str, str] | None = None,
    is_file: Callable[[Path], bool] = Path.is_file,
) -> dict[str, Path]:
    """The installed channels and their executables, in :data:`fm.browser.session.CHANNELS` order (Edge first), looked
    up at the paths Playwright resolves the channels to (:data:`BROWSER_PATHS`). Nothing is launched."""
    env = os.environ if environ is None else environ
    family = "win32" if platform.startswith("win") else "darwin" if platform == "darwin" else "linux"
    found: dict[str, Path] = {}
    for channel in CHANNELS:
        for variable, tail in BROWSER_PATHS[family].get(channel, ()):
            root = env.get(variable) if variable is not None else None
            if variable is not None and not root:
                continue
            path = Path(root) / tail if root else Path(tail)
            if is_file(path):
                found[channel] = path
                break
    return found


def run_checks(
    *,
    online: bool = False,
    now: datetime | None = None,
    launch_browser: bool = False,
    config_loader: Callable[[], Config] = load_config,
    store_opener: Callable[[], Store] = Store.open,
    state_db: Path | None = None,
    session_loader: Callable[[], EspnSession] = load_session,
    profile_dir: Path | None = None,
    browsers: Callable[[], Mapping[str, Path]] = find_browsers,
    tick_state: Path | None = None,
    run_lock: RunLock | None = None,
    pause_reader: Callable[[], PauseState | None] = pause_state,
    client_factory: EspnClientFactory = _default_client,
    http_client: httpx.Client | None = None,
) -> list[Check]:
    """Every check, in the order the module docstring lists them. Pure data; nothing is written anywhere unless
    ``launch_browser`` lets the session check take the run lock.

    The session verdict comes from the last tick's record in ``tick_state`` (default ``tick-state.json``) when it is
    fresh. ``session_loader`` (which launches a browser on the profile) is called only when ``launch_browser`` is set,
    ``profile_dir`` (default ``fm.paths.browser_profile_dir()``) holds a profile, ``run_lock`` (default the real one,
    as ``fm doctor``) can be taken without waiting, and the record is stale or ``online`` needs the cookies. The store
    checks open ``state.db`` through ``store_opener`` only when ``state_db`` (default ``fm.paths.state_db()``) exists.
    ``online`` adds the ESPN reads (through ``client_factory``) and the channel ping (through ``http_client`` when
    given, else the channel's own client).
    """
    at = datetime.now(UTC) if now is None else now.astimezone(UTC)
    profile = profile_dir if profile_dir is not None else paths.browser_profile_dir()
    state_file = tick_state if tick_state is not None else paths.config_dir() / TICK_STATE_FILE
    database = state_db if state_db is not None else paths.state_db()
    lock = run_lock if run_lock is not None else RunLock(command="fm doctor")
    record = _tick_record(state_file)

    config_check, config = _config(config_loader)
    session_check, session = _session(
        profile, session_loader, at, record=record.session, launch=launch_browser, need_cookies=online, lock=lock
    )
    checks = [config_check, session_check, _browser(browsers()), _notify(config), _tick(record, state_file, at, lock)]
    if database.exists():
        checks.extend(_store_checks(store_opener, config, pause_reader(), at))
    else:
        checks.extend(_no_state(config, pause_reader()))
    if online:
        if config is None:
            checks.append(_skipped("espn"))
        else:
            checks.extend(_espn(league, session, client_factory) for league in config.leagues)
        checks.append(_channel(config, http_client))
    return checks


# --- offline checks ---------------------------------------------------------------------------------------------------


def _skipped(name: str, reason: str = "config.toml did not load") -> Check:
    return Check(name, CheckStatus.SKIP, reason)


def _config(loader: Callable[[], Config]) -> tuple[Check, Config | None]:
    try:
        config = loader()
    except ConfigError as exc:
        fix = "fm config check" if paths.config_file().is_file() else "fm setup"
        return Check("config", CheckStatus.FAIL, str(exc), fix=fix), None
    keys = ", ".join(league.key for league in config.leagues)
    detail = f"config.toml valid: {len(config.leagues)} league(s) ({keys})"
    if ADVISOR_KEY in config.missing_secrets():
        detail += f"; {ADVISOR_KEY} missing from .env, so the advisor writes template rationales only"
        return Check("config", CheckStatus.WARN, detail, fix="fm setup"), config
    return Check("config", CheckStatus.OK, detail), config


@dataclass(frozen=True, slots=True)
class _SessionRecord:
    """The session verdict a tick left in ``tick-state.json`` (:func:`fm.jobs.tick.session_record`)."""

    status: str
    checked_at: datetime
    expires_at: datetime | None
    detail: str


@dataclass(frozen=True, slots=True)
class _TickRecord:
    last_tick: datetime | None = None
    session: _SessionRecord | None = None


def _tick_record(path: Path) -> _TickRecord:
    """``last_tick`` and the session record from the tick's state file; either is ``None`` when the file is missing,
    unreadable, or the value is malformed or naive (the tick itself ignores such a file, so it has no memory either)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _TickRecord()
    if not isinstance(data, dict):
        return _TickRecord()
    return _TickRecord(last_tick=_aware(data.get("last_tick")), session=_session_record(data.get("session")))


def _session_record(value: object) -> _SessionRecord | None:
    if not isinstance(value, dict):
        return None
    status, detail = value.get("status"), value.get("detail")
    checked_at = _aware(value.get("checked_at"))
    if not isinstance(status, str) or checked_at is None:
        return None
    return _SessionRecord(
        status=status,
        checked_at=checked_at,
        expires_at=_aware(value.get("expires_at")),
        detail=detail if isinstance(detail, str) else "",
    )


def _aware(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return None if parsed.tzinfo is None else parsed.astimezone(UTC)


def _session(
    profile: Path,
    loader: Callable[[], EspnSession],
    now: datetime,
    *,
    record: _SessionRecord | None,
    launch: bool,
    need_cookies: bool,
    lock: RunLock,
) -> tuple[Check, EspnSession | None]:
    """The session verdict and, when it was loaded and ESPN reads can use it, the session itself."""
    name = "session"
    if not profile_exists(profile):
        detail = "no browser profile yet: nothing is signed in to ESPN"
        return Check(name, CheckStatus.FAIL, detail, fix="fm login"), None
    fresh = record is not None and now - record.checked_at < SESSION_RECORD_FRESH
    if launch and (need_cookies or not fresh):
        try:
            lock.acquire()
        except RunLockHeld as exc:
            held_by = exc.holder.describe() if exc.holder is not None else "another fm process"
            if record is not None and fresh:
                return _from_record(record, now), None
            return _not_checked(record, now, f"{held_by} holds the run lock, so no browser is started beside it"), None
        except OSError as exc:
            return _not_checked(record, now, f"the run lock could not be taken ({exc})"), None
        try:
            return _load_session(loader, now)
        finally:
            lock.release()
    if record is not None and fresh:
        return _from_record(record, now), None
    return _not_checked(record, now, "fm doctor checks it", fix="fm doctor" if not launch else None), None


def _load_session(loader: Callable[[], EspnSession], now: datetime) -> tuple[Check, EspnSession | None]:
    name = "session"
    try:
        session = loader()
    except NotLoggedInError as exc:
        return Check(name, CheckStatus.FAIL, str(exc), fix="fm login"), None
    except BrowserError as exc:
        detail = f"could not open the browser profile to read the session ({exc}); another fm process may hold it"
        return Check(name, CheckStatus.WARN, detail), None
    except AuthError as exc:
        return Check(name, CheckStatus.FAIL, f"could not read the ESPN session: {exc}", fix="fm login"), None
    detail = describe(session, now)
    match session.status(now):
        case SessionStatus.EXPIRED:
            return Check(name, CheckStatus.FAIL, detail, fix="fm login"), None
        case SessionStatus.EXPIRING:
            return Check(name, CheckStatus.WARN, detail, fix="fm login"), session
        case _:
            return Check(name, CheckStatus.OK, detail), session


def _from_record(record: _SessionRecord, now: datetime) -> Check:
    """The verdict a fresh tick record gives. A cookie verdict (ok, expiring, expired) is judged again against ``now``
    from the recorded expiry date, so the line reads exactly as a fresh load would."""
    name = "session"
    checked = f" (the tick checked it {_age(now - record.checked_at)} ago)"
    status = record.status
    detail = record.detail
    if status in tuple(SessionStatus) and record.expires_at is not None:
        cookie = EspnSession(espn_s2="", swid="", expires_at=record.expires_at)
        status, detail = cookie.status(now).value, describe(cookie, now)
    match status:
        case SessionStatus.OK:
            return Check(name, CheckStatus.OK, detail + checked)
        case SessionStatus.EXPIRING:
            return Check(name, CheckStatus.WARN, detail + checked, fix="fm login")
        case "unavailable":
            return Check(name, CheckStatus.WARN, detail + checked + "; another fm process may have held the profile")
        case _:  # expired, missing: nothing reads or writes until fm login
            return Check(name, CheckStatus.FAIL, detail + checked, fix="fm login")


def _not_checked(record: _SessionRecord | None, now: datetime, why: str, *, fix: str | None = None) -> Check:
    if record is None:
        last = "no tick has recorded it yet"
    else:
        last = f"the last tick checked it {_age(now - record.checked_at)} ago ({record.status})"
    detail = f"ESPN session not checked recently: {last}; {why}"
    return Check("session", CheckStatus.WARN, detail, fix=fix if fix is not None else "fm login --check")


def _browser(installed: Mapping[str, Path]) -> Check:
    if not installed:
        detail = "no supported browser found; install Microsoft Edge or Google Chrome (fm login drives it)"
        return Check("browser", CheckStatus.FAIL, detail)
    found = "; ".join(f"{BROWSER_NAMES.get(channel, channel)} at {path}" for channel, path in installed.items())
    return Check("browser", CheckStatus.OK, found)


def _notify(config: Config | None) -> Check:
    if config is None:
        return _skipped("notify")
    channel = config.notify.channel
    if channel == "none":
        detail = (
            'no phone channel (notify.channel = "none"): proposals wait for fm proposals or the dashboard, and '
            "alerts stay in the logs"
        )
        return Check("notify", CheckStatus.WARN, detail, fix="fm notify setup")
    missing = [name for name in config.missing_secrets() if name != ADVISOR_KEY]
    if missing:
        detail = f"{channel} is not set up: {', '.join(missing)} missing from .env"
        return Check("notify", CheckStatus.FAIL, detail, fix="fm notify setup")
    return Check("notify", CheckStatus.OK, f"{channel} configured")


def _tick(record: _TickRecord, path: Path, now: datetime, lock: RunLock) -> Check:
    last = record.last_tick
    if last is None:
        detail = f"no tick has run yet (no last tick in {path.name})"
        return Check("tick", CheckStatus.FAIL, detail, fix=_tick_fix(lock))
    age = now - last
    detail = f"last tick {_age(age)} ago ({render.stamp(last)})"
    if age >= TICK_FAIL_AFTER:
        return Check("tick", CheckStatus.FAIL, f"{detail}; ticks have stopped", fix=_tick_fix(lock))
    if age >= TICK_WARN_AFTER:
        return Check("tick", CheckStatus.WARN, f"{detail}; the next tick is overdue", fix=_tick_fix(lock))
    return Check("tick", CheckStatus.OK, detail)


def _tick_fix(lock: RunLock) -> str:
    """``fm run`` when nothing holds the run lock; when something does (``fm run``, or a tick that is stuck), starting
    another loop would only wait on it, so the fix points at the log instead. A lock file that does not exist is not
    probed (probing would create it)."""
    try:
        held = lock.path.exists() and lock.is_locked()
    except OSError:
        held = False
    return TICK_LATE_FIX if held else "fm run"


def _no_state(config: Config | None, paused: PauseState | None) -> list[Check]:
    """The store checks on an installation that has no ``state.db`` yet: WARN, not FAIL (nothing is broken, nothing
    has run), and not opened, since opening would create and migrate it."""
    fix = "fm sync" if config is not None else "fm setup"
    syncs = [f"sync:{league.key}" for league in config.leagues] if config is not None else ["sync"]
    checks = [Check(name, CheckStatus.WARN, NO_STATE, fix=fix) for name in syncs]
    checks.append(_pause(paused))
    checks.extend(Check(name, CheckStatus.WARN, NO_STATE, fix=fix) for name in ("proposals", "deadlines", "llm"))
    return checks


def _store_checks(
    opener: Callable[[], Store], config: Config | None, paused: PauseState | None, now: datetime
) -> list[Check]:
    """The checks that read ``state.db``: syncs, pause, proposals, deadlines, Claude spend."""
    try:
        store = opener()
    except (SchemaError, sqlite3.Error, OSError) as exc:
        reason = f"state.db did not open: {exc}"
        return [
            Check("store", CheckStatus.FAIL, reason, fix="fm sync"),
            _skipped("sync", reason),
            _pause(paused),
            _skipped("proposals", reason),
            _skipped("deadlines", reason),
            _skipped("llm", reason),
        ]
    with store:
        checks = [_sync(store, league, now) for league in config.leagues] if config is not None else [_skipped("sync")]
        checks.append(_pause(paused))
        open_rows = store.proposals.open()
        keys = {row.row_id: row.key for row in store.leagues.all()}
        checks.append(_proposals(open_rows))
        checks.append(_deadlines(open_rows, keys, now))
        checks.append(_llm(store, config, now))
    return checks


def _sync(store: Store, league: League, now: datetime) -> Check:
    name, fix = f"sync:{league.key}", f"fm sync --league {league.key}"
    row = store.leagues.by_key(league.key)
    if row is None:
        detail = f"{league.sport.upper()} league {league.espn_league_id} has never been synced"
        return Check(name, CheckStatus.FAIL, detail, fix=fix)
    age = now - row.as_of
    detail = f"{row.name or 'league'} synced {_age(age)} ago ({render.stamp(row.as_of)})"
    if age >= SYNC_FAIL_AFTER:
        return Check(name, CheckStatus.FAIL, f"{detail}; the stored roster and standings are stale", fix=fix)
    if age >= SYNC_WARN_AFTER:
        return Check(name, CheckStatus.WARN, f"{detail}; the tick syncs hourly", fix=fix)
    return Check(name, CheckStatus.OK, detail)


def _pause(paused: PauseState | None) -> Check:
    if paused is None:
        return Check("pause", CheckStatus.OK, "not paused")
    detail = f"{paused.describe()}; nothing executes or auto-approves until resumed"
    return Check("pause", CheckStatus.WARN, detail, fix="fm resume")


def _proposals(open_rows: Sequence[ProposalRow]) -> Check:
    if not open_rows:
        return Check("proposals", CheckStatus.OK, "no open proposals")
    counts = {status: sum(1 for row in open_rows if row.status == status) for status in OPEN_PROPOSAL_STATUSES}
    by_status = ", ".join(f"{count} {status}" for status, count in counts.items() if count)
    return Check("proposals", CheckStatus.OK, f"{len(open_rows)} open: {by_status}")


def _deadlines(open_rows: Sequence[ProposalRow], keys: Mapping[int, str], now: datetime) -> Check:
    due = sorted(
        ((row.deadline, row) for row in open_rows if row.deadline is not None and row.deadline - now <= DEADLINE_SOON),
        key=lambda pair: pair[0],
    )
    if not due:
        return Check("deadlines", CheckStatus.OK, f"no open proposal is due within {_age(DEADLINE_SOON)}")
    parts: list[str] = []
    for deadline, row in due:
        left = deadline - now
        when = f"overdue by {_age(-left)}" if left <= timedelta(0) else f"due in {_age(left)}"
        parts.append(f"{keys.get(row.league_id, '?')} #{row.row_id} {row.kind} ({row.status}) {when}")
    return Check("deadlines", CheckStatus.WARN, "; ".join(parts), fix="fm proposals list")


def _llm(store: Store, config: Config | None, now: datetime) -> Check:
    if config is None:
        return _skipped("llm")
    budget = config.llm.daily_budget_usd
    spent = store.llm_usage.cost_since(_day_start(now))
    if budget <= 0:
        detail = f"daily Claude budget is $0.00, so the advisor is off (template rationales only); ${spent:.2f} used"
        return Check("llm", CheckStatus.OK, detail)
    detail = f"Claude spend since 00:00 UTC: ${spent:.2f} of ${budget:.2f}"
    if spent >= budget:
        return Check("llm", CheckStatus.WARN, f"{detail}; the daily cap is reached, calls are refused until 00:00 UTC")
    return Check("llm", CheckStatus.OK, detail)


def _day_start(now: datetime) -> datetime:
    """00:00 UTC of ``now``'s day, as ``fm.advisor.client.day_start`` opens the budget window (not imported: that
    module loads the Anthropic SDK)."""
    return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


# --- online checks ----------------------------------------------------------------------------------------------------


def _espn(league: League, session: EspnSession | None, factory: EspnClientFactory) -> Check:
    name = f"espn:{league.key}"
    if session is None:
        return Check(name, CheckStatus.SKIP, "no usable ESPN session to read with (see session)")
    try:
        with factory(league, session) as client:
            read = client.get_view("mStatus")
    except EspnAuthError as exc:
        return Check(name, CheckStatus.FAIL, f"ESPN refused the session: {exc}", fix="fm login")
    except EspnClientError as exc:
        return Check(name, CheckStatus.FAIL, f"ESPN read failed: {exc}")
    status = LeagueStatus.model_validate(read.data.get("status") or {})
    detail = (
        f"ESPN answered mStatus: scoring period {status.latest_scoring_period}, "
        f"matchup period {status.current_matchup_period}"
    )
    return Check(name, CheckStatus.OK, detail)


def _channel(config: Config | None, http_client: httpx.Client | None) -> Check:
    if config is None:
        return _skipped("channel")
    if config.notify.channel == "none":
        return Check("channel", CheckStatus.SKIP, 'no phone channel to ping (notify.channel = "none")')
    try:
        channel = open_channel(config, client=http_client)
    except NotifyError as exc:
        return Check("channel", CheckStatus.FAIL, str(exc), fix="fm notify setup")
    try:
        if isinstance(channel, TelegramChannel):
            return Check("channel", CheckStatus.OK, f"telegram bot @{channel.username()} answered getMe")
        if isinstance(channel, NtfyChannel):
            return _ntfy_health(channel.client)
        return Check("channel", CheckStatus.SKIP, f"no ping for the {channel.name} channel")
    except NotifyError as exc:
        return Check("channel", CheckStatus.FAIL, f"telegram getMe failed: {exc}", fix="fm notify setup")
    except httpx.HTTPError as exc:
        detail = f"ntfy server {DEFAULT_SERVER} unreachable: {type(exc).__name__}: {exc}"
        return Check("channel", CheckStatus.FAIL, detail)
    finally:
        channel.close()


def _ntfy_health(client: httpx.Client) -> Check:
    response = client.get(f"{DEFAULT_SERVER}{NTFY_HEALTH_PATH}", timeout=REQUEST_TIMEOUT)
    try:
        body = response.json()
    except ValueError:
        body = None
    healthy = body.get("healthy") if isinstance(body, dict) else None
    if response.status_code != 200 or healthy is False:
        detail = f"ntfy server {DEFAULT_SERVER} answered HTTP {response.status_code}" + (
            " and reports unhealthy" if healthy is False else ""
        )
        return Check("channel", CheckStatus.FAIL, detail)
    return Check("channel", CheckStatus.OK, f"ntfy server {DEFAULT_SERVER} is healthy (HTTP {response.status_code})")


# --- helpers ----------------------------------------------------------------------------------------------------------


def _age(delta: timedelta) -> str:
    """``12 min``, ``3 h 5 min``, ``2 d 4 h``: a duration for a detail line, never negative."""
    seconds = max(0, int(delta.total_seconds()))
    minutes, hours, days = seconds // 60, seconds // 3600, seconds // 86400
    if days >= 1:
        return f"{days} d {hours % 24} h"
    if hours >= 1:
        return f"{hours} h {minutes % 60} min"
    return f"{minutes} min"


__all__ = [
    "ADVISOR_KEY",
    "BROWSER_NAMES",
    "BROWSER_PATHS",
    "DEADLINE_SOON",
    "NO_STATE",
    "NTFY_HEALTH_PATH",
    "SESSION_RECORD_FRESH",
    "SYNC_FAIL_AFTER",
    "SYNC_WARN_AFTER",
    "TICK_FAIL_AFTER",
    "TICK_LATE_FIX",
    "TICK_STATE_FILE",
    "TICK_WARN_AFTER",
    "Check",
    "CheckStatus",
    "EspnClientFactory",
    "find_browsers",
    "has_failure",
    "run_checks",
]
