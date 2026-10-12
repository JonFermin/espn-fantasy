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
  (FAIL). The cookie comes from the injectable loader, which is called only once the profile exists, so a fresh
  install never starts a browser; a profile that will not open (held by another ``fm`` process) is a WARN.
- ``browser``: Microsoft Edge or Google Chrome is installed, found where Playwright's registry looks for the channel
  executables (:func:`find_browsers`) without launching anything.
- ``notify``: the ``[notify]`` channel's secrets are in ``.env``.
- ``tick``: the last tick in ``tick-state.json``: WARN after :data:`TICK_WARN_AFTER`, FAIL after
  :data:`TICK_FAIL_AFTER` or when no tick ever ran.
- ``sync:<league>``: the store's ``as_of`` per configured league: WARN after :data:`SYNC_WARN_AFTER` (the tick syncs
  a league hourly, so an older sync means the ticks are not syncing), FAIL after :data:`SYNC_FAIL_AFTER` or before
  the first sync.
- ``pause``: the kill switch (a WARN while thrown).
- ``proposals``: what is open, by status. ``deadlines``: open proposals due within :data:`DEADLINE_SOON` or overdue.
- ``llm``: today's recorded Claude spend against ``[llm] daily_budget_usd``; the cap reached is a WARN.

``--online`` adds ``espn:<league>`` (an ``mStatus`` read through :class:`fm.espn.client.EspnClient`) and ``channel``
(Telegram ``getMe``, or the ntfy server's ``/v1/health``).

Verdicts: FAIL means something cannot do its job and ``fix`` names the command that repairs it; WARN means it works but
needs attention; SKIP means the check could not run because an earlier one failed (no config, no session). This
module reads only: it never touches ESPN's write path and imports neither ``fm.executor`` nor the browser flows,
and it stays clear of the tick's and the advisor's import graphs so the dashboard can load it cheaply.
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
from fm.notify import NotifyError, NtfyChannel, TelegramChannel, open_channel
from fm.notify.ntfy import DEFAULT_SERVER, REQUEST_TIMEOUT
from fm.proposals import PauseState, pause_state
from fm.store import OPEN_PROPOSAL_STATUSES, ProposalRow, SchemaError, Store

TICK_STATE_FILE = "tick-state.json"
"""``fm.jobs.tick.STATE_FILE``, read here directly: importing the tick would pull in the executor and the flows."""
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
    config_loader: Callable[[], Config] = load_config,
    store_opener: Callable[[], Store] = Store.open,
    session_loader: Callable[[], EspnSession] = load_session,
    profile_dir: Path | None = None,
    browsers: Callable[[], Mapping[str, Path]] = find_browsers,
    tick_state: Path | None = None,
    pause_reader: Callable[[], PauseState | None] = pause_state,
    client_factory: EspnClientFactory = _default_client,
    http_client: httpx.Client | None = None,
) -> list[Check]:
    """Every check, in the order the module docstring lists them. Pure data; nothing is written anywhere.

    ``session_loader`` is called only when ``profile_dir`` (default ``fm.paths.browser_profile_dir()``) holds a
    profile. ``online`` adds the ESPN reads (through ``client_factory``) and the channel ping (through
    ``http_client`` when given, else the channel's own client).
    """
    at = datetime.now(UTC) if now is None else now.astimezone(UTC)
    profile = profile_dir if profile_dir is not None else paths.browser_profile_dir()
    state_file = tick_state if tick_state is not None else paths.config_dir() / TICK_STATE_FILE

    config_check, config = _config(config_loader)
    session_check, session = _session(profile, session_loader, at)
    checks = [config_check, session_check, _browser(browsers()), _notify(config), _tick(state_file, at)]
    checks.extend(_store_checks(store_opener, config, pause_reader(), at))
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


def _session(
    profile: Path, loader: Callable[[], EspnSession], now: datetime
) -> tuple[Check, EspnSession | None]:
    """The session verdict and, when ESPN reads can use it, the session itself."""
    name = "session"
    if not profile_exists(profile):
        detail = "no browser profile yet: nothing is signed in to ESPN"
        return Check(name, CheckStatus.FAIL, detail, fix="fm login"), None
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
    missing = [name for name in config.missing_secrets() if name != ADVISOR_KEY]
    if missing:
        detail = f"{channel} is not set up: {', '.join(missing)} missing from .env"
        return Check("notify", CheckStatus.FAIL, detail, fix="fm notify setup")
    return Check("notify", CheckStatus.OK, f"{channel} configured")


def _tick(path: Path, now: datetime) -> Check:
    last = _last_tick(path)
    if last is None:
        return Check("tick", CheckStatus.FAIL, f"no tick has run yet (no last tick in {path.name})", fix="fm run")
    age = now - last
    detail = f"last tick {_age(age)} ago ({render.stamp(last)})"
    if age >= TICK_FAIL_AFTER:
        return Check("tick", CheckStatus.FAIL, f"{detail}; ticks have stopped", fix="fm run")
    if age >= TICK_WARN_AFTER:
        return Check("tick", CheckStatus.WARN, f"{detail}; the next tick is overdue", fix="fm run")
    return Check("tick", CheckStatus.OK, detail)


def _last_tick(path: Path) -> datetime | None:
    """``last_tick`` from the tick's state file; ``None`` when the file is missing, unreadable or naive (the tick
    itself ignores such a file, so it has no memory either)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get("last_tick") if isinstance(data, dict) else None
        if not isinstance(value, str):
            return None
        parsed = datetime.fromisoformat(value)
    except (OSError, ValueError):
        return None
    return None if parsed.tzinfo is None else parsed.astimezone(UTC)


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
    "NTFY_HEALTH_PATH",
    "SYNC_FAIL_AFTER",
    "SYNC_WARN_AFTER",
    "TICK_FAIL_AFTER",
    "TICK_STATE_FILE",
    "TICK_WARN_AFTER",
    "Check",
    "CheckStatus",
    "EspnClientFactory",
    "find_browsers",
    "has_failure",
    "run_checks",
]
