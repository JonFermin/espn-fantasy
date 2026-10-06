"""Real-league capture (ROADMAP #14): read every ESPN view, capture each write flow's request with the guard aborting
it, prove nothing changed, and turn the captures into scrubbed fixtures. ``scripts/capture/README.md`` has the protocol.

    uv run python scripts/capture/capture.py reads                # every read view + the unknown-settling probes
    uv run python scripts/capture/capture.py webclient            # calendars, stat splits, transaction code (bundle)
    uv run python scripts/capture/capture.py snapshot             # league state before any write flow
    uv run python scripts/capture/capture.py explore URL [STEPS]  # one guarded page visit: screenshot + aria snapshot
    uv run python scripts/capture/capture.py writes               # a person signs in, drives each flow; guard aborts
    uv run python scripts/capture/capture.py verify               # re-read the leagues and compare with the snapshot
    uv run python scripts/capture/capture.py fixtures             # scrub + trim into tests/fixtures/espn/real/
    uv run python scripts/capture/capture.py web-login            # fallback: a sign-in window that stays open
    uv run python scripts/capture/capture.py trade-review         # trade builder up to its review step, never Send
    uv run python scripts/capture/capture.py derive-trade         # the offer's request, from ESPN's code, offline

Every browser session starts with :class:`guard.WriteGuard` installed and proven before any ESPN page opens. Raw
captures hold real manager names and ids, so ``--raw`` defaults to the system temp dir and may not be inside the repo;
only ``fixtures`` writes into the repo, and only scrubbed data. URLs given to ``explore`` may use ``{league}``,
``{team}``, ``{season}`` and ``{sport}``, filled from the ``--league`` entry of config.toml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fixture_writer import WEBCLIENT_FILE, make_fixtures
from guard import FANTASY_DOMAIN, WRITE_HOST, BlockedRequest, GuardError, GuardProofError, WriteGuard
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, Response
from reads import (
    RequestLog,
    SnapshotError,
    capture_league,
    compare_snapshots,
    league_client,
    periods_left_in_matchup,
    snapshot,
)
from webclient import (
    BUNDLE_MARKER,
    Calendar,
    CalendarError,
    WebClientError,
    error_codes,
    extract_calendars,
    extract_stat_settings,
    extract_transaction_code,
    match_calendar,
)

from fm.browser.session import BrowserSession, LaunchOptions, open_browser
from fm.config import Config, League, load_config
from fm.espn.auth import EspnSession, harvest_session
from fm.espn.client import EspnClient, View

DEFAULT_RAW = Path(tempfile.gettempdir()) / "espn-fantasy-capture"
REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "espn" / "real"
SNAPSHOT_BEFORE = "snapshot_before.json"
TEAM_URL = "https://fantasy.espn.com/{sport}/team?leagueId={league}&teamId={team}&seasonId={season}"
TRADE_BUILDER_URL = (
    "https://fantasy.espn.com/{sport}/team/trade?leagueId={league}&teamId={partner}&fromTeamId={team}&seasonId={season}"
)
TRADE_REVIEW = "trade_review"
DERIVED_WRITES = "writes_derived"
DERIVE_JS = Path(__file__).resolve().parent / "derive_trade.cjs"
LOGIN_REQUIRED = "Log in Required"
"""The heading the web app shows when the profile has no OneID web session (the API cookies may still work)."""
LOGIN_POLL_MS = 2000
LOGIN_SETTLED_POLLS = 3
"""Polls in a row without the heading on a fantasy page before the team page is reloaded to confirm a sign-in."""
_GAME_IN_URL = re.compile(r"/games/(ffl|fba)/")


# --- shared -----------------------------------------------------------------------------------------------------------


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def guarded_browser(
    *, headless: bool = True, pace_pages: bool = True, sign_in: bool = False
) -> Iterator[tuple[BrowserSession, WriteGuard, Page]]:
    """The persistent profile with the write guard installed and proven on ``about:blank`` before anything else."""
    guard = WriteGuard(pace_reads=pace_pages, sign_in=sign_in)
    with open_browser(LaunchOptions(headless=headless)) as browser:
        context = browser.context
        guard.install(context)
        page = context.pages[0] if context.pages else context.new_page()
        for extra in context.pages[1:]:
            extra.close()
        if page.url != "about:blank":
            page.goto("about:blank")
        guard.prove(page)
        try:
            yield browser, guard, page
        finally:
            guard.check()


def load_espn_session(*, headless: bool = True) -> EspnSession:
    """The cookies from the profile, read inside a guarded browser that opens no ESPN page."""
    with guarded_browser(headless=headless) as (browser, _guard, _page):
        return harvest_session(browser.context)


def leagues(config: Config, keys: Sequence[str]) -> list[League]:
    return [config.league(key) for key in keys] if keys else list(config.leagues)


def league_url(template: str, league: League | None) -> str:
    """Fill ``{league}``, ``{team}``, ``{season}`` and ``{sport}`` from a configured league, so ids stay in config."""
    if league is None:
        return template
    sport = "football" if league.sport == "nfl" else "basketball"
    return template.format(league=league.espn_league_id, team=league.team_id, season=league.season, sport=sport)


def open_espn_page(page: Page, guard: WriteGuard, url: str, settle_ms: int) -> None:
    """Navigate, let the app settle, then prove the guard again from the ESPN origin before anything else."""
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_timeout(settle_ms)
    guard.prove(page)


def web_login_missing(page: Page) -> bool:
    return page.get_by_role("heading", name=LOGIN_REQUIRED).count() > 0


def guard_report(guard: WriteGuard) -> dict[str, Any]:
    return {
        "blocked": [record.as_json() for record in guard.blocked],
        "passed": guard.passed,
        "reads": [read.as_json() for read in guard.reads],
        "websockets": guard.websockets,
        "leaks": guard.leaks,
        "errors": guard.errors,
    }


# --- snapshot / verify ------------------------------------------------------------------------------------------------


def load_calendar(raw: Path, league: League) -> Calendar:
    """The league's season calendar: ``calendar_<game>.json`` from ``webclient``, else the committed fixture."""
    for path in (raw / f"calendar_{league.game}.json", FIXTURES / league.game / "calendar.json"):
        if path.exists():
            data = read_json(path)
            if data.get("game") == league.game and data.get("season") == league.season:
                return Calendar.from_json(data)
    raise CalendarError(f"{league.key}: no {league.season} {league.game} calendar; run `capture.py webclient` first")


def snapshot_periods(client: EspnClient, league: League, raw: Path) -> list[int]:
    """The current scoring period through the end of its matchup, from ``mSettings`` and the season calendar."""
    settings = client.get_view(View.SETTINGS, key="snapshot").data
    current = settings.get("scoringPeriodId") or (settings.get("status") or {}).get("latestScoringPeriod")
    if not isinstance(current, int):
        raise SnapshotError(f"{league.key}: mSettings names no current scoring period")
    schedule = (settings.get("settings") or {}).get("scheduleSettings") or {}
    if not isinstance(schedule.get("matchupPeriods"), dict) or not isinstance(schedule.get("periodTypeId"), int):
        raise SnapshotError(f"{league.key}: mSettings has no scheduleSettings.matchupPeriods/periodTypeId")
    calendar = load_calendar(raw, league)
    return periods_left_in_matchup(calendar.matchup_days(schedule["matchupPeriods"], schedule["periodTypeId"]), current)


def cmd_snapshot(args: argparse.Namespace) -> int:
    config = load_config()
    session = load_espn_session()
    log = RequestLog()
    result: dict[str, Any] = {}
    for league in leagues(config, args.league):
        with league_client(league, session, args.raw, log) as client:
            try:
                periods = snapshot_periods(client, league, args.raw)
            except (CalendarError, SnapshotError) as exc:
                print(exc)
                return 2
            result[league.key] = snapshot(client, league.team_id, periods)
        taken = result[league.key]
        print(
            f"{league.key}: scoring periods {periods[0]}-{periods[-1]}, {len(taken['transactions'])} transaction "
            f"records, {len(taken['pending'])} pending items"
        )
    write_json(args.raw / SNAPSHOT_BEFORE, result)
    return 0


def verify_leagues(raw: Path, chosen: Sequence[League], session: EspnSession) -> bool:
    """Re-read each league over the snapshot's periods and compare; prints the outcome and returns True when nothing
    of ours changed."""
    before = read_json(raw / SNAPSHOT_BEFORE)
    log = RequestLog()
    report: dict[str, Any] = {"checked_at": datetime.now(UTC).isoformat(), "leagues": {}}
    clean = True
    for league in chosen:
        earlier = before.get(league.key)
        if earlier is None:
            print(f"{league.key}: not in {SNAPSHOT_BEFORE}; run `capture.py snapshot` first")
            return False
        try:
            with league_client(league, session, raw, log) as client:
                after = snapshot(client, league.team_id, earlier.get("periods") or [])
            problems, notes = compare_snapshots(earlier, after, league.team_id)
        except (SnapshotError, ValueError, KeyError) as exc:
            print(f"{league.key}: cannot verify: {exc}")
            return False
        report["leagues"][league.key] = {"problems": problems, "notes": notes, "after": after}
        clean = clean and not problems
        print(f"{league.key}: {'CHANGED' if problems else 'unchanged'} ({len(notes)} notes about other teams)")
        for line in problems:
            print(f"  PROBLEM {line}")
    write_json(raw / f"verify_{datetime.now(UTC):%Y%m%dT%H%M%S}.json", report)
    return clean


def cmd_verify(args: argparse.Namespace) -> int:
    config = load_config()
    return 0 if verify_leagues(args.raw, leagues(config, args.league), load_espn_session()) else 1


# --- reads ------------------------------------------------------------------------------------------------------------


def cmd_reads(args: argparse.Namespace) -> int:
    config = load_config()
    session = load_espn_session()
    log = RequestLog()
    report: dict[str, Any] = {"captured_at": datetime.now(UTC).isoformat(), "leagues": {}}
    for league in leagues(config, args.league):
        with league_client(league, session, args.raw, log) as client:
            result = capture_league(client, league, extra_views=args.view)
        report["leagues"][league.key] = result.as_json()
        print(f"{league.key}: {len(result.captures)} captures, {len(result.failures)} failures")
        for failure in result.failures:
            print(f"  FAILED {failure}")
    report["requests"] = log.summary()
    report["request_log"] = log.entries
    report["session"] = {"espn_s2_expires": session.expires_at.isoformat() if session.expires_at else None}
    write_json(args.raw / "reads.json", report)
    return 0


# --- webclient --------------------------------------------------------------------------------------------------------


def cmd_webclient(args: argparse.Namespace) -> int:
    """What the web client ships as code, from the bundle a guarded page load fetches anyway: each league's season
    calendar (matched against its pro schedule from ``reads``), the stat split labels and the transaction code."""
    config = load_config()
    chosen = leagues(config, args.league)
    reads = read_json(args.raw / "reads.json")
    bundles: dict[str, bytes] = {}

    def keep(response: Response) -> None:
        if BUNDLE_MARKER in response.url and response.url not in bundles:
            bundles[response.url] = response.body()

    with guarded_browser() as (browser, guard, page):
        browser.context.on("response", keep)
        open_espn_page(page, guard, league_url(TEAM_URL, chosen[0]), args.settle_ms)
    if not bundles:
        print("the web client bundle was not loaded; nothing extracted")
        return 1
    url, body = next(iter(bundles.items()))
    text = body.decode("utf-8", errors="replace")
    build = url.split("/kona/")[1].split("/")[0] if "/kona/" in url else None
    try:
        code = extract_transaction_code(text)
    except WebClientError as exc:
        print(f"transaction code: {exc}")
        return 1
    write_json(
        args.raw / WEBCLIENT_FILE,
        {
            "webClientBuild": build,
            "bundle": url,
            "bundleSha256": hashlib.sha256(body).hexdigest(),
            "transactionCode": code,
            "statSettings": extract_stat_settings(text),
        },
    )
    print(f"transaction code: {len(code['excerpts'])} excerpts (build {build})")
    calendars = extract_calendars(text)
    for league in chosen:
        captures = reads["leagues"][league.key]["captures"]
        pro = next(c for c in reversed(captures) if c["name"] == "pro_schedule")
        schedule = read_json(args.raw / "cache" / pro["saved"])
        try:
            calendar = match_calendar(calendars, schedule, weekly_game=league.sport == "nfl")
        except CalendarError as exc:
            print(f"{league.key}: {exc}")
            return 1
        payload = {"game": league.game, "season": league.season, "webClientBuild": build, **calendar.as_json()}
        payload["errorCodes"] = error_codes(text)
        write_json(args.raw / f"calendar_{league.game}.json", payload)
        print(f"{league.key}: calendar with {len(calendar.scoring_periods)} scoring periods (build {build})")
    return 0


# --- explore ----------------------------------------------------------------------------------------------------------


def run_steps(page: Page, steps: Sequence[dict[str, Any]], out: Path, guard: WriteGuard) -> None:
    """Exploration steps: ``{"click": {"role", "name", "nth", "exact"}}``, ``{"text": ...}``, ``{"role": ..., "name":
    ...}`` (count matches), ``{"wait": ms}``, ``{"shot": name}``, ``{"goto": url}``. The guard is checked after each."""
    for index, step in enumerate(steps):
        if "goto" in step:
            page.goto(step["goto"], wait_until="domcontentloaded")
        elif "click" in step:
            spec = step["click"]
            locator = page.get_by_role(spec["role"], name=spec.get("name"), exact=spec.get("exact", False))
            locator.nth(spec.get("nth", 0)).click(timeout=spec.get("timeout", 15000))
        elif "text" in step:
            page.get_by_text(step["text"], exact=step.get("exact", False)).nth(step.get("nth", 0)).click()
        elif "role" in step:
            print(page.get_by_role(step["role"], name=step.get("name")).count(), "matches for", step)
        elif "wait" in step:
            page.wait_for_timeout(step["wait"])
        elif "shot" in step:
            dump_page(page, out, step["shot"])
        guard.check()
        print(f"step {index}: {step}")


def dump_page(page: Page, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    (out / f"{name}.aria.txt").write_text(page.locator("body").aria_snapshot(), encoding="utf-8")
    print(f"saved {name}")


def cmd_explore(args: argparse.Namespace) -> int:
    steps = json.loads(Path(args.steps).read_text(encoding="utf-8")) if args.steps else []
    out = args.raw / "explore" / args.name
    league = load_config().league(args.league[0]) if args.league else None
    steps = [{**step, "goto": league_url(step["goto"], league)} if "goto" in step else step for step in steps]
    network: list[str] = []
    with guarded_browser(headless=not args.headed, pace_pages=not args.no_page_pacing) as (browser, guard, page):
        browser.context.on("response", lambda response: _log_response(network, response))
        open_espn_page(page, guard, league_url(args.url, league), args.settle_ms)
        try:
            dump_page(page, out, "start")
            run_steps(page, steps, out, guard)
        finally:
            write_json(out / "guard.json", {**guard_report(guard), "network": network})
    return 0


def _log_response(network: list[str], response: Response) -> None:
    request = response.request
    if request.resource_type in ("fetch", "xhr", "document") and "espncdn" not in request.url:
        network.append(f"{request.method} {response.status} {request.resource_type} {request.url[:160]}")


# --- writes -----------------------------------------------------------------------------------------------------------


def is_transaction_capture(record: BlockedRequest) -> bool:
    return WRITE_HOST in record.url or "/transactions" in record.url or "/pendingTransactions" in record.url


def save_write(out: Path, record: BlockedRequest, counts: dict[str, int]) -> Path:
    """One aborted write as ``writes/<game>_<type>[_<executionType>]_<n>.json`` (raw: scrubbed by ``fixtures``)."""
    match = _GAME_IN_URL.search(record.url)
    game = match.group(1) if match else "unknown"
    body = record.body if isinstance(record.body, dict) else {}
    label = str(body.get("type") or "request")
    if body.get("executionType") not in (None, "EXECUTE"):
        label += f"_{body['executionType']}"
    key = f"{game}_{label}"
    counts[key] = counts.get(key, 0) + 1
    path = out / f"{key}_{counts[key]}.json"
    write_json(path, record.as_json())
    return path


def wait_for_web_login(page: Page, guard: WriteGuard, url: str, *, minutes: float, settle_ms: int) -> bool:
    """Wait while a person signs in in this guarded window; True once the team page renders without the
    "Log in Required" heading. Every request the guard aborts meanwhile is printed, so a sign-in request it stopped
    is visible. False when the time runs out or the window closes."""
    deadline = time.monotonic() + minutes * 60
    reported = len(guard.captures)
    clear = 0
    while time.monotonic() < deadline:
        try:
            page.wait_for_timeout(LOGIN_POLL_MS)
            on_fantasy = urlsplit(page.url).hostname == FANTASY_DOMAIN
            clear = clear + 1 if on_fantasy and not web_login_missing(page) else 0
            if clear >= LOGIN_SETTLED_POLLS:
                open_espn_page(page, guard, url, settle_ms)  # a fresh load, so a half-rendered page cannot pass
                if not web_login_missing(page):
                    return True
                clear = 0
        except PlaywrightError:
            return False  # the window was closed
        guard.check()
        for record in guard.captures[reported:]:
            print(f"  the guard aborted {record.method} {record.url[:120]} ({record.reason})")
        reported = len(guard.captures)
    return False


def capture_writes(page: Page, guard: WriteGuard, out: Path, minutes: float) -> int:
    """Save every transaction request the guard aborts until the window closes or ``minutes`` pass; returns the count.
    A guard failure raises :class:`guard.GuardError`."""
    counts: dict[str, int] = {}
    seen = 0
    other = 0
    deadline = time.monotonic() + minutes * 60
    while time.monotonic() < deadline:
        try:
            page.wait_for_timeout(1000)
        except PlaywrightError:
            break  # the window was closed
        guard.check()
        fresh = guard.captures[seen:]
        seen += len(fresh)
        for record in fresh:
            if is_transaction_capture(record):
                saved = save_write(out, record, counts)
                print(f"captured and aborted: {saved.name}")
            else:
                # Kept raw (outside the repo, never made into fixtures) so a write sent elsewhere is visible.
                other += 1
                write_json(out / "other" / f"aborted_{other}.json", record.as_json())
                print(f"  aborted, not a transaction: {record.method} {record.url[:120]} ({record.reason})")
    return sum(counts.values())


def cmd_writes(args: argparse.Namespace) -> int:
    """Assisted capture: a person drives each write flow in a guarded window; every transaction request is aborted
    by the guard and saved. Closing the window (or the time limit) ends the session, and the leagues are verified
    whatever happened in the window (a failed sign-in or a guard failure included)."""
    config = load_config()
    chosen = leagues(config, args.league)
    if not (args.raw / SNAPSHOT_BEFORE).exists():
        print("run `capture.py snapshot` first: the verification compares against it")
        return 2
    url = league_url(TEAM_URL, chosen[0])
    session: EspnSession | None = None
    outcome = 0
    try:
        with guarded_browser(headless=args.headless) as (browser, guard, page):
            session = harvest_session(browser.context)
            browser.context.add_init_script(SEND_TRAP_JS)  # no offer is sent or answered from this window either
            open_espn_page(page, guard, url, args.settle_ms)
            if web_login_missing(page):
                if args.headless:
                    print(f"The web app shows {LOGIN_REQUIRED!r} and nobody can sign in headless; drop --headless.")
                    return 2
                print(
                    f"The web app shows {LOGIN_REQUIRED!r}: the profile has the API cookies but no web sign-in. Sign "
                    "in in this window (click Log In and finish the Disney sign-in, one-time code included; reload "
                    "the page if it still says Log in Required afterwards). The write guard stays on; the sign-in is "
                    "served from registerdisney.go.com, which it should not block, and every request it aborts is "
                    f"listed below. The capture starts once the team page shows your team (waiting up to "
                    f"{args.login_minutes:g} minutes)."
                )
                if not wait_for_web_login(page, guard, url, minutes=args.login_minutes, settle_ms=args.settle_ms):
                    print(
                        "No web sign-in. If the guard aborted a request the sign-in needed (listed above), run "
                        "`capture.py web-login`, sign in there, close its window, then run this command again."
                    )
                    outcome = 2
            if outcome == 0:
                print(
                    "Drive each flow in the window until the app sends its request (the guard aborts it and the app "
                    "shows an error): a bench swap, a free-agent add/drop and a waiver claim, in each league. No "
                    "trades here: Send, Accept and Decline are disabled in this window, and `capture.py trade-review` "
                    "covers the trade builder. Never touch a real pending offer. Close the window when done."
                )
                print(f"{capture_writes(page, guard, args.raw / 'writes', args.minutes)} write requests captured")
    except GuardError as exc:
        print(f"STOPPED: the write guard failed: {exc}")
        outcome = 1
    print("verifying the leagues")
    clean = verify_leagues(args.raw, chosen, session if session is not None else load_espn_session())
    return outcome if clean else 1


def cmd_web_login(args: argparse.Namespace) -> int:
    """Fallback when the write guard blocks the sign-in inside ``writes``: a headed window on the persistent profile
    for a manual sign-in that stays open until you close it (``fm login`` clears ESPN's cookies and closes its window
    as soon as the API cookies land, before the web app has a session). League writes stay blocked: the guard runs in
    sign-in mode, which lets only non-GET requests to ESPN hosts outside ``fantasy.espn.com`` through. Afterwards a
    fresh headless browser checks whether the web session survived the restart that ``writes`` will need."""
    chosen = leagues(load_config(), args.league)
    url = league_url(TEAM_URL, chosen[0])
    with guarded_browser(headless=False, sign_in=True) as (_browser, guard, page):
        try:
            page.goto(url, wait_until="domcontentloaded")
            print("Sign in in this window (click Log In), wait until the team page shows your team, then close it.")
            deadline = time.monotonic() + args.minutes * 60
            while time.monotonic() < deadline:
                try:
                    page.wait_for_timeout(1000)
                except PlaywrightError:
                    break  # the window was closed
                guard.check()
        finally:
            write_json(args.raw / f"web_login_{datetime.now(UTC):%Y%m%dT%H%M%S}.json", guard_report(guard))
    print(f"the sign-in guard let {len(guard.passed)} non-GET requests to ESPN hosts through (see web_login_*.json)")
    with guarded_browser() as (_browser, check_guard, check_page):
        open_espn_page(check_page, check_guard, url, args.settle_ms)
        missing = web_login_missing(check_page)
    if missing:
        print(f"After a browser restart the web app still shows {LOGIN_REQUIRED!r}: the web session did not persist.")
        return 3
    print("The web session survived a browser restart; run `capture.py snapshot` and then `writes`.")
    return 0


# --- trade review -----------------------------------------------------------------------------------------------------

FORBIDDEN_CONTROL = re.compile(
    r"\b(send|propose|submit|accept|decline|reject|cancel|confirm|counter|veto|withdraw|delete|drop)\b", re.IGNORECASE
)
"""Accessible names ``trade-review`` refuses to click: anything that could send, answer or cancel an offer."""

SEND_TRAP_JS = """
(() => {
  if (window.__fmSendTrap) return;
  const forbidden = /send trade|propose trade|send|submit|accept|decline|counter/i;
  const hit = (node) => {
    const el = node && node.closest ? node.closest('button, a, input, [role="button"], [role="link"]') : null;
    if (!el) return false;
    const text = [el.getAttribute('aria-label'), el.textContent, el.value].filter(Boolean).join(' ');
    return forbidden.test(text);
  };
  const trap = (event) => {
    const target = event.type === 'submit' ? (event.submitter || event.target) : event.target;
    if (event.type === 'keydown' && event.key !== 'Enter' && event.key !== ' ') return;
    if (event.type === 'submit' || hit(target)) {
      event.preventDefault();
      event.stopImmediatePropagation();
      window.__fmSendTrapHits = (window.__fmSendTrapHits || 0) + 1;
    }
  };
  for (const type of ['pointerdown', 'pointerup', 'mousedown', 'mouseup', 'click', 'keydown', 'submit']) {
    window.addEventListener(type, trap, true);
  }
  window.__fmSendTrap = true;
})();
"""
"""A capturing listener on every document of the ``trade-review`` context that swallows clicks, key presses and form
submits on any control reading like Send/Propose/Accept/Decline, so not even a person at the headed window can send an
offer through the page. The network guard and trade lock stay the real defence; this is the layer in front of them."""

REVIEW_STATE_JS = """
() => {
  const seen = (el) => !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
  const label = (el) => (el.getAttribute('aria-label') || el.textContent || '').trim().replace(/\\s+/g, ' ');
  const named = (el) => el.getAttribute('aria-label') || el.getAttribute('name') || '';
  return {
    path: location.pathname,
    headings: [...document.querySelectorAll('h1, h2, h3, h4, [role="heading"]')].filter(seen).map(label),
    buttons: [...document.querySelectorAll('button, [role="button"]')].filter(seen).map((el) => ({
      name: label(el),
      disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
    })),
    selects: [...document.querySelectorAll('select')].filter(seen).map((el) => ({
      name: named(el),
      value: el.value,
      options: [...el.options].map((o) => ({ value: o.value, text: o.text.trim(), selected: o.selected })),
    })),
    textareas: [...document.querySelectorAll('textarea')].filter(seen).map((el) => ({
      name: named(el),
      placeholder: el.placeholder,
      maxLength: el.maxLength,
      value: el.value,
    })),
    checked: [...document.querySelectorAll('input[type="checkbox"]:checked')].map(
      (el) => el.getAttribute('aria-label') || el.getAttribute('name') || ''
    ),
    dialogs: [...document.querySelectorAll('[role="dialog"], [role="alertdialog"], [aria-modal="true"]')]
      .filter(seen)
      .map((el) => ({
        label: el.getAttribute('aria-label') || '',
        text: el.innerText.split('\\n').map((line) => line.trim()).filter(Boolean),
      })),
    trapHits: window.__fmSendTrapHits || 0,
  };
}
"""


def offer_teams(snap: dict[str, Any]) -> set[int]:
    """Every team in any trade record or pending item of a snapshot: the teams of the real offers, never disturbed."""
    teams: set[int] = set()
    for field_name in ("transactions", "pending"):
        for record in (snap.get(field_name) or {}).values():
            if str(record.get("type", "")).startswith("TRADE"):
                teams.update(int(team) for team in record.get("team_ids") or [] if team)
                if record.get("team_id"):
                    teams.add(int(record["team_id"]))
    return teams


def _reserved_player_ids(value: Any) -> set[int]:
    if isinstance(value, dict):
        found = {int(value["playerId"])} if isinstance(value.get("playerId"), int) else set()
        for child in value.values():
            found |= _reserved_player_ids(child)
        return found
    if isinstance(value, list):
        return {pid for child in value for pid in _reserved_player_ids(child)}
    return set()


def tradeable(team: dict[str, Any]) -> list[dict[str, Any]]:
    """Roster entries the builder lets a person tick: not trade-locked, not held by a pending offer or move."""
    roster = team.get("roster") or {}
    reserved = _reserved_player_ids(roster.get("tradeReservedEntries"))
    picks = []
    for entry in roster.get("entries") or []:
        pool = entry.get("playerPoolEntry") or {}
        player_id = entry.get("playerId") or pool.get("id")
        if pool.get("tradeLocked") or entry.get("pendingTransactionIds") or player_id in reserved:
            continue
        name = (pool.get("player") or {}).get("fullName")
        if isinstance(player_id, int) and isinstance(name, str) and name:
            picks.append({"id": player_id, "name": name})
    return sorted(picks, key=lambda pick: pick["id"])


def choose_trade(client: EspnClient, league: League, snap: dict[str, Any], partner: int | None) -> dict[str, Any]:
    """A one-for-one offer to a team outside every real offer: the lowest-id tradeable player on each side."""
    busy = offer_teams(snap)
    if partner is None:
        team_ids = sorted(team.id for team in client.teams().data.teams)
        partner = next((team for team in team_ids if team != league.team_id and team not in busy), None)
        if partner is None:
            raise SnapshotError(f"{league.key}: every other team is part of a real offer")
    elif partner in busy or partner == league.team_id:
        raise SnapshotError(f"{league.key}: team {partner} is ours or part of a real offer; pick another")
    rosters = client.get_view(View.ROSTER, key="trade_review").data
    teams = {team["id"]: team for team in rosters.get("teams") or []}
    ours, theirs = tradeable(teams.get(league.team_id, {})), tradeable(teams.get(partner, {}))
    if not ours or not theirs:
        raise SnapshotError(f"{league.key}: no tradeable player on one side of a trade with team {partner}")
    settings = client.get_view(View.SETTINGS, key="trade_review").data
    return {
        "game": league.game,
        "seasonId": league.season,
        "leagueId": league.espn_league_id,
        "latestScoringPeriod": (settings.get("status") or {}).get("latestScoringPeriod"),
        "fromTeamId": league.team_id,
        "toTeamId": partner,
        "players": [
            {**ours[0], "teamId": league.team_id, "side": "ours"},
            {**theirs[0], "teamId": partner, "side": "theirs"},
        ],
    }


def safe_click(page: Page, role: str, name: str) -> None:
    """Click the one control with this role and exact accessible name; refuse a forbidden name or an ambiguous match."""
    if FORBIDDEN_CONTROL.search(name):
        raise GuardError(f"refusing to click {role} {name!r}: it reads like a send, answer or cancel")
    locator = page.get_by_role(role, name=name, exact=True)  # type: ignore[arg-type]
    if locator.count() != 1:
        raise GuardError(f"expected one {role} named {name!r}, found {locator.count()}; stopping")
    locator.click(timeout=15000)


def review_trade(
    page: Page, guard: WriteGuard, league: League, selection: dict[str, Any], out: Path, settle_ms: int
) -> dict[str, Any]:
    """Tick one player per side in the trade builder and press Continue; read the review step and leave. The trade
    canary runs before the page is opened and again before the first click; Send Trade Proposal is never touched."""
    canaries = guard.trade_canary(page, game=league.game, season=league.season)
    sport = "football" if league.sport == "nfl" else "basketball"
    url = TRADE_BUILDER_URL.format(
        sport=sport,
        league=league.espn_league_id,
        partner=selection["toTeamId"],
        team=league.team_id,
        season=league.season,
    )
    open_espn_page(page, guard, url, settle_ms)
    canaries += guard.trade_canary(page, game=league.game, season=league.season)
    if page.evaluate("window.__fmSendTrap === true") is not True:
        raise GuardProofError("the send trap is not installed on the trade builder")
    report: dict[str, Any] = {
        "league": league.key,
        "game": league.game,
        "observed_at": datetime.now(UTC).isoformat(),
        "builder_url": url,
        "selection": selection,
        "canary": [record.as_json() for record in canaries],
    }
    if web_login_missing(page):
        report["status"] = "login required"
        return report
    dump_page(page, out, "builder")
    report["builder"] = page.evaluate(REVIEW_STATE_JS)
    for player in selection["players"]:
        safe_click(page, "checkbox", f"Trade {player['name']}")
        page.wait_for_timeout(500)
        guard.check()
    reads_from, blocked_from = len(guard.reads), len(guard.captures)
    safe_click(page, "button", "Continue")
    page.wait_for_timeout(settle_ms)
    guard.check()
    dump_page(page, out, "review")
    state = page.evaluate(REVIEW_STATE_JS)
    report["review"] = state
    report["review_reads"] = [read.as_json() for read in guard.reads[reads_from:]]
    report["review_blocked"] = [record.as_json() for record in guard.captures[blocked_from:]]
    report["send_button_present"] = any(
        re.search(r"send trade proposal", button["name"], re.IGNORECASE) for button in state["buttons"]
    )
    report["status"] = "review reached" if report["send_button_present"] else "review not reached"
    page.goto("about:blank")
    guard.check()
    return report


def cmd_trade_review(args: argparse.Namespace) -> int:
    """Open each league's trade builder in a guarded window, tick one player per side, press Continue and record the
    review step (its reads and DOM). Send Trade Proposal is never clicked: the click helper refuses it, a page-level
    trap swallows it, and the trade lock would abort its request. Verifies every league afterwards."""
    config = load_config()
    chosen = leagues(config, args.league)
    if not (args.raw / SNAPSHOT_BEFORE).exists():
        print("run `capture.py snapshot` first: the verification compares against it")
        return 2
    snaps = read_json(args.raw / SNAPSHOT_BEFORE)
    session: EspnSession | None = None
    outcome = 0
    try:
        with guarded_browser(headless=args.headless) as (browser, guard, page):
            session = harvest_session(browser.context)
            browser.context.add_init_script(SEND_TRAP_JS)
            log = RequestLog()
            for league in chosen:
                with league_client(league, session, args.raw, log) as client:
                    selection = choose_trade(client, league, snaps[league.key], args.partner)
                out = args.raw / TRADE_REVIEW / league.game
                write_json(out / "selection.json", selection)
                report = review_trade(page, guard, league, selection, out, args.settle_ms)
                write_json(out / "review.json", report)
                print(f"{league.key}: {report['status']} ({len(report.get('review_reads', []))} reads on the review)")
                for record in report.get("review_blocked", []):
                    print(f"  the guard aborted {record['method']} {record['url'][:120]} ({record['reason']})")
                if report["status"] != "review reached":
                    outcome = 2
    except (GuardError, SnapshotError) as exc:
        print(f"STOPPED: {exc}")
        outcome = 1
    print("verifying the leagues")
    clean = verify_leagues(args.raw, chosen, session if session is not None else load_espn_session())
    return outcome if clean else 1


# --- derive trade -----------------------------------------------------------------------------------------------------

DERIVED_NOTE = (
    "derived, not observed: produced by running ESPN's saved web-client code (proposeTrade, createTransaction, the "
    "model serializer) offline in Node with its send function stubbed (scripts/capture/derive_trade.cjs); no request "
    "was sent or attempted"
)


def derive_trade(webclient: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    """The path and wire body ESPN's own ``proposeTrade`` builds from ``inputs`` (see ``derive_trade.cjs``)."""
    done = subprocess.run(
        ["node", str(DERIVE_JS), str(webclient)],
        input=json.dumps(inputs),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"derive_trade.cjs failed: {done.stderr.strip()}")
    result = json.loads(done.stdout)
    return {"path": result["path"], "body": json.loads(result["data"])}


def cmd_derive_trade(args: argparse.Namespace) -> int:
    """The ``TRADE_PROPOSAL`` request for each league's ``trade-review`` selection, derived from the saved client code
    (never sent), into ``writes_derived/<game>_TRADE_PROPOSAL.json`` for ``fixtures``."""
    config = load_config()
    webclient = args.raw / WEBCLIENT_FILE
    if not webclient.exists():
        webclient = FIXTURES / WEBCLIENT_FILE
    session = load_espn_session()
    code = read_json(webclient)
    for league in leagues(config, args.league):
        folder = args.raw / TRADE_REVIEW / league.game
        if not (folder / "selection.json").exists():
            print(f"{league.key}: no trade-review selection; run `capture.py trade-review` first")
            return 2
        selection = read_json(folder / "selection.json")
        review = read_json(folder / "review.json") if (folder / "review.json").exists() else {}
        expiry = next((select for select in (review.get("review") or {}).get("selects", []) if select["options"]), None)
        days = int(re.sub(r"\D", "", expiry["value"]) or 2) if expiry else 2
        textarea = next(iter((review.get("review") or {}).get("textareas", [])), None)
        inputs = {
            **{key: selection[key] for key in ("game", "seasonId", "leagueId", "latestScoringPeriod")},
            "swid": session.swid if session.swid.startswith("{") else "{" + session.swid + "}",
            "fromTeamId": selection["fromTeamId"],
            "toTeamId": selection["toTeamId"],
            "trade": [{"id": p["id"], "teamId": p["teamId"]} for p in selection["players"]],
            "expirationDate": (datetime.now(UTC) + timedelta(days=days)).isoformat(timespec="milliseconds")[:-6] + "Z",
            "comment": textarea["value"] if textarea else "",
        }
        derived = derive_trade(webclient, inputs)
        versions = {
            match.group(1)
            for read in review.get("review_reads") or []
            if (match := re.search(r"[?&]platformVersion=([0-9a-f]{40})", read["url"]))
        }
        query = f"?platformVersion={versions.pop()}" if len(versions) == 1 else ""
        record = {
            "provenance": DERIVED_NOTE,
            "derived_at": datetime.now(UTC).isoformat(),
            "webClientBuild": code.get("webClientBuild"),
            "bundleSha256": code.get("bundleSha256"),
            "method": "POST",
            "url": f"https://{WRITE_HOST}/apis/v3/{derived['path']}{query}",
            "headers": {
                "accept": "application/json",
                "content-type": "application/json",
                "x-fantasy-platform": "espn-fantasy-web",
                "x-fantasy-source": "kona",
            },
            "body": derived["body"],
            "inputs": {
                "players": selection["players"],
                "expiry_days": days,
                "expiry_source": "the review step's expiry select" if expiry else "the builder's default (2 Days)",
                "comment_source": "the review step's textarea" if textarea else "empty",
            },
            "sources": {
                "body": "service.proposeTrade -> createTransaction -> model get(), executed",
                "path": "the `post` excerpt (games/{game}/seasons/{season}/segments/0/leagues/{id}/transactions/)",
                "host": "the `writeHost` excerpt (lm-api-writes.fantasy.espn.com)",
                "headers": "requestDefaults (Accept, X-Fantasy-Source) and post (Content-Type) excerpts; the "
                "X-Fantasy-Platform value is not in the excerpts and is taken from the observed captures",
                "expirationDate": "an ISO string with milliseconds, as the observed 2026-10-06 ffl abort sent; the "
                "value is now + expiry_days, which is what that abort shows (sent at +48 h for 2 Days)",
                "platformVersion": "the client appends it to every request (requestConfig) but its value is not in "
                "the excerpts; it is the value the same client sent on the review step's own reads"
                if query
                else "not observed on the review step, so the derived URL carries none",
            },
            "unknowns": [
                "the order of the TRADE items is the order of transactionListByAction.trade, which the UI builds; "
                "the observed ffl abort listed our players first",
            ],
        }
        path = args.raw / DERIVED_WRITES / f"{league.game}_TRADE_PROPOSAL.json"
        write_json(path, record)
        print(f"{league.key}: derived {path.name} ({len(derived['body'].get('items', []))} items)")
    return 0


# --- fixtures ---------------------------------------------------------------------------------------------------------


def cmd_fixtures(args: argparse.Namespace) -> int:
    config = load_config()
    index = make_fixtures(args.raw, args.dest, leagues(config, args.league))
    total = sum(entry["bytes"] for entry in index["files"].values())
    print(f"{len(index['files'])} fixtures, {total / 1024:.0f} KiB, under {args.dest}")
    return 0


# --- main -------------------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="unscrubbed capture dir (never the repo)")
    parser.add_argument("--league", action="append", default=[], help="league key from config.toml (repeatable)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshot", help="league state (rest of the matchup) before any write flow")
    sub.add_parser("verify", help="re-read the leagues and compare with the snapshot")
    reads = sub.add_parser("reads", help="every read view plus the probes")
    reads.add_argument("--view", action="append", default=[], help="extra league view to capture (repeatable)")
    webclient = sub.add_parser(
        "webclient", aliases=["calendar"], help="calendars, stat split labels and transaction code from the bundle"
    )
    webclient.add_argument("--settle-ms", type=int, default=8000)
    explore = sub.add_parser("explore", help="one guarded page visit")
    explore.add_argument("url")
    explore.add_argument("steps", nargs="?", help="JSON file of steps")
    explore.add_argument("--name", default="page")
    explore.add_argument("--headed", action="store_true")
    explore.add_argument("--no-page-pacing", action="store_true", help="let the page's own API reads go unpaced")
    explore.add_argument("--settle-ms", type=int, default=8000)
    writes = sub.add_parser("writes", help="assisted write-flow capture in a guarded, headed window")
    writes.add_argument("--minutes", type=float, default=30.0)
    writes.add_argument("--login-minutes", type=float, default=10.0, help="how long to wait for a web sign-in")
    writes.add_argument("--headless", action="store_true", help="for checking the plumbing only")
    writes.add_argument("--settle-ms", type=int, default=10000)
    web_login = sub.add_parser("web-login", help="a sign-in window that stays open (league writes still blocked)")
    web_login.add_argument("--minutes", type=float, default=15.0)
    web_login.add_argument("--settle-ms", type=int, default=10000)
    trade = sub.add_parser("trade-review", help="trade builder up to its review step; never sends an offer")
    trade.add_argument("--partner", type=int, help="the other team (default: the lowest id outside every real offer)")
    trade.add_argument("--headless", action="store_true", help="needs a web sign-in that survives a restart")
    trade.add_argument("--settle-ms", type=int, default=8000)
    sub.add_parser("derive-trade", help="the offer request from ESPN's code, run offline; nothing is sent")
    fixtures = sub.add_parser("fixtures", help="scrub + trim the raw captures into the repo")
    fixtures.add_argument("--dest", type=Path, default=FIXTURES)
    args = parser.parse_args(argv)
    raw = args.raw.resolve()
    if raw == REPO or REPO in raw.parents:
        parser.error("--raw must be outside the repo: raw captures are unscrubbed")
    handlers = {
        "snapshot": cmd_snapshot,
        "verify": cmd_verify,
        "reads": cmd_reads,
        "webclient": cmd_webclient,
        "calendar": cmd_webclient,
        "explore": cmd_explore,
        "writes": cmd_writes,
        "web-login": cmd_web_login,
        "trade-review": cmd_trade_review,
        "derive-trade": cmd_derive_trade,
        "fixtures": cmd_fixtures,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
