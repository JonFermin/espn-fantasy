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
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fixture_writer import WEBCLIENT_FILE, make_fixtures
from guard import FANTASY_DOMAIN, WRITE_HOST, BlockedRequest, GuardError, WriteGuard
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
                    "shows an error): a bench swap, a free-agent add/drop, a waiver claim, and a trade proposal, in "
                    "each league. Never touch a real pending offer. Close the window when done."
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
        "fixtures": cmd_fixtures,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
