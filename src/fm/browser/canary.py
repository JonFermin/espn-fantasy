"""The selector canary (DESIGN section 6.3, ROADMAP #28): a read-only check that ESPN's pages and views still look the
way the executor expects, run daily so drift is found before a real move depends on it.

Two checks per league, both read-only:

- **Selectors.** Every selector in the registry (:func:`fm.browser.selectors.registered_selectors`) is looked up on its
  page, scoped the way the registry says (``within`` a row). What counts as drift follows the selector's
  :class:`fm.browser.selectors.Presence`: an ``ALWAYS`` selector that resolves nowhere is drift, a ``NEVER`` marker
  (the "Log in Required" heading) that shows is drift, and a ``SOMETIMES`` selector (an open slot, MOVE on an unlocked
  player) is only observed, never alerted on. A selector that needs a click to appear (``after``) is skipped: the
  canary loads pages and counts locators, and never clicks, so nothing it does can save anything. A page the canary
  has no address for (:data:`PAGE_ADDRESSES`) is a finding of its own, so a new page cannot go unwatched.
- **Read views.** Each view the executor and the planners depend on is read through :class:`fm.espn.client.EspnClient`
  and parsed by ``fm.espn.models`` / ``fm.espn.settings``. A schema failure is drift; an HTTP or auth failure is not
  drift, but it means the view could not be checked, so it is reported as a failure.

The result is a :class:`CanaryReport` per league. :func:`drift_alert` turns reports into one
``fm.notify.Message`` for ``fm.notify.send_alert``, and :func:`drift_payload` into the structured payload. Nothing
here sends: ``fm canary`` (``fm.commands.canary``) does, and only on a live run. Selectors live in
``fm.browser.selectors`` alone; this module iterates that registry and spells none out.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fm.browser import selectors
from fm.browser.flows import LocatorLike, PageLike
from fm.browser.selectors import Presence, Selector, WebPage
from fm.config import League
from fm.espn.client import PENDING_OFFER_TYPES, EspnClient, EspnClientError, EspnRead, EspnSchemaError
from fm.espn.models import LeagueStatus, MatchupsView
from fm.notify.base import Message
from fm.notify.messages import alert

PAGE_LOAD_TIMEOUT_MS = 30_000.0
"""How long loading a page may take."""
ANCHOR_TIMEOUT_MS = 10_000.0
"""How long to wait for a page's first always-present selector before judging the page by what is there."""
FREE_AGENT_PROBE = 10
"""Players the free-agent view is asked for: enough to parse, little to pull."""
ALERT_LINES = 12
"""Findings an alert lists before it says how many more there are; the structured payload keeps all of them."""


class FindingKind(StrEnum):
    """What the canary found wrong. The first three are drift: ESPN changed. The rest mean it could not check."""

    MISSING = "missing"  # an ALWAYS selector resolves to nothing on a loaded page
    WARNING = "warning"  # a NEVER marker (Log in Required) shows
    UNPARSEABLE = "unparseable"  # a read view no longer parses
    NO_ADDRESS = "no_address"  # a registered page the canary does not know how to open
    PAGE_FAILED = "page_failed"  # a page did not load
    VIEW_FAILED = "view_failed"  # a read view could not be read (HTTP, auth)
    LOCATOR_ERROR = "locator_error"  # a locator raised instead of counting

    @property
    def is_drift(self) -> bool:
        return self in DRIFT_KINDS


DRIFT_KINDS = frozenset({FindingKind.MISSING, FindingKind.WARNING, FindingKind.UNPARSEABLE})

VIEW_EXPECTED = "parses"
"""The ``expected`` of a finding about a read view."""


@dataclass(frozen=True, slots=True)
class CanaryFinding:
    """One thing wrong: what was checked, where, what was expected and what happened."""

    kind: FindingKind
    subject: str
    """A selector key (``roster.table``), a view (``mRoster``) or a page (``roster``)."""
    expected: str
    """A :class:`Presence` value for a selector, :data:`VIEW_EXPECTED` for a view."""
    detail: str
    page: WebPage | None = None
    """The page of a selector finding; ``None`` for a view."""
    url: str | None = None
    """The address the page was opened at, when there was one."""

    @property
    def is_drift(self) -> bool:
        return self.kind.is_drift

    def line(self) -> str:
        where = f"[{self.page.value}] " if self.page is not None else "[view] "
        return f"{where}{self.subject} ({self.expected}): {self.kind.value}: {self.detail}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "drift": self.is_drift,
            "subject": self.subject,
            "page": None if self.page is None else self.page.value,
            "expected": self.expected,
            "detail": self.detail,
            "url": self.url,
        }


@dataclass(frozen=True, slots=True)
class Skipped:
    """A selector the canary did not judge, and why."""

    key: str
    reason: str


@dataclass(frozen=True, slots=True)
class SelectorCheck:
    """What the selector pass saw."""

    findings: tuple[CanaryFinding, ...] = ()
    resolved: tuple[str, ...] = ()
    """Keys that resolved (an ALWAYS selector that did, a SOMETIMES one that happened to be there)."""
    absent_sometimes: tuple[str, ...] = ()
    """SOMETIMES selectors that were not on the page: expected, never alerted."""
    skipped: tuple[Skipped, ...] = ()
    clear: tuple[str, ...] = ()
    """NEVER markers that were not on the page, as a healthy page has it."""


@dataclass(frozen=True, slots=True)
class ViewCheck:
    """What the view pass saw."""

    findings: tuple[CanaryFinding, ...] = ()
    parsed: tuple[str, ...] = ()
    skipped: tuple[Skipped, ...] = ()


@dataclass(frozen=True, slots=True)
class CanaryReport:
    """One league's canary run."""

    league: str
    """The league's config key."""
    selector_check: SelectorCheck | None
    """``None`` when no browser was used (a fixture run)."""
    view_check: ViewCheck | None
    """``None`` when the views were not read."""
    findings: tuple[CanaryFinding, ...] = field(default=())

    @property
    def drift(self) -> tuple[CanaryFinding, ...]:
        """The findings that mean ESPN changed."""
        return tuple(finding for finding in self.findings if finding.is_drift)

    @property
    def failures(self) -> tuple[CanaryFinding, ...]:
        """The findings that mean something could not be checked."""
        return tuple(finding for finding in self.findings if not finding.is_drift)

    @property
    def ok(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict[str, Any]:
        sel, views = self.selector_check, self.view_check
        return {
            "league": self.league,
            "ok": self.ok,
            "findings": [finding.as_dict() for finding in self.findings],
            "selectors": None
            if sel is None
            else {
                "resolved": list(sel.resolved),
                "absent_sometimes": list(sel.absent_sometimes),
                "clear": list(sel.clear),
                "skipped": [{"key": item.key, "reason": item.reason} for item in sel.skipped],
            },
            "views": None
            if views is None
            else {
                "parsed": list(views.parsed),
                "skipped": [{"key": item.key, "reason": item.reason} for item in views.skipped],
            },
        }


# --- pages ------------------------------------------------------------------------------------------------------------

type PageAddress = Callable[[League], str]
"""Where the canary opens a page for a league."""


def _team_page(league: League) -> str:
    return selectors.team_page_url(league.game, league.espn_league_id, league.team_id, league.season)


PAGE_ADDRESSES: Mapping[WebPage, PageAddress] = {WebPage.ROSTER: _team_page}
"""How to open each registered page. A page in the registry without an entry is reported as :attr:`FindingKind.
NO_ADDRESS`, and ``tests/browser/test_canary.py`` fails until it has one. The address builders themselves belong in
``fm.browser.selectors``; add a page there, then a line here."""


def check_selectors(
    page: PageLike,
    league: League,
    *,
    registry: Sequence[Selector] | None = None,
    addresses: Mapping[WebPage, PageAddress] | None = None,
) -> SelectorCheck:
    """Open each registered page for ``league`` and judge every selector by its :class:`Presence`.

    Loads pages and counts locators; clicks nothing, so a selector that appears only after a click (``after``) is
    skipped. ``registry`` and ``addresses`` default to the real ones; tests pass their own.
    """
    chosen = tuple(selectors.registered_selectors() if registry is None else registry)
    known = addresses if addresses is not None else PAGE_ADDRESSES
    by_key = {item.key: item for item in chosen}
    findings: list[CanaryFinding] = []
    resolved: list[str] = []
    absent: list[str] = []
    skipped: list[Skipped] = []
    clear: list[str] = []
    pages = list(dict.fromkeys(item.page for item in chosen))
    for web_page in pages:
        batch = [item for item in chosen if item.page is web_page]
        address = known.get(web_page)
        if address is None:
            findings.append(
                CanaryFinding(
                    FindingKind.NO_ADDRESS,
                    web_page.value,
                    Presence.ALWAYS.value,
                    f"the canary has no address for the {web_page.value} page, so its {len(batch)} selectors were "
                    "not checked (add it to fm.browser.canary.PAGE_ADDRESSES)",
                    page=web_page,
                )
            )
            skipped.extend(Skipped(item.key, "no address for the page") for item in batch)
            continue
        url = address(league)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT_MS)
        except Exception as exc:
            findings.append(
                CanaryFinding(
                    FindingKind.PAGE_FAILED,
                    web_page.value,
                    Presence.ALWAYS.value,
                    f"the {web_page.value} page did not load: {_short(exc)}",
                    page=web_page,
                    url=url,
                )
            )
            skipped.extend(Skipped(item.key, "the page did not load") for item in batch)
            continue
        part = _check_page(page, web_page, url, batch, by_key)
        findings.extend(part.findings)
        resolved.extend(part.resolved)
        absent.extend(part.absent_sometimes)
        skipped.extend(part.skipped)
        clear.extend(part.clear)
    return SelectorCheck(tuple(findings), tuple(resolved), tuple(absent), tuple(skipped), tuple(clear))


def _check_page(
    page: PageLike, web_page: WebPage, url: str, batch: Sequence[Selector], by_key: Mapping[str, Selector]
) -> SelectorCheck:
    """Judge one loaded page's selectors, in registration order."""
    _wait_for_anchor(page, batch)
    findings: list[CanaryFinding] = []
    resolved: list[str] = []
    absent: list[str] = []
    skipped: list[Skipped] = []
    clear: list[str] = []
    warned = False
    for item in batch:
        if item.presence is not Presence.NEVER:
            continue
        count, error = _count(page, item, by_key)
        if error is not None:
            findings.append(_locator_error(item, url, error))
        elif count:
            warned = True
            findings.append(
                CanaryFinding(
                    FindingKind.WARNING,
                    item.key,
                    item.presence.value,
                    f"{count} match(es) for {item.description}; a healthy page does not show it",
                    page=web_page,
                    url=url,
                )
            )
        else:
            clear.append(item.key)
    broken: set[str] = set()
    for item in batch:
        if item.presence is Presence.NEVER:
            continue
        if item.after is not None:
            skipped.append(Skipped(item.key, f"appears after a click on {item.after}; the canary never clicks"))
            continue
        if warned:
            skipped.append(Skipped(item.key, "the page shows a warning marker, so it is not the page to judge"))
            continue
        blocked = _blocking_parent(item, by_key, broken)
        if blocked is not None:
            skipped.append(Skipped(item.key, f"its container {blocked} is not on the page"))
            continue
        count, error = _count(page, item, by_key)
        if error is not None:
            broken.add(item.key)
            findings.append(_locator_error(item, url, error))
        elif count:
            resolved.append(item.key)
        elif item.presence is Presence.ALWAYS:
            broken.add(item.key)
            findings.append(
                CanaryFinding(
                    FindingKind.MISSING,
                    item.key,
                    item.presence.value,
                    f"resolves to nothing on {url}: {item.description}",
                    page=web_page,
                    url=url,
                )
            )
        else:
            broken.add(item.key)  # a SOMETIMES container that is absent leaves nothing inside it to judge
            absent.append(item.key)
    return SelectorCheck(tuple(findings), tuple(resolved), tuple(absent), tuple(skipped), tuple(clear))


def _wait_for_anchor(page: PageLike, batch: Sequence[Selector]) -> None:
    """Give a client-rendered page time to draw: wait for its first always-present, unscoped selector. A page that
    never shows it is judged by what is there, so the wait's failure is not a finding."""
    for item in batch:
        if item.presence is Presence.ALWAYS and item.within is None and item.after is None:
            try:
                item.locate(page).first.wait_for(state="visible", timeout=ANCHOR_TIMEOUT_MS)
            except Exception:
                return
            return


def _locator(page: PageLike, item: Selector, by_key: Mapping[str, Selector]) -> LocatorLike:
    """``item`` looked up inside its container chain (``within``), as a flow would."""
    if item.within is None:
        return item.locate(page)
    return item.locate(_locator(page, by_key[item.within], by_key))


def _count(page: PageLike, item: Selector, by_key: Mapping[str, Selector]) -> tuple[int, Exception | None]:
    try:
        return _locator(page, item, by_key).count(), None
    except Exception as exc:
        return 0, exc


def _blocking_parent(item: Selector, by_key: Mapping[str, Selector], broken: set[str]) -> str | None:
    """The container (``within`` chain) already found missing or absent: ``item`` has nothing to be found in."""
    parent = item.within
    while parent is not None:
        if parent in broken:
            return parent
        parent = by_key[parent].within
    return None


def _locator_error(item: Selector, url: str, error: Exception) -> CanaryFinding:
    return CanaryFinding(
        FindingKind.LOCATOR_ERROR,
        item.key,
        item.presence.value,
        f"the locator raised instead of counting: {_short(error)}",
        page=item.page,
        url=url,
    )


# --- read views -------------------------------------------------------------------------------------------------------


def check_views(reader: EspnClient) -> ViewCheck:
    """Read every view through ``reader`` (read-only by construction: the client has no write method) and parse it.

    Scoreboard needs a matchup period; it comes from the roster read, else the first of the season's matchup periods.
    """
    findings: list[CanaryFinding] = []
    parsed: list[str] = []
    skipped: list[Skipped] = []

    def attempt[T](view: str, read: Callable[[], T]) -> T | None:
        try:
            result = read()
        except EspnSchemaError as exc:
            findings.append(CanaryFinding(FindingKind.UNPARSEABLE, view, VIEW_EXPECTED, _short(exc)))
            return None
        except EspnClientError as exc:
            findings.append(CanaryFinding(FindingKind.VIEW_FAILED, view, VIEW_EXPECTED, _short(exc)))
            return None
        parsed.append(view)
        return result

    attempt("mSettings", reader.settings)
    attempt("mTeam+mStandings", reader.teams)
    rosters = attempt("mRoster", reader.rosters)
    matchups = attempt("mMatchup", reader.matchups)
    period = _matchup_period(rosters.data.status if rosters is not None else None, matchups)
    if period is None:
        skipped.append(Skipped("mMatchupScore+mScoreboard", "no matchup period to ask for"))
    else:
        attempt("mMatchupScore+mScoreboard", lambda: reader.scoreboard(period))
    attempt("kona_player_info", lambda: reader.free_agents(limit=FREE_AGENT_PROBE))
    attempt("mPendingTransactions", reader.pending_transactions)
    attempt("mTransactions2", lambda: reader.transactions(types=PENDING_OFFER_TYPES))
    attempt("proTeamSchedules_wl", reader.pro_schedule)
    return ViewCheck(tuple(findings), tuple(parsed), tuple(skipped))


def _matchup_period(status: LeagueStatus | None, matchups: EspnRead[MatchupsView] | None) -> int | None:
    if status is not None and status.current_matchup_period:
        return status.current_matchup_period
    if matchups is not None:
        periods = matchups.data.matchup_periods
        if periods:
            return periods[0]
    return None


# --- the run and its report -------------------------------------------------------------------------------------------


def run_canary(
    league: League,
    *,
    reader: EspnClient | None,
    page: PageLike | None,
    registry: Sequence[Selector] | None = None,
    addresses: Mapping[WebPage, PageAddress] | None = None,
) -> CanaryReport:
    """Check ``league``'s pages (with ``page``) and read views (with ``reader``); either may be left out."""
    selector_check = None if page is None else check_selectors(page, league, registry=registry, addresses=addresses)
    view_check = None if reader is None else check_views(reader)
    findings = (selector_check.findings if selector_check else ()) + (view_check.findings if view_check else ())
    return CanaryReport(league.key, selector_check, view_check, findings)


def drift_payload(reports: Sequence[CanaryReport]) -> dict[str, Any]:
    """The structured drift report: per league, every finding (which selector or view, its page, what was expected,
    what happened) plus what was checked. JSON-able."""
    return {
        "ok": all(report.ok for report in reports),
        "drift": any(report.drift for report in reports),
        "leagues": [report.as_dict() for report in reports],
    }


def drift_alert(reports: Sequence[CanaryReport]) -> Message | None:
    """The alert for ``fm.notify.send_alert``'s message (``None`` when every report is clean).

    Drift leads the title; findings that only mean a check could not run follow in the body. The link opens the first
    page involved, where the fix starts.
    """
    bad = [report for report in reports if not report.ok]
    if not bad:
        return None
    drift = sum(len(report.drift) for report in bad)
    failed = sum(len(report.failures) for report in bad)
    if drift:
        title = f"ESPN drift: {drift} finding{'s' if drift != 1 else ''}"
        title += f" and {failed} unchecked" if failed else ""
    else:
        title = f"ESPN canary could not check {failed} thing{'s' if failed != 1 else ''}"
    lines: list[str] = []
    for report in bad:
        lines.extend(f"{report.league}: {finding.line()}" for finding in (*report.drift, *report.failures))
    if len(lines) > ALERT_LINES:
        lines = [*lines[:ALERT_LINES], f"... and {len(lines) - ALERT_LINES} more (fm canary lists them all)"]
    link = next((finding.url for report in bad for finding in report.findings if finding.url), None)
    return alert(title, "\n".join(lines), link=link)


def _short(error: BaseException, limit: int = 300) -> str:
    text = " ".join(str(error).split()) or type(error).__name__
    return text if len(text) <= limit else text[: limit - 3] + "..."
