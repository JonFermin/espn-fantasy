"""Test doubles for the executor and its flows: a fake ESPN read API, write transport, page and browser (ROADMAP #19).

Nothing here opens a socket or starts a browser, so executor and flow tests run offline (CLAUDE.md):

- :class:`FakeEspnApi` serves ESPN read views from in-memory JSON through an ``httpx.MockTransport``, so the real
  :class:`fm.espn.client.EspnClient` parses them. Bodies are plain dicts that a test, or a fake write landing, mutates
  to change what the next read sees.
- :class:`FakeTransport` stands in for the browser-session write transport: it records every request the executor
  sends and answers from a script of :class:`Reply` values (a 2xx, an ESPN error, a 5xx, a timeout). ``on_send`` applies
  a write that lands to the fake league, so verification can see it.
- :class:`FakePage`, :class:`FakeElement` and :class:`FakeLocator` model a page as a tree of elements with roles,
  accessible names, text, labels and CSS selectors, and answer the locators flows use
  (``fm.browser.flows.PageLike``) with Playwright's semantics where they matter: strict mode, a missing element is a
  ``playwright`` ``TimeoutError``, hidden or disabled elements are not actionable. ``on_click`` handlers script what a
  click does (open a dialog, save a move); every action lands in ``FakePage.actions``.
- :class:`FakeBrowser` hands out pages and writes stand-in traces; :func:`fake_runtime` and :class:`FakeOpener` wire
  it all into an ``fm.executor.Runtime`` for ``fm.executor.execute(..., opener=...)``.

Production code never imports this module.
"""

from __future__ import annotations

import base64
import json
import zipfile
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from re import Pattern
from typing import Any

import httpx
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from fm.browser.flows import WriteRequest, WriteResponse, WriteTimeoutError
from fm.espn.auth import EspnSession
from fm.espn.client import READS_HOST, EspnClient
from fm.espn.ids import Game
from fm.executor.runtime import Runtime
from fm.store import LeagueRow

FAKE_SWID = "{00000000-0000-0000-0000-000000000000}"
"""The placeholder account id fakes put in ``memberId``, like the scrubbed fixtures."""
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
"""What a fake screenshot writes: a valid 1x1 PNG."""


def view_key(request: httpx.Request) -> str:
    """``mTeam+mStandings``: the ``view=`` params of a read, joined, as :class:`FakeEspnApi` keys its bodies."""
    return "+".join(request.url.params.get_list("view"))


# --- reads ------------------------------------------------------------------------------------------------------------


class FakeEspnApi:
    """ESPN's read host (``lm-api-reads``) in memory. Answers GETs by view; anything else fails the test."""

    def __init__(self, views: Mapping[str, Any] | None = None) -> None:
        self.views: dict[str, Any] = dict(views or {})
        self.requests: list[httpx.Request] = []
        self._failures: list[tuple[str | None, httpx.Response]] = []

    def serve(self, view: str, body: Any) -> Any:
        """Answer ``view`` (``"mRoster"``, ``"mTeam+mStandings"``) with ``body``: JSON-able data, an
        ``httpx.Response``, or a callable taking the ``httpx.Request`` and returning either. Returns ``body``."""
        self.views[view] = body
        return body

    def serve_file(self, view: str, path: Path) -> Any:
        """Serve the JSON in ``path`` (a fresh copy a test may mutate) and return it."""
        return self.serve(view, json.loads(path.read_text(encoding="utf-8")))

    def fail(self, status: int, *, view: str | None = None, body: Any = None, times: int = 1) -> None:
        """Answer the next ``times`` reads of ``view`` (any view when ``None``) with HTTP ``status``."""
        payload = body if body is not None else {"messages": [f"fake HTTP {status}"], "details": []}
        self._failures.extend((view, httpx.Response(status, json=payload)) for _ in range(times))

    def reads(self, view: str | None = None) -> int:
        """How many reads arrived, of ``view`` or of anything."""
        return sum(1 for request in self.requests if view is None or view_key(request) == view)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method != "GET" or request.url.host != READS_HOST:
            raise AssertionError(
                f"the fake read API answers GETs to {READS_HOST} only, got {request.method} {request.url}"
            )
        self.requests.append(request)
        key = view_key(request)
        for index, (view, response) in enumerate(self._failures):
            if view is None or view == key:
                del self._failures[index]
                return response
        if key not in self.views:
            raise AssertionError(f"no fake body for view {key!r}; serving {sorted(self.views)}")
        body = self.views[key]
        if callable(body):
            body = body(request)
        if isinstance(body, httpx.Response):
            return body
        return httpx.Response(200, json=body)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(
        self, game: Game | str, league_id: int, season: int, *, session: EspnSession | None = None
    ) -> EspnClient:
        """A real :class:`EspnClient` over this fake: no capture, no pacing, one attempt per read, no sleeping."""
        return EspnClient(
            game,
            league_id,
            season,
            session,
            client=httpx.Client(transport=self.transport()),
            capture=False,
            min_interval_s=0.0,
            max_attempts=1,
            sleep=lambda _seconds: None,
        )

    def client_for(self, league: LeagueRow) -> EspnClient:
        return self.client(Game.from_sport(league.sport), league.espn_league_id, league.season)


# --- writes -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reply:
    """One scripted answer of :class:`FakeTransport`."""

    status: int = 200
    body: Any = None
    """The JSON body; a 2xx without one answers ``{"id": "fake-transaction-N"}``."""
    timeout: bool = False
    """Raise ``WriteTimeoutError`` instead of answering."""
    error: Exception | None = None
    """Raise this instead of answering (``WriteUncertainError``, ``WriteRefusedError``, ...)."""
    applies: bool | None = None
    """Whether the write lands (``FakeTransport.on_send`` runs). Default: only for a 2xx answer."""

    @property
    def lands(self) -> bool:
        if self.applies is not None:
            return self.applies
        return not self.timeout and self.error is None and 200 <= self.status < 300

    @classmethod
    def espn_error(cls, status: int, code: str, message: str = "") -> Reply:
        """ESPN's error shape: ``{"messages": [...], "details": [{"type": code, "message": ...}]}``."""
        text = message or code
        return cls(
            status=status,
            body={"messages": [text], "details": [{"type": code, "message": text, "shortMessage": text}]},
        )

    @classmethod
    def timed_out(cls, *, applies: bool = False) -> Reply:
        """No answer in time; ``applies=True`` when ESPN applied the write anyway."""
        return cls(timeout=True, applies=applies)


class FakeTransport:
    """``fm.browser.flows.WriteTransport`` that records each request and answers from a script; never sends anything.

    Replies are used in order, then ``default`` (a plain 200) for every later send.
    """

    def __init__(
        self,
        *replies: Reply,
        on_send: Callable[[WriteRequest], object] | None = None,
        default: Reply | None = None,
    ) -> None:
        self.replies: deque[Reply] = deque(replies)
        self.on_send = on_send
        self.default = default if default is not None else Reply()
        self.sent: list[WriteRequest] = []
        self.timeouts: list[float] = []

    def send(self, request: WriteRequest, *, timeout_s: float) -> WriteResponse:
        self.sent.append(request)
        self.timeouts.append(timeout_s)
        reply = self.replies.popleft() if self.replies else self.default
        if reply.lands and self.on_send is not None:
            self.on_send(request)
        if reply.timeout:
            raise WriteTimeoutError(f"no answer within {timeout_s:g} s (fake)")
        if reply.error is not None:
            raise reply.error
        body = reply.body
        if body is None and 200 <= reply.status < 300:
            body = {"id": f"fake-transaction-{len(self.sent)}"}
        return WriteResponse(status=reply.status, body=body, text="" if body is None else json.dumps(body))


# --- pages ------------------------------------------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return " ".join(text.split())


def text_matches(actual: str | None, expected: str | Pattern[str], exact: bool | None = None) -> bool:
    """Playwright's text matching: a regex searches; a string is a case-insensitive substring, or with ``exact`` a
    case-sensitive whole match, whitespace normalized either way."""
    if actual is None:
        return False
    if isinstance(expected, Pattern):
        return expected.search(actual) is not None
    have, want = _normalize(actual), _normalize(expected)
    return have == want if exact else want.casefold() in have.casefold()


@dataclass(eq=False)
class FakeElement:
    """One element of a fake page. Elements compare by identity; build trees with ``children`` or :meth:`add`."""

    role: str | None = None
    """ARIA role for ``get_by_role``: ``button``, ``row``, ``option``, ``dialog``, ..."""
    name: str | None = None
    """Accessible name; ``get_by_role(name=...)`` falls back to ``label``, then ``text``."""
    text: str | None = None
    label: str | None = None
    test_id: str | None = None
    selectors: tuple[str, ...] = ()
    """CSS selectors this element answers in ``locator(...)`` (the ones ``fm.browser.selectors`` registers)."""
    visible: bool = True
    enabled: bool = True
    checked: bool = False
    value: str = ""
    options: tuple[str, ...] = ()
    """Values a ``select_option`` may pick."""
    children: list[FakeElement] = field(default_factory=list)
    on_click: Callable[[FakePage], object] | None = None
    """What a click does, given the page: show a dialog, apply a move to a :class:`FakeEspnApi` body."""
    error: Exception | None = None
    """Raised by every action on the element, e.g. a ``playwright`` ``TimeoutError`` for a click that hangs."""
    clicks: int = 0

    def add(self, *children: FakeElement) -> FakeElement:
        self.children.extend(children)
        return self

    def walk(self) -> Iterator[FakeElement]:
        """This element and every descendant, depth first."""
        yield self
        for child in self.children:
            yield from child.walk()

    def descendants(self) -> Iterator[FakeElement]:
        for child in self.children:
            yield from child.walk()

    @property
    def accessible_name(self) -> str | None:
        return self.name or self.label or self.text

    @property
    def content(self) -> str:
        """The element's text with its descendants', as ``inner_text`` and ``filter(has_text=...)`` read it."""
        own = self.text or self.name or ""
        parts = [own, *(child.content for child in self.children)]
        return " ".join(part for part in parts if part)


type _Pick = Callable[[list[FakeElement]], list[FakeElement]]


def _unique(elements: Iterable[FakeElement]) -> list[FakeElement]:
    found: list[FakeElement] = []
    for element in elements:
        if not any(element is seen for seen in found):
            found.append(element)
    return found


def _role(role: str, name: str | Pattern[str] | None, exact: bool | None) -> Callable[[FakeElement], bool]:
    def match(element: FakeElement) -> bool:
        if element.role != role or not element.visible:
            return False
        return name is None or text_matches(element.accessible_name, name, exact)

    return match


def _test_id(test_id: str | Pattern[str]) -> Callable[[FakeElement], bool]:
    def match(element: FakeElement) -> bool:
        if element.test_id is None:
            return False
        return (
            test_id.search(element.test_id) is not None if isinstance(test_id, Pattern) else element.test_id == test_id
        )

    return match


class FakeLocator:
    """A lazy query over a :class:`FakePage`, re-run at every action like a Playwright locator."""

    def __init__(self, page: FakePage, resolve: Callable[[], list[FakeElement]], description: str) -> None:
        self._page = page
        self._resolve = resolve
        self.description = description

    def __repr__(self) -> str:
        return f"FakeLocator({self.description})"

    # --- building ---

    def _derive(self, pick: _Pick, description: str) -> FakeLocator:
        return FakeLocator(self._page, lambda: pick(self._resolve()), f"{self.description} >> {description}")

    def _inside(self, match: Callable[[FakeElement], bool], description: str) -> FakeLocator:
        def pick(found: list[FakeElement]) -> list[FakeElement]:
            return _unique(element for root in found for element in root.descendants() if match(element))

        return self._derive(pick, description)

    @property
    def first(self) -> FakeLocator:
        return self._derive(lambda found: found[:1], "first")

    @property
    def last(self) -> FakeLocator:
        return self._derive(lambda found: found[-1:], "last")

    def nth(self, index: int) -> FakeLocator:
        def pick(found: list[FakeElement]) -> list[FakeElement]:
            return [found[index]] if -len(found) <= index < len(found) else []

        return self._derive(pick, f"nth={index}")

    def filter(self, *, has_text: str | Pattern[str] | None = None) -> FakeLocator:
        if has_text is None:
            return self
        return self._derive(
            lambda found: [element for element in found if text_matches(element.content, has_text)],
            f"has_text={has_text!r}",
        )

    def all(self) -> list[FakeLocator]:
        return [self.nth(index) for index in range(self.count())]

    def get_by_role(
        self, role: str, *, name: str | Pattern[str] | None = None, exact: bool | None = None
    ) -> FakeLocator:
        return self._inside(_role(role, name, exact), f"role={role} name={name!r}")

    def get_by_text(self, text: str | Pattern[str], *, exact: bool | None = None) -> FakeLocator:
        return self._inside(lambda element: text_matches(element.text, text, exact), f"text={text!r}")

    def get_by_label(self, text: str | Pattern[str], *, exact: bool | None = None) -> FakeLocator:
        return self._inside(lambda element: text_matches(element.label, text, exact), f"label={text!r}")

    def locator(self, selector: str, /) -> FakeLocator:
        return self._inside(lambda element: selector in element.selectors, selector)

    # --- resolving ---

    def _missing(self, what: str) -> PlaywrightTimeoutError:
        return PlaywrightTimeoutError(
            f"Timeout {self._page.timeout_ms:g}ms exceeded: waiting for {self.description} ({what})"
        )

    def _found(self) -> list[FakeElement]:
        """The current matches; like Playwright, any use of a locator on a closed page raises."""
        self._page.ensure_open()
        return self._resolve()

    def _one(self, what: str) -> FakeElement:
        found = self._found()
        if not found:
            raise self._missing(what)
        if len(found) > 1:
            raise PlaywrightError(f"strict mode violation: {self.description} resolved to {len(found)} elements")
        return found[0]

    def _actionable(self, what: str) -> FakeElement:
        element = self._one(what)
        if element.error is not None:
            self._page.log(f"{what} failed", self.description)
            raise element.error
        if not element.visible or not element.enabled:
            raise self._missing(f"{what}: element is {'hidden' if not element.visible else 'disabled'}")
        return element

    # --- actions ---

    def click(self, *, timeout: float | None = None) -> None:
        element = self._actionable("click")
        element.clicks += 1
        self._page.log("click", self.description)
        if element.on_click is not None:
            element.on_click(self._page)

    def fill(self, value: str, *, timeout: float | None = None) -> None:
        element = self._actionable("fill")
        element.value = value
        self._page.log("fill", f"{self.description} = {value!r}")

    def check(self, *, timeout: float | None = None) -> None:
        self._actionable("check").checked = True
        self._page.log("check", self.description)

    def uncheck(self, *, timeout: float | None = None) -> None:
        self._actionable("uncheck").checked = False
        self._page.log("uncheck", self.description)

    def select_option(
        self,
        value: str | Sequence[str] | None = None,
        *,
        label: str | Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> list[str]:
        element = self._actionable("select_option")
        wanted = value if value is not None else label
        choices = [wanted] if isinstance(wanted, str) else list(wanted or ())
        if not choices or any(choice not in element.options for choice in choices):
            raise self._missing(f"select_option: {choices} not among {list(element.options)}")
        element.value = choices[0]
        self._page.log("select", f"{self.description} = {choices[0]!r}")
        return choices

    def press(self, key: str, *, timeout: float | None = None) -> None:
        self._actionable("press")
        self._page.log("press", f"{self.description} {key}")

    # --- state ---

    def count(self) -> int:
        return len(self._found())

    def is_visible(self) -> bool:
        found = self._found()
        if len(found) > 1:
            raise PlaywrightError(f"strict mode violation: {self.description} resolved to {len(found)} elements")
        return bool(found) and found[0].visible

    def is_enabled(self) -> bool:
        return self._one("is_enabled").enabled

    def is_checked(self) -> bool:
        return self._one("is_checked").checked

    def inner_text(self, *, timeout: float | None = None) -> str:
        return self._one("inner_text").content

    def text_content(self, *, timeout: float | None = None) -> str | None:
        return self._one("text_content").content

    def wait_for(self, *, state: str | None = None, timeout: float | None = None) -> None:
        wanted = state or "visible"
        found = self._found()
        if len(found) > 1:
            raise PlaywrightError(f"strict mode violation: {self.description} resolved to {len(found)} elements")
        element = found[0] if found else None
        reached = {
            "attached": element is not None,
            "detached": element is None,
            "visible": element is not None and element.visible,
            "hidden": element is None or not element.visible,
        }.get(wanted)
        if reached is None:
            raise ValueError(f"unknown wait state {wanted!r}")
        if not reached:
            raise self._missing(f"to be {wanted}")


type Screen = Sequence[FakeElement] | Callable[[], Sequence[FakeElement]]
"""What a URL shows: fixed elements, or a function rendering fresh ones on every visit (form state starts over)."""


class FakePage:
    """A page as a list of top-level :class:`FakeElement` trees. ``goto`` swaps in ``screens[url]`` when there is one;
    ``on_click`` handlers call :meth:`show` / :meth:`add` to change what is on screen."""

    def __init__(
        self,
        *elements: FakeElement,
        url: str = "about:blank",
        title: str = "",
        screens: Mapping[str, Screen] | None = None,
    ) -> None:
        self._url = url
        self._title = title
        self.elements: list[FakeElement] = list(elements)
        self.screens: dict[str, Screen] = dict(screens or {})
        self.goto_errors: dict[str, Exception] = {}
        """Navigations that fail, by URL."""
        self.actions: list[tuple[str, str]] = []
        """Everything done to the page, in order: ``("goto", url)``, ``("click", locator)``, ``("fill", ...)``."""
        self.screenshots: list[Path] = []
        self.timeout_ms = 30_000.0
        self.navigation_timeout_ms = 30_000.0
        self.closed = False

    # --- the page surface flows use ---

    @property
    def url(self) -> str:
        return self._url

    def goto(self, url: str, *, timeout: float | None = None, wait_until: str | None = None) -> None:
        self.log("goto", url)
        error = self.goto_errors.get(url)
        if error is not None:
            raise error
        self._url = url
        screen = self.screens.get(url)
        if screen is not None:
            self.elements = list(screen() if callable(screen) else screen)

    def get_by_role(
        self, role: str, *, name: str | Pattern[str] | None = None, exact: bool | None = None
    ) -> FakeLocator:
        return self._search(_role(role, name, exact), f"role={role} name={name!r}")

    def get_by_text(self, text: str | Pattern[str], *, exact: bool | None = None) -> FakeLocator:
        return self._search(lambda element: text_matches(element.text, text, exact), f"text={text!r}")

    def get_by_label(self, text: str | Pattern[str], *, exact: bool | None = None) -> FakeLocator:
        return self._search(lambda element: text_matches(element.label, text, exact), f"label={text!r}")

    def get_by_test_id(self, test_id: str | Pattern[str]) -> FakeLocator:
        return self._search(_test_id(test_id), f"test_id={test_id!r}")

    def locator(self, selector: str, /) -> FakeLocator:
        return self._search(lambda element: selector in element.selectors, selector)

    def wait_for_load_state(self, state: str | None = None, *, timeout: float | None = None) -> None:
        self.log("wait_for_load_state", state or "load")

    def screenshot(self, *, path: str | Path | None = None, full_page: bool | None = None) -> bytes:
        self.log("screenshot", "" if path is None else str(path))
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(PNG_1X1)
            self.screenshots.append(target)
        return PNG_1X1

    def set_default_timeout(self, timeout: float) -> None:
        self.timeout_ms = float(timeout)

    def set_default_navigation_timeout(self, timeout: float) -> None:
        self.navigation_timeout_ms = float(timeout)

    def title(self) -> str:
        return self._title

    def close(self) -> None:
        self.closed = True

    # --- for tests and on_click handlers ---

    def show(self, *elements: FakeElement, url: str | None = None) -> None:
        """Replace what is on screen (a new view after a click); optionally change the URL too."""
        self.elements = list(elements)
        if url is not None:
            self._url = url

    def add(self, *elements: FakeElement) -> None:
        """Put more top-level elements on screen (a dialog opening)."""
        self.elements.extend(elements)

    def ensure_open(self) -> None:
        if self.closed:
            raise PlaywrightError("Target page, context or browser has been closed")

    def log(self, action: str, detail: str) -> None:
        self.ensure_open()
        self.actions.append((action, detail))

    def did(self, action: str) -> list[str]:
        """The details of every logged ``action`` (``page.did("click")``)."""
        return [detail for kind, detail in self.actions if kind == action]

    def _search(self, match: Callable[[FakeElement], bool], description: str) -> FakeLocator:
        def resolve() -> list[FakeElement]:
            return _unique(element for root in self.elements for element in root.walk() if match(element))

        return FakeLocator(self, resolve, description)


class FakeBrowser:
    """``fm.executor.BrowserLike`` that hands out fake pages and writes stand-in trace files.

    ``page`` is the page every ``new_page`` returns (so a test can inspect it), or a factory called each time;
    by default each call gets a blank :class:`FakePage`.
    """

    def __init__(self, page: FakePage | Callable[[], FakePage] | None = None) -> None:
        self._page = page
        self.opened: list[FakePage] = []
        self.traces: list[Path] = []
        self.tracing = False

    def new_page(self) -> FakePage:
        if isinstance(self._page, FakePage):
            page = self._page
        elif self._page is not None:
            page = self._page()
        else:
            page = FakePage()
        page.closed = False
        self.opened.append(page)
        return page

    def start_trace(self) -> None:
        if self.tracing:
            raise PlaywrightError("Tracing has been already started")
        self.tracing = True

    def stop_trace(self, path: Path) -> None:
        if not self.tracing:
            raise PlaywrightError("Must start tracing before stopping")
        self.tracing = False
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("trace.txt", "fake Playwright trace\n")
        self.traces.append(path)


# --- wiring -----------------------------------------------------------------------------------------------------------


def fake_runtime(
    api: FakeEspnApi,
    league: LeagueRow,
    *,
    transport: FakeTransport | None = None,
    browser: FakeBrowser | None = None,
    member_id: str | None = FAKE_SWID,
) -> Runtime:
    """An ``fm.executor.Runtime`` over fakes: ``api`` for reads, ``transport`` for writes, ``browser`` for UI mode."""
    return Runtime(
        reader=api.client_for(league),
        transport=transport if transport is not None else FakeTransport(),
        browser=browser if browser is not None else FakeBrowser(),
        member_id=member_id,
    )


class FakeOpener:
    """``fm.executor.RuntimeOpener`` that hands out one prepared runtime and records each call."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.calls: list[tuple[int, bool]] = []
        """``(league row id, dry_run)`` per run that got as far as opening the runtime."""

    def __call__(self, league: LeagueRow, *, dry_run: bool) -> AbstractContextManager[Runtime]:
        self.calls.append((league.row_id, dry_run))
        return nullcontext(self.runtime)
