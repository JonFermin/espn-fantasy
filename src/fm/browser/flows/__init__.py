"""Executor flows: the flow protocol (API mode + UI mode) and the registry the executor looks flows up in.

A flow carries out one family of proposal kinds on ESPN (``set_lineup`` for the lineup kinds, ``add_drop``,
``claim_waiver``, the trade flows; DESIGN section 6.3). It decides nothing and never sends a write on its own:
:func:`fm.executor.execute` drives every flow through the same steps and owns what makes a write safe.

1. :meth:`Flow.check`: preconditions read through the API (``ctx.reader`` is an :class:`fm.espn.client.EspnClient`):
   nothing affected is locked, slots are eligible, the add is still a free agent, no duplicate claim or offer. A
   failure blocks the run before anything is written.
2. API mode, :meth:`Flow.build_request`: the web app's own transaction request (the DESIGN 6.3 envelope). The executor
   checks it (a POST to this league's endpoint on :data:`WRITES_HOST`, our team, ``isLeagueManager`` false, no
   credentials in the headers), saves it to the audit folder and sends it once from the browser session with a hard
   timeout. A dry run stops after saving it.
3. UI mode, :meth:`Flow.run_ui`: a role/text-locator click-through of the same action on a :class:`PageLike`, used when
   the flow has no API mode for the move or ESPN rejected the API request in a way the UI may get past
   (:meth:`Flow.ui_may_follow`). Every state-changing click goes through :meth:`UiDriver.confirm`; in a dry run the
   first one stops the walk before anything is clicked. Selectors live in ``fm.browser.selectors``; flows use role and
   text locators.
4. :meth:`Flow.verify`: a re-read through the API showing the change. Unverified means failed (CLAUDE.md).

What the executor adds around these steps: the single-use execution token, at most one API request per attempt and
never a retry, a timeout or a 5xx marks the attempt ``unknown`` and forces a re-read with no fallback, UI mode follows
only a definite rejection or an API mode that could not send the move at all (:class:`ModeUnavailableError`), and the
request, response, screenshots and trace go to ``audit/``.

Writing a flow
--------------
One module per flow family in this package. It registers its flow on import, so nobody edits a shared file::

    from fm.browser.flows import Flow, FlowContext, Mode, Preconditions, UiDriver, Verification, WriteRequest
    from fm.browser.flows import register_flow
    from fm.proposals import LineupPayload, ProposalKind


    class SetLineup(Flow[LineupPayload]):
        name = "set_lineup"
        kinds = (ProposalKind.BENCH_INACTIVE, ProposalKind.LINEUP)
        payload_type = LineupPayload
        modes = (Mode.API, Mode.UI)

        def check(self, ctx: FlowContext[LineupPayload]) -> Preconditions: ...
        def build_request(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> WriteRequest: ...
        def run_ui(self, ctx: FlowContext[LineupPayload], ui: UiDriver, pre: Preconditions) -> None: ...
        def verify(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> Verification: ...


    register_flow(SetLineup())

:func:`flow_for` imports every public module of this package once (:func:`discover`) before it looks a flow up, the
way ``fm.cli`` finds command modules; ``_private`` modules are skipped. A ``(kind, sport)`` pair is served by one flow,
and claiming it twice is an error. ``flow_registry`` is the process-wide registry; tests build their own
:class:`FlowRegistry`. ``fm.browser.fakes`` holds the fake page, read API, write transport and browser that flow tests
run against.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from re import Pattern
from types import ModuleType
from typing import Any, ClassVar, Protocol

from fm.config import Sport
from fm.decide.registry import normalize_sport
from fm.espn.client import EspnClient
from fm.espn.ids import Game
from fm.proposals.payloads import Payload
from fm.proposals.policy import ProposalKind, kind_spec
from fm.store import SPORTS, LeagueRow, ProposalRow

WRITES_HOST = "lm-api-writes.fantasy.espn.com"
"""ESPN's write host (DESIGN 6.3). Only the executor sends anything here, and only for an approved proposal."""
WRITES_API_ROOT = f"https://{WRITES_HOST}/apis/v3/games"
MAX_TEXT_CHARS = 4000
"""How much of a response body that is not JSON an execution row keeps (the audit file keeps all of it)."""
CREDENTIAL_HEADERS: frozenset[str] = frozenset({"cookie", "authorization", "proxy-authorization"})
"""Headers a write request may not set (lower case): the browser session supplies the cookies."""
REDACTED = "<redacted>"

_ERROR_CODE = re.compile(r"\b(?:TRAN|FAILED|AUTH|GENERAL)_[A-Z0-9_]+\b")
BUSINESS_CODE_PREFIXES: tuple[str, ...] = ("TRAN_", "FAILED_")
"""ESPN error codes for a league rule (``TRAN_LINEUP_LOCKED``, ``FAILED_ROSTERLOCK``): the UI enforces the same rule."""
AUTH_CODE_PREFIX = "AUTH_"


class Mode(StrEnum):
    """How a flow carries a move out; also the ``executions.mode`` column."""

    API = "api"  # the web app's own transaction request, sent from the browser session
    UI = "ui"  # role/text-locator click-through, the fallback


def league_write_root(game: Game | str, season: int, league_id: int) -> str:
    """``https://lm-api-writes.fantasy.espn.com/apis/v3/games/{game}/seasons/{season}/segments/0/leagues/{id}``."""
    return f"{WRITES_API_ROOT}/{Game.coerce(game).value}/seasons/{season}/segments/0/leagues/{league_id}"


def transactions_url(game: Game | str, season: int, league_id: int) -> str:
    """The one endpoint the web app writes through: ``POST .../leagues/{id}/transactions/``."""
    return f"{league_write_root(game, season, league_id)}/transactions/"


def jsonable(value: Any) -> Any:
    """``value`` as plain JSON types (dicts, lists, str, numbers), for the audit files and the JSON columns.

    Mappings and sequences are copied; datetimes, paths and anything else ``json`` cannot encode become strings.
    """
    return json.loads(json.dumps(value, default=str))


# --- what a flow sees and returns -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FlowContext[P: Payload]:
    """Everything a flow gets for one proposal: the move, the league, the read client and the run's clock."""

    proposal: ProposalRow
    league: LeagueRow
    kind: ProposalKind
    payload: P
    reader: EspnClient
    """API reads for preconditions and verification. Reads only: the client has no write method."""
    now: datetime
    dry_run: bool = False
    member_id: str | None = None
    """The account's SWID (``{...}``), the envelope's ``memberId``; ``None`` when the session carried none."""

    @property
    def game(self) -> Game:
        return Game.from_sport(self.league.sport)

    @property
    def team_id(self) -> int:
        """Our team in the league: every write acts as this team and no other."""
        return self.league.team_id

    @property
    def scoring_period_id(self) -> int | None:
        """The scoring period the proposal was made for (a week in NFL, a day in NBA), when it named one."""
        return self.proposal.scoring_period_id

    @property
    def transactions_url(self) -> str:
        return transactions_url(self.game, self.league.season, self.league.espn_league_id)


@dataclass(frozen=True, slots=True)
class Preconditions:
    """What :meth:`Flow.check` found: every reason the move must not run (none means clear to write), plus the facts it
    read, as JSON-able data for the audit folder and for the later steps of the same run."""

    failures: tuple[str, ...] = ()
    observed: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "failures": list(self.failures), "observed": jsonable(self.observed)}


@dataclass(frozen=True, slots=True)
class Verification:
    """What :meth:`Flow.verify` found on re-reading the league: whether it shows the requested change."""

    matched: bool
    detail: str = ""
    expected: Mapping[str, Any] = field(default_factory=dict)
    observed: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "detail": self.detail,
            "expected": jsonable(self.expected),
            "observed": jsonable(self.observed),
        }


# --- API mode: requests, responses, transports ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WriteRequest:
    """One API-mode write: a JSON body POSTed to ESPN's transactions endpoint.

    Cookies never belong here: the browser session adds them when the executor sends the request (and the executor
    refuses a request that sets :data:`CREDENTIAL_HEADERS`). ``headers`` holds anything else the web app sends (#14
    captures which).
    """

    url: str
    body: Mapping[str, Any]
    headers: Mapping[str, str] = field(default_factory=dict)
    method: str = "POST"

    def to_json(self) -> dict[str, Any]:
        """The request as the audit folder and ``executions.request`` keep it, credential headers masked."""
        headers = {
            name: REDACTED if name.lower() in CREDENTIAL_HEADERS else value for name, value in self.headers.items()
        }
        return {"method": self.method, "url": self.url, "headers": headers, "body": jsonable(self.body)}


class WriteOutcome(StrEnum):
    """What an HTTP answer to a write means for the move."""

    ACCEPTED = "accepted"  # 2xx: ESPN took it; the re-read decides whether it really happened
    REJECTED = "rejected"  # 3xx/4xx: ESPN refused it, so nothing was applied
    UNKNOWN = "unknown"  # 5xx: a gateway may have answered for a backend that applied it


@dataclass(frozen=True, slots=True)
class WriteResponse:
    """ESPN's answer to a write: the status, the parsed JSON body (``None`` when it was not JSON) and the raw text."""

    status: int
    body: Any = None
    text: str = ""
    elapsed_s: float | None = None

    @classmethod
    def from_text(cls, status: int, text: str, *, elapsed_s: float | None = None) -> WriteResponse:
        """Parse ``text`` as JSON when it is JSON."""
        try:
            body = json.loads(text) if text.strip() else None
        except ValueError:
            body = None
        return cls(status=status, body=body, text=text, elapsed_s=elapsed_s)

    @property
    def outcome(self) -> WriteOutcome:
        if 200 <= self.status < 300:
            return WriteOutcome.ACCEPTED
        if self.status >= 500:
            return WriteOutcome.UNKNOWN
        return WriteOutcome.REJECTED

    @property
    def error_codes(self) -> tuple[str, ...]:
        """ESPN's error codes (``details[].type``, plus any ``TRAN_``/``FAILED_``/``AUTH_`` code in the text)."""
        codes: list[str] = []
        if isinstance(self.body, Mapping):
            details = self.body.get("details")
            if isinstance(details, list):
                for detail in details:
                    if isinstance(detail, Mapping) and isinstance(detail.get("type"), str) and detail["type"]:
                        codes.append(detail["type"])
        codes.extend(_ERROR_CODE.findall(self.text))
        return tuple(dict.fromkeys(codes))

    @property
    def message(self) -> str:
        """ESPN's own words for an error (``details[].message`` or ``messages[]``), or ``""``."""
        if not isinstance(self.body, Mapping):
            return ""
        parts: list[str] = []
        details = self.body.get("details")
        if isinstance(details, list):
            for detail in details:
                if isinstance(detail, Mapping):
                    text = detail.get("message") or detail.get("shortMessage")
                    if isinstance(text, str) and text:
                        parts.append(text)
        messages = self.body.get("messages")
        if not parts and isinstance(messages, list):
            parts.extend(str(message) for message in messages if message)
        return "; ".join(dict.fromkeys(parts))

    def describe(self) -> str:
        """``HTTP 409 TRAN_LINEUP_LOCKED: Lineup is locked`` for logs and the CLI."""
        text = f"HTTP {self.status}"
        if self.error_codes:
            text += " " + ", ".join(self.error_codes)
        if self.message:
            text += f": {self.message}"
        return text

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status,
            "outcome": self.outcome.value,
            "error_codes": list(self.error_codes),
            "elapsed_s": self.elapsed_s,
        }
        if self.body is not None:
            data["body"] = jsonable(self.body)
        else:
            data["text"] = self.text[:MAX_TEXT_CHARS]
        return data


class WriteTransport(Protocol):
    """Sends one API-mode write. The executor calls it at most once per attempt and never retries.

    ``timeout_s`` is a hard limit for the whole exchange. Raise :class:`WriteTimeoutError` when it runs out and
    :class:`WriteUncertainError` for any failure after which the request may have reached ESPN;
    :class:`WriteRefusedError` only when nothing was sent. Any HTTP answer, error statuses included, is returned.
    """

    def send(self, request: WriteRequest, *, timeout_s: float) -> WriteResponse: ...


class WriteError(Exception):
    """A write did not complete normally."""


class WriteRefusedError(WriteError):
    """Refused before anything was sent: nothing reached ESPN."""


class WriteUncertainError(WriteError):
    """The request may or may not have reached ESPN: the write is UNKNOWN until a re-read says otherwise."""


class WriteTimeoutError(WriteUncertainError):
    """No answer within the hard timeout."""


class ModeUnavailableError(Exception):
    """The flow cannot carry this move out in this mode (the action is not captured yet, the page lacks the control).
    Nothing was sent, so the executor may try the next mode."""


class DryRunStop(BaseException):
    """Raised by :meth:`UiDriver.confirm` in a dry run: the walk reached its first write and stops before clicking.

    A ``BaseException`` so a flow's ``except Exception`` cannot swallow it and keep walking.
    """

    def __init__(self, what: str) -> None:
        super().__init__(what)
        self.what = what


# --- UI mode: the page surface flows use ------------------------------------------------------------------------------


class LocatorLike(Protocol):
    """The slice of ``playwright.sync_api.Locator`` flows use; the fake page's locator has the same shape."""

    def click(self, *, timeout: float | None = None) -> None: ...
    def fill(self, value: str, *, timeout: float | None = None) -> None: ...
    def check(self, *, timeout: float | None = None) -> None: ...
    def uncheck(self, *, timeout: float | None = None) -> None: ...
    def select_option(
        self,
        value: str | Sequence[str] | None = None,
        *,
        label: str | Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> list[str]: ...
    def press(self, key: str, *, timeout: float | None = None) -> None: ...
    def count(self) -> int: ...
    def is_visible(self) -> bool: ...
    def is_enabled(self) -> bool: ...
    def is_checked(self) -> bool: ...
    def inner_text(self, *, timeout: float | None = None) -> str: ...
    def text_content(self, *, timeout: float | None = None) -> str | None: ...
    def wait_for(self, *, state: Any = None, timeout: float | None = None) -> None: ...
    @property
    def first(self) -> LocatorLike: ...
    @property
    def last(self) -> LocatorLike: ...
    def nth(self, index: int) -> LocatorLike: ...
    def filter(self, *, has_text: str | Pattern[str] | None = None) -> LocatorLike: ...
    def all(self) -> Sequence[LocatorLike]: ...
    def get_by_role(
        self, role: Any, *, name: str | Pattern[str] | None = None, exact: bool | None = None
    ) -> LocatorLike: ...
    def get_by_text(self, text: str | Pattern[str], *, exact: bool | None = None) -> LocatorLike: ...
    def get_by_label(self, text: str | Pattern[str], *, exact: bool | None = None) -> LocatorLike: ...
    def locator(self, selector: str, /) -> LocatorLike: ...


class PageLike(Protocol):
    """The slice of ``playwright.sync_api.Page`` flows use; ``fm.browser.fakes.FakePage`` has the same shape."""

    @property
    def url(self) -> str: ...
    def goto(self, url: str, *, timeout: float | None = None, wait_until: Any = None) -> object: ...
    def get_by_role(
        self, role: Any, *, name: str | Pattern[str] | None = None, exact: bool | None = None
    ) -> LocatorLike: ...
    def get_by_text(self, text: str | Pattern[str], *, exact: bool | None = None) -> LocatorLike: ...
    def get_by_label(self, text: str | Pattern[str], *, exact: bool | None = None) -> LocatorLike: ...
    def get_by_test_id(self, test_id: str | Pattern[str]) -> LocatorLike: ...
    def locator(self, selector: str, /) -> LocatorLike: ...
    def wait_for_load_state(self, state: Any = None, *, timeout: float | None = None) -> None: ...
    def screenshot(self, *, path: str | Path | None = None, full_page: bool | None = None) -> bytes: ...
    def set_default_timeout(self, timeout: float) -> None: ...
    def set_default_navigation_timeout(self, timeout: float) -> None: ...
    def title(self) -> str: ...
    def close(self) -> None: ...


class UiDriver(Protocol):
    """What UI mode hands a flow: the page, screenshots into the audit folder, and the write gate.

    :meth:`confirm` is the only way a UI flow may make a state-changing click (the final Submit, a "Here" that saves a
    lineup move). The executor counts and caps those clicks, treats any failure after one as an unknown outcome, and in
    a dry run stops at the first by raising :class:`DryRunStop` before clicking.
    """

    @property
    def page(self) -> PageLike: ...
    @property
    def dry_run(self) -> bool: ...
    def screenshot(self, name: str) -> None: ...
    def confirm(self, target: LocatorLike, *, what: str) -> None: ...


# --- the flow protocol ------------------------------------------------------------------------------------------------


def rejection_allows_ui(response: WriteResponse) -> bool:
    """Whether UI mode may follow this API answer: only a definite rejection that a click-through could get past.

    Not after an acceptance or an unknown outcome (that would be a retry), not after a 401 or an ``AUTH_`` code (the
    page is signed out too), and not after an ESPN league-rule code (``TRAN_``/``FAILED_``), which the UI enforces the
    same way. A rejection without one of those (a 400 for a changed envelope, a 403 or 404 for a moved endpoint) means
    the API path broke, which is what the UI fallback is for.
    """
    if response.outcome is not WriteOutcome.REJECTED:
        return False
    codes = response.error_codes
    if response.status == 401 or any(code.startswith(AUTH_CODE_PREFIX) for code in codes):
        return False
    return not any(code.startswith(BUSINESS_CODE_PREFIXES) for code in codes)


def transaction_id_from(response: WriteResponse) -> str | None:
    """The ESPN transaction id in a write's answer (``id``, ``transactionId``, or the same under ``transaction``)."""
    body = response.body
    for candidate in (body, body.get("transaction") if isinstance(body, Mapping) else None):
        if not isinstance(candidate, Mapping):
            continue
        for key in ("id", "transactionId"):
            value = candidate.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, int) and not isinstance(value, bool):
                return str(value)
    return None


class Flow[P: Payload](ABC):
    """One family of ESPN moves, in API mode, UI mode or both (DESIGN 6.3). Subclasses set the class attributes and
    implement :meth:`check` and :meth:`verify`, plus :meth:`build_request` for API mode and :meth:`run_ui` for UI mode.

    Flows are stateless: one instance serves every proposal and every league of its sports.
    """

    name: ClassVar[str]
    """Short id for logs and the audit trail: ``set_lineup``, ``add_drop``, ``claim_waiver``."""
    kinds: ClassVar[tuple[ProposalKind, ...]]
    """The proposal kinds this flow carries out."""
    payload_type: ClassVar[type[Payload]]
    """The payload model those kinds carry (``fm.proposals.policy.KINDS``); checked at registration."""
    modes: ClassVar[tuple[Mode, ...]] = (Mode.API, Mode.UI)
    """Modes in the order the executor tries them: API first, the UI click-through as the fallback."""
    sports: ClassVar[tuple[Sport, ...]] = SPORTS

    @abstractmethod
    def check(self, ctx: FlowContext[P]) -> Preconditions:
        """Read the league through ``ctx.reader`` and list every reason the move must not run now.

        Raise ``fm.espn.client.EspnClientError`` (what the reader raises) when the league cannot be read: the executor
        then refuses without spending the execution token, so the proposal can run once ESPN answers again.
        """

    def build_request(self, ctx: FlowContext[P], pre: Preconditions) -> WriteRequest:
        """API mode: the transaction request the web app would send for this move. Pure: never send anything."""
        raise ModeUnavailableError(f"{self.name} has no API mode")

    def run_ui(self, ctx: FlowContext[P], ui: UiDriver, pre: Preconditions) -> None:
        """UI mode: click the move through on ``ui.page``, sending every state-changing click through ``ui.confirm``."""
        raise ModeUnavailableError(f"{self.name} has no UI mode")

    @abstractmethod
    def verify(self, ctx: FlowContext[P], pre: Preconditions) -> Verification:
        """Re-read the league through ``ctx.reader``: does it show the move? The executor may call this more than once
        (re-reads are safe), including before a fallback to check that a rejected request left nothing behind."""

    def ui_may_follow(self, response: WriteResponse) -> bool:
        """Whether UI mode may run after this API answer. Defaults to :func:`rejection_allows_ui`; a flow that maps
        ESPN's error codes (#25) can be stricter, never looser: the executor never falls back after an unknown."""
        return rejection_allows_ui(response)

    def transaction_id(self, response: WriteResponse) -> str | None:
        """ESPN's id for the transaction a write created, stored on the execution row."""
        return transaction_id_from(response)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({getattr(self, 'name', '?')})"


# --- registry ---------------------------------------------------------------------------------------------------------


class DuplicateFlowError(ValueError):
    """A ``(kind, sport)`` pair already has a flow."""


class UnknownFlowError(LookupError):
    """No flow is registered for a ``(kind, sport)`` pair."""


@dataclass(frozen=True, slots=True)
class FlowRegistration:
    """One ``(kind, sport)`` pair and the flow that serves it."""

    kind: ProposalKind
    sport: Sport
    flow: Flow[Any]

    @property
    def key(self) -> tuple[ProposalKind, Sport]:
        return (self.kind, self.sport)

    @property
    def label(self) -> str:
        """``nfl:lineup -> set_lineup`` for logs and errors."""
        return f"{self.sport}:{self.kind.value} -> {self.flow.name}"


class FlowRegistry:
    """Flows keyed by ``(kind, sport)``. Registration checks the flow is complete and consistent with the kinds."""

    def __init__(self) -> None:
        self._entries: dict[tuple[ProposalKind, Sport], FlowRegistration] = {}

    def register(self, flow: Flow[Any]) -> Flow[Any]:
        """Bind ``flow`` to every ``(kind, sport)`` it serves, all or nothing. Raises ``DuplicateFlowError`` when a pair
        is taken and ``TypeError`` / ``ValueError`` for a flow that is incomplete or does not fit its kinds."""
        registrations = [FlowRegistration(kind, sport, flow) for kind, sport in _validated_pairs(flow)]
        for registration in registrations:
            existing = self._entries.get(registration.key)
            if existing is not None:
                raise DuplicateFlowError(
                    f"{registration.sport}:{registration.kind.value} is served by {existing.flow!r} already; "
                    f"{flow!r} cannot claim it too"
                )
        for registration in registrations:
            self._entries[registration.key] = registration
        return flow

    def get(self, kind: ProposalKind | str, sport: Game | str) -> Flow[Any] | None:
        registration = self._entries.get((kind_spec(kind).kind, normalize_sport(sport)))
        return None if registration is None else registration.flow

    def flow_for(self, kind: ProposalKind | str, sport: Game | str) -> Flow[Any]:
        """The flow for ``(kind, sport)``; ``UnknownFlowError`` names what the sport has instead."""
        flow = self.get(kind, sport)
        if flow is None:
            normalized = normalize_sport(sport)
            known = ", ".join(entry.label for entry in self._entries.values() if entry.sport == normalized)
            raise UnknownFlowError(
                f"no executor flow for {kind_spec(kind).kind.value} proposals in {normalized}; "
                f"registered: {known or 'none'}"
            )
        return flow

    def registered(self) -> tuple[FlowRegistration, ...]:
        """Every registration, in registration order."""
        return tuple(self._entries.values())

    def unregister(self, flow: Flow[Any]) -> int:
        """Remove every pair ``flow`` serves; returns how many were removed."""
        keys = [key for key, entry in self._entries.items() if entry.flow is flow]
        for key in keys:
            del self._entries[key]
        return len(keys)

    def clear(self) -> None:
        self._entries.clear()

    def __contains__(self, key: object) -> bool:
        """``(ProposalKind.LINEUP, "nfl") in registry``."""
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        kind, sport = key
        try:
            return self.get(kind, sport) is not None
        except (ValueError, TypeError, LookupError):
            return False

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[FlowRegistration]:
        return iter(self.registered())


def _validated_pairs(flow: Flow[Any]) -> list[tuple[ProposalKind, Sport]]:
    """The ``(kind, sport)`` pairs ``flow`` serves, after checking it is a complete flow that fits those kinds."""
    if not isinstance(flow, Flow):
        raise TypeError(f"expected a Flow instance, got {flow!r}")
    cls = type(flow)
    name = getattr(flow, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} needs a name")
    kinds = tuple(kind_spec(kind).kind for kind in _class_tuple(flow, "kinds"))
    if not kinds:
        raise ValueError(f"flow {name} serves no proposal kinds")
    payload_type = getattr(flow, "payload_type", None)
    for kind in kinds:
        expected = kind_spec(kind).payload_type
        if not isinstance(payload_type, type) or not issubclass(expected, payload_type):
            shown = getattr(payload_type, "__name__", repr(payload_type))
            raise TypeError(f"flow {name} takes {shown}, but {kind.value} proposals carry {expected.__name__}")
    modes = tuple(Mode(mode) for mode in _class_tuple(flow, "modes"))
    if not modes or len(set(modes)) != len(modes):
        raise ValueError(f"flow {name} must list each of its modes once, got {modes!r}")
    if Mode.API in modes and cls.build_request is Flow.build_request:
        raise TypeError(f"flow {name} lists API mode but does not implement build_request")
    if Mode.UI in modes and cls.run_ui is Flow.run_ui:
        raise TypeError(f"flow {name} lists UI mode but does not implement run_ui")
    sports: list[Sport] = []
    for sport in _class_tuple(flow, "sports"):
        normalized = normalize_sport(sport)
        if normalized not in sports:
            sports.append(normalized)
    if not sports:
        raise ValueError(f"flow {name} serves no sport")
    return [(kind, sport) for kind in dict.fromkeys(kinds) for sport in sports]


def _class_tuple(flow: Flow[Any], attribute: str) -> tuple[Any, ...]:
    value = getattr(flow, attribute, None)
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise TypeError(f"{type(flow).__name__}.{attribute} must be a tuple, got {value!r}")
    return tuple(value)


flow_registry = FlowRegistry()
"""The process-wide registry flow modules register on at import."""


def register_flow[F: Flow[Any]](flow: F) -> F:
    """Register ``flow`` on the process-wide registry; returns it unchanged."""
    flow_registry.register(flow)
    return flow


def discover(package: ModuleType | None = None) -> tuple[str, ...]:
    """Import every public module of ``package`` (this one by default) in name order, so each registers its flows.

    Imports are cached, so calling this again is cheap and registers nothing twice. Returns the module names.
    """
    pkg = package if package is not None else sys.modules[__name__]
    names: list[str] = []
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda module: module.name):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{pkg.__name__}.{info.name}")
        names.append(info.name)
    return tuple(names)


def flow_for(kind: ProposalKind | str, sport: Game | str) -> Flow[Any]:
    """The process-wide flow for ``(kind, sport)``, after importing this package's flow modules."""
    discover()
    return flow_registry.flow_for(kind, sport)


def registered_flows() -> tuple[FlowRegistration, ...]:
    """Every process-wide registration, after importing this package's flow modules."""
    discover()
    return flow_registry.registered()
