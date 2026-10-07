"""The UI fallback drill (DESIGN sections 6.3 and 13, ROADMAP #41): a weekly dry run of every UI-mode flow against the
live site that stops before the final save, so the click-through stays known-good for the day API mode breaks.

For each league the drill reads the current views (our roster, the league's settings, the player pool), picks a target
for each flow that has a UI mode (:data:`PLANNERS`: a swap of two unlocked players for ``set_lineup``; a free agent and
a droppable player for ``add_drop``; a waiver player and a droppable player for ``claim_waiver``) and runs the flow's
own :meth:`fm.browser.flows.Flow.run_ui` on a live page. The walk opens the pages, clicks everything short of the save
(MOVE, Drop Player, Continue) and checks that each expected control resolves. It ends at the flow's first
:meth:`fm.browser.flows.UiDriver.confirm`, which here only waits for the control to show and then stops the walk.
Then the drill navigates away, which discards the half-made move, and closes the page. Nothing is proposed or stored:
the targets are real players from the read views only so the pages show the controls a real move would use, and no
roster change is implied.

Nothing the drill does can save, in four layers. Any one of them alone is enough to keep the league unchanged:

1. **The browser context is a dry run.** ``fm drill`` opens it as ``live_opener(...)(row, dry_run=True)``, the way ``fm
   execute --dry-run`` and ``fm canary`` do: every request that is not a GET to ``espn.com`` and everything sent to
   ESPN's write host is aborted, and the runtime's write transport refuses every send. :func:`run_drills` refuses a
   runtime the opener did not mark ``dry_run`` (:attr:`fm.executor.Runtime.dry_run`, set by ``opener(row,
   dry_run=True)``) or whose transport is not that :class:`fm.executor.RefusingTransport` (:class:`DrillError`).
2. **The driver cannot confirm.** The flow gets a :class:`DrillUi` in place of the executor's. Its ``confirm`` waits for
   the control (so a missing one fails the drill), records the step it reached and raises
   :class:`fm.browser.flows.DryRunStop`; the control it is handed can never be clicked, and ``confirm`` returning
   instead of raising is a :class:`DrillSafetyError`.
3. **The page refuses the save.** The page the flow sees is a :class:`GuardedPage`: a click (or an Enter or Space
   press, a check, a select) on any locator that could be a final-save control (the lineup's HERE, the player list's
   Add and Claim, the roster-fix dialog's Confirm) or that cannot be classified raises :class:`FinalSaveClickError`
   before it reaches the page, even for a flow that skips ``ui.confirm``. A locator is a final-save control when its
   name matches a probe built from the players the plan and the preconditions name, and, whatever the player, when
   its role name or text starts with Confirm, Add, Claim or Here. Both errors are ``BaseException`` so a flow's
   ``except Exception`` cannot swallow them.
4. **A watcher on the page's requests.** A request to ESPN's write host or a non-GET to a ``/transactions`` path is
   recorded when the page makes it (:class:`RequestWatch`). The context aborts it, but the drill reports it as a
   safety finding all the same.

A drill fails (:class:`DrillFailureKind`) when a flow's walk raised before its confirm (a page or control did not
show, ESPN's "Log in Required" page came up), when it ended without reaching a confirm, when the views could not be
read, when a registered flow has no planner (so a new UI flow cannot go unwatched, as with the canary's page
addresses), when every flow was skipped (nothing was checked), and for anything a safety layer caught. A flow with
no target right now (nobody on waivers, every player locked) is skipped with its reason, never alerted on. The result
is a :class:`DrillReport` per league; :func:`drill_alert` turns reports into one ``fm.notify.Message`` for
``fm.notify.send_alert``, and :func:`drill_payload` into the structured payload. Nothing here sends: ``fm drill``
(``fm.commands.drill``) does.

Scheduling: ``fm schedule`` installs the tick only, and this module is out of the tick on purpose (a drill opens a
browser for minutes). To run it weekly, register a second scheduled job that runs ``fm drill`` (the wrapper
``fm.jobs.scheduler_windows`` or ``fm.jobs.scheduler_macos`` writes for the tick shows how), or have the tick's
health checks call ``fm.commands.drill.run_live`` when the last drill is a week old. The alert is sent by the
command, so either path alerts the same way. Selectors live in ``fm.browser.selectors`` alone; the drill spells none
out and clicks only what the flows click.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from re import Pattern
from typing import Any
from urllib.parse import urlsplit

from fm import paths
from fm.browser import selectors
from fm.browser.flows import (
    WRITES_HOST,
    DryRunStop,
    Flow,
    FlowContext,
    FlowRegistry,
    LocatorLike,
    Mode,
    PageLike,
    discover,
    flow_registry,
    jsonable,
)
from fm.browser.flows import Preconditions as FlowPreconditions
from fm.config import League
from fm.espn.client import EspnClient, EspnClientError
from fm.espn.ids import Game
from fm.espn.models import PlayersView, RosterEntry, TeamRoster
from fm.espn.settings import LeagueSettings, SlotKind
from fm.executor import (
    DEFAULT_UI_TIMEOUT_S,
    AuditLog,
    RefusingTransport,
    Runtime,
    UiSession,
    dry_run_block_reason,
)
from fm.notify.base import Message
from fm.notify.messages import alert
from fm.proposals import AddDropPayload, LineupMove, LineupPayload, Payload, WaiverPayload
from fm.store import LeagueRow, ProposalRow

logger = logging.getLogger(__name__)

POOL_PROBE = 25
"""Players the pool read is asked for: enough to hold a free agent and a waiver player, little to pull."""
ALERT_LINES = 12
"""Findings an alert lists before it says how many more there are; the structured payload keeps all of them."""
BLANK_PAGE = "about:blank"
"""Where the drill navigates after a walk: away from the half-made move."""
DRILL_ACTOR = "fm drill"
MAX_PROBE_NAMES = 40
"""Player names the final-save probes are built from (their add/drop pairs grow with the square)."""


class DrillError(RuntimeError):
    """The drill was refused before it touched a page: a runtime that is not a dry run, an unknown flow name."""


class DrillSafetyError(BaseException):
    """A safety layer stopped the drill. A ``BaseException`` so a flow's ``except Exception`` cannot swallow it."""


class FinalSaveClickError(DrillSafetyError):
    """The walk tried to click something that could save (or something the guard could not classify)."""


class NoDrillTargetError(Exception):
    """The views show no player a flow could be drilled with right now: the flow is skipped, never alerted on."""


class DrillStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DrillFailureKind(StrEnum):
    """What the drill found wrong."""

    UI_FAILED = "ui_failed"  # the walk raised before its confirm: a page or control did not show, or signed out
    NO_CONFIRM = "no_confirm"  # the walk ended without reaching a confirm
    NO_PLANNER = "no_planner"  # a registered flow has a UI mode and the drill has no target planner for it
    SETUP_FAILED = "setup_failed"  # the views or the flow's preconditions could not be read
    NOT_DRILLED = "not_drilled"  # every flow was skipped, so nothing was checked
    UNSAFE = "unsafe"  # a safety layer caught a save attempt


@dataclass(frozen=True, slots=True)
class DrillFinding:
    """One thing wrong: which flow, what and where."""

    kind: DrillFailureKind
    flow: str
    detail: str
    url: str | None = None
    """The last page the walk opened, when there was one."""

    def line(self) -> str:
        return f"{self.flow}: {self.kind.value}: {self.detail}"

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "flow": self.flow, "detail": self.detail, "url": self.url}


@dataclass(frozen=True, slots=True)
class DrillResult:
    """One flow's drill."""

    flow: str
    status: DrillStatus
    target: str = ""
    """What the walk was run on (``swap A (UTIL) and B (Bench)``)."""
    reached: str | None = None
    """The confirm the walk stopped before: the step that would have saved."""
    pages: tuple[str, ...] = ()
    """The addresses the walk opened, in order."""
    clicks: tuple[str, ...] = ()
    """The clicks it made (MOVE, Drop Player, Continue): none of them saves."""
    skipped: str | None = None
    notes: tuple[str, ...] = ()
    """What the flow's preconditions said about the move now (a closed lock, a pending claim): not drill failures,
    since the drill never saves."""
    artifacts: tuple[str, ...] = ()
    """Screenshots, relative to the audit root."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "flow": self.flow,
            "status": self.status.value,
            "target": self.target,
            "reached": self.reached,
            "pages": list(self.pages),
            "clicks": list(self.clicks),
            "skipped": self.skipped,
            "notes": list(self.notes),
            "artifacts": list(self.artifacts),
        }


@dataclass(frozen=True, slots=True)
class DrillReport:
    """One league's drill, like :class:`fm.browser.canary.CanaryReport`."""

    league: str
    """The league's config key."""
    results: tuple[DrillResult, ...] = ()
    findings: tuple[DrillFinding, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def unsafe(self) -> tuple[DrillFinding, ...]:
        return tuple(finding for finding in self.findings if finding.kind is DrillFailureKind.UNSAFE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "league": self.league,
            "ok": self.ok,
            "findings": [finding.as_dict() for finding in self.findings],
            "flows": [result.as_dict() for result in self.results],
        }


# --- targets ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DrillViews:
    """What the planners read: the current views of one league."""

    game: Game
    settings: LeagueSettings
    roster: TeamRoster
    """Our roster."""
    pool: PlayersView
    """The top of the player pool (free agents and players on waivers)."""


@dataclass(frozen=True, slots=True)
class DrillPlan:
    """What one flow is drilled with."""

    payload: Payload
    description: str
    names: tuple[str, ...]
    """The players' names, which the guard builds its final-save probes from."""
    scoring_period_id: int | None = None


type Planner = Callable[[DrillViews], DrillPlan]
"""Picks a flow's drill target from the views; raises :class:`NoDrillTargetError` when there is none."""


def _kinds(views: DrillViews) -> dict[int, SlotKind]:
    return {slot.slot_id: slot.kind for slot in views.settings.lineup_slots}


def _name(entry: RosterEntry) -> str:
    return entry.player.full_name.strip()


def _unique_name(entry: RosterEntry, roster: TeamRoster) -> bool:
    """The walk finds a player's row by his name, so it must not match another row (as ``set_lineup`` requires)."""
    needle = _name(entry).casefold()
    return bool(needle) and sum(1 for other in roster.entries if needle in _name(other).casefold()) == 1


def swap_target(views: DrillViews) -> DrillPlan:
    """A swap of two unlocked players in different lineup slots, each eligible for the other's: a starter and a bench
    player first. The flow's UI walk makes it as MOVE on one row and HERE on the other (the confirm)."""
    kinds = _kinds(views)
    movable = [
        entry
        for entry in views.roster.entries
        if kinds.get(entry.lineup_slot_id) in (SlotKind.ACTIVE, SlotKind.BENCH)
        and not entry.lineup_locked
        and _unique_name(entry, views.roster)
    ]
    best: tuple[int, RosterEntry, RosterEntry] | None = None
    for first in movable:
        for second in movable:
            if first.lineup_slot_id == second.lineup_slot_id:
                continue
            if second.lineup_slot_id not in first.player.eligible_slots:
                continue
            if first.lineup_slot_id not in second.player.eligible_slots:
                continue
            starter_for_bench = (kinds[first.lineup_slot_id], kinds[second.lineup_slot_id]) == (
                SlotKind.ACTIVE,
                SlotKind.BENCH,
            )
            rank = 0 if starter_for_bench else 1
            if best is None or rank < best[0]:
                best = (rank, first, second)
    if best is None:
        raise NoDrillTargetError("no two unlocked players on our roster can trade lineup slots")
    _, first, second = best
    payload = LineupPayload(
        moves=(
            LineupMove(espn_id=first.player_id, from_slot_id=first.lineup_slot_id, to_slot_id=second.lineup_slot_id),
            LineupMove(espn_id=second.player_id, from_slot_id=second.lineup_slot_id, to_slot_id=first.lineup_slot_id),
        )
    )
    label_first = selectors.slot_label(views.game, first.lineup_slot_id)
    label_second = selectors.slot_label(views.game, second.lineup_slot_id)
    return DrillPlan(
        payload,
        f"swap {_name(first)} ({label_first}) and {_name(second)} ({label_second})",
        (_name(first), _name(second)),
    )


def _drop_candidate(views: DrillViews) -> RosterEntry:
    """A player the roster-fix page offers a Drop Player button for: droppable, unlocked, off IR (bench first)."""
    kinds = _kinds(views)
    droppable = [
        entry
        for entry in views.roster.entries
        if entry.player.droppable is not False
        and not entry.lineup_locked
        and not entry.player_pool_entry.roster_locked
        and kinds.get(entry.lineup_slot_id) in (SlotKind.ACTIVE, SlotKind.BENCH)
        and _unique_name(entry, views.roster)
    ]
    droppable.sort(key=lambda entry: kinds[entry.lineup_slot_id] is not SlotKind.BENCH)
    if not droppable:
        raise NoDrillTargetError("no player on our roster can be dropped (all locked, undroppable or on IR)")
    return droppable[0]


def add_drop_target(views: DrillViews) -> DrillPlan:
    """The first free agent of the pool and a droppable player: the roster-fix walk (Drop, Continue, Confirm)."""
    entry = next(
        (
            item
            for item in views.pool.players
            if item.is_free_agent and not item.roster_locked and item.player.full_name
        ),
        None,
    )
    if entry is None:
        raise NoDrillTargetError("the pool read shows no free agent")
    drop = _drop_candidate(views)
    name = entry.player.full_name.strip()
    return DrillPlan(
        AddDropPayload(add_espn_id=entry.id, drop_espn_id=drop.player_id),
        f"add {name} and drop {_name(drop)}",
        (name, _name(drop)),
    )


def claim_target(views: DrillViews) -> DrillPlan:
    """The first player on waivers and a droppable player, with no bid: the roster-fix walk for a claim."""
    entry = next(
        (item for item in views.pool.players if item.is_on_waivers and item.player.full_name),
        None,
    )
    if entry is None:
        raise NoDrillTargetError("the pool read shows nobody on waivers")
    drop = _drop_candidate(views)
    name = entry.player.full_name.strip()
    return DrillPlan(
        WaiverPayload(add_espn_id=entry.id, drop_espn_id=drop.player_id),
        f"claim {name} and drop {_name(drop)}",
        (name, _name(drop)),
    )


PLANNERS: Mapping[str, Planner] = {
    "set_lineup": swap_target,
    "add_drop": add_drop_target,
    "claim_waiver": claim_target,
}
"""How to pick a target for each flow with a UI mode, by flow name. A UI flow without an entry is reported as
:attr:`DrillFailureKind.NO_PLANNER`, and ``tests/browser/test_drills.py`` fails until it has one. The one-click Add on
the player list (an add into roster room) has no walk of its own to drill, and the canary watches that page's
selectors."""


# --- the guard: layer 3 -----------------------------------------------------------------------------------------------


def _same(text: str) -> str:
    return " ".join(text.split()).casefold()


@dataclass(frozen=True, slots=True)
class _Spec:
    """What a locator was built from, as far as the guard can tell: a role and name, or a text. Anything else (a label,
    a test id, a CSS selector) is unknown, and an unknown locator is never clicked."""

    role: str | None = None
    name: str | Pattern[str] | None = None
    text: str | Pattern[str] | None = None
    exact: bool | None = None
    known: bool = True

    def describe(self) -> str:
        if not self.known:
            return "an unclassified locator"
        wanted = self.name if self.role is not None else self.text
        shown = wanted.pattern if isinstance(wanted, Pattern) else wanted
        return f"{self.role or 'text'}[{shown}]"

    def confirm_like(self) -> bool:
        """Whether the name this locator looks for starts with Confirm, Add, Claim or Here (any alternative of a
        pattern counts), whoever the player is: the shapes of every save control, which the probes only spell out for
        the players the drill knows of."""
        wanted = self.name if self.role is not None else self.text
        if wanted is None:
            return False
        if isinstance(wanted, Pattern):
            return any(_SAVE_WORDS.match(lead) for lead in _leads(wanted.pattern))
        return _SAVE_WORDS.match(wanted.strip()) is not None

    def matches(self, accessible_name: str) -> bool:
        """Whether this locator would pick a control with this name. One that names nothing picks any."""
        if not self.known:
            return True
        wanted = self.name if self.role is not None else self.text
        if wanted is None:
            return True
        if isinstance(wanted, Pattern):
            return wanted.search(accessible_name) is not None
        if self.exact:
            return _same(accessible_name) == _same(wanted)
        return _same(wanted) in _same(accessible_name)


_UNKNOWN = _Spec(known=False)

_SAVE_WORDS = re.compile(r"(?:confirm|add|claim|here)\b", re.IGNORECASE)
"""What the name of every control that could save starts with (``Confirm move of ...``, ``Add ... for ...``, ``Claim
... for ...``, HERE), compared case-insensitively."""
_PATTERN_NOISE = re.compile(r"(?:\^|\\s[*+?]?|\\b|\s|\(\?[a-zA-Z-]*[:)]|\()+")
_FLAG_GROUP = re.compile(r"\(\?[a-zA-Z-]*\)")


def _leads(pattern: str) -> list[str]:
    """What each alternative of a regex begins with, past anchors, whitespace and opening groups: for
    ``^\\s*(?:confirm move of\\s.+|move)\\s*$`` that is ``confirm move of ...`` and ``move ...``. Only the alternatives
    of the groups the pattern opens with count (the ``|`` in ``continue to (?:add|claim) ...`` sits in a later one)."""
    first = _PATTERN_NOISE.match(pattern)
    start = first.end() if first else 0
    noise = pattern[:start]
    lead_depth = noise.count("(") - len(_FLAG_GROUP.findall(noise))
    leads = [pattern[start : start + 40]]
    depth, index = lead_depth, start
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            index += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "|" and depth <= lead_depth:
            skipped = _PATTERN_NOISE.match(pattern, index + 1)
            begin = skipped.end() if skipped else index + 1
            leads.append(pattern[begin : begin + 40])
        index += 1
    return leads


def final_save_probes(names: Sequence[str]) -> tuple[str, ...]:
    """The accessible names of every control that could save, for ``names``' players: the lineup editor's HERE
    (``Confirm move of <Player> to <Slot>``, or ``Move`` on an empty row), the player list's ``Add``/``Claim`` button
    and the roster-fix dialog's ``Confirm add|claim <Player> and drop <Player>``. The shapes are the ones
    ``fm.browser.selectors`` documents."""
    probes = ["Move"]
    for name in names:
        probes += [
            f"Confirm move of {name} to Bench",
            f"Add {name} Guard for Team",
            f"Claim {name} Guard for Team",
        ]
    for added in names:
        for dropped in names:
            probes += [f"Confirm add {added} and drop {dropped}", f"Confirm claim {added} and drop {dropped}"]
    return tuple(dict.fromkeys(probes))


_ACTIVATION_KEYS = frozenset({"enter", "return", "numpadenter", "space"})


def _activates(key: str) -> bool:
    """Whether pressing ``key`` (``Enter``, ``Control+Enter``, ``Space``, a lone space...) would activate a control."""
    last = " " if key.strip() == "" else key.rsplit("+", 1)[-1].strip().lower()
    return last == " " or last in _ACTIVATION_KEYS


def observed_names(observed: Any) -> tuple[str, ...]:
    """The player names in a flow's ``Preconditions.observed`` (JSON-able data): every string under a ``name`` key, at
    any depth. The probes are built from these as well as from the plan, because a flow builds its locators from the
    names the preconditions read, which need not be the plan's."""
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key == "name" and isinstance(item, str) and item.strip():
                    found.append(item.strip())
                else:
                    walk(item)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item)

    walk(observed)
    return tuple(dict.fromkeys(found))


class _Guard:
    """The guard's state, shared by one page and the locators derived from it."""

    def __init__(self, probes: Sequence[str]) -> None:
        self.probes = tuple(probes)
        self.pages: list[str] = []
        self.clicks: list[str] = []
        self.refused: list[str] = []

    def before_click(self, spec: _Spec) -> None:
        """Raise :class:`FinalSaveClickError` unless ``spec`` is known and cannot pick a final-save control."""
        if not spec.known:
            self.refused.append(spec.describe())
            raise FinalSaveClickError(f"refused to click {spec.describe()}: the guard cannot tell it from a save")
        if spec.confirm_like():
            self.refused.append(spec.describe())
            raise FinalSaveClickError(
                f"refused to click {spec.describe()}: a name that starts with Confirm, Add, Claim or Here is a save"
            )
        for probe in self.probes:
            if spec.matches(probe):
                self.refused.append(spec.describe())
                raise FinalSaveClickError(f"refused to click {spec.describe()}: it can pick the save control {probe!r}")
        self.clicks.append(spec.describe())


class GuardedLocator:
    """A locator that refuses to click anything that could be a final save (:class:`FinalSaveClickError`), and passes
    every other action through. Locators derived from it keep what they were built from."""

    def __init__(self, inner: LocatorLike, guard: _Guard, spec: _Spec) -> None:
        self._inner = inner
        self._guard = guard
        self._spec = spec

    def __repr__(self) -> str:
        return f"GuardedLocator({self._spec.describe()})"

    def _derive(self, inner: LocatorLike, spec: _Spec | None = None) -> GuardedLocator:
        return GuardedLocator(inner, self._guard, self._spec if spec is None else spec)

    # --- actions ---

    def click(self, *, timeout: float | None = None) -> None:
        self._guard.before_click(self._spec)
        self._inner.click(timeout=timeout)

    def fill(self, value: str, *, timeout: float | None = None) -> None:
        self._inner.fill(value, timeout=timeout)

    def check(self, *, timeout: float | None = None) -> None:
        self._guard.before_click(self._spec)  # a checked box on a save control can submit as well
        self._inner.check(timeout=timeout)

    def uncheck(self, *, timeout: float | None = None) -> None:
        self._inner.uncheck(timeout=timeout)

    def select_option(
        self,
        value: str | Sequence[str] | None = None,
        *,
        label: str | Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> list[str]:
        self._guard.before_click(self._spec)
        return self._inner.select_option(value, label=label, timeout=timeout)

    def press(self, key: str, *, timeout: float | None = None) -> None:
        if _activates(key):  # Enter or Space on a focused save control is a click
            self._guard.before_click(self._spec)
        self._inner.press(key, timeout=timeout)

    # --- state ---

    def count(self) -> int:
        return self._inner.count()

    def is_visible(self) -> bool:
        return self._inner.is_visible()

    def is_enabled(self) -> bool:
        return self._inner.is_enabled()

    def is_checked(self) -> bool:
        return self._inner.is_checked()

    def inner_text(self, *, timeout: float | None = None) -> str:
        return self._inner.inner_text(timeout=timeout)

    def text_content(self, *, timeout: float | None = None) -> str | None:
        return self._inner.text_content(timeout=timeout)

    def wait_for(self, *, state: Any = None, timeout: float | None = None) -> None:
        self._inner.wait_for(state=state, timeout=timeout)

    # --- narrowing ---

    @property
    def first(self) -> GuardedLocator:
        return self._derive(self._inner.first)

    @property
    def last(self) -> GuardedLocator:
        return self._derive(self._inner.last)

    def nth(self, index: int) -> GuardedLocator:
        return self._derive(self._inner.nth(index))

    def filter(self, *, has_text: str | Pattern[str] | None = None) -> GuardedLocator:
        return self._derive(self._inner.filter(has_text=has_text))

    def all(self) -> list[GuardedLocator]:
        return [self._derive(item) for item in self._inner.all()]

    def get_by_role(
        self, role: Any, *, name: str | Pattern[str] | None = None, exact: bool | None = None
    ) -> GuardedLocator:
        return self._derive(self._inner.get_by_role(role, name=name, exact=exact), _Spec(str(role), name, None, exact))

    def get_by_text(self, text: str | Pattern[str], *, exact: bool | None = None) -> GuardedLocator:
        return self._derive(self._inner.get_by_text(text, exact=exact), _Spec(None, None, text, exact))

    def get_by_label(self, text: str | Pattern[str], *, exact: bool | None = None) -> GuardedLocator:
        return self._derive(self._inner.get_by_label(text, exact=exact), _UNKNOWN)

    def locator(self, selector: str, /) -> GuardedLocator:
        return self._derive(self._inner.locator(selector), _UNKNOWN)


class GuardedPage:
    """The page a drilled flow sees: every locator it builds is a :class:`GuardedLocator`, and the addresses it opens
    are recorded."""

    def __init__(self, inner: PageLike, guard: _Guard) -> None:
        self._inner = inner
        self._guard = guard

    @property
    def url(self) -> str:
        return self._inner.url

    def goto(self, url: str, *, timeout: float | None = None, wait_until: Any = None) -> object:
        self._guard.pages.append(url)
        return self._inner.goto(url, timeout=timeout, wait_until=wait_until)

    def get_by_role(
        self, role: Any, *, name: str | Pattern[str] | None = None, exact: bool | None = None
    ) -> GuardedLocator:
        found = self._inner.get_by_role(role, name=name, exact=exact)
        return GuardedLocator(found, self._guard, _Spec(str(role), name, None, exact))

    def get_by_text(self, text: str | Pattern[str], *, exact: bool | None = None) -> GuardedLocator:
        return GuardedLocator(self._inner.get_by_text(text, exact=exact), self._guard, _Spec(None, None, text, exact))

    def get_by_label(self, text: str | Pattern[str], *, exact: bool | None = None) -> GuardedLocator:
        return GuardedLocator(self._inner.get_by_label(text, exact=exact), self._guard, _UNKNOWN)

    def get_by_test_id(self, test_id: str | Pattern[str]) -> GuardedLocator:
        return GuardedLocator(self._inner.get_by_test_id(test_id), self._guard, _UNKNOWN)

    def locator(self, selector: str, /) -> GuardedLocator:
        return GuardedLocator(self._inner.locator(selector), self._guard, _UNKNOWN)

    def wait_for_load_state(self, state: Any = None, *, timeout: float | None = None) -> None:
        self._inner.wait_for_load_state(state, timeout=timeout)

    def screenshot(self, *, path: str | Path | None = None, full_page: bool | None = None) -> bytes:
        return self._inner.screenshot(path=path, full_page=full_page)

    def set_default_timeout(self, timeout: float) -> None:
        self._inner.set_default_timeout(timeout)

    def set_default_navigation_timeout(self, timeout: float) -> None:
        self._inner.set_default_navigation_timeout(timeout)

    def title(self) -> str:
        return self._inner.title()

    def close(self) -> None:
        self._inner.close()


# --- the driver: layer 2 ----------------------------------------------------------------------------------------------


class DrillUi:
    """The :class:`fm.browser.flows.UiDriver` a drilled flow gets. It is a dry-run :class:`fm.executor.UiSession` over
    a :class:`GuardedPage`, so screenshots and the confirm's wait for the control are the executor's own, with one more
    rule: the control handed to ``confirm`` can never be clicked, and ``confirm`` that comes back instead of raising
    :class:`fm.browser.flows.DryRunStop` is a :class:`DrillSafetyError`."""

    def __init__(self, page: PageLike, audit: AuditLog, probes: Sequence[str], *, prefix: str = "drill") -> None:
        self.guard = _Guard(probes)
        self._page = GuardedPage(page, self.guard)
        self.session = UiSession(self._page, audit, dry_run=True, prefix=prefix)
        self.reached: list[str] = []
        """The confirms the walk got to, in order: the saves it did not make."""

    @property
    def page(self) -> PageLike:
        return self._page

    @property
    def dry_run(self) -> bool:
        return True

    def screenshot(self, name: str) -> None:
        self.session.screenshot(name)

    def confirm(self, target: LocatorLike, *, what: str) -> None:
        self.reached.append(what)
        self.session.confirm(GuardedLocator(target, self.guard, _UNKNOWN), what=what)  # waits for it; never clicks
        raise DrillSafetyError(f"confirm({what!r}) returned instead of stopping the drill before the save")


# --- the request watcher: layer 4 -------------------------------------------------------------------------------------


class RequestWatch:
    """Listens to a page's requests (when the page has an ``on`` hook, as Playwright's has) and records what the dry run
    guard aborts. A request to ESPN's write host, or a non-GET to a ``/transactions`` path, is a save attempt."""

    def __init__(self) -> None:
        self.blocked: list[str] = []
        """Every request the dry-run rule aborts (:func:`fm.executor.dry_run_block_reason`): ESPN's own beacons too."""
        self.save_attempts: list[str] = []

    def record(self, request: Any) -> None:
        method = str(getattr(request, "method", "") or "").upper()
        url = str(getattr(request, "url", "") or "")
        if dry_run_block_reason(method, url) is None:
            return
        self.blocked.append(f"{method} {url}")
        parts = urlsplit(url)
        host = (parts.hostname or "").lower().rstrip(".")
        if host == WRITES_HOST or (method != "GET" and "/transactions" in parts.path):
            self.save_attempts.append(f"{method} {url}")

    def attach(self, page: Any) -> None:
        hook = getattr(page, "on", None)
        if callable(hook):
            hook("request", self.record)


# --- running a flow's drill -------------------------------------------------------------------------------------------


def unsaved_league_row(league: League, at: datetime) -> LeagueRow:
    """An unsaved row with the fields a flow reads; the drill does not use the store."""
    return LeagueRow(
        key=league.key,
        sport=league.sport,
        espn_league_id=league.espn_league_id,
        season=league.season,
        team_id=league.team_id,
        as_of=at,
    )


def _proposal_row(flow: Flow[Any], plan: DrillPlan, at: datetime) -> ProposalRow:
    """The proposal a flow's context holds. It is never stored or proposed: flows read only its scoring period."""
    return ProposalRow(
        league_id=0,
        kind=flow.kinds[0].value,
        policy="off",
        scoring_period_id=plan.scoring_period_id,
        payload=jsonable(plan.payload.model_dump(mode="json")),
        created_by=DRILL_ACTOR,
        created_at=at,
    )


def _short(error: BaseException, limit: int = 300) -> str:
    text = " ".join(str(error).split()) or type(error).__name__
    text = f"{type(error).__name__}: {text}" if not isinstance(error, DrillSafetyError) else text
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _leave(page: PageLike) -> None:
    """Navigate away (which drops a half-made move) and close the page. Best effort: a page that will not leave is
    closed anyway, and nothing here is evidence."""
    try:
        page.goto(BLANK_PAGE)
    except Exception as exc:
        logger.warning("drill: could not navigate away from the walk's page: %s", exc)
    try:
        page.close()
    except Exception as exc:
        logger.warning("drill: could not close the walk's page: %s", exc)


def _drill_flow(
    flow: Flow[Any],
    plan: DrillPlan,
    league: League,
    row: LeagueRow,
    runtime: Runtime,
    audit: AuditLog,
    at: datetime,
) -> tuple[DrillResult, tuple[DrillFinding, ...]]:
    """Run one flow's UI walk on a fresh page, dry. Never raises for what the walk or a safety layer did."""
    try:
        ctx = FlowContext(
            proposal=_proposal_row(flow, plan, at),
            league=row,
            kind=flow.kinds[0],
            payload=plan.payload,
            reader=runtime.reader,
            now=at,
            dry_run=True,
            member_id=runtime.member_id,
        )
        pre: FlowPreconditions = flow.check(ctx)
    except Exception as exc:  # an unreadable league (EspnClientError) or a flow that cannot check: not a walk to judge
        detail = f"the flow's preconditions could not be read: {_short(exc)}"
        finding = DrillFinding(DrillFailureKind.SETUP_FAILED, flow.name, detail)
        return DrillResult(flow.name, DrillStatus.FAILED, plan.description), (finding,)
    try:
        page = runtime.browser.new_page()
        timeout_ms = DEFAULT_UI_TIMEOUT_S * 1000
        page.set_default_timeout(timeout_ms)
        page.set_default_navigation_timeout(timeout_ms)
    except Exception as exc:
        finding = DrillFinding(DrillFailureKind.UI_FAILED, flow.name, f"could not open a page: {_short(exc)}")
        return DrillResult(flow.name, DrillStatus.FAILED, plan.description), (finding,)
    watch = RequestWatch()
    watch.attach(page)
    names = tuple(dict.fromkeys([*plan.names, *observed_names(pre.observed)]))[:MAX_PROBE_NAMES]
    ui = DrillUi(page, audit, final_save_probes(names), prefix=f"drill-{flow.name}")
    findings: list[DrillFinding] = []
    stopped = False
    try:
        flow.run_ui(ctx, ui, pre)
    except DryRunStop:
        stopped = True
    except DrillSafetyError as exc:
        findings.append(DrillFinding(DrillFailureKind.UNSAFE, flow.name, _short(exc)))
    except Exception as exc:
        logger.warning("drill: %s's UI walk failed: %s", flow.name, exc)
        findings.append(DrillFinding(DrillFailureKind.UI_FAILED, flow.name, _walk_failure(exc, pre)))
    else:
        findings.append(
            DrillFinding(DrillFailureKind.NO_CONFIRM, flow.name, "the walk ended without reaching a confirm")
        )
    finally:
        try:
            if findings:
                ui.screenshot("failed")
        except Exception as exc:  # a screenshot that fails must not stop the walk's page from being left
            logger.warning("drill: could not take the failure screenshot: %s", exc)
        _leave(page)
    if watch.save_attempts:
        detail = f"the page sent a write ({'; '.join(watch.save_attempts)}); the dry-run guard aborted it"
        findings.append(DrillFinding(DrillFailureKind.UNSAFE, flow.name, detail))
    last_page = ui.guard.pages[-1] if ui.guard.pages else None
    findings = [finding if finding.url else _at(finding, last_page) for finding in findings]
    result = DrillResult(
        flow.name,
        DrillStatus.PASSED if stopped and not findings else DrillStatus.FAILED,
        plan.description,
        reached=ui.reached[0] if ui.reached else None,
        pages=tuple(ui.guard.pages),
        clicks=tuple(ui.guard.clicks),
        notes=pre.failures,
        artifacts=tuple(ui.session.artifacts),
    )
    return result, tuple(findings)


def _at(finding: DrillFinding, url: str | None) -> DrillFinding:
    return DrillFinding(finding.kind, finding.flow, finding.detail, url)


def _walk_failure(error: Exception, pre: FlowPreconditions) -> str:
    """Why the walk failed, with what the preconditions said when they had a view on it."""
    detail = f"the walk failed before its confirm: {_short(error)}"
    if pre.failures:
        detail += f" (preconditions: {'; '.join(pre.failures)})"
    return detail


# --- the run and its report -------------------------------------------------------------------------------------------


def drillable_flows(registry: FlowRegistry, sport: str) -> tuple[Flow[Any], ...]:
    """Every flow in ``registry`` with a UI mode that serves ``sport``, once each, in registration order."""
    found: list[Flow[Any]] = []
    for registration in registry.registered():
        flow = registration.flow
        if registration.sport == sport and Mode.UI in flow.modes and flow not in found:
            found.append(flow)
    return tuple(found)


def read_views(league: League, reader: EspnClient) -> DrillViews:
    """Our roster, the league's settings and the top of the player pool, read through ``reader`` (reads only).
    Raises ``EspnClientError`` when ESPN cannot be read or answers with something unparseable."""
    game = Game.coerce(league.game)
    rosters = reader.rosters().data
    try:
        roster = rosters.roster(league.team_id)
    except KeyError as exc:
        raise EspnClientError(f"ESPN's rosters have no team {league.team_id} (our team_id)") from exc
    settings = reader.settings().data
    pool = reader.free_agents(limit=POOL_PROBE).data
    return DrillViews(game, settings, roster, pool)


def run_drills(
    league: League,
    *,
    runtime: Runtime,
    flows: Sequence[str] | None = None,
    registry: FlowRegistry | None = None,
    planners: Mapping[str, Planner] | None = None,
    audit_root: Path | None = None,
    now: datetime | None = None,
) -> DrillReport:
    """Drill every UI flow of ``league``'s sport (or just ``flows``, by name) on ``runtime``'s browser.

    ``runtime`` must be a dry run's (opened with ``dry_run=True``, so ``Runtime.dry_run`` is set, and its transport a
    :class:`fm.executor.RefusingTransport`), or :class:`DrillError`.
    ``registry`` defaults to the process-wide flows and ``planners`` to :data:`PLANNERS`; tests pass their own.
    Screenshots go under ``audit_root`` (default: ``drills/`` in the audit folder).
    """
    if not runtime.dry_run or not isinstance(runtime.transport, RefusingTransport):
        raise DrillError(
            "the drill needs a dry-run runtime (Runtime.dry_run set, a transport that refuses every send): "
            "open it with dry_run=True"
        )
    chosen = registry
    if chosen is None:
        discover()
        chosen = flow_registry
    available = drillable_flows(chosen, league.sport)
    if flows:
        unknown = sorted(set(flows) - {flow.name for flow in available})
        if unknown:
            known = ", ".join(flow.name for flow in available) or "none"
            raise DrillError(f"no UI flow {', '.join(unknown)} for {league.sport}; with a UI mode: {known}")
        available = tuple(flow for flow in available if flow.name in flows)
    at = now if now is not None else datetime.now(UTC)
    row = unsaved_league_row(league, at)
    plan_by = PLANNERS if planners is None else planners
    try:
        views = read_views(league, runtime.reader)
    except EspnClientError as exc:
        finding = DrillFinding(DrillFailureKind.SETUP_FAILED, "*", f"the views could not be read: {_short(exc)}")
        return DrillReport(league.key, (), (finding,))
    root = audit_root if audit_root is not None else paths.audit_dir() / "drills"
    results: list[DrillResult] = []
    findings: list[DrillFinding] = []
    for flow in available:
        planner = plan_by.get(flow.name)
        if planner is None:
            detail = (
                f"{flow.name} has a UI mode and the drill has no planner for it, so its fallback is not "
                "drilled (add one to fm.browser.drills.PLANNERS)"
            )
            findings.append(DrillFinding(DrillFailureKind.NO_PLANNER, flow.name, detail))
            results.append(DrillResult(flow.name, DrillStatus.FAILED))
            continue
        try:
            plan = planner(views)
        except NoDrillTargetError as exc:
            results.append(DrillResult(flow.name, DrillStatus.SKIPPED, skipped=str(exc)))
            continue
        folder = root / at.strftime("%Y-%m-%d") / f"{league.key}-{flow.name}-{at:%H%M%S}"
        folder.mkdir(parents=True, exist_ok=True)
        result, found = _drill_flow(flow, plan, league, row, runtime, AuditLog(folder, root), at)
        results.append(result)
        findings.extend(found)
    if results and all(result.status is DrillStatus.SKIPPED for result in results):
        reasons = "; ".join(f"{result.flow}: {result.skipped}" for result in results)
        findings.append(
            DrillFinding(
                DrillFailureKind.NOT_DRILLED, "*", f"every flow was skipped, so nothing was checked ({reasons})"
            )
        )
    return DrillReport(league.key, tuple(results), tuple(findings))


def drill_payload(reports: Sequence[DrillReport]) -> dict[str, Any]:
    """The structured drill report: per league, every finding and what each flow's drill did. JSON-able."""
    return {
        "ok": all(report.ok for report in reports),
        "unsafe": any(report.unsafe for report in reports),
        "leagues": [report.as_dict() for report in reports],
    }


def drill_alert(reports: Sequence[DrillReport]) -> Message | None:
    """The alert for ``fm.notify.send_alert`` (``None`` when every report is clean): one message for the whole run.

    A save attempt leads the title. The link opens the last page of the first failed walk, where the fix starts.
    """
    bad = [report for report in reports if not report.ok]
    if not bad:
        return None
    count = sum(len(report.findings) for report in bad)
    unsafe = sum(len(report.unsafe) for report in bad)
    if unsafe:
        title = f"UI drill SAFETY: {unsafe} save attempt{'s' if unsafe != 1 else ''} caught"
    else:
        title = f"UI fallback drill failed: {count} finding{'s' if count != 1 else ''}"
    lines = [f"{report.league}: {finding.line()}" for report in bad for finding in report.findings]
    if len(lines) > ALERT_LINES:
        lines = [*lines[:ALERT_LINES], f"... and {len(lines) - ALERT_LINES} more (fm drill lists them all)"]
    link = next((finding.url for report in bad for finding in report.findings if finding.url), None)
    return alert(title, "\n".join(lines), link=link)


def could_not_run_alert(reason: str) -> Message:
    """The alert for a drill that could not start (the session expired, the browser would not open)."""
    return alert("UI fallback drill could not run", " ".join(reason.split()))
