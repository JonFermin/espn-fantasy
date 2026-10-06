"""What ESPN's web pages look like to the UI-mode flows: page addresses, slot labels and role/text selectors.

CLAUDE.md: selectors live only here, and flows use role and text locators. A flow never spells out a role, a button
label or a URL. It asks this module for a :class:`Selector` and calls :meth:`Selector.locate` on the page or on a
row. The helpers below cover the lookups that take a value: :func:`player_row` finds a player's row and
:func:`empty_slot_row` an open slot's. Every selector is registered with the page it belongs to, so the read-only
canary (ROADMAP #28) can assert that each one still resolves. :class:`Presence` says what to expect of it on a healthy
page, ``within`` names the selector a scoped one is looked up inside, and ``after`` names the control whose click
reveals it.

Status (ROADMAP #14): the team-page address and the "Log in Required" heading are what the real-league capture saw.
The roster controls (``MOVE`` on a player's row, then ``HERE`` on the row he goes to, which saves the move at once)
follow ESPN's long-standing lineup editor but have not been driven yet: that needs the web sign-in the guarded UI
capture is waiting for. Until it lands, API mode is the write path (docs/espn-api.md section 4), the UI fallback
uses what is here, and the canary is the drift alarm. Add/drop (#27) and trades (#44) add their pages to
:class:`WebPage` and register their selectors in this module.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from re import Pattern
from types import MappingProxyType

from fm.browser.flows import LocatorLike, PageLike
from fm.espn.ids import Game, ids_for

WEB_ROOT = "https://fantasy.espn.com"
"""The web app's origin (the API hosts are ``lm-api-reads`` and ``lm-api-writes``)."""
_SPORT_PATHS: Mapping[Game, str] = MappingProxyType({Game.FFL: "football", Game.FBA: "basketball"})


class WebPage(StrEnum):
    """The ESPN pages UI mode works on."""

    ROSTER = "roster"  # the team page: our roster by lineup slot, where lineup moves are made


def team_page_url(
    game: Game | str, league_id: int, team_id: int, season: int, scoring_period_id: int | None = None
) -> str:
    """Our team page (``fantasy.espn.com/{football|basketball}/team?leagueId=...&teamId=...&seasonId=...``), showing
    the lineup of ``scoring_period_id`` when one is given (a week in NFL, a day in NBA)."""
    url = f"{WEB_ROOT}/{_SPORT_PATHS[Game.coerce(game)]}/team?leagueId={league_id}&teamId={team_id}&seasonId={season}"
    if scoring_period_id:
        url += f"&scoringPeriodId={scoring_period_id}"
    return url


_SLOT_LABELS: Mapping[Game, Mapping[int, str]] = MappingProxyType(
    {
        Game.FFL: MappingProxyType({20: "Bench", 23: "FLEX"}),
        Game.FBA: MappingProxyType({12: "Bench"}),
    }
)
"""Where the roster page labels a slot differently from ``fm.espn.ids`` (``BE`` is "Bench", ``RB/WR/TE`` is "FLEX")."""


def slot_label(game: Game | str, slot_id: int) -> str:
    """How the roster page labels a lineup slot in its slot column (``QB``, ``FLEX``, ``UTIL``, ``Bench``, ``IR``)."""
    resolved = Game.coerce(game)
    return _SLOT_LABELS[resolved].get(slot_id) or ids_for(resolved).slot_label(slot_id)


def _whole_word(text: str) -> Pattern[str]:
    """A control whose accessible name is ``text`` and nothing else, in any case: ``MOVE`` or ``Move``, never
    ``Remove``."""
    return re.compile(rf"^\s*{re.escape(text)}\s*$", re.IGNORECASE)


class Presence(StrEnum):
    """When a loaded, signed-in page shows a selector: what the canary may expect of it."""

    ALWAYS = "always"  # every healthy page: missing means ESPN's page changed
    SOMETIMES = "sometimes"  # only in some states (an open slot, an unlocked player, after a click)
    NEVER = "never"  # a warning marker: a healthy page does not show it (Log in Required)


@dataclass(frozen=True, slots=True)
class Selector:
    """A role or text locator on one ESPN page. ``role`` set: ``get_by_role(role, name=name, exact=exact)``; otherwise
    ``get_by_text(text, exact=exact)``."""

    key: str
    """``page.thing``: a stable id for the canary and for error messages."""
    page: WebPage
    description: str
    role: str | None = None
    name: str | Pattern[str] | None = None
    text: str | Pattern[str] | None = None
    exact: bool | None = None
    presence: Presence = Presence.ALWAYS
    within: str | None = None
    """The key of the selector this one is looked up inside (a control in a row)."""
    after: str | None = None
    """The key of the control whose click makes this one appear."""

    def __post_init__(self) -> None:
        if (self.role is None) == (self.text is None):
            raise ValueError(f"selector {self.key} needs a role or a text, not both")

    def locate(self, scope: PageLike | LocatorLike) -> LocatorLike:
        """The Playwright locator for this selector on ``scope``: a page, or a locator such as a row."""
        if self.role is not None:
            return scope.get_by_role(self.role, name=self.name, exact=self.exact)
        assert self.text is not None  # __post_init__ guarantees one of the two
        return scope.get_by_text(self.text, exact=self.exact)


_REGISTRY: dict[str, Selector] = {}


def _register(selector: Selector) -> Selector:
    if selector.key in _REGISTRY:
        raise ValueError(f"selector {selector.key} is registered twice")
    for reference in (selector.within, selector.after):
        if reference is not None and reference not in _REGISTRY:
            raise ValueError(f"selector {selector.key} refers to {reference}, which is not registered (yet)")
    _REGISTRY[selector.key] = selector
    return selector


def registered_selectors() -> tuple[Selector, ...]:
    """Every selector, in registration order (named like :func:`fm.browser.flows.registered_flows`, so the name is
    not :func:`fm.decide.registry.registered`)."""
    return tuple(_REGISTRY.values())


def selectors_for(page: WebPage) -> tuple[Selector, ...]:
    """The selectors of one page, in registration order."""
    return tuple(selector for selector in _REGISTRY.values() if selector.page is page)


def selector(key: str) -> Selector:
    """A selector by key; ``KeyError`` names the known keys."""
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"no selector {key!r}; known: {', '.join(_REGISTRY)}") from None


# --- the roster (team) page -------------------------------------------------------------------------------------------

LOGIN_REQUIRED = _register(
    Selector(
        "roster.login_required",
        WebPage.ROSTER,
        "the heading ESPN shows instead of the team when the profile has no web sign-in (the API cookies may still "
        "work); seen by the #14 capture",
        role="heading",
        name="Log in Required",
        presence=Presence.NEVER,
    )
)
ROSTER_TABLE = _register(
    Selector("roster.table", WebPage.ROSTER, "the roster table: one row per lineup slot", role="table")
)
ROSTER_ROW = _register(
    Selector(
        "roster.row",
        WebPage.ROSTER,
        "a roster row: the slot label, the player (or Empty) and the move control; filtered by name in player_row",
        role="row",
    )
)
_SLOT_CELL_ROLE = "cell"
SLOT_CELL = _register(
    Selector(
        "roster.slot_cell",
        WebPage.ROSTER,
        "the slot column of a row, named by slot_label; slot_cell matches the label exactly",
        role=_SLOT_CELL_ROLE,
        within="roster.row",
    )
)
EMPTY_SLOT = _register(
    Selector(
        "roster.empty_slot",
        WebPage.ROSTER,
        "the player column of a slot nobody fills",
        text="Empty",
        exact=True,
        presence=Presence.SOMETIMES,
        within="roster.row",
    )
)
MOVE_BUTTON = _register(
    Selector(
        "roster.move",
        WebPage.ROSTER,
        "MOVE on an unlocked player's row: picks him up and marks the rows he may go to (changes nothing by itself)",
        role="button",
        name=_whole_word("move"),
        presence=Presence.SOMETIMES,
        within="roster.row",
    )
)
HERE_BUTTON = _register(
    Selector(
        "roster.here",
        WebPage.ROSTER,
        "HERE on a destination row after MOVE: puts the player there, swapping with whoever holds it; saves at once",
        role="button",
        name=_whole_word("here"),
        presence=Presence.SOMETIMES,
        within="roster.row",
        after="roster.move",
    )
)


def player_row(scope: PageLike | LocatorLike, player_name: str) -> LocatorLike:
    """The roster row showing ``player_name`` (a case-insensitive text match, as Playwright's ``has_text`` does)."""
    return ROSTER_ROW.locate(scope).filter(has_text=player_name)


def slot_cell(row: LocatorLike, label: str) -> LocatorLike:
    """The slot column of ``row`` when it reads exactly ``label`` (:func:`slot_label`): ``RB`` never matches
    ``RB/WR``."""
    return row.get_by_role(_SLOT_CELL_ROLE, name=label, exact=True)


def empty_slot_row(scope: PageLike | LocatorLike, label: str) -> LocatorLike | None:
    """The first row of slot ``label`` that nobody fills (its player column reads Empty), or ``None``."""
    for row in ROSTER_ROW.locate(scope).filter(has_text=EMPTY_SLOT.text).all():
        if slot_cell(row, label).count() and EMPTY_SLOT.locate(row).count():
            return row
    return None
