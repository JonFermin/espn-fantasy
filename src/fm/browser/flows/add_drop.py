"""``add_drop``: a free-agent add, a drop, or both in one transaction, in API mode with a UI click-through fallback
(DESIGN 6.3, docs/espn-api.md section 4 "add_drop").

It serves the ``add_drop`` proposals of both sports. Their payload is a :class:`fm.proposals.AddDropPayload`: the
free agent to add, the player to drop, or both. The executor (:func:`fm.executor.execute`) drives the steps below and
owns the write-safety rules around them. ``fm.decide.waivers`` already times its adds to the league's roster lock and
skips a free agent whose game has started, so these preconditions are the backstop, not the first refusal.

Preconditions, read through the API (``mSettings``, ``mRoster``, ``proTeamSchedules_wl`` and the add's
``kona_playercard``). Every failure is listed at once:

- The move is for ESPN's current scoring period (``status.latestScoringPeriod``): the web app sends an add for the
  current period, and which day a later one would land on is unverified.
- Adds and drops close at the league's roster lock, ``plugin.transaction_cutoff(pro_team_id, period, schedule,
  lock_type=settings.roster_lock_type)`` per player moved: the day's first tip under the real NBA league's
  ``FIRSTGAME_SCORINGPERIOD``, each player's own kickoff under the NFL league's ``INDIVIDUAL_GAME``. An ``UNKNOWN``
  roster lock type (ESPN's weekly types parse as one) is a refusal, never a guess. ESPN's own ``rosterLocked`` flag
  refuses too.
- The add is a free agent now: a player on waivers is claimed, not added (``claim_waiver``), and a rostered one is
  nobody's to add.
- The drop is on our roster for the period and not on ESPN's undroppable list.
- Afterwards the roster holds no more players than the league's active and bench slots (IR is extra): a full roster
  needs the drop in the same proposal.

API mode sends the web client's ``FREEAGENT`` envelope (``ROSTER`` for a bare drop) with ``ADD``/``DROP`` items,
built by :mod:`fm.browser.transactions`; an add without a drop goes without ``memberId``, as the player list's
one-click Add did in the capture (``fba/write_FREEAGENT_1.json``), and an add with a drop carries it, as the roster-fix
page's request did (``fba/write_FREEAGENT_2.json``). UI mode follows the same two paths: with roster room, the player
list's ``Add <Name> ...`` button is the one state-changing click (so it is the
:meth:`fm.browser.flows.UiDriver.confirm`); with a drop, the roster-fix page's ``Drop Player <Name>``, then
``Continue to add ... and drop ...``, then the
"Confirm Transaction" dialog's ``Confirm add ... and drop ...``, which is the confirm. A bare drop has no captured
click-through and is left to API mode. Before the first click the league is read again, and the walk stops if the add
or the drop is no longer where the preconditions found him. The UI comes after a rejected API request only when
:func:`fm.browser.transactions.ui_may_follow` allows it. Verification re-reads ``mRoster``: the add is on our roster
and the drop is not.

:mod:`fm.browser.flows.waiver` (claims and their cancellation) shares the league read, the player facts, the cutoff
rule and the roster-fix walk defined here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fm.browser import selectors
from fm.browser.flows import (
    Flow,
    FlowContext,
    Mode,
    ModeUnavailableError,
    PageLike,
    Preconditions,
    UiDriver,
    Verification,
    WriteRequest,
    WriteResponse,
    register_flow,
)
from fm.browser.selectors import RosterFixType
from fm.browser.transactions import Envelope, TransactionType, add_item, drop_item, transaction_request, ui_may_follow
from fm.espn.client import EspnSchemaError
from fm.espn.ids import Game
from fm.espn.models import POOL_FREE_AGENT, POOL_WAIVERS, PoolEntry, ProSchedule, RostersView, TeamRoster
from fm.espn.settings import LeagueSettings, LockType, SlotKind
from fm.proposals import AddDropPayload, Payload, ProposalKind
from fm.sports.base import FREE_AGENT_TEAM, plugin_for

SIGNED_OUT_PAGE = (
    "says Log in Required: the browser profile has the API cookies but no ESPN web sign-in, so the click-through "
    "cannot run (sign in on fantasy.espn.com in the fm browser profile; API mode does not need it)"
)
SAVE_FAILED = "ESPN's page reported that the save failed (Oops! Looks like something went wrong)"
"""What the UI walk raises after its confirm when the page shows the failure text; the executor then re-reads."""


# --- what a precondition read finds -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeagueRead:
    """One precondition read of the league: its settings, the period's rosters, the pro schedule, ESPN's current
    scoring period and the one the proposal is for."""

    settings: LeagueSettings
    rosters: RostersView
    schedule: ProSchedule
    latest: int
    target: int

    @property
    def unmapped_lock(self) -> str | None:
        """Why no cutoff can be computed: the roster lock type is one the plugins refuse to read; else ``None``."""
        if self.settings.roster_lock_type is not LockType.UNKNOWN:
            return None
        raw = self.settings.roster_lock_type_raw or LockType.UNKNOWN.value
        return (
            f"the league's roster lock type {raw} is not mapped, so when adds and drops close is unknown; "
            "the move is refused rather than timed by a guess"
        )

    @property
    def lock_rule(self) -> str:
        """How the league's roster lock reads, for messages."""
        if self.settings.roster_lock_type is LockType.INDIVIDUAL_GAME:
            return "adds and drops close at each player's own game"
        return "adds and drops close for everyone at the period's first game"

    def our_roster(self, team_id: int) -> TeamRoster | None:
        try:
            return self.rosters.roster(team_id)
        except KeyError:
            return None

    def ir_slots(self) -> frozenset[int]:
        return frozenset(slot.slot_id for slot in self.settings.lineup_slots if slot.kind is SlotKind.IR)

    def roster_count(self, roster: TeamRoster) -> int:
        """Players on the roster outside IR: what counts against the league's roster size."""
        ir = self.ir_slots()
        return sum(1 for entry in roster.entries if entry.lineup_slot_id not in ir)


@dataclass(frozen=True, slots=True)
class PlayerFacts:
    """One player as the precondition read found him: the pool entry of an add, the roster entry of a drop."""

    espn_id: int
    name: str
    status: str | None
    """ESPN's pool status: ``FREEAGENT``, ``WAIVERS`` or ``ONTEAM``."""
    on_team_id: int | None
    pro_team_id: int | None
    roster_locked: bool
    droppable: bool | None
    slot: int | None
    """His lineup slot when he is on our roster."""
    cutoff: datetime | None
    """When adds and drops of him close in the period under the league's roster lock; ``None`` when they do not
    (a team without a game under per-game locking), or when the lock type is unmapped."""
    waiver_process_date: datetime | None

    @property
    def label(self) -> str:
        return f"{self.name} ({self.espn_id})" if self.name else f"player {self.espn_id}"

    def to_json(self) -> dict[str, Any]:
        return {
            "espn_id": self.espn_id,
            "name": self.name,
            "status": self.status,
            "on_team_id": self.on_team_id,
            "pro_team_id": self.pro_team_id,
            "roster_locked": self.roster_locked,
            "droppable": self.droppable,
            "slot": self.slot,
            "cutoff": None if self.cutoff is None else self.cutoff.isoformat(),
            "waiver_process_date": None if self.waiver_process_date is None else self.waiver_process_date.isoformat(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PlayerFacts:
        return cls(
            espn_id=int(data["espn_id"]),
            name=str(data["name"]),
            status=None if data["status"] is None else str(data["status"]),
            on_team_id=None if data["on_team_id"] is None else int(data["on_team_id"]),
            pro_team_id=None if data["pro_team_id"] is None else int(data["pro_team_id"]),
            roster_locked=bool(data["roster_locked"]),
            droppable=None if data["droppable"] is None else bool(data["droppable"]),
            slot=None if data["slot"] is None else int(data["slot"]),
            cutoff=_instant(data["cutoff"]),
            waiver_process_date=_instant(data["waiver_process_date"]),
        )


@dataclass(frozen=True, slots=True)
class MoveFacts:
    """What the preconditions found about an add and/or a drop, kept in ``Preconditions.observed`` for the later steps
    of the same run."""

    scoring_period_id: int
    latest_scoring_period: int
    roster_lock_type: str
    roster_count: int
    roster_size: int
    add: PlayerFacts | None
    drop: PlayerFacts | None

    def to_json(self) -> dict[str, Any]:
        return {
            "scoring_period_id": self.scoring_period_id,
            "latest_scoring_period": self.latest_scoring_period,
            "roster_lock_type": self.roster_lock_type,
            "roster_count": self.roster_count,
            "roster_size": self.roster_size,
            "add": None if self.add is None else self.add.to_json(),
            "drop": None if self.drop is None else self.drop.to_json(),
        }

    @classmethod
    def from_observed(cls, observed: Mapping[str, Any]) -> MoveFacts:
        """Read :meth:`to_json` back. Raises ``ValueError`` when the preconditions did not get that far."""
        try:
            return cls(
                scoring_period_id=int(observed["scoring_period_id"]),
                latest_scoring_period=int(observed["latest_scoring_period"]),
                roster_lock_type=str(observed["roster_lock_type"]),
                roster_count=int(observed["roster_count"]),
                roster_size=int(observed["roster_size"]),
                add=None if observed["add"] is None else PlayerFacts.from_json(observed["add"]),
                drop=None if observed["drop"] is None else PlayerFacts.from_json(observed["drop"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"the preconditions are incomplete ({type(exc).__name__}: {exc})") from exc


def _instant(value: Any) -> datetime | None:
    return None if value is None else datetime.fromisoformat(str(value))


def stamp(at: datetime) -> str:
    """``2026-10-20 23:00 UTC`` for messages."""
    return at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


# --- reads ------------------------------------------------------------------------------------------------------------


def require_period(view: RostersView, scoring_period_id: int) -> None:
    """ESPN echoes the period a roster read asked for (docs/espn-api.md section 1 #16); an answer for another period
    is unusable (a refusal before the write, an unreadable re-read after it), never a verdict on the move."""
    if view.scoring_period_id is not None and view.scoring_period_id != scoring_period_id:
        raise EspnSchemaError(
            f"asked for scoring period {scoring_period_id}'s rosters, ESPN answered with scoring period "
            f"{view.scoring_period_id}'s"
        )


def read_league[P: Payload](ctx: FlowContext[P]) -> LeagueRead:
    """Settings, the proposal's period's rosters (ESPN's current period without one), the pro schedule."""
    view = ctx.reader.rosters(ctx.scoring_period_id).data
    settings = ctx.reader.settings().data
    latest: int | None = None
    if view.status is not None and view.status.latest_scoring_period:
        latest = view.status.latest_scoring_period
    elif settings.current_scoring_period:
        latest = settings.current_scoring_period
    if latest is None:  # a read problem, not the league's state: refuse, and leave the token unspent
        raise EspnSchemaError("neither mRoster's status nor mSettings says which scoring period is current")
    target = ctx.scoring_period_id or latest
    require_period(view, target)
    schedule = ctx.reader.pro_schedule().data
    return LeagueRead(settings=settings, rosters=view, schedule=schedule, latest=latest, target=target)


def pool_entry[P: Payload](ctx: FlowContext[P], player_id: int) -> PoolEntry | None:
    """The player's ``kona_playercard`` entry (his pool status, team, lock flags), or ``None`` when ESPN has none."""
    try:
        return ctx.reader.player_cards([player_id]).data.entry(player_id)
    except KeyError:
        return None


def cutoff_for(read: LeagueRead, game: Game, pro_team_id: int | None) -> datetime | None:
    """When adds and drops of a player on ``pro_team_id`` close in the period under the league's roster lock; ``None``
    when the lock type is unmapped (:attr:`LeagueRead.unmapped_lock` is the failure for that) or nothing locks him."""
    if read.unmapped_lock is not None:
        return None
    team = pro_team_id if pro_team_id is not None else FREE_AGENT_TEAM
    return plugin_for(game).transaction_cutoff(
        team, read.target, read.schedule, lock_type=read.settings.roster_lock_type
    )


def player_facts(entry: PoolEntry, read: LeagueRead, game: Game, *, slot: int | None = None) -> PlayerFacts:
    return PlayerFacts(
        espn_id=entry.id,
        name=entry.player.full_name,
        status=entry.status,
        on_team_id=entry.rostered_team_id,
        pro_team_id=entry.player.pro_team_id,
        roster_locked=entry.roster_locked,
        droppable=entry.player.droppable,
        slot=slot,
        cutoff=cutoff_for(read, game, entry.player.pro_team_id),
        waiver_process_date=entry.waiver_process_date,
    )


def drop_facts(roster: TeamRoster, read: LeagueRead, game: Game, player_id: int) -> PlayerFacts | None:
    """The drop as our roster shows him, or ``None`` when he is not on it."""
    try:
        entry = roster.entry(player_id)
    except KeyError:
        return None
    return player_facts(entry.player_pool_entry, read, game, slot=entry.lineup_slot_id)


# --- preconditions ----------------------------------------------------------------------------------------------------


def closed(player: PlayerFacts, what: str, read: LeagueRead, now: datetime) -> str | None:
    """Why ``what`` ("adding", "dropping") the player is closed now, or ``None``: the league's cutoff has passed, or
    ESPN flags him roster-locked."""
    if player.cutoff is not None and now >= player.cutoff:
        return f"{what} {player.label} closed at {stamp(player.cutoff)}: {read.lock_rule}"
    if player.roster_locked:
        return f"ESPN marks {player.label} roster-locked: his game has started"
    return None


def add_failures(add: PlayerFacts, team_id: int, read: LeagueRead, now: datetime) -> list[str]:
    """Every reason the free agent cannot be added now."""
    failures: list[str] = []
    if add.status == POOL_WAIVERS:
        until = "" if add.waiver_process_date is None else f" until {stamp(add.waiver_process_date)}"
        failures.append(f"{add.label} is on waivers{until}: claim him with a waiver proposal instead of adding him")
    elif add.on_team_id is not None:
        whose = "already on our team" if add.on_team_id == team_id else f"on team {add.on_team_id}"
        failures.append(f"{add.label} is {whose}, not a free agent")
    elif add.status != POOL_FREE_AGENT:
        failures.append(f"{add.label} is not a free agent (ESPN status {add.status!r})")
    problem = closed(add, "adding", read, now)
    if problem is not None:
        failures.append(problem)
    return failures


def drop_failures(drop: PlayerFacts, read: LeagueRead, now: datetime, *, locks_apply: bool = True) -> list[str]:
    """Every reason the player cannot be dropped: ESPN's undroppable list, and (when the drop happens in the current
    period) the roster lock."""
    failures: list[str] = []
    if drop.droppable is False:
        failures.append(f"{drop.label} is on ESPN's undroppable list")
    if locks_apply:
        problem = closed(drop, "dropping", read, now)
        if problem is not None:
            failures.append(problem)
    return failures


def not_on_roster(player_id: int, team_id: int, period: int) -> str:
    return f"player {player_id} is not on team {team_id} in scoring period {period}"


def room_failure(facts: MoveFacts) -> str | None:
    """The roster would be over the league's size after the move: it needs a drop."""
    after = facts.roster_count + (1 if facts.add is not None else 0) - (1 if facts.drop is not None else 0)
    if after > facts.roster_size:
        return (
            f"the roster would hold {after} players and the league allows {facts.roster_size}: "
            "add a drop to the proposal"
        )
    return None


# --- UI mode ----------------------------------------------------------------------------------------------------------


def open_page(page: PageLike, url: str, anchor: selectors.Selector, what: str) -> None:
    """Load ``url`` and wait for ``anchor``, a control every healthy copy of the page shows; a page without a web
    sign-in (the ``Log in Required`` heading) is a :class:`ModeUnavailableError`."""
    page.goto(url, wait_until="domcontentloaded")
    try:
        anchor.locate(page).first.wait_for(state="visible")
    except Exception as exc:
        if selectors.LOGIN_REQUIRED.locate(page).count():
            raise ModeUnavailableError(f"{what} {SIGNED_OUT_PAGE}") from exc
        raise
    if selectors.LOGIN_REQUIRED.locate(page).count():
        raise ModeUnavailableError(f"{what} {SIGNED_OUT_PAGE}")


def check_saved(page: PageLike) -> None:
    """After the confirm click: the page must not show ESPN's failure text."""
    if selectors.SAVE_FAILED.locate(page).count():
        raise RuntimeError(SAVE_FAILED)


def roster_fix_walk[P: Payload](
    ctx: FlowContext[P], ui: UiDriver, kind: RosterFixType, add: PlayerFacts, drop: PlayerFacts, *, what: str
) -> None:
    """The roster-fix page for ``add`` (``type=add`` or ``type=claim``): pick ``drop``, Continue, then Confirm in the
    dialog, which is the one state-changing click (``ui.confirm``). The page's own refusals (no drop button, a
    ``Can't drop`` notice) are :class:`ModeUnavailableError` before anything is clicked."""
    page = ui.page
    league = ctx.league
    url = selectors.roster_fix_url(ctx.game, league.espn_league_id, league.season, ctx.team_id, add.espn_id, kind)
    open_page(page, url, selectors.CONTINUE_BUTTON, "the roster-fix page")
    ui.screenshot("rosterfix")
    button = selectors.drop_player_button(page, drop.name)
    if not button.count():
        if selectors.undroppable_notice(page, drop.name).count():
            raise ModeUnavailableError(f"the roster-fix page says it can't drop {drop.name} (ESPN's undroppable list)")
        raise ModeUnavailableError(f"the roster-fix page offers no Drop Player button for {drop.name}")
    button.click()
    selectors.continue_button(page, add.name, drop.name).click()
    dialog = selectors.CONFIRM_DIALOG.locate(page)
    dialog.wait_for(state="visible")
    ui.confirm(selectors.confirm_transaction_button(dialog, add.name, drop.name), what=what)
    check_saved(page)


def require_unchanged[P: Payload](ctx: FlowContext[P], facts: MoveFacts) -> None:
    """Re-read before the first click: the add must still be where the preconditions found him (a free agent, or on
    waivers for a claim) and the drop still on our roster, and neither newly roster-locked (a claim's players may
    be locked for today already, which the preconditions allowed), or the click-through would not make the move that
    was approved."""
    problems: list[str] = []
    if facts.add is not None:
        entry = pool_entry(ctx, facts.add.espn_id)
        if entry is None or entry.status != facts.add.status or entry.rostered_team_id is not None:
            now_status = "gone" if entry is None else f"{entry.status} on team {entry.rostered_team_id or 0}"
            problems.append(f"{facts.add.label} is no longer {facts.add.status} ({now_status})")
        elif entry.roster_locked and not facts.add.roster_locked:
            problems.append(f"{facts.add.label} is roster-locked now")
    if facts.drop is not None:
        view = ctx.reader.rosters(facts.scoring_period_id).data
        require_period(view, facts.scoring_period_id)
        try:
            entry = view.roster(ctx.team_id).entry(facts.drop.espn_id)
        except KeyError:
            problems.append(f"{facts.drop.label} is no longer on team {ctx.team_id}")
        else:
            if entry.player_pool_entry.roster_locked and not facts.drop.roster_locked:
                problems.append(f"{facts.drop.label} is roster-locked now")
    if problems:
        raise ModeUnavailableError("the league changed since the preconditions were read: " + "; ".join(problems))


# --- the flow ---------------------------------------------------------------------------------------------------------


class AddDrop(Flow[AddDropPayload]):
    """A free-agent add, a drop, or both (``add_drop``) in both sports."""

    name = "add_drop"
    kinds = (ProposalKind.ADD_DROP,)
    payload_type = AddDropPayload
    modes = (Mode.API, Mode.UI)

    # --- preconditions ------------------------------------------------------------------------------------------------

    def check(self, ctx: FlowContext[AddDropPayload]) -> Preconditions:
        read = read_league(ctx)
        payload = ctx.payload
        failures: list[str] = []
        if read.target != read.latest:
            when = "over" if read.target < read.latest else "not current yet"
            failures.append(
                f"scoring period {read.target} is {when}: ESPN is on scoring period {read.latest}, and a free-agent "
                "move is made in the current period"
            )
        if read.unmapped_lock is not None:
            failures.append(read.unmapped_lock)
        roster = read.our_roster(ctx.team_id)
        if roster is None:
            failures.append(f"ESPN's rosters for scoring period {read.target} have no team {ctx.team_id} (our team_id)")
            return Preconditions(
                tuple(failures), {"scoring_period_id": read.target, "latest_scoring_period": read.latest}
            )
        add: PlayerFacts | None = None
        if payload.add_espn_id is not None:
            entry = pool_entry(ctx, payload.add_espn_id)
            if entry is None:
                failures.append(f"ESPN has no player {payload.add_espn_id}")
            else:
                add = player_facts(entry, read, ctx.game)
                failures.extend(add_failures(add, ctx.team_id, read, ctx.now))
        drop: PlayerFacts | None = None
        if payload.drop_espn_id is not None:
            drop = drop_facts(roster, read, ctx.game, payload.drop_espn_id)
            if drop is None:
                failures.append(not_on_roster(payload.drop_espn_id, ctx.team_id, read.target))
            else:
                failures.extend(drop_failures(drop, read, ctx.now))
        facts = MoveFacts(
            scoring_period_id=read.target,
            latest_scoring_period=read.latest,
            roster_lock_type=read.settings.roster_lock_type.value,
            roster_count=read.roster_count(roster),
            roster_size=read.settings.roster_size,
            add=add,
            drop=drop,
        )
        room = room_failure(facts)
        if room is not None:
            failures.append(room)
        return Preconditions(tuple(failures), facts.to_json())

    # --- API mode -----------------------------------------------------------------------------------------------------

    def build_request(self, ctx: FlowContext[AddDropPayload], pre: Preconditions) -> WriteRequest:
        facts = MoveFacts.from_observed(pre.observed)
        payload = ctx.payload
        items: list[Mapping[str, Any]] = []
        if payload.add_espn_id is not None:
            items.append(add_item(payload.add_espn_id, ctx.team_id))
        if payload.drop_espn_id is not None:
            items.append(drop_item(payload.drop_espn_id, ctx.team_id))
        adding = payload.add_espn_id is not None
        one_click = adding and payload.drop_espn_id is None  # the player list's Add sends no memberId (captured)
        envelope = Envelope(
            team_id=ctx.team_id,
            type=TransactionType.FREEAGENT if adding else TransactionType.ROSTER,
            member_id=None if one_click else ctx.member_id,
            scoring_period_id=facts.latest_scoring_period,
            items=tuple(items),
        )
        return transaction_request(ctx.league, envelope)

    def ui_may_follow(self, response: WriteResponse) -> bool:
        return ui_may_follow(response)

    # --- UI mode ------------------------------------------------------------------------------------------------------

    def run_ui(self, ctx: FlowContext[AddDropPayload], ui: UiDriver, pre: Preconditions) -> None:
        facts = MoveFacts.from_observed(pre.observed)
        if facts.add is None:
            raise ModeUnavailableError(
                "a bare drop has no captured click-through; API mode sends it as a ROSTER transaction with a DROP item"
            )
        require_unchanged(ctx, facts)
        if facts.drop is None:
            self._one_click_add(ctx, ui, facts.add)
            return
        roster_fix_walk(
            ctx, ui, RosterFixType.ADD, facts.add, facts.drop, what=f"add {facts.add.name} and drop {facts.drop.name}"
        )

    def _one_click_add(self, ctx: FlowContext[AddDropPayload], ui: UiDriver, add: PlayerFacts) -> None:
        """The player list's ``Add <Name> ...`` button: with roster room (a precondition) the click sends the request,
        so it is the confirm."""
        page = ui.page
        league = ctx.league
        url = selectors.players_page_url(ctx.game, league.espn_league_id, ctx.team_id, league.season)
        open_page(page, url, selectors.STATUS_FILTER, "the player list")
        ui.screenshot("players")
        button = selectors.add_button(page, add.name)
        if not button.count():
            raise ModeUnavailableError(
                f"the player list shows no Add button for {add.name}: he is not on its first page, or no longer a "
                "free agent; API mode needs no list"
            )
        ui.confirm(button, what=f"add {add.name}")
        check_saved(page)

    # --- verification -------------------------------------------------------------------------------------------------

    def verify(self, ctx: FlowContext[AddDropPayload], pre: Preconditions) -> Verification:
        facts = MoveFacts.from_observed(pre.observed)
        payload = ctx.payload
        view = ctx.reader.rosters(facts.scoring_period_id).data
        require_period(view, facts.scoring_period_id)
        expected = {"add": payload.add_espn_id, "drop": payload.drop_espn_id}
        try:
            roster = view.roster(ctx.team_id)
        except KeyError:
            return Verification(False, f"the re-read has no team {ctx.team_id}", expected, {})
        ids = set(roster.player_ids)
        observed = {
            "add_on_roster": None if payload.add_espn_id is None else payload.add_espn_id in ids,
            "drop_on_roster": None if payload.drop_espn_id is None else payload.drop_espn_id in ids,
        }
        wrong: list[str] = []
        if payload.add_espn_id is not None and payload.add_espn_id not in ids:
            wrong.append(f"{_who(facts.add, payload.add_espn_id)} is not on team {ctx.team_id}")
        if payload.drop_espn_id is not None and payload.drop_espn_id in ids:
            wrong.append(f"{_who(facts.drop, payload.drop_espn_id)} is still on team {ctx.team_id}")
        detail = "; ".join(wrong) if wrong else "the roster shows the move"
        return Verification(not wrong, detail, expected, observed)


def _who(player: PlayerFacts | None, espn_id: int) -> str:
    return player.label if player is not None else f"player {espn_id}"


ADD_DROP = register_flow(AddDrop())
"""The process-wide ``add_drop`` flow (``fm.browser.flows.flow_for`` finds it)."""
