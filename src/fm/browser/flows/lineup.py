"""``set_lineup``: one scoring period's lineup moves, in API mode with a UI click-through fallback (DESIGN 6.3).

It serves the ``bench_inactive`` and ``lineup`` proposals of both sports. Their payload is a
:class:`fm.proposals.LineupPayload`: slot moves for one scoring period, with both sides of every swap. The executor
(:func:`fm.executor.execute`) drives the steps below and owns the write-safety rules around them.

Preconditions, read through the API (``mRoster`` for the period, plus ``mSettings``). Every failure is listed at once:

- The period is ESPN's current one (``status.latestScoringPeriod``) or a later one in the season. A past lineup
  cannot change. With no period on the proposal, the move is for the current one.
- Each player is on our roster in the slot the move starts from, and is not ``lineupLocked`` (his game has
  started). No player is moved twice.
- Each destination is a slot this league uses (``lineupSlotCounts``) and is among the player's ``eligibleSlots``.
- Afterwards no active or IR slot the moves fill holds more players than the league has. Bench room is left to ESPN,
  whose real leagues report ``isBenchUnlimited``.

API mode sends the web client's ``ROSTER`` envelope, or ``FUTURE_ROSTER`` for a later period, built by
:mod:`fm.browser.transactions` with one ``LINEUP`` item per move. UI mode opens our team page for the period and makes
each move with ESPN's lineup editor. ``MOVE`` on the player's row, then ``HERE`` on the row he goes to, which saves at
once (each ``HERE`` is a :meth:`fm.browser.flows.UiDriver.confirm`). That covers swaps and moves into open slots, so the
moves are planned into those steps before anything is clicked. A lineup that does not break down that way, or a
player name that would match two rows, is left to API mode (:class:`fm.browser.flows.ModeUnavailableError`). Before
the first click the roster is read again, and the walk stops if any player is no longer where the preconditions found
him. The UI comes after a rejected API request only when :func:`fm.browser.transactions.ui_may_follow` allows it.
Verification re-reads ``mRoster`` for the period: every moved player must be in his new slot.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from fm.browser.transactions import TransactionType, lineup_envelope, lineup_type, transaction_request, ui_may_follow
from fm.espn.client import EspnSchemaError
from fm.espn.ids import Game, ids_for
from fm.espn.models import RostersView, TeamRoster
from fm.espn.settings import LeagueSettings, SlotKind
from fm.proposals import LineupMove, LineupPayload, ProposalKind

SIGNED_OUT = (
    "the team page says Log in Required: the browser profile has the API cookies but no ESPN web sign-in, so the "
    "click-through cannot run (sign in on fantasy.espn.com in the fm browser profile; API mode does not need it)"
)


@dataclass(frozen=True, slots=True)
class _Player:
    """One of our players as the precondition read found him."""

    espn_id: int
    name: str
    slot: int
    locked: bool
    eligible: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Lineup:
    """What :meth:`SetLineup.check` read, kept in ``Preconditions.observed`` for the later steps of the same run."""

    scoring_period_id: int
    latest_scoring_period: int
    transaction_type: TransactionType | None
    players: Mapping[int, _Player]
    slot_counts: Mapping[int, int]
    bench_slots: frozenset[int]

    def to_json(self) -> dict[str, Any]:
        return {
            "scoring_period_id": self.scoring_period_id,
            "latest_scoring_period": self.latest_scoring_period,
            "transaction_type": None if self.transaction_type is None else self.transaction_type.value,
            "slot_counts": {str(slot): count for slot, count in self.slot_counts.items()},
            "bench_slots": sorted(self.bench_slots),
            "roster": {
                str(player.espn_id): {
                    "name": player.name,
                    "slot": player.slot,
                    "locked": player.locked,
                    "eligible": list(player.eligible),
                }
                for player in self.players.values()
            },
        }

    @classmethod
    def from_observed(cls, observed: Mapping[str, Any]) -> _Lineup:
        """Read :meth:`to_json` back. Raises ``ValueError`` when the preconditions did not get that far."""
        try:
            raw_type = observed["transaction_type"]
            players = {
                int(pid): _Player(
                    espn_id=int(pid),
                    name=str(data["name"]),
                    slot=int(data["slot"]),
                    locked=bool(data["locked"]),
                    eligible=tuple(int(slot) for slot in data["eligible"]),
                )
                for pid, data in observed["roster"].items()
            }
            return cls(
                scoring_period_id=int(observed["scoring_period_id"]),
                latest_scoring_period=int(observed["latest_scoring_period"]),
                transaction_type=None if raw_type is None else TransactionType(raw_type),
                players=players,
                slot_counts={int(slot): int(count) for slot, count in observed["slot_counts"].items()},
                bench_slots=frozenset(int(slot) for slot in observed["bench_slots"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"set_lineup preconditions are incomplete ({type(exc).__name__}: {exc})") from exc


@dataclass(frozen=True, slots=True)
class _UiStep:
    """One save in ESPN's lineup editor: ``MOVE`` on ``player``'s row, then ``HERE`` on ``partner``'s row (a swap) or
    on an empty row of ``to_slot``."""

    player: _Player
    to_slot: int
    partner: _Player | None = None


class SetLineup(Flow[LineupPayload]):
    """Lineup moves for one scoring period (``bench_inactive``, ``lineup``) in both sports."""

    name = "set_lineup"
    kinds = (ProposalKind.BENCH_INACTIVE, ProposalKind.LINEUP)
    payload_type = LineupPayload
    modes = (Mode.API, Mode.UI)

    # --- preconditions ------------------------------------------------------------------------------------------------

    def check(self, ctx: FlowContext[LineupPayload]) -> Preconditions:
        view = ctx.reader.rosters(ctx.scoring_period_id).data
        settings = ctx.reader.settings().data
        latest = _latest_period(view, settings)
        if latest is None:  # a read problem, not the league's state: refuse, and leave the token unspent
            raise EspnSchemaError("neither mRoster's status nor mSettings says which scoring period is current")
        target = ctx.scoring_period_id or latest
        _require_period(view, target)
        failures: list[str] = []
        transaction_type: TransactionType | None = None
        try:
            transaction_type = lineup_type(target, latest)
        except ValueError as exc:
            failures.append(f"{exc}; a past lineup cannot change")
        final = _final_period(view, settings)
        if final is not None and target > final:
            failures.append(f"scoring period {target} is after the season's last ({final})")
        try:
            roster = view.roster(ctx.team_id)
        except KeyError:
            failures.append(f"ESPN's rosters for scoring period {target} have no team {ctx.team_id} (our team_id)")
            return Preconditions(tuple(failures), {"scoring_period_id": target, "latest_scoring_period": latest})
        lineup = _Lineup(
            scoring_period_id=target,
            latest_scoring_period=latest,
            transaction_type=transaction_type,
            players=_players(roster),
            slot_counts=settings.slot_counts,
            bench_slots=frozenset(slot.slot_id for slot in settings.lineup_slots if slot.kind is SlotKind.BENCH),
        )
        failures.extend(_move_failures(ctx.payload.moves, lineup, ctx.game, ctx.team_id))
        observed = lineup.to_json()
        observed["moves"] = [_describe_move(move, lineup, ctx.game) for move in ctx.payload.moves]
        return Preconditions(tuple(failures), observed)

    # --- API mode -----------------------------------------------------------------------------------------------------

    def build_request(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> WriteRequest:
        lineup = _Lineup.from_observed(pre.observed)
        envelope = lineup_envelope(
            team_id=ctx.team_id,
            member_id=ctx.member_id,
            scoring_period_id=lineup.scoring_period_id,
            latest_scoring_period=lineup.latest_scoring_period,
            moves=ctx.payload.moves,
        )
        return transaction_request(ctx.league, envelope)

    def ui_may_follow(self, response: WriteResponse) -> bool:
        return ui_may_follow(response)

    # --- UI mode ------------------------------------------------------------------------------------------------------

    def run_ui(self, ctx: FlowContext[LineupPayload], ui: UiDriver, pre: Preconditions) -> None:
        lineup = _Lineup.from_observed(pre.observed)
        steps = _ui_steps(ctx.payload.moves, lineup)
        _require_unchanged(ctx, lineup)
        page = ui.page
        league = ctx.league
        url = selectors.team_page_url(
            ctx.game, league.espn_league_id, ctx.team_id, league.season, lineup.scoring_period_id
        )
        _open_roster(page, url)
        ui.screenshot("roster")
        for step in steps:
            label = selectors.slot_label(ctx.game, step.to_slot)
            selectors.MOVE_BUTTON.locate(selectors.player_row(page, step.player.name)).click()
            if step.partner is not None:
                target = selectors.player_row(page, step.partner.name)
                what = f"move {step.player.name} to {label} for {step.partner.name}"
            else:
                found = selectors.empty_slot_row(page, label)
                if found is None:
                    raise ModeUnavailableError(f"the team page shows no open {label} slot for {step.player.name}")
                target = found
                what = f"move {step.player.name} to {label}"
            ui.confirm(selectors.HERE_BUTTON.locate(target), what=what)
            selectors.HERE_BUTTON.locate(page).first.wait_for(state="hidden")  # the editor closed: the move saved

    # --- verification -------------------------------------------------------------------------------------------------

    def verify(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> Verification:
        lineup = _Lineup.from_observed(pre.observed)
        view = ctx.reader.rosters(lineup.scoring_period_id).data
        _require_period(view, lineup.scoring_period_id)
        expected = {str(move.espn_id): move.to_slot_id for move in ctx.payload.moves}
        try:
            roster = view.roster(ctx.team_id)
        except KeyError:
            return Verification(False, f"the re-read has no team {ctx.team_id}", expected, {})
        slots = {entry.player_id: entry.lineup_slot_id for entry in roster.entries}
        observed = {str(move.espn_id): slots.get(move.espn_id) for move in ctx.payload.moves}
        wrong: list[str] = []
        for move in ctx.payload.moves:
            actual = slots.get(move.espn_id)
            if actual == move.to_slot_id:
                continue
            where = f"in {_slot(ctx.game, actual)}" if actual is not None else f"not on team {ctx.team_id}"
            wrong.append(f"{_who(move.espn_id, lineup)} is {where}, expected in {_slot(ctx.game, move.to_slot_id)}")
        detail = "; ".join(wrong) if wrong else "every moved player is in his new slot"
        return Verification(not wrong, detail, expected, observed)


SET_LINEUP = register_flow(SetLineup())
"""The process-wide ``set_lineup`` flow (``fm.browser.flows.flow_for`` finds it)."""


# --- preconditions ----------------------------------------------------------------------------------------------------


def _require_period(view: RostersView, scoring_period_id: int) -> None:
    """ESPN echoes the period a roster read asked for; an answer for another period is unusable (a refusal before the
    write, an unreadable re-read after it), never a verdict on the lineup."""
    if view.scoring_period_id is not None and view.scoring_period_id != scoring_period_id:
        raise EspnSchemaError(
            f"asked for scoring period {scoring_period_id}'s rosters, ESPN answered with scoring period "
            f"{view.scoring_period_id}'s"
        )


def _latest_period(view: RostersView, settings: LeagueSettings) -> int | None:
    """ESPN's current scoring period, the one ``ROSTER`` moves are for (``status.latestScoringPeriod``)."""
    if view.status is not None and view.status.latest_scoring_period:
        return view.status.latest_scoring_period
    return settings.current_scoring_period or None


def _final_period(view: RostersView, settings: LeagueSettings) -> int | None:
    if view.status is not None and view.status.final_scoring_period:
        return view.status.final_scoring_period
    return settings.final_scoring_period or None


def _players(roster: TeamRoster) -> dict[int, _Player]:
    return {
        entry.player_id: _Player(
            espn_id=entry.player_id,
            name=entry.player.full_name,
            slot=entry.lineup_slot_id,
            locked=entry.lineup_locked,
            eligible=tuple(entry.player.eligible_slots),
        )
        for entry in roster.entries
    }


def _move_failures(moves: Sequence[LineupMove], lineup: _Lineup, game: Game, team_id: int) -> list[str]:
    """Every reason the moves cannot be made on the lineup as read."""
    failures: list[str] = []
    counts = Counter(move.espn_id for move in moves)
    for espn_id, times in counts.items():
        if times > 1:
            failures.append(f"{_who(espn_id, lineup)} is moved {times} times; give each player one move")
    used = {slot for slot, count in lineup.slot_counts.items() if count > 0}
    after = {player.espn_id: player.slot for player in lineup.players.values()}
    for move in moves:
        player = lineup.players.get(move.espn_id)
        if player is None:
            period = lineup.scoring_period_id
            failures.append(f"player {move.espn_id} is not on team {team_id} in scoring period {period}")
            continue
        who = _who(move.espn_id, lineup)
        if player.slot != move.from_slot_id:
            failures.append(f"{who} is in {_slot(game, player.slot)}, not {_slot(game, move.from_slot_id)}")
        if player.locked:
            failures.append(f"{who} is locked: his game has started")
        if move.to_slot_id not in used:
            failures.append(f"{_slot(game, move.to_slot_id)} is not a slot in this league's lineup")
        elif move.to_slot_id not in player.eligible:
            allowed = ", ".join(_slot(game, slot) for slot in player.eligible if slot in used) or "no slot"
            failures.append(f"{who} cannot play {_slot(game, move.to_slot_id)}; he is eligible for {allowed}")
        after[move.espn_id] = move.to_slot_id
    filled = Counter(after.values())
    for slot in sorted({move.to_slot_id for move in moves} & (used - lineup.bench_slots)):
        have = lineup.slot_counts[slot]
        if filled[slot] > have:
            failures.append(
                f"{_slot(game, slot)} would hold {filled[slot]} players and the league has {have}: "
                "move the player it holds out in the same proposal"
            )
    return failures


def _describe_move(move: LineupMove, lineup: _Lineup, game: Game) -> dict[str, Any]:
    return {
        "espn_id": move.espn_id,
        "name": _who(move.espn_id, lineup),
        "from": _slot(game, move.from_slot_id),
        "to": _slot(game, move.to_slot_id),
    }


def _who(espn_id: int, lineup: _Lineup) -> str:
    player = lineup.players.get(espn_id)
    return f"{player.name} ({espn_id})" if player is not None and player.name else f"player {espn_id}"


def _slot(game: Game, slot_id: int) -> str:
    """``UTIL (11)``: the slot's label with its ESPN id."""
    return f"{ids_for(game).slot_label(slot_id)} ({slot_id})"


# --- UI mode ----------------------------------------------------------------------------------------------------------


def _ui_steps(moves: Sequence[LineupMove], lineup: _Lineup) -> list[_UiStep]:
    """The moves as lineup-editor saves: swaps first where a move has a partner going the other way, else a move into
    an open slot. Raises ``ModeUnavailableError`` before anything is clicked when the moves do not break down so."""
    pending = list(moves)
    filled = Counter(player.slot for player in lineup.players.values())
    steps: list[_UiStep] = []
    while pending:
        step = _next_step(pending, lineup, filled)
        if step is None:
            raise ModeUnavailableError(
                "the click-through makes swaps and moves into open slots, and these moves do not break down into "
                "those; API mode can make them in one transaction"
            )
        steps.append(step)
    for step in steps:
        for player in (step.player, step.partner):
            if player is not None:
                _require_unique_row(player, lineup)
    return steps


def _next_step(pending: list[LineupMove], lineup: _Lineup, filled: Counter[int]) -> _UiStep | None:
    """Take the first move that can be clicked now off ``pending`` (and its partner, for a swap)."""
    for move in pending:
        partner = next(
            (
                other
                for other in pending
                if other is not move and other.from_slot_id == move.to_slot_id and other.to_slot_id == move.from_slot_id
            ),
            None,
        )
        if partner is not None:
            pending.remove(move)
            pending.remove(partner)
            return _UiStep(lineup.players[move.espn_id], move.to_slot_id, lineup.players[partner.espn_id])
        if filled[move.to_slot_id] < lineup.slot_counts.get(move.to_slot_id, 0):
            pending.remove(move)
            filled[move.from_slot_id] -= 1
            filled[move.to_slot_id] += 1
            return _UiStep(lineup.players[move.espn_id], move.to_slot_id)
    return None


def _require_unique_row(player: _Player, lineup: _Lineup) -> None:
    """The click-through finds a player's row by his name, so it must not match another row on the page."""
    if not player.name.strip():
        raise ModeUnavailableError(f"player {player.espn_id} has no name to find his row by")
    needle = player.name.casefold()
    matches = [other.name for other in lineup.players.values() if needle in other.name.casefold()]
    if len(matches) > 1:
        raise ModeUnavailableError(f"{player.name}'s row cannot be told apart by name from {sorted(matches)}")


def _require_unchanged(ctx: FlowContext[LineupPayload], lineup: _Lineup) -> None:
    """Re-read the roster before the first click: every moved player must still be where the preconditions found him,
    and unlocked, or the click-through would not make the moves that were approved."""
    view = ctx.reader.rosters(lineup.scoring_period_id).data
    _require_period(view, lineup.scoring_period_id)
    roster = view.roster(ctx.team_id)
    entries = {entry.player_id: entry for entry in roster.entries}
    problems: list[str] = []
    for move in ctx.payload.moves:
        who = _who(move.espn_id, lineup)
        entry = entries.get(move.espn_id)
        if entry is None:
            problems.append(f"{who} is no longer on team {ctx.team_id}")
        elif entry.lineup_slot_id != move.from_slot_id:
            problems.append(f"{who} is in {_slot(ctx.game, entry.lineup_slot_id)} now")
        elif entry.lineup_locked:
            problems.append(f"{who} is locked now")
    if problems:
        raise ModeUnavailableError("the roster changed since the preconditions were read: " + "; ".join(problems))


def _open_roster(page: PageLike, url: str) -> None:
    """Load our team page and wait for the roster; a page without a web sign-in is :data:`SIGNED_OUT`."""
    page.goto(url, wait_until="domcontentloaded")
    try:
        selectors.ROSTER_TABLE.locate(page).first.wait_for(state="visible")
    except Exception as exc:
        if selectors.LOGIN_REQUIRED.locate(page).count():
            raise ModeUnavailableError(SIGNED_OUT) from exc
        raise
    if selectors.LOGIN_REQUIRED.locate(page).count():
        raise ModeUnavailableError(SIGNED_OUT)
