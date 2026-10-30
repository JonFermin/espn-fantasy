"""``ir_activation``: bring a player back from IR once his injury status no longer allows him there (both sports).

ESPN lets a player sit in an IR slot only while his designation is IR-eligible (:data:`fm.decide.trades.IR_ELIGIBLE`:
OUT, injury reserve, suspension). Once he is listed day-to-day or active, ESPN blocks the team's adds and trades until
he leaves the slot, and no other decision moves him: the lineup planners keep IR players where they are, and streaming
never drops or activates one. This decision proposes the fix, one ESPN write per proposal (the executor sends one
request per approved proposal, DESIGN 6.3):

- **Room on the roster** (fewer players outside IR than :attr:`LeagueSettings.roster_size`): a ``lineup`` proposal
  moving him from the IR slot to the bench, served by the ``set_lineup`` flow; the lineup planners take him from
  there.
- **A full roster:** a drop-only ``add_drop`` proposal, of the rostered player whose loss costs the rest-of-season
  value least (:meth:`fm.model.valuation.RosterValuer.gain` with the returning player counted on the roster). The
  move off IR follows on a later run, once the drop has gone through and the roster has room. Never dropped: a player
  in an IR slot, an untouchable, a player in another open proposal, one whose game has locked him, or one the
  valuation could not value.

Both are approval-only, whatever the league's lineup setting says (``max_setting="approve"``): the user chose that
(2026-10-29), and a drop is never auto in any case. While a player is stuck, :func:`stuck_on_ir` is also what the
streaming and waiver decisions check before proposing an add ESPN would refuse.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from fm.config import Config
from fm.decide import registry as _registry
from fm.decide.trades import IR_ELIGIBLE
from fm.espn.ids import InjuryStatus
from fm.espn.settings import LeagueSettings
from fm.model.valuation import ValuationError, load_valuation
from fm.proposals import AddDropPayload, LineupMove, LineupPayload, PolicyError, ProposalKind, propose
from fm.proposals.policy import as_utc, find_untouchables, parse_payload, stored_settings
from fm.sports.base import ScheduleLike, plugin_for
from fm.store import LeagueRow, PlayerRow, ProposalRow, RosterEntryRow, Store

DECISION_KIND: Final = "ir_activation"
CREATED_BY: Final = "decide.ir_activation"


class IrActivationError(ValueError):
    """The league cannot be read for the decision (not synced, no roster for our team)."""


@dataclass(frozen=True, slots=True)
class Stuck:
    """A player of ours in an IR slot whose injury status no longer allows it."""

    entry: RosterEntryRow
    player: PlayerRow | None

    @property
    def espn_id(self) -> int:
        return self.entry.espn_id

    @property
    def name(self) -> str:
        return self.player.full_name if self.player is not None else f"ESPN {self.espn_id}"

    @property
    def status(self) -> str:
        return (self.player.injury_status if self.player is not None else None) or "unknown"


@dataclass(frozen=True, slots=True)
class IrActivationProposals:
    """What :func:`propose_ir_activation` stored and the refusals and notes it ran into."""

    stuck: tuple[Stuck, ...] = ()
    proposals: tuple[ProposalRow, ...] = ()
    blocked: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def stuck_on_ir(
    store: Store, league: LeagueRow, settings: LeagueSettings, *, period: int | None = None
) -> tuple[Stuck, ...]:
    """Our players in an IR slot whose injury status is known and not IR-eligible, in ``period`` (default: the latest
    synced roster's). A status the store does not have (``UNKNOWN``) is a gap in the data, not a reason to block adds or
    drop anyone, so it does not count. Empty before the first sync."""
    if league.row_id is None:
        return ()
    target = period if period is not None else store.rosters.latest_period(league.row_id)
    if target is None:
        return ()
    ir_slot = plugin_for(league.sport).ids.ir_slot
    stuck: list[Stuck] = []
    for entry in store.rosters.team(league.row_id, target, league.team_id):
        if entry.lineup_slot_id != ir_slot:
            continue
        player = store.players.get(league.sport, entry.espn_id)
        status = settings.ids.injury_status(player.injury_status if player is not None else None)
        if status is not InjuryStatus.UNKNOWN and status not in IR_ELIGIBLE:
            stuck.append(Stuck(entry, player))
    return tuple(stuck)


def stuck_warning(league: LeagueRow, stuck: Sequence[Stuck]) -> str:
    """The note a decision that would add a player leaves while ``stuck`` blocks the team's moves."""
    names = ", ".join(f"{one.name} ({one.status})" for one in stuck)
    return (
        f"{league.key}: {names} in an IR slot without an IR-eligible status: ESPN blocks our adds until he is moved "
        "off IR, so no add is proposed (ir_activation proposes the fix)"
    )


def propose_ir_activation(
    store: Store,
    config: Config,
    league: LeagueRow,
    *,
    schedule: ScheduleLike,
    now: datetime,
    settings: LeagueSettings | None = None,
) -> IrActivationProposals:
    """The ``ir_activation`` decision (module docstring): for each stuck player, the move off IR when the roster has
    room, else the drop that makes room. A draft policy refuses is reported in ``blocked``, not raised."""
    at = as_utc(now)
    synced = settings if settings is not None else stored_settings(store, league)
    if synced is None or league.row_id is None:
        raise IrActivationError(f"league {league.key!r} has no synced settings; run fm sync")
    period = store.rosters.latest_period(league.row_id)
    if period is None:
        raise IrActivationError(f"no roster for team {league.team_id} of league {league.key!r}; run fm sync")
    stuck = stuck_on_ir(store, league, synced, period=period)
    if not stuck:
        return IrActivationProposals()
    plugin = plugin_for(league.sport)
    team = store.rosters.team(league.row_id, period, league.team_id)
    outside_ir = sum(1 for entry in team if entry.lineup_slot_id != plugin.ids.ir_slot)
    room = synced.roster_size - outside_ir
    stored: list[ProposalRow] = []
    blocked: list[str] = []
    warnings: list[str] = []
    for one in stuck:
        try:
            if room > 0:
                deadline = _lineup_lock(league, synced, one, period, schedule)
                stored.append(_activate(store, config, league, one, plugin.ids.bench_slot, deadline, at))
                room -= 1
                continue
            drop, closed = _cheapest_drop(store, config, league, synced, team, stuck, period, at, schedule, warnings)
            if drop is None:
                why = (
                    f"drops closed for period {period} at the league's roster lock; the drop is proposed when the "
                    "next period opens"
                    if closed
                    else "no player can be dropped to make room"
                )
                warnings.append(f"{league.key}: {one.name} is stuck on IR: {why}")
                continue
            stored.append(_drop(store, config, league, synced, one, drop, period, schedule, at))
        except PolicyError as exc:
            blocked.append(str(exc))
        break  # one at a time: the next player's fix depends on the roster this one leaves
    return IrActivationProposals(stuck=stuck, proposals=tuple(stored), blocked=tuple(blocked), warnings=tuple(warnings))


def _activate(
    store: Store,
    config: Config,
    league: LeagueRow,
    one: Stuck,
    bench_slot: int,
    deadline: datetime | None,
    at: datetime,
) -> ProposalRow:
    """The move off IR, for whichever period is current when it runs (no period on the proposal: the ``set_lineup``
    flow then moves him in ESPN's current one), so one open proposal per player serves until his next lineup lock."""
    move = LineupMove(espn_id=one.espn_id, from_slot_id=one.entry.lineup_slot_id, to_slot_id=bench_slot)
    return propose(
        store,
        config,
        league,
        ProposalKind.LINEUP,
        LineupPayload(moves=(move,)),
        created_by=CREATED_BY,
        engine_numbers={"activate": {"espn_id": one.espn_id, "name": one.name, "status": one.status}},
        rationale=(
            f"Activate {one.name} from IR: ESPN lists him {one.status}, which an IR slot does not allow, and ESPN "
            "blocks our adds and trades until he is moved. He goes to the bench; the lineup planner starts him from "
            "there."
        ),
        deadline=deadline,
        dedupe_key=f"ir_activate:{league.season}:{one.espn_id}",
        max_setting="approve",
        now=at,
    )


def _drop(
    store: Store,
    config: Config,
    league: LeagueRow,
    settings: LeagueSettings,
    one: Stuck,
    drop: tuple[RosterEntryRow, PlayerRow, float],
    period: int,
    schedule: ScheduleLike,
    at: datetime,
) -> ProposalRow:
    entry, player, loss = drop
    deadline = _drop_cutoff(league, settings, player, period, schedule)
    return propose(
        store,
        config,
        league,
        ProposalKind.ADD_DROP,
        AddDropPayload(drop_espn_id=entry.espn_id),
        created_by=CREATED_BY,
        scoring_period_id=period,
        engine_numbers={
            "activate": {"espn_id": one.espn_id, "name": one.name, "status": one.status},
            "drop": {"espn_id": entry.espn_id, "name": player.full_name},
            "value_lost": loss,
        },
        rationale=(
            f"Drop {player.full_name} to make room for {one.name}, who is {one.status} and can no longer sit on IR "
            f"(ESPN blocks our adds and trades until he is moved). Of the players we could drop, {player.full_name} "
            f"is the one we miss least: {_loss_text(loss)} Once the drop goes through, the move of {one.name} off IR "
            "is proposed next."
        ),
        deadline=deadline,
        dedupe_key=f"ir_drop:{league.season}:{period}:{one.espn_id}:{entry.espn_id}",
        max_setting="approve",
        now=at,
    )


def _lineup_lock(
    league: LeagueRow, settings: LeagueSettings, one: Stuck, period: int, schedule: ScheduleLike
) -> datetime | None:
    """When ``one`` locks in ``period`` under the league's lineup lock; ``None`` without a game (or a known lock)."""
    if one.player is None or one.player.pro_team_id is None:
        return None
    try:
        return plugin_for(league.sport).lock_time(
            one.player.pro_team_id, period, schedule, lock_type=settings.lineup_lock_type
        )
    except ValueError:  # an UNKNOWN lock type: no deadline rather than a guessed one
        return None


def _loss_text(loss: float) -> str:
    if loss <= 0.05:
        return "the free agents who would fill in for him are worth as much over the rest of the season."
    return f"losing him costs {loss:.1f} points of rest-of-season value."


def _drop_cutoff(
    league: LeagueRow, settings: LeagueSettings, player: PlayerRow, period: int, schedule: ScheduleLike
) -> datetime | None:
    """When ESPN stops taking a drop of ``player`` in ``period``: the league's roster lock for his pro team."""
    if player.pro_team_id is None:
        return None
    plugin = plugin_for(league.sport)
    return plugin.transaction_cutoff(player.pro_team_id, period, schedule, lock_type=settings.roster_lock_type)


def _cheapest_drop(
    store: Store,
    config: Config,
    league: LeagueRow,
    settings: LeagueSettings,
    team: Sequence[RosterEntryRow],
    stuck: Sequence[Stuck],
    period: int,
    at: datetime,
    schedule: ScheduleLike,
    warnings: list[str],
) -> tuple[tuple[RosterEntryRow, PlayerRow, float] | None, bool]:
    """The droppable player whose loss costs the least rest-of-season value, with the value lost (points); among
    equal losses (bench players the lineup never needs), the one whose own rest-of-season value is lowest. A player
    ESPN no longer takes a drop of in ``period`` (past the league's roster lock for his team) is passed over; the
    flag says whether that left nobody."""
    ir_slot = plugin_for(league.sport).ids.ir_slot
    candidates = [entry for entry in team if entry.lineup_slot_id != ir_slot and not entry.lineup_locked]
    policy = config.league(league.key).policy
    barred = set(find_untouchables(store, league.sport, policy, (entry.espn_id for entry in candidates)))
    barred |= _committed(store, league)
    barred |= {one.espn_id for one in stuck}
    try:
        valuer = load_valuation(store, league, now=at, settings=settings, schedule=schedule).valuer()
    except ValuationError as exc:
        warnings.append(f"{league.key}: no drop chosen for the IR fix: {exc}")
        return None, False
    closed = False
    best: tuple[tuple[float, float], tuple[RosterEntryRow, PlayerRow, float]] | None = None
    for entry in candidates:
        player = store.players.get(league.sport, entry.espn_id)
        if entry.espn_id in barred or player is None or entry.espn_id not in valuer.roster:
            continue
        if not valuer.outlooks[entry.espn_id].has_projection:  # his value is unknown, not zero: never dropped
            continue
        cutoff = _drop_cutoff(league, settings, player, period, schedule)
        if cutoff is not None and cutoff <= at:
            closed = True
            continue
        loss = -valuer.gain(drop=entry.espn_id)
        key = (round(loss, 6), valuer.ros(entry.espn_id))  # bench players often tie at no loss: the weakest goes
        if best is None or key < best[0]:
            best = (key, (entry, player, loss))
    return (None, closed) if best is None else (best[1], False)


def _committed(store: Store, league: LeagueRow) -> set[int]:
    """Players another open proposal of the league moves, adds or drops: a new drop leaves them alone."""
    players: set[int] = set()
    for row in store.proposals.open(league.row_id):
        try:
            payload = parse_payload(row)
        except (PolicyError, ValueError):
            continue
        if isinstance(payload, AddDropPayload):
            players.update(i for i in (payload.add_espn_id, payload.drop_espn_id) if i is not None)
        elif isinstance(payload, LineupPayload):
            players.update(move.espn_id for move in payload.moves)
        else:
            players.update(_ints(getattr(payload, "outgoing_players", ())))
    return players


def _ints(values: Iterable[object]) -> list[int]:
    return [value for value in values if isinstance(value, int)]


_registry.register("nfl", DECISION_KIND, propose_ir_activation)
_registry.register("nba", DECISION_KIND, propose_ir_activation)
