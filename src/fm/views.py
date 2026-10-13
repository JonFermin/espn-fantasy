"""Shared read views (DESIGN section 12, ROADMAP #52): the JSON-able read model behind the MCP tools and the dashboard.

The MCP server (:mod:`fm.mcp_server`) and the local dashboard show the same things, so they are built once, here. Each
view is one public function that returns plain JSON-able dicts: strings, numbers, booleans, ``None``, lists and dicts,
with times as ISO 8601 strings in UTC and numbers rounded, and no ANSI text. A surface is then a thin wrapper that maps
its arguments in and :class:`ViewError` out.

- :func:`status`: per league, the sync state, the team's record and FAAB, the captured pro schedule and the open
  proposals, plus the kill switch.
- :func:`lineup`: :func:`fm.decide.lineup.plan_lineup`'s plan for the synced scoring period. It returns the recommended
  lineup, the current and recommended totals and the moves. Drafts come back as suggestions with their policy verdict,
  and nothing is stored.
- :func:`waivers`: :func:`fm.decide.waivers.decide_waivers` with ``store_proposals=False``. It returns the ranked (add,
  drop) pairs, the moves it would make, the replacement levels and the warnings. Nothing is stored.
- :func:`trade_eval`: :func:`fm.decide.trades.evaluate_trade` for a deal written ``"give A, B get C"``.
- :func:`list_proposals`: the proposal queue as it stands, newest first.
- :func:`proposal_dict` and :func:`verdict_dict`: one proposal row and one policy verdict as every view shows them. The
  execution token never leaves the store.

Inputs
------
Every view opens the config and the store per call (``FM_CONFIG_DIR`` / ``FM_CACHE_DIR`` as for the CLI), so a
long-lived server or dashboard sees what ``fm sync`` stored since it started. ``league`` is a key from ``config.toml``;
``None`` means every configured league. The clock is explicit: ``as_of`` is the caller's ISO 8601 text, which
:func:`parse_as_of` turns into an instant (a naive time is UTC). Without it the view uses ``now``, and without that the
wall clock. Locks, deadlines and availability are judged at that instant.

The pro schedule and the league's ``mMatchup`` schedule come from the captures ``fm sync`` indexed under the cache dir.
When one is missing the output says so instead of guessing. ``schedule_path`` and ``matchups_path`` inject a recorded
``proTeamSchedules_wl`` / ``mMatchup`` view instead, as the CLI's ``--schedule FILE`` and ``--matchups FILE`` do. An
injected file names one league's season, so it needs a single league.

Errors
------
A problem the caller has to fix raises :class:`ViewError`, a :class:`ValueError` whose message is meant for a person.
That covers a ``config.toml`` that does not load, an unknown league, a timestamp that is not ISO 8601, an injected file
that cannot be read, a deal that cannot be read and a league :func:`trade_eval` cannot value. The MCP server turns it
into a ``ToolError`` and the dashboard into an HTTP 4xx, so this module imports neither fastmcp nor a web framework. A
per-league problem in a view over several leagues (not synced, no roster, no pro schedule) is not an error: that
league's block says ``ok: false`` and why, and the other leagues still answer.

No write path
-------------
Workers propose, the executor acts (CLAUDE.md). Every view here is a read:

- This module imports nothing from ``fm.executor`` or ``fm.browser`` and never opens an ESPN session. Every read comes
  from the store and the cache dir the sync filled. ``tests/test_views.py`` pins this with an AST check of this file's
  imports and a subprocess walk of ``sys.modules``.
- It stores nothing. It imports no propose, approve, reject, execute, pause or resume function from
  :mod:`fm.proposals`. Lineup drafts and waiver moves are judged with :func:`fm.proposals.evaluate` (the policy
  verdict, itself a read) and returned, never proposed. Storing a ``proposed`` row is ``create_proposal``'s job in
  :mod:`fm.mcp_server`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

import fm.render.status as render_status
from fm import paths
from fm.config import Config, ConfigError, League, load_config
from fm.decide import registry
from fm.decide.lineup import DECISION_KIND, LineupDecision, LineupDraft, LineupError, LineupPlan, plan_lineup
from fm.decide.trades import (
    DEFAULT_SEED,
    SideImpact,
    TradeContext,
    TradeError,
    TradeEvaluation,
    evaluate_trade,
    load_trade_context,
    parse_trade_text,
)
from fm.decide.waivers import WAIVERS_KIND, WaiverDecision, WaiverError, WaiverMove, decide_waivers
from fm.espn.client import View
from fm.espn.ids import ids_for
from fm.espn.models import MatchupsView, ProSchedule
from fm.jobs.sync import ESPN_SOURCE
from fm.model.valuation import DEFAULT_PLAYOFF_WEIGHT, PlayerOutlook, ValuationError
from fm.proposals import ProposalError, Verdict, evaluate, parse_payload
from fm.proposals.pause import pause_state
from fm.proposals.policy import stored_settings
from fm.sports.base import plugin_for
from fm.store import LeagueRow, ProposalRow, RawSnapshotRow, Store

DEFAULT_TRADE_RUNS = 10_000
"""Seasons :func:`trade_eval` simulates when the league's ``mMatchup`` schedule is available."""
_GONE_FROM_CACHE = "is gone from the cache"


class ViewError(ValueError):
    """A problem the caller has to fix, such as an unknown league or a bad timestamp. The message is for a person.

    Surfaces map it: the MCP server to a ``ToolError``, the dashboard to an HTTP 4xx.
    """


# --- inputs -----------------------------------------------------------------------------------------------------------


def parse_as_of(text: str | None, *, now: datetime | None = None) -> datetime:
    """The instant a view is judged at, aware and in UTC.

    ``text`` is ISO 8601, such as ``2026-10-04T15:00Z``, and a naive time is UTC. Without ``text`` it is ``now``, and
    without ``now`` the wall clock. Raises :class:`ViewError` for text that is not a timestamp.
    """
    if text is None:
        return datetime.now(UTC) if now is None else _utc(now)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ViewError(f"as_of {text!r} is not an ISO 8601 timestamp (try 2026-10-04T15:00Z)") from None
    return _utc(parsed)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _load_config() -> Config:
    try:
        return load_config()
    except ConfigError as exc:
        raise ViewError(str(exc)) from None


def _select(config: Config, key: str | None) -> list[League]:
    if key is None:
        return list(config.leagues)
    known = {league.key: league for league in config.leagues}
    if key not in known:
        raise ViewError(f"no league {key!r} in config.toml; known: {', '.join(known) or 'none'}")
    return [known[key]]


def _at_least(name: str, value: int | None, minimum: int) -> None:
    """A count the caller passed is in range. The MCP schema already enforces it; a dashboard query may not."""
    if value is not None and value < minimum:
        raise ViewError(f"{name} must be at least {minimum}, got {value}")


# --- formatting -------------------------------------------------------------------------------------------------------


def _num(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def _time(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def _person(espn_id: int, name: str | None, position: str | None = None) -> dict[str, Any]:
    return {"espn_id": espn_id, "name": name or f"player {espn_id}", "position": position}


def _outlook_person(outlook: PlayerOutlook | None) -> dict[str, Any] | None:
    return None if outlook is None else _person(outlook.espn_id, outlook.name, outlook.position)


def verdict_dict(verdict: Verdict) -> dict[str, Any]:
    """A policy verdict: whether the move is allowed, the setting it would run under, and every reason it is not."""
    return {"allowed": verdict.allowed, "setting": verdict.setting, "reasons": list(verdict.reasons)}


def proposal_dict(row: ProposalRow, league_key: str | None = None) -> dict[str, Any]:
    """A stored proposal, with a one-line summary of its payload. The execution token never leaves the store."""
    try:
        summary = parse_payload(row).summary()
    except (ProposalError, ValidationError):
        summary = json.dumps(row.payload, sort_keys=True)
    return {
        "id": row.row_id,
        "league": league_key,
        "kind": row.kind,
        "status": row.status,
        "policy": row.policy,
        "scoring_period_id": row.scoring_period_id,
        "payload": row.payload,
        "summary": summary,
        "rationale": row.rationale,
        "deadline": _time(row.deadline),
        "created_by": row.created_by,
        "created_at": _time(row.created_at),
        "decided_by": row.decided_by,
        "decided_at": _time(row.decided_at),
    }


# --- the captures the sync left in the cache, or a file the caller names (no network, no browser) ---------------------


def _read[M: BaseModel](model: type[M], path: Path, *, missing: str) -> tuple[M | None, str]:
    """``path`` parsed as ``model``, else ``None`` and what went wrong (``missing`` when the file is not there)."""
    try:
        return model.model_validate_json(path.read_bytes()), ""
    except FileNotFoundError:
        return None, missing
    except (OSError, ValidationError, ValueError) as exc:
        return None, f"could not be read ({type(exc).__name__})"


def _injected[M: BaseModel](model: type[M], name: str, path: Path | None, selected: Sequence[League]) -> M | None:
    """The file the caller named in ``name``, read before anything is planned so a bad one fails the whole call."""
    if path is None:
        return None
    if len(selected) != 1:
        raise ViewError(f"{name} is one league's season; pass league as well")
    view, problem = _read(model, path, missing="does not exist")
    if view is None:
        raise ViewError(f"{name} {path}: {problem}")
    return view


def _injected_schedule(path: Path | None, selected: Sequence[League]) -> ProSchedule | None:
    schedule = _injected(ProSchedule, "schedule_path", path, selected)
    if schedule is not None and not schedule.scoring_periods:
        raise ViewError(f"schedule_path {path}: has no scoring periods; pass a recorded proTeamSchedules_wl view")
    return schedule


def _injected_matchups(path: Path | None, selected: Sequence[League]) -> MatchupsView | None:
    matchups = _injected(MatchupsView, "matchups_path", path, selected)
    if matchups is not None and not matchups.schedule:
        raise ViewError(f"matchups_path {path}: has no matchups; pass a recorded mMatchup view")
    return matchups


def _schedule_snapshot(store: Store, configured: League) -> RawSnapshotRow | None:
    """The newest captured pro schedule for the league's game and season (captures are season-level)."""
    marker = f"/games/{configured.game}/seasons/{configured.season}"
    rows = store.raw_snapshots.find(ESPN_SOURCE, View.PRO_SCHEDULES.value)
    return next((row for row in reversed(rows) if row.url is not None and marker in row.url), None)


def _cached_schedule(store: Store, configured: League) -> tuple[ProSchedule | None, RawSnapshotRow | None, str]:
    """The captured pro schedule, its snapshot row, and (when there is none) why."""
    snapshot = _schedule_snapshot(store, configured)
    if snapshot is None:
        return (
            None,
            None,
            (
                f"{configured.key}: no pro schedule for {configured.game} {configured.season} is captured under "
                f"{paths.cache_dir()}; run fm sync or fm lineup once with an ESPN session"
            ),
        )
    schedule, problem = _read(ProSchedule, paths.cache_dir() / snapshot.path, missing=_GONE_FROM_CACHE)
    if schedule is None:
        return None, snapshot, f"{configured.key}: the captured pro schedule {snapshot.path} {problem}"
    return schedule, snapshot, ""


def _schedule(store: Store, configured: League, injected: ProSchedule | None) -> tuple[ProSchedule | None, str]:
    """The injected pro schedule, else the captured one; ``None`` and the reason when there is neither."""
    if injected is not None:
        return injected, ""
    schedule, _, note = _cached_schedule(store, configured)
    return schedule, note


def _cached_matchups(store: Store, configured: League, row: LeagueRow) -> tuple[MatchupsView | None, str]:
    """The newest captured ``mMatchup`` view for the league, else ``None`` and the reason."""
    snapshots = store.raw_snapshots.find(ESPN_SOURCE, View.MATCHUP.value, league_id=row.row_id)
    if not snapshots:
        return None, f"{configured.key}: no mMatchup schedule is captured, so title odds are not simulated"
    view, problem = _read(MatchupsView, paths.cache_dir() / snapshots[-1].path, missing=_GONE_FROM_CACHE)
    if view is None:
        return None, f"{configured.key}: the captured mMatchup {snapshots[-1].path} {problem}"
    return view, ""


# --- status -----------------------------------------------------------------------------------------------------------


def status(league: str | None = None) -> dict[str, Any]:
    """Per configured league (or just ``league``): the sync state, the team's record and FAAB, the captured pro
    schedule and the open proposals; plus whether the kill switch is on."""
    config = _load_config()
    selected = _select(config, league)
    with Store.open() as store:
        blocks = [_status_block(_league_status(store, configured)) for configured in selected]
    switch = pause_state()
    return {
        "kill_switch": {"on": switch is not None, "detail": None if switch is None else switch.describe()},
        "leagues": blocks,
    }


def _league_status(store: Store, configured: League) -> render_status.LeagueStatus:
    row = store.leagues.by_key(configured.key)
    if row is None:
        return render_status.LeagueStatus(configured)
    schedule, snapshot, _ = _cached_schedule(store, configured)
    return render_status.LeagueStatus(
        configured,
        league=row,
        settings=stored_settings(store, row),
        team=store.teams.get(row.row_id, row.team_id),
        roster_period=store.rosters.latest_period(row.row_id),
        schedule_captured=snapshot.fetched_at if schedule is not None and snapshot is not None else None,
        schedule_periods=schedule.scoring_periods if schedule is not None else (),
        open_proposals=store.proposals.open(row.row_id),
    )


def _status_block(state: render_status.LeagueStatus) -> dict[str, Any]:
    configured, league = state.configured, state.league
    block: dict[str, Any] = {
        "league": configured.key,
        "sport": configured.sport,
        "season": configured.season,
        "espn_league_id": configured.espn_league_id,
        "team_id": configured.team_id,
        "synced": league is not None,
    }
    if league is None:
        block["note"] = "not synced; run fm sync"
        return block
    block["name"] = league.name
    block["synced_at"] = _time(league.as_of)
    block["roster_period"] = state.roster_period
    team = state.team
    block["team"] = (
        None
        if team is None
        else {
            "name": team.name,
            "wins": team.wins,
            "losses": team.losses,
            "ties": team.ties,
            "points_for": _num(team.points_for, 1),
            "points_against": _num(team.points_against, 1),
            "playoff_seed": team.playoff_seed,
            "waiver_rank": team.waiver_rank,
        }
    )
    settings = state.settings
    uses_faab = settings is not None and settings.acquisition.uses_faab
    block["faab"] = (
        None
        if not uses_faab or team is None or settings is None
        else {"spent": team.acquisition_budget_spent, "budget": settings.acquisition.budget}
    )
    periods = state.schedule_periods
    block["pro_schedule"] = {
        "captured_at": _time(state.schedule_captured),
        "first_period": periods[0] if periods else None,
        "last_period": periods[-1] if periods else None,
    }
    block["open_proposals"] = [proposal_dict(row, configured.key) for row in state.open_proposals]
    return block


# --- lineup -----------------------------------------------------------------------------------------------------------


def lineup(
    league: str | None = None,
    *,
    as_of: str | None = None,
    now: datetime | None = None,
    opponent_team_id: int | None = None,
    schedule_path: Path | None = None,
) -> dict[str, Any]:
    """The lineup plan for each league's synced scoring period, judged at ``as_of`` (else ``now``).

    ``opponent_team_id`` is this period's opponent; his synced roster gives the win probability, so it needs a single
    league. Each league's block holds the recommended starters per slot with projections, the current and recommended
    expected points, the moves, and the proposals the engine would draft with their policy verdict. Nothing is stored.
    """
    _at_least("opponent_team_id", opponent_team_id, 1)
    config = _load_config()
    at = parse_as_of(as_of, now=now)
    selected = _select(config, league)
    if opponent_team_id is not None and len(selected) != 1:
        raise ViewError("opponent_team_id names one league's opponent; pass league as well")
    injected = _injected_schedule(schedule_path, selected)
    with Store.open() as store:
        blocks = [
            _lineup_block(store, config, configured, at=at, opponent=opponent_team_id, injected=injected)
            for configured in selected
        ]
    return {"as_of": _time(at), "leagues": blocks}


def _lineup_block(
    store: Store,
    config: Config,
    configured: League,
    *,
    at: datetime,
    opponent: int | None,
    injected: ProSchedule | None,
) -> dict[str, Any]:
    key = configured.key
    row = store.leagues.by_key(key)
    if row is None:
        return {"league": key, "ok": False, "error": "not synced; run fm sync"}
    if registry.get(row.sport, DECISION_KIND) is None:
        return {"league": key, "ok": False, "error": f"no lineup decision is registered for {row.sport}"}
    period = store.rosters.latest_period(row.row_id)
    if period is None:
        return {"league": key, "ok": False, "error": "no roster snapshot; run fm sync"}
    schedule, note = _schedule(store, configured, injected)
    if schedule is None:
        return {"league": key, "ok": False, "error": f"no lineup planned: {note}"}
    try:
        decision = plan_lineup(
            store, row, schedule=schedule, now=at, period=period, opponent_team_id=opponent, weights=None
        )
    except LineupError as exc:
        return {"league": key, "ok": False, "error": str(exc)}
    warnings = [*decision.warnings, *_period_warning(row, schedule, period, at)]
    return _lineup_dict(store, config, row, decision, at=at, warnings=warnings, opponent=opponent)


def _period_warning(row: LeagueRow, schedule: ProSchedule, period: int, at: datetime) -> list[str]:
    current = plugin_for(row.sport).scoring_period_at(at, schedule)
    if current is None:
        return [f"{row.key}: the pro schedule has no scoring period at {_time(at)}; planned the synced period {period}"]
    if current != period:
        return [f"{row.key}: roster synced for scoring period {period} but period {current} is current; run fm sync"]
    return []


def _lineup_dict(
    store: Store,
    config: Config,
    row: LeagueRow,
    decision: LineupDecision,
    *,
    at: datetime,
    warnings: Sequence[str],
    opponent: int | None,
) -> dict[str, Any]:
    ids = ids_for(row.sport)
    inputs, best, current = decision.inputs, decision.best, decision.current
    team = store.teams.get(row.row_id, row.team_id)
    rival = store.teams.get(row.row_id, opponent) if opponent is not None else None
    order = [*inputs.slot_counts, ids.bench_slot, ids.ir_slot]
    rank = {slot_id: index for index, slot_id in enumerate(order)}
    players = sorted(inputs.players, key=lambda p: (rank.get(best.slots[p.espn_id], len(order)), -p.expected, p.label))
    roster = [
        {
            "slot": ids.slot_label(best.slots[player.espn_id]),
            "slot_id": best.slots[player.espn_id],
            "from_slot": ids.slot_label(player.slot_id),
            "moved": best.slots[player.espn_id] != player.slot_id,
            "espn_id": player.espn_id,
            "name": player.label,
            "position": player.position,
            "projection": _num(player.points, 2),
            "p_active": _num(player.p_active, 2),
            "expected": _num(player.expected, 2),
            "has_game": player.has_game,
            "designation": player.designation,
            "locked": player.locked,
            "lock_at": _time(player.lock_at),
        }
        for player in players
    ]
    labels = {player.espn_id: player.label for player in inputs.players}
    moves = [
        {
            "espn_id": move.espn_id,
            "name": labels.get(move.espn_id),
            "from_slot": ids.slot_label(move.from_slot_id),
            "to_slot": ids.slot_label(move.to_slot_id),
        }
        for move in best.moves
    ]
    outlook = decision.outlook
    return {
        "league": row.key,
        "ok": True,
        "as_of": _time(at),
        "scoring_period": inputs.scoring_period_id,
        "roster_synced_at": _time(inputs.roster_as_of),
        "team": team.name if team is not None else f"team {row.team_id}",
        "opponent": None
        if outlook is None
        else {
            "name": rival.name if rival is not None else None if opponent is None else f"team {opponent}",
            "projected": _num(outlook.opponent_mean, 1),
            "sd": _num(outlook.opponent_sd, 1),
        },
        "current": _plan_dict(current),
        "recommended": {
            **_plan_dict(best),
            "objective": None if best.objective is None else str(best.objective),
            "open_slots": [ids.slot_label(slot_id) for slot_id in best.open_slots],
            "idle_starters": [labels.get(espn_id, str(espn_id)) for espn_id in best.idle_starters],
        },
        "lineup": roster,
        "moves": moves,
        "suggested_proposals": [_draft_dict(store, config, row, draft, at=at) for draft in decision.drafts],
        "warnings": list(warnings),
    }


def _plan_dict(plan: LineupPlan) -> dict[str, Any]:
    return {
        "expected_points": _num(plan.expected, 1),
        "sd": _num(plan.sd, 1),
        "win_probability": _num(plan.win_probability),
    }


def _draft_dict(store: Store, config: Config, row: LeagueRow, draft: LineupDraft, *, at: datetime) -> dict[str, Any]:
    """A draft the engine would propose, with what policy says about it. Nothing is stored."""
    verdict = evaluate(
        store,
        config,
        row,
        draft.kind,
        draft.payload,
        scoring_period_id=draft.scoring_period_id,
        deadline=draft.deadline,
        now=at,
    )
    return {
        "kind": draft.kind.value,
        "scoring_period_id": draft.scoring_period_id,
        "deadline": _time(draft.deadline),
        "rationale": draft.rationale,
        "payload": draft.payload.model_dump(mode="json"),
        "numbers": draft.engine_numbers,
        "policy": verdict_dict(verdict),
    }


# --- waivers ----------------------------------------------------------------------------------------------------------


def waivers(
    league: str | None = None,
    *,
    as_of: str | None = None,
    now: datetime | None = None,
    top: int = 10,
    max_moves: int | None = None,
    schedule_path: Path | None = None,
) -> dict[str, Any]:
    """The waiver plan for each league, judged at ``as_of`` (else ``now``).

    Each league's block holds the moves the engine would make (each with its bid and policy verdict), the ``top``
    ranked (add, drop) pairs by rest-of-season value, the replacement levels and every skipped candidate as a warning.
    ``max_moves`` caps the plan. Nothing is stored.
    """
    _at_least("top", top, 0)
    _at_least("max_moves", max_moves, 1)
    config = _load_config()
    at = parse_as_of(as_of, now=now)
    selected = _select(config, league)
    injected = _injected_schedule(schedule_path, selected)
    with Store.open() as store:
        blocks = [
            _waiver_block(store, config, configured, at=at, top=top, max_moves=max_moves, injected=injected)
            for configured in selected
        ]
    return {"as_of": _time(at), "leagues": blocks}


def _waiver_block(
    store: Store,
    config: Config,
    configured: League,
    *,
    at: datetime,
    top: int,
    max_moves: int | None,
    injected: ProSchedule | None,
) -> dict[str, Any]:
    key = configured.key
    row = store.leagues.by_key(key)
    if row is None:
        return {"league": key, "ok": False, "error": "not synced; run fm sync"}
    if registry.get(row.sport, WAIVERS_KIND) is None:
        return {"league": key, "ok": False, "error": f"no waivers decision is registered for {row.sport}"}
    schedule, note = _schedule(store, configured, injected)
    try:
        decision: WaiverDecision = decide_waivers(
            store, config, row, now=at, schedule=schedule, max_moves=max_moves, store_proposals=False
        )
    except (WaiverError, ValuationError) as exc:
        return {"league": key, "ok": False, "error": str(exc)}
    warnings = [*decision.warnings, *(() if schedule is not None else (note,))]
    return {
        "league": key,
        "ok": True,
        "as_of": _time(at),
        "scoring_period": decision.scoring_period,
        "wire": {"size": len(decision.wire), "synced_at": _time(decision.wire.as_of)},
        "moves": [_waiver_dict(store, config, row, move, at=at, with_verdict=True) for move in decision.moves],
        "ranked_total": len(decision.ranked),
        "ranked": [_waiver_dict(store, config, row, move, at=at, with_verdict=False) for move in decision.ranked[:top]],
        "replacement_levels": [
            {"slot": level.label, "player": level.name, "value": _num(level.value, 1)}
            for level in decision.replacement.values()
        ],
        "warnings": warnings,
    }


def _waiver_dict(
    store: Store, config: Config, row: LeagueRow, move: WaiverMove, *, at: datetime, with_verdict: bool
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": move.kind.value,
        "add": _outlook_person(move.add),
        "drop": _outlook_person(move.drop),
        "gain": _num(move.gain, 2),
        "start_period": move.start,
        "scoring_period_id": move.scoring_period,
        "deadline": _time(move.deadline),
        "add_value": _num(move.add_value, 1),
        "add_vor": _num(move.add_vor, 1),
        "drop_value": _num(move.drop_value, 1),
        "drop_vor": _num(move.drop_vor, 1),
        "bid": move.bid,
        "payload": move.payload.model_dump(mode="json"),
        "rationale": move.rationale(),
    }
    if with_verdict:
        out["policy"] = verdict_dict(
            evaluate(
                store,
                config,
                row,
                move.kind,
                move.payload,
                scoring_period_id=move.scoring_period,
                deadline=move.deadline,
                now=at,
            )
        )
    return out


# --- trades -----------------------------------------------------------------------------------------------------------


def trade_eval(
    deal: str,
    league: str | None = None,
    *,
    as_of: str | None = None,
    now: datetime | None = None,
    runs: int = DEFAULT_TRADE_RUNS,
    seed: int = DEFAULT_SEED,
    schedule_path: Path | None = None,
    matchups_path: Path | None = None,
) -> dict[str, Any]:
    """One trade, written ``"give A, B get C"`` (players by name or ESPN id), judged at ``as_of`` (else ``now``).

    It returns both sides' rest-of-season lineup value and title odds, legality, P(accept), a recommendation and the
    payload ``create_proposal`` takes. Title odds are simulated over ``runs`` seasons with ``seed`` (the same seed gives
    the same numbers) when the league's ``mMatchup`` schedule is available, and the warnings say when it is not.
    ``league`` may be left out only when one league is configured. A deal that cannot be read, or a league that is not
    synced or cannot be valued, raises :class:`ViewError`.
    """
    _at_least("runs", runs, 1)
    config = _load_config()
    at = parse_as_of(as_of, now=now)
    chosen = _select(config, league)
    if len(chosen) != 1:
        raise ViewError("several leagues are configured; pass league")
    configured = chosen[0]
    schedule_file = _injected_schedule(schedule_path, chosen)
    matchups_file = _injected_matchups(matchups_path, chosen)
    with Store.open() as store:
        row = store.leagues.by_key(configured.key)
        if row is None:
            raise ViewError(f"{configured.key}: not synced; run fm sync")
        schedule, schedule_note = _schedule(store, configured, schedule_file)
        matchups, matchups_note = (
            (matchups_file, "") if matchups_file is not None else _cached_matchups(store, configured, row)
        )
        notes = [note for note in (schedule_note, matchups_note) if note]
        notes.append("market values are not consulted; P(accept) uses our own ranks")
        try:
            ctx = load_trade_context(
                store,
                row,
                now=at,
                config=config,
                schedule=schedule,
                matchups=matchups,
                market=None,
                playoff_weight=DEFAULT_PLAYOFF_WEIGHT,
            )
            found = evaluate_trade(ctx, parse_trade_text(ctx, deal), runs=runs, seed=seed)
        except TradeError as exc:
            raise ViewError(f"{configured.key}: {exc}") from None
        return _trade_dict(ctx, found, at=at, notes=notes)


def _trade_side(ctx: TradeContext, side: SideImpact) -> dict[str, Any]:
    before, after = side.odds_before, side.odds_after
    return {
        "team_id": side.team_id,
        "name": side.name,
        "gives": [_person(i, ctx.name(i), _position(ctx, i)) for i in side.gives],
        "gets": [_person(i, ctx.name(i), _position(ctx, i)) for i in side.gets],
        "ros_before": _num(side.ros_before, 1),
        "ros_after": _num(side.ros_after, 1),
        "delta_ros": _num(side.delta_ros, 2),
        "title_before": None if before is None else _num(before.title),
        "title_after": None if after is None else _num(after.title),
        "delta_title": _num(side.delta_title),
        "delta_playoffs": _num(side.delta_playoffs),
        "delta_bye": _num(side.delta_bye),
    }


def _position(ctx: TradeContext, espn_id: int) -> str | None:
    player = ctx.players.get(espn_id)
    return player.position if player is not None else None


def _trade_dict(ctx: TradeContext, found: TradeEvaluation, *, at: datetime, notes: Iterable[str]) -> dict[str, Any]:
    spec, legality, acceptance = found.spec, found.legality, found.acceptance
    return {
        "league": ctx.league.key,
        "as_of": _time(at),
        "with_team": {"team_id": spec.other_team_id, "name": ctx.team_name(spec.other_team_id)},
        "give": [_person(i, ctx.name(i), _position(ctx, i)) for i in spec.give],
        "get": [_person(i, ctx.name(i), _position(ctx, i)) for i in spec.get],
        "recommendation": found.recommendation,
        "reason": found.reason,
        "legal": legality.legal,
        "problems": list(legality.problems),
        "legality_notes": list(legality.notes),
        "ours": _trade_side(ctx, found.ours),
        "theirs": _trade_side(ctx, found.theirs),
        "acceptance": {
            "p_accept": _num(acceptance.p_accept),
            "market_value_to_them": _num(acceptance.receive_value, 0),
            "market_value_from_them": _num(acceptance.send_value, 0),
            "surplus": _num(acceptance.surplus),
            "their_lineup_gain": _num(acceptance.need_gain, 2),
            "their_forced_drops": acceptance.drops,
            "basis": list(acceptance.basis),
            "summary": acceptance.describe(),
        },
        "score": _num(found.score, 4),
        "score_basis": found.score_basis,
        "value_unit": found.unit,
        "simulated": found.simulated,
        "runs": found.runs if found.simulated else 0,
        "seed": found.seed,
        "fit": list(found.fit),
        "proposal_payload": spec.payload().model_dump(mode="json"),
        "warnings": [*found.warnings, *ctx.warnings, *notes],
    }


# --- the proposal queue -----------------------------------------------------------------------------------------------


def list_proposals(league: str | None = None, *, include_closed: bool = False, limit: int = 50) -> dict[str, Any]:
    """The proposal queue, newest first: the open proposals, or with ``include_closed`` also the ones already decided
    or finished (rejected, expired, verified, failed). ``count`` is the total before ``limit`` cuts the list."""
    _at_least("limit", limit, 1)
    config = _load_config()
    selected = _select(config, league)
    out: list[dict[str, Any]] = []
    with Store.open() as store:
        for configured in selected:
            row = store.leagues.by_key(configured.key)
            if row is None:
                continue
            rows = store.proposals.find(league_id=row.row_id) if include_closed else store.proposals.open(row.row_id)
            out.extend(proposal_dict(found, configured.key) for found in rows)
    out.sort(key=lambda item: (item["created_at"] or "", item["id"] or 0), reverse=True)
    return {"count": len(out), "proposals": out[:limit]}
