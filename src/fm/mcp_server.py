"""The MCP server (DESIGN section 12, ROADMAP #40): read tools plus ``create_proposal``, which never acts.

It lets Claude Code or Claude Desktop answer "who should I start at FLEX?" from the store ``fm sync`` filled. Run it
with ``fm mcp`` (stdio). Tools, all returning JSON-able dicts (no ANSI text):

- ``status``: per league, the sync state, the team's record and FAAB, the captured pro schedule, the kill switch and the
  open proposals.
- ``lineup``: :func:`fm.decide.lineup.plan_lineup`'s plan for the synced scoring period: the recommended lineup, the
  current and recommended totals and the moves. Drafts are returned as suggestions with their policy verdict; nothing
  is stored.
- ``waivers``: :func:`fm.decide.waivers.decide_waivers` with ``store_proposals=False``: the ranked (add, drop) pairs,
  the moves it would make, the replacement levels and the warnings. Nothing is stored.
- ``trade_eval``: :func:`fm.decide.trades.evaluate_trade` for a deal written ``"give A, B get C"``.
- ``list_proposals``: the proposal queue as it stands.
- ``create_proposal``: :func:`fm.proposals.propose` with the policy verdict. It stores one ``proposed`` row, tagged
  ``created_by = "mcp"``, and nothing else.

No write path
-------------
Workers propose, the executor acts (CLAUDE.md). This module is a worker, so it has no way to act:

- It imports nothing from ``fm.executor`` or ``fm.browser`` and never opens an ESPN session: every read comes from the
  store and the cache dir the sync filled (the pro schedule and the ``mMatchup`` schedule are used when captured, and
  the output says when they are not). The test suite pins this with an AST check of this file's imports and a
  subprocess walk of ``sys.modules``.
- It imports no approve, reject, execute, pause or resume function from :mod:`fm.proposals`, and registers no tool for
  one; the tool list is an allowlist (:data:`TOOL_NAMES`) the tests compare against, and they also scan every tool name
  and description for words that would suggest otherwise.
- ``create_proposal`` stores a row in status ``proposed`` and stops. A trade kind is approve-only by policy and a lineup
  proposal that policy would run as ``auto`` (approved on its own at T-15) is stored as ``approve`` instead: only the
  engine's own drafts may be auto-approved, never a move Claude suggested.
- Policy refusals come back as a result (``status: "blocked"`` with every reason), never as an exception.

Every tool opens the store and config per call (``FM_CONFIG_DIR`` / ``FM_CACHE_DIR`` as for the CLI), so a long-lived
server sees what ``fm sync`` stored since it started.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field, ValidationError

import fm.render.status as render_status
from fm import paths
from fm.config import Config, ConfigError, League, load_config
from fm.decide import registry
from fm.decide.lineup import DECISION_KIND, LineupDecision, LineupDraft, LineupError, plan_lineup
from fm.decide.trades import (
    DEFAULT_SEED,
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
from fm.proposals import PolicyError, ProposalError, ProposalKind, Verdict, evaluate, kind_spec, parse_payload, propose
from fm.proposals.pause import pause_state
from fm.proposals.policy import as_utc, stored_settings
from fm.sports.base import plugin_for
from fm.store import LeagueRow, ProposalRow, RawSnapshotRow, Store

SERVER_NAME = "espn-fantasy"
MCP_CREATED_BY = "mcp"
"""``proposals.created_by`` of every proposal this server stores."""
TOOL_NAMES = ("status", "lineup", "waivers", "trade_eval", "list_proposals", "create_proposal")
"""The whole tool list. Adding a tool means changing this tuple and the test that guards it."""
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
STORES_PROPOSAL = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
INSTRUCTIONS = (
    "Personal ESPN fantasy manager for NFL (ffl) and NBA (fba) leagues. The read tools (status, lineup, waivers, "
    "trade_eval, list_proposals) answer from the local store and change nothing. create_proposal records a suggestion "
    "for the user to review in the fm CLI or on their phone; it never changes a league, and trades always need the "
    "user's explicit review. Names are looked up for display, but every move is made of ESPN ids."
)
MAX_RATIONALE = 2000
DEFAULT_TRADE_RUNS = 10_000

LeagueArg = Annotated[
    str | None,
    Field(description="League key from config.toml (for example 'ffl' or 'fba'). Default: every configured league."),
]
AsOfArg = Annotated[
    str | None,
    Field(
        description="Pin the clock (ISO 8601, e.g. 2026-10-04T15:00Z; a naive time is UTC): locks, deadlines and "
        "availability are judged at this instant. Default: now."
    ),
]


def _now() -> datetime:
    """The wall clock; tests replace it."""
    return datetime.now(UTC)


# --- small helpers ----------------------------------------------------------------------------------------------------


def _config() -> Config:
    try:
        return load_config()
    except ConfigError as exc:
        raise ToolError(str(exc)) from None


def _select(config: Config, key: str | None) -> list[League]:
    if key is None:
        return list(config.leagues)
    known = {league.key: league for league in config.leagues}
    if key not in known:
        raise ToolError(f"no league {key!r} in config.toml; known: {', '.join(known) or 'none'}")
    return [known[key]]


def _parse_as_of(text: str | None) -> datetime:
    if text is None:
        return _now()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ToolError(f"as_of {text!r} is not an ISO 8601 timestamp (try 2026-10-04T15:00Z)") from None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _num(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def _time(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def _person(espn_id: int, name: str | None, position: str | None = None) -> dict[str, Any]:
    return {"espn_id": espn_id, "name": name or f"player {espn_id}", "position": position}


def _outlook_person(outlook: PlayerOutlook | None) -> dict[str, Any] | None:
    return None if outlook is None else _person(outlook.espn_id, outlook.name, outlook.position)


def _verdict(verdict: Verdict) -> dict[str, Any]:
    return {"allowed": verdict.allowed, "setting": verdict.setting, "reasons": list(verdict.reasons)}


def _proposal(row: ProposalRow, league_key: str | None = None) -> dict[str, Any]:
    """A proposal for the caller. The execution token never leaves the store."""
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


# --- the captures the sync left in the cache (no network, no browser) ---------------------------------------------


def _read_schedule(path: Path) -> tuple[ProSchedule | None, str]:
    try:
        return ProSchedule.model_validate_json(path.read_bytes()), ""
    except FileNotFoundError:
        return None, "is gone from the cache"
    except (OSError, ValidationError, ValueError) as exc:
        return None, f"could not be read ({type(exc).__name__})"


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
    schedule, problem = _read_schedule(paths.cache_dir() / snapshot.path)
    if schedule is None:
        return None, snapshot, f"{configured.key}: the captured pro schedule {snapshot.path} {problem}"
    return schedule, snapshot, ""


def _cached_matchups(store: Store, configured: League, row: LeagueRow) -> tuple[MatchupsView | None, str]:
    """The newest captured ``mMatchup`` view for the league, else ``None`` and the reason."""
    snapshots = store.raw_snapshots.find(ESPN_SOURCE, View.MATCHUP.value, league_id=row.row_id)
    if not snapshots:
        return None, f"{configured.key}: no mMatchup schedule is captured, so title odds are not simulated"
    path = paths.cache_dir() / snapshots[-1].path
    try:
        return MatchupsView.model_validate_json(path.read_bytes()), ""
    except FileNotFoundError:
        return None, f"{configured.key}: the captured mMatchup {snapshots[-1].path} is gone from the cache"
    except (OSError, ValidationError, ValueError) as exc:
        return (
            None,
            f"{configured.key}: the captured mMatchup {snapshots[-1].path} could not be read ({type(exc).__name__})",
        )


# --- status -----------------------------------------------------------------------------------------------------------


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


def _status_block(status: render_status.LeagueStatus) -> dict[str, Any]:
    configured, league = status.configured, status.league
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
    block["roster_period"] = status.roster_period
    team = status.team
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
    settings = status.settings
    uses_faab = settings is not None and settings.acquisition.uses_faab
    block["faab"] = (
        None
        if not uses_faab or team is None or settings is None
        else {"spent": team.acquisition_budget_spent, "budget": settings.acquisition.budget}
    )
    periods = status.schedule_periods
    block["pro_schedule"] = {
        "captured_at": _time(status.schedule_captured),
        "first_period": periods[0] if periods else None,
        "last_period": periods[-1] if periods else None,
    }
    block["open_proposals"] = [_proposal(row, configured.key) for row in status.open_proposals]
    return block


def _status(league: LeagueArg = None) -> dict[str, Any]:
    config = _config()
    with Store.open() as store:
        blocks = [_status_block(_league_status(store, configured)) for configured in _select(config, league)]
    switch = pause_state()
    return {
        "kill_switch": {"on": switch is not None, "detail": None if switch is None else switch.describe()},
        "leagues": blocks,
    }


# --- lineup -----------------------------------------------------------------------------------------------------------


def _lineup_block(
    store: Store, config: Config, configured: League, *, at: datetime, opponent: int | None
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
    schedule, _, note = _cached_schedule(store, configured)
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


def _plan_dict(plan: Any) -> dict[str, Any]:
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
        "policy": _verdict(verdict),
    }


def _lineup(
    league: LeagueArg = None,
    as_of: AsOfArg = None,
    opponent_team_id: Annotated[
        int | None,
        Field(
            ge=1,
            description="This period's opponent (ESPN team id); his synced roster gives the win probability. "
            "Needs a single league.",
        ),
    ] = None,
) -> dict[str, Any]:
    config = _config()
    at = _parse_as_of(as_of)
    selected = _select(config, league)
    if opponent_team_id is not None and len(selected) != 1:
        raise ToolError("opponent_team_id names one league's opponent; pass league as well")
    with Store.open() as store:
        blocks = [_lineup_block(store, config, configured, at=at, opponent=opponent_team_id) for configured in selected]
    return {"as_of": _time(at), "leagues": blocks}


# --- waivers ----------------------------------------------------------------------------------------------------------


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
        out["policy"] = _verdict(
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


def _waiver_block(
    store: Store, config: Config, configured: League, *, at: datetime, top: int, max_moves: int | None
) -> dict[str, Any]:
    key = configured.key
    row = store.leagues.by_key(key)
    if row is None:
        return {"league": key, "ok": False, "error": "not synced; run fm sync"}
    if registry.get(row.sport, WAIVERS_KIND) is None:
        return {"league": key, "ok": False, "error": f"no waivers decision is registered for {row.sport}"}
    schedule, _, note = _cached_schedule(store, configured)
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


def _waivers(
    league: LeagueArg = None,
    as_of: AsOfArg = None,
    top: Annotated[int, Field(ge=0, le=100, description="Ranked (add, drop) pairs to return.")] = 10,
    max_moves: Annotated[int | None, Field(ge=1, description="Plan at most this many moves.")] = None,
) -> dict[str, Any]:
    config = _config()
    at = _parse_as_of(as_of)
    with Store.open() as store:
        blocks = [
            _waiver_block(store, config, configured, at=at, top=top, max_moves=max_moves)
            for configured in _select(config, league)
        ]
    return {"as_of": _time(at), "leagues": blocks}


# --- trades -----------------------------------------------------------------------------------------------------------


def _trade_side(ctx: TradeContext, side: Any) -> dict[str, Any]:
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


def _trade_eval(
    deal: Annotated[str, Field(description='The deal, as in "give A, B get C": players by name or ESPN id.')],
    league: LeagueArg = None,
    as_of: AsOfArg = None,
    runs: Annotated[
        int, Field(ge=100, le=50_000, description="Simulated seasons (only when the league's schedule is captured).")
    ] = DEFAULT_TRADE_RUNS,
    seed: Annotated[
        int, Field(description="The simulation's seed; the same seed gives the same numbers.")
    ] = DEFAULT_SEED,
) -> dict[str, Any]:
    config = _config()
    at = _parse_as_of(as_of)
    chosen = _select(config, league)
    if len(chosen) != 1:
        raise ToolError("several leagues are configured; pass league")
    configured = chosen[0]
    with Store.open() as store:
        row = store.leagues.by_key(configured.key)
        if row is None:
            raise ToolError(f"{configured.key}: not synced; run fm sync")
        schedule, _, schedule_note = _cached_schedule(store, configured)
        matchups, matchups_note = _cached_matchups(store, configured, row)
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
            raise ToolError(f"{configured.key}: {exc}") from None
        return _trade_dict(ctx, found, at=at, notes=notes)


# --- the proposal queue -----------------------------------------------------------------------------------------------


def _list_proposals(
    league: LeagueArg = None,
    include_closed: Annotated[
        bool, Field(description="Include proposals already decided or finished (rejected, expired, verified, failed).")
    ] = False,
    limit: Annotated[int, Field(ge=1, le=200, description="At most this many, newest first.")] = 50,
) -> dict[str, Any]:
    config = _config()
    selected = _select(config, league)
    out: list[dict[str, Any]] = []
    with Store.open() as store:
        for configured in selected:
            row = store.leagues.by_key(configured.key)
            if row is None:
                continue
            rows = store.proposals.find(league_id=row.row_id) if include_closed else store.proposals.open(row.row_id)
            out.extend(_proposal(found, configured.key) for found in rows)
    out.sort(key=lambda item: (item["created_at"] or "", item["id"] or 0), reverse=True)
    return {"count": len(out), "proposals": out[:limit]}


def _dedupe_key(league: LeagueRow, kind: ProposalKind, payload: Mapping[str, Any]) -> str:
    """One open proposal per league, kind and payload: asking twice finds the first."""
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"{MCP_CREATED_BY}:{league.key}:{league.season}:{kind.value}:{digest}"


def _create_proposal(
    league: Annotated[str, Field(description="League key from config.toml (for example 'ffl' or 'fba').")],
    kind: Annotated[
        str,
        Field(
            description="Proposal kind: bench_inactive, lineup, add_drop, waiver, waiver_cancel, trade_propose, "
            "trade_accept, trade_decline or trade_cancel."
        ),
    ],
    payload: Annotated[
        dict[str, Any],
        Field(
            description="The move, as ESPN ids. lineup and bench_inactive: {moves: [{espn_id, from_slot_id, "
            "to_slot_id}]}. add_drop: {add_espn_id, drop_espn_id}. waiver: {add_espn_id, drop_espn_id, bid_amount}. "
            "trade_propose: {other_team_id, give_espn_ids, get_espn_ids}. trade_accept and trade_decline add "
            "espn_transaction_id. waiver_cancel and trade_cancel: {espn_transaction_id}."
        ),
    ],
    rationale: Annotated[str, Field(min_length=1, max_length=MAX_RATIONALE, description="Why, in a few sentences.")],
    scoring_period_id: Annotated[
        int | None, Field(ge=1, description="The scoring period the move is for (counts toward that week's cap).")
    ] = None,
    deadline: Annotated[
        str | None,
        Field(description="ISO 8601 time after which the move stops making sense (lock, waiver run). Default: none."),
    ] = None,
) -> dict[str, Any]:
    config = _config()
    try:
        spec = kind_spec(kind)
    except PolicyError as exc:
        return _refused("invalid", [str(exc)])
    try:
        model = spec.payload_type.model_validate(payload)
    except ValidationError as exc:
        reasons = [f"payload.{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        return _refused("invalid", reasons)
    try:
        due = None if deadline is None else as_utc(_parse_as_of(deadline))
    except ToolError as exc:
        return _refused("invalid", [str(exc).replace("as_of", "deadline")])
    known = {configured.key for configured in config.leagues}
    if league not in known:
        return _refused(
            "invalid", [f"no league {league!r} in config.toml; known: {', '.join(sorted(known)) or 'none'}"]
        )
    at = _now()
    with Store.open() as store:
        row = store.leagues.by_key(league)
        if row is None:
            return _refused("invalid", [f"{league}: not synced; run fm sync"])
        open_before = {found.row_id for found in store.proposals.open(row.row_id)}
        try:
            with store.db.transaction():
                stored = propose(
                    store,
                    config,
                    row,
                    spec.kind,
                    model,
                    created_by=MCP_CREATED_BY,
                    scoring_period_id=scoring_period_id,
                    engine_numbers={"origin": MCP_CREATED_BY},
                    rationale=rationale,
                    deadline=due,
                    dedupe_key=_dedupe_key(row, spec.kind, model.model_dump(mode="json")),
                    now=at,
                )
                note = None
                if stored.row_id not in open_before and stored.policy == "auto":
                    # Only the engine's own drafts may be approved on their own at T-15, never a move Claude suggested.
                    stored = store.proposals.update(stored.model_copy(update={"policy": "approve"}))
                    note = (
                        "policy auto was lowered to approve: a suggestion made through MCP is never approved on its own"
                    )
        except PolicyError as exc:
            verdict = evaluate(
                store, config, row, spec.kind, model, scoring_period_id=scoring_period_id, deadline=due, now=at
            )
            return _refused("blocked", list(verdict.reasons) or [str(exc)], policy=_verdict(verdict))
        existing = stored.row_id in open_before
        return {
            "status": "existing" if existing else "proposed",
            "proposal": _proposal(stored, league),
            "reasons": [],
            "note": note,
            "message": (f"already open as #{stored.row_id}" if existing else f"stored as proposal #{stored.row_id}")
            + f" (policy {stored.policy}); it waits for the user's review and nothing has been changed in ESPN",
        }


def _refused(status: str, reasons: list[str], *, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "proposal": None,
        "policy": policy,
        "reasons": reasons,
        "note": None,
        "message": "nothing was stored" + (": " + "; ".join(reasons) if reasons else ""),
    }


# --- the server -------------------------------------------------------------------------------------------------------


def build_server() -> FastMCP:
    """A fresh server with exactly the tools in :data:`TOOL_NAMES`."""
    server = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)
    server.tool(
        _status,
        name="status",
        description="Per league: sync state, our record and FAAB, the captured pro schedule, the kill-switch state "
        "and the open proposals.",
        annotations=READ_ONLY,
    )
    server.tool(
        _lineup,
        name="lineup",
        description="Plan this scoring period's lineup from the synced roster: the recommended starters per slot with "
        "projections, expected points and win probability, the moves from the current lineup and the proposals the "
        "engine would suggest. A plan only; nothing is stored.",
        annotations=READ_ONLY,
    )
    server.tool(
        _waivers,
        name="waivers",
        description="Rank (add, drop) pairs for the waiver wire and free agents by rest-of-season value, with the "
        "moves the engine would suggest, the bid, replacement levels and every skipped candidate. A ranking only; "
        "nothing is stored.",
        annotations=READ_ONLY,
    )
    server.tool(
        _trade_eval,
        name="trade_eval",
        description='Judge one trade given as "give A, B get C": both sides\' rest-of-season lineup value and title '
        "odds, legality, P(accept) and a recommendation. Returns the payload to hand to create_proposal.",
        annotations=READ_ONLY,
    )
    server.tool(
        _list_proposals,
        name="list_proposals",
        description="The proposal queue with each proposal's kind, status, policy, payload and rationale.",
        annotations=READ_ONLY,
    )
    server.tool(
        _create_proposal,
        name="create_proposal",
        description="Record a suggested move for the user to review (CLI or phone). It goes through the league's "
        "policy and guardrails and is stored as proposed, or refused with every reason; duplicates return the "
        "existing one. It never changes a league: trades always need the user's explicit review.",
        annotations=STORES_PROPOSAL,
    )
    return server
