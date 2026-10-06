"""``fm lineup``: a :class:`fm.decide.lineup.LineupDecision` as text.

The block shows the lineup the optimizer recommends (``decision.best``) slot by slot with each player's projection,
``p_active`` and expected points, the totals and win probability of the current and recommended lineups, then each
draft with its policy verdict (computed by the command, one string per draft), its deadline and the numbers behind
every move, and finally the decision's warnings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fm.decide.lineup import LineupCandidate, LineupDecision, LineupDraft, LineupPlan, MatchupOutlook, Objective
from fm.espn.ids import IdMaps
from fm.render import INDENT, Align, columns, percent, points, signed, stamp, warning_lines
from fm.store import LeagueRow

_DESIGNATIONS = {"OUT": "OUT", "INJURY_RESERVE": "IR", "SUSPENSION": "suspended", "DOUBTFUL": "doubtful"}
_ROSTER_ALIGN: tuple[Align, ...] = ("<", "<", "<", ">", ">", ">", "<")


def lineup_lines(
    decision: LineupDecision,
    *,
    league: LeagueRow,
    ids: IdMaps,
    team_name: str,
    opponent: str | None,
    verdicts: Sequence[str],
    as_of: str,
) -> list[str]:
    """The whole block for one league. ``verdicts`` has one line per draft, in ``decision.drafts`` order."""
    inputs = decision.inputs
    head = f"{league.key}: {league.name or 'unnamed league'}, scoring period {inputs.scoring_period_id} lineup for "
    lines = [head + f"{team_name} (as of {as_of}; roster synced {stamp(inputs.roster_as_of)})"]
    if decision.outlook is not None:
        lines.append(f"{INDENT}opponent: {opponent or 'given outlook'}, {_outlook(decision.outlook)}")
    lines.extend(_totals(decision))
    lines.append("")
    lines.extend(_roster(decision, ids))
    lines.append("")
    if not decision.drafts:
        lines.append(f"{INDENT}lineup stands: nothing to move")
    for draft, verdict in zip(decision.drafts, verdicts, strict=True):
        lines.extend(_draft(draft, verdict))
    lines.extend(warning_lines(decision.warnings))
    return lines


def _outlook(outlook: MatchupOutlook) -> str:
    return f"projected {points(outlook.opponent_mean)} points (sd {points(outlook.opponent_sd)})"


def _plan_summary(plan: LineupPlan) -> str:
    text = f"{points(plan.expected)} expected points (sd {points(plan.sd)})"
    if plan.win_probability is not None:
        text += f", win probability {percent(plan.win_probability)}"
    return text


def _totals(decision: LineupDecision) -> list[str]:
    current, best = decision.current, decision.best
    lines = [f"{INDENT}current lineup: {_plan_summary(current)}"]
    recommended = f"{INDENT}recommended:    {_plan_summary(best)} ({signed(best.expected - current.expected)})"
    if best.objective is Objective.WIN_PROBABILITY:
        recommended += ", maximizing win probability (lopsided matchup)"
    lines.append(recommended)
    if best.open_slots:
        lines.append(f"{INDENT}{len(best.open_slots)} active slot(s) stay empty: nobody eligible is available")
    if best.idle_starters:
        lines.append(f"{INDENT}{len(best.idle_starters)} starter(s) will not play and have no replacement")
    return lines


def _roster(decision: LineupDecision, ids: IdMaps) -> list[str]:
    """The recommended lineup, active slots in the league's order, then the bench and IR."""
    inputs, best = decision.inputs, decision.best
    order = [*inputs.slot_counts, ids.bench_slot, ids.ir_slot]
    rank = {slot_id: index for index, slot_id in enumerate(order)}
    rows: list[Sequence[str]] = [("slot", "player", "pos", "proj", "p(act)", "exp", "note")]
    players = sorted(inputs.players, key=lambda p: (rank.get(best.slots[p.espn_id], len(order)), -p.expected, p.label))
    for player in players:
        to_slot = best.slots[player.espn_id]
        rows.append(
            (
                ids.slot_label(to_slot),
                player.label,
                player.position or "-",
                points(player.points),
                f"{player.p_active:.2f}",
                points(player.expected),
                _note(player, to_slot, ids),
            )
        )
    return columns(rows, align=_ROSTER_ALIGN)


def _note(player: LineupCandidate, to_slot: int, ids: IdMaps) -> str:
    notes: list[str] = []
    if to_slot != player.slot_id:
        notes.append(f"from {ids.slot_label(player.slot_id)}")
    if not player.has_game:
        notes.append("no game")
    elif player.designation in _DESIGNATIONS:
        notes.append(_DESIGNATIONS[player.designation])
    if player.locked:
        notes.append("locked")
    elif player.lock_at is not None:
        notes.append(f"locks {stamp(player.lock_at)}")
    return ", ".join(notes)


def _draft(draft: LineupDraft, verdict: str) -> list[str]:
    numbers = draft.engine_numbers
    lines = [f"{INDENT}{draft.kind.value}: {verdict}; due {stamp(draft.deadline)}", f"{INDENT * 2}{draft.rationale}"]
    rows: list[Sequence[str]] = [_move_row(move) for move in numbers.get("moves", ())]
    lines.extend(columns(rows, align=("<", "<", "<", ">", ">", ">", "<"), indent=INDENT * 2))
    return lines


def _move_row(move: dict[str, Any]) -> tuple[str, ...]:
    lock = move.get("lock_at")
    return (
        f"{move.get('name') or move['espn_id']}",
        f"({move.get('position') or '-'})",
        f"{move['from']} -> {move['to']}:",
        f"{points(move['points'])} proj",
        f"x {move['p_active']:.2f} =",
        f"{points(move['expected'])} exp",
        "" if lock is None else f"locks {lock[:16].replace('T', ' ')}Z",
    )
