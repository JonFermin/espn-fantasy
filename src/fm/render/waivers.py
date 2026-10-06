"""``fm waivers``: a :class:`fm.decide.waivers.WaiverDecision` as text.

The block names the wire the decision read, the moves it proposes with their policy verdict (one string per move,
computed by the command), the numbers behind each (gain, rest-of-season values, value over replacement, bid), then the
top of the ranking so the next-best pairs are visible, the replacement level per slot, and every warning: the
decision names each skipped candidate and why.
"""

from __future__ import annotations

from collections.abc import Sequence

from fm.decide.waivers import WaiverDecision, WaiverMove
from fm.model.valuation import PlayerOutlook
from fm.proposals import ProposalKind
from fm.render import INDENT, Align, columns, points, signed, stamp, warning_lines

_RANK_ALIGN: tuple[Align, ...] = (">", "<", "<", "<", ">", ">", ">", ">", ">", "<")


def waiver_lines(decision: WaiverDecision, *, verdicts: Sequence[str], as_of: str, top: int) -> list[str]:
    """The whole block for one league. ``verdicts`` has one line per move, in ``decision.moves`` order."""
    league, wire = decision.league, decision.wire
    lines = [
        f"{league.key}: {league.name or 'unnamed league'}, waivers and free agents for scoring period "
        f"{decision.scoring_period} (as of {as_of}; wire of {len(wire)} synced {stamp(wire.as_of, none='never')})"
    ]
    if not decision.moves:
        lines.append(f"{INDENT}no move worth proposing")
    for move, verdict in zip(decision.moves, verdicts, strict=True):
        lines.extend(_move(move, verdict))
    ranked = decision.ranked[:top]
    if ranked:
        lines.append("")
        lines.append(f"{INDENT}top {len(ranked)} of {len(decision.ranked)} ranked pairs (rest-of-season points):")
        lines.extend(_ranking(ranked))
    replacement = decision.replacement
    if replacement:
        levels = ", ".join(
            f"{level.label} {level.name or 'nobody'} {points(level.value)}" for level in replacement.values()
        )
        lines.append(f"{INDENT}replacement level: {levels}")
    lines.extend(warning_lines(decision.warnings))
    return lines


def _label(outlook: PlayerOutlook) -> str:
    return f"{outlook.name} ({outlook.position})" if outlook.position else outlook.name


def _move(move: WaiverMove, verdict: str) -> list[str]:
    lines = [f"{INDENT}{move.kind.value}: {verdict}; due {stamp(move.deadline)}", f"{INDENT * 2}{move.rationale()}"]
    detail = (
        f"gain {signed(move.gain)} from period {move.start}; add {_label(move.add)} ROS {points(move.add_value)}"
        f" (VOR {signed(move.add_vor or 0.0)})"
    )
    if move.drop is not None:
        detail += f"; drop {_label(move.drop)} ROS {points(move.drop_value)} (VOR {signed(move.drop_vor or 0.0)})"
    lines.append(f"{INDENT * 2}{detail}")
    if move.bidding is not None and move.bid is not None:
        bidding = move.bidding
        lines.append(
            f"{INDENT * 2}bid ${move.bid}: {bidding.share:.0%} of the roster's ROS value buys the "
            f"${bidding.budget_left} left; cap ${bidding.cap}, minimum ${bidding.minimum_bid}"
        )
    return lines


def _ranking(ranked: Sequence[WaiverMove]) -> list[str]:
    rows: list[Sequence[str]] = [("#", "move", "add", "drop", "gain", "add ROS", "VOR", "drop ROS", "bid", "due")]
    for index, move in enumerate(ranked, start=1):
        rows.append(
            (
                str(index),
                "claim" if move.kind is ProposalKind.WAIVER else "add",
                _label(move.add),
                "-" if move.drop is None else _label(move.drop),
                signed(move.gain),
                points(move.add_value),
                signed(move.add_vor or 0.0),
                points(move.drop_value),
                "-" if move.bid is None else f"${move.bid}",
                stamp(move.deadline),
            )
        )
    return columns(rows, align=_RANK_ALIGN)
