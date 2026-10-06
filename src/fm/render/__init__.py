"""CLI rendering: plain-text lines for recommendations and reports (``fm status``, ``fm lineup``, ``fm waivers``).

Every renderer here takes the decision modules' own result objects (:class:`fm.decide.lineup.LineupDecision`,
:class:`fm.decide.waivers.WaiverDecision`, the store rows) and returns the lines to print, so the command modules in
``fm.commands`` stay thin and the output can be snapshot-tested without a terminal. Nothing here reads the store or
the clock: timestamps are rendered as given, in UTC with a ``Z`` suffix, as ``fm proposals list`` does. Plain text,
not rich tables, so the same lines read in a terminal, in a log and in a notification.

- :mod:`fm.render.status`: one block per league (sync state, team record, FAAB, open proposals).
- :mod:`fm.render.lineup`: the recommended lineup, its expected points and win probability, the drafts and their moves.
- :mod:`fm.render.waivers`: the moves proposed, the ranked (add, drop) pairs, replacement levels and warnings.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Literal

INDENT = "  "
type Align = Literal["<", ">"]


def stamp(at: datetime | None, *, none: str = "-") -> str:
    """``YYYY-MM-DD HH:MMZ`` in UTC, or ``none`` for ``None``."""
    return none if at is None else f"{at.astimezone(UTC):%Y-%m-%d %H:%M}Z"


def points(value: float | None, *, none: str = "-") -> str:
    """A points figure with one decimal."""
    return none if value is None else f"{value:.1f}"


def signed(value: float) -> str:
    """A gain with its sign: ``+5.1``, ``-0.4``, ``+0.0``."""
    return f"{value:+.1f}"


def percent(value: float | None, *, none: str = "-") -> str:
    """A probability as a whole percentage."""
    return none if value is None else f"{value:.0%}"


def columns(rows: Iterable[Sequence[str]], *, align: Sequence[Align] = (), indent: str = INDENT) -> list[str]:
    """Pad ``rows`` (sequences of cell strings) into aligned columns, each line prefixed with ``indent``.

    ``align`` gives each column's alignment (``<`` left, ``>`` right); columns beyond it are left-aligned. Trailing
    whitespace is stripped so a short last cell leaves no padding behind.
    """
    table = [list(row) for row in rows]
    if not table:
        return []
    width = max(len(row) for row in table)
    widths = [max(len(row[i]) if i < len(row) else 0 for row in table) for i in range(width)]
    lines: list[str] = []
    for row in table:
        cells: list[str] = []
        for i, w in enumerate(widths):
            cell = row[i] if i < len(row) else ""
            side = align[i] if i < len(align) else "<"
            cells.append(f"{cell:{side}{w}}")
        lines.append((indent + " ".join(cells)).rstrip())
    return lines


def warning_lines(warnings: Iterable[str], *, indent: str = INDENT) -> list[str]:
    """``warning: ...`` lines, one per warning, as ``fm sync`` prints them."""
    return [f"{indent}warning: {warning}" for warning in warnings]


__all__ = ["INDENT", "Align", "columns", "percent", "points", "signed", "stamp", "warning_lines"]
