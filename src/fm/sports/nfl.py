"""NFL plugin for ESPN ``ffl`` leagues: weekly scoring periods, locks at kickoff, and slot eligibility including
``FLEX`` (RB/WR/TE) and ``OP`` (QB/RB/WR/TE, the superflex slot).

Slot eligibility follows ESPN's position-to-slot rules for ``ffl``, by position label from
:data:`fm.espn.ids.FFL_POSITIONS`; the ids come from :data:`fm.espn.ids.FFL`, never literals. A player ESPN lists with
extra eligibility (a TE also eligible at WR) carries that in his own ``eligibleSlots``, which callers prefer when they
have it; this table is the sport rule and the fallback. ``TQB`` (slot 1, team quarterback) is not supported: ESPN no
longer offers it to new leagues and no position in the id maps fills it.

Lock times come from the ESPN pro schedule (any :class:`fm.sports.base.ScheduleLike`), never from a weekday or a
kickoff time: Thursday, Friday and Saturday games, the international 9:30 a.m. ET slot and Monday night all fall out
of each game's start. The scoring period is the NFL week (ESPN's ``scoringPeriodId``), and a week gives way to the
next :data:`NFL_GAME_DURATION` after its last kickoff, which is how a tick finds the current week between syncs. The
league's lock type (per game or everyone at the week's first kickoff) is read from its settings and passed in by the
caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from types import MappingProxyType
from typing import Final

from fm.espn.ids import FFL, Game
from fm.sports.base import PeriodKind, SportPlugin

NFL_GAME_DURATION: Final = timedelta(hours=4)
"""How long after kickoff an NFL game is assumed to run (about 3 h 15 min typical, longer with overtime or a delay);
only used to decide when one week's lineups give way to the next."""

_SLOT_POSITIONS_BY_LABEL: Mapping[str, frozenset[str]] = {
    "QB": frozenset({"QB"}),
    "RB": frozenset({"RB"}),
    "RB/WR": frozenset({"RB", "WR"}),
    "WR": frozenset({"WR"}),
    "WR/TE": frozenset({"WR", "TE"}),
    "TE": frozenset({"TE"}),
    "OP": frozenset({"QB", "RB", "WR", "TE"}),  # offensive player: the superflex slot
    "RB/WR/TE": frozenset({"RB", "WR", "TE"}),  # FLEX
    "DT": frozenset({"DT"}),
    "DE": frozenset({"DE"}),
    "LB": frozenset({"LB"}),
    "DL": frozenset({"DT", "DE"}),
    "CB": frozenset({"CB"}),
    "S": frozenset({"S"}),
    "DB": frozenset({"CB", "S"}),
    "DP": frozenset({"DT", "DE", "LB", "CB", "S"}),  # any individual defensive player
    "D/ST": frozenset({"D/ST"}),
    "K": frozenset({"K"}),
    "P": frozenset({"P"}),
    "HC": frozenset({"HC"}),
}


def _by_slot_id(table: Mapping[str, frozenset[str]]) -> Mapping[int, frozenset[str]]:
    """Resolve slot and position labels through the ffl id maps, so a typo fails at import rather than in a lineup."""
    resolved: dict[int, frozenset[str]] = {}
    for slot_label, positions in table.items():
        for position in positions:
            FFL.position_id(position)  # KeyError on an unknown position label
        resolved[FFL.slot_id(slot_label)] = positions
    return MappingProxyType(resolved)


NFL_SLOT_POSITIONS: Mapping[int, frozenset[str]] = _by_slot_id(_SLOT_POSITIONS_BY_LABEL)
"""Active ``ffl`` lineup slot id to the position labels that may fill it (bench and IR take every position)."""


class NflPlugin(SportPlugin):
    """ESPN fantasy football: weekly periods, per-game locks at kickoff, FLEX and OP eligibility."""

    game = Game.FFL
    sport = "nfl"
    period_kind = PeriodKind.WEEK
    game_duration = NFL_GAME_DURATION

    @property
    def slot_positions(self) -> Mapping[int, frozenset[str]]:
        return NFL_SLOT_POSITIONS


NFL: Final = NflPlugin()
PLUGIN: Final = NFL  # discovered by fm.sports.base.plugin_for("nfl")
