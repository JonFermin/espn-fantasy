"""Policy presets: ``preset = "hands_off" | "balanced" | "ask_first"`` on a ``[[league]]`` (DESIGN section 11).

A preset is plain data per sport: the ``[league.policy]`` and ``[league.trade_finder]`` keys it supplies. ``fm.config``
applies it while validating a league, filling only the keys the file leaves unset, so a key written in the file always
wins and a league without a preset loads exactly as it did before presets existed. The names are what ``fm setup``
offers and ``fm config set-preset`` writes; ``PRESET_DESCRIPTIONS`` is the one-line pitch each shows.

CLAUDE.md's invariants hold here as they do in ``fm.config.Policy``: no preset carries a ``trade*`` policy key (trades
are approval-only and never configurable) and none sets ``add_drop`` or ``waiver`` to ``auto``. NBA ``add`` stays
``approve`` in every preset because NBA streaming reports no news-free gain yet. ``tests/test_presets.py`` holds the
table to those rules, so a new preset cannot loosen them unnoticed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fm.config import Sport

PresetName = Literal["hands_off", "balanced", "ask_first"]
PRESET_NAMES: tuple[PresetName, ...] = ("hands_off", "balanced", "ask_first")

PRESET_DESCRIPTIONS: Mapping[PresetName, str] = MappingProxyType(
    {
        "hands_off": "lineups are set on their own; in NFL a clearly good free-agent add fires by itself; "
        "drops, claims and trades wait for your approval",
        "balanced": "lineups are set on their own; every add, drop, claim and trade waits for your approval",
        "ask_first": "every move waits for your approval, and the trade finder stays off",
    }
)


@dataclass(frozen=True)
class Preset:
    """The keys one preset supplies for one sport. Missing keys fall through to the model defaults."""

    name: PresetName
    sport: Sport
    policy: Mapping[str, object]
    trade_finder: Mapping[str, object]

    @property
    def description(self) -> str:
        return PRESET_DESCRIPTIONS[self.name]


# The table in DESIGN section 11, one row per (preset, sport). Lineup kinds are the only ones that may be auto
# besides add; add_drop and waiver are approve everywhere; the weekly cap follows the sport's cadence.
_POLICY: Mapping[PresetName, Mapping[Sport, Mapping[str, object]]] = {
    "hands_off": {
        "nfl": {
            "bench_inactive": "auto",
            "lineup": "auto",
            "add": "auto",
            "auto_add_min_gain": 15,
            "add_drop": "approve",
            "waiver": "approve",
            "max_transactions_per_week": 3,
        },
        "nba": {
            "bench_inactive": "auto",
            "lineup": "auto",
            "add": "approve",
            "add_drop": "approve",
            "waiver": "approve",
            "max_transactions_per_week": 7,
        },
    },
    "balanced": {
        "nfl": {
            "bench_inactive": "auto",
            "lineup": "auto",
            "add": "approve",
            "add_drop": "approve",
            "waiver": "approve",
            "max_transactions_per_week": 3,
        },
        "nba": {
            "bench_inactive": "auto",
            "lineup": "auto",
            "add": "approve",
            "add_drop": "approve",
            "waiver": "approve",
            "max_transactions_per_week": 7,
        },
    },
    "ask_first": {
        "nfl": {
            "bench_inactive": "approve",
            "lineup": "approve",
            "add": "approve",
            "add_drop": "approve",
            "waiver": "approve",
            "max_transactions_per_week": 3,
        },
        "nba": {
            "bench_inactive": "approve",
            "lineup": "approve",
            "add": "approve",
            "add_drop": "approve",
            "waiver": "approve",
            "max_transactions_per_week": 7,
        },
    },
}

# The trade finder only drafts offers for approval (never sends one), so the active presets turn it on with stricter
# thresholds than the model defaults; ask_first leaves it off and the thresholds at their defaults.
_FINDER_ON: Mapping[str, object] = {
    "enabled": True,
    "min_accept": 0.5,
    "min_title_gain": 0.015,
    "min_ros_gain": 0.3,
    "max_offers": 1,
}
_TRADE_FINDER: Mapping[PresetName, Mapping[str, object]] = {
    "hands_off": _FINDER_ON,
    "balanced": _FINDER_ON,
    "ask_first": {"enabled": False},
}


def preset_for(name: str, sport: str) -> Preset:
    """The preset ``name`` for ``sport``. ``ValueError`` names the known presets or sports."""
    for preset_name, by_sport in _POLICY.items():
        if preset_name != name:
            continue
        for known_sport, policy in by_sport.items():
            if known_sport == sport:
                return Preset(
                    name=preset_name,
                    sport=known_sport,
                    policy=MappingProxyType(dict(policy)),
                    trade_finder=MappingProxyType(dict(_TRADE_FINDER[preset_name])),
                )
        raise ValueError(f"no {name} preset for sport {sport!r}; known: {', '.join(by_sport)}")
    raise ValueError(f"unknown preset {name!r}; choose one of {', '.join(PRESET_NAMES)}")
