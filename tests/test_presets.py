"""The preset table (PLUG_AND_PLAY section 1) and the invariants no preset may loosen.

Loading a preset league through ``config.toml`` is covered by ``tests/test_config.py``; this file holds the data
itself to the spec table and to CLAUDE.md: no ``trade*`` policy key, no ``auto`` add_drop or waiver, NBA ``add``
never ``auto``.
"""

from __future__ import annotations

import pytest

from fm.config import ESPN_GAME, Policy, TradeFinder
from fm.presets import PRESET_DESCRIPTIONS, PRESET_NAMES, Preset, preset_for

SPORTS = tuple(ESPN_GAME)
EVERY = [(name, sport) for name in PRESET_NAMES for sport in SPORTS]


def ids(name: str, sport: str) -> str:
    return f"{name}-{sport}"


class TestInvariants:
    @pytest.mark.parametrize(("name", "sport"), EVERY, ids=[ids(*case) for case in EVERY])
    def test_no_preset_holds_a_trade_key_or_an_auto_drop_or_claim(self, name: str, sport: str) -> None:
        preset = preset_for(name, sport)
        assert not [key for key in preset.policy if key.startswith("trade")]
        assert preset.policy["add_drop"] == "approve" and preset.policy["waiver"] == "approve"
        assert "auto" not in (preset.policy["add_drop"], preset.policy["waiver"])

    @pytest.mark.parametrize(("name", "sport"), EVERY, ids=[ids(*case) for case in EVERY])
    def test_every_preset_validates_on_its_own(self, name: str, sport: str) -> None:
        preset = preset_for(name, sport)
        Policy.model_validate(dict(preset.policy))  # the Policy validator rejects trade keys and an auto add alone
        TradeFinder.model_validate(dict(preset.trade_finder))

    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_nba_add_stays_approve(self, name: str) -> None:
        assert preset_for(name, "nba").policy["add"] == "approve"
        assert "auto_add_min_gain" not in preset_for(name, "nba").policy

    def test_presets_are_immutable(self) -> None:
        preset = preset_for("balanced", "nfl")
        with pytest.raises(TypeError):
            preset.policy["lineup"] = "off"  # type: ignore[index]  # the point of the test
        with pytest.raises(AttributeError):
            preset.name = "ask_first"  # type: ignore[misc]
        assert preset_for("balanced", "nfl").policy == preset.policy  # a fresh copy each call, same data


class TestTable:
    def test_hands_off(self) -> None:
        nfl = preset_for("hands_off", "nfl")
        assert nfl.policy == {
            "bench_inactive": "auto",
            "lineup": "auto",
            "add": "auto",
            "auto_add_min_gain": 15,
            "add_drop": "approve",
            "waiver": "approve",
            "max_transactions_per_week": 3,
        }
        nba = preset_for("hands_off", "nba")
        assert nba.policy == {
            "bench_inactive": "auto",
            "lineup": "auto",
            "add": "approve",
            "add_drop": "approve",
            "waiver": "approve",
            "max_transactions_per_week": 7,
        }
        for preset in (nfl, nba):
            assert preset.trade_finder == {
                "enabled": True,
                "min_accept": 0.5,
                "min_title_gain": 0.015,
                "min_ros_gain": 0.3,
                "max_offers": 1,
            }

    def test_balanced_is_hands_off_without_the_auto_add(self) -> None:
        for sport in SPORTS:
            balanced, hands_off = preset_for("balanced", sport), preset_for("hands_off", sport)
            assert balanced.policy["add"] == "approve" and "auto_add_min_gain" not in balanced.policy
            assert {k: v for k, v in balanced.policy.items() if k != "add"} == {
                k: v for k, v in hands_off.policy.items() if k not in ("add", "auto_add_min_gain")
            }
            assert balanced.trade_finder == hands_off.trade_finder

    def test_ask_first_approves_everything_and_leaves_the_finder_off(self) -> None:
        for sport, cap in (("nfl", 3), ("nba", 7)):
            preset = preset_for("ask_first", sport)
            assert preset.policy == {
                "bench_inactive": "approve",
                "lineup": "approve",
                "add": "approve",
                "add_drop": "approve",
                "waiver": "approve",
                "max_transactions_per_week": cap,
            }
            assert preset.trade_finder == {"enabled": False}  # the thresholds stay at the model defaults
            assert TradeFinder.model_validate(dict(preset.trade_finder)) == TradeFinder(enabled=False)

    def test_weekly_cap_follows_the_sport(self) -> None:
        for name in PRESET_NAMES:
            assert preset_for(name, "nfl").policy["max_transactions_per_week"] == 3
            assert preset_for(name, "nba").policy["max_transactions_per_week"] == 7


class TestLookup:
    def test_names_and_descriptions(self) -> None:
        assert PRESET_NAMES == ("hands_off", "balanced", "ask_first")
        assert set(PRESET_DESCRIPTIONS) == set(PRESET_NAMES)
        for name in PRESET_NAMES:
            for sport in SPORTS:
                preset = preset_for(name, sport)
                assert isinstance(preset, Preset)
                assert (preset.name, preset.sport) == (name, sport)
                assert preset.description == PRESET_DESCRIPTIONS[name]
                assert preset.description and "\n" not in preset.description

    def test_unknown_preset_or_sport(self) -> None:
        with pytest.raises(ValueError, match="unknown preset 'yolo'; choose one of hands_off, balanced, ask_first"):
            preset_for("yolo", "nfl")
        with pytest.raises(ValueError, match="no balanced preset for sport 'mlb'; known: nfl, nba"):
            preset_for("balanced", "mlb")
