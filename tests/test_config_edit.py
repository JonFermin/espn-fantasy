"""``fm.config_edit``: rendering a fresh file, appending leagues, setting values, and the write rules (validate before
replace, atomic, ``config.toml.bak``, comments and key order kept, CRLF kept).

Every test works on a temp file: the real ``~/.config/espn-fantasy`` is never touched (conftest points
``FM_CONFIG_DIR`` elsewhere, and these tests pass explicit paths anyway).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fm.config import Config, ConfigError, League, Llm, Notify, Policy, TradeFinder, load_config
from fm.config_edit import (
    add_leagues,
    backup_path,
    render_config,
    set_notify_channel,
    set_value,
    write_config,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE = FIXTURES / "config.sample.toml"

NFL: dict[str, Any] = {"key": "nfl", "sport": "nfl", "espn_league_id": 1234567, "season": 2026, "team_id": 4}
NBA: dict[str, Any] = {"key": "nba", "sport": "nba", "espn_league_id": 7654321, "season": 2027, "team_id": 9}

COMMENTED = """\
# My leagues. Hand-written, so every comment here must survive an edit.

[[league]]
key = "nfl"                   # short name
sport = "nfl"
espn_league_id = 1234567
season = 2026
team_id = 4

[league.policy]
bench_inactive = "auto"       # off | approve | auto
lineup = "approve"
untouchables = ["Sample Player"]
# Trades are approval-only and have no setting here.

[[league]]
key = "nba"
sport = "nba"
espn_league_id = 7654321
season = 2027
team_id = 9

[llm]
daily_budget_usd = 1.5

[notify]
channel = "telegram"          # telegram | ntfy
"""


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return write(tmp_path / "config.toml", COMMENTED)


def load(path: Path) -> Config:
    return load_config(path, environ={})


class TestRender:
    def test_round_trips_through_load_config(self, tmp_path: Path) -> None:
        text = render_config(
            [
                {**NFL, "preset": "balanced"},
                {**NBA, "policy": {"lineup": "approve", "untouchables": ["Sample Player", 4242424]}},
            ],
            notify={"channel": "ntfy"},
        )
        config = load(write(tmp_path / "config.toml", text))
        nfl, nba = config.leagues
        assert nfl == League(**NFL, preset="balanced")
        assert nfl.policy.lineup == "auto" and nfl.trade_finder.enabled  # the preset's values
        assert nba == League(**NBA, policy=Policy(lineup="approve", untouchables=("Sample Player", 4242424)))
        assert nba.preset is None and nba.trade_finder == TradeFinder()
        assert config.llm == Llm() and config.notify == Notify(channel="ntfy")

    def test_writes_only_the_keys_it_is_given(self) -> None:
        text = render_config([{**NFL, "preset": "balanced", "policy": None}])
        assert 'preset = "balanced"' in text
        assert "\n[league.policy]\n" not in text and "lineup" not in text  # so set-preset keeps deciding the policy
        assert "[llm]" in text and "[notify]" in text  # the defaults, written out so they can be edited
        assert 'channel = "telegram"' in text and 'model = "claude-opus-5-5"' in text

    def test_is_commented(self) -> None:
        text = render_config([NFL])
        lines = text.splitlines()
        assert lines[0].startswith("# ") and "Secrets never go in this file" in text
        assert "Trades are approval-only" in text
        assert any(line.startswith('key = "nfl"') and "# short name" in line for line in lines)
        assert any(line.startswith("espn_league_id = 1234567") and "# leagueId=" in line for line in lines)
        assert any(line.startswith('sport = "nfl"') and "# nfl | nba" in line for line in lines)
        assert "\n\n\n" not in text  # single blank lines between tables

    def test_invalid_entry_is_rejected_with_its_location(self) -> None:
        with pytest.raises(ConfigError, match=r"league\[1\]\.team_id: Input should be greater than or equal to 1"):
            render_config([NFL, {**NBA, "team_id": 0}])
        with pytest.raises(ConfigError, match="duplicate league key 'nfl'"):
            render_config([NFL, {**NFL, "espn_league_id": 2}])
        with pytest.raises(ConfigError, match="trades are approval-only"):
            render_config([{**NFL, "policy": {"trade": "auto"}}])
        with pytest.raises(ConfigError, match=r"at least one \[\[league\]\]"):
            render_config([])

    def test_llm_and_notify_tables(self, tmp_path: Path) -> None:
        text = render_config([NFL], llm={"model": "claude-other", "daily_budget_usd": 0.5}, notify={"channel": "ntfy"})
        config = load(write(tmp_path / "config.toml", text))
        assert config.llm == Llm(model="claude-other", daily_budget_usd=0.5)
        assert config.notify.channel == "ntfy"


class TestWrite:
    def test_fresh_file_has_no_backup(self, tmp_path: Path) -> None:
        path = tmp_path / "new" / "config.toml"
        config = write_config(path, render_config([NFL]))
        assert config.leagues[0].key == "nfl" and load(path) == config
        assert not backup_path(path).exists()
        assert not path.with_name("config.toml.tmp").exists()

    def test_replace_keeps_the_previous_file_as_bak(self, config_path: Path) -> None:
        before = config_path.read_bytes()
        write_config(config_path, render_config([NFL]))
        assert backup_path(config_path).read_bytes() == before
        assert config_path.read_bytes() != before
        assert not config_path.with_name("config.toml.tmp").exists()

    def test_invalid_text_leaves_the_file_byte_identical(self, config_path: Path) -> None:
        before = config_path.read_bytes()
        with pytest.raises(ConfigError, match="league\\[0\\]\\.policy\\.lineup: Input should be 'off'") as info:
            write_config(config_path, COMMENTED.replace('lineup = "approve"', 'lineup = "always"'))
        assert str(config_path) in str(info.value)  # located at the real file, not a temp one
        assert config_path.read_bytes() == before
        assert not backup_path(config_path).exists()
        assert not config_path.with_name("config.toml.tmp").exists()

    def test_syntax_error_is_rejected(self, config_path: Path) -> None:
        with pytest.raises(ConfigError, match="invalid TOML"):
            write_config(config_path, "[[league]\n")


class TestAddLeagues:
    def test_appends_new_leagues_and_skips_present_ones(self, config_path: Path) -> None:
        before = config_path.read_bytes()
        entries = [
            {**NFL, "key": "renamed"},  # same (sport, espn_league_id): present, skipped
            {"key": "work", "sport": "nfl", "espn_league_id": 55, "season": 2026, "team_id": 2, "preset": "ask_first"},
            {"key": "nba2", "sport": "nba", "espn_league_id": 66, "season": 2027, "team_id": 3},
        ]
        assert add_leagues(config_path, entries) == ("work", "nba2")
        config = load(config_path)
        assert [league.key for league in config.leagues] == ["nfl", "nba", "work", "nba2"]
        assert config.league("work").preset == "ask_first" and config.league("work").policy.lineup == "approve"
        assert config.league("nba2").policy == Policy()
        text = config_path.read_text(encoding="utf-8")
        assert text.startswith(COMMENTED.split("[llm]")[0])  # the existing part is untouched, comments included
        assert text.index('key = "work"') < text.index("[llm]")  # new tables sit with the other leagues
        assert backup_path(config_path).read_bytes() == before

    def test_nothing_new_writes_nothing(self, config_path: Path) -> None:
        before = config_path.read_bytes()
        assert add_leagues(config_path, [NFL, {**NBA, "team_id": 1}]) == ()
        assert config_path.read_bytes() == before
        assert not backup_path(config_path).exists()

    def test_bad_entry_is_rejected_and_the_file_kept(self, config_path: Path) -> None:
        before = config_path.read_bytes()
        with pytest.raises(ConfigError, match="duplicate league key 'nba'"):
            add_leagues(config_path, [{**NBA, "espn_league_id": 2}])
        with pytest.raises(ConfigError, match=r"league\[2\]\.sport: Input should be 'nfl' or 'nba'"):
            add_leagues(config_path, [{**NBA, "sport": "mlb", "espn_league_id": 2}])
        assert config_path.read_bytes() == before

    def test_file_without_leagues(self, tmp_path: Path) -> None:
        path = write(tmp_path / "config.toml", '[notify]\nchannel = "ntfy"\n')
        assert add_leagues(path, [NFL]) == ("nfl",)
        config = load(path)
        assert config.leagues[0] == League(**NFL) and config.notify.channel == "ntfy"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            add_leagues(tmp_path / "nope.toml", [NFL])


class TestSetValue:
    def test_preset_keeps_comments_and_order(self, config_path: Path) -> None:
        before = config_path.read_bytes()
        config = set_value(config_path, "nfl", "preset", "hands_off")
        nfl = config.league("nfl")
        assert nfl.preset == "hands_off"
        assert (nfl.policy.bench_inactive, nfl.policy.lineup) == ("auto", "approve")  # the file's lineup wins
        assert nfl.policy.add == "auto" and nfl.policy.auto_add_min_gain == 15
        assert nfl.policy.untouchables == ("Sample Player",) and nfl.trade_finder.enabled
        text = config_path.read_text(encoding="utf-8")
        expected = COMMENTED.replace("team_id = 4\n", 'team_id = 4\npreset = "hands_off"\n')
        assert text == expected  # one line added, everything else byte for byte: comments, order, blank lines
        assert backup_path(config_path).read_bytes() == before

    def test_replacing_a_value_keeps_its_comment_line(self, config_path: Path) -> None:
        set_value(config_path, "nfl", "policy.bench_inactive", "approve")
        text = config_path.read_text(encoding="utf-8")
        assert 'bench_inactive = "approve"    # off | approve | auto\n' in text  # the comment stays in its column
        assert "# Trades are approval-only and have no setting here.\n" in text
        assert "# My leagues. Hand-written" in text
        assert load(config_path).league("nfl").policy.bench_inactive == "approve"

    def test_creates_the_table_on_the_way(self, config_path: Path) -> None:
        set_value(config_path, "nfl", "trade_finder.enabled", True)
        set_value(config_path, "nba", "policy.max_transactions_per_week", 5)
        config = load(config_path)
        assert config.league("nfl").trade_finder == TradeFinder(enabled=True)
        assert config.league("nba").policy.max_transactions_per_week == 5
        text = config_path.read_text(encoding="utf-8")
        nfl_part = text[: text.index('key = "nba"')]
        assert "[league.trade_finder]\nenabled = true\n\n" in nfl_part
        assert "[league.policy]\nmax_transactions_per_week = 5\n" in text[text.index('key = "nba"') :]

    def test_lists_and_removal(self, config_path: Path) -> None:
        set_value(config_path, "nfl", "policy.untouchables", ["Sample Player", 4242424])
        assert load(config_path).league("nfl").policy.untouchables == ("Sample Player", 4242424)
        set_value(config_path, "nfl", "policy.untouchables", None)
        assert load(config_path).league("nfl").policy.untouchables == ()
        assert "untouchables" not in config_path.read_text(encoding="utf-8")
        set_value(config_path, "nfl", "trade_finder.enabled", None)  # nothing to remove: no table, no change
        assert "[league.trade_finder]" not in config_path.read_text(encoding="utf-8")

    def test_same_value_writes_nothing(self, config_path: Path) -> None:
        before = config_path.read_bytes()
        set_value(config_path, "nfl", "policy.lineup", "approve")
        assert config_path.read_bytes() == before
        assert not backup_path(config_path).exists()

    def test_invalid_value_is_rejected_and_the_file_byte_identical(self, config_path: Path) -> None:
        before = config_path.read_bytes()
        cases = [
            ("nfl", "policy.lineup", "always", "league\\[0\\]\\.policy\\.lineup: Input should be 'off'"),
            ("nfl", "preset", "yolo", "league\\[0\\]\\.preset: Input should be 'hands_off', 'balanced' or 'ask_first'"),
            ("nfl", "policy.trade", "auto", "trades are approval-only"),
            ("nfl", "policy.add_drop", "auto", "league\\[0\\]\\.policy\\.add_drop: Input should be 'off' or 'approve'"),
            ("nfl", "policy.nope", 1, "league\\[0\\]\\.policy\\.nope: Extra inputs are not permitted"),
            ("nba", "nickname", "x", "league\\[1\\]\\.nickname: Extra inputs are not permitted"),
        ]
        for league, key, value, message in cases:
            with pytest.raises(ConfigError, match=message) as info:
                set_value(config_path, league, key, value)
            assert str(config_path) in str(info.value)
            assert config_path.read_bytes() == before
        assert not backup_path(config_path).exists()
        assert not config_path.with_name("config.toml.tmp").exists()

    def test_unknown_league(self, config_path: Path) -> None:
        with pytest.raises(ConfigError, match="no league 'mlb'; known: nfl, nba"):
            set_value(config_path, "mlb", "preset", "balanced")

    def test_a_key_that_is_not_a_table(self, config_path: Path) -> None:
        with pytest.raises(ConfigError, match="league 'nfl': season is not a table"):
            set_value(config_path, "nfl", "season.x", 1)
        with pytest.raises(ValueError, match="invalid key"):
            set_value(config_path, "nfl", "policy.", 1)

    def test_crlf_file_keeps_crlf(self, tmp_path: Path) -> None:
        path = write(tmp_path / "config.toml", COMMENTED.replace("\n", "\r\n"))
        set_value(path, "nfl", "preset", "balanced")
        set_value(path, "nfl", "trade_finder.enabled", False)
        raw = path.read_bytes()
        assert raw.count(b"\r\n") == raw.count(b"\n")
        assert b'preset = "balanced"\r\n' in raw
        assert load(path).league("nfl").preset == "balanced"

    def test_sample_fixture_survives_an_edit_unchanged_elsewhere(self, tmp_path: Path) -> None:
        path = write(tmp_path / "config.toml", SAMPLE.read_text(encoding="utf-8"))
        original = load_config(SAMPLE, environ={})
        set_value(path, "nba", "preset", "ask_first")
        edited = load(path)
        assert edited.league("nfl") == original.league("nfl")
        assert edited.league("nba").preset == "ask_first"
        assert edited.league("nba").policy.max_transactions_per_week == 7  # the explicit key
        assert edited.league("nba").trade_finder == TradeFinder(enabled=False)
        before, after = SAMPLE.read_text(encoding="utf-8"), path.read_text(encoding="utf-8")
        assert after.replace('preset = "ask_first"\n', "") == before


class TestSetNotifyChannel:
    def test_sets_the_channel_and_keeps_the_comment(self, config_path: Path) -> None:
        config = set_notify_channel(config_path, "ntfy")
        assert config.notify.channel == "ntfy"
        assert 'channel = "ntfy"              # telegram | ntfy\n' in config_path.read_text(encoding="utf-8")

    def test_creates_the_table(self, tmp_path: Path) -> None:
        path = write(tmp_path / "config.toml", COMMENTED.split("[notify]")[0])
        assert set_notify_channel(path, "ntfy").notify.channel == "ntfy"
        assert path.read_text(encoding="utf-8").endswith('daily_budget_usd = 1.5\n\n[notify]\nchannel = "ntfy"\n')

    def test_invalid_channel_is_rejected(self, config_path: Path) -> None:
        before = config_path.read_bytes()
        with pytest.raises(ConfigError, match="notify.channel: Input should be 'telegram' or 'ntfy' \\(got 'sms'\\)"):
            set_notify_channel(config_path, "sms")
        assert config_path.read_bytes() == before
