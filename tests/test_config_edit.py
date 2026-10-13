"""``fm.config_edit``: rendering a fresh file, appending leagues, setting values, and the write rules (validate before
replace, atomic, ``config.toml.bak``, comments and key order kept, CRLF kept).

Every test works on a temp file: the real ``~/.config/espn-fantasy`` is never touched (conftest points
``FM_CONFIG_DIR`` elsewhere, and these tests pass explicit paths anyway).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import fm.config_edit as config_edit
from fm.config import Config, ConfigError, League, Llm, Notify, Policy, TradeFinder, load_config
from fm.config_edit import (
    TEMP_PREFIX,
    AddLeaguesResult,
    LeagueUpdate,
    add_leagues,
    backup_path,
    edit_lock_path,
    preset_overrides,
    remove_keys,
    render_config,
    set_notify_channel,
    set_value,
    set_values,
    write_config,
)
from fm.jobs.runlock import RunLock, exclusive_lock

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


def temps(path: Path) -> list[Path]:
    """Temp files an atomic write left beside ``path`` (there should never be any)."""
    return sorted(path.parent.glob(TEMP_PREFIX + "*"))


class TestWrite:
    def test_fresh_file_has_no_backup(self, tmp_path: Path) -> None:
        path = tmp_path / "new" / "config.toml"
        config = write_config(path, render_config([NFL]))
        assert config.leagues[0].key == "nfl" and load(path) == config
        assert not backup_path(path).exists()
        assert temps(path) == []

    def test_replace_keeps_the_previous_file_as_bak(self, config_path: Path) -> None:
        before = config_path.read_bytes()
        write_config(config_path, render_config([NFL]))
        assert backup_path(config_path).read_bytes() == before
        assert config_path.read_bytes() != before
        assert temps(config_path) == []

    def test_invalid_text_leaves_the_file_byte_identical(self, config_path: Path) -> None:
        before = config_path.read_bytes()
        with pytest.raises(ConfigError, match="league\\[0\\]\\.policy\\.lineup: Input should be 'off'") as info:
            write_config(config_path, COMMENTED.replace('lineup = "approve"', 'lineup = "always"'))
        assert str(config_path) in str(info.value)  # located at the real file, not a temp one
        assert config_path.read_bytes() == before
        assert not backup_path(config_path).exists()
        assert temps(config_path) == []


class TestConcurrentWriters:
    """Two processes editing config.toml (fm setup, the dashboard, fm config set-preset) never share a temp file and
    never lose each other's edit."""

    def test_each_write_uses_its_own_temp_file_beside_the_target(
        self, config_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stray = config_path.with_name("config.toml.tmp")
        stray.write_text("another writer's temp file", encoding="utf-8")  # the old fixed name: never touched now
        moved: list[Path] = []
        real_replace = os.replace

        def spy(src: Any, dst: Any) -> None:
            moved.append(Path(src))
            real_replace(src, dst)

        monkeypatch.setattr(config_edit.os, "replace", spy)
        set_value(config_path, "nfl", "preset", "balanced")
        set_value(config_path, "nfl", "preset", "ask_first")
        first, second = moved
        assert first != second and all(path.parent == config_path.parent for path in moved)
        assert all(path.name.startswith(TEMP_PREFIX) for path in moved)
        assert stray.read_text(encoding="utf-8") == "another writer's temp file"
        assert temps(config_path) == []

    def test_an_edit_waits_for_another_writer_and_applies_on_top_of_its_change(self, config_path: Path) -> None:
        other = RunLock(edit_lock_path(config_path), command="another fm process").acquire()
        done: list[Config] = []
        failed: list[BaseException] = []

        def edit() -> None:
            try:
                done.append(set_value(config_path, "nfl", "policy.lineup", "auto"))
            except BaseException as exc:  # surfaced below
                failed.append(exc)

        worker = threading.Thread(target=edit)
        worker.start()
        time.sleep(0.5)
        assert done == [] and failed == []  # waiting: it has not read the file yet
        # The other writer's edit lands while ours waits; ours must read the file only once it has the lock.
        config_path.write_text(COMMENTED.replace("daily_budget_usd = 1.5", "daily_budget_usd = 3.0"), encoding="utf-8")
        other.release()
        worker.join(timeout=10)
        assert failed == [] and len(done) == 1
        config = load(config_path)
        assert config.llm.daily_budget_usd == 3.0 and config.league("nfl").policy.lineup == "auto"  # both edits kept

    def test_a_writer_that_never_lets_go_is_reported_not_waited_on_forever(
        self, config_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config_edit, "EDIT_LOCK_TIMEOUT_S", 0.3)
        before = config_path.read_bytes()
        with exclusive_lock(edit_lock_path(config_path), command="fm ui", timeout=0):
            with pytest.raises(ConfigError, match=r"is being edited by fm ui \(pid \d+\)"):
                set_value(config_path, "nfl", "preset", "balanced")
        assert config_path.read_bytes() == before

    def test_a_replace_refused_once_is_retried(self, config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        real_replace = os.replace
        calls: list[Path] = []

        def busy_once(src: Any, dst: Any) -> None:
            calls.append(Path(dst))
            if len(calls) == 1:
                raise PermissionError(13, "The process cannot access the file because it is being used")
            real_replace(src, dst)

        monkeypatch.setattr(config_edit.os, "replace", busy_once)
        assert set_value(config_path, "nfl", "preset", "balanced").league("nfl").preset == "balanced"
        assert calls == [config_path, config_path] and temps(config_path) == []

    def test_a_replace_refused_twice_fails_cleanly(self, config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def always_busy(src: Any, dst: Any) -> None:
            raise PermissionError(13, "being used by another process")

        before = config_path.read_bytes()
        monkeypatch.setattr(config_edit.os, "replace", always_busy)
        with pytest.raises(ConfigError, match="being used by another process"):
            set_value(config_path, "nfl", "preset", "balanced")
        assert config_path.read_bytes() == before and temps(config_path) == []

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
        assert add_leagues(config_path, entries) == AddLeaguesResult(added=("work", "nba2"), unchanged=("nfl",))
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
        result = add_leagues(config_path, [NFL, {**NBA, "team_id": 1}])
        assert result.added == () and result.unchanged == ("nfl",)
        assert config_path.read_bytes() == before
        assert not backup_path(config_path).exists()

    def test_a_present_league_whose_season_or_team_moved_is_reported_not_rewritten(self, config_path: Path) -> None:
        """Re-running setup after ESPN renewed a league (new season) or after the team was corrected: add_leagues
        says what differs and leaves the file alone; the caller updates it through set_values."""
        before = config_path.read_bytes()
        renewed = {**NFL, "key": "football", "season": 2027}
        moved = {**NBA, "team_id": 1}
        result = add_leagues(config_path, [renewed, moved])
        assert result == AddLeaguesResult(
            changed=(
                LeagueUpdate(key="nfl", fields={"season": (2026, 2027)}),
                LeagueUpdate(key="nba", fields={"team_id": (9, 1)}),
            )
        )
        assert config_path.read_bytes() == before  # add_leagues never changes a league that is present
        for update in result.changed:
            set_values(config_path, update.key, update.updates)
        config = load(config_path)
        assert config.league("nfl").season == 2027 and config.league("nba").team_id == 1
        assert "# My leagues. Hand-written" in config_path.read_text(encoding="utf-8")

    def test_a_new_league_and_a_moved_one_in_one_call(self, config_path: Path) -> None:
        extra = {"key": "work", "sport": "nfl", "espn_league_id": 55, "season": 2026, "team_id": 2}
        result = add_leagues(config_path, [extra, {**NBA, "season": 2028}])
        assert result.added == ("work",) and result.unchanged == ()
        assert result.changed == (LeagueUpdate(key="nba", fields={"season": (2027, 2028)}),)
        assert load(config_path).league("nba").season == 2027  # appended "work" only; nba untouched

    def test_bad_entry_is_rejected_and_the_file_kept(self, config_path: Path) -> None:
        before = config_path.read_bytes()
        with pytest.raises(ConfigError, match="duplicate league key 'nba'"):
            add_leagues(config_path, [{**NBA, "espn_league_id": 2}])
        with pytest.raises(ConfigError, match=r"league\[2\]\.sport: Input should be 'nfl' or 'nba'"):
            add_leagues(config_path, [{**NBA, "sport": "mlb", "espn_league_id": 2}])
        assert config_path.read_bytes() == before

    def test_file_without_leagues(self, tmp_path: Path) -> None:
        path = write(tmp_path / "config.toml", '[notify]\nchannel = "ntfy"\n')
        assert add_leagues(path, [NFL]).added == ("nfl",)
        config = load(path)
        assert config.leagues[0] == League(**NFL) and config.notify.channel == "ntfy"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            add_leagues(tmp_path / "nope.toml", [NFL])
        assert list(tmp_path.iterdir()) == []  # no lock file for a file that is not there


class TestPresetOverrides:
    def test_lists_written_keys_that_differ_from_the_preset(self, config_path: Path) -> None:
        # nfl writes bench_inactive = "auto", lineup = "approve" and untouchables, and has no preset yet.
        assert preset_overrides(config_path, "nfl") == []
        assert preset_overrides(config_path, "nfl", preset="ask_first") == ["policy.bench_inactive"]
        assert preset_overrides(config_path, "nfl", preset="balanced") == ["policy.lineup"]
        set_value(config_path, "nfl", "preset", "hands_off")
        assert preset_overrides(config_path, "nfl") == ["policy.lineup"]  # untouchables is not the preset's key
        assert preset_overrides(config_path, "nba", preset="hands_off") == []

    def test_reads_a_raw_league_table_too(self) -> None:
        table = {
            "key": "nfl",
            "sport": "nfl",
            "preset": "ask_first",
            "policy": {"bench_inactive": "auto", "lineup": "approve", "max_transactions_per_week": 3},
            "trade_finder": {"enabled": True, "min_accept": 0.6},
        }
        assert preset_overrides(table) == ["policy.bench_inactive", "trade_finder.enabled"]
        assert preset_overrides({**table, "preset": "balanced"}) == ["policy.lineup", "trade_finder.min_accept"]
        assert preset_overrides({"key": "x", "sport": "nba"}) == []

    def test_unknown_league_or_preset(self, config_path: Path) -> None:
        with pytest.raises(ConfigError, match="no league 'mlb'; known: nfl, nba"):
            preset_overrides(config_path, "mlb")
        with pytest.raises(ValueError, match="unknown preset 'yolo'"):
            preset_overrides(config_path, "nfl", preset="yolo")
        with pytest.raises(ValueError, match="needs the league key"):
            preset_overrides(config_path)

    def test_remove_keys_hands_them_back_to_the_preset_in_one_edit(self, config_path: Path) -> None:
        set_value(config_path, "nfl", "preset", "balanced")
        before = config_path.read_bytes()
        config = remove_keys(config_path, "nfl", ["policy.lineup", "trade_finder.enabled"])  # the second is absent
        assert config.league("nfl").policy.lineup == "auto"  # the preset decides it now
        assert backup_path(config_path).read_bytes() == before  # one edit, one backup
        text = config_path.read_text(encoding="utf-8")
        assert "lineup" not in text and "# Trades are approval-only and have no setting here.\n" in text
        assert 'bench_inactive = "auto"       # off | approve | auto\n' in text


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
        message = "notify.channel: Input should be 'telegram', 'ntfy' or 'none' \\(got 'sms'\\)"
        with pytest.raises(ConfigError, match=message):
            set_notify_channel(config_path, "sms")
        assert config_path.read_bytes() == before

    def test_none_means_no_phone_and_needs_no_channel_secrets(self, config_path: Path) -> None:
        config = set_notify_channel(config_path, "none")
        assert config.notify.channel == "none"
        assert config.required_secrets() == ("ANTHROPIC_API_KEY",)
        assert 'channel = "none"              # telegram | ntfy\n' in config_path.read_text(encoding="utf-8")


class TestNoChannel:
    def test_render_writes_channel_none_when_asked(self, tmp_path: Path) -> None:
        for notify in ("none", {"channel": "none"}):
            text = render_config([NFL], notify=notify)
            assert 'channel = "none"' in text and "telegram | ntfy | none" in text
            config = load(write(tmp_path / "config.toml", text))
            assert config.notify == Notify(channel="none") and config.missing_secrets() == ("ANTHROPIC_API_KEY",)
        assert 'channel = "telegram"' in render_config([NFL])  # the default is unchanged

    def test_open_channel_refuses_none_with_a_notify_error(self, tmp_path: Path) -> None:
        from fm.notify import NotifyError, open_channel

        config = load(write(tmp_path / "config.toml", render_config([NFL], notify="none")))
        with pytest.raises(NotifyError, match=r"no phone channel configured \(notify.channel = none\)"):
            open_channel(config)
