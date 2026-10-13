"""config.toml + .env loading: the sample fixture, defaults, every validation rule, dir override, ``fm config check``.

The CLI tests register ``fm.commands.config_cmd`` on a private root instead of importing ``fm.cli``, so a half-written
command module from another task cannot break this file; ``tests/test_cli.py`` covers discovery of the real root.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import typer
from pydantic import SecretStr, ValidationError
from typer.testing import CliRunner

from fm import paths
from fm.commands import config_cmd
from fm.config import (
    OPTIONAL_SECRET_ENV_VARS,
    SECRET_ENV_VARS,
    Config,
    ConfigError,
    League,
    Llm,
    Notify,
    Policy,
    Secrets,
    TradeFinder,
    config_paths,
    load_config,
)
from fm.config_edit import backup_path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE = FIXTURES / "config.sample.toml"
HOME = FIXTURES / "home" / "config.toml"

MINIMAL = """\
[[league]]
key = "nfl"
sport = "nfl"
espn_league_id = 1234567
season = 2026
team_id = 4
"""

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_inherited_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """conftest drops the API keys; also drop the channel variables so secret assertions are deterministic."""
    for name in SECRET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def write(directory: Path, text: str, name: str = "config.toml") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def secret(value: SecretStr | None) -> str | None:
    return None if value is None else value.get_secret_value()


def _root() -> None:
    pass


def cli() -> typer.Typer:
    """A root with only the config group registered, the way ``fm.cli`` would register it."""
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    config_cmd.register(root)
    return root


class TestValid:
    def test_sample_fixture_loads_both_leagues(self) -> None:
        config = load_config(SAMPLE, environ={})
        nfl, nba = config.leagues
        assert (nfl.key, nfl.sport, nfl.game) == ("nfl", "nfl", "ffl")
        assert (nfl.espn_league_id, nfl.season, nfl.team_id) == (1234567, 2026, 4)
        assert nfl.policy == Policy(
            bench_inactive="auto",
            lineup="approve",
            add_drop="approve",
            waiver="approve",
            max_transactions_per_week=3,
            max_faab_pct_per_bid=0.35,
            untouchables=("Sample Player", 4242424),
        )
        assert (nba.key, nba.sport, nba.game) == ("nba", "nba", "fba")
        assert (nba.espn_league_id, nba.season, nba.team_id) == (7654321, 2027, 9)
        assert nba.policy.max_transactions_per_week == 7
        assert nba.policy.max_faab_pct_per_bid == 0.35  # omitted keys take the default
        assert nba.policy.untouchables == ()
        assert config.llm == Llm(model="claude-opus-5-5", daily_budget_usd=2.0)
        assert config.notify == Notify(channel="telegram")
        assert config.secrets == Secrets()  # no .env next to the fixture

    def test_omitted_sections_take_defaults(self, tmp_path: Path) -> None:
        config = load_config(write(tmp_path, MINIMAL), environ={})
        (league,) = config.leagues
        assert league.policy == Policy()
        assert Policy().bench_inactive == "auto"  # the one auto default (DESIGN section 18)
        assert Policy().lineup == "auto"  # lineup optimizations fire at T-15 when unanswered
        assert (Policy().add, Policy().add_drop, Policy().waiver) == ("approve", "approve", "approve")
        assert Policy().auto_add_min_gain is None
        assert (Policy().max_transactions_per_week, Policy().max_faab_pct_per_bid) == (3, 0.35)
        assert Policy().untouchables == ()
        assert config.llm == Llm() and config.llm.model == "claude-opus-5-5"
        assert config.notify == Notify() and config.notify.channel == "telegram"

    def test_league_lookup_by_key(self) -> None:
        config = load_config(SAMPLE, environ={})
        assert config.league("nba").espn_league_id == 7654321
        with pytest.raises(KeyError, match="no league 'mlb'.*known: nfl, nba"):
            config.league("mlb")

    def test_models_are_frozen(self) -> None:
        config = load_config(SAMPLE, environ={})
        with pytest.raises(ValidationError, match="frozen"):
            config.llm.model = "other"
        with pytest.raises(ValidationError, match="frozen"):
            config.leagues[0].policy.lineup = "auto"

    def test_python_construction_enforces_the_same_rules(self) -> None:
        with pytest.raises(ValidationError, match="approval-only"):
            Policy.model_validate({"trade": "auto"})
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            League(key="nfl", sport="nfl", espn_league_id=0, season=2026, team_id=1)
        league = League(key="nba", sport="nba", espn_league_id=1, season=2027, team_id=1)
        assert league.game == "fba" and league.policy == Policy()
        with pytest.raises(ValidationError, match=r"at least one \[\[league\]\]"):
            Config()
        with pytest.raises(ValidationError, match="duplicate league key 'nba'"):
            Config(leagues=(league, league))
        assert Config(leagues=(league,)).league("nba") is league


HANDS_OFF_NFL = Policy(
    bench_inactive="auto",
    lineup="auto",
    add="auto",
    auto_add_min_gain=15,
    add_drop="approve",
    waiver="approve",
    max_transactions_per_week=3,
)
FINDER_ON = TradeFinder(enabled=True, min_accept=0.5, min_title_gain=0.015, min_ros_gain=0.3, max_offers=1)


class TestPresets:
    """``preset`` fills the unset policy and trade_finder keys; without one nothing changes (PLUG_AND_PLAY 1)."""

    def test_fixture_home_loads_to_the_same_config_as_before(self) -> None:
        config = load_config(HOME, environ={})
        assert config == Config(
            leagues=(
                League(
                    key="nfl",
                    sport="nfl",
                    espn_league_id=1234567,
                    season=2026,
                    team_id=1,
                    policy=Policy(
                        bench_inactive="auto",
                        lineup="approve",
                        add_drop="approve",
                        waiver="approve",
                        max_transactions_per_week=3,
                        max_faab_pct_per_bid=0.35,
                        untouchables=("Josh Allen",),
                    ),
                ),
            ),
            llm=Llm(model="claude-opus-5-5", daily_budget_usd=0.0),
            notify=Notify(channel="telegram"),
        )
        (league,) = config.leagues
        assert league.preset is None and league.trade_finder == TradeFinder()
        assert load_config(SAMPLE, environ={}).leagues[0].preset is None

    def test_preset_fills_every_unset_key(self, tmp_path: Path) -> None:
        (league,) = load_config(write(tmp_path, MINIMAL + 'preset = "hands_off"\n'), environ={}).leagues
        assert league.preset == "hands_off"
        assert league.policy == HANDS_OFF_NFL  # max_faab_pct_per_bid and untouchables keep the model defaults
        assert league.trade_finder == FINDER_ON

    def test_explicit_keys_win(self, tmp_path: Path) -> None:
        text = (
            MINIMAL + 'preset = "hands_off"\n'
            '[league.policy]\nlineup = "approve"\nuntouchables = ["Sample Player"]\n'
            "[league.trade_finder]\nenabled = false\n"
        )
        (league,) = load_config(write(tmp_path, text), environ={}).leagues
        assert league.policy == HANDS_OFF_NFL.model_copy(
            update={"lineup": "approve", "untouchables": ("Sample Player",)}
        )
        assert league.trade_finder == FINDER_ON.model_copy(update={"enabled": False})  # the thresholds still apply

    def test_nba_and_ask_first(self, tmp_path: Path) -> None:
        nba = MINIMAL.replace('key = "nfl"', 'key = "nba"').replace('sport = "nfl"', 'sport = "nba"')
        (hands_off,) = load_config(write(tmp_path, nba + 'preset = "hands_off"\n'), environ={}).leagues
        assert (hands_off.policy.add, hands_off.policy.auto_add_min_gain) == ("approve", None)  # NBA add never auto
        assert hands_off.policy.max_transactions_per_week == 7 and hands_off.policy.lineup == "auto"
        (ask_first,) = load_config(write(tmp_path, nba + 'preset = "ask_first"\n'), environ={}).leagues
        assert ask_first.policy == Policy(bench_inactive="approve", lineup="approve", max_transactions_per_week=7)
        assert ask_first.trade_finder == TradeFinder(enabled=False)

    def test_python_construction_applies_the_preset(self) -> None:
        league = League(key="nfl", sport="nfl", espn_league_id=1, season=2026, team_id=1, preset="balanced")
        assert league.policy.lineup == "auto" and league.policy.add == "approve" and league.trade_finder == FINDER_ON
        explicit = League(
            key="nfl",
            sport="nfl",
            espn_league_id=1,
            season=2026,
            team_id=1,
            preset="hands_off",
            policy=Policy(lineup="approve"),  # only the field given counts as set
        )
        assert explicit.policy == HANDS_OFF_NFL.model_copy(update={"lineup": "approve"})
        with pytest.raises(ValidationError, match="Input should be 'hands_off', 'balanced' or 'ask_first'"):
            League(key="nfl", sport="nfl", espn_league_id=1, season=2026, team_id=1, preset="yolo")  # type: ignore[arg-type]

    def test_input_is_not_mutated(self) -> None:
        data = {"key": "nfl", "sport": "nfl", "espn_league_id": 1, "season": 2026, "team_id": 1, "preset": "balanced"}
        policy = {"lineup": "approve"}
        League.model_validate({**data, "policy": policy})
        assert policy == {"lineup": "approve"} and "policy" not in data

    def test_preset_with_a_bad_sport_reports_the_sport(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL.replace('sport = "nfl"', 'sport = "mlb"') + 'preset = "balanced"\n')
        with pytest.raises(ConfigError) as info:
            load_config(path, environ={})
        assert "league[0].sport: Input should be 'nfl' or 'nba' (got 'mlb')" in str(info.value)
        assert str(info.value).startswith("invalid configuration (1 problem)")


BAD_CASES: dict[str, tuple[str, list[str]]] = {
    "no league table": ("[llm]\nmodel = 'x'\n", ["at least one [[league]] table is required"]),
    "empty league array": ("league = []\n", ["at least one [[league]] table is required"]),
    "unknown league key": (
        MINIMAL + 'nickname = "x"\n',
        ["league[0].nickname: Extra inputs are not permitted (got 'x')"],
    ),
    "unknown top-level table": (MINIMAL + "[advisor]\nx = 1\n", ["advisor: Extra inputs are not permitted"]),
    "lineup outside the literal": (
        MINIMAL + '[league.policy]\nlineup = "always"\n',
        ["league[0].policy.lineup: Input should be 'off', 'approve' or 'auto' (got 'always')"],
    ),
    "add_drop cannot be auto": (
        MINIMAL + '[league.policy]\nadd_drop = "auto"\n',
        ["league[0].policy.add_drop: Input should be 'off' or 'approve' (got 'auto')"],
    ),
    "an auto add needs a threshold": (
        MINIMAL + '[league.policy]\nadd = "auto"\n',
        ['add = "auto" needs auto_add_min_gain'],
    ),
    "the auto add threshold is positive": (
        MINIMAL + '[league.policy]\nadd = "auto"\nauto_add_min_gain = 0\n',
        ["league[0].policy.auto_add_min_gain: Input should be greater than 0"],
    ),
    "waiver cannot be auto": (
        MINIMAL + '[league.policy]\nwaiver = "auto"\n',
        ["league[0].policy.waiver: Input should be 'off' or 'approve' (got 'auto')"],
    ),
    "trade policy is not configurable": (
        MINIMAL + '[league.policy]\ntrade = "auto"\ntrade_accept = "approve"\n',
        ["league[0].policy: trade, trade_accept: trades are approval-only and not configurable"],
    ),
    "faab cap above 1": (
        MINIMAL + "[league.policy]\nmax_faab_pct_per_bid = 1.5\n",
        ["league[0].policy.max_faab_pct_per_bid: Input should be less than or equal to 1 (got 1.5)"],
    ),
    "negative weekly cap": (
        MINIMAL + "[league.policy]\nmax_transactions_per_week = -1\n",
        ["league[0].policy.max_transactions_per_week: Input should be greater than or equal to 0 (got -1)"],
    ),
    "placeholder league id": (
        MINIMAL.replace("1234567", "0"),
        ["league[0].espn_league_id: Input should be greater than or equal to 1 (got 0)"],
    ),
    "placeholder team id": (
        MINIMAL.replace("team_id = 4", "team_id = 0"),
        ["league[0].team_id: Input should be greater than or equal to 1 (got 0)"],
    ),
    "unknown sport": (
        MINIMAL.replace('sport = "nfl"', 'sport = "mlb"'),
        ["league[0].sport: Input should be 'nfl' or 'nba' (got 'mlb')"],
    ),
    "key with spaces": (
        MINIMAL.replace('key = "nfl"', 'key = "NFL Main"'),
        ["league[0].key: String should match pattern", "(got 'NFL Main')"],
    ),
    "two-digit season": (
        MINIMAL.replace("season = 2026", "season = 26"),
        ["league[0].season: Input should be greater than or equal to 2000 (got 26)"],
    ),
    "duplicate key": (MINIMAL + MINIMAL.replace("1234567", "7654321"), ["duplicate league key 'nfl'"]),
    "same espn league twice": (
        MINIMAL + MINIMAL.replace('key = "nfl"', 'key = "nfl2"'),
        ["leagues 'nfl' and 'nfl2' point at the same ESPN nfl league 1234567"],
    ),
    "negative budget": (
        MINIMAL + "[llm]\ndaily_budget_usd = -1\n",
        ["llm.daily_budget_usd: Input should be greater than or equal to 0 (got -1)"],
    ),
    "unknown channel": (
        MINIMAL + '[notify]\nchannel = "sms"\n',
        ["notify.channel: Input should be 'telegram', 'ntfy' or 'none' (got 'sms')"],
    ),
    "unknown preset": (
        MINIMAL + 'preset = "yolo"\n',
        ["league[0].preset: Input should be 'hands_off', 'balanced' or 'ask_first' (got 'yolo')"],
    ),
    "an explicit auto add under a preset still needs a threshold": (
        MINIMAL + 'preset = "balanced"\n[league.policy]\nadd = "auto"\n',
        ['league[0].policy: add = "auto" needs auto_add_min_gain'],
    ),
    "a preset cannot smuggle a trade key past the policy": (
        MINIMAL + 'preset = "hands_off"\n[league.policy]\ntrade = "auto"\n',
        ["league[0].policy: trade: trades are approval-only and not configurable"],
    ),
    "toml syntax": ("[[league]\n", ["invalid TOML"]),
}


def test_an_auto_add_with_a_threshold_is_valid(tmp_path: Path) -> None:
    text = MINIMAL + '[league.policy]\nadd = "auto"\nauto_add_min_gain = 20\nadd_drop = "approve"\n'
    (league,) = load_config(write(tmp_path, text), environ={}).leagues
    assert (league.policy.add, league.policy.auto_add_min_gain, league.policy.add_drop) == ("auto", 20.0, "approve")


class TestInvalid:
    @pytest.mark.parametrize(("text", "expected"), list(BAD_CASES.values()), ids=list(BAD_CASES))
    def test_rejected_with_a_located_message(self, tmp_path: Path, text: str, expected: list[str]) -> None:
        path = write(tmp_path, text)
        with pytest.raises(ConfigError) as info:
            load_config(path, environ={})
        message = str(info.value)
        assert str(path) in message
        for fragment in expected:
            assert fragment in message, message

    def test_each_problem_is_reported_once(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL + '[league.policy]\nlineup = "always"\nmax_faab_pct_per_bid = 2\n')
        with pytest.raises(ConfigError) as info:
            load_config(path, environ={})
        lines = str(info.value).splitlines()
        assert lines[0] == "invalid configuration (2 problems):"
        assert len(lines) == 3
        assert "at least" not in str(info.value)  # no "too few leagues" cascade from the failed league

    def test_secrets_table_in_the_toml_is_refused_without_echoing_it(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL + '[secrets]\nanthropic_api_key = "sk-should-not-leak"\n')
        with pytest.raises(ConfigError) as info:
            load_config(path, environ={})
        message = str(info.value)
        assert message == f"{path}: [secrets] belongs in {tmp_path / '.env'}, not in config.toml"
        assert "sk-should-not-leak" not in message

    def test_missing_file_names_the_path(self, tmp_path: Path) -> None:
        path = tmp_path / "nope.toml"
        with pytest.raises(ConfigError, match="not found") as info:
            load_config(path, environ={})
        assert str(path) in str(info.value)


class TestSecrets:
    def test_env_file_values_become_secrets(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL)
        write(
            tmp_path,
            'ANTHROPIC_API_KEY="sk-from-file"\nTELEGRAM_BOT_TOKEN=123:abc\nTELEGRAM_CHAT_ID=42\nODDS_API_KEY=\nUNRELATED=1\n',
            ".env",
        )
        config = load_config(path, environ={})
        assert secret(config.secrets.anthropic_api_key) == "sk-from-file"
        assert secret(config.secrets.telegram_bot_token) == "123:abc"
        assert config.secrets.telegram_chat_id == 42
        assert config.secrets.odds_api_key is None  # blank means unset
        assert config.secrets.ntfy_topic is None and config.secrets.ntfy_reply_topic is None
        assert config.required_secrets() == ("ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
        assert config.missing_secrets() == ()

    def test_secret_values_never_appear_in_repr_or_dumps(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL)
        write(tmp_path, "ANTHROPIC_API_KEY=sk-from-file\nNTFY_TOPIC=topic-from-file\n", ".env")
        config = load_config(path, environ={})
        for rendered in (repr(config), str(config), config.model_dump_json(), repr(config.secrets)):
            assert "sk-from-file" not in rendered and "topic-from-file" not in rendered

    def test_environment_overrides_the_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = write(tmp_path, MINIMAL)
        write(tmp_path, "TELEGRAM_CHAT_ID=42\nANTHROPIC_API_KEY=sk-from-file\n", ".env")
        config = load_config(path, environ={"TELEGRAM_CHAT_ID": "7"})
        assert config.secrets.telegram_chat_id == 7
        assert secret(config.secrets.anthropic_api_key) == "sk-from-file"
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "9")
        assert load_config(path).secrets.telegram_chat_id == 9  # the default mapping is os.environ

    def test_blank_values_count_as_unset(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL)
        config = load_config(path, environ={"ANTHROPIC_API_KEY": "   ", "TELEGRAM_CHAT_ID": ""})
        assert config.secrets.anthropic_api_key is None and config.secrets.telegram_chat_id is None
        assert config.missing_secrets() == ("ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")

    def test_bad_chat_id_is_located_in_env_without_echoing_it(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL)
        env_path = write(tmp_path, "TELEGRAM_CHAT_ID=not-a-number-SECRET\n", ".env")
        with pytest.raises(ConfigError) as info:
            load_config(path, environ={})
        message = str(info.value)
        assert f"{env_path}: TELEGRAM_CHAT_ID: Input should be a valid integer" in message
        assert "SECRET" not in message

    def test_required_secrets_follow_the_channel(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL + '[notify]\nchannel = "ntfy"\n')
        config = load_config(path, environ={})
        assert config.required_secrets() == ("ANTHROPIC_API_KEY", "NTFY_TOPIC", "NTFY_REPLY_TOPIC")
        assert config.missing_secrets() == config.required_secrets()
        complete = load_config(path, environ={"ANTHROPIC_API_KEY": "k", "NTFY_TOPIC": "t", "NTFY_REPLY_TOPIC": "r"})
        assert complete.missing_secrets() == ()

    def test_secrets_model_knows_its_variables(self) -> None:
        assert SECRET_ENV_VARS == (
            "ANTHROPIC_API_KEY",
            "ODDS_API_KEY",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
            "NTFY_TOPIC",
            "NTFY_REPLY_TOPIC",
        )
        assert set(OPTIONAL_SECRET_ENV_VARS) <= set(SECRET_ENV_VARS)
        secrets = Secrets.model_validate({"ANTHROPIC_API_KEY": "k", "PATH": "ignored"})
        assert secrets.is_set("ANTHROPIC_API_KEY") and not secrets.is_set("ODDS_API_KEY")
        with pytest.raises(KeyError, match="NOT_A_SECRET is not a known secret"):
            secrets.is_set("NOT_A_SECRET")


class TestDirOverride:
    def test_default_paths_follow_fm_config_dir(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"  # where conftest points FM_CONFIG_DIR
        assert config_paths() == (config_dir / "config.toml", config_dir / ".env")
        assert config_paths() == (paths.config_file(), paths.env_file())

    def test_loads_from_fm_config_dir(self) -> None:
        config_dir = paths.config_dir()
        write(config_dir, MINIMAL)
        write(config_dir, "ANTHROPIC_API_KEY=sk-dir\n", ".env")
        config = load_config(environ={})
        assert config.leagues[0].key == "nfl"
        assert secret(config.secrets.anthropic_api_key) == "sk-dir"

    def test_switching_the_dir_switches_the_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        other = tmp_path / "other"
        write(other, MINIMAL.replace('key = "nfl"', 'key = "work"'))
        monkeypatch.setenv("FM_CONFIG_DIR", str(other))
        assert config_paths()[0] == other / "config.toml"
        assert load_config(environ={}).leagues[0].key == "work"

    def test_missing_default_file_is_reported_with_its_path(self) -> None:
        with pytest.raises(ConfigError, match="not found") as info:
            load_config(environ={})
        assert str(paths.config_file()) in str(info.value)

    def test_explicit_path_reads_the_env_beside_it(self, tmp_path: Path) -> None:
        write(paths.config_dir(), "ANTHROPIC_API_KEY=sk-config-dir\n", ".env")
        elsewhere = tmp_path / "elsewhere"
        path = write(elsewhere, MINIMAL)
        write(elsewhere, "ANTHROPIC_API_KEY=sk-sibling\n", ".env")
        assert config_paths(path) == (path, elsewhere / ".env")
        assert secret(load_config(path, environ={}).secrets.anthropic_api_key) == "sk-sibling"
        # An explicit env_path wins over the sibling rule.
        chosen = load_config(path, env_path=paths.env_file(), environ={})
        assert secret(chosen.secrets.anthropic_api_key) == "sk-config-dir"


class TestCheckCommand:
    def test_sample_fixture_passes(self) -> None:
        result = runner.invoke(cli(), ["config", "check", "--path", str(SAMPLE)])
        assert result.exit_code == 0, result.output
        assert f"config: {SAMPLE}" in result.output
        assert f"env:    {FIXTURES / '.env'} (not found)" in result.output
        assert "nfl: nfl (ffl) league 1234567, season 2026, team 4" in result.output
        assert "nba: nba (fba) league 7654321, season 2027, team 9" in result.output
        assert "bench_inactive=auto lineup=approve add=approve add_drop=approve waiver=approve" in result.output
        assert "max 3 transactions/week, bids <= 35% of FAAB, 2 untouchable(s)" in result.output
        assert "max 7 transactions/week, bids <= 35% of FAAB, 0 untouchable(s)" in result.output
        assert "llm:    claude-opus-5-5, $2.00/day budget" in result.output
        assert "notify: telegram" in result.output
        for name in ("ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            assert f"{name}: missing" in result.output
        assert "ODDS_API_KEY: unset (optional)" in result.output
        assert "NTFY_TOPIC" not in result.output  # the other channel's variables are not listed
        assert result.output.rstrip().endswith("ok: config.toml is valid; 3 required secret(s) missing from .env")

    def test_complete_setup_reports_set_without_printing_values(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL)
        env = "ANTHROPIC_API_KEY=sk-secret-value\nTELEGRAM_BOT_TOKEN=tok-secret\nTELEGRAM_CHAT_ID=987654321\n"
        write(tmp_path, env, ".env")
        result = runner.invoke(cli(), ["config", "check", "-p", str(path)])
        assert result.exit_code == 0, result.output
        assert f"env:    {tmp_path / '.env'}\n" in result.output
        for name in ("ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            assert f"{name}: set" in result.output
        assert "ok: config.toml is valid; all required secrets set" in result.output
        for value in ("sk-secret-value", "tok-secret", "987654321"):
            assert value not in result.output

    def test_invalid_config_exits_one_and_lists_the_problems(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL + '[league.policy]\nlineup = "always"\n[notify]\nchannel = "sms"\n')
        result = runner.invoke(cli(), ["config", "check", "--path", str(path)])
        assert result.exit_code == 1, result.output
        assert result.stdout == ""  # problems go to stderr
        assert "error: invalid configuration (2 problems):" in result.stderr
        assert "league[0].policy.lineup: Input should be 'off', 'approve' or 'auto' (got 'always')" in result.stderr
        assert "notify.channel: Input should be 'telegram', 'ntfy' or 'none' (got 'sms')" in result.stderr

    def test_default_path_is_the_config_dir_file(self) -> None:
        result = runner.invoke(cli(), ["config", "check"])
        assert result.exit_code == 1, result.output
        assert f"error: {paths.config_file()} not found" in result.stderr
        write(paths.config_dir(), MINIMAL)
        result = runner.invoke(cli(), ["config", "check"])
        assert result.exit_code == 0, result.output
        assert f"config: {paths.config_file()}" in result.output

    def test_preset_is_shown_with_the_policy_it_gives(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL + 'preset = "hands_off"\n[league.policy]\nlineup = "approve"\n')
        result = runner.invoke(cli(), ["config", "check", "--path", str(path)])
        assert result.exit_code == 0, result.output
        assert "nfl: nfl (ffl) league 1234567, season 2026, team 4, preset hands_off" in result.output
        assert (
            "bench_inactive=auto lineup=approve add=auto (min gain 15) add_drop=approve waiver=approve" in result.output
        )
        assert "trade finder: drafts up to 1 offer(s) a period for approval when P(accept) >= 50%" in result.output
        assert ", preset" not in runner.invoke(cli(), ["config", "check", "--path", str(SAMPLE)]).output

    def test_group_help(self) -> None:
        result = runner.invoke(cli(), ["config", "--help"])
        assert result.exit_code == 0, result.output
        assert "check" in result.output and "set-preset" in result.output
        result = runner.invoke(cli(), ["config", "check", "--help"])
        assert result.exit_code == 0, result.output
        assert "--path" in result.output and "-p" in result.output


class TestSetPresetCommand:
    def test_writes_the_preset_and_shows_the_resulting_policy(self, tmp_path: Path) -> None:
        path = write(tmp_path, SAMPLE.read_text(encoding="utf-8"))
        before = path.read_bytes()
        result = runner.invoke(cli(), ["config", "set-preset", "nba", "hands_off", "--path", str(path)])
        assert result.exit_code == 0, result.output
        # The sample writes lineup = "approve" for nba, so the preset's "lineups are set on their own" is not true.
        assert f"{path}: nba uses the hands_off preset, with keys written in config.toml overriding it" in result.output
        assert "lineups are set on their own" not in result.output
        assert (
            "note: these keys written in config.toml override the preset: policy.lineup; remove them or run with "
            "--reset-overrides" in result.output
        )
        assert "nba: nba (fba) league 7654321, season 2027, team 9, preset hands_off" in result.output
        assert "bench_inactive=auto lineup=approve add=approve add_drop=approve waiver=approve" in result.output
        assert "max 7 transactions/week" in result.output  # the file's explicit keys still win
        assert "trade finder: drafts up to 1 offer(s)" in result.output
        assert backup_path(path).read_bytes() == before
        config = load_config(path, environ={})
        assert config.league("nba").preset == "hands_off" and config.league("nfl").preset is None
        newline = b"\r\n" if b"\r\n" in before else b"\n"  # write_text uses the platform's; the edit keeps it
        assert path.read_bytes().replace(b'preset = "hands_off"' + newline, b"") == before  # one line added, no more

    def test_reset_overrides_removes_the_written_keys_so_the_preset_decides(self, tmp_path: Path) -> None:
        path = write(tmp_path, SAMPLE.read_text(encoding="utf-8"))
        before = path.read_bytes()
        result = runner.invoke(
            cli(), ["config", "set-preset", "nba", "hands_off", "--reset-overrides", "--path", str(path)]
        )
        assert result.exit_code == 0, result.output
        assert f"{path}: nba uses the hands_off preset: lineups are set on their own" in result.output
        assert "removed from config.toml so the preset decides them: policy.lineup" in result.output
        assert "note:" not in result.output
        assert "bench_inactive=auto lineup=auto add=approve" in result.output
        assert backup_path(path).read_bytes() == before  # one edit: the preset and the removal together
        nba = load_config(path, environ={}).league("nba")
        assert nba.preset == "hands_off" and nba.policy.lineup == "auto"
        assert nba.policy.max_transactions_per_week == 7  # a written key equal to the preset's stays
        nfl_part = path.read_text(encoding="utf-8").split('key = "nba"')[0]
        assert 'lineup = "approve"' in nfl_part  # the other league is untouched

    def test_a_written_key_equal_to_the_preset_is_no_override(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL + '[league.policy]\nlineup = "approve"\nuntouchables = ["Sample Player"]\n')
        result = runner.invoke(cli(), ["config", "set-preset", "nfl", "ask_first", "--path", str(path)])
        assert result.exit_code == 0, result.output
        assert "note:" not in result.output and "uses the ask_first preset: every move waits" in result.output

    def test_default_path_is_the_config_dir_file(self) -> None:
        write(paths.config_dir(), MINIMAL)
        result = runner.invoke(cli(), ["config", "set-preset", "nfl", "ask_first"])
        assert result.exit_code == 0, result.output
        assert f"{paths.config_file()}: nfl uses the ask_first preset" in result.output
        assert load_config(environ={}).league("nfl").preset == "ask_first"

    def test_unknown_preset_is_a_usage_error_and_touches_nothing(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL)
        before = path.read_bytes()
        result = runner.invoke(cli(), ["config", "set-preset", "nfl", "yolo", "--path", str(path)])
        assert result.exit_code == 2, result.output
        plain = " ".join(re.sub(r"[\u2500-\u257f]", " ", result.output).split())  # typer may draw a wrapped box
        assert "unknown preset 'yolo'; choose one of hands_off, balanced, ask_first" in plain
        assert path.read_bytes() == before and not backup_path(path).exists()

    def test_unknown_league_exits_one(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL)
        result = runner.invoke(cli(), ["config", "set-preset", "mlb", "balanced", "--path", str(path)])
        assert result.exit_code == 1, result.output
        assert f"error: {path}: no league 'mlb'; known: nfl" in result.stderr
        assert not backup_path(path).exists()

    def test_invalid_file_exits_one_and_stays_byte_identical(self, tmp_path: Path) -> None:
        path = write(tmp_path, MINIMAL + '[league.policy]\nlineup = "always"\n')
        before = path.read_bytes()
        result = runner.invoke(cli(), ["config", "set-preset", "nfl", "balanced", "--path", str(path)])
        assert result.exit_code == 1, result.output
        assert "league[0].policy.lineup: Input should be 'off', 'approve' or 'auto' (got 'always')" in result.stderr
        assert path.read_bytes() == before and not backup_path(path).exists()

    def test_help(self) -> None:
        result = runner.invoke(cli(), ["config", "set-preset", "--help"])
        assert result.exit_code == 0, result.output
        assert "hands_off, balanced, ask_first" in result.output
        assert "config.toml.bak" in result.output
