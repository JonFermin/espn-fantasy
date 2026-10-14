"""``fm setup`` (ROADMAP #53, docs/DESIGN.md section 12.2): the wizard over fake collaborators, offline.

Every external effect is a fake from :class:`Fakes`: the session loader and login, discovery and league
confirmation, the phone setup (which writes made-up ntfy topics or Telegram values to ``.env`` the way the real one
does), the sync, the two scheduler jobs, the ``fm bot`` probe and the doctor. The config dir is the per-test
``FM_CONFIG_DIR`` from ``conftest``. League and team names are synthetic.
"""

from __future__ import annotations

import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pytest
import typer
from dotenv import dotenv_values, set_key
from typer.testing import CliRunner

from fm import paths
from fm.commands import config_cmd
from fm.commands import setup as setup_cmd
from fm.config import Config, Secrets, load_config
from fm.espn.auth import EspnSession, NotLoggedInError
from fm.espn.discover import DiscoveredLeague, DiscoveryError, LeagueRef
from fm.notify import NotifyError
from fm.notify.ntfy import REPLY_TOPIC_ENV, TOPIC_ENV
from fm.setup import (
    SUMMARY_HINT,
    AcceptDefaults,
    Job,
    Prompts,
    SetupDeps,
    SetupError,
    SetupOptions,
    SetupResult,
    StepError,
    describe_league,
    league_key,
    parse_bot_pids,
    read_secrets,
    run_setup,
)

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
SESSION = EspnSession(espn_s2="s2-test", swid="{0000-TEST-SWID}", expires_at=datetime.now(UTC) + timedelta(days=60))
NFL = DiscoveredLeague(
    sport="nfl",
    espn_league_id=1234567,
    season=2026,
    team_id=3,
    league_name="Sample Football League",
    team_name="Sample Team",
)
NBA = DiscoveredLeague(
    sport="nba",
    espn_league_id=7654321,
    season=2027,
    team_id=5,
    league_name="Sample Hoops League",
    team_name="Sample Five",
)
NFL_URL = "https://fantasy.espn.com/football/team?leagueId=1234567&teamId=3&seasonId=2026"
TOPIC = "espn-fantasy-PhoneTopicForSetupTests_01"
REPLY = "espn-fantasy-ReplyTopicForSetupTests_02"
ANTHROPIC_KEY = "sk-ant-test-setup-secret"
ODDS_KEY = "odds-test-setup-secret"

runner = CliRunner()


@dataclass
class Fakes:
    """Every collaborator of the wizard, recording what it was asked to do in ``calls``."""

    session: EspnSession | Exception = SESSION
    login_session: EspnSession = SESSION
    discovered: Sequence[DiscoveredLeague] | DiscoveryError = (NFL, NBA)
    confirmed: dict[tuple[str, int], DiscoveredLeague | DiscoveryError] = field(default_factory=dict)
    autostart_installed: bool | None = False
    wake_installed: bool | None = False
    job_error: str | None = None
    bots: tuple[int, ...] = ()
    no_start: bool = True
    """Whether the autostart job can be registered without starting fm run (Task Scheduler can, launchd cannot)."""
    sync_errors: dict[str, str] = field(default_factory=dict)
    channel_error: str | None = None
    calls: list[tuple[Any, ...]] = field(default_factory=list)

    def deps(self) -> SetupDeps:
        return SetupDeps(
            load_session=self.load_session,
            login=self.login,
            discover=self.discover,
            confirm_league=self.confirm_league,
            setup_channel=self.setup_channel,
            sync=self.sync,
            autostart=Job(
                "fm run at logon",
                "fm autostart install",
                self._installed("autostart"),
                self._install("autostart"),
                self._install("autostart", start=False) if self.no_start else None,
            ),
            wake_timer=Job("the wake timer", "fm schedule install", self._installed("wake"), self._install("wake")),
            bot_pids=lambda: self.bots,
            doctor=lambda: ["OK    config  config.toml valid", "WARN  tick    no tick has run yet  → fix: fm run"],
        )

    def load_session(self) -> EspnSession:
        self.calls.append(("load_session",))
        if isinstance(self.session, Exception):
            raise self.session
        return self.session

    def login(self) -> EspnSession:
        self.calls.append(("login",))
        return self.login_session

    def discover(self, session: EspnSession) -> Sequence[DiscoveredLeague]:
        self.calls.append(("discover",))
        if isinstance(self.discovered, DiscoveryError):
            raise self.discovered
        return self.discovered

    def confirm_league(self, ref: LeagueRef, session: EspnSession) -> DiscoveredLeague:
        self.calls.append(("confirm", ref.sport, ref.espn_league_id, ref.team_id))
        answer = self.confirmed.get((ref.sport, ref.espn_league_id))
        if answer is None:
            raise DiscoveryError(f"league {ref.espn_league_id} not found")
        if isinstance(answer, DiscoveryError):
            raise answer
        return answer

    def setup_channel(
        self,
        channel: Literal["telegram", "ntfy"],
        env_path: Path,
        found: Secrets,
        *,
        prompts: Prompts,
        echo: Any,
    ) -> None:
        self.calls.append(("channel", channel))
        if self.channel_error is not None:
            raise NotifyError(self.channel_error)
        if channel == "ntfy":
            set_key(env_path, TOPIC_ENV, TOPIC, quote_mode="never")
            set_key(env_path, REPLY_TOPIC_ENV, REPLY, quote_mode="never")
        else:
            set_key(env_path, "TELEGRAM_BOT_TOKEN", "123456:TEST-token", quote_mode="never")
            set_key(env_path, "TELEGRAM_CHAT_ID", "4242", quote_mode="never")
        echo(f"{channel} set up")

    def sync(self, config: Config, keys: Sequence[str], session: EspnSession) -> list[str]:
        self.calls.append(("sync", *keys))
        for key in keys:
            if key in self.sync_errors:
                raise StepError(self.sync_errors[key])
        return [f"{key}: synced" for key in keys]

    def _installed(self, name: str) -> Any:
        def installed() -> bool | None:
            return self.autostart_installed if name == "autostart" else self.wake_installed

        return installed

    def _install(self, name: str, *, start: bool = True) -> Any:
        def install() -> list[str]:
            self.calls.append(("install", name) if start else ("install", name, "no-start"))
            if self.job_error is not None:
                raise StepError(self.job_error)
            if name == "autostart":
                self.autostart_installed = True
            else:
                self.wake_installed = True
            return [f"installed {name}"]

        return install

    def kinds(self) -> list[str]:
        return [str(call[0]) for call in self.calls]


class Scripted:
    """Answers the wizard's questions in order, recording each; any question beyond the script fails the test."""

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    def _next(self, question: str) -> Any:
        self.asked.append(question)
        assert self.answers, f"unexpected question: {question}"
        return self.answers.pop(0)

    def confirm(self, question: str, *, default: bool) -> bool:
        return bool(self._next(question))

    def choose(self, question: str, choices: Sequence[str], *, default: str) -> str:
        answer = self._next(question)
        assert answer in choices, (answer, choices)
        return str(answer)

    def text(self, question: str, *, default: str = "") -> str:
        return str(self._next(question))

    def secret(self, question: str) -> str:
        return str(self._next(question))


class Recording(AcceptDefaults):
    """:class:`AcceptDefaults` that records each yes/no question's default and answers the ones in ``answers``
    itself, so a test can check what the terminal would have offered."""

    def __init__(self, answers: dict[str, bool] | None = None) -> None:
        self.answers = dict(answers or {})
        self.defaults: dict[str, bool] = {}

    def confirm(self, question: str, *, default: bool) -> bool:
        self.defaults[question] = default
        return self.answers.get(question, default)


def config_path() -> Path:
    return paths.config_dir() / "config.toml"


def env_path() -> Path:
    return paths.config_dir() / ".env"


def wizard(
    fakes: Fakes, options: SetupOptions | None = None, prompts: Prompts | None = None
) -> tuple[SetupResult, str]:
    lines: list[str] = []
    result = run_setup(
        options if options is not None else SetupOptions(),
        fakes.deps(),
        prompts=prompts if prompts is not None else AcceptDefaults(),
        echo=lines.append,
        now=NOW,
    )
    return result, "\n".join(lines)


def raw_leagues() -> dict[str, dict[str, Any]]:
    data = tomllib.loads(config_path().read_text(encoding="utf-8"))
    return {table["key"]: table for table in data["league"]}


# --- the CLI: fm setup --yes, then fm config check, then a re-run ----------------------------------------------------


def _root() -> None:
    pass


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    setup_cmd.register(root)
    config_cmd.register(root)
    return root


def run(*args: str, expect: int = 0) -> str:
    result = runner.invoke(cli(), list(args), catch_exceptions=False)
    assert result.exit_code == expect, result.output
    return result.output


def test_setup_yes_writes_a_config_check_accepts_and_a_rerun_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = Fakes()
    monkeypatch.setattr(setup_cmd, "live_deps", fakes.deps)
    output = run("setup", "--yes")
    assert 'NFL  Sample Football League  -  team 3 "Sample Team"' in output
    assert "nfl: NFL  Sample Football League" in output and "preset balanced" in output
    assert output.rstrip().endswith(SUMMARY_HINT)
    assert "WARN  tick    no tick has run yet" in output  # the doctor's lines close the run

    checked = run("config", "check")
    assert "nfl: nfl (ffl) league 1234567, season 2026, team 3, preset balanced" in checked
    assert "nba: nba (fba) league 7654321, season 2027, team 5, preset balanced" in checked
    assert "notify: ntfy" in checked and "ok: config.toml is valid" in checked
    assert dotenv_values(env_path()) == {TOPIC_ENV: TOPIC, REPLY_TOPIC_ENV: REPLY}  # no API key: --yes skips secrets
    assert fakes.calls.count(("channel", "ntfy")) == 1
    assert [call for call in fakes.calls if call[0] == "sync"] == [("sync", "nfl"), ("sync", "nba")]
    assert [call for call in fakes.calls if call[0] == "install"] == [("install", "autostart"), ("install", "wake")]

    before = config_path().read_bytes()
    env_before = env_path().read_bytes()
    fakes.calls.clear()
    output = run("setup", "--yes")
    assert config_path().read_bytes() == before  # byte-identical
    assert env_path().read_bytes() == env_before
    assert not (paths.config_dir() / "config.toml.bak").exists()  # never rewritten, not even identically
    assert "already has everything; left unchanged" in output and "ntfy is already set up" in output
    assert "Nothing new to sync." in output
    assert [kind for kind in fakes.kinds() if kind in ("channel", "sync", "install", "login")] == []


def test_fm_setup_reports_a_setup_error_and_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = Fakes(discovered=DiscoveryError("ESPN rejected the session (HTTP 401); run `fm login`"))
    monkeypatch.setattr(setup_cmd, "live_deps", fakes.deps)
    result = runner.invoke(cli(), ["setup", "--yes"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "Could not discover your leagues: ESPN rejected the session" in result.output
    assert "error: no league to set up: pass the URL of your team page with --league-url" in result.stderr
    assert not config_path().exists() and not env_path().exists()  # nothing written before the leagues settle


def test_the_cli_flags_reach_the_wizard(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = Fakes(discovered=DiscoveryError("fan API down"), confirmed={("nfl", 1234567): NFL})
    monkeypatch.setattr(setup_cmd, "live_deps", fakes.deps)
    output = run(
        "setup",
        "--yes",
        "--league-url",
        NFL_URL,
        "--preset",
        "ask_first",
        "--channel",
        "none",
        "--no-sync",
        "--no-autostart",
    )
    assert raw_leagues()["nfl"]["preset"] == "ask_first"
    assert load_config().notify.channel == "none"
    assert "Skipped (--no-sync)" in output and "Skipped (--no-autostart)" in output
    assert [kind for kind in fakes.kinds() if kind in ("channel", "sync", "install")] == []


def test_without_yes_the_terminal_asks_and_hides_the_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = Fakes()
    monkeypatch.setattr(setup_cmd, "live_deps", fakes.deps)
    answers = ["y", "hands_off", "n", ANTHROPIC_KEY, "", "none"]  # NFL yes, its preset, NBA no, keys, no phone
    result = runner.invoke(cli(), ["setup"], input="\n".join(answers) + "\n", catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Manage NFL  Sample Football League" in result.output and "Preset for nfl" in result.output
    assert ANTHROPIC_KEY not in result.output  # typed hidden, never echoed
    assert {key: table["preset"] for key, table in raw_leagues().items()} == {"nfl": "hands_off"}
    assert dotenv_values(env_path()) == {"ANTHROPIC_API_KEY": ANTHROPIC_KEY}
    assert load_config().notify.channel == "none"


# --- leagues ----------------------------------------------------------------------------------------------------------


def test_discovery_failure_falls_back_to_the_league_urls() -> None:
    fakes = Fakes(discovered=DiscoveryError("fan API answered HTTP 500"), confirmed={("nfl", 1234567): NFL})
    result, output = wizard(fakes, SetupOptions(league_urls=(NFL_URL, "https://example.com/not-a-league")))
    assert ("confirm", "nfl", 1234567, 3) in fakes.calls  # the URL's team is checked, not trusted
    assert "Could not discover your leagues: fan API answered HTTP 500" in output
    assert "Skipped https://example.com/not-a-league: not an ESPN fantasy league URL" in output
    assert result.added == ("nfl",) and list(raw_leagues()) == ["nfl"]


def test_without_urls_a_failed_discovery_asks_for_them() -> None:
    fakes = Fakes(discovered=DiscoveryError("no route"), confirmed={("nfl", 1234567): NFL, ("nba", 7654321): NBA})
    nba_url = "https://fantasy.espn.com/basketball/league?leagueId=7654321"
    prompts = Scripted(
        NFL_URL, nba_url, "",  # the URLs, then an empty line
        True, "hands_off", True, "ask_first",  # manage each league, and its preset
        "", "",  # no API keys
        "none",  # no phone
    )  # fmt: skip
    result, _ = wizard(fakes, prompts=prompts)
    assert prompts.asked[:3] == ["League URL"] * 3 and not prompts.answers
    assert result.added == ("nfl", "nba")
    assert {key: table["preset"] for key, table in raw_leagues().items()} == {"nfl": "hands_off", "nba": "ask_first"}


def test_a_discovered_league_without_a_team_is_confirmed_and_a_refused_one_skipped() -> None:
    bare = DiscoveredLeague(sport="nfl", espn_league_id=1234567, season=2026, team_hint=9)
    other = DiscoveredLeague(sport="nba", espn_league_id=7654321, season=2027)
    refused = DiscoveryError("team 9 is not this account's team (this account owns team 3)")
    fakes = Fakes(discovered=(bare, other), confirmed={("nfl", 1234567): NFL, ("nba", 7654321): refused})
    result, output = wizard(fakes)
    assert ("confirm", "nfl", 1234567, None) in fakes.calls  # the hint travels in the ref, never as the team
    assert "Skipped NBA league 7654321: team 9 is not this account's team" in output
    assert result.added == ("nfl",) and raw_leagues()["nfl"]["team_id"] == 3


def test_declining_every_new_league_on_a_fresh_install_writes_nothing() -> None:
    prompts = Scripted(False, False)
    with pytest.raises(SetupError, match="no league selected"):
        wizard(Fakes(), prompts=prompts)
    assert not config_path().exists() and not env_path().exists()


def test_keys_are_short_and_unique() -> None:
    assert league_key("nfl", []) == "nfl"
    assert league_key("nfl", ["nfl", "nba"]) == "nfl2"
    assert league_key("nba", ["nba", "nba2"]) == "nba3"
    second = DiscoveredLeague(sport="nfl", espn_league_id=2222222, season=2026, team_id=1)
    result, _ = wizard(Fakes(discovered=(NFL, second)))
    assert result.added == ("nfl", "nfl2")


def test_a_league_is_shown_by_name_and_team() -> None:
    assert describe_league(NFL) == 'NFL  Sample Football League  -  team 3 "Sample Team"'
    unnamed = DiscoveredLeague(sport="nba", espn_league_id=7654321, season=2027, team_id=5)
    assert describe_league(unnamed) == "NBA  league 7654321  -  team 5"


# --- existing config.toml ---------------------------------------------------------------------------------------------

HAND_WRITTEN = """# my own notes stay
[[league]]
key = "football"   # my own key
sport = "nfl"
espn_league_id = 1234567
season = 2026
team_id = 3

[league.policy]
lineup = "approve"   # I approve lineups myself

[notify]
channel = "ntfy"
"""


def write_existing(text: str = HAND_WRITTEN, *, ntfy: bool = True) -> None:
    config_path().write_text(text, encoding="utf-8")
    if ntfy:
        env_path().write_text(f"{TOPIC_ENV}={TOPIC}\n{REPLY_TOPIC_ENV}={REPLY}\n", encoding="utf-8")


def test_an_existing_league_is_never_rewritten_and_a_new_one_is_appended() -> None:
    write_existing()
    fakes = Fakes()
    prompts = Scripted(True, "balanced", "", "", "ntfy")  # manage the new NBA league, its preset, no keys, ntfy
    result, output = wizard(fakes, prompts=prompts)
    text = config_path().read_text(encoding="utf-8")
    original_league = HAND_WRITTEN[: HAND_WRITTEN.index("[notify]")]
    assert original_league in text  # byte for byte, comments included
    assert "  " + describe_league(NFL) + "  (in config.toml as football)" in output
    leagues = raw_leagues()
    assert leagues["football"] == {
        "key": "football",
        "sport": "nfl",
        "espn_league_id": 1234567,
        "season": 2026,
        "team_id": 3,
        "policy": {"lineup": "approve"},
    }
    assert leagues["nba"]["preset"] == "balanced"
    assert (result.added, result.present, result.updated) == (("nba",), ("football",), ())
    assert [call for call in fakes.calls if call[0] == "sync"] == [("sync", "nba")]  # only the new league syncs


def test_a_changed_season_is_reported_and_left_alone_unless_confirmed() -> None:
    write_existing(HAND_WRITTEN.replace("season = 2026", "season = 2025"))
    before = config_path().read_bytes()
    prompts = Scripted(False, "", "", "ntfy", False)  # skip the NBA league, no keys, ntfy, do not update
    result, output = wizard(Fakes(), prompts=prompts)
    assert "football in config.toml differs from ESPN: season 2025 -> 2026." in output
    assert prompts.asked[-1] == "Update football (season 2025 -> 2026)?"
    assert config_path().read_bytes() == before and result.declined == ("football",)
    assert result.synced == ()

    fakes = Fakes(discovered=(NFL,))
    result, output = wizard(fakes)  # --yes confirms
    assert result.updated == ("football",) and result.synced == ("football",)
    text = config_path().read_text(encoding="utf-8")
    assert text == HAND_WRITTEN  # only the season changed, comments and all
    assert (paths.config_dir() / "config.toml.bak").read_bytes() == before


def test_a_changed_team_is_kept_under_yes_and_updated_through_set_values_once_confirmed() -> None:
    """Review fix: the team decides whose roster the automation acts on, so --yes keeps config.toml's team and says
    what differs; "Update?" defaults to no and only an explicit yes changes it."""
    write_existing(HAND_WRITTEN.replace("team_id = 3", "team_id = 8"))
    before = config_path().read_bytes()
    result, output = wizard(Fakes(discovered=(NFL,)))
    assert "team_id 8 -> 3" in output and result.updated == () and result.declined == ("football",)
    assert "ESPN says this account's team in football is team 3; config.toml has team 8." in output
    assert config_path().read_bytes() == before and result.synced == ()

    prompts = Recording(answers={"Update football (team_id 8 -> 3)?": True})
    result, output = wizard(Fakes(discovered=(NFL,)), prompts=prompts)
    assert prompts.defaults["Update football (team_id 8 -> 3)?"] is False  # asked, defaulting to no
    assert result.updated == ("football",) and raw_leagues()["football"]["team_id"] == 3


def test_yes_over_a_config_missing_a_discovered_league_adds_and_updates_nothing() -> None:
    """Review fix (the guard): --yes on an installation must not opt a league into automation. The account's NBA
    league is not in config.toml and ESPN names another team for the NFL one; --yes changes neither, and says so."""
    write_existing(HAND_WRITTEN.replace("team_id = 3", "team_id = 8"))
    before = config_path().read_bytes()
    fakes = Fakes()  # discovery finds NFL (team 3) and NBA
    result, output = wizard(fakes)
    assert (result.added, result.updated, result.declined) == ((), (), ("football",))
    assert config_path().read_bytes() == before and not (paths.config_dir() / "config.toml.bak").exists()
    assert f"Skipped {describe_league(NBA)}. (not in config.toml; fm setup without --yes asks" in output
    assert "--league-url <url> adds it" in output
    assert [call for call in fakes.calls if call[0] == "sync"] == []


def test_a_new_league_defaults_to_managed_on_a_fresh_install_or_by_url_only() -> None:
    prompts = Recording()
    wizard(Fakes(), prompts=prompts)  # no config.toml yet: setting the leagues up is the point
    assert prompts.defaults[f"Manage {describe_league(NFL)}?"] is True
    assert prompts.defaults[f"Manage {describe_league(NBA)}?"] is True

    config_path().write_text(HAND_WRITTEN, encoding="utf-8")  # NBA left out of an existing installation
    prompts = Recording()
    result, _ = wizard(Fakes(), prompts=prompts)
    assert prompts.defaults == {f"Manage {describe_league(NBA)}?": False} and result.added == ()

    nba_url = "https://fantasy.espn.com/basketball/team?leagueId=7654321&teamId=5"
    fakes = Fakes(confirmed={("nba", 7654321): NBA})
    result, output = wizard(fakes, SetupOptions(league_urls=(nba_url,)))  # --yes --league-url adds it
    assert result.added == ("nba",) and raw_leagues()["nba"]["team_id"] == 5
    assert [call for call in fakes.calls if call[0] == "sync"] == [("sync", "nba")]


def test_yes_moves_a_league_to_the_current_season_but_never_to_a_future_one() -> None:
    write_existing(HAND_WRITTEN.replace("season = 2026", "season = 2025"))
    result, _ = wizard(Fakes(discovered=(NFL,)))  # 2026 is the current NFL season on NOW (Oct 13, 2026)
    assert result.updated == ("football",) and raw_leagues()["football"]["season"] == 2026

    renewed = DiscoveredLeague(
        sport="nfl", espn_league_id=1234567, season=2027, team_id=3, league_name="Sample Football League"
    )
    before = config_path().read_bytes()
    result, output = wizard(Fakes(discovered=(renewed,)))
    assert result.updated == () and result.declined == ("football",) and config_path().read_bytes() == before
    assert "Season 2027 is not the current NFL season (2026); it changes only when you confirm it" in output
    prompts = Recording(answers={"Update football (season 2026 -> 2027)?": True})
    result, _ = wizard(Fakes(discovered=(renewed,)), prompts=prompts)
    assert prompts.defaults["Update football (season 2026 -> 2027)?"] is False
    assert result.updated == ("football",) and raw_leagues()["football"]["season"] == 2027


def test_an_existing_config_that_does_not_load_stops_setup_untouched() -> None:
    config_path().write_text("[[league]]\nkey = 'nfl'\n", encoding="utf-8")
    fakes = Fakes()
    with pytest.raises(SetupError, match="exists but does not load, so setup leaves it alone"):
        wizard(fakes)
    assert fakes.calls == []  # not even the session was read
    assert config_path().read_text(encoding="utf-8") == "[[league]]\nkey = 'nfl'\n"


def test_choosing_another_channel_sets_it_in_the_existing_config() -> None:
    write_existing()
    fakes = Fakes(discovered=(NFL,))
    result, output = wizard(fakes, SetupOptions(channel="telegram"))
    assert ("channel", "telegram") in fakes.calls
    assert result.channel == "telegram" and load_config().notify.channel == "telegram"
    assert 'Set channel = "telegram" under [notify] (was "ntfy").' in output
    assert raw_leagues()["football"]["policy"] == {"lineup": "approve"}


# --- session ----------------------------------------------------------------------------------------------------------


def test_no_session_runs_the_login_flow() -> None:
    fakes = Fakes(session=NotLoggedInError("no browser profile yet; run `fm login`"))
    _, output = wizard(fakes)
    assert fakes.kinds()[:3] == ["load_session", "login", "discover"]
    assert "No ESPN session yet: a browser window opens" in output


def test_an_expired_session_signs_in_again_and_an_expiring_one_offers_renewal() -> None:
    expired = EspnSession(espn_s2="s2", swid="{X}", expires_at=NOW - timedelta(days=1))
    fakes = Fakes(session=expired)
    _, output = wizard(fakes, SetupOptions(sync=False, autostart=False))
    assert "login" in fakes.kinds() and "ESPN session expired on" in output

    expiring = EspnSession(espn_s2="s2", swid="{X}", expires_at=NOW + timedelta(days=3))
    fakes = Fakes(session=expiring)
    wizard(fakes, SetupOptions(sync=False, autostart=False))  # --yes keeps the session that still works
    assert "login" not in fakes.kinds()
    fakes = Fakes(session=expiring, discovered=(NFL,))
    prompts = Scripted(True, "", "", "none")  # renew, no keys, no phone; the league is already configured
    wizard(fakes, SetupOptions(sync=False, autostart=False), prompts=prompts)
    assert prompts.asked[0] == "Renew the ESPN session now?" and "login" in fakes.kinds()


def test_a_session_that_cannot_be_read_stops_setup() -> None:
    from fm.browser.session import BrowserLaunchError

    fakes = Fakes(session=BrowserLaunchError("msedge failed to launch: profile in use"))
    with pytest.raises(SetupError, match="could not read the ESPN session"):
        wizard(fakes)
    assert not config_path().exists()


# --- keys and phone ---------------------------------------------------------------------------------------------------


def test_api_keys_are_saved_hidden_and_never_printed() -> None:
    prompts = Scripted(True, "balanced", True, "balanced", f" {ANTHROPIC_KEY} ", ODDS_KEY, "none")
    _, output = wizard(Fakes(), prompts=prompts)
    assert dotenv_values(env_path()) == {"ANTHROPIC_API_KEY": ANTHROPIC_KEY, "ODDS_API_KEY": ODDS_KEY}
    assert ANTHROPIC_KEY not in output and ODDS_KEY not in output
    assert "Saved ANTHROPIC_API_KEY" in output
    assert load_config().missing_secrets() == ()  # channel none: the Anthropic key is all that is required

    prompts = Scripted("none")  # a re-run asks for neither key again
    _, output = wizard(Fakes(), prompts=prompts)
    assert "ANTHROPIC_API_KEY is already set in .env." in output and not prompts.answers


def test_a_key_only_in_this_shells_environment_is_saved_to_env_and_never_printed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review fix: a key exported only in the shell running setup was reported as already set and never saved, so
    fm run at logon (which reads .env, not that shell) ran without it. --yes saves it; no value is ever printed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    _, output = wizard(Fakes(), SetupOptions(channel="none"))
    assert dotenv_values(env_path()).get("ANTHROPIC_API_KEY") == ANTHROPIC_KEY
    assert "ANTHROPIC_API_KEY is set in this shell's environment only" in output
    assert "Saved ANTHROPIC_API_KEY to" in output and ANTHROPIC_KEY not in output
    _, output = wizard(Fakes(), SetupOptions(channel="none"))
    assert "ANTHROPIC_API_KEY is already set in .env." in output  # saved once, then found in the file


def test_a_shell_only_key_the_user_declines_to_save_stays_out_of_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", ODDS_KEY)
    prompts = Recording(answers={"Save ODDS_API_KEY from this shell's environment to .env?": False})
    _, output = wizard(Fakes(), SetupOptions(channel="none"), prompts=prompts)
    assert prompts.defaults["Save ODDS_API_KEY from this shell's environment to .env?"] is True
    assert not env_path().exists() or "ODDS_API_KEY" not in dotenv_values(env_path())
    assert "Left ODDS_API_KEY out of .env" in output and ODDS_KEY not in output


def test_a_bad_env_value_is_named_but_never_shown() -> None:
    env_path().write_text("TELEGRAM_CHAT_ID=not-a-number-secret\n", encoding="utf-8")
    with pytest.raises(SetupError, match="invalid value for TELEGRAM_CHAT_ID") as raised:
        read_secrets(env_path(), environ={})
    assert "not-a-number-secret" not in str(raised.value)


def test_a_phone_setup_that_does_not_finish_is_a_warning() -> None:
    fakes = Fakes(channel_error="the code did not reach the bot in time; run fm notify setup again")
    result, output = wizard(fakes, SetupOptions(channel="telegram"))
    assert "warning: telegram is not set up yet: the code did not reach the bot in time" in output
    assert "Run fm notify setup --channel telegram to finish it." in output
    assert result.channel == "telegram" and load_config().notify.channel == "telegram"


def test_no_phone_skips_the_channel_setup() -> None:
    fakes = Fakes()
    result, output = wizard(fakes, SetupOptions(channel="none"))
    assert "channel" not in fakes.kinds() and result.channel == "none"
    assert load_config().notify.channel == "none" and "No phone" in output


# --- sync, autostart, summary -----------------------------------------------------------------------------------------


def test_a_failed_sync_is_a_warning_and_the_rest_goes_on() -> None:
    fakes = Fakes(sync_errors={"nfl": "ESPN answered HTTP 503"})
    result, output = wizard(fakes)
    assert "warning: nfl did not sync: ESPN answered HTTP 503; fm sync --league nfl retries" in output
    assert result.synced == ("nba",) and "  nba: synced" in output
    assert ("install", "autostart") in fakes.calls


def test_installed_jobs_are_left_alone_and_a_failed_install_is_a_warning() -> None:
    fakes = Fakes(autostart_installed=True, wake_installed=None, job_error="Register-ScheduledTask failed (1)")
    _, output = wizard(fakes)
    assert "fm run at logon: already installed (fm autostart install reinstalls it)." in output
    assert ("install", "wake") in fakes.calls  # the scheduler could not tell, so it is (re)installed
    assert "warning: the wake timer is not installed: Register-ScheduledTask failed (1); run fm schedule install" in (
        output
    )


def test_a_running_fm_bot_is_reported_and_left_running() -> None:
    fakes = Fakes(bots=(4321, 4322))
    _, output = wizard(fakes)
    assert "note: fm bot is running (pid 4321, 4322). fm run replaces it" in output
    assert "both poll the ntfy channel" in output and "setup leaves it running" in output


def test_with_fm_bot_running_fm_run_is_registered_but_not_started() -> None:
    """Review fix: setup used to start fm run beside a running fm bot, and both then polled the phone channel. Now the
    logon job is registered without starting, and setup says to stop fm bot, then run fm autostart install."""
    fakes = Fakes(bots=(4321,))
    _, output = wizard(fakes)
    installs = [call for call in fakes.calls if call[0] == "install"]
    assert installs == [("install", "autostart", "no-start"), ("install", "wake")]  # the wake timer as usual
    assert "fm run is registered but not started, so it does not poll beside fm bot." in output
    assert "Stop fm bot (Ctrl+C in its\nwindow), then run fm autostart install to start fm run now" in output
    assert "(or log out and back in)" in output


def test_with_fm_bot_running_a_scheduler_that_cannot_register_only_installs_nothing() -> None:
    fakes = Fakes(bots=(4321,), no_start=False)  # launchd starts a loaded agent at once
    _, output = wizard(fakes)
    assert [call for call in fakes.calls if call[0] == "install"] == [("install", "wake")]
    assert "warning: fm run at logon is not installed: this scheduler starts fm run as soon as" in output
    assert "Stop fm bot first (Ctrl+C in its window), then run fm autostart install." in output


def test_the_live_autostart_job_registers_without_starting_where_the_scheduler_can() -> None:
    from fm.commands.autostart import NO_START_BACKENDS

    job = setup_cmd.live_deps().autostart  # builds the Job only; nothing is installed or queried here
    assert (job.install_no_start is not None) == (sys.platform in NO_START_BACKENDS)


def test_without_fm_bot_fm_run_is_installed_and_started() -> None:
    fakes = Fakes()
    wizard(fakes)
    assert ("install", "autostart") in fakes.calls and ("install", "autostart", "no-start") not in fakes.calls


def test_the_summary_is_the_doctor_then_the_next_steps() -> None:
    _, output = wizard(Fakes())
    lines = output.splitlines()
    assert lines[-4:] == [
        "OK    config  config.toml valid",
        "WARN  tick    no tick has run yet  → fix: fm run",
        "",
        "Dashboard: fm ui. Ask Claude: fm claude install.",
    ]


@pytest.mark.parametrize(
    ("listing", "pids"),
    [
        ("1200 C:\\repo\\.venv\\Scripts\\fm.exe bot\n1201 C:\\repo\\.venv\\Scripts\\fm.exe run", [1200]),
        ('900 "C:\\Users\\x\\.local\\bin\\uv.exe" run fm bot --once\n901 python -m fm bot', [900, 901]),
        ("  77 /Users/x/repo/.venv/bin/python /Users/x/repo/.venv/bin/fm bot\n  78 /bin/zsh", [77]),
        ("55 fm botany\n56 fm setup\n57 notfm bot", []),
        ("12 powershell -Command Get-CimInstance Win32_Process -Filter \"CommandLine like '%bot%'\"", []),
        ("99 fm bot", []),  # this process
    ],
)
def test_fm_bot_is_found_in_a_process_listing(listing: str, pids: list[int]) -> None:
    assert parse_bot_pids(listing, exclude=(99,)) == pids
