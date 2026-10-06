"""The in-house NBA baseline (ROADMAP #43): per-minute rates, DARKO minutes adjusted for teammates out, back-to-backs
and blowout risk, the loader that fetches it, and its place in the blend and the backtest.

Offline. The model runs on small synthetic teams whose game logs are exact (every stat is a fixed rate times the minutes
played), so each adjustment can be checked against the arithmetic it claims. The loader runs against stub feeds, the
parsers against the recorded stats.nba.com payloads in ``tests/fixtures/sources/nba_stats/``, and the evaluation against
the synthetic NBA points-league replay in ``tests/fixtures/backtest/nba/`` (see its README and ``generate.py``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest
from typer.testing import CliRunner

from fm.cli import app
from fm.eval.backtest import BacktestData, load_fixture, run_backtest, with_blend
from fm.model import baseline_nba
from fm.model.baseline_nba import (
    DEFAULT_NBA_PARAMS,
    MINUTES_BACK_TO_BACK,
    MINUTES_BLOWOUT,
    MINUTES_BOUND,
    MINUTES_TEAMMATES_OUT,
    NBA_BASELINE_LABEL,
    NBA_BASELINE_SOURCE,
    NBA_BASELINE_STATS,
    NbaAppearance,
    NbaBaselineError,
    NbaBaselineLoader,
    NbaBaselineParams,
    NbaDayContext,
    NbaGameHistory,
    NbaPlayerBaseline,
    absorption_from_on_off,
    blowout_probabilities,
    darko_rates,
    nba_baseline_rows,
    nba_board_spreads,
    nba_league_rates,
    nba_out_probabilities,
    nba_period_date,
    nba_recency_weight,
    nba_schedule_context,
    project_nba_day,
    recent_minutes,
    regressed_nba_rates,
    starter_weight,
)
from fm.model.ids_nba import NbaCrosswalk
from fm.model.projections import (
    BLEND,
    ESPN,
    BlendWeights,
    ProjectionSourceRegistry,
    projection_source,
)
from fm.model.value_nba import blend_day, day_sources
from fm.sources.base import Fetched, SourceUnavailable
from fm.sources.darko import DarkoProjection, DarkoSource
from fm.sources.nba_schedule import NbaGame, NbaSchedule, NbaScheduleSource
from fm.sources.nba_stats import NbaStatsSource, on_off_frame, parse_result_sets
from fm.sources.odds import EspnScoreboardSource, Scoreboard, ScoreboardGame
from fm.store import PlayerIdRow, PlayerRow, ProjectionRow, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
STATS = FIXTURES / "sources" / "nba_stats"
NBA_BACKTEST = FIXTURES / "backtest" / "nba"
SEASON = 2027
NOW = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
DAY = date(2026, 11, 20)
BOS_ID, NYK_ID = 1610612738, 1610612752
runner = CliRunner()

RATES = {
    "FGM": 0.2,
    "FGA": 0.42,
    "FG3M": 0.05,
    "FG3A": 0.14,
    "FTM": 0.08,
    "FTA": 0.1,
    "OREB": 0.03,
    "DREB": 0.12,
    "AST": 0.1,
    "STL": 0.03,
    "BLK": 0.02,
    "TOV": 0.06,
    "PF": 0.07,
}
MINUTES = (36.0, 33.0, 32.0, 30.0, 28.0, 24.0, 20.0, 16.0, 11.0, 10.0)
IDS = tuple(range(101, 111))
"""nba.com ids of the ten BOS players of ``team()``, in minutes order."""


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store.open(tmp_path / "state.db")


# --- synthetic teams ---


def darko(
    nba_id: int,
    minutes: float,
    *,
    usage: float = 1.0,
    team_id: int | None = BOS_ID,
    available: bool = True,
    age: float = 27.0,
) -> DarkoProjection:
    """A DARKO row whose per-minute talent equals ``RATES`` (pace 100: per-100 = rate x 48), usage columns scaled."""
    scale = {"FGA": usage, "FTA": usage, "AST": usage, "TOV": usage}
    per = {key: RATES[key] * scale.get(key, 1.0) * 48.0 for key in RATES}
    return DarkoProjection(
        nba_id=nba_id,
        name=f"Player {nba_id}",
        team_id=team_id,
        team="BOS" if team_id == BOS_ID else None,
        available=available,
        age=age,
        minutes=minutes,
        pace=100.0,
        per_100={
            "pts": (2 * RATES["FGM"] + RATES["FG3M"] + RATES["FTM"]) * 48.0,
            "orb": per["OREB"],
            "drb": per["DREB"],
            "ast": per["AST"],
            "pf": per["PF"],
            "blk": per["BLK"],
            "stl": per["STL"],
            "tov": per["TOV"],
            "fta": per["FTA"],
            "fga": per["FGA"],
            "fg3a": per["FG3A"],
        },
        rates={"fg_pct": RATES["FGM"] / RATES["FGA"], "fg3_pct": RATES["FG3M"] / RATES["FG3A"], "ft_pct": 0.8},
        dpm={},
    )


def team(usage: Mapping[int, float] | None = None) -> list[DarkoProjection]:
    """Ten BOS players on ``MINUTES``; ``usage`` scales a player's talent in the usage columns."""
    shares = usage or {}
    return [darko(nba_id, minutes, usage=shares.get(nba_id, 1.0)) for nba_id, minutes in zip(IDS, MINUTES, strict=True)]


def appearance(
    nba_id: int, game: str, day: date, minutes: float, *, team_code: str = "BOS", usage: float = 1.0
) -> NbaAppearance:
    stats = {
        column: rate * minutes * (usage if column in baseline_nba.NBA_USAGE_COLUMNS else 1.0)
        for column, rate in RATES.items()
    }
    return NbaAppearance(nba_id, game, day, team_code, minutes, stats)


def games(
    days: list[date],
    *,
    absent: Mapping[int, set[int]] | None = None,
    minutes: Mapping[int, float] | None = None,
    extra: Mapping[tuple[int, int], float] | None = None,
    second_night: Mapping[int, float] | None = None,
) -> NbaGameHistory:
    """One game per ``days`` entry for the ten BOS players. ``absent`` maps a game index to the players who miss it;
    ``extra`` gives (game index, player) a different load than nominal; ``second_night`` maps a player to his minutes on
    a game the day after another."""
    nominal = dict(zip(IDS, MINUTES, strict=True)) | dict(minutes or {})
    rows: list[NbaAppearance] = []
    for index, day in enumerate(days):
        night = index > 0 and (day - days[index - 1]).days == 1
        for nba_id in IDS:
            if nba_id in (absent or {}).get(index, set()):
                continue
            load = (extra or {}).get((index, nba_id), nominal[nba_id])
            if night and second_night and nba_id in second_night:
                load = second_night[nba_id]
            rows.append(appearance(nba_id, f"G{index:03d}", day, load))
    return NbaGameHistory(rows)


def season_days(count: int = 24, *, start: date = date(2026, 10, 1)) -> list[date]:
    """Games on days 0 and 1 of every three: half of them second nights."""
    return [start + timedelta(days=3 * (index // 2) + index % 2) for index in range(count)]


def context(**fields: Any) -> NbaDayContext:
    return NbaDayContext(**({"day": DAY, "teams": frozenset({"BOS"})} | fields))


def minutes_of(projection: baseline_nba.NbaDayProjection) -> dict[int, float]:
    return {nba_id: player.minutes for nba_id, player in projection.players.items()}


def assert_accounted(player: NbaPlayerBaseline) -> None:
    """Every minute of the final number is a prior or a recorded adjustment."""
    assert player.prior_minutes + sum(a.minutes for a in player.adjustments) == pytest.approx(player.minutes)


# --- registration ---


def test_the_baseline_registers_itself_for_the_nba_with_a_loader() -> None:
    source = projection_source("nba", NBA_BASELINE_SOURCE)
    assert (source.name, source.label, source.stored) == (NBA_BASELINE_SOURCE, NBA_BASELINE_LABEL, False)
    assert isinstance(source.loader, NbaBaselineLoader)
    assert source.loadable and source.sport == "nba"
    baseline_nba._register()  # what a second import would do: nothing
    assert projection_source("nba", NBA_BASELINE_SOURCE) is source


def test_blend_day_picks_registered_sources_up() -> None:
    registry = ProjectionSourceRegistry()
    registry.register("nba", ESPN, stored=True)
    loader = NbaBaselineLoader()
    registry.register("nba", NBA_BASELINE_SOURCE, label=NBA_BASELINE_LABEL, loader=loader)
    day = day_sources(registry)
    assert day.names("nba") == (ESPN, NBA_BASELINE_SOURCE)
    assert day.lookup("nba", NBA_BASELINE_SOURCE).loader is loader


def test_the_stats_of_a_line_are_espn_abbreviations() -> None:
    from fm.espn.ids import FBA

    assert {FBA.stat_id(stat) for stat in (*NBA_BASELINE_STATS, "GP")}
    assert set(baseline_nba.NBA_LINE_STAT.values()) <= set(NBA_BASELINE_STATS)
    assert set(baseline_nba.NBA_LINE_STAT) == set(baseline_nba.NBA_RATE_COLUMNS) == set(baseline_nba.NBA_DARKO_KEYS)


@pytest.mark.parametrize(
    "bad",
    [
        {"half_life_days": 0.0},
        {"max_minutes": 60.0},
        {"usage_cap": -0.1},
        {"starter_ceiling": 10.0},
        {"b2b_cap": float("nan")},
        {"blowout_sd": 0.0},
    ],
)
def test_parameters_refuse_nonsense(bad: dict[str, Any]) -> None:
    with pytest.raises(NbaBaselineError):
        NbaBaselineParams(**bad)


# --- game logs ---


def test_game_logs_parse_from_the_recorded_payload() -> None:
    payload = (STATS / "leaguegamelog_2025-26.json").read_bytes()
    history = NbaGameHistory.from_frame(parse_result_sets(payload)["LeagueGameLog"])
    assert len(history) == 6 and set(history.players) == {203999, 1628983, 1641705}
    jokic = history.appearances(203999)
    assert [(a.day, a.team, a.minutes, a.stats["FGA"]) for a in jokic] == [
        (date(2026, 4, 10), "DEN", 34.0, 19.0),
        (date(2026, 4, 12), "DEN", 29.0, jokic[1].stats["FGA"]),
    ]
    assert history.latest_team(203999) == "DEN" and history.last_day(203999) == date(2026, 4, 12)
    assert jokic[0].usage == pytest.approx(19 + 0.44 * 8 + 3)


def test_a_game_log_frame_needs_its_columns_and_skips_rows_without_minutes() -> None:
    with pytest.raises(NbaBaselineError, match="lacks columns"):
        NbaGameHistory.from_frame(pl.DataFrame({"PLAYER_ID": [1]}))
    header = ["PLAYER_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MIN", *baseline_nba.NBA_RATE_COLUMNS]
    rows = [
        [1, "BOS", "G1", "2026-11-01", 30.0, *([1] * 13)],
        [2, "BOS", "G1", "2026-11-01", 0, *([1] * 13)],
        [3, "BOS", "G1", "2026-11-01T00:00:00", None, *([1] * 13)],
    ]
    history = NbaGameHistory.from_frame(pl.DataFrame(rows, schema=header, orient="row"))
    assert history.players == (1,)


def test_history_before_a_day_never_sees_that_day() -> None:
    history = games(season_days(6))
    cutoff = season_days(6)[3]
    known = history.before(cutoff)
    assert max(a.day for a in known.all_appearances()) < cutoff
    assert len(known) == 30


def test_second_nights_come_from_the_teams_own_calendar() -> None:
    history = games(season_days(6))
    flags = [history.second_night("BOS", f"G{index:03d}") for index in range(6)]
    assert flags == [False, True, False, True, False, True]
    assert history.second_night("NYK", "G001") is False


def test_a_split_compares_games_with_and_without_a_teammate() -> None:
    days = season_days(10)
    history = games(
        days, absent={index: {101} for index in (5, 6, 7)}, extra={(index, 108): 24.0 for index in (5, 6, 7)}
    )
    split = history.without("BOS", 101, 108)
    assert split is not None
    assert (split.games_with, split.games_without) == (7, 3)
    assert (split.minutes_with, split.minutes_without) == (16.0, 24.0)
    assert split.usage_with == pytest.approx(split.usage_without)
    assert history.without("BOS", 999, 108) is None  # a player who never played for the team
    assert history.without("NYK", 101, 108) is None


def test_the_second_night_split_separates_the_nights() -> None:
    history = games(season_days(8), second_night={101: 30.0})
    split = history.second_night_split(101, "BOS")
    assert split is not None
    assert (split.games_second_night, split.games_other) == (4, 4)
    assert (split.minutes_second_night, split.minutes_other) == (30.0, 36.0)
    assert history.second_night_split(1, "BOS") is None


# --- per-minute rates ---


def test_recent_games_weigh_more() -> None:
    assert nba_recency_weight(DAY, DAY - timedelta(days=45), 45.0) == pytest.approx(0.5)
    assert nba_recency_weight(DAY, DAY, 45.0) == 1.0 and nba_recency_weight(DAY, DAY + timedelta(days=3), 45.0) == 1.0
    assert nba_recency_weight(DAY, DAY - timedelta(days=90), 45.0) == pytest.approx(0.25)


def test_rates_are_pulled_toward_darko_by_pseudo_minutes() -> None:
    prior = darko_rates(darko(1, 30.0))
    assert prior["FGA"] == pytest.approx(RATES["FGA"])
    hot = [appearance(1, f"G{i}", DAY - timedelta(days=i + 1), 30.0, usage=2.0) for i in range(10)]
    estimate = regressed_nba_rates(hot, DAY, darko=prior, league={}, params=DEFAULT_NBA_PARAMS)
    assert estimate is not None and estimate.prior == "darko" and estimate.games == 10
    # 300 minutes of evidence (recency-weighted a little less) against 500 pseudo-minutes
    assert RATES["FGA"] < estimate.per_minute["FGA"] < 2 * RATES["FGA"]
    assert estimate.per_minute["STL"] == pytest.approx(RATES["STL"])  # unmoved columns stay put
    many = [appearance(1, f"H{i}", DAY - timedelta(days=i + 1), 30.0, usage=2.0) for i in range(200)]
    heavy = regressed_nba_rates(many, DAY, darko=prior, league={}, params=DEFAULT_NBA_PARAMS)
    assert heavy is not None and heavy.per_minute["FGA"] > estimate.per_minute["FGA"]
    assert heavy.per_minute["FGA"] == pytest.approx(2 * RATES["FGA"], rel=0.15)
    unplayed = regressed_nba_rates([], DAY, darko=prior, league={}, params=DEFAULT_NBA_PARAMS)
    assert unplayed is not None and unplayed.per_minute == prior


def test_rates_without_darko_use_the_league_and_without_either_nothing() -> None:
    history = games(season_days(6))
    league = nba_league_rates(history)
    assert league["FGA"] == pytest.approx(RATES["FGA"]) and nba_league_rates(NbaGameHistory([])) == {}
    estimate = regressed_nba_rates(history.appearances(101), DAY, darko={}, league=league, params=DEFAULT_NBA_PARAMS)
    assert estimate is not None and estimate.prior == "league"
    assert regressed_nba_rates([], DAY, darko={}, league={}, params=DEFAULT_NBA_PARAMS) is None
    assert darko_rates(darko(1, 0.0)) == {}


def test_recent_minutes_need_enough_games() -> None:
    history = games(season_days(6))
    assert recent_minutes(history.appearances(101), DAY, DEFAULT_NBA_PARAMS) == pytest.approx(36.0)
    assert recent_minutes(history.appearances(101)[:4], DAY, DEFAULT_NBA_PARAMS) is None


def test_absorption_comes_from_the_on_off_split() -> None:
    frame = on_off_frame(parse_result_sets((STATS / "teamplayeronoffdetails_OKC_2025-26.json").read_bytes()))
    absorbed = absorption_from_on_off(frame)
    gilgeous = absorbed[1628983]
    # off court the team uses fewer possessions a minute than on court, so most but not all of his production stays
    assert DEFAULT_NBA_PARAMS.absorption_floor <= gilgeous <= 1.0
    assert all(DEFAULT_NBA_PARAMS.absorption_floor <= share <= 1.0 for share in absorbed.values())
    with pytest.raises(NbaBaselineError, match="lacks columns"):
        absorption_from_on_off(pl.DataFrame({"MIN": [1.0]}))
    frame = pl.DataFrame(
        {
            "VS_PLAYER_ID": [5, 5, 6],
            "COURT_STATUS": ["On", "Off", "On"],
            "MIN": [30.0, 18.0, 30.0],
            "FGA": [60.0, 30.0, 60.0],
            "FTA": [0.0, 0.0, 0.0],
            "TOV": [0.0, 0.0, 0.0],
        }
    )
    assert absorption_from_on_off(frame) == {5: pytest.approx(0.833, abs=1e-3)}  # 30 / 18 against 60 / 30


# --- context helpers ---


def test_blowouts_get_likelier_with_the_spread_and_either_side_can_lose_one() -> None:
    even = blowout_probabilities(0.0)
    assert even[0] == pytest.approx(even[1])
    favorite = blowout_probabilities(-14.0)
    assert favorite[0] > even[0] > favorite[1]
    assert blowout_probabilities(14.0) == pytest.approx(favorite[::-1])
    assert 0.0 < favorite[0] < 1.0


def test_starters_are_weighted_by_minutes() -> None:
    assert (starter_weight(10.0), starter_weight(18.0), starter_weight(32.0), starter_weight(40.0)) == (
        0.0,
        0.0,
        1.0,
        1.0,
    )
    assert starter_weight(25.0) == pytest.approx(0.5)


def game(day: date, number: int, home: str, away: str, home_id: int = BOS_ID, away_id: int = NYK_ID) -> NbaGame:
    return NbaGame.model_validate(
        {
            "gameId": f"00226{number:05d}",
            "gameDateEst": f"{day.isoformat()}T00:00:00Z",
            "gameStatusText": "7:30 pm ET",
            "gameDateTimeUTC": f"{day.isoformat()}T00:30:00Z",
            "homeTeam": {"teamId": home_id, "teamTricode": home},
            "awayTeam": {"teamId": away_id, "teamTricode": away},
        }
    )


def schedule() -> NbaSchedule:
    first = date(2026, 10, 20)
    games_ = (
        game(first, 1, "BOS", "NYK"),
        game(first + timedelta(days=1), 2, "NYK", "DEN", NYK_ID, 1610612743),
        game(first + timedelta(days=2), 3, "BOS", "DEN", BOS_ID, 1610612743),
    )
    return NbaSchedule("2026", "00", None, games_, ())


def test_period_one_is_the_first_regular_season_day() -> None:
    games_ = schedule()
    assert nba_period_date(games_, 1) == date(2026, 10, 20) and nba_period_date(games_, 3) == date(2026, 10, 22)
    assert nba_period_date(games_, 0) is None
    assert nba_period_date(NbaSchedule("2026", "00", None, (), ()), 1) is None


def test_the_schedule_names_the_teams_playing_and_the_second_nights() -> None:
    games_ = schedule()
    assert nba_schedule_context(games_, date(2026, 10, 21)) == (frozenset({"NYK", "DEN"}), frozenset({"NYK"}))
    assert nba_schedule_context(games_, date(2026, 10, 22)) == (frozenset({"BOS", "DEN"}), frozenset({"DEN"}))
    assert nba_schedule_context(games_, date(2026, 10, 23)) == (frozenset(), frozenset())


def board(*lines: tuple[str, str, float | None]) -> Scoreboard:
    entries = []
    for number, (home, away, spread) in enumerate(lines):
        entry: dict[str, Any] = {
            "event_id": str(number),
            "kickoff": "2026-10-20T23:30:00Z",
            "name": f"{away} at {home}",
            "short_name": f"{away} @ {home}",
            "state": "pre",
            "home": {"id": 2, "abbreviation": home},
            "away": {"id": 18, "abbreviation": away},
        }
        if spread is not None:
            entry["line"] = {"provider": "DraftKings", "spread": spread, "over_under": 224.5}
        entries.append(ScoreboardGame.model_validate(entry))
    return Scoreboard(games=entries)


def test_each_team_gets_its_own_spread_from_a_scoreboard() -> None:
    spreads = nba_board_spreads(board(("BOS", "NY", -6.5), ("GS", "DEN", None), ("NO", "SA", 3.0)))
    assert spreads == {"BOS": -6.5, "NYK": 6.5, "NOP": 3.0, "SAS": -3.0}  # ESPN's NY, NO, SA read as nba.com's


def test_a_players_chance_of_being_out_is_the_designations() -> None:
    crosswalk = NbaCrosswalk(
        [
            PlayerIdRow(sport="nba", espn_id=espn, source="nba", source_id=str(nba), origin="x", as_of=NOW)
            for espn, nba in ((1, 11), (2, 12), (3, 13), (4, 14))
        ]
    )
    players = [
        PlayerRow(sport="nba", espn_id=1, full_name="Out", injury_status="OUT", as_of=NOW),
        PlayerRow(sport="nba", espn_id=2, full_name="Healthy", as_of=NOW),
        PlayerRow(sport="nba", espn_id=3, full_name="Gone", active=False, as_of=NOW),
        PlayerRow(sport="nba", espn_id=4, full_name="Doubtful", injury_status="DOUBTFUL", as_of=NOW),
        PlayerRow(sport="nba", espn_id=9, full_name="Unmapped", injury_status="OUT", as_of=NOW),
    ]
    out = nba_out_probabilities(players, crosswalk)
    assert out[11] == 1.0 and out[13] == 1.0 and 12 not in out and 9 not in out
    assert 0.0 < out[14] < 1.0


# --- the day's projection ---


def test_a_quiet_day_projects_the_prior_minutes_and_the_regressed_rates() -> None:
    result = project_nba_day(team(), games(season_days()), context())
    assert len(result.players) == 10 and result.notes == ("blowout risk skipped for BOS: no line",)
    star = result.players[101]
    assert star.minutes == 36.0 and star.prior_minutes == 36.0 and star.adjustments == () and star.usage_bump == 0.0
    assert star.rates.prior == "darko" and star.rates.games == 24
    line = star.line
    assert set(line) == {*NBA_BASELINE_STATS, "GP"} and line["GP"] == 1.0 and line["MIN"] == 36.0
    assert line["FGA"] == pytest.approx(RATES["FGA"] * 36.0)
    assert line["3PM"] == pytest.approx(RATES["FG3M"] * 36.0) and line["TO"] == pytest.approx(RATES["TOV"] * 36.0)
    assert line["PTS"] == pytest.approx(2 * line["FGM"] + line["3PM"] + line["FTM"])
    assert line["REB"] == pytest.approx(line["OREB"] + line["DREB"])
    assert_accounted(star)


def test_a_player_whose_team_does_not_play_gets_no_line() -> None:
    assert project_nba_day(team(), games(season_days()), context(teams=frozenset({"NYK"}))).players == {}
    assert project_nba_day(team(), games(season_days()), context(teams=frozenset())).notes == (
        f"no talent row belongs to a team that plays on {DAY}",
    )


def test_a_teammate_who_is_out_hands_his_minutes_to_the_others() -> None:
    result = project_nba_day(team(), games(season_days()), context(out={101: 1.0}))
    assert 101 not in result.players and len(result.players) == 9
    minutes = minutes_of(result)
    assert sum(minutes.values()) == pytest.approx(240.0 - 36.0 + 36.0)  # the team's minutes are conserved
    share = 36.0 / sum(MINUTES[1:])
    for nba_id, prior in zip(IDS[1:], MINUTES[1:], strict=True):
        assert minutes[nba_id] == pytest.approx(prior * (1 + share))  # no split to go on: proportional to playing time
        assert result.players[nba_id].adjustments[0].kind == MINUTES_TEAMMATES_OUT
        assert_accounted(result.players[nba_id])
    assert "Player 101 out (p=1.00, 36.0 min): proportional share" in result.players[102].adjustments[0].detail


def test_a_teammate_who_may_be_out_hands_over_a_share_of_his_minutes() -> None:
    result = project_nba_day(team(), games(season_days()), context(out={101: 0.5}))
    assert 101 in result.players and result.players[101].minutes == 36.0  # his own line is the one for if he plays
    gained = sum(player.minutes - player.prior_minutes for nba_id, player in result.players.items() if nba_id != 101)
    assert gained == pytest.approx(18.0)


def test_a_without_split_beats_the_proportional_share_as_the_evidence_grows() -> None:
    days = season_days(24)
    hosts = {index: {101} for index in range(2, 24, 2)}  # 11 games without the star
    history = games(days, absent=hosts, extra={(index, 108): 28.0 for index in hosts})
    plain = project_nba_day(team(), games(days), context(out={101: 1.0})).players[108]
    informed = project_nba_day(team(), history, context(out={101: 1.0})).players[108]
    assert informed.minutes > plain.minutes + 3.0
    assert "without-split over 11 games, weight 0.58" in informed.adjustments[0].detail
    thin = games(days, absent={2: {101}}, extra={(2, 108): 28.0})
    assert project_nba_day(team(), thin, context(out={101: 1.0})).players[108].minutes == pytest.approx(plain.minutes)
    assert_accounted(informed)


def test_gains_and_totals_are_bounded_and_the_cut_is_recorded() -> None:
    out = {101: 1.0, 102: 1.0, 103: 1.0, 104: 1.0}
    result = project_nba_day(team(), games(season_days()), context(out=out))
    assert len(result.players) == 6
    for player in result.players.values():
        assert player.minutes <= DEFAULT_NBA_PARAMS.max_minutes
        assert player.minutes - player.prior_minutes <= DEFAULT_NBA_PARAMS.max_gain + 1e-9
        assert_accounted(player)
    assert MINUTES_BOUND in {a.kind for player in result.players.values() for a in player.adjustments}
    assert sum(minutes_of(result).values()) < 240.0  # what the caps cut is dropped, not re-spread
    tight = NbaBaselineParams(max_gain=2.0)
    capped = project_nba_day(team(), games(season_days()), context(out={101: 1.0}), params=tight)
    assert all(p.minutes - p.prior_minutes <= 2.0 + 1e-9 for p in capped.players.values())


def test_a_player_darko_marks_unavailable_or_zeroes_is_out_and_still_vacates_minutes() -> None:
    talent = team()
    talent[0] = darko(101, 36.0, available=False)
    talent[1] = darko(102, 0.0)
    result = project_nba_day(talent, games(season_days()), context())
    assert 101 not in result.players and 102 not in result.players
    assert result.players[103].minutes > 32.0  # the star's 36 and the second option's 33 (from his own games) go round


def test_usage_follows_the_absent_players_possessions() -> None:
    history = games(season_days(), absent={})
    talent = team(usage={101: 2.0})
    base = project_nba_day(talent, history, context(out={101: 1.0}))
    bump = base.players[102].usage_bump
    assert 0.0 < bump <= DEFAULT_NBA_PARAMS.usage_cap
    lower = project_nba_day(talent, history, context(out={101: 1.0}, absorption={101: 0.7}))
    assert lower.players[102].usage_bump == pytest.approx(bump * 0.7 / DEFAULT_NBA_PARAMS.default_absorption)
    line = base.players[102].line
    assert line["FGA"] == pytest.approx(
        base.players[102].rates.per_minute["FGA"] * (1 + bump) * base.players[102].minutes
    )
    assert line["STL"] == pytest.approx(
        base.players[102].rates.per_minute["STL"] * base.players[102].minutes
    )  # not usage
    capped = project_nba_day(talent, history, context(out={101: 1.0}), params=NbaBaselineParams(usage_cap=0.01))
    assert capped.players[102].usage_bump == 0.01 and "bounded" in capped.players[102].usage_detail


def test_a_second_night_costs_starters_minutes_and_the_bench_gets_them() -> None:
    history = games(season_days())
    quiet = project_nba_day(team(), history, context())
    tired = project_nba_day(team(), history, context(back_to_back=frozenset({"BOS"})))
    star, bench = tired.players[101], tired.players[108]
    # his own 12 second nights show no drop, so they outweigh the 4% prior two to one: 36 x 0.04 x 6 / 18 minutes lost
    assert star.minutes == pytest.approx(36.0 - 36.0 * DEFAULT_NBA_PARAMS.b2b_prior * 6 / 18)
    assert star.minutes < quiet.players[101].minutes and bench.minutes > quiet.players[108].minutes
    assert {a.kind for p in tired.players.values() for a in p.adjustments} == {MINUTES_BACK_TO_BACK}
    assert sum(minutes_of(tired).values()) == pytest.approx(sum(minutes_of(quiet).values()))
    assert (
        project_nba_day(team(), history, context(teams=frozenset({"BOS"}), back_to_back=frozenset({"NYK"})))
        .players[101]
        .minutes
        == 36.0
    )
    for player in tired.players.values():
        assert_accounted(player)


def test_a_players_own_second_night_record_outweighs_the_prior_as_it_grows() -> None:
    days = season_days(24)
    plain = project_nba_day(team(), games(days), context(back_to_back=frozenset({"BOS"})))
    tired = project_nba_day(team(), games(days, second_night={101: 28.0}), context(back_to_back=frozenset({"BOS"})))
    assert tired.players[101].minutes < plain.players[101].minutes - 1.0
    detail = tired.players[101].adjustments[0].detail
    assert "12 second nights" in detail and "observed" in detail
    capped = project_nba_day(
        team(),
        games(days, second_night={101: 10.0}),
        context(back_to_back=frozenset({"BOS"})),
        params=NbaBaselineParams(b2b_cap=0.05),
    )
    assert capped.players[101].minutes == pytest.approx(36.0 * 0.95)


def test_blowout_risk_comes_from_the_spread_and_skips_without_a_line() -> None:
    history = games(season_days())
    favorite = project_nba_day(team(), history, context(spreads={"BOS": -14.0}))
    wins, losses = blowout_probabilities(-14.0)
    expected = wins * DEFAULT_NBA_PARAMS.blowout_loss_win + losses * DEFAULT_NBA_PARAMS.blowout_loss_loss
    star = favorite.players[101]
    assert star.minutes == pytest.approx(36.0 - expected) and star.adjustments[0].kind == MINUTES_BLOWOUT
    assert favorite.players[108].minutes > 16.0 and favorite.notes == ()
    even = project_nba_day(team(), history, context(spreads={"BOS": 0.0}))
    assert 36.0 - even.players[101].minutes < 36.0 - star.minutes
    assert favorite.players[110].adjustments[0].minutes > 0.0  # the bench picks them up
    assert "BOS" in project_nba_day(team(), history, context()).notes[0]
    huge = project_nba_day(team(), history, context(spreads={"BOS": -40.0}))
    assert 36.0 - huge.players[101].minutes <= DEFAULT_NBA_PARAMS.blowout_cap + 1e-9


def test_the_adjustments_stack_in_order_and_every_one_is_accounted_for() -> None:
    result = project_nba_day(
        team(usage={101: 1.5}),
        games(season_days()),
        context(out={101: 1.0, 105: 0.4}, spreads={"BOS": -9.0}, back_to_back=frozenset({"BOS"})),
    )
    kinds = [[a.kind for a in player.adjustments] for player in result.players.values()]
    assert any(MINUTES_TEAMMATES_OUT in row and MINUTES_BACK_TO_BACK in row and MINUTES_BLOWOUT in row for row in kinds)
    for player in result.players.values():
        assert_accounted(player)
        assert 0.0 <= player.minutes <= DEFAULT_NBA_PARAMS.max_minutes
    explanation = result.players[102].explain()
    assert explanation.startswith("Player 102 (BOS):") and "teammates_out" in explanation and "usage" in explanation


def test_nothing_after_the_day_leaks_into_it() -> None:
    days = season_days(24)
    history = games(days)
    future = [appearance(nba_id, f"F{nba_id}", DAY + timedelta(days=1), 48.0, usage=5.0) for nba_id in IDS]
    leaky = NbaGameHistory([*history.all_appearances(), *future])
    clean = project_nba_day(team(), history, context(out={101: 1.0}))
    assert project_nba_day(team(), leaky, context(out={101: 1.0})).players == clean.players


def test_players_darko_lacks_come_from_their_own_games_if_they_are_recent_enough() -> None:
    history = games(season_days())
    talent = team()[1:]  # no DARKO row for 101
    result = project_nba_day(talent, history, context())
    assert result.players[101].rates.prior == "league" and result.players[101].minutes == pytest.approx(36.0)
    old = project_nba_day(talent, history, context(day=date(2027, 3, 1)))
    assert 101 not in old.players
    few = NbaGameHistory(list(history.appearances(101))[:3])
    assert 101 not in project_nba_day(talent, few, context()).players
    no_minutes = project_nba_day([darko(120, 0.0), darko(121, 20.0)], NbaGameHistory([]), context())
    assert no_minutes.players[121].rates.prior == "darko" and no_minutes.players[121].minutes == 20.0


def test_a_darko_row_without_a_team_takes_the_one_in_the_game_logs() -> None:
    talent = [darko(nba_id, minutes, team_id=None) for nba_id, minutes in zip(IDS, MINUTES, strict=True)]
    assert len(project_nba_day(talent, games(season_days()), context()).players) == 10


# --- the loader ---


class StubTalent:
    clock = staticmethod(lambda: NOW)

    def __init__(self, rows: list[DarkoProjection], *, fail: bool = False) -> None:
        self.rows, self.fail, self.calls = rows, fail, 0

    def projections(self, **options: Any) -> Fetched[list[DarkoProjection]]:
        self.calls += 1
        if self.fail:
            raise SourceUnavailable("darko is down")
        return Fetched(self.rows, NOW - timedelta(hours=2), "darko", "talent", "current", cached=True)


class StubStats:
    def __init__(self, history: NbaGameHistory | None, *, fail_on_off: bool = False) -> None:
        self.history, self.fail_on_off, self.on_off_teams = history, fail_on_off, []

    def game_logs(self, season: str, **options: Any) -> Fetched[pl.DataFrame]:
        assert season == "2026-27"
        if self.history is None:
            raise SourceUnavailable("stats.nba.com is down")
        header = ["PLAYER_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MIN", *baseline_nba.NBA_RATE_COLUMNS]
        rows = [
            [
                a.nba_id,
                a.team,
                a.game_id,
                a.day.isoformat(),
                a.minutes,
                *(a.stats[c] for c in baseline_nba.NBA_RATE_COLUMNS),
            ]
            for a in self.history.all_appearances()
        ]
        return Fetched(
            pl.DataFrame(rows, schema=header, orient="row"), NOW - timedelta(hours=5), "nba_stats", "game_logs", "x"
        )

    def on_off(self, team_id: int, season: str, **options: Any) -> Fetched[pl.DataFrame]:
        self.on_off_teams.append(team_id)
        if self.fail_on_off:
            raise SourceUnavailable("on/off is down")
        frame = pl.DataFrame(
            {
                "VS_PLAYER_ID": [101, 101],
                "COURT_STATUS": ["On", "Off"],
                "MIN": [34.0, 14.0],
                "FGA": [70.0, 14.0],
                "FTA": [20.0, 6.0],
                "TOV": [12.0, 5.0],
            }
        )
        return Fetched(frame, NOW, "nba_stats", "on_off", "x")


class StubSchedule:
    def __init__(self, games_: NbaSchedule | None = None) -> None:
        self.games_ = games_

    def schedule(self, **options: Any) -> Fetched[NbaSchedule]:
        if self.games_ is None:
            raise SourceUnavailable("the CDN is down")
        return Fetched(
            self.games_, NOW - timedelta(hours=1), "nba_schedule", "schedule", "league", warnings=("a cup note",)
        )


class StubLines:
    def __init__(self, scoreboard: Scoreboard | None) -> None:
        self.scoreboard_ = scoreboard
        self.days: list[date] = []

    def scoreboard(self, game: object = None, *, day: date | None = None, **options: Any) -> Fetched[Scoreboard]:
        assert day is not None
        self.days.append(day)
        if self.scoreboard_ is None:
            raise SourceUnavailable("ESPN is down")
        return Fetched(self.scoreboard_, NOW, "espn_scoreboard", "scoreboard", "x")


def crosswalk(ids: tuple[int, ...] = IDS) -> NbaCrosswalk:
    return NbaCrosswalk(
        PlayerIdRow(sport="nba", espn_id=espn_of(nba), source="nba", source_id=str(nba), origin="x", as_of=NOW)
        for nba in ids
    )


def espn_of(nba_id: int) -> int:
    return 5_000_000 + nba_id


def loader_for(
    *,
    talent: StubTalent | None = None,
    stats: StubStats | None = None,
    calendar: StubSchedule | None = None,
    lines: StubLines | None = None,
    walk: NbaCrosswalk | None = None,
) -> tuple[NbaBaselineLoader, StubTalent, StubStats, StubLines]:
    talent = talent or StubTalent(team())
    stats = stats or StubStats(games(season_days()))
    lines = lines or StubLines(board(("BOS", "NY", -6.5)))
    calendar = calendar or StubSchedule(season_calendar())
    loader = NbaBaselineLoader(
        talent=cast(DarkoSource, talent),
        stats=cast(NbaStatsSource, stats),
        schedule=cast(NbaScheduleSource, calendar),
        lines=cast(EspnScoreboardSource, lines),
        crosswalk=walk if walk is not None else crosswalk(),
    )
    return loader, talent, stats, lines


def season_calendar() -> NbaSchedule:
    """Opening night is 2026-11-19, so scoring period 2 is the day the stubs' games happen on (2026-11-20, a second
    night for BOS)."""
    first = date(2026, 11, 19)
    return NbaSchedule(
        "2026",
        "00",
        None,
        (game(first, 1, "BOS", "NYK"), game(first + timedelta(days=1), 2, "BOS", "NYK")),
        (),
    )


def test_the_loader_builds_rows_by_espn_id_with_provenance(store: Store) -> None:
    loader, _, stats, lines = loader_for()
    fetched = loader(store, SEASON, 2, {})
    assert (fetched.source, fetched.dataset, fetched.degraded, fetched.stale) == (
        NBA_BASELINE_SOURCE,
        "game_logs",
        False,
        False,
    )
    assert fetched.as_of == NOW - timedelta(hours=5) and fetched.cached is False
    assert {row.espn_id for row in fetched.data} == {espn_of(nba) for nba in IDS}
    row = next(row for row in fetched.data if row.espn_id == espn_of(101))
    assert (row.sport, row.source, row.kind, row.season, row.scoring_period_id) == (
        "nba",
        NBA_BASELINE_SOURCE,
        "projected",
        SEASON,
        2,
    )
    assert row.stats["MIN"] < 36.0  # the second night of a back-to-back and BOS's spread both take starters' minutes
    assert lines.days == [date(2026, 11, 20)] and stats.on_off_teams == []  # nobody is out: no on/off pull
    assert fetched.warnings == ("a cup note",)
    assert set(loader.explanations) == {espn_of(nba) for nba in IDS}
    assert_accounted(loader.explanations[espn_of(101)])


def test_players_who_may_be_out_shape_the_day_and_pull_their_teams_on_off_split(store: Store) -> None:
    store.players.upsert(PlayerRow(sport="nba", espn_id=espn_of(101), full_name="Star", injury_status="OUT", as_of=NOW))
    loader, _, stats, _ = loader_for()
    fetched = loader(store, SEASON, 2, {})
    assert espn_of(101) not in {row.espn_id for row in fetched.data}
    assert stats.on_off_teams == [BOS_ID]
    assert loader.explanations[espn_of(102)].adjustments[0].kind == MINUTES_TEAMMATES_OUT
    assert "absorption 0.70" in loader.explanations[espn_of(102)].usage_detail  # the on/off split replaced the default
    failing, _, stats, _ = loader_for(stats=StubStats(games(season_days()), fail_on_off=True))
    again = failing(store, SEASON, 2, {})
    assert any("no on/off split for BOS" in warning for warning in again.warnings) and again.data


def test_missing_lines_skip_blowout_risk_with_a_warning(store: Store) -> None:
    loader, *_ = loader_for(lines=StubLines(None))
    fetched = loader(store, SEASON, 2, {})
    warnings = " | ".join(fetched.warnings)
    assert (
        "no lines for 2026-11-20, blowout risk skipped" in warnings
        and "blowout risk skipped for BOS: no line" in warnings
    )
    assert fetched.data and not fetched.degraded


@pytest.mark.parametrize("broken", ["talent", "logs", "calendar"])
def test_a_missing_required_feed_degrades_to_an_empty_result(store: Store, broken: str) -> None:
    loader, *_ = loader_for(
        talent=StubTalent([], fail=True) if broken == "talent" else None,
        stats=StubStats(None) if broken == "logs" else None,
        calendar=StubSchedule(None) if broken == "calendar" else None,
    )
    fetched = loader(store, SEASON, 2, {})
    assert fetched.data == () and fetched.degraded and fetched.source == NBA_BASELINE_SOURCE
    assert len(fetched.warnings) == 1 and fetched.warnings[0].startswith(
        "baseline_nba: unavailable, blending without it ("
    )
    assert loader.explanations == {}


def test_a_period_outside_the_schedule_and_a_bad_log_frame_degrade_too(store: Store) -> None:
    loader, *_ = loader_for()
    assert loader(store, SEASON, 0, {}).degraded
    assert "no regular-season day for period 0" in loader(store, SEASON, 0, {}).warnings[0]

    class BadLogs(StubStats):
        def game_logs(self, season: str, **options: Any) -> Fetched[pl.DataFrame]:
            return Fetched(pl.DataFrame({"PLAYER_ID": [1]}), NOW, "nba_stats", "game_logs", "x")

    bad, *_ = loader_for(stats=BadLogs(None))
    assert "lacks columns" in bad(store, SEASON, 2, {}).warnings[0]


def test_players_the_crosswalk_lacks_are_counted_not_loaded(store: Store) -> None:
    loader, *_ = loader_for(walk=crosswalk(IDS[:3]))
    assert {row.espn_id for row in loader(store, SEASON, 2, {}).data} == {espn_of(nba) for nba in IDS[:3]}
    nobody, *_ = loader_for(walk=NbaCrosswalk([]))
    fetched = nobody(store, SEASON, 2, {})
    assert fetched.data == () and any("maps none of 10 players" in warning for warning in fetched.warnings)


def test_baseline_rows_name_who_they_could_not_map() -> None:
    projection = project_nba_day(team(), games(season_days()), context())
    converted = nba_baseline_rows(projection, crosswalk(IDS[:9]), season=SEASON, scoring_period=2, as_of=NOW)
    assert len(converted.rows) == 9 and converted.unmapped == ("110",)
    assert converted.warnings == ("baseline_nba: blowout risk skipped for BOS: no line",)


def test_the_blend_takes_the_baseline_in_beside_espn(store: Store) -> None:
    store.projections.upsert(
        ProjectionRow(
            sport="nba",
            espn_id=espn_of(101),
            source=ESPN,
            kind="projected",
            season=SEASON,
            scoring_period_id=0,
            stats={"PTS": 2500.0, "GP": 50.0, "MIN": 1800.0},
            as_of=NOW,
        )
    )
    registry = ProjectionSourceRegistry()
    registry.register("nba", ESPN, stored=True)
    loader, *_ = loader_for()
    registry.register("nba", NBA_BASELINE_SOURCE, loader=loader)
    weights = BlendWeights.parse("[nba.default]\nespn = 1\nbaseline_nba = 1\n", sources=registry)
    result = blend_day(store, SEASON, 2, weights=weights, sources=registry)
    row = next(row for row in result.rows if row.espn_id == espn_of(101))
    assert result.blend.sources_for(espn_of(101), SEASON, 2) == (NBA_BASELINE_SOURCE, ESPN)
    own = loader.explanations[espn_of(101)].line
    assert row.source == BLEND and row.stats["PTS"] == pytest.approx((own["PTS"] + 50.0) / 2)
    assert {r.source for r in store.projections.for_period("nba", SEASON, 2)} == {NBA_BASELINE_SOURCE, ESPN, BLEND}


# --- the backtest ---


def read_inputs() -> tuple[list[DarkoProjection], NbaGameHistory, dict[int, NbaDayContext], NbaCrosswalk]:
    raw = json.loads((NBA_BACKTEST / "inputs.json").read_text(encoding="utf-8"))
    talent = [DarkoProjection.model_validate(entry["talent"]) for entry in raw["players"]]
    history = NbaGameHistory.from_frame(pl.DataFrame(raw["logs"], schema=raw["columns"], orient="row"))
    walk = NbaCrosswalk(
        PlayerIdRow(
            sport="nba",
            espn_id=entry["espn_id"],
            source="nba",
            source_id=str(entry["talent"]["nba_id"]),
            origin="x",
            as_of=NOW,
        )
        for entry in raw["players"]
    )
    contexts = {
        int(period): NbaDayContext(
            date.fromisoformat(day["day"]),
            frozenset(day["teams"]),
            {int(nba_id): chance for nba_id, chance in day["out"].items()},
            day["spreads"],
            frozenset(day["back_to_back"]),
        )
        for period, day in raw["days"].items()
    }
    return talent, history, contexts, walk


def baseline_backtest() -> BacktestData:
    """The NBA fixture with the baseline's rows for every period, computed from the fixture's own game logs."""
    talent, history, contexts, walk = read_inputs()
    rows: list[ProjectionRow] = []
    for period, day in contexts.items():
        rows.extend(
            nba_baseline_rows(
                project_nba_day(talent, history, day), walk, season=2027, scoring_period=period, as_of=NOW
            ).rows
        )
    return load_fixture(NBA_BACKTEST).with_source(NBA_BASELINE_SOURCE, rows)


WEIGHTS = "[nba.default]\nespn = 1\ndarko = 1\nbaseline_nba = 1\n"


def test_the_blend_with_the_baseline_is_no_worse_than_the_blend_without_it() -> None:
    data = baseline_backtest()
    weights = BlendWeights.parse(WEIGHTS)
    without = with_blend(data, weights, sources=["espn", "darko"], name="blend_without")
    both = with_blend(without, weights, sources=["espn", "darko", NBA_BASELINE_SOURCE], name="blend_with")
    report = run_backtest(both, sources=["blend_without", "blend_with"], restrict_to_common=True)
    plain, extended = report.source("blend_without").overall, report.source("blend_with").overall
    assert plain.samples == extended.samples > 200
    assert extended.mae <= plain.mae
    assert extended.rmse <= plain.rmse
    assert abs(extended.bias) <= abs(plain.bias)
    for position, error in report.source("blend_with").by_position.items():
        assert error.mae <= report.source("blend_without").by_position[position].mae + 1.0  # no position falls apart
    lineups = report.source("blend_without").lineups, report.source("blend_with").lineups
    assert (lineups[1].efficiency or 0.0) >= (lineups[0].efficiency or 0.0) - 1e-9


def test_the_baseline_alone_is_competitive_with_the_sources_it_joins() -> None:
    report = run_backtest(baseline_backtest(), restrict_to_common=True)
    baseline = report.source(NBA_BASELINE_SOURCE).overall
    assert baseline.samples == report.source("espn").overall.samples > 200
    assert baseline.mae <= min(report.source("espn").overall.mae, report.source("darko").overall.mae) * 1.05


def test_every_baseline_row_is_a_per_game_line_for_a_player_who_plays() -> None:
    data = baseline_backtest()
    rows = data.projections[NBA_BASELINE_SOURCE]
    assert rows and all(set(row.stats) == {*NBA_BASELINE_STATS, "GP"} for row in rows)
    assert all(row.stats["MIN"] > 0 and row.stats["PTS"] > 0 for row in rows)
    espn = {(row.scoring_period_id, row.espn_id) for row in data.projections["espn"]}
    assert {(row.scoring_period_id, row.espn_id) for row in rows} <= espn  # nobody the others know to be out


def test_the_command_runs_the_nba_fixture() -> None:
    result = runner.invoke(app, ["backtest", "--sport", "nba", "--fixtures", str(FIXTURES / "backtest")])
    assert result.exit_code == 0, result.output
    assert "blend" in result.output and "espn" in result.output and "darko" in result.output
