"""nflverse adapter against recorded nflreadpy downloads (no network).

nflreadpy fetches with ``requests``, so respx cannot intercept it. The ``recorded`` fixture replaces the one method
every nflreadpy loader funnels through (``NflverseDownloader._download_file``) with a server for the trimmed real files
under tests/fixtures/sources/nflverse, keyed by the URL's file name. The real loaders (season filters, roof cleanup, URL
building) still run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from nflreadpy.config import CacheMode, get_config
from nflreadpy.downloader import NflverseDownloader

from fm.sources.base import RateLimiter
from fm.sources.nflverse import NflverseSource

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sources" / "nflverse"
T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
SEASON = 2026

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/"
DYNASTYPROCESS = "https://github.com/dynastyprocess/data/raw/master/files/"
FFOPPORTUNITY = "https://github.com/ffverse/ffopportunity/releases/download/"


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


class RecordedDownloads:
    """Serves nflreadpy download URLs from the fixture directory, parsed the way nflreadpy parses them."""

    def __init__(self) -> None:
        self.urls: list[str] = []
        self.fail_with: Exception | None = None

    def __call__(self, url: str, **kwargs: object) -> pl.DataFrame:
        self.urls.append(url)
        if self.fail_with is not None:
            raise self.fail_with
        path = FIXTURES / url.rsplit("/", 1)[1]
        if not path.is_file():
            raise AssertionError(f"no recorded fixture for {url}")
        if path.suffix == ".parquet":
            return pl.read_parquet(path)
        return pl.read_csv(path, null_values=["NA", "NULL", ""])


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> RecordedDownloads:
    served = RecordedDownloads()
    # A callable instance is not a descriptor, so nflreadpy's ``self._download_file(url, **kw)`` reaches it as-is.
    monkeypatch.setattr(NflverseDownloader, "_download_file", served)
    return served


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def source(recorded: RecordedDownloads, clock: FakeClock, tmp_path: Path) -> NflverseSource:
    return NflverseSource(cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock)


def columns(frame: pl.DataFrame, *names: str) -> None:
    missing = [name for name in names if name not in frame.columns]
    assert not missing, f"missing columns {missing}"


# --- datasets ---


def test_player_stats_weekly(source: NflverseSource, recorded: RecordedDownloads, tmp_path: Path) -> None:
    result = source.player_stats(SEASON)
    frame = result.data
    columns(frame, "player_id", "player_display_name", "position", "team", "week", "targets", "target_share")
    columns(frame, "carries", "rushing_yards", "receptions", "receiving_yards", "fantasy_points", "fantasy_points_ppr")
    assert frame["season"].unique().to_list() == [SEASON]
    assert frame.schema["week"] == pl.Int32  # dtypes arrive exactly as nflverse ships them
    assert recorded.urls == [f"{NFLVERSE}stats_player/stats_player_week_{SEASON}.parquet"]
    assert (result.as_of, result.cached, result.source, result.dataset) == (T0, False, "nflverse", "player_stats")
    raw = tmp_path / "cache" / "nflverse" / "player_stats" / f"{SEASON}_week.parquet"
    assert result.raw_path == raw and pl.read_parquet(raw).equals(frame)


def test_player_stats_summary_level_is_part_of_the_key(source: NflverseSource, recorded: RecordedDownloads) -> None:
    with pytest.raises(AssertionError, match="no recorded fixture"):
        source.player_stats(SEASON, summary_level="reg")
    assert recorded.urls[-1] == f"{NFLVERSE}stats_player/stats_player_reg_{SEASON}.parquet"


def test_snap_counts(source: NflverseSource, recorded: RecordedDownloads) -> None:
    frame = source.snap_counts(SEASON).data
    columns(frame, "pfr_player_id", "player", "position", "team", "week", "offense_snaps", "offense_pct")
    assert frame.filter(pl.col("player") == "Jahmyr Gibbs").height > 0
    assert recorded.urls == [f"{NFLVERSE}snap_counts/snap_counts_{SEASON}.parquet"]


def test_injuries_with_practice_status(source: NflverseSource, recorded: RecordedDownloads) -> None:
    frame = source.injuries(SEASON).data
    columns(frame, "gsis_id", "full_name", "team", "week", "report_status", "practice_status", "report_primary_injury")
    out = frame.filter((pl.col("gsis_id") == "00-0041512") & (pl.col("week") == 4))  # Jadarian Price, week 4
    assert out["report_status"].to_list() == ["Out"]
    assert out["practice_status"].to_list() == ["Did Not Participate In Practice"]
    assert recorded.urls == [f"{NFLVERSE}injuries/injuries_{SEASON}.parquet"]


def test_depth_charts(source: NflverseSource, recorded: RecordedDownloads) -> None:
    frame = source.depth_charts(SEASON).data
    columns(frame, "dt", "team", "player_name", "espn_id", "gsis_id", "pos_grp", "pos_abb", "pos_rank")
    gibbs = frame.filter(pl.col("gsis_id") == "00-0039139")
    assert gibbs["team"].to_list() == ["DET"] and gibbs["pos_rank"].to_list() == [1]
    assert recorded.urls == [f"{NFLVERSE}depth_charts/depth_charts_{SEASON}.parquet"]


def test_schedules_carry_betting_lines(source: NflverseSource, recorded: RecordedDownloads) -> None:
    frame = source.schedules(SEASON).data
    columns(frame, "game_id", "week", "gameday", "weekday", "gametime", "away_team", "home_team")
    columns(frame, "spread_line", "total_line", "away_moneyline", "home_moneyline", "roof", "surface", "stadium")
    assert frame["season"].unique().to_list() == [SEASON]
    assert set(frame["roof"].drop_nulls().unique().to_list()) <= {"dome", "outdoors", "closed", "open"}
    assert frame.filter(pl.col("week") == 4)["total_line"].null_count() == 0
    assert recorded.urls == [f"{NFLVERSE}schedules/games.parquet"]


def test_ff_playerids_maps_espn_sleeper_and_gsis(source: NflverseSource, recorded: RecordedDownloads) -> None:
    frame = source.ff_playerids().data
    columns(frame, "name", "position", "team", "espn_id", "sleeper_id", "gsis_id", "fantasypros_id", "pfr_id")
    allen = frame.filter(pl.col("sleeper_id") == 4984)
    assert allen["name"].to_list() == ["Josh Allen"]
    assert allen["espn_id"].to_list() == [3918298] and allen["gsis_id"].to_list() == ["00-0034857"]
    assert recorded.urls == [f"{DYNASTYPROCESS}db_playerids.csv"]


def test_ff_opportunity_expected_points(source: NflverseSource, recorded: RecordedDownloads) -> None:
    frame = source.ff_opportunity(SEASON).data
    columns(frame, "player_id", "full_name", "position", "week", "total_fantasy_points_exp", "total_fantasy_points")
    columns(frame, "rec_attempt", "rush_attempt", "rec_yards_gained_exp", "rush_yards_gained_exp")
    assert recorded.urls == [f"{FFOPPORTUNITY}latest-data/ep_weekly_{SEASON}.parquet"]


def test_ff_rankings_weekly_ecr(source: NflverseSource, recorded: RecordedDownloads) -> None:
    frame = source.ff_rankings().data
    columns(frame, "fantasypros_id", "player_name", "pos", "team", "rank", "ecr", "sd", "best", "worst")
    top = frame.sort("rank").row(0, named=True)
    assert (top["player_name"], top["pos"], top["ecr"]) == ("Josh Allen", "QB", 1.0)
    assert recorded.urls == [f"{DYNASTYPROCESS}fp_latest_weekly.csv"]


def test_ff_rankings_type_selects_the_file(source: NflverseSource, recorded: RecordedDownloads) -> None:
    with pytest.raises(AssertionError, match="no recorded fixture"):
        source.ff_rankings("draft")
    assert recorded.urls == [f"{DYNASTYPROCESS}db_fpecr_latest.csv"]


def test_fixture_players_line_up_across_datasets(source: NflverseSource) -> None:
    # The same players appear in every recorded dataset so later crosswalk/blend tests can join them.
    ids = source.ff_playerids().data["gsis_id"].drop_nulls().to_list()
    assert set(source.player_stats(SEASON).data["player_id"].to_list()) <= set(ids)
    assert set(source.injuries(SEASON).data["gsis_id"].to_list()) <= set(ids)
    assert set(source.ff_opportunity(SEASON).data["player_id"].to_list()) <= set(ids)


# --- caching and freshness ---


def test_second_call_is_served_from_cache(
    source: NflverseSource, recorded: RecordedDownloads, clock: FakeClock
) -> None:
    first = source.player_stats(SEASON)
    clock.advance(hours=5)
    again = source.player_stats(SEASON)
    assert len(recorded.urls) == 1
    assert again.cached is True and again.as_of == first.as_of == T0
    assert again.data.equals(first.data)


def test_schedule_ttl_is_short_because_lines_move(
    source: NflverseSource, recorded: RecordedDownloads, clock: FakeClock
) -> None:
    source.schedules(SEASON)
    clock.advance(minutes=4)
    assert source.schedules(SEASON).cached is True
    later = clock.advance(minutes=2)
    refreshed = source.schedules(SEASON)
    assert refreshed.cached is False and refreshed.as_of == later
    assert len(recorded.urls) == 2


def test_failed_refresh_serves_the_stale_frame(
    source: NflverseSource, recorded: RecordedDownloads, clock: FakeClock
) -> None:
    first = source.injuries(SEASON)
    clock.advance(hours=4)
    recorded.fail_with = ConnectionError("Failed to download: offline")  # what nflreadpy raises
    stale = source.injuries(SEASON)
    assert stale.stale is True and stale.as_of == T0 and stale.data.equals(first.data)
    assert "offline" in stale.warnings[0]


def test_nflreadpy_cache_is_switched_off(source: NflverseSource) -> None:
    assert get_config().cache_mode == CacheMode.OFF


def test_custom_loader_is_a_seam(clock: FakeClock, tmp_path: Path, recorded: RecordedDownloads) -> None:
    class Loader:
        def load_schedules(self, seasons: int | list[int] | bool | None = True) -> pl.DataFrame:
            return pl.DataFrame({"season": [seasons], "week": [1], "spread_line": [-3.5]})

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"unexpected loader call {name}")

    source = NflverseSource(loader=Loader(), cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock)  # type: ignore[arg-type]
    frame = source.schedules(SEASON).data
    assert frame["season"].to_list() == [SEASON] and frame["spread_line"].to_list() == [-3.5]
    assert recorded.urls == []
