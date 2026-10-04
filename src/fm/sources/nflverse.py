"""nflverse datasets through ``nflreadpy`` (Polars). ``nfl_data_py`` is archived; never use it.

Datasets, by nflverse name: weekly player stats, snap counts, injuries with practice status, depth charts, schedules
with betting lines, ``ff_playerids`` (the DynastyProcess ID map: ESPN, Sleeper, GSIS, FantasyPros, ...),
``ff_opportunity`` (expected fantasy points) and ``ff_rankings`` (FantasyPros ECR). Each method returns the frame
exactly as nflreadpy delivers it; column names follow the nflverse data dictionaries
(https://nflreadr.nflverse.com/articles/). ``player_id`` in stats and ``gsis_id`` elsewhere are GSIS ids.

nflreadpy keeps its own 24 hour in-memory cache, which would defeat a 5 minute schedule TTL inside a long-running
process (``fm bot``), so the adapter switches it off and :mod:`fm.sources.base` owns caching (parquet under
``cache_dir()/sources/nflverse/``) and the ``as_of`` stamps.

Offline tests patch ``nflreadpy.downloader.NflverseDownloader._download_file`` to serve recorded files by URL, which
exercises nflreadpy's real loaders; a custom ``loader`` is the other seam.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar, Literal, Protocol, Unpack

import polars as pl
from nflreadpy.config import CacheMode, update_config

# nflreadpy re-exports these at package level, but each shares its name with the submodule that defines it, which
# makes the package attribute ambiguous to a type checker. Import from the defining modules instead.
from nflreadpy.load_depth_charts import load_depth_charts
from nflreadpy.load_ffverse import load_ff_opportunity, load_ff_playerids, load_ff_rankings
from nflreadpy.load_injuries import load_injuries
from nflreadpy.load_schedules import load_schedules
from nflreadpy.load_snap_counts import load_snap_counts
from nflreadpy.load_stats import load_player_stats

from fm.sources.base import Fetched, FetchOptions, RateLimiter, Source, utcnow

type SummaryLevel = Literal["week", "reg", "post", "reg+post"]
type RankingsType = Literal["draft", "week", "all"]
type OpportunityStat = Literal["weekly", "pbp_pass", "pbp_rush"]
type Seasons = int | list[int] | bool | None


class NflreadLoader(Protocol):
    """The subset of nflreadpy this adapter calls. Tests may substitute an object with the same methods."""

    def load_player_stats(self, seasons: Seasons = ..., summary_level: SummaryLevel = ...) -> pl.DataFrame: ...
    def load_snap_counts(self, seasons: Seasons = ...) -> pl.DataFrame: ...
    def load_injuries(self, seasons: Seasons = ...) -> pl.DataFrame: ...
    def load_depth_charts(self, seasons: Seasons = ...) -> pl.DataFrame: ...
    def load_schedules(self, seasons: Seasons = ...) -> pl.DataFrame: ...
    def load_ff_playerids(self) -> pl.DataFrame: ...
    def load_ff_opportunity(
        self,
        seasons: int | list[int] | None = ...,
        stat_type: OpportunityStat = ...,
        model_version: Literal["latest", "v1.0.0"] = ...,
    ) -> pl.DataFrame: ...
    def load_ff_rankings(self, type: RankingsType = ...) -> pl.DataFrame: ...


class NflreadpyLoader:
    """The default loader: nflreadpy's own functions."""

    load_player_stats = staticmethod(load_player_stats)
    load_snap_counts = staticmethod(load_snap_counts)
    load_injuries = staticmethod(load_injuries)
    load_depth_charts = staticmethod(load_depth_charts)
    load_schedules = staticmethod(load_schedules)
    load_ff_playerids = staticmethod(load_ff_playerids)
    load_ff_opportunity = staticmethod(load_ff_opportunity)
    load_ff_rankings = staticmethod(load_ff_rankings)


def disable_nflreadpy_cache() -> None:
    """Make nflreadpy download on every call; this package's cache decides when that happens."""
    update_config(cache_mode=CacheMode.OFF)


class NflverseSource(Source):
    """nflverse data, one method per dataset. All methods return ``Fetched[pl.DataFrame]``."""

    name: ClassVar[str] = "nflverse"
    min_interval: ClassVar[float] = 1.0
    """GitHub release downloads: one per second is polite and nowhere near the rate limit."""
    ttl: ClassVar[Mapping[str, timedelta]] = {
        "player_stats": timedelta(hours=6),  # nflverse rebuilds stats nightly
        "snap_counts": timedelta(hours=6),  # refreshed about every 6 h in season
        "injuries": timedelta(hours=3),  # practice reports land Wed-Fri afternoons
        "depth_charts": timedelta(hours=6),  # daily
        "schedules": timedelta(minutes=5),  # betting lines move; nflverse re-pulls every ~5 min
        "ff_playerids": timedelta(hours=24),
        "ff_opportunity": timedelta(hours=6),
        "ff_rankings": timedelta(hours=6),
    }

    def __init__(
        self,
        *,
        loader: NflreadLoader | None = None,
        cache_root: Path | None = None,
        limiter: RateLimiter | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        super().__init__(cache_root=cache_root, limiter=limiter, clock=clock)
        self.loader: NflreadLoader = loader if loader is not None else NflreadpyLoader()
        disable_nflreadpy_cache()

    def player_stats(
        self,
        season: int,
        *,
        summary_level: SummaryLevel = "week",
        **options: Unpack[FetchOptions],
    ) -> Fetched[pl.DataFrame]:
        """Player stat lines per week (or per season with another ``summary_level``); ``player_id`` is the GSIS id."""
        return self.fetch_frame(
            "player_stats",
            f"{season}_{summary_level}",
            load=lambda: self.loader.load_player_stats(seasons=season, summary_level=summary_level),
            meta={"season": season, "summary_level": summary_level},
            **options,
        )

    def snap_counts(self, season: int, **options: Unpack[FetchOptions]) -> Fetched[pl.DataFrame]:
        """Offense/defense/special-teams snaps and percentages per player-game (Pro Football Reference ids)."""
        return self.fetch_frame(
            "snap_counts",
            str(season),
            load=lambda: self.loader.load_snap_counts(seasons=season),
            meta={"season": season},
            **options,
        )

    def injuries(self, season: int, **options: Unpack[FetchOptions]) -> Fetched[pl.DataFrame]:
        """Official injury reports: ``report_status`` (game designation) and ``practice_status`` per player-week."""
        return self.fetch_frame(
            "injuries",
            str(season),
            load=lambda: self.loader.load_injuries(seasons=season),
            meta={"season": season},
            **options,
        )

    def depth_charts(self, season: int, **options: Unpack[FetchOptions]) -> Fetched[pl.DataFrame]:
        """Depth chart snapshots (one per ``dt`` timestamp) with ``pos_abb``, ``pos_rank`` and ESPN/GSIS ids."""
        return self.fetch_frame(
            "depth_charts",
            str(season),
            load=lambda: self.loader.load_depth_charts(seasons=season),
            meta={"season": season},
            **options,
        )

    def schedules(self, season: int, **options: Unpack[FetchOptions]) -> Fetched[pl.DataFrame]:
        """Games with kickoff (``gameday``/``gametime``), ``spread_line``, ``total_line``, moneylines, roof, surface."""
        return self.fetch_frame(
            "schedules",
            str(season),
            load=lambda: self.loader.load_schedules(seasons=season),
            meta={"season": season},
            **options,
        )

    def ff_playerids(self, **options: Unpack[FetchOptions]) -> Fetched[pl.DataFrame]:
        """DynastyProcess ID map: ``espn_id``, ``sleeper_id``, ``gsis_id``, ``fantasypros_id``, ``pfr_id`` and more."""
        return self.fetch_frame("ff_playerids", "all", load=self.loader.load_ff_playerids, **options)

    def ff_opportunity(
        self,
        season: int,
        *,
        stat_type: OpportunityStat = "weekly",
        **options: Unpack[FetchOptions],
    ) -> Fetched[pl.DataFrame]:
        """ffopportunity expected points per player-week (``total_fantasy_points_exp`` and components)."""
        return self.fetch_frame(
            "ff_opportunity",
            f"{season}_{stat_type}",
            load=lambda: self.loader.load_ff_opportunity(seasons=season, stat_type=stat_type),
            meta={"season": season, "stat_type": stat_type},
            **options,
        )

    def ff_rankings(self, type: RankingsType = "week", **options: Unpack[FetchOptions]) -> Fetched[pl.DataFrame]:
        """FantasyPros expert consensus ranks (``ecr``, ``sd``, ``best``, ``worst``); ``week`` is the weekly ECR."""
        return self.fetch_frame(
            "ff_rankings",
            type,
            load=lambda: self.loader.load_ff_rankings(type=type),
            meta={"type": type},
            **options,
        )
