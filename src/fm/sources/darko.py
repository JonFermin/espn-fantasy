"""DARKO projections (Kostya Medvedovsky): per-100-possession talent with projected minutes and pace, plus daily lines.

DARKO's Shiny app went offline in June 2026 (its successor, darko.app, only offers a client-side CSV button), but the
public Google Sheet behind it still exports CSV. The "talent" tab (``gid=284274620``) has one row per player:
``nba_id``, ``player_name``, position, ``available``, ``tm_id``/``team_name`` (``-999`` and blank for free agents),
projected ``minutes`` and ``pace``, per-100-possession rates (``pts_100``, ``orb_100``, ``drb_100``, ``ast_100``,
``stl_100``, ``blk_100``, ``tov_100``, ``pf_100``, ``fga_100``, ``fg3a_100``, ``fta_100``, ``rim_fga_100``),
shooting percentages (``fg_pct``, ``fg2_pct``, ``fg3_pct``, ``ft_pct``), usage, rebound and assist rates, and DPM. The
"daily" tab (``gid=951008984``) has per-game box projections (``minutes``, ``pts``, ``blk``, ``orb``, ``drb``, ``ast``,
``stl``, ``fg3a``, ``fg3m``) for a date, with opponent and ``game_id`` when the player's team plays that day.

Projections are stat lines, never points (CLAUDE.md). :meth:`DarkoProjection.per_game` turns the rates into a
per-game line the way darko.app's Fantasy Lab does: ``stat_100 × possessions / 100`` with ``possessions = pace ×
minutes / 48``, and made shots as attempts × percentage. Keys are nba.com-style lowercase (``pts``, ``reb``, ``ast``,
``fgm``, ``fga``, ``fg3m``, ``ftm``, ...); the NBA plugin maps them onto ESPN stat ids. The sheet URLs are constructor
arguments so a mirror replaces them without a code change, and a Google error page (HTML) fails the column check,
lands in ``.rejected.csv`` and leaves the last good copy in place.
"""

from __future__ import annotations

import csv
import io
import logging
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import ClassVar, Unpack

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from fm.sources.base import Fetched, FetchOptions, HttpSource, RateLimiter, SourceSchemaError, utcnow

logger = logging.getLogger(__name__)

SHEET_ID = "1mhwOLqPu2F9026EQiVxFPIN1t9RGafGpl-dokaIsm9c"
TALENT_GID = "284274620"
DAILY_GID = "951008984"
TALENT_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={TALENT_GID}"
DAILY_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={DAILY_GID}"

PER_100_COLUMNS = (
    "pts_100",
    "orb_100",
    "drb_100",
    "ast_100",
    "pf_100",
    "blk_100",
    "stl_100",
    "tov_100",
    "fta_100",
    "fga_100",
    "rim_fga_100",
    "fg3a_100",
)
RATE_COLUMNS = (
    "usg_pct",
    "orb_pct",
    "drb_pct",
    "ast_pct",
    "blk_pct",
    "stl_pct",
    "tov_pct",
    "ft_ar",
    "fg_pct",
    "fg2_pct",
    "fg3_pct",
    "ft_pct",
    "fg3_ar",
    "rim_fg_pct",
)
DPM_COLUMNS = ("dpm", "o_dpm", "d_dpm", "box_odpm", "box_ddpm", "on_off_odpm", "on_off_ddpm")
TALENT_REQUIRED = (
    "nba_id",
    "player_name",
    "minutes",
    "pace",
    "pts_100",
    "orb_100",
    "drb_100",
    "ast_100",
    "stl_100",
    "blk_100",
    "tov_100",
    "fga_100",
    "fg3a_100",
    "fta_100",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
)
DAILY_STATS = ("pts", "blk", "orb", "drb", "ast", "stl", "fg3a", "fg3m")
DAILY_REQUIRED = ("nba_id", "date", "player_name", "minutes", *DAILY_STATS)


def _num(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.upper() in ("NA", "NAN", "NULL", "NONE"):
        return None
    return float(text)


def _id(value: object) -> int | None:
    """Ids arrive float-formatted (``203999.00``); ``-999`` is DARKO's "none" (no team, no game)."""
    number = _num(value)
    return None if number is None or number < 0 else int(number)


def _text(value: object) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def _flag(value: object, *, default: bool) -> bool:
    number = _num(value)
    return default if number is None else bool(number)


class DarkoModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DarkoProjection(DarkoModel):
    """One talent row. ``per_100``/``rates``/``dpm`` keep DARKO's column names (minus the ``_100`` suffix)."""

    nba_id: int
    name: str
    position: str | None = None
    """Listed position (``C-F``, ``G``)."""
    x_position: str | None = None
    """DARKO's modeled position bucket (``c_pos``, ``pg_pos``)."""
    team_id: int | None = None
    """nba.com team id; ``None`` for free agents."""
    team: str | None = None
    available: bool = True
    age: float | None = None
    career_games: int | None = None
    minutes: float
    """Projected minutes per game."""
    pace: float
    """Projected team possessions per 48 minutes."""
    per_100: dict[str, float]
    rates: dict[str, float]
    dpm: dict[str, float]

    @property
    def possessions(self) -> float:
        """Possessions the player is on the floor for per game."""
        return self.pace * self.minutes / 48

    def per_game(self) -> dict[str, float]:
        """The per-game stat line: ``min``, ``pts``, ``oreb``, ``dreb``, ``reb``, ``ast``, ``stl``, ``blk``, ``tov``,
        ``pf``, ``fga``, ``fgm``, ``fg3a``, ``fg3m``, ``fta``, ``ftm``. Zero across the board at zero minutes."""
        scale = self.possessions / 100

        def per(key: str) -> float:
            return self.per_100.get(key, 0.0) * scale

        fga, fg3a, fta = per("fga"), per("fg3a"), per("fta")
        oreb, dreb = per("orb"), per("drb")
        return {
            "min": self.minutes,
            "pts": per("pts"),
            "oreb": oreb,
            "dreb": dreb,
            "reb": oreb + dreb,
            "ast": per("ast"),
            "stl": per("stl"),
            "blk": per("blk"),
            "tov": per("tov"),
            "pf": per("pf"),
            "fga": fga,
            "fgm": fga * self.rates.get("fg_pct", 0.0),
            "fg3a": fg3a,
            "fg3m": fg3a * self.rates.get("fg3_pct", 0.0),
            "fta": fta,
            "ftm": fta * self.rates.get("ft_pct", 0.0),
        }


class DarkoDailyProjection(DarkoModel):
    """One daily-tab row: the per-game box projection for ``day``; ``game_id`` is set when the team plays."""

    nba_id: int
    name: str
    day: date
    team: str | None = None
    opponent: str | None = None
    game_id: str | None = None
    """nba.com game id, zero-padded to 10 digits (``0042500402``); the sheet stores it as an integer."""
    available: bool = True
    minutes: float
    stats: dict[str, float]
    """``min``, ``pts``, ``oreb``, ``dreb``, ``reb``, ``ast``, ``stl``, ``blk``, ``fg3a``, ``fg3m``."""

    @property
    def has_game(self) -> bool:
        return self.game_id is not None


def _rows(payload: bytes, required: tuple[str, ...], label: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    fields = [name.strip() for name in reader.fieldnames or () if isinstance(name, str)]
    missing = [column for column in required if column not in fields]
    if missing:
        hint = " (not a CSV?)" if len(fields) <= 1 else ""
        raise SourceSchemaError(f"{label}: missing columns {missing}{hint}")
    return [
        {key.strip(): value if isinstance(value, str) else "" for key, value in row.items() if isinstance(key, str)}
        for row in reader
    ]


def _required_number(row: Mapping[str, str], column: str) -> float:
    number = _num(row.get(column))
    if number is None:
        raise ValueError(f"{column} is blank")
    return number


def _talent_row(row: Mapping[str, str]) -> DarkoProjection:
    nba_id = _id(row.get("nba_id"))
    name = _text(row.get("player_name"))
    if nba_id is None or name is None:
        raise ValueError("nba_id and player_name are required")
    career = _num(row.get("career_game_num"))
    return DarkoProjection(
        nba_id=nba_id,
        name=name,
        position=_text(row.get("position")),
        x_position=_text(row.get("x_position")),
        team_id=_id(row.get("tm_id")),
        team=_text(row.get("team_name")),
        available=_flag(row.get("available"), default=True),
        age=_num(row.get("age")),
        career_games=int(career) if career is not None else None,
        minutes=_required_number(row, "minutes"),
        pace=_required_number(row, "pace"),
        per_100={
            column.removesuffix("_100"): value
            for column in PER_100_COLUMNS
            if (value := _num(row.get(column))) is not None
        },
        rates={column: value for column in RATE_COLUMNS if (value := _num(row.get(column))) is not None},
        dpm={column: value for column in DPM_COLUMNS if (value := _num(row.get(column))) is not None},
    )


def _daily_row(row: Mapping[str, str]) -> DarkoDailyProjection:
    nba_id = _id(row.get("nba_id"))
    name = _text(row.get("player_name"))
    day = _text(row.get("date"))
    if nba_id is None or name is None or day is None:
        raise ValueError("nba_id, player_name and date are required")
    game = _id(row.get("game_id"))
    stats = {column: _required_number(row, column) for column in DAILY_STATS}
    minutes = _required_number(row, "minutes")
    return DarkoDailyProjection(
        nba_id=nba_id,
        name=name,
        day=date.fromisoformat(day[:10]),
        team=_text(row.get("team_name")),
        opponent=_text(row.get("opp_tm_name")),
        game_id=f"{game:010d}" if game is not None else None,
        available=_flag(row.get("available"), default=True),
        minutes=minutes,
        stats={
            "min": minutes,
            "pts": stats["pts"],
            "oreb": stats["orb"],
            "dreb": stats["drb"],
            "reb": stats["orb"] + stats["drb"],
            "ast": stats["ast"],
            "stl": stats["stl"],
            "blk": stats["blk"],
            "fg3a": stats["fg3a"],
            "fg3m": stats["fg3m"],
        },
    )


def _parse[T](
    payload: bytes, required: tuple[str, ...], build: Callable[[Mapping[str, str]], T], label: str
) -> list[T]:
    """Rows that fail validation are skipped and counted; a sheet with rows but no valid one is a schema error."""
    rows = _rows(payload, required, label)
    parsed: list[T] = []
    skipped = 0
    for row in rows:
        try:
            parsed.append(build(row))
        except (ValueError, ValidationError):
            skipped += 1
    if rows and not parsed:
        raise SourceSchemaError(f"{label}: none of {len(rows)} rows validated")
    if skipped:
        logger.warning("%s: skipped %d of %d rows that did not validate", label, skipped, len(rows))
    return parsed


def parse_talent(payload: bytes) -> list[DarkoProjection]:
    return _parse(payload, TALENT_REQUIRED, _talent_row, "darko talent")


def parse_daily(payload: bytes) -> list[DarkoDailyProjection]:
    return _parse(payload, DAILY_REQUIRED, _daily_row, "darko daily")


class DarkoSource(HttpSource):
    """DARKO's sheet tabs as typed rows; raw CSV captured under ``cache_dir()/sources/darko/``."""

    name: ClassVar[str] = "darko"
    min_interval: ClassVar[float] = 1.0
    base_headers: ClassVar[Mapping[str, str]] = {"Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8"}
    ttl: ClassVar[Mapping[str, timedelta]] = {
        "talent": timedelta(hours=6),  # DARKO updates once a day
        "daily": timedelta(hours=1),  # game-day lines move with lineup news
    }

    def __init__(
        self,
        *,
        talent_url: str = TALENT_URL,
        daily_url: str = DAILY_URL,
        client: httpx.Client | None = None,
        sleep: Callable[[float], object] = time.sleep,
        cache_root: Path | None = None,
        limiter: RateLimiter | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        super().__init__(client=client, sleep=sleep, cache_root=cache_root, limiter=limiter, clock=clock)
        self.talent_url = talent_url
        self.daily_url = daily_url

    def projections(self, **options: Unpack[FetchOptions]) -> Fetched[list[DarkoProjection]]:
        """Current per-100 talent with projected minutes and pace, one row per player."""
        return self.fetch(
            "talent",
            "current",
            download=lambda: self.get_bytes(self.talent_url),
            parse=parse_talent,
            ext="csv",
            meta={"url": self.talent_url},
            **options,
        )

    def daily(self, **options: Unpack[FetchOptions]) -> Fetched[list[DarkoDailyProjection]]:
        """Per-game box projections for the date DARKO last published."""
        return self.fetch(
            "daily",
            "current",
            download=lambda: self.get_bytes(self.daily_url),
            parse=parse_daily,
            ext="csv",
            meta={"url": self.daily_url},
            **options,
        )
