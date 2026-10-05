"""Projection sources, the per-stat blend, and projection uncertainty (DESIGN section 8.1).

Projections are stat lines, never points (CLAUDE.md): the store holds one :class:`fm.store.ProjectionRow` per
(player, scoring period, source, kind), keyed by ESPN stat abbreviation, and each league scores them with its own items
(:mod:`fm.model.scoring`). This module has four parts.

**Sources.** A projection source is a name per sport kept in a registry, the way decision modules register in
:mod:`fm.decide.registry`: the blend weights file names sources, :func:`blend_period` loads each one, and the in-house
baselines (ROADMAP #42, #43) add themselves with :func:`register_source` without touching a shared file. A source says
where its rows come from: ``stored`` sources are written to ``projections`` by another job (``espn``: the sync job
stores ESPN's lines), and the others have a ``loader`` that fetches them (``sleeper``: :class:`SleeperLoader` reads the
week's projections through the Sleeper adapter, whose cache the sync job refreshes, and joins them to ESPN ids through
the crosswalk the sync job saved). ``darko`` (NBA) is registered so the weights file can name it; its loader needs the
NBA crosswalk and stat mapping (ROADMAP #17, #24) and is attached there.

**Sleeper lines.** :func:`sleeper_line` maps Sleeper's keys onto ESPN abbreviations (:data:`SLEEPER_STATS`,
:data:`SLEEPER_DEFENSE_STATS` for team defenses) and :func:`sleeper_rows` keys them by ESPN id through
:class:`fm.model.ids.Crosswalk`, reporting the players it cannot map. Sleeper's projected first downs are left out
(:data:`SLEEPER_PROJECTION_EXCLUDED`): in the 2026 week-4 capture they run about twice ESPN's and, for quarterbacks,
above the projected completions, which no passing first-down count can be. Its actual first downs are sane and kept.

**Blend.** :func:`blend` averages the sources' lines stat by stat with weights per (sport, source, position) read from
``data/blend_weights.toml`` (:class:`BlendWeights`, resolved through ``fm.paths.data_file`` like ``id_overrides.csv``).
A stat is averaged over the sources that carry it, with their weights renormalised, so a stat only ESPN projects
(``P300``, the D/ST brackets, ``FTD``) keeps ESPN's value and a source that does not project a stat does not pull it to
zero. The cost: both sources also omit stats they project as zero, so when one omits a rare event (a receiver's
rushing touchdown) that the other projects small, the small value stands instead of being halved, an overstatement
of a tenth of a point or so. A line with no non-zero stat is different: ESPN projects an empty line for a player on
bye, a projection of zero, so it pulls every stat of the blend toward zero. Derived stats follow their inputs
(:func:`blend_line`): before averaging, each line gets the every-N counts, totals and scaled stats its own stats give
(:data:`LINE_DERIVED`), so a league that scores ``PY25`` or ``TT`` sees Sleeper's yards and turnovers as well as
ESPN's, and where both sources project a league's scored stats the blend scores as the weighted mean of their scores.
Brackets keep ESPN's probabilities, and percentages are recomputed from the blended makes and attempts. Blended rows
go under the reserved source name :data:`BLEND`, stamped with the oldest input's ``as_of``; :func:`blend_period`
writes them and deletes an earlier run's row for a player nothing projects any more.

**Uncertainty.** A projection's standard deviation in league points is ``max(floor, cv * points)`` with the
coefficient of variation by position from the same file (:class:`SdModel`, :meth:`BlendWeights.projection_sd`), the
starting point DESIGN asks for. :func:`fit_position_sd` estimates the coefficients from stored projections and actuals
(the backtest's residuals, ROADMAP #33), and the weight tuning (ROADMAP #39), which owns the file, writes them into it.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, Unpack

import numpy as np

from fm import paths
from fm.config import Sport
from fm.espn.ids import FFL, Game, IdMaps, ids_for
from fm.espn.settings import LeagueSettings
from fm.model.ids import SLEEPER, Crosswalk, is_team_defense
from fm.model.scoring import (
    DERIVATIONS,
    MAX_DERIVATION_DEPTH,
    Bracket,
    Ratio,
    Rule,
    Scorer,
    ScoringError,
    derive_stats,
    rule_inputs,
)
from fm.sources.base import Fetched, FetchOptions
from fm.sources.sleeper import SleeperSource, SleeperStatLine
from fm.store import SPORTS, ProjectionKind, ProjectionRow, Store, utc_now

ESPN: Final = "espn"
DARKO: Final = "darko"
BLEND: Final = "blend"
"""The source name of blended rows; never a registered source."""
DEFAULT_WEIGHTS = paths.data_file("blend_weights.toml")
SOURCE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
RESERVED_WEIGHT_KEYS: tuple[str, ...] = ("default", "sd")
"""Tables of a sport in the weights file that are not positions."""
STORED_DATASET: Final = "projections"
"""``Fetched.dataset`` of rows read back from the store."""
TEAM_DEFENSE_POSITION: Final = "D/ST"
"""The position of an ESPN D/ST player id (``-16000 - proTeamId``), used when the ``players`` table lacks the row."""
FFL.position_id(TEAM_DEFENSE_POSITION)  # KeyError at import if the id maps ever rename it


def _sport(value: Game | str) -> Sport:
    """``nfl`` / ``nba`` from a sport, an ESPN game key or a ``Game``. Raises ``ValueError``."""
    return "nfl" if Game.coerce(value) is Game.FFL else "nba"


def position_for(sport: Game | str, espn_id: int, positions: Mapping[int, str | None] | None = None) -> str | None:
    """A player's position label from ``positions`` (ESPN id to label, usually the ``players`` table), falling back
    to ``D/ST`` for an NFL team-defense id, so a defense missing from the table still gets its scoring overrides."""
    position = positions.get(espn_id) if positions else None
    if position is None and _sport(sport) == "nfl" and is_team_defense(espn_id):
        return TEAM_DEFENSE_POSITION
    return position


# --- the source registry ----------------------------------------------------------------------------------------------

type ProjectionLoader = Callable[[Store, int, int, FetchOptions], Fetched[tuple[ProjectionRow, ...]]]
"""``loader(store, season, scoring_period, options)``: a source's projected rows for the period, keyed by ESPN id and
ESPN stat abbreviation, with the fetch's provenance (``as_of``, ``stale``, ``degraded``, ``warnings``). ``options``
are the :class:`fm.sources.base.FetchOptions` the caller passes through."""


class DuplicateProjectionSourceError(ValueError):
    """``(sport, name)`` is already registered."""


class UnknownProjectionSourceError(LookupError):
    """Nothing is registered under ``(sport, name)``."""


@dataclass(frozen=True, slots=True)
class ProjectionSource:
    """One registered source: its name (the ``source`` column of ``projections`` and the key in the weights file), the
    sport it projects, and where its rows come from: ``stored`` (another job writes them to the store) or ``loader``.
    A source with neither is known to the weights file but cannot be loaded yet; pass its rows to :func:`blend`."""

    sport: Sport
    name: str
    label: str
    loader: ProjectionLoader | None = None
    stored: bool = False

    @property
    def key(self) -> tuple[Sport, str]:
        return (self.sport, self.name)

    @property
    def loadable(self) -> bool:
        return self.stored or self.loader is not None


class ProjectionSourceRegistry:
    """Projection sources keyed by ``(sport, name)``; iteration follows registration order."""

    def __init__(self) -> None:
        self._entries: dict[tuple[Sport, str], ProjectionSource] = {}

    def register(
        self,
        sport: Game | str,
        name: str,
        *,
        label: str | None = None,
        loader: ProjectionLoader | None = None,
        stored: bool = False,
    ) -> ProjectionSource:
        """Add a source. Raises :class:`DuplicateProjectionSourceError` when ``(sport, name)`` is taken, ``ValueError``
        for an unsupported sport, a malformed name, the reserved name ``blend``, or both ``loader`` and ``stored``, and
        ``TypeError`` for a loader that is not callable."""
        if not isinstance(name, str) or not SOURCE_PATTERN.match(name):
            raise ValueError(f"projection source name should be a lower-case identifier like 'sleeper', got {name!r}")
        if name == BLEND:
            raise ValueError(f"{BLEND!r} is the blended output, not a source")
        if loader is not None and not callable(loader):
            raise TypeError(f"projection source {name!r}: loader should be callable, got {loader!r}")
        if loader is not None and stored:
            raise ValueError(f"projection source {name!r} is either stored by another job or loaded, not both")
        source = ProjectionSource(sport=_sport(sport), name=name, label=label or name, loader=loader, stored=stored)
        if source.key in self._entries:
            raise DuplicateProjectionSourceError(f"projection source {source.sport}:{name} is registered twice")
        self._entries[source.key] = source
        return source

    def get(self, sport: Game | str, name: str) -> ProjectionSource | None:
        return self._entries.get((_sport(sport), name))

    def lookup(self, sport: Game | str, name: str) -> ProjectionSource:
        """The source for ``(sport, name)``. Raises :class:`UnknownProjectionSourceError` naming the alternatives."""
        source = self.get(sport, name)
        if source is None:
            normalized = _sport(sport)
            available = ", ".join(self.names(normalized)) or "nothing"
            raise UnknownProjectionSourceError(
                f"no {name!r} projection source is registered for {normalized}; registered: {available}"
            )
        return source

    def registered(self, sport: Game | str | None = None) -> tuple[ProjectionSource, ...]:
        """Every source, or those for one sport, in registration order."""
        if sport is None:
            return tuple(self._entries.values())
        normalized = _sport(sport)
        return tuple(source for source in self._entries.values() if source.sport == normalized)

    def names(self, sport: Game | str) -> tuple[str, ...]:
        return tuple(source.name for source in self.registered(sport))

    def unregister(self, sport: Game | str, name: str) -> ProjectionSource:
        """Remove and return a source (to re-register it with a loader, say). Raises
        :class:`UnknownProjectionSourceError` when there is none."""
        source = self.lookup(sport, name)
        del self._entries[source.key]
        return source

    def clear(self) -> None:
        self._entries.clear()

    def __contains__(self, key: object) -> bool:
        """``("nfl", "sleeper") in registry``; the sport may be a game key or ``Game``."""
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        sport, name = key
        if not isinstance(sport, Game | str) or not isinstance(name, str):
            return False
        try:
            return (_sport(sport), name) in self._entries
        except ValueError:
            return False

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ProjectionSource]:
        return iter(tuple(self._entries.values()))


source_registry = ProjectionSourceRegistry()
"""The process-wide registry; the v1 sources register below, at import."""


def register_source(
    sport: Game | str,
    name: str,
    *,
    label: str | None = None,
    loader: ProjectionLoader | None = None,
    stored: bool = False,
) -> ProjectionSource:
    """Register a projection source on the process-wide registry (the in-house baselines do this at import)."""
    return source_registry.register(sport, name, label=label, loader=loader, stored=stored)


def projection_source(sport: Game | str, name: str) -> ProjectionSource:
    return source_registry.lookup(sport, name)


def projection_sources(sport: Game | str | None = None) -> tuple[ProjectionSource, ...]:
    return source_registry.registered(sport)


# --- Sleeper lines ----------------------------------------------------------------------------------------------------

SLEEPER_DEFENSE: Final = "DEF"
"""Sleeper's position for a team defense, whose player id is the team code; its lines read the defensive keys."""
SLEEPER_REGULAR_SEASON: Final = "regular"
"""The only ``season_type`` that maps onto ESPN's NFL scoring periods (regular-season weeks)."""
SLEEPER_KINDS: Mapping[str, ProjectionKind] = MappingProxyType({"proj": "projected", "stat": "actual"})
"""``SleeperStatLine.category`` to the store's ``kind``."""

SLEEPER_STATS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "pass_att": ("PA",),
        "pass_cmp": ("PC",),
        "pass_inc": ("INC",),
        "pass_yd": ("PY",),
        "pass_td": ("PTD",),
        "pass_2pt": ("2PC",),
        "pass_int": ("INTT",),
        "pass_sack": ("SKD",),
        "pass_fd": ("PFD",),
        "pass_td_40p": ("PTD40",),
        "pass_td_50p": ("PTD50",),
        "rush_att": ("RA",),
        "rush_yd": ("RY",),
        "rush_td": ("RTD",),
        "rush_2pt": ("2PR",),
        "rush_fd": ("RFD",),
        "rush_td_40p": ("RTD40",),
        "rush_td_50p": ("RTD50",),
        "rec": ("REC",),
        "rec_yd": ("REY",),
        "rec_td": ("RETD",),
        "rec_2pt": ("2PRE",),
        "rec_tgt": ("RET",),
        "rec_fd": ("REFD",),
        "rec_td_40p": ("RETD40",),
        "rec_td_50p": ("RETD50",),
        "kr_yd": ("KR",),
        "pr_yd": ("PR",),
        "kr_td": ("KRTD",),
        "pr_td": ("PRTD",),
        "fum": ("FUM",),
        "fum_lost": ("FUML",),
        "fum_rec_td": ("FTD",),
        "gp": ("GP",),
        "fgm": ("FG",),
        "fga": ("FGA",),
        "fgmiss": ("FGM",),
        "fgm_yds": ("FGY",),
        "fgmiss_yds": ("FGMY",),
        "fgm_40_49": ("FG40",),
        "fga_40_49": ("FGA40",),
        "fgmiss_40_49": ("FGM40",),
        "fgm_50p": ("FG50P", "FG50"),
        "fga_50p": ("FGA50P", "FGA50"),
        "fgmiss_50p": ("FGM50P", "FGM50"),
        "xpm": ("PAT",),
        "xpa": ("PATA",),
        "xpmiss": ("PATM",),
    }
)
"""Sleeper stat key to the ESPN abbreviations it fills, for every player but a team defense. ESPN's own lines carry
``FG50P`` (50+) and ``FG50`` (50-59) with one value, so both get Sleeper's ``fgm_50p``. Sleeper's own points
(``pts_ppr`` and friends), ADP fields, bonus counters and play-length splits with no ESPN counterpart are ignored."""

SLEEPER_DEFENSE_STATS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "sack": ("SK",),
        "int": ("INT",),
        "fum_rec": ("FR",),
        "ff": ("FF",),
        "safe": ("SF",),
        "blk_kick": ("BLKK",),
        "def_td": ("DEFRETTD",),
        "pass_int_td": ("INTTD",),
        "def_fum_td": ("FRTD",),
        "def_kr_td": ("KRTD",),
        "def_pr_td": ("PRTD",),
        "def_kr_yd": ("KR",),
        "def_pr_yd": ("PR",),
        "pts_allow": ("PTSA",),
        "yds_allow": ("YA",),
        "tkl": ("TK",),
        "tkl_solo": ("TKS",),
        "tkl_ast": ("TKA",),
        "tkl_loss": ("STF",),
        "pass_def": ("PD",),
        "gp": ("GP",),
    }
)
"""The keys of a ``DEF`` line, which reads nothing else: on a quarterback's line ``pass_int_td`` is a pick-six thrown,
not a return, and a defense's ``pr_yd`` is not the ``def_pr_yd`` ESPN's ``PR`` matches. ``def_td`` is the sum of the
interception and fumble return touchdowns, as ESPN's ``DEFRETTD`` is. ``DPTSA`` and the ``DPA`` brackets come from
``PTSA`` in the scorer, the way ESPN's projections carry them."""

SLEEPER_SUMS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "FG0": ("fgm_0_19", "fgm_20_29", "fgm_30_39"),
        "FGA0": ("fga_0_19", "fga_20_29", "fga_30_39"),
        "FGM0": ("fgmiss_0_19", "fgmiss_20_29", "fgmiss_30_39"),
    }
)
"""ESPN's 0-39 yard kicking stats are the sum of Sleeper's finer distance splits."""

SLEEPER_PROJECTION_EXCLUDED: frozenset[str] = frozenset({"pass_fd", "rush_fd", "rec_fd"})
"""Keys dropped from projected lines (not actuals). In the 2026 week-4 capture Sleeper projected Josh Allen for 22.6
passing first downs on 19.1 completions, Jauan Jennings for 4.1 receiving first downs on 3.4 catches, and its running
backs for first downs on about half their carries, twice ESPN's rate. Its actual first downs (8 on 22 completions)
are sane, so only the projections are left out."""


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def sleeper_line(
    stats: Mapping[str, float], *, position: str | None = None, projected: bool = False
) -> dict[str, float]:
    """A Sleeper stat dict as an ESPN-abbreviation stat line. ``position`` ``DEF`` reads the defensive keys only;
    ``projected`` drops :data:`SLEEPER_PROJECTION_EXCLUDED`. Non-numeric and non-finite values are skipped."""
    excluded = SLEEPER_PROJECTION_EXCLUDED if projected else frozenset[str]()
    defense = position == SLEEPER_DEFENSE
    line: dict[str, float] = {}
    for key, abbreviations in (SLEEPER_DEFENSE_STATS if defense else SLEEPER_STATS).items():
        value = _number(stats.get(key)) if key not in excluded else None
        if value is None:
            continue
        for abbreviation in abbreviations:
            line[abbreviation] = value
    if not defense:
        for abbreviation, parts in SLEEPER_SUMS.items():
            present = [value for key in parts if (value := _number(stats.get(key))) is not None]
            if present and abbreviation not in line:
                line[abbreviation] = math.fsum(present)
    return line


@dataclass(frozen=True, slots=True)
class Converted:
    """Rows from a source's lines plus what could not be converted: ``unmapped`` are source ids with no ESPN id (the
    crosswalk's job to fix, through ``data/id_overrides.csv``); ``warnings`` summarise every skip for the output."""

    rows: tuple[ProjectionRow, ...]
    unmapped: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def sleeper_rows(lines: Iterable[SleeperStatLine], crosswalk: Crosswalk, *, as_of: datetime) -> Converted:
    """Sleeper projections (or week stats) as ``sleeper`` rows keyed by ESPN id and ESPN abbreviation.

    Season and week come from each line; ``category`` decides ``kind`` (``proj`` is projected, ``stat`` actual).
    Placeholders (ADP only) are skipped silently. Skipped with a warning: lines outside the regular season (their weeks
    are not ESPN scoring periods), unknown categories, and players the crosswalk cannot map (also listed in
    ``unmapped``). When two lines describe one player-week the later one wins. ``as_of`` is the fetch's.
    """
    rows: dict[tuple[int, ProjectionKind, int, int], ProjectionRow] = {}
    unmapped: list[str] = []
    off_season: list[str] = []
    categories: dict[str, None] = {}
    considered = 0
    for line in lines:
        if line.is_placeholder:
            continue
        if line.season_type != SLEEPER_REGULAR_SEASON:
            off_season.append(line.season_type)
            continue
        kind = SLEEPER_KINDS.get(line.category)
        if kind is None:
            categories[line.category] = None
            continue
        considered += 1
        espn_id = crosswalk.espn_id(SLEEPER, line.player_id)
        if espn_id is None:
            unmapped.append(line.player_id)
            continue
        position = line.position or (SLEEPER_DEFENSE if line.player_id.isalpha() else None)
        rows[(espn_id, kind, line.season, line.week)] = ProjectionRow(
            sport="nfl",
            espn_id=espn_id,
            source=SLEEPER,
            kind=kind,
            season=line.season,
            scoring_period_id=line.week,
            stats=sleeper_line(line.stats, position=position, projected=kind == "projected"),
            as_of=as_of,
        )
    warnings: list[str] = []
    if unmapped:
        shown = ", ".join(unmapped[:5]) + (f" and {len(unmapped) - 5} more" if len(unmapped) > 5 else "")
        warnings.append(f"sleeper: {len(unmapped)} of {considered} lines belong to players with no ESPN id ({shown})")
    if off_season:
        kinds = ", ".join(sorted(set(off_season)))
        warnings.append(f"sleeper: skipped {len(off_season)} lines outside the regular season ({kinds})")
    if categories:
        warnings.append(f"sleeper: skipped lines of unknown category {', '.join(sorted(categories))}")
    return Converted(tuple(rows.values()), tuple(unmapped), tuple(warnings))


class SleeperLoader:
    """The ``sleeper`` source's loader: the week's projections through the adapter, joined to ESPN ids.

    ``source`` defaults to a fresh :class:`fm.sources.sleeper.SleeperSource` over the cache dir (closed after each
    call), so the cached copy the sync job refreshed is served while it is fresh; ``crosswalk`` defaults to the one
    saved in the store. The adapter degrades instead of raising when Sleeper breaks, so the result may be ``stale`` or
    an empty ``degraded`` one, and the blend then runs on ESPN alone.
    """

    def __init__(self, source: SleeperSource | None = None, *, crosswalk: Crosswalk | None = None) -> None:
        self.source = source
        self.crosswalk = crosswalk

    def __call__(
        self, store: Store, season: int, scoring_period: int, options: FetchOptions
    ) -> Fetched[tuple[ProjectionRow, ...]]:
        adapter = self.source if self.source is not None else SleeperSource()
        try:
            fetched = adapter.projections(season, scoring_period, **options)
        finally:
            if self.source is None:
                adapter.close()
        walk = self.crosswalk if self.crosswalk is not None else Crosswalk.from_store(store)
        converted = sleeper_rows(fetched.data, walk, as_of=fetched.as_of)
        return Fetched(
            converted.rows,
            fetched.as_of,
            fetched.source,
            fetched.dataset,
            fetched.key,
            cached=fetched.cached,
            stale=fetched.stale,
            degraded=fetched.degraded,
            warnings=(*fetched.warnings, *converted.warnings),
            raw_path=fetched.raw_path,
        )


source_registry.register("nfl", ESPN, label="ESPN projections (stored by fm sync)", stored=True)
source_registry.register("nfl", SLEEPER, label="Sleeper (RotoWire) projections", loader=SleeperLoader())
source_registry.register("nba", ESPN, label="ESPN projections (stored by fm sync)", stored=True)
source_registry.register("nba", DARKO, label="DARKO projections (loader arrives with the NBA crosswalk)")


# --- blend weights and the uncertainty model --------------------------------------------------------------------------

DEFAULT_SD_FLOOR: Final = 2.0
DEFAULT_SD_CV: Final = 0.55
MIN_SD_SAMPLES: Final = 20
"""Player-periods a position needs before :func:`fit_position_sd` reports an estimate for it."""


class BlendWeightsError(ValueError):
    """``blend_weights.toml`` cannot be used as given; the message names the file and the table."""


@dataclass(frozen=True, slots=True)
class SdEstimate:
    """Residual statistics for one position from :func:`fit_position_sd`: ``sd`` in league points over ``samples``
    player-periods, and ``cv`` (``sd / mean_projected``) in the form the weights file stores."""

    position: str
    samples: int
    sd: float
    cv: float
    mean_projected: float
    mean_actual: float


@dataclass(frozen=True, slots=True)
class SdModel:
    """Projection uncertainty by position: ``sd = max(floor, cv(position) * |points|)``; ``default`` is the
    coefficient of variation for a position without its own."""

    floor: float = DEFAULT_SD_FLOOR
    default: float = DEFAULT_SD_CV
    by_position: Mapping[str, float] = MappingProxyType({})

    def cv(self, position: str | None = None) -> float:
        if position is None:
            return self.default
        return self.by_position.get(position, self.default)

    def sd(self, points: float, position: str | None = None) -> float:
        return max(self.floor, self.cv(position) * abs(points))

    def updated(self, estimates: Mapping[str, SdEstimate]) -> SdModel:
        """This model with the positions in ``estimates`` replaced by their fitted coefficients of variation."""
        merged = {**self.by_position, **{position: estimate.cv for position, estimate in estimates.items()}}
        return SdModel(floor=self.floor, default=self.default, by_position=MappingProxyType(merged))


def _non_negative(value: object, at: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BlendWeightsError(f"{at}: expected a number, got {value!r}")
    if not math.isfinite(value) or value < 0:
        raise BlendWeightsError(f"{at}: expected a finite number >= 0, got {value!r}")
    return float(value)


def _position_label(ids: IdMaps, label: str, at: str) -> str:
    try:
        ids.position_id(label)
    except KeyError:
        known = ", ".join(ids.positions.values())
        raise BlendWeightsError(f"{at}: {label!r} is not a {ids.game.sport} position (one of {known})") from None
    return label


def _parse_weight_table(raw: object, sport: Sport, sources: ProjectionSourceRegistry, at: str) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise BlendWeightsError(f"{at}: expected a table of source = weight entries, got {raw!r}")
    table: dict[str, float] = {}
    for name, value in raw.items():
        if sources.get(sport, str(name)) is None:
            known = ", ".join(sources.names(sport)) or "nothing"
            raise BlendWeightsError(
                f"{at}: {name!r} is not a registered {sport} projection source (registered: {known}); "
                "import the module that registers it before loading the weights"
            )
        table[str(name)] = _non_negative(value, f"{at}.{name}")
    return table


def _parse_sd_table(raw: object, ids: IdMaps, at: str) -> SdModel:
    if not isinstance(raw, Mapping):
        raise BlendWeightsError(f"{at}: expected a table, got {raw!r}")
    floor = DEFAULT_SD_FLOOR
    default = DEFAULT_SD_CV
    by_position: dict[str, float] = {}
    for key, value in raw.items():
        label = str(key)
        if label == "floor":
            floor = _non_negative(value, f"{at}.floor")
        elif label == "default":
            default = _non_negative(value, f"{at}.default")
        else:
            by_position[_position_label(ids, label, at)] = _non_negative(value, f"{at}.{label}")
    return SdModel(floor=floor, default=default, by_position=MappingProxyType(by_position))


@dataclass(frozen=True)
class BlendWeights:
    """Per-(sport, source, position) blend weights and the uncertainty model, as read from ``blend_weights.toml``.

    ``weights(sport, position)`` merges the sport's ``default`` table with the position's overrides and drops sources
    weighted 0; it is what :func:`blend` divides by. Build one with :meth:`load` (the committed file unless a path is
    given), :meth:`parse` (TOML text) or :meth:`uniform` (equal weights for every registered source: the DESIGN
    starting point, and the baseline a tuned file must beat).
    """

    tables: Mapping[Sport, Mapping[str, Mapping[str, float]]]
    """sport -> ``default`` or position label -> source -> weight."""
    sd_models: Mapping[Sport, SdModel]
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None, *, sources: ProjectionSourceRegistry = source_registry) -> BlendWeights:
        """Read the weights file (``DEFAULT_WEIGHTS`` unless given). A missing file is an error, never silence."""
        target = DEFAULT_WEIGHTS if path is None else path
        if not target.is_file():
            raise BlendWeightsError(f"blend weights file not found: {target}")
        parsed = cls.parse(target.read_text(encoding="utf-8"), where=str(target), sources=sources)
        return cls(parsed.tables, parsed.sd_models, path=target)

    @classmethod
    def parse(
        cls, text: str, *, where: str = "blend_weights.toml", sources: ProjectionSourceRegistry = source_registry
    ) -> BlendWeights:
        """Parse TOML text. Raises :class:`BlendWeightsError` naming ``where`` and the table for an unknown sport,
        position or source, a weight that is not a finite number >= 0, a sport without a ``default`` table, or a
        ``default`` table that weights every source 0."""
        try:
            raw = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise BlendWeightsError(f"{where}: {exc}") from exc
        tables: dict[Sport, Mapping[str, Mapping[str, float]]] = {}
        sd_models: dict[Sport, SdModel] = {}
        for sport_key, sport_table in raw.items():
            if sport_key not in SPORTS:
                raise BlendWeightsError(f"{where}: [{sport_key}] is not a sport (expected {' or '.join(SPORTS)})")
            sport: Sport = "nfl" if sport_key == "nfl" else "nba"
            if not isinstance(sport_table, Mapping):
                raise BlendWeightsError(f"{where}: [{sport}] should be a table, got {sport_table!r}")
            ids = ids_for(sport)
            by_position: dict[str, Mapping[str, float]] = {}
            sd_model = SdModel()
            for key, value in sport_table.items():
                label = str(key)
                at = f"{where}: [{sport}.{label}]"
                if label == "sd":
                    sd_model = _parse_sd_table(value, ids, at)
                    continue
                if label != "default":
                    _position_label(ids, label, at)
                by_position[label] = MappingProxyType(_parse_weight_table(value, sport, sources, at))
            default = by_position.get("default")
            if default is None:
                raise BlendWeightsError(f"{where}: [{sport}] has no [{sport}.default] table")
            if not any(weight > 0 for weight in default.values()):
                raise BlendWeightsError(f"{where}: [{sport}.default] weights every source 0, so nothing would blend")
            tables[sport] = MappingProxyType(by_position)
            sd_models[sport] = sd_model
        return cls(MappingProxyType(tables), MappingProxyType(sd_models))

    @classmethod
    def uniform(cls, *, sources: ProjectionSourceRegistry = source_registry) -> BlendWeights:
        """Equal weights for every registered source of each sport that has one, with the default uncertainty model."""
        tables: dict[Sport, Mapping[str, Mapping[str, float]]] = {}
        for sport in SPORTS:
            names = sources.names(sport)
            if names:
                tables[sport] = MappingProxyType({"default": MappingProxyType(dict.fromkeys(names, 1.0))})
        return cls(MappingProxyType(tables), MappingProxyType({sport: SdModel() for sport in tables}))

    @property
    def sports(self) -> tuple[Sport, ...]:
        return tuple(self.tables)

    def positions(self, sport: Game | str) -> tuple[str, ...]:
        """Positions with their own override table."""
        return tuple(key for key in self._table(sport) if key not in RESERVED_WEIGHT_KEYS)

    def sources(self, sport: Game | str) -> tuple[str, ...]:
        """Every source the sport's tables name, weighted or not."""
        names: dict[str, None] = {}
        for table in self._table(sport).values():
            names.update(dict.fromkeys(table))
        return tuple(names)

    def weights(self, sport: Game | str, position: str | None = None) -> dict[str, float]:
        """Source -> weight for a position (the default table with the position's overrides); zero weights are
        dropped. Raises :class:`BlendWeightsError` for a sport the file has no table for."""
        table = self._table(sport)
        merged = dict(table["default"])
        if position is not None and position in table and position not in RESERVED_WEIGHT_KEYS:
            merged.update(table[position])
        return {source: weight for source, weight in merged.items() if weight > 0}

    def sd_model(self, sport: Game | str) -> SdModel:
        return self.sd_models.get(_sport(sport), SdModel())

    def projection_sd(self, points: float, *, sport: Game | str, position: str | None = None) -> float:
        """Standard deviation in league points of a projection worth ``points`` at ``position``."""
        return self.sd_model(sport).sd(points, position)

    def _table(self, sport: Game | str) -> Mapping[str, Mapping[str, float]]:
        normalized = _sport(sport)
        table = self.tables.get(normalized)
        if table is None:
            raise BlendWeightsError(f"no blend weights for {normalized}" + (f" in {self.path}" if self.path else ""))
        return table


# --- the blend --------------------------------------------------------------------------------------------------------


def _reads_a_bracket_or_ratio(rules: Mapping[str, Rule], stat: str, depth: int = 0) -> bool:
    rule = rules.get(stat)
    if rule is None or depth >= MAX_DERIVATION_DEPTH:
        return False
    if isinstance(rule, Bracket | Ratio):
        return True
    return any(_reads_a_bracket_or_ratio(rules, name, depth + 1) for name in rule_inputs(rule))


LINE_DERIVED: Mapping[Game, tuple[str, ...]] = MappingProxyType(
    {
        game: tuple(stat for stat in rules if not _reads_a_bracket_or_ratio(rules, stat))
        for game, rules in DERIVATIONS.items()
    }
)
"""The derived stats each line gets from its own stats before a blend of two or more (every-N counts, totals, scaled
and difference stats), so each blends as the weighted mean of every source's own value. Brackets are left out, and so
is what reads one (``DPA14`` is ``PA14``): ESPN projects them as probabilities, which another source's 0/1 indicator
on its mean yards would drag toward 0 or 1. Percentages are left out too (:data:`RECOMPUTED_RATIOS`)."""

RECOMPUTED_RATIOS: Mapping[Game, tuple[str, ...]] = MappingProxyType(
    {
        game: tuple(stat for stat, rule in rules.items() if isinstance(rule, Ratio))
        for game, rules in DERIVATIONS.items()
    }
)
"""Percentages, which a blend recomputes from its blended makes and attempts rather than averaging them."""


def _is_zero_line(line: Mapping[str, float]) -> bool:
    """True for a stat line with no non-zero stat: a projection of zero (ESPN's empty line for a player on bye)."""
    return not any(value != 0 for value in line.values())


def _completed(line: Mapping[str, float], game: Game, source: str) -> dict[str, float]:
    try:
        return derive_stats(line, game, LINE_DERIVED[game])
    except ScoringError as exc:
        raise ScoringError(f"{source} line: {exc}") from exc


def blend_line(
    lines: Mapping[str, Mapping[str, float]], weights: Mapping[str, float], *, sport: Game | str
) -> dict[str, float]:
    """One stat line from several sources' lines: each stat is the weighted mean over the sources that carry it.

    ``lines`` is source -> stat line and ``weights`` source -> weight; a source weighted 0 or absent from ``weights``
    is ignored, and a single weighted line is the blend as it is. With two or more:

    - a line with no non-zero stat (ESPN's empty line for a player on bye) is a projection of zero, not a missing
      one: it counts as 0 for every stat of the blend;
    - every other line first gets the derived stats it lacks that its own stats give (:data:`LINE_DERIVED`, through
      :func:`fm.model.scoring.derive_stats`). So ``PY25`` is the mean of each source's ``floor(yards / 25)``, the form
      ESPN's projections carry, and not ESPN's count beside averaged yards; and ``TT`` is the mean of each source's
      turnovers. A stat one source carries and no other derives (a bracket probability, ``PFD``) keeps that source's
      value exactly;
    - percentages (:data:`RECOMPUTED_RATIOS`) are then recomputed from the blended makes and attempts, so they stay
      volume-weighted.

    Stats keep the order they are first seen. Raises :class:`fm.model.scoring.ScoringError` for a non-numeric or
    non-finite value in a line that is completed.
    """
    used = {source: line for source, line in lines.items() if weights.get(source, 0.0) > 0}
    if len(used) == 1:
        (line,) = used.values()
        return {stat: float(value) for stat, value in line.items()}
    game = Game.coerce(sport)
    zero = {source for source, line in used.items() if _is_zero_line(line)}
    completed = {
        source: dict.fromkeys(line, 0.0) if source in zero else _completed(line, game, source)
        for source, line in used.items()
    }
    order = dict.fromkeys(stat for line in completed.values() for stat in line)
    totals = dict.fromkeys(order, 0.0)
    mass = dict.fromkeys(order, 0.0)
    lone: dict[str, float | None] = {}
    for source, line in completed.items():
        weight = weights[source]
        for stat, value in (dict.fromkeys(order, 0.0) if source in zero else line).items():
            totals[stat] += weight * value
            mass[stat] += weight
            lone[stat] = value if stat not in lone else None
    blended = {stat: value if (value := lone[stat]) is not None else totals[stat] / mass[stat] for stat in order}
    ratios = tuple(stat for stat in RECOMPUTED_RATIOS[game] if stat in blended)
    if ratios:
        parts = {stat: value for stat, value in blended.items() if stat not in ratios}
        recomputed = derive_stats(parts, game, ratios)
        blended.update((stat, recomputed[stat]) for stat in ratios if stat in recomputed)
    return blended


@dataclass(frozen=True, slots=True)
class Blend:
    """Blended rows plus provenance: ``inputs`` maps ``(espn_id, season, scoring_period_id)`` to the sources that fed
    each row; ``warnings`` note rows that could not be blended and why."""

    rows: tuple[ProjectionRow, ...]
    inputs: Mapping[tuple[int, int, int], tuple[str, ...]]
    warnings: tuple[str, ...] = ()

    def sources_for(self, espn_id: int, season: int, scoring_period_id: int) -> tuple[str, ...]:
        return self.inputs.get((espn_id, season, scoring_period_id), ())


def blend(
    rows: Iterable[ProjectionRow],
    *,
    weights: BlendWeights,
    positions: Mapping[int, str | None] | None = None,
) -> Blend:
    """Blend projected rows of one sport into one ``blend`` row per (player, season, scoring period).

    ``positions`` (ESPN id -> position label) selects each player's weights; an unknown position gets the sport's
    default table (an NFL defense id falls back to ``D/ST``, see :func:`position_for`). Each player's stats are
    :func:`blend_line` of his weighted sources' lines: derived stats follow every source's own stats, and an empty
    line is a projection of zero that counts in the blend (and in ``inputs``). Rows already under :data:`BLEND` are
    ignored, and so are actuals (an actual is not blended). Rows of a source the weights file does not name are dropped
    with a warning; a source weighted 0 for the position is simply skipped, and a player left with no weighted source
    gets no row (also warned). Each row's ``as_of`` is the oldest of its inputs'. Two rows of one source for one
    player-period: the later wins. Raises ``ValueError`` for rows of two sports and
    :class:`fm.model.scoring.ScoringError` (naming the player and source) for a non-finite stat in a line it completes.
    """
    groups: dict[tuple[int, int, int], dict[str, ProjectionRow]] = {}
    sport: Sport | None = None
    actuals = 0
    for row in rows:
        if row.source == BLEND:
            continue
        if row.kind != "projected":
            actuals += 1
            continue
        if sport is None:
            sport = row.sport
        elif row.sport != sport:
            raise ValueError(f"blend one sport at a time: got {sport} and {row.sport} rows")
        groups.setdefault((row.espn_id, row.season, row.scoring_period_id), {})[row.source] = row
    warnings: list[str] = []
    if actuals:
        warnings.append(f"{actuals} rows were actuals, not projections; they were not blended")
    if sport is None:
        return Blend((), MappingProxyType({}), tuple(warnings))

    named = set(weights.sources(sport))
    unnamed = sorted({source for group in groups.values() for source in group} - named)
    blended: list[ProjectionRow] = []
    inputs: dict[tuple[int, int, int], tuple[str, ...]] = {}
    unweighted = 0
    for (espn_id, season, period), by_source in sorted(groups.items()):
        table = weights.weights(sport, position_for(sport, espn_id, positions))
        used = {source: row for source, row in by_source.items() if source in table}
        if not used:
            unweighted += any(source in named for source in by_source)
            continue
        try:
            stats = blend_line({source: row.stats for source, row in used.items()}, table, sport=sport)
        except ScoringError as exc:
            raise ScoringError(f"{sport}: ESPN {espn_id}, {season} period {period}: {exc}") from exc
        blended.append(
            ProjectionRow(
                sport=sport,
                espn_id=espn_id,
                source=BLEND,
                kind="projected",
                season=season,
                scoring_period_id=period,
                stats=stats,
                as_of=min(row.as_of for row in used.values()),
            )
        )
        inputs[(espn_id, season, period)] = tuple(sorted(used))
    warnings.extend(
        f"{sport}: projection source {source!r} has no entry in the blend weights; its rows were not blended"
        for source in unnamed
    )
    if unweighted:
        warnings.append(
            f"{sport}: {unweighted} players had rows only from sources weighted 0 at their position; no blended row"
        )
    return Blend(tuple(blended), MappingProxyType(inputs), tuple(warnings))


@dataclass(frozen=True, slots=True)
class PeriodBlend:
    """What :func:`blend_period` did: the :class:`Blend`, each source's load with its provenance (``as_of``,
    ``stale``, ``degraded``) for the output, how many rows it wrote, every warning in one place, and ``retired``: the
    ESPN ids whose blended row from an earlier run it deleted, since nothing projects them now."""

    blend: Blend
    loads: tuple[Fetched[tuple[ProjectionRow, ...]], ...]
    saved: int
    warnings: tuple[str, ...] = ()
    retired: tuple[int, ...] = ()

    @property
    def rows(self) -> tuple[ProjectionRow, ...]:
        return self.blend.rows

    def load(self, source: str) -> Fetched[tuple[ProjectionRow, ...]] | None:
        return next((fetched for fetched in self.loads if fetched.source == source), None)


def load_stored(
    store: Store, sport: Game | str, source: str, season: int, scoring_period: int
) -> Fetched[tuple[ProjectionRow, ...]]:
    """A stored source's projected rows for the period, as a :class:`fm.sources.base.Fetched` stamped with the
    oldest row's ``as_of``; with none stored it is an empty ``degraded`` result naming the fix."""
    normalized = _sport(sport)
    rows = tuple(store.projections.for_period(normalized, season, scoring_period, source=source))
    key = f"{normalized}_{season}_{scoring_period}"
    if not rows:
        missing = f"{source}: no {normalized} projections stored for {season} period {scoring_period}; run fm sync"
        return Fetched(rows, utc_now(), source, STORED_DATASET, key, cached=True, degraded=True, warnings=(missing,))
    return Fetched(rows, min(row.as_of for row in rows), source, STORED_DATASET, key, cached=True)


def blend_period(
    store: Store,
    sport: Game | str,
    season: int,
    scoring_period: int,
    *,
    weights: BlendWeights,
    sources: ProjectionSourceRegistry = source_registry,
    save: bool = True,
    **options: Unpack[FetchOptions],
) -> PeriodBlend:
    """Load every registered source of the sport for the period, blend them, and save.

    Stored sources are read from ``projections``; loaded ones are fetched (``options`` pass through to their adapters)
    and, with ``save``, written there under their own name so the inputs of every blend stay replayable. A source the
    weights file weights but nothing can load is warned about, and so are rows a loader returns for another sport or
    period. Positions come from the ``players`` table. The blend is written under :data:`BLEND` in the same
    transaction, replacing each player's previous blended row.

    A stored blended row this run does not replace belongs to a player no weighted source projects now: a loader
    failed or dropped him, the weights changed, or the crosswalk moved his id. It is deleted in the same transaction
    rather than left for a reader of ``projections`` to take as current (an empty line would read as a projection of
    zero); ``retired`` lists those players and a warning counts them.
    """
    normalized = _sport(sport)
    weighted = set(weights.sources(normalized))
    loads: list[Fetched[tuple[ProjectionRow, ...]]] = []
    notes: list[str] = []
    rows: list[ProjectionRow] = []
    fetched_rows: list[ProjectionRow] = []
    for source in sources.registered(normalized):
        if source.stored:
            fetched = load_stored(store, normalized, source.name, season, scoring_period)
        elif source.loader is not None:
            fetched = source.loader(store, season, scoring_period, options)
        else:
            if source.name in weighted:
                notes.append(
                    f"{normalized}: projection source {source.name!r} is weighted but has no loader; "
                    "its rows were not blended"
                )
            continue
        target = (normalized, season, scoring_period, "projected")
        wanted = [row for row in fetched.data if (row.sport, row.season, row.scoring_period_id, row.kind) == target]
        if len(wanted) != len(fetched.data):
            notes.append(f"{source.name}: dropped {len(fetched.data) - len(wanted)} rows for another sport or period")
        loads.append(fetched)
        rows.extend(wanted)
        if not source.stored:
            fetched_rows.extend(wanted)
    players = store.players.many(normalized, {row.espn_id for row in rows})
    result = blend(rows, weights=weights, positions={player.espn_id: player.position for player in players})
    saved = 0
    retired: list[int] = []
    if save:
        current = {row.espn_id for row in result.rows}
        with store.db.transaction():
            retired = [
                row.espn_id
                for row in store.projections.for_period(normalized, season, scoring_period, source=BLEND)
                if row.espn_id not in current
            ]
            store.projections.delete(normalized, season, scoring_period, BLEND, retired)
            saved = store.projections.upsert_many([*fetched_rows, *result.rows])
    warnings = [*(warning for fetched in loads for warning in fetched.warnings), *notes, *result.warnings]
    if retired:
        warnings.append(
            f"{normalized}: {len(retired)} players with a blended row from an earlier run have no weighted source "
            "now; their blended rows were deleted"
        )
    return PeriodBlend(result, tuple(loads), saved, tuple(warnings), retired=tuple(retired))


# --- projections with points and uncertainty --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Projection:
    """A stat line scored for one league: ``points`` under its scoring items and ``sd`` from the uncertainty model.
    Expected value is ``p_active * points`` (:func:`fm.model.availability.expected_points`). In a category league the
    items carry no points, so use ``stats`` (or :meth:`fm.model.scoring.Scorer.categories`)."""

    espn_id: int
    season: int
    scoring_period_id: int
    position: str | None
    source: str
    kind: ProjectionKind
    stats: Mapping[str, float]
    points: float
    sd: float
    as_of: datetime


def project(
    rows: Iterable[ProjectionRow],
    settings: LeagueSettings,
    *,
    weights: BlendWeights,
    positions: Mapping[int, str | None] | None = None,
    scorer: Scorer | None = None,
) -> list[Projection]:
    """Score projection rows (blended or single-source) for a league, with a standard deviation each, in row order.

    ``positions`` (ESPN id -> label) picks each player's ``pointsOverrides`` and uncertainty; pass them, since a D/ST
    or a TE-premium league scores differently by position. Raises ``ValueError`` for a row of another sport.
    """
    scoring = scorer if scorer is not None else Scorer(settings)
    sport = _sport(settings.game)
    projections: list[Projection] = []
    for row in rows:
        if row.sport != sport:
            raise ValueError(f"ESPN {row.espn_id}'s projection is {row.sport}; league {settings.league_id} is {sport}")
        position = position_for(sport, row.espn_id, positions)
        earned = scoring.points(row.stats, position=position)
        projections.append(
            Projection(
                espn_id=row.espn_id,
                season=row.season,
                scoring_period_id=row.scoring_period_id,
                position=position,
                source=row.source,
                kind=row.kind,
                stats=MappingProxyType(dict(row.stats)),
                points=earned,
                sd=weights.projection_sd(earned, sport=sport, position=position),
                as_of=row.as_of,
            )
        )
    return projections


def fit_position_sd(
    projected: Iterable[ProjectionRow],
    actual: Iterable[ProjectionRow],
    settings: LeagueSettings,
    *,
    positions: Mapping[int, str | None],
    min_samples: int = MIN_SD_SAMPLES,
    scorer: Scorer | None = None,
) -> dict[str, SdEstimate]:
    """Residual standard deviation by position, in the league's points, from projected and actual rows.

    Rows pair on (ESPN id, season, scoring period); a pair needs a known position and a non-zero projection (an empty
    projected line is a bye, not a miss). Positions with fewer than ``min_samples`` pairs (at least 2) are left out.
    The result feeds :meth:`SdModel.updated`.
    """
    scoring = scorer if scorer is not None else Scorer(settings)
    sport = _sport(settings.game)
    actuals = {
        (row.espn_id, row.season, row.scoring_period_id): row
        for row in actual
        if row.kind == "actual" and row.source != BLEND
    }
    pairs: dict[str, list[tuple[float, float]]] = {}
    for row in projected:
        if row.kind != "projected":
            continue
        observed = actuals.get((row.espn_id, row.season, row.scoring_period_id))
        position = position_for(sport, row.espn_id, positions)
        if observed is None or position is None:
            continue
        expected = scoring.points(row.stats, position=position)
        if expected == 0.0:
            continue
        pairs.setdefault(position, []).append((expected, scoring.points(observed.stats, position=position)))
    estimates: dict[str, SdEstimate] = {}
    for position, samples in sorted(pairs.items()):
        if len(samples) < max(2, min_samples):
            continue
        matrix = np.asarray(samples, dtype=float)
        expected_points, observed_points = matrix[:, 0], matrix[:, 1]
        sd = float(np.std(observed_points - expected_points, ddof=1))
        mean_projected = float(expected_points.mean())
        estimates[position] = SdEstimate(
            position=position,
            samples=len(samples),
            sd=sd,
            cv=sd / abs(mean_projected) if mean_projected else math.inf,
            mean_projected=mean_projected,
            mean_actual=float(observed_points.mean()),
        )
    return estimates
