"""NFL player ID crosswalk: ESPN ids <-> GSIS and Sleeper ids (DESIGN section 7, "Player identity").

Every NFL player has one canonical row keyed by ESPN id; the other sources key by their own ids (nflverse stats and
injuries by GSIS id, Sleeper projections and trending by Sleeper id). The crosswalk is built in three layers:

1. ``ff_playerids``, the DynastyProcess ID map shipped with nflverse (``NflverseSource.ff_playerids``): every row with
   an ``espn_id`` contributes its ``gsis_id`` and ``sleeper_id``. Sleeper's own ``espn_id`` field is too sparse to use
   (none for rookies).
2. ``data/id_overrides.csv``, a reviewed file for the gaps and mistakes: ``espn_id,source,source_id,name,note``, one
   mapping per line, ``#`` comments allowed. A row replaces whatever ff_playerids said for that player *and* for that
   source id; an empty ``source_id`` removes a wrong mapping.
3. Team defenses, which no ID map carries. ESPN's D/ST player ids are ``-16000 - proTeamId`` and Sleeper's are the team
   code, so those mappings are derived from :data:`fm.espn.ids.FFL_PRO_TEAMS`. GSIS has no team ids, so a D/ST is
   complete with a Sleeper id alone.

Invariants: a source id maps to one ESPN player, and a player has at most one id per source (the ``player_ids`` table
enforces the same). :meth:`Crosswalk.unmapped` reports ESPN players with no id for a source; :func:`check_rostered`
is the gate DESIGN asks for: sync fails loudly when any player on any roster in a managed league is unmapped, because
a silent id mismatch is the likeliest way to make a wrong decision. The fix is one line in the overrides file. The
gate covers NFL leagues and refuses any other: the sync job dispatches on the league's sport, and an NBA league is
checked by the NBA crosswalk's own gate (ROADMAP #17).

:meth:`Crosswalk.save` replaces the sport's ``player_ids`` rows atomically and :meth:`Crosswalk.from_store` reads them
back, so jobs that only translate ids (projection blend, availability) never touch nflverse.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Unpack

import polars as pl

from fm import paths
from fm.espn.ids import FFL_PRO_TEAMS
from fm.sources.base import Fetched, FetchOptions
from fm.store import PlayerIdRow, PlayerRow, Sport, Store

if TYPE_CHECKING:
    from fm.sources.nflverse import NflverseSource

SPORT: Sport = "nfl"
GSIS = "gsis"
SLEEPER = "sleeper"
SOURCES: tuple[str, ...] = (GSIS, SLEEPER)
"""The id systems the crosswalk covers, in report order."""

ORIGIN_FF_PLAYERIDS = "ff_playerids"
ORIGIN_OVERRIDE = "override"
ORIGIN_TEAM_DEFENSE = "team_defense"

TEAM_DEFENSE_BASE = -16000
"""ESPN D/ST player ids are ``TEAM_DEFENSE_BASE - proTeamId`` (the Eagles, pro team 21, are ``-16021``)."""

type TeamCodeSource = Literal["nflverse", "sleeper"]
TEAM_CODES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "nflverse": MappingProxyType({"LAR": "LA", "WSH": "WAS"}),
        "sleeper": MappingProxyType({"WSH": "WAS"}),
    }
)
"""Pro-team abbreviations other sources spell differently from ESPN; every other code matches ESPN's."""

DEFAULT_OVERRIDES = paths.data_file("id_overrides.csv")
OVERRIDE_COLUMNS = ("espn_id", "source", "source_id", "name", "note")
REQUIRED_OVERRIDE_COLUMNS = ("espn_id", "source", "source_id")
ID_FORMATS: Mapping[str, re.Pattern[str]] = MappingProxyType(
    {
        GSIS: re.compile(r"\d{2}-\d{7}"),  # 00-0034857
        SLEEPER: re.compile(r"\d+|[A-Z]{2,3}"),  # 4984, or a team code for a defense
    }
)
NULL_CELLS = frozenset({"", "NA", "NULL", "nan", "NaN"})
MAX_NAMES_IN_WARNING = 5


class CrosswalkError(ValueError):
    """The ID map or the overrides file cannot be used as given."""


@dataclass(frozen=True, slots=True)
class UnmappedPlayer:
    """An ESPN player with no id in one or more sources. ``name``/``position``/``pro_team`` come from the ``players``
    table when the player is stored there."""

    espn_id: int
    missing: tuple[str, ...]
    name: str | None = None
    position: str | None = None
    pro_team: str | None = None

    def __str__(self) -> str:
        """``Jahmyr Gibbs (RB DET, ESPN 4429795): no gsis id`` or ``ESPN 999999: no gsis, sleeper id``."""
        detail = " ".join(part for part in (self.position, self.pro_team) if part)
        if self.name:
            where = f"{detail}, ESPN {self.espn_id}" if detail else f"ESPN {self.espn_id}"
            who = f"{self.name} ({where})"
        else:
            who = f"ESPN {self.espn_id}" + (f" ({detail})" if detail else "")
        return f"{who}: no {', '.join(self.missing)} id"


class UnmappedPlayersError(CrosswalkError):
    """The gate: players that must be mapped are not. ``unmapped`` lists them, sorted by name."""

    def __init__(self, unmapped: Iterable[UnmappedPlayer], *, context: str = "players") -> None:
        self.unmapped = tuple(unmapped)
        listing = "\n".join(f"  {entry}" for entry in self.unmapped)
        super().__init__(
            f"{context}: {len(self.unmapped)} unmapped; add rows to {DEFAULT_OVERRIDES.name} and sync again:\n{listing}"
        )


@dataclass(frozen=True, slots=True)
class Override:
    """One line of the overrides file. ``source_id`` ``None`` means: remove the mapping ff_playerids has."""

    espn_id: int
    source: str
    source_id: str | None
    name: str | None = None
    note: str | None = None


# --- teams ------------------------------------------------------------------------------------------------------------


def team_code(espn_abbrev: str, source: TeamCodeSource) -> str:
    """A pro team as ``source`` spells it (``WSH`` -> ``WAS`` everywhere; ``LAR`` -> ``LA`` in nflverse)."""
    return TEAM_CODES[source].get(espn_abbrev, espn_abbrev)


def team_defense_id(pro_team_id: int) -> int:
    """The ESPN player id of a pro team's D/ST."""
    return TEAM_DEFENSE_BASE - pro_team_id


def team_defense_pro_team_id(espn_id: int) -> int | None:
    """The pro team behind a D/ST player id, or ``None`` for an ordinary player."""
    team_id = TEAM_DEFENSE_BASE - espn_id
    return team_id if team_id != 0 and team_id in FFL_PRO_TEAMS else None


def is_team_defense(espn_id: int) -> bool:
    return team_defense_pro_team_id(espn_id) is not None


def sources_for(espn_id: int) -> tuple[str, ...]:
    """The id systems a player is expected in: all of them for a player, Sleeper only for a D/ST (GSIS has no teams)."""
    return (SLEEPER,) if is_team_defense(espn_id) else SOURCES


# --- overrides --------------------------------------------------------------------------------------------------------


def load_overrides(path: Path | None = None) -> tuple[Override, ...]:
    """Read the overrides file (``DEFAULT_OVERRIDES`` unless given). A missing file is an error, never silence."""
    target = DEFAULT_OVERRIDES if path is None else path
    if not target.is_file():
        raise CrosswalkError(f"overrides file not found: {target}")
    return parse_overrides(target.read_text(encoding="utf-8"), where=str(target))


def parse_overrides(text: str, *, where: str = "overrides") -> tuple[Override, ...]:
    """Parse overrides CSV text: a header, then one mapping per line; ``#`` lines and blank lines are ignored.

    Raises :class:`CrosswalkError` naming ``where`` and the line for a bad header, field count, ``espn_id``, source,
    id format, or a duplicate (``espn_id``, ``source``) / (``source``, ``source_id``).
    """
    numbered = [
        (number, line)
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not numbered:
        return ()
    header_number, header_line = numbered[0]
    columns = [name.strip().lower() for name in next(csv.reader([header_line]))]
    missing = [name for name in REQUIRED_OVERRIDE_COLUMNS if name not in columns]
    if missing:
        raise CrosswalkError(
            f"{where}:{header_number}: header lacks {', '.join(missing)}; expected {','.join(OVERRIDE_COLUMNS)}"
        )
    overrides: list[Override] = []
    players_seen: dict[tuple[int, str], int] = {}
    ids_seen: dict[tuple[str, str], int] = {}
    for number, line in numbered[1:]:
        at = f"{where}:{number}"
        values = next(csv.reader([line]))
        if len(values) != len(columns):
            raise CrosswalkError(f"{at}: expected {len(columns)} fields, got {len(values)}")
        override = _parse_override({name: value.strip() for name, value in zip(columns, values, strict=True)}, at)
        player_key = (override.espn_id, override.source)
        if player_key in players_seen:
            raise CrosswalkError(
                f"{at}: ESPN {override.espn_id} {override.source} already set on line {players_seen[player_key]}"
            )
        players_seen[player_key] = number
        if override.source_id is not None:
            id_key = (override.source, override.source_id)
            if id_key in ids_seen:
                raise CrosswalkError(
                    f"{at}: {override.source} id {override.source_id} already used on line {ids_seen[id_key]}"
                )
            ids_seen[id_key] = number
        overrides.append(override)
    return tuple(overrides)


def _parse_override(record: Mapping[str, str], at: str) -> Override:
    try:
        espn_id = int(record["espn_id"])
    except ValueError:
        raise CrosswalkError(f"{at}: espn_id {record['espn_id']!r} is not an integer") from None
    if espn_id == 0:
        raise CrosswalkError(f"{at}: espn_id must not be 0")
    source = record["source"].lower()
    if source not in SOURCES:
        raise CrosswalkError(f"{at}: unknown source {record['source']!r}; expected one of {', '.join(SOURCES)}")
    source_id = record["source_id"] or None
    if source_id is not None and ID_FORMATS[source].fullmatch(source_id) is None:
        raise CrosswalkError(f"{at}: {source_id!r} does not look like a {source} id")
    return Override(espn_id, source, source_id, record.get("name") or None, record.get("note") or None)


# --- the crosswalk ----------------------------------------------------------------------------------------------------


class Crosswalk:
    """An immutable ESPN <-> source id map for NFL players, with O(1) lookups both ways.

    Built by :func:`build_crosswalk` or read back with :meth:`from_store`. ``warnings`` are the build's data-quality
    notes (ff_playerids rows without an ESPN id, conflicting rows that were dropped, overrides that no longer change
    anything); sync surfaces them next to the source adapters' own.
    """

    def __init__(self, rows: Iterable[PlayerIdRow], *, warnings: Iterable[str] = ()) -> None:
        by_player: dict[int, dict[str, PlayerIdRow]] = {}
        by_source: dict[tuple[str, str], PlayerIdRow] = {}
        for row in rows:
            if row.sport != SPORT:
                raise CrosswalkError(f"crosswalk rows must be {SPORT}, got {row.sport} for ESPN {row.espn_id}")
            ids = by_player.setdefault(row.espn_id, {})
            if row.source in ids:
                raise CrosswalkError(
                    f"ESPN {row.espn_id} has two {row.source} ids: {ids[row.source].source_id} and {row.source_id}"
                )
            key = (row.source, row.source_id)
            if key in by_source:
                raise CrosswalkError(
                    f"{row.source} id {row.source_id} maps to ESPN {by_source[key].espn_id} and ESPN {row.espn_id}"
                )
            ids[row.source] = row
            by_source[key] = row
        self._by_player = by_player
        self._by_source = by_source
        self._rows = tuple(sorted(by_source.values(), key=lambda row: (row.espn_id, row.source)))
        self.warnings = tuple(warnings)

    @property
    def rows(self) -> tuple[PlayerIdRow, ...]:
        """Every mapping, ordered by (``espn_id``, ``source``)."""
        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def espn_id(self, source: str, source_id: str | int) -> int | None:
        """The ESPN player a source id belongs to (``sleeper`` ``"4984"`` -> ``3918298``)."""
        row = self._by_source.get((source, str(source_id).strip()))
        return None if row is None else row.espn_id

    def source_id(self, espn_id: int, source: str) -> str | None:
        row = self._by_player.get(espn_id, {}).get(source)
        return None if row is None else row.source_id

    def gsis_id(self, espn_id: int) -> str | None:
        return self.source_id(espn_id, GSIS)

    def sleeper_id(self, espn_id: int) -> str | None:
        return self.source_id(espn_id, SLEEPER)

    def ids_for(self, espn_id: int) -> dict[str, str]:
        """``source -> source_id`` for one player; empty when unknown."""
        return {source: row.source_id for source, row in sorted(self._by_player.get(espn_id, {}).items())}

    def mapped(self, source: str | None = None) -> set[int]:
        """ESPN ids with an id in ``source`` (in any source when ``None``)."""
        if source is None:
            return set(self._by_player)
        return {espn_id for espn_id, ids in self._by_player.items() if source in ids}

    def unmapped(
        self,
        espn_ids: Iterable[int],
        *,
        sources: Sequence[str] = SOURCES,
        players: Iterable[PlayerRow] = (),
    ) -> list[UnmappedPlayer]:
        """The players among ``espn_ids`` lacking an id in any of ``sources`` they are expected in (a D/ST only in
        Sleeper), named from ``players`` where possible; sorted by name, then id."""
        known = {player.espn_id: player for player in players}
        found: list[UnmappedPlayer] = []
        for espn_id in set(espn_ids):
            expected = sources_for(espn_id)
            missing = tuple(s for s in sources if s in expected and self.source_id(espn_id, s) is None)
            if not missing:
                continue
            player = known.get(espn_id)
            found.append(
                UnmappedPlayer(
                    espn_id,
                    missing,
                    name=player.full_name if player else None,
                    position=player.position if player else None,
                    pro_team=player.pro_team if player else None,
                )
            )
        return sorted(found, key=lambda entry: (entry.name or "", entry.espn_id))

    def require_mapped(
        self,
        espn_ids: Iterable[int],
        *,
        sources: Sequence[str] = SOURCES,
        players: Iterable[PlayerRow] = (),
        context: str = "players",
    ) -> None:
        """Raise :class:`UnmappedPlayersError` unless every one of ``espn_ids`` is mapped (see :meth:`unmapped`)."""
        missing = self.unmapped(espn_ids, sources=sources, players=players)
        if missing:
            raise UnmappedPlayersError(missing, context=context)

    def save(self, store: Store) -> int:
        """Replace the sport's ``player_ids`` rows with this crosswalk in one transaction; returns the row count.

        A replace rather than an upsert (``PlayerIdRepo.replace_sport``): when an override re-points a source id to
        another player, the old row would otherwise collide with the table's one-owner-per-source-id constraint.
        """
        return store.player_ids.replace_sport(SPORT, self._rows)

    @classmethod
    def from_store(cls, store: Store) -> Crosswalk:
        """The crosswalk last saved to the store (empty before the first sync)."""
        return cls(store.player_ids.for_sport(SPORT))


class _Mappings:
    """Build-time state: (espn_id, source) -> (source_id, origin), one owner per source id, conflicts noted."""

    def __init__(self) -> None:
        self.by_player: dict[tuple[int, str], tuple[str, str]] = {}
        self.owner: dict[tuple[str, str], int] = {}
        self.warnings: list[str] = []

    def add(self, espn_id: int, source: str, source_id: str, origin: str) -> bool:
        """Add unless the player already has a ``source`` id or the id already has an owner; first row wins."""
        current = self.by_player.get((espn_id, source))
        if current is not None:
            if current[0] != source_id:
                self.warnings.append(
                    f"{origin}: ESPN {espn_id} has {source} ids {current[0]} and {source_id}; kept {current[0]}"
                )
            return False
        owner = self.owner.get((source, source_id))
        if owner is not None:
            self.warnings.append(
                f"{origin}: {source} id {source_id} is listed for ESPN {owner} and ESPN {espn_id}; kept ESPN {owner}"
            )
            return False
        self.by_player[(espn_id, source)] = (source_id, origin)
        self.owner[(source, source_id)] = espn_id
        return True

    def override(self, override: Override) -> None:
        """Apply one override: drop what the player had for the source and what the id was attached to, then set."""
        current = self.by_player.get((override.espn_id, override.source))
        who = f"ESPN {override.espn_id} {override.source}"
        if override.source_id is None:
            if current is None:
                self.warnings.append(f"{ORIGIN_OVERRIDE}: {who} removes nothing; the row can go")
            self.remove(override.espn_id, override.source)
            return
        if current is not None and current[0] == override.source_id and current[1] != ORIGIN_OVERRIDE:
            self.warnings.append(
                f"{ORIGIN_OVERRIDE}: {who} = {override.source_id} is now in {current[1]}; the row can go"
            )
        self.remove(override.espn_id, override.source)
        previous_owner = self.owner.get((override.source, override.source_id))
        if previous_owner is not None:
            self.remove(previous_owner, override.source)
        self.by_player[(override.espn_id, override.source)] = (override.source_id, ORIGIN_OVERRIDE)
        self.owner[(override.source, override.source_id)] = override.espn_id

    def remove(self, espn_id: int, source: str) -> None:
        current = self.by_player.pop((espn_id, source), None)
        if current is not None:
            del self.owner[(source, current[0])]

    def rows(self, as_of: datetime) -> list[PlayerIdRow]:
        return [
            PlayerIdRow(sport=SPORT, espn_id=espn_id, source=source, source_id=source_id, origin=origin, as_of=as_of)
            for (espn_id, source), (source_id, origin) in self.by_player.items()
        ]


def _clean_id(value: object) -> str | None:
    """An id cell as text: ints and whole floats lose the decimal, blanks and NA markers are ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    return None if text in NULL_CELLS else text


def build_crosswalk(frame: pl.DataFrame, overrides: Iterable[Override] = (), *, as_of: datetime) -> Crosswalk:
    """ff_playerids rows -> mappings, then the overrides, then the derived team defenses.

    ``frame`` is ``ff_playerids`` as nflreadpy delivers it (``espn_id``, ``gsis_id``, ``sleeper_id`` and ``name`` are
    used; the ids may arrive as ints, floats or text). Rows without an ESPN id contribute nothing and are counted in
    ``warnings``; a second id for a player, or a second player for an id, keeps the first and is noted there too.
    ``as_of`` stamps every row (the fetch's ``as_of``).
    """
    missing = [column for column in ("espn_id", "gsis_id", "sleeper_id") if column not in frame.columns]
    if missing:
        raise CrosswalkError(f"ff_playerids is missing columns {', '.join(missing)}")
    mappings = _Mappings()
    without_espn: list[str | None] = []
    columns = [column for column in ("espn_id", "gsis_id", "sleeper_id", "name") if column in frame.columns]
    for record in frame.select(columns).iter_rows(named=True):
        espn_text = _clean_id(record["espn_id"])
        try:
            espn_id = int(espn_text) if espn_text is not None else None
        except ValueError:
            espn_id = None
        if espn_id is None:
            without_espn.append(_clean_id(record.get("name")))
            continue
        for source, column in ((GSIS, "gsis_id"), (SLEEPER, "sleeper_id")):
            source_id = _clean_id(record[column])
            if source_id is not None:
                mappings.add(espn_id, source, source_id, ORIGIN_FF_PLAYERIDS)
    for override in overrides:
        mappings.override(override)
    for team_id, abbrev in FFL_PRO_TEAMS.items():
        if team_id == 0:
            continue
        espn_id = team_defense_id(team_id)
        if (espn_id, SLEEPER) in mappings.by_player:
            continue  # an override spoke
        mappings.add(espn_id, SLEEPER, team_code(abbrev, "sleeper"), ORIGIN_TEAM_DEFENSE)
    warnings: list[str] = []
    if without_espn:
        names = [name for name in without_espn if name]
        shown = ", ".join(names[:MAX_NAMES_IN_WARNING])
        more = f" and {len(names) - MAX_NAMES_IN_WARNING} more" if len(names) > MAX_NAMES_IN_WARNING else ""
        warnings.append(
            f"{ORIGIN_FF_PLAYERIDS}: {len(without_espn)} of {frame.height} rows have no ESPN id"
            + (f" ({shown}{more})" if shown else "")
        )
    warnings.extend(mappings.warnings)
    return Crosswalk(mappings.rows(as_of), warnings=warnings)


def fetch_crosswalk(
    source: NflverseSource,
    *,
    overrides: Path | None = None,
    **options: Unpack[FetchOptions],
) -> Fetched[Crosswalk]:
    """Build the crosswalk from the adapter's ``ff_playerids`` and the overrides file (``DEFAULT_OVERRIDES`` unless
    given). The result carries the fetch's provenance (``as_of``, ``cached``, ``stale``) and its warnings followed by
    the build's, so a sync can surface everything from one place."""
    fetched = source.ff_playerids(**options)
    crosswalk = build_crosswalk(fetched.data, load_overrides(overrides), as_of=fetched.as_of)
    return Fetched(
        crosswalk,
        fetched.as_of,
        fetched.source,
        fetched.dataset,
        fetched.key,
        cached=fetched.cached,
        stale=fetched.stale,
        degraded=fetched.degraded,
        warnings=(*fetched.warnings, *crosswalk.warnings),
        raw_path=fetched.raw_path,
    )


def check_rostered(
    store: Store,
    league_id: int,
    *,
    crosswalk: Crosswalk | None = None,
    scoring_period_id: int | None = None,
    sources: Sequence[str] = SOURCES,
) -> set[int]:
    """The unmapped-rostered-player gate for an NFL league: every player on any roster in the league must have an id in
    each of ``sources`` it is expected in, or :class:`UnmappedPlayersError` names the offenders (with names from
    ``players``).

    Checks the latest roster snapshot unless ``scoring_period_id`` is given, against ``crosswalk`` or the one saved in
    the store. Returns the ESPN ids checked; empty before the first roster snapshot.

    This is the NFL crosswalk's gate only, so callers dispatch on the league's sport: a league of another sport raises
    :class:`CrosswalkError` (its own crosswalk checks it) and a league id the store does not know raises
    ``LookupError``, both before any roster is read, so neither can pass vacuously.
    """
    league = store.leagues.get(league_id)
    if league is None:
        raise LookupError(f"no league with id {league_id} in the store")
    if league.sport != SPORT:
        raise CrosswalkError(
            f"league {league_id} ({league.key} {league.season}) is an {league.sport} league; the {SPORT} crosswalk "
            f"gates {SPORT} leagues only, so check it with the {league.sport} crosswalk"
        )
    period = scoring_period_id if scoring_period_id is not None else store.rosters.latest_period(league_id)
    if period is None:
        return set()
    rostered = store.rosters.rostered_ids(league_id, period)
    if not rostered:
        return rostered
    walk = crosswalk if crosswalk is not None else Crosswalk.from_store(store)
    walk.require_mapped(
        rostered,
        sources=sources,
        players=store.players.many(SPORT, rostered),
        context=f"rostered players in league {league_id} (scoring period {period})",
    )
    return rostered
