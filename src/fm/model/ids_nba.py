"""NBA player ID crosswalk: ESPN ids <-> nba.com person ids (DESIGN section 7, "Player identity").

Every NBA player has one canonical row keyed by ESPN id; stats.nba.com (``fm.sources.nba_stats``) and DARKO
(``fm.sources.darko``) key players by nba.com person id, and no public ID map joins the two. The crosswalk is built in
two layers, mirroring :mod:`fm.model.ids`:

1. A match on (normalised name, team, position). Each ESPN player (a ``players`` row) is matched to the nba.com
   persons the sources list (:class:`NbaPerson`): the same normalised name (diacritics, punctuation and
   ``Jr.``/``III`` suffixes removed), with his team picking between namesakes, and a candidate that disagrees on both
   team and position rejected as someone else. Teams compare as nba.com tricodes; positions as the guard, forward
   and center groups of ESPN's default position plus every position his ESPN slots make him eligible at. A player no
   nba.com name fits is tried once more on his team by surname and a first name one spelling shortens
   (``Herb Jones`` -> ``Herbert Jones``, ``Nicolas Claxton`` -> ``Nic Claxton``, but never ``Caleb`` -> ``Cody``).
   Such partial matches and name matches that disagree with ESPN on team or position stay in the crosswalk and are
   called out in ``warnings``; a player with no match, or several, is left unmapped with a :class:`MatchNote` saying
   why.
2. ``data/id_overrides_nba.csv``, the reviewed file, in the NFL file's layout (``espn_id,source,source_id,name,note``,
   ``#`` comments allowed). A row pins the player to that nba.com id, or with an empty ``source_id`` keeps him
   unmapped, whatever the match would say. Matching runs around the pins: a pinned player and a pinned nba.com id are
   out of play, so the match never contradicts a pin, and a player whose namesake is pinned to someone else is told
   so rather than quietly re-pointed.

Invariants: an nba.com id maps to one ESPN player, and a player has at most one id per source (the ``player_ids`` table
enforces the same). :func:`check_nba_rostered` is the gate DESIGN asks for: sync fails loudly, naming each unmapped
rostered player and why the match failed, and the fix is one line in the overrides file. Like the NFL gate it refuses
a league of another sport. :meth:`NbaCrosswalk.save` replaces the sport's ``player_ids`` rows atomically and
:meth:`NbaCrosswalk.from_store` reads them back, so jobs that only translate ids never touch the sources.

:func:`fetch_nba_crosswalk` gathers the persons: stats.nba.com's player splits per season (everyone who played, with
his latest team), DARKO's talent sheet (positions, and players without a game yet) and the active players in
``nba_api``'s bundled table (names and ids only), so the crosswalk can be built before opening night.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Unpack

import polars as pl
from nba_api.stats.static import players as nba_static_players
from nba_api.stats.static import teams as nba_static_teams

from fm import paths
from fm.espn.ids import FBA
from fm.model.ids import (
    MAX_NAMES_IN_WARNING,
    ORIGIN_OVERRIDE,
    OVERRIDE_COLUMNS,
    REQUIRED_OVERRIDE_COLUMNS,
    CrosswalkError,
    Override,
    UnmappedPlayer,
    UnmappedPlayersError,
)
from fm.sources.base import Fetched, FetchOptions, SourceError
from fm.store import PlayerIdRow, PlayerRow, Sport, Store

if TYPE_CHECKING:
    from fm.sources.darko import DarkoProjection, DarkoSource
    from fm.sources.nba_stats import NbaStatsSource

NBA_SPORT: Sport = "nba"
NBA_SOURCE = "nba"
"""The id system: nba.com person ids, as stats.nba.com (``PLAYER_ID``) and DARKO (``nba_id``) key players."""

ORIGIN_NAME_MATCH = "name_match"
ORIGIN_PARTIAL_MATCH = "partial_match"

NBA_OVERRIDES = paths.data_file("id_overrides_nba.csv")
NBA_ID_FORMAT = re.compile(r"[1-9]\d{0,9}")
"""An nba.com person id as the sources print it: ``2544`` (LeBron James) through ``1642843`` (a 2025 rookie)."""

FREE_AGENT = "FA"
ESPN_TEAM_ALIASES: Mapping[str, str] = MappingProxyType(
    {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
)
"""ESPN's own pro-team abbreviations (``proTeamSchedules_wl``) that differ from the nba.com tricodes
:data:`fm.espn.ids.FBA_PRO_TEAMS` uses; every other code is the same on both sides."""

NBA_TEAM_CODES: Mapping[int, str] = MappingProxyType(
    {int(team["id"]): str(team["abbreviation"]) for team in nba_static_teams.get_teams()}
)
"""nba.com team id (``1610612759``) to tricode (``SAS``), from ``nba_api``'s bundled table."""

POSITION_GROUPS: Mapping[str, str] = MappingProxyType(
    {
        "PG": "G",
        "SG": "G",
        "G": "G",
        "GUARD": "G",
        "SF": "F",
        "PF": "F",
        "F": "F",
        "FORWARD": "F",
        "C": "C",
        "CENTER": "C",
    }
)
"""Position words in any source's vocabulary (ESPN ``PF``, nba.com ``F-C``, DARKO ``Center-Forward`` or ``c_pos``) to
the three groups the match compares."""

PERSON_COLUMNS = ("PLAYER_ID", "PLAYER_NAME")
"""What a stats.nba.com player frame must carry; ``TEAM_ABBREVIATION``, ``POSITION`` and ``GAME_DATE`` are used when
present."""

_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
_PLAIN_LETTERS = str.maketrans(
    {"đ": "d", "ð": "d", "ø": "o", "ł": "l", "ı": "i", "ß": "ss", "æ": "ae", "œ": "oe", "þ": "th"}
)
_DROPPED = re.compile(r"[.'’‘ʼ`]")
_SEPARATORS = re.compile(r"[^a-z0-9]+")
_NON_LETTERS = re.compile(r"[^A-Z]+")
_SINGLE_POSITION_SLOTS: Mapping[int, str] = MappingProxyType(
    {slot_id: label for slot_id, label in FBA.lineup_slots.items() if label in FBA.positions.values()}
)
_MIN_PREFIX = 2


class UnmappedNbaPlayersError(UnmappedPlayersError):
    """The NBA gate: players with no nba.com id. ``unmapped`` lists them sorted by name and ``reasons`` holds why the
    match failed, by ESPN id, where the crosswalk knows it; the fix is a row in ``id_overrides_nba.csv``."""

    def __init__(
        self,
        unmapped: Iterable[UnmappedPlayer],
        *,
        context: str = "players",
        reasons: Mapping[int, str] | None = None,
    ) -> None:
        self.unmapped = tuple(unmapped)
        self.reasons = dict(reasons or {})
        listing = "\n".join(
            f"  {entry}" + (f" ({self.reasons[entry.espn_id]})" if entry.espn_id in self.reasons else "")
            for entry in self.unmapped
        )
        # The NFL error's message names the NFL overrides file, so build this one directly.
        CrosswalkError.__init__(
            self,
            f"{context}: {len(self.unmapped)} unmapped; add rows to {NBA_OVERRIDES.name} and sync again:\n{listing}",
        )


@dataclass(frozen=True, slots=True)
class NbaPerson:
    """One nba.com person as a source lists him: id, display name, team tricode (``None`` when the source does not
    say, or for a free agent) and listed position in the source's own vocabulary (see :func:`position_groups`).
    ``aliases`` are other spellings seen for the same id."""

    nba_id: int
    name: str
    team: str | None = None
    position: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    def __str__(self) -> str:
        detail = " ".join(part for part in (self.position, self.team) if part)
        return f"nba {self.nba_id} {self.name}" + (f" ({detail})" if detail else "")


class MatchKind(StrEnum):
    """What a :class:`MatchNote` reports. ``DISPUTED`` and ``PARTIAL`` players are mapped; the rest are not."""

    NO_MATCH = "no_match"
    AMBIGUOUS = "ambiguous"
    DISPUTED = "disputed"
    PARTIAL = "partial"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class MatchNote:
    """Why an ESPN player has no nba.com id, or why his match deserves a look. ``candidates`` are the nba.com ids in
    play: the ones that fit equally, the one matched with a caveat, or near misses worth an override."""

    espn_id: int
    name: str
    kind: MatchKind
    reason: str
    candidates: tuple[int, ...] = ()

    def __str__(self) -> str:
        return f"{self.name} (ESPN {self.espn_id}): {self.reason}"


# --- names, teams, positions ------------------------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    """A name as both sides spell it: ASCII letters and digits, lower case, single spaces, no periods or apostrophes,
    no ``Jr.``/``Sr.``/``II``-``V`` suffix (``Luka Dončić`` -> ``luka doncic``, ``P.J. Washington`` -> ``pj
    washington``, ``Jaren Jackson Jr.`` -> ``jaren jackson``, ``Shai Gilgeous-Alexander`` -> ``shai gilgeous
    alexander``)."""
    decomposed = unicodedata.normalize("NFKD", name.lower().translate(_PLAIN_LETTERS))
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    tokens = [token for token in _SEPARATORS.split(_DROPPED.sub("", ascii_text)) if token]
    while len(tokens) > 1 and tokens[-1] in _SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def position_groups(position: str | None) -> frozenset[str]:
    """The groups (``G``, ``F``, ``C``) a position string names, in any source's vocabulary; empty when the position
    is missing or uses words the table lacks."""
    if not position:
        return frozenset()
    tokens = _NON_LETTERS.split(position.upper().removesuffix("_POS"))
    return frozenset(POSITION_GROUPS[token] for token in tokens if token in POSITION_GROUPS)


def espn_position_groups(player: PlayerRow) -> frozenset[str]:
    """The groups of an ESPN player's default position and of every position his eligible slots name (an ``SG``
    eligible at ``SF`` is a guard and a forward). Combination slots (``G/F``, ``UTIL``) say nothing about him."""
    labels = [player.position, *(_SINGLE_POSITION_SLOTS.get(slot_id) for slot_id in player.eligible_slot_ids)]
    return frozenset(group for label in labels for group in position_groups(label))


def tricode(label: str | None) -> str | None:
    """A pro-team label as an nba.com tricode: ESPN's own spellings translated (``NY`` -> ``NYK``), the free-agent
    pseudo team and blanks ``None``."""
    code = (label or "").strip().upper()
    if not code or code == FREE_AGENT:
        return None
    return ESPN_TEAM_ALIASES.get(code, code)


def espn_tricode(player: PlayerRow) -> str | None:
    """An ESPN player's pro team as an nba.com tricode: from ``pro_team_id`` through
    :data:`fm.espn.ids.FBA_PRO_TEAMS` when ESPN sent a known one, else from the ``pro_team`` label."""
    if player.pro_team_id is not None and player.pro_team_id in FBA.pro_teams:
        return tricode(FBA.pro_teams[player.pro_team_id])
    return tricode(player.pro_team)


def nba_tricode(team_id: int | None) -> str | None:
    """The tricode of an nba.com team id; ``None`` for a free agent (``None``) or an id the bundled table lacks."""
    return None if team_id is None else NBA_TEAM_CODES.get(team_id)


# --- persons from the sources -----------------------------------------------------------------------------------------


def persons_from_frame(frame: pl.DataFrame) -> list[NbaPerson]:
    """Persons from a stats.nba.com player frame (``player_splits``, ``game_logs``): ``PLAYER_ID`` and ``PLAYER_NAME``
    are required, ``TEAM_ABBREVIATION`` and ``POSITION`` used when present. A player on several rows (game logs across
    a trade) takes the team of his latest ``GAME_DATE`` (else his last row) and keeps every spelling of his name."""
    missing = [column for column in PERSON_COLUMNS if column not in frame.columns]
    if missing:
        raise CrosswalkError(f"stats.nba.com frame is missing columns {', '.join(missing)}")
    wanted = (*PERSON_COLUMNS, "TEAM_ABBREVIATION", "POSITION", "GAME_DATE")
    records = list(frame.select([column for column in wanted if column in frame.columns]).iter_rows(named=True))
    if "GAME_DATE" in frame.columns:
        records.sort(key=lambda record: str(record["GAME_DATE"] or ""))  # ISO dates: text order is date order
    persons: dict[int, NbaPerson] = {}
    for record in records:
        nba_id, name = _int_or_none(record["PLAYER_ID"]), _text_or_none(record["PLAYER_NAME"])
        if nba_id is None or name is None:
            continue
        team = tricode(_text_or_none(record.get("TEAM_ABBREVIATION")))
        position = _text_or_none(record.get("POSITION"))
        current = persons.get(nba_id)
        if current is None:
            persons[nba_id] = NbaPerson(nba_id, name, team=team, position=position)
            continue
        persons[nba_id] = NbaPerson(
            nba_id,
            current.name,
            team=team or current.team,
            position=position or current.position,
            aliases=_aliases(current, (name,)),
        )
    return list(persons.values())


def persons_from_darko(projections: Iterable[DarkoProjection]) -> list[NbaPerson]:
    """Persons from DARKO's talent rows: the listed position, else DARKO's modeled bucket (``c_pos``), and the team
    through its nba.com id (free agents carry none)."""
    return [
        NbaPerson(
            projection.nba_id,
            projection.name,
            team=nba_tricode(projection.team_id),
            position=projection.position or projection.x_position,
        )
        for projection in projections
    ]


def persons_from_static() -> list[NbaPerson]:
    """The active players in ``nba_api``'s bundled player table: names and ids only, no team or position. It ships
    with the package (offline) and trails the league by a draft or two, so it only fills gaps the live sources leave,
    such as a player who has not played this season."""
    found: list[NbaPerson] = []
    for entry in nba_static_players.get_players():
        record: Mapping[str, Any] = entry
        nba_id, name = _int_or_none(record.get("id")), _text_or_none(record.get("full_name"))
        if record.get("is_active") and nba_id is not None and name is not None:
            found.append(NbaPerson(nba_id, name))
    return found


def merge_persons(*groups: Iterable[NbaPerson]) -> list[NbaPerson]:
    """One person per nba.com id across sources, in first-seen order: the first group to name a team or a position
    wins it, later groups fill the gaps, and every other spelling of the name becomes an alias. Pass the freshest
    source first."""
    merged: dict[int, NbaPerson] = {}
    for group in groups:
        for person in group:
            current = merged.get(person.nba_id)
            if current is None:
                merged[person.nba_id] = person
                continue
            merged[person.nba_id] = NbaPerson(
                person.nba_id,
                current.name,
                team=current.team or person.team,
                position=current.position or person.position,
                aliases=_aliases(current, person.names),
            )
    return list(merged.values())


def _aliases(person: NbaPerson, spellings: Iterable[str]) -> tuple[str, ...]:
    return tuple(name for name in dict.fromkeys((*person.aliases, *spellings)) if name != person.name)


# --- overrides --------------------------------------------------------------------------------------------------------


def load_nba_overrides(path: Path | None = None) -> tuple[Override, ...]:
    """Read the overrides file (``NBA_OVERRIDES`` unless given). A missing file is an error, never silence."""
    target = NBA_OVERRIDES if path is None else path
    if not target.is_file():
        raise CrosswalkError(f"overrides file not found: {target}")
    return parse_nba_overrides(target.read_text(encoding="utf-8"), where=str(target))


def parse_nba_overrides(text: str, *, where: str = "overrides") -> tuple[Override, ...]:
    """Parse overrides CSV text: a header, then one mapping per line; ``#`` lines and blank lines are ignored.

    The NFL parser's rules with the NBA id system: raises :class:`CrosswalkError` naming ``where`` and the line for a
    bad header, field count or ``espn_id``, a source other than ``nba``, an id that is not an nba.com person id, or a
    duplicate (``espn_id``, ``source``) / (``source``, ``source_id``).
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
    players_seen: dict[int, int] = {}
    ids_seen: dict[str, int] = {}
    for number, line in numbered[1:]:
        at = f"{where}:{number}"
        values = next(csv.reader([line]))
        if len(values) != len(columns):
            raise CrosswalkError(f"{at}: expected {len(columns)} fields, got {len(values)}")
        override = _parse_override({name: value.strip() for name, value in zip(columns, values, strict=True)}, at)
        if override.espn_id in players_seen:
            raise CrosswalkError(
                f"{at}: ESPN {override.espn_id} {override.source} already set on line {players_seen[override.espn_id]}"
            )
        players_seen[override.espn_id] = number
        if override.source_id is not None:
            if override.source_id in ids_seen:
                raise CrosswalkError(
                    f"{at}: {override.source} id {override.source_id} already used on line "
                    f"{ids_seen[override.source_id]}"
                )
            ids_seen[override.source_id] = number
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
    if source != NBA_SOURCE:
        raise CrosswalkError(f"{at}: unknown source {record['source']!r}; expected {NBA_SOURCE}")
    source_id = record["source_id"] or None
    if source_id is not None and NBA_ID_FORMAT.fullmatch(source_id) is None:
        raise CrosswalkError(f"{at}: {source_id!r} does not look like an nba.com person id")
    return Override(espn_id, source, source_id, record.get("name") or None, record.get("note") or None)


# --- the crosswalk ----------------------------------------------------------------------------------------------------


class NbaCrosswalk:
    """An immutable ESPN <-> nba.com id map for NBA players, with O(1) lookups both ways.

    Built by :func:`build_nba_crosswalk` or read back with :meth:`from_store`. ``warnings`` summarise the build's
    data-quality notes (unmatched and ambiguous players, matches that disagree with ESPN, partial matches); ``notes``
    carry the per-player detail behind them, which the gate quotes. A crosswalk read from the store has neither.
    """

    def __init__(
        self,
        rows: Iterable[PlayerIdRow],
        *,
        warnings: Iterable[str] = (),
        notes: Iterable[MatchNote] = (),
    ) -> None:
        by_player: dict[int, dict[str, PlayerIdRow]] = {}
        by_source: dict[tuple[str, str], PlayerIdRow] = {}
        for row in rows:
            if row.sport != NBA_SPORT:
                raise CrosswalkError(f"crosswalk rows must be {NBA_SPORT}, got {row.sport} for ESPN {row.espn_id}")
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
        self.notes = tuple(sorted(notes, key=lambda note: (note.name, note.espn_id)))
        self._notes = {note.espn_id: note for note in self.notes}

    @property
    def rows(self) -> tuple[PlayerIdRow, ...]:
        """Every mapping, ordered by (``espn_id``, ``source``)."""
        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def espn_id(self, nba_id: int | str) -> int | None:
        """The ESPN player an nba.com person id belongs to (``203999`` -> ``3112335``)."""
        row = self._by_source.get((NBA_SOURCE, str(nba_id).strip()))
        return None if row is None else row.espn_id

    def nba_id(self, espn_id: int) -> int | None:
        """An ESPN player's nba.com person id, as the sources key him."""
        row = self._by_player.get(espn_id, {}).get(NBA_SOURCE)
        return None if row is None else int(row.source_id)

    def mapped(self) -> set[int]:
        """ESPN ids with an nba.com id."""
        return {espn_id for espn_id, ids in self._by_player.items() if NBA_SOURCE in ids}

    def note(self, espn_id: int) -> MatchNote | None:
        """The build's note on a player (why he has no id, or what to check about his match), if any."""
        return self._notes.get(espn_id)

    def unmapped(self, espn_ids: Iterable[int], *, players: Iterable[PlayerRow] = ()) -> list[UnmappedPlayer]:
        """The players among ``espn_ids`` without an nba.com id, named from ``players`` where possible; sorted by
        name, then id."""
        known = {player.espn_id: player for player in players}
        found: list[UnmappedPlayer] = []
        for espn_id in set(espn_ids):
            if self.nba_id(espn_id) is not None:
                continue
            player = known.get(espn_id)
            found.append(
                UnmappedPlayer(
                    espn_id,
                    (NBA_SOURCE,),
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
        players: Iterable[PlayerRow] = (),
        context: str = "players",
    ) -> None:
        """Raise :class:`UnmappedNbaPlayersError` unless every one of ``espn_ids`` has an nba.com id, quoting the
        build's reason for each player it has one for."""
        missing = self.unmapped(espn_ids, players=players)
        if missing:
            reasons = {entry.espn_id: note.reason for entry in missing if (note := self.note(entry.espn_id))}
            raise UnmappedNbaPlayersError(missing, context=context, reasons=reasons)

    def save(self, store: Store) -> int:
        """Replace the sport's ``player_ids`` rows with this crosswalk in one transaction; returns the row count.

        A replace rather than an upsert (``PlayerIdRepo.replace_sport``): when a pin moves an nba.com id to another
        player, the old row would otherwise collide with the table's one-owner-per-source-id constraint.
        """
        return store.player_ids.replace_sport(NBA_SPORT, self._rows)

    @classmethod
    def from_store(cls, store: Store) -> NbaCrosswalk:
        """The crosswalk last saved to the store (empty before the first sync)."""
        return cls(store.player_ids.for_sport(NBA_SPORT))


# --- the match --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Match:
    person: NbaPerson
    origin: str


class _Matcher:
    """Matches ESPN players to persons, one person per player and vice versa: exact names for everyone first, then
    the partial match for whoever no name fits. ``taken`` starts with the ids the overrides pin."""

    def __init__(self, persons: Iterable[NbaPerson], pinned_ids: Mapping[int, int]) -> None:
        self.by_name: dict[str, list[NbaPerson]] = {}
        self.by_surname: dict[tuple[str, str], list[NbaPerson]] = {}
        self.taken: dict[int, int] = dict(pinned_ids)  # nba id -> ESPN id
        self.pinned_ids = frozenset(pinned_ids)
        self.matches: dict[int, _Match] = {}
        self.notes: dict[int, MatchNote] = {}
        seen: set[int] = set()
        for person in persons:
            if person.nba_id in seen:
                continue
            seen.add(person.nba_id)
            for spelling in person.names:
                key = normalize_name(spelling)
                if not key:
                    continue
                _append_once(self.by_name.setdefault(key, []), person)
                parts = _name_parts(key)
                team = tricode(person.team)
                if parts is not None and team is not None:
                    _append_once(self.by_surname.setdefault((parts[1], team), []), person)

    def run(self, players: Sequence[PlayerRow]) -> None:
        shared = Counter(normalize_name(player.full_name) for player in players)
        leftovers: list[PlayerRow] = []
        for player in players:
            key = normalize_name(player.full_name)
            if shared[key] > 1:
                self.note(player, MatchKind.AMBIGUOUS, f"{shared[key]} ESPN players share this name")
            elif not self.match_name(player, key):
                leftovers.append(player)
        self.match_partial(leftovers)

    def match_name(self, player: PlayerRow, key: str) -> bool:
        """Settle a player whose name an nba.com person carries (matched or noted); ``False`` when none does."""
        candidates = self.by_name.get(key, [])
        if not candidates:
            return False
        available = [person for person in candidates if person.nba_id not in self.taken]
        if not available:
            owners = ", ".join(f"{person} to {self._owner(person.nba_id)}" for person in candidates)
            self.note(player, MatchKind.NO_MATCH, f"the nba.com player with this name is mapped already: {owners}")
            return True
        ranked: list[tuple[bool, NbaPerson, bool | None, bool | None]] = []
        for person in available:
            team, position = _agreement(player, person)
            if team is False and position is False:
                continue  # a different person who shares the name
            ranked.append((team is True, person, team, position))  # only his team tells namesakes apart
        if not ranked:
            listing = ", ".join(str(person) for person in available)
            self.note(
                player,
                MatchKind.NO_MATCH,
                f"ESPN lists him {_describe(player)}, but the nba.com player with this name is not: {listing}",
                tuple(person.nba_id for person in available),
            )
            return True
        best = max(score for score, *_ in ranked)
        top = sorted((entry for entry in ranked if entry[0] == best), key=lambda entry: entry[1].nba_id)
        if len(top) > 1:
            self.note(
                player,
                MatchKind.AMBIGUOUS,
                "several nba.com players fit: " + ", ".join(str(entry[1]) for entry in top),
                tuple(entry[1].nba_id for entry in top),
            )
            return True
        _, person, team, position = top[0]
        self.accept(player, person, ORIGIN_NAME_MATCH)
        disputed = [what for what, agreed in (("team", team), ("position", position)) if agreed is False]
        if disputed:
            self.note(
                player,
                MatchKind.DISPUTED,
                f"matched {person} by name, but ESPN lists him {_describe(player)}; the {' and '.join(disputed)} "
                f"{'disagree' if len(disputed) > 1 else 'disagrees'}",
                (person.nba_id,),
            )
        return True

    def match_partial(self, players: Sequence[PlayerRow]) -> None:
        """Surname, team and a first name one spelling shortens, for players no nba.com name fits. A person two of
        them fit goes to neither."""
        fits: dict[int, list[NbaPerson]] = {}
        near: dict[int, list[NbaPerson]] = {}
        for player in players:
            parts, team = _name_parts(normalize_name(player.full_name)), espn_tricode(player)
            if parts is None or team is None:
                fits[player.espn_id] = near[player.espn_id] = []
                continue
            first, surname = parts
            near[player.espn_id] = pool = [
                person for person in self.by_surname.get((surname, team), []) if person.nba_id not in self.taken
            ]
            fits[player.espn_id] = [
                person for person in pool if _agreement(player, person)[1] is not False and _shortens(first, person)
            ]
        wanted = Counter(person.nba_id for found in fits.values() for person in found)
        for player in players:
            found = fits[player.espn_id]
            listing = ", ".join(str(person) for person in found)
            if len(found) > 1:
                reason = f"several nba.com players fit by surname, team and first name: {listing}"
                self.note(player, MatchKind.AMBIGUOUS, reason, tuple(person.nba_id for person in found))
            elif found and wanted[found[0].nba_id] > 1:
                reason = f"{listing} fits him and another ESPN player by surname, team and first name"
                self.note(player, MatchKind.AMBIGUOUS, reason, (found[0].nba_id,))
            elif found:
                (person,) = found
                self.accept(player, person, ORIGIN_PARTIAL_MATCH)
                reason = f"matched {person} by surname, team and first name; confirm it with an override"
                self.note(player, MatchKind.PARTIAL, reason, (person.nba_id,))
            else:
                close = [person for person in near[player.espn_id] if person.nba_id not in self.taken]
                hint = f"; on his team: {', '.join(str(person) for person in close)}" if close else ""
                reason = f"no nba.com player named {player.full_name!r}{hint}"
                self.note(player, MatchKind.NO_MATCH, reason, tuple(person.nba_id for person in close))

    def accept(self, player: PlayerRow, person: NbaPerson, origin: str) -> None:
        self.taken[person.nba_id] = player.espn_id
        self.matches[player.espn_id] = _Match(person, origin)

    def note(self, player: PlayerRow, kind: MatchKind, reason: str, candidates: tuple[int, ...] = ()) -> None:
        self.notes[player.espn_id] = MatchNote(player.espn_id, player.full_name, kind, reason, candidates)

    def _owner(self, nba_id: int) -> str:
        how = "by an override" if nba_id in self.pinned_ids else "by name"
        return f"ESPN {self.taken[nba_id]} ({how})"


def _append_once(bucket: list[NbaPerson], person: NbaPerson) -> None:
    if person not in bucket:
        bucket.append(person)


def _name_parts(normalized: str) -> tuple[str, str] | None:
    """(first name, surname) of a normalised name; ``None`` for a single word."""
    first, _, surname = normalized.partition(" ")
    return (first, surname) if surname else None


def _shortens(first: str, person: NbaPerson) -> bool:
    """True when ``first`` and one of the person's first names are the same name, one shortening the other
    (``herb``/``herbert``, ``nic``/``nicolas``); a shared initial alone (``caleb``/``cody``) is not enough."""
    for spelling in person.names:
        parts = _name_parts(normalize_name(spelling))
        if parts is None:
            continue
        shorter, longer = sorted((first, parts[0]), key=len)
        if len(shorter) >= _MIN_PREFIX and longer.startswith(shorter):
            return True
    return False


def _agreement(player: PlayerRow, person: NbaPerson) -> tuple[bool | None, bool | None]:
    """Whether the team and the position agree; ``None`` where either side does not say."""
    espn_team, nba_team = espn_tricode(player), tricode(person.team)
    team = espn_team == nba_team if espn_team is not None and nba_team is not None else None
    espn_groups, nba_groups = espn_position_groups(player), position_groups(person.position)
    position = bool(espn_groups & nba_groups) if espn_groups and nba_groups else None
    return team, position


def _describe(player: PlayerRow) -> str:
    detail = " ".join(part for part in (player.position, player.pro_team) if part)
    return f"as {detail}" if detail else "without a team or position"


def build_nba_crosswalk(
    players: Iterable[PlayerRow],
    persons: Iterable[NbaPerson],
    overrides: Iterable[Override] = (),
    *,
    as_of: datetime,
) -> NbaCrosswalk:
    """ESPN players (``players`` rows of the ``nba`` sport) matched to nba.com persons by name, team and position
    around the overrides' pins, then the pins themselves. ``as_of`` stamps every row (the inputs' ``as_of``).

    The crosswalk covers the players given plus every pinned player: :meth:`NbaCrosswalk.save` replaces the sport's
    rows, so pass every player that needs an id (rostered and in the pool). Notes go to ``notes`` and a summary per
    kind to ``warnings``; a pinned player has no note unless his pin removes his id, which the gate then explains.
    """
    pins = {override.espn_id: override for override in overrides}
    roster: dict[int, PlayerRow] = {}
    for player in players:
        if player.sport != NBA_SPORT:
            raise CrosswalkError(f"players must be {NBA_SPORT}, got {player.sport} for ESPN {player.espn_id}")
        roster.setdefault(player.espn_id, player)
    pinned_ids = {int(pin.source_id): pin.espn_id for pin in pins.values() if pin.source_id is not None}
    matcher = _Matcher(persons, pinned_ids)
    matcher.run([player for espn_id, player in sorted(roster.items()) if espn_id not in pins])
    rows = [
        PlayerIdRow(
            sport=NBA_SPORT,
            espn_id=espn_id,
            source=NBA_SOURCE,
            source_id=str(match.person.nba_id),
            origin=match.origin,
            as_of=as_of,
        )
        for espn_id, match in matcher.matches.items()
    ]
    notes = list(matcher.notes.values())
    for pin in pins.values():
        if pin.source_id is not None:
            rows.append(
                PlayerIdRow(
                    sport=NBA_SPORT,
                    espn_id=pin.espn_id,
                    source=NBA_SOURCE,
                    source_id=pin.source_id,
                    origin=ORIGIN_OVERRIDE,
                    as_of=as_of,
                )
            )
        else:
            known = roster.get(pin.espn_id)
            name = known.full_name if known else pin.name or f"ESPN {pin.espn_id}"
            reason = f"{NBA_OVERRIDES.name} keeps him unmapped (an empty source_id); give him an nba.com id there"
            notes.append(MatchNote(pin.espn_id, name, MatchKind.REMOVED, reason))
    return NbaCrosswalk(rows, warnings=_summarise(len(roster), notes), notes=notes)


_SUMMARIES: tuple[tuple[MatchKind, str], ...] = (
    (MatchKind.NO_MATCH, ORIGIN_NAME_MATCH + ": {count} of {total} ESPN players have no nba.com match ({names})"),
    (MatchKind.AMBIGUOUS, ORIGIN_NAME_MATCH + ": {count} ESPN players fit more than one way ({names}); add overrides"),
    (MatchKind.DISPUTED, ORIGIN_NAME_MATCH + ": {count} matches disagree with ESPN on team or position ({names})"),
    (MatchKind.PARTIAL, ORIGIN_PARTIAL_MATCH + ": {count} matches by surname, team and first name ({names}); confirm"),
)
"""One build warning per kind of note (unmapped players, then mapped ones worth a look); pinned players excepted."""


def _summarise(total: int, notes: Iterable[MatchNote]) -> list[str]:
    by_kind: dict[MatchKind, list[MatchNote]] = {}
    for note in sorted(notes, key=lambda note: (note.name, note.espn_id)):
        by_kind.setdefault(note.kind, []).append(note)
    return [
        template.format(count=len(by_kind[kind]), total=total, names=_names(by_kind[kind]))
        for kind, template in _SUMMARIES
        if kind in by_kind
    ]


def _names(notes: Sequence[MatchNote]) -> str:
    shown = ", ".join(note.name for note in notes[:MAX_NAMES_IN_WARNING])
    more = len(notes) - MAX_NAMES_IN_WARNING
    return shown + (f" and {more} more" if more > 0 else "")


def _int_or_none(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# --- through the adapters ---------------------------------------------------------------------------------------------


def fetch_nba_crosswalk(
    players: Iterable[PlayerRow],
    stats: NbaStatsSource,
    seasons: str | Sequence[str],
    *,
    darko: DarkoSource | None = None,
    overrides: Path | None = None,
    **options: Unpack[FetchOptions],
) -> Fetched[NbaCrosswalk]:
    """Build the crosswalk from the persons the sources list and the overrides file (``NBA_OVERRIDES`` unless given).

    ``seasons`` are nba.com labels, current first (``("2026-27", "2025-26")``): before opening night the current
    season's splits are empty and last season's carry everyone who played. Teams come from the current season, then
    DARKO (when given), then the earlier seasons; ``nba_api``'s bundled table of active players fills in last.

    A source that fails is a warning and makes the result ``degraded`` rather than an error, since the others still
    match most players; a sync should keep the saved crosswalk rather than replace it with a degraded one. The result
    carries the oldest input's ``as_of``, is ``cached``/``stale`` when its inputs were, and lists the adapters'
    warnings, then the build's.
    """
    labels = (seasons,) if isinstance(seasons, str) else tuple(seasons)
    if not labels:
        raise ValueError("at least one season is required")
    inputs: list[Fetched[Any]] = []
    by_season: list[list[NbaPerson]] = []
    warnings: list[str] = []
    failed = False
    for season in labels:
        try:
            splits = stats.player_splits(season, "Base", **options)
        except SourceError as exc:
            warnings.append(f"{stats.name} {season}: unavailable, matching without it ({exc})")
            failed = True
            by_season.append([])
            continue
        inputs.append(splits)
        warnings.extend(splits.warnings)
        by_season.append(persons_from_frame(splits.data))
    talent: list[NbaPerson] = []
    if darko is not None:
        try:
            fetched_talent = darko.projections(**options)
        except SourceError as exc:
            warnings.append(f"{darko.name}: unavailable, matching without it ({exc})")
            failed = True
        else:
            inputs.append(fetched_talent)
            warnings.extend(fetched_talent.warnings)
            talent = persons_from_darko(fetched_talent.data)
    persons = merge_persons(by_season[0], talent, *by_season[1:], persons_from_static())
    as_of = min((fetched.as_of for fetched in inputs), default=stats.clock())
    crosswalk = build_nba_crosswalk(players, persons, load_nba_overrides(overrides), as_of=as_of)
    return Fetched(
        crosswalk,
        as_of,
        "+".join(dict.fromkeys(fetched.source for fetched in inputs)) or stats.name,
        "crosswalk",
        "+".join(labels),
        cached=bool(inputs) and all(fetched.cached for fetched in inputs),
        stale=any(fetched.stale for fetched in inputs),
        degraded=failed or any(fetched.degraded for fetched in inputs),
        warnings=(*warnings, *crosswalk.warnings),
        raw_path=inputs[0].raw_path if inputs else None,
    )


# --- the unmapped-rostered-player gate --------------------------------------------------------------------------------


def check_nba_rostered(
    store: Store,
    league_id: int,
    *,
    crosswalk: NbaCrosswalk | None = None,
    scoring_period_id: int | None = None,
) -> set[int]:
    """The unmapped-rostered-player gate for an NBA league: every player on any roster in the league must have an
    nba.com id, or :class:`UnmappedNbaPlayersError` names the offenders (with names from ``players`` and the reason
    the match failed, when ``crosswalk`` is a fresh build).

    Checks the latest roster snapshot unless ``scoring_period_id`` is given, against ``crosswalk`` or the one saved in
    the store. Returns the ESPN ids checked; empty before the first roster snapshot.

    This is the NBA crosswalk's gate only, as :func:`fm.model.ids.check_rostered` is the NFL one's: a league of another
    sport raises :class:`CrosswalkError` and a league id the store does not know raises ``LookupError``, both before
    any roster is read, so neither can pass vacuously.
    """
    league = store.leagues.get(league_id)
    if league is None:
        raise LookupError(f"no league with id {league_id} in the store")
    if league.sport != NBA_SPORT:
        raise CrosswalkError(
            f"league {league_id} ({league.key} {league.season}) is an {league.sport} league; the {NBA_SPORT} "
            f"crosswalk gates {NBA_SPORT} leagues only, so check it with the {league.sport} crosswalk"
        )
    period = scoring_period_id if scoring_period_id is not None else store.rosters.latest_period(league_id)
    if period is None:
        return set()
    rostered = store.rosters.rostered_ids(league_id, period)
    if not rostered:
        return rostered
    walk = crosswalk if crosswalk is not None else NbaCrosswalk.from_store(store)
    walk.require_mapped(
        rostered,
        players=store.players.many(NBA_SPORT, rostered),
        context=f"rostered players in league {league_id} (scoring period {period})",
    )
    return rostered
