"""The sync job: pull league state from ESPN and the external sources into the store (DESIGN sections 5, 7, 13).

For every configured league, :func:`sync_league` reads the views in :mod:`fm.espn.client` and writes what they say:

- ``mSettings`` -> the ``leagues`` row (named from the settings) and ``league_settings`` (the parsed settings as JSON,
  pointing at the raw capture it came from, the shape :func:`fm.proposals.policy.stored_settings` reads back);
- ``mTeam`` + ``mStandings`` -> ``teams`` (records, seeds, waiver ranks, FAAB spent);
- ``mRoster`` for the current scoring period -> ``roster_snapshots``, each team's rows replaced together;
- ``kona_player_info`` -> the top of the wire by ownership (``pool_size`` players) and ``kona_playercard`` for every
  rostered and pooled player -> ``players`` (position and pro team decoded through :mod:`fm.espn.ids`) and
  ``projections``: ESPN's projected and actual stat lines for the period and the season, keyed by the sport's stat
  abbreviations (:class:`fm.sports.base.StatSchema`), never points. An empty line is kept: ESPN projects one for a
  player on bye, and that is a projection of zero, not a missing one;
- every :class:`fm.espn.client.EspnRead` capture -> a ``raw_snapshots`` row. The client never writes to the store; this
  job is what indexes its captures.

All of a league's reads happen first and its rows are then written in one transaction, so a failed read leaves the
previous sync in place rather than half of a new one. The current scoring period is ESPN's (``scoringPeriodId`` on
the settings, else on the teams view); nothing here assumes a weekday or a week number.

Then the sources and the unmapped-rostered-player gate, dispatched on each stored league's sport
(:attr:`fm.store.LeagueRow.sport`):

- NFL: the id crosswalk (:func:`fm.model.ids.fetch_crosswalk` over nflverse ``ff_playerids`` plus the overrides file)
  is saved to ``player_ids`` and :func:`fm.model.ids.check_rostered` runs for every NFL league. A failed gate is
  recorded on the league's :class:`LeagueSync` (``gate.error``) rather than raised, so the rest of the report still
  comes out; :attr:`SyncReport.ok` is ``False`` and ``fm sync`` exits 1 naming the players. The ESPN state stays
  stored, because it is correct: the fix is a row in ``data/id_overrides.csv`` and another sync. Sleeper's projections
  for each NFL league's current week are refreshed through the adapter's cache and written to ``projections`` as
  ``sleeper`` stat lines (ESPN ids through the crosswalk, ESPN abbreviations through
  :func:`fm.model.projections.sleeper_rows`), the inputs :func:`fm.model.projections.blend_period` blends with ESPN's.
- NBA: the NBA crosswalk (:func:`fm.model.ids_nba.fetch_nba_crosswalk` over stats.nba.com's player splits for this
  season and last, DARKO's talent sheet and ``nba_api``'s bundled table, plus ``data/id_overrides_nba.csv``) matches
  every NBA player this sync wrote, is saved to ``player_ids``, and :func:`fm.model.ids_nba.check_nba_rostered` runs
  for every NBA league, its failure recorded the same way. Each gate refuses a league of the other sport, which is why
  the dispatch exists.

The NBA crosswalk's sources degrade rather than fail: one that is down with no cached copy is a warning and makes
the build ``degraded``. A degraded build does not replace the crosswalk an earlier sync saved, and the gate then
checks the saved one; with none saved yet, it is saved, as the best there is. (The nflverse ID map has no such mode:
with nothing cached a failed download raises.)

Every source result is reported as a :class:`SourceSync` carrying the adapter's ``as_of``, ``cached``, ``stale`` and
``degraded`` flags and its ``warnings`` (for the crosswalk: the adapter's, then the build's). This is deliberate:
Sleeper's undocumented endpoints fail as an HTTP 200 full of junk rather than an error, so a quiet break has to be
visible in the sync output. A stale or degraded source is reported, not a failure. ``force`` and the other
:class:`fm.sources.base.FetchOptions` pass through to every source; ESPN is always read live.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Unpack

from fm.config import Config, League, Sport
from fm.espn.auth import EspnSession
from fm.espn.client import DEFAULT_POOL_LIMIT, PLAYER_CARD_BATCH, ClientOptions, EspnClient, EspnRead
from fm.espn.ids import IdMaps, ids_for
from fm.espn.models import Player, PlayersView, RostersView, Team, TeamsView
from fm.espn.settings import LeagueSettings
from fm.model.ids import Crosswalk, UnmappedPlayer, UnmappedPlayersError, check_rostered, fetch_crosswalk
from fm.model.ids_nba import NBA_SPORT, NbaCrosswalk, check_nba_rostered, fetch_nba_crosswalk
from fm.model.projections import sleeper_rows
from fm.sources.base import Fetched, FetchOptions
from fm.sources.darko import DarkoSource
from fm.sources.nba_stats import NbaStatsSource, nba_season
from fm.sources.nflverse import NflverseSource
from fm.sources.sleeper import SleeperSource
from fm.sports.base import StatSchema
from fm.store import (
    LeagueRow,
    LeagueSettingsRow,
    PlayerRow,
    ProjectionKind,
    ProjectionRow,
    RawSnapshotRow,
    RosterEntryRow,
    Store,
    TeamRow,
)

ESPN_SOURCE = "espn"
"""``source`` of the rows this job writes from ESPN: ``projections`` stat lines and ``raw_snapshots``."""
DEFAULT_POOL_SIZE = 100
"""Free agents (and players on waivers) kept per league, most-owned first; paged ``DEFAULT_POOL_LIMIT`` at a time."""
NFL_SPORT: Sport = "nfl"
KEPT_CROSSWALK = "degraded, so the crosswalk an earlier sync saved was kept"


class SyncError(RuntimeError):
    """The sync could not run as asked: an unknown league key, or ESPN left out something the job needs."""


# --- results ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateResult:
    """The unmapped-rostered-player gate for one league: how many rostered players it checked, and the failure."""

    checked: int
    error: UnmappedPlayersError | None = None

    @property
    def passed(self) -> bool:
        return self.error is None

    @property
    def unmapped(self) -> tuple[UnmappedPlayer, ...]:
        return () if self.error is None else self.error.unmapped


@dataclass(frozen=True, slots=True)
class LeagueSync:
    """What one league's sync read and wrote. ``league_id`` is the store's row id, ``espn_league_id`` ESPN's, and
    ``sport`` the stored row's, which picks the gate."""

    key: str
    sport: Sport
    league_id: int
    espn_league_id: int
    season: int
    scoring_period_id: int
    name: str | None
    as_of: datetime
    teams: int
    rosters: int
    """Teams whose roster snapshot was written."""
    rostered: int
    """Distinct players on those rosters."""
    players: int
    projections: int
    snapshots: int
    warnings: tuple[str, ...] = ()
    gate: GateResult | None = None
    """The league's unmapped-rostered-player gate: ``None`` until :func:`sync` runs it (:func:`sync_league` alone
    does not)."""
    player_ids: tuple[int, ...] = ()
    """ESPN ids of every ``players`` row written (rostered, then the pool): who the sport's crosswalk must cover."""

    @property
    def ok(self) -> bool:
        return self.gate is None or self.gate.passed

    def describe(self) -> str:
        """One line: ``nfl: Fixture League (ESPN 1234567, 2026), scoring period 4: 10 teams, ...``."""
        name = self.name or "league"
        return (
            f"{self.key}: {name} (ESPN {self.espn_league_id}, {self.season}), scoring period {self.scoring_period_id}: "
            f"{self.teams} teams, {self.rostered} rostered on {self.rosters} rosters, {self.players} players, "
            f"{self.projections} stat lines, {self.snapshots} captures"
        )


@dataclass(frozen=True, slots=True)
class SourceSync:
    """One source dataset as the sync saw it: the adapter's provenance flags and warnings, plus what was stored."""

    source: str
    dataset: str
    key: str
    as_of: datetime
    cached: bool
    stale: bool
    degraded: bool
    warnings: tuple[str, ...]
    stored: int = 0
    """Rows written to the store from this dataset."""
    detail: str = ""
    """What the dataset held, for the report (``60 id mappings saved``)."""

    @classmethod
    def from_fetched(
        cls, fetched: Fetched[Any], *, stored: int = 0, detail: str = "", warnings: Iterable[str] = ()
    ) -> SourceSync:
        """``warnings`` (what storing the data found) follow the adapter's own."""
        return cls(
            source=fetched.source,
            dataset=fetched.dataset,
            key=fetched.key,
            as_of=fetched.as_of,
            cached=fetched.cached,
            stale=fetched.stale,
            degraded=fetched.degraded,
            warnings=(*fetched.warnings, *warnings),
            stored=stored,
            detail=detail,
        )

    @property
    def fresh(self) -> bool:
        """Downloaded in this sync and usable."""
        return not (self.cached or self.stale or self.degraded)

    @property
    def state(self) -> str:
        if self.degraded:
            return "DEGRADED (nothing usable; see warnings)"
        if self.stale:
            return "STALE (refresh failed, serving the last good copy)"
        return "cached" if self.cached else "fresh"

    def describe(self) -> str:
        """One line: ``nflverse/ff_playerids[all]: as of 2026-10-04 15:00 UTC, fresh; 60 id mappings saved``."""
        line = f"{self.source}/{self.dataset}[{self.key}]: as of {self.as_of.astimezone(UTC):%Y-%m-%d %H:%M} UTC, "
        line += self.state
        if self.detail:
            line += f"; {self.detail}"
        return line


@dataclass(frozen=True, slots=True)
class SyncReport:
    leagues: tuple[LeagueSync, ...]
    sources: tuple[SourceSync, ...]

    @property
    def ok(self) -> bool:
        """Every gate passed. A stale or degraded source is reported, not a failure: the engine runs on what it has."""
        return all(league.ok for league in self.leagues)

    @property
    def failed(self) -> tuple[LeagueSync, ...]:
        """The leagues whose gate failed."""
        return tuple(league for league in self.leagues if not league.ok)

    @property
    def gate_errors(self) -> tuple[UnmappedPlayersError, ...]:
        return tuple(league.gate.error for league in self.leagues if league.gate is not None and league.gate.error)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Every league and source warning, in report order."""
        found = [warning for league in self.leagues for warning in league.warnings]
        found.extend(warning for source in self.sources for warning in source.warnings)
        return tuple(found)


# --- one league -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LeagueReads:
    """Every ESPN response a league sync needs, read before anything is written."""

    settings: EspnRead[LeagueSettings]
    teams: EspnRead[TeamsView]
    scoring_period_id: int
    rosters: EspnRead[RostersView]
    pool: tuple[EspnRead[PlayersView], ...]
    cards: tuple[EspnRead[PlayersView], ...]

    @property
    def others(self) -> tuple[EspnRead[Any], ...]:
        """Every read but the settings, in request order."""
        return (self.teams, self.rosters, *self.pool, *self.cards)


def sync_league(store: Store, league: League, client: EspnClient, *, pool_size: int = DEFAULT_POOL_SIZE) -> LeagueSync:
    """Read one league's state through ``client`` and write it to the store in one transaction. No sources, no gate."""
    if pool_size < 0:
        raise ValueError(f"pool_size must be >= 0, got {pool_size!r}")
    reads = _read_league(league, client, pool_size)
    with store.db.transaction():
        return _write_league(store, league, reads)


def _read_league(league: League, client: EspnClient, pool_size: int) -> _LeagueReads:
    settings = client.settings()
    teams = client.teams()
    period = settings.data.current_scoring_period
    if period is None:
        period = teams.data.scoring_period_id
    if period is None:
        raise SyncError(
            f"{league.key}: ESPN did not report the current scoring period for league {league.espn_league_id}"
        )
    rosters = client.rosters(period)

    pool: list[EspnRead[PlayersView]] = []
    offset = 0
    while offset < pool_size:
        limit = min(DEFAULT_POOL_LIMIT, pool_size - offset)
        page = client.free_agents(period, limit=limit, offset=offset)
        pool.append(page)
        if len(page.data.players) < limit:
            break
        offset += limit

    ids: dict[int, None] = {}  # insertion-ordered set: rostered players first, then the pool
    for roster in rosters.data.teams:
        ids.update(dict.fromkeys(roster.player_ids))
    for page in pool:
        ids.update(dict.fromkeys(page.data.player_ids))
    cards = [client.player_cards(batch, scoring_period=period) for batch in _batches(list(ids), PLAYER_CARD_BATCH)]
    return _LeagueReads(settings, teams, period, rosters, tuple(pool), tuple(cards))


def _write_league(store: Store, league: League, reads: _LeagueReads) -> LeagueSync:
    settings = reads.settings.data
    period = reads.scoring_period_id
    warnings: list[str] = []

    row = store.leagues.upsert(
        LeagueRow(
            key=league.key,
            sport=league.sport,
            espn_league_id=league.espn_league_id,
            season=league.season,
            team_id=league.team_id,
            name=settings.name or None,
            as_of=reads.settings.as_of,
        )
    )
    league_id = row.row_id
    sport = row.sport
    ids = ids_for(sport)
    schema = StatSchema.for_game(sport)
    settings_snapshot = _index_capture(store, reads.settings, league_id)
    snapshots = int(settings_snapshot is not None)
    snapshots += sum(_index_capture(store, read, league_id) is not None for read in reads.others)
    store.settings.upsert(
        LeagueSettingsRow(
            league_id=league_id,
            settings=settings.model_dump(mode="json"),
            raw_snapshot_id=settings_snapshot.id if settings_snapshot is not None else None,
            as_of=reads.settings.as_of,
        )
    )

    teams = reads.teams.data
    store.teams.upsert_many(_team_row(league_id, team, reads.teams.as_of) for team in teams.teams)
    known = sorted(team.id for team in teams.teams)
    if league.team_id not in known:
        warnings.append(
            f"team {league.team_id} is not in ESPN league {league.espn_league_id} (teams: "
            f"{', '.join(str(team) for team in known) or 'none'}); check team_id in config.toml"
        )

    players: dict[int, PlayerRow] = {}
    rostered: set[int] = set()
    for roster in reads.rosters.data.teams:
        entries = [
            RosterEntryRow(
                league_id=league_id,
                scoring_period_id=period,
                team_id=roster.team_id,
                espn_id=entry.player_id,
                lineup_slot_id=entry.lineup_slot_id,
                acquisition_type=entry.acquisition_type,
                acquisition_date=entry.acquisition_date,
                lineup_locked=entry.lineup_locked,
                as_of=reads.rosters.as_of,
            )
            for entry in roster.entries
        ]
        store.rosters.replace(league_id, period, roster.team_id, entries)
        for entry in roster.entries:
            players[entry.player_id] = _player_row(sport, entry.player, ids, reads.rosters.as_of)
            rostered.add(entry.player_id)
    for page in reads.pool:
        for entry in page.data.players:
            players[entry.id] = _player_row(sport, entry.player, ids, page.as_of)

    projections = 0
    for cards in reads.cards:
        lines: list[ProjectionRow] = []
        for entry in cards.data.players:
            players[entry.id] = _player_row(sport, entry.player, ids, cards.as_of)
            lines.extend(_projection_rows(sport, entry.player, league.season, period, schema, cards.as_of))
        projections += store.projections.upsert_many(lines)
    store.players.upsert_many(players.values())

    return LeagueSync(
        key=league.key,
        sport=sport,
        league_id=league_id,
        espn_league_id=league.espn_league_id,
        season=league.season,
        scoring_period_id=period,
        name=settings.name or None,
        as_of=reads.settings.as_of,
        teams=len(teams.teams),
        rosters=len(reads.rosters.data.teams),
        rostered=len(rostered),
        players=len(players),
        projections=projections,
        snapshots=snapshots,
        warnings=tuple(warnings),
        player_ids=tuple(players),
    )


def _index_capture(store: Store, read: EspnRead[Any], league_id: int | None) -> RawSnapshotRow | None:
    """The ``raw_snapshots`` row for a read's capture (``None`` when the client captured nothing)."""
    capture = read.capture
    if capture is None:
        return None
    return store.raw_snapshots.insert(
        RawSnapshotRow(
            source=ESPN_SOURCE,
            kind=capture.kind,
            league_id=league_id,
            scoring_period_id=capture.scoring_period_id,
            url=capture.url,
            params=capture.params,
            path=capture.relative_path,
            sha256=capture.sha256,
            size_bytes=capture.size_bytes,
            status_code=capture.status_code,
            fetched_at=capture.fetched_at,
        )
    )


def _team_row(league_id: int, team: Team, as_of: datetime) -> TeamRow:
    overall = team.record.overall
    return TeamRow(
        league_id=league_id,
        team_id=team.id,
        name=team.display_name,
        abbrev=team.abbrev,
        division_id=team.division_id,
        wins=overall.wins,
        losses=overall.losses,
        ties=overall.ties,
        points_for=overall.points_for,
        points_against=overall.points_against,
        playoff_seed=team.playoff_seed,
        waiver_rank=team.waiver_rank,
        acquisition_budget_spent=team.transaction_counter.acquisition_budget_spent,
        as_of=as_of,
    )


def _player_row(sport: Sport, player: Player, ids: IdMaps, as_of: datetime) -> PlayerRow:
    name = player.full_name or " ".join(part for part in (player.first_name, player.last_name) if part)
    position_id = player.default_position_id
    pro_team_id = player.pro_team_id
    return PlayerRow(
        sport=sport,
        espn_id=player.id,
        full_name=name or f"Player {player.id}",
        default_position_id=position_id,
        position=ids.position_label(position_id) if position_id is not None else None,
        pro_team_id=pro_team_id,
        pro_team=ids.pro_team(pro_team_id) if pro_team_id is not None else None,
        eligible_slot_ids=list(player.eligible_slots),
        injury_status=player.injury_status,
        injured=player.injured,
        active=player.active,
        as_of=as_of,
    )


def _projection_rows(
    sport: Sport, player: Player, season: int, period: int, schema: StatSchema, as_of: datetime
) -> list[ProjectionRow]:
    """ESPN's projected and actual lines for the period and the season (``0``), as abbreviation-keyed stat lines.

    A line ESPN sends empty is kept (a bye projects zero); a line it does not send is absent. A period's line is the
    game's "Game" split (:data:`fm.espn.models.STAT_SPLIT_GAME`: 1 in ``ffl``, 5 in ``fba``).
    """
    rows: list[ProjectionRow] = []
    kinds: tuple[tuple[ProjectionKind, bool], ...] = (("projected", True), ("actual", False))
    for kind, projected in kinds:
        for scoring_period in (period, 0):
            line = player.stat_entry(
                season=season, scoring_period=scoring_period, projected=projected, game=schema.game
            )
            if line is None:
                continue
            rows.append(
                ProjectionRow(
                    sport=sport,
                    espn_id=player.id,
                    source=ESPN_SOURCE,
                    kind=kind,
                    season=season,
                    scoring_period_id=scoring_period,
                    stats=schema.from_espn(line.stats),
                    as_of=as_of,
                )
            )
    return rows


def _batches[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --- the whole job ----------------------------------------------------------------------------------------------------


def sync(
    store: Store,
    config: Config,
    *,
    session: EspnSession | None,
    leagues: Iterable[str] | None = None,
    nflverse: NflverseSource | None = None,
    sleeper: SleeperSource | None = None,
    nba_stats: NbaStatsSource | None = None,
    darko: DarkoSource | None = None,
    espn_options: ClientOptions | None = None,
    pool_size: int = DEFAULT_POOL_SIZE,
    overrides: Path | None = None,
    nba_overrides: Path | None = None,
    **fetch: Unpack[FetchOptions],
) -> SyncReport:
    """Sync every configured league (or the ``leagues`` keys given), then the sources and the gate for each sport.

    ``session`` carries the ESPN cookies (``None`` reads public leagues only). ``nflverse``, ``sleeper``,
    ``nba_stats`` and ``darko`` default to fresh adapters over the cache dir; ``espn_options`` are passed to each
    :class:`EspnClient`; ``overrides`` and ``nba_overrides`` are the crosswalk overrides files
    (``data/id_overrides.csv`` and ``data/id_overrides_nba.csv`` by default); ``fetch`` (``force``, ``max_age``,
    ``fresh_since``) goes to every source. ESPN, crosswalk and source failures raise (a league already synced stays
    synced); a failed gate is recorded in the report, whose ``ok`` is then ``False``.
    """
    if pool_size < 0:
        raise ValueError(f"pool_size must be >= 0, got {pool_size!r}")
    selected = select_leagues(config, leagues)
    synced: list[LeagueSync] = []
    options: ClientOptions = espn_options if espn_options is not None else {}
    for league in selected:
        with EspnClient.for_league(league, session, **options) as client:
            synced.append(sync_league(store, league, client, pool_size=pool_size))

    sources: list[SourceSync] = []
    if any(item.sport == NFL_SPORT for item in synced):
        synced, nfl_sources = _sync_nfl_sources(store, synced, nflverse, sleeper, overrides, fetch)
        sources.extend(nfl_sources)
    if any(item.sport == NBA_SPORT for item in synced):
        synced, nba_sources = _sync_nba_sources(store, synced, nba_stats, darko, nba_overrides, fetch)
        sources.extend(nba_sources)
    return SyncReport(tuple(synced), tuple(sources))


def select_leagues(config: Config, keys: Iterable[str] | None) -> tuple[League, ...]:
    """The configured leagues under ``keys`` (one key may be a bare string), once each in the order given; every
    league when ``keys`` is ``None``. An unknown key raises :class:`SyncError` naming the known ones."""
    if keys is None:
        return config.leagues
    chosen: list[League] = []
    for key in (keys,) if isinstance(keys, str) else keys:
        try:
            league = config.league(key)
        except KeyError as exc:
            raise SyncError(exc.args[0]) from None
        if league not in chosen:
            chosen.append(league)
    return tuple(chosen)


def _sync_nfl_sources(
    store: Store,
    synced: Sequence[LeagueSync],
    nflverse: NflverseSource | None,
    sleeper: SleeperSource | None,
    overrides: Path | None,
    fetch: FetchOptions,
) -> tuple[list[LeagueSync], list[SourceSync]]:
    """The NFL crosswalk (saved, then the gate for each NFL league) and Sleeper's projections for each current week,
    stored as ``sleeper`` stat lines."""
    sources: list[SourceSync] = []
    crosswalk = fetch_crosswalk(nflverse if nflverse is not None else NflverseSource(), overrides=overrides, **fetch)
    stored = crosswalk.data.save(store)
    sources.append(SourceSync.from_fetched(crosswalk, stored=stored, detail=f"{stored} id mappings saved"))
    gated = [_gate_nfl(store, item, crosswalk.data) if item.sport == NFL_SPORT else item for item in synced]

    weeks = sorted({(item.season, item.scoring_period_id) for item in synced if item.sport == NFL_SPORT})
    source = sleeper if sleeper is not None else SleeperSource()
    try:
        for season, week in weeks:
            lines = source.projections(season, week, **fetch)
            converted = sleeper_rows(lines.data, crosswalk.data, as_of=lines.as_of)
            saved = store.projections.upsert_many(converted.rows)
            detail = f"{len(lines.data)} stat lines, {saved} stored"
            sources.append(SourceSync.from_fetched(lines, stored=saved, detail=detail, warnings=converted.warnings))
    finally:
        if sleeper is None:
            source.close()
    return gated, sources


def _gate_nfl(store: Store, item: LeagueSync, crosswalk: Crosswalk) -> LeagueSync:
    """:func:`fm.model.ids.check_rostered` for an NFL league, its failure recorded on the result rather than raised."""
    try:
        checked = check_rostered(store, item.league_id, crosswalk=crosswalk, scoring_period_id=item.scoring_period_id)
    except UnmappedPlayersError as exc:
        return replace(item, gate=GateResult(checked=item.rostered, error=exc))
    return replace(item, gate=GateResult(checked=len(checked)))


def _sync_nba_sources(
    store: Store,
    synced: Sequence[LeagueSync],
    nba_stats: NbaStatsSource | None,
    darko: DarkoSource | None,
    overrides: Path | None,
    fetch: FetchOptions,
) -> tuple[list[LeagueSync], list[SourceSync]]:
    """The NBA crosswalk over every NBA player this sync wrote (saved), then the gate for each NBA league.

    The persons come from stats.nba.com's splits for the latest NBA league's season and the one before (before opening
    night the current season is empty), DARKO's talent sheet and ``nba_api``'s bundled table. With no NBA player
    written there is nothing to match, and the gates check the saved crosswalk.
    """
    leagues = [item for item in synced if item.sport == NBA_SPORT]
    players = store.players.many(NBA_SPORT, {espn_id for item in leagues for espn_id in item.player_ids})
    sources: list[SourceSync] = []
    walk: NbaCrosswalk | None = None
    if players:
        season = max(item.season for item in leagues)
        stats = nba_stats if nba_stats is not None else NbaStatsSource()
        talent = darko if darko is not None else DarkoSource()
        try:
            seasons = (nba_season(season), nba_season(season - 1))
            crosswalk = fetch_nba_crosswalk(players, stats, seasons, darko=talent, overrides=overrides, **fetch)
        finally:
            if darko is None:
                talent.close()
        kept = _keeps_saved_crosswalk(store, NBA_SPORT, crosswalk)
        stored = 0 if kept else crosswalk.data.save(store)
        detail = KEPT_CROSSWALK if kept else f"{stored} id mappings saved"
        sources.append(SourceSync.from_fetched(crosswalk, stored=stored, detail=detail))
        walk = None if kept else crosswalk.data
    gated = [_gate_nba(store, item, walk) if item.sport == NBA_SPORT else item for item in synced]
    return gated, sources


def _gate_nba(store: Store, item: LeagueSync, crosswalk: NbaCrosswalk | None) -> LeagueSync:
    """:func:`fm.model.ids_nba.check_nba_rostered` for an NBA league (against ``crosswalk``, else the saved one), its
    failure recorded on the result rather than raised."""
    try:
        checked = check_nba_rostered(
            store, item.league_id, crosswalk=crosswalk, scoring_period_id=item.scoring_period_id
        )
    except UnmappedPlayersError as exc:
        return replace(item, gate=GateResult(checked=item.rostered, error=exc))
    return replace(item, gate=GateResult(checked=len(checked)))


def _keeps_saved_crosswalk(store: Store, sport: Sport, crosswalk: Fetched[Any]) -> bool:
    """A degraded build (a source down with no cached copy) leaves the crosswalk an earlier sync saved in place."""
    return crosswalk.degraded and bool(store.player_ids.for_sport(sport))
