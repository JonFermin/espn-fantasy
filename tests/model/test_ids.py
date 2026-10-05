"""NFL player ID crosswalk: ff_playerids mapping, overrides, team defenses, the store round trip, the rostered gate.

The ID map is the recorded DynastyProcess file under tests/fixtures/sources/nflverse (14 mapped players and one row
with no ids), parsed the way nflreadpy parses it. No network.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from fm.model.ids import (
    DEFAULT_OVERRIDES,
    GSIS,
    ORIGIN_FF_PLAYERIDS,
    ORIGIN_OVERRIDE,
    ORIGIN_TEAM_DEFENSE,
    SLEEPER,
    SOURCES,
    Crosswalk,
    CrosswalkError,
    Override,
    UnmappedPlayer,
    UnmappedPlayersError,
    build_crosswalk,
    check_rostered,
    fetch_crosswalk,
    is_team_defense,
    load_overrides,
    parse_overrides,
    sources_for,
    team_code,
    team_defense_id,
    team_defense_pro_team_id,
)
from fm.sources.base import RateLimiter
from fm.sources.nflverse import NflreadLoader, NflverseSource
from fm.store import LeagueRow, PlayerIdRow, PlayerRow, RosterEntryRow, Sport, Store, TeamRow

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sources" / "nflverse" / "db_playerids.csv"
AS_OF = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
TEAM_DEFENSES = 32
NO_ESPN_ID_WARNING = "ff_playerids: 1 of 15 rows have no ESPN id (Trinidad Chambliss)"

ALLEN, MAHOMES, GIBBS = 3918298, 3139477, 4429795
NOBODY = 999999  # an ESPN id no ID map knows
EAGLES_DST = -16021  # ESPN pro team 21
JOKIC = 3112335  # an NBA player's ESPN id


def ff_playerids() -> pl.DataFrame:
    """The recorded DynastyProcess file, parsed the way nflreadpy parses CSV."""
    return pl.read_csv(FIXTURE, null_values=["NA", "NULL", ""])


def player(espn_id: int, name: str, position: str | None = None, pro_team: str | None = None) -> PlayerRow:
    return PlayerRow(sport="nfl", espn_id=espn_id, full_name=name, position=position, pro_team=pro_team, as_of=AS_OF)


@pytest.fixture
def walk() -> Crosswalk:
    return build_crosswalk(ff_playerids(), as_of=AS_OF)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


# --- mapping ---


def test_maps_espn_to_gsis_and_sleeper_from_ff_playerids(walk: Crosswalk) -> None:
    assert walk.gsis_id(ALLEN) == "00-0034857" and walk.sleeper_id(ALLEN) == "4984"
    assert walk.espn_id(GSIS, "00-0034857") == ALLEN and walk.espn_id(SLEEPER, "4984") == ALLEN
    assert walk.espn_id(SLEEPER, 4984) == ALLEN  # ints straight out of a frame work too
    assert walk.ids_for(MAHOMES) == {GSIS: "00-0033873", SLEEPER: "4046"}
    assert walk.ids_for(NOBODY) == {} and walk.gsis_id(NOBODY) is None and walk.espn_id(SLEEPER, "0") is None
    assert len(walk) == 14 * 2 + TEAM_DEFENSES
    assert len(walk.mapped(GSIS)) == 14 and len(walk.mapped(SLEEPER)) == 14 + TEAM_DEFENSES
    assert walk.mapped() == walk.mapped(SLEEPER)
    allen_gsis = next(row for row in walk.rows if row.espn_id == ALLEN and row.source == GSIS)
    assert allen_gsis == PlayerIdRow(
        sport="nfl", espn_id=ALLEN, source=GSIS, source_id="00-0034857", origin=ORIGIN_FF_PLAYERIDS, as_of=AS_OF
    )
    assert [(row.espn_id, row.source) for row in walk.rows] == sorted((row.espn_id, row.source) for row in walk.rows)


def test_rows_without_an_espn_id_are_reported_not_mapped(walk: Crosswalk) -> None:
    assert walk.warnings == (NO_ESPN_ID_WARNING,)
    assert walk.espn_id(GSIS, "") is None


def test_ids_are_normalised_from_ints_floats_and_text() -> None:
    data = pl.DataFrame(
        {
            "espn_id": [4984.0, 4046.0, None],
            "gsis_id": [" 00-0034857 ", "NA", "00-0000001"],
            "sleeper_id": ["4984", "", "7"],
            "name": ["Josh Allen", "Patrick Mahomes", "Nobody"],
        }
    )
    walk = build_crosswalk(data, as_of=AS_OF)
    assert walk.ids_for(4984) == {GSIS: "00-0034857", SLEEPER: "4984"}
    assert walk.ids_for(4046) == {}
    assert walk.espn_id(GSIS, "00-0000001") is None
    assert walk.warnings == ("ff_playerids: 1 of 3 rows have no ESPN id (Nobody)",)


def test_conflicting_rows_keep_the_first_and_warn() -> None:
    data = pl.DataFrame(
        {
            "espn_id": [1, 1, 2, 3],
            "gsis_id": ["00-0000001", "00-0000009", None, None],
            "sleeper_id": ["10", None, "10", "30"],
            "name": ["A", "A again", "B", "C"],
        }
    )
    walk = build_crosswalk(data, as_of=AS_OF)
    assert walk.gsis_id(1) == "00-0000001" and walk.espn_id(SLEEPER, "10") == 1
    assert walk.sleeper_id(2) is None and walk.sleeper_id(3) == "30"
    assert walk.warnings == (
        "ff_playerids: ESPN 1 has gsis ids 00-0000001 and 00-0000009; kept 00-0000001",
        "ff_playerids: sleeper id 10 is listed for ESPN 1 and ESPN 2; kept ESPN 1",
    )


def test_missing_columns_raise() -> None:
    with pytest.raises(CrosswalkError, match="missing columns gsis_id, sleeper_id"):
        build_crosswalk(pl.DataFrame({"espn_id": [1]}), as_of=AS_OF)


def test_team_defenses_are_derived_for_sleeper_only(walk: Crosswalk) -> None:
    assert team_defense_id(21) == EAGLES_DST and team_defense_pro_team_id(EAGLES_DST) == 21
    assert is_team_defense(EAGLES_DST) and not is_team_defense(ALLEN) and not is_team_defense(-16000)
    assert sources_for(EAGLES_DST) == (SLEEPER,) and sources_for(ALLEN) == SOURCES
    assert walk.sleeper_id(EAGLES_DST) == "PHI" and walk.gsis_id(EAGLES_DST) is None
    assert walk.sleeper_id(team_defense_id(28)) == "WAS"  # ESPN spells Washington WSH, Sleeper WAS
    assert walk.espn_id(SLEEPER, "HOU") == team_defense_id(34)
    defenses = [row for row in walk.rows if row.origin == ORIGIN_TEAM_DEFENSE]
    assert len(defenses) == TEAM_DEFENSES and all(row.source == SLEEPER for row in defenses)
    assert walk.unmapped([EAGLES_DST]) == []


def test_team_codes_follow_each_source() -> None:
    assert team_code("WSH", "sleeper") == "WAS" and team_code("WSH", "nflverse") == "WAS"
    assert team_code("LAR", "nflverse") == "LA" and team_code("LAR", "sleeper") == "LAR"
    assert team_code("KC", "nflverse") == "KC"


def test_crosswalk_rejects_inconsistent_rows() -> None:
    def row(espn_id: int, source: str, source_id: str, sport: Sport = "nfl") -> PlayerIdRow:
        return PlayerIdRow(sport=sport, espn_id=espn_id, source=source, source_id=source_id, origin="test", as_of=AS_OF)

    with pytest.raises(CrosswalkError, match="two sleeper ids"):
        Crosswalk([row(1, SLEEPER, "1"), row(1, SLEEPER, "2")])
    with pytest.raises(CrosswalkError, match="maps to ESPN 1 and ESPN 2"):
        Crosswalk([row(1, SLEEPER, "1"), row(2, SLEEPER, "1")])
    with pytest.raises(CrosswalkError, match="must be nfl"):
        Crosswalk([row(1, "nba", "1", sport="nba")])


# --- overrides ---


def test_parse_overrides_skips_comments_and_blank_lines() -> None:
    text = (
        "# reviewed by hand\n\nespn_id,source,source_id,name,note\n"
        f"{NOBODY},sleeper,13999,Nobody Special,ff_playerids lacks him\n"
        f"{GIBBS},gsis,,Jahmyr Gibbs,drop a wrong id\n"
    )
    assert parse_overrides(text) == (
        Override(NOBODY, SLEEPER, "13999", "Nobody Special", "ff_playerids lacks him"),
        Override(GIBBS, GSIS, None, "Jahmyr Gibbs", "drop a wrong id"),
    )
    assert parse_overrides("# nothing yet\n") == ()


def test_override_adds_a_missing_mapping() -> None:
    walk = build_crosswalk(ff_playerids(), [Override(NOBODY, SLEEPER, "13999", "Nobody Special")], as_of=AS_OF)
    assert walk.sleeper_id(NOBODY) == "13999" and walk.espn_id(SLEEPER, "13999") == NOBODY
    row = next(row for row in walk.rows if row.espn_id == NOBODY)
    assert row.origin == ORIGIN_OVERRIDE and row.as_of == AS_OF
    assert walk.unmapped([NOBODY]) == [UnmappedPlayer(NOBODY, (GSIS,))]
    assert walk.warnings == (NO_ESPN_ID_WARNING,)


def test_override_repoints_a_source_id_to_one_owner() -> None:
    # ff_playerids has Sleeper 9221 as Gibbs; an override says it is really NOBODY. A source id has one owner.
    walk = build_crosswalk(ff_playerids(), [Override(NOBODY, SLEEPER, "9221")], as_of=AS_OF)
    assert walk.espn_id(SLEEPER, "9221") == NOBODY
    assert walk.sleeper_id(GIBBS) is None and walk.gsis_id(GIBBS) == "00-0039139"
    assert walk.unmapped([GIBBS]) == [UnmappedPlayer(GIBBS, (SLEEPER,))]


def test_override_replaces_a_players_id_for_a_source() -> None:
    walk = build_crosswalk(ff_playerids(), [Override(GIBBS, SLEEPER, "9999")], as_of=AS_OF)
    assert walk.sleeper_id(GIBBS) == "9999" and walk.espn_id(SLEEPER, "9221") is None


def test_override_with_an_empty_id_removes_the_mapping() -> None:
    walk = build_crosswalk(ff_playerids(), [Override(GIBBS, GSIS, None)], as_of=AS_OF)
    assert walk.gsis_id(GIBBS) is None and walk.espn_id(GSIS, "00-0039139") is None
    assert walk.sleeper_id(GIBBS) == "9221"


def test_overrides_that_change_nothing_are_flagged() -> None:
    walk = build_crosswalk(
        ff_playerids(), [Override(ALLEN, SLEEPER, "4984"), Override(NOBODY, GSIS, None)], as_of=AS_OF
    )
    assert walk.sleeper_id(ALLEN) == "4984"
    assert walk.warnings == (
        NO_ESPN_ID_WARNING,
        f"override: ESPN {ALLEN} sleeper = 4984 is now in ff_playerids; the row can go",
        f"override: ESPN {NOBODY} gsis removes nothing; the row can go",
    )


def test_override_beats_the_derived_team_defense() -> None:
    walk = build_crosswalk(ff_playerids(), [Override(EAGLES_DST, SLEEPER, "PHL")], as_of=AS_OF)
    assert walk.sleeper_id(EAGLES_DST) == "PHL" and walk.espn_id(SLEEPER, "PHI") is None
    assert len([row for row in walk.rows if row.origin == ORIGIN_TEAM_DEFENSE]) == TEAM_DEFENSES - 1


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("x,sleeper,1,,", "espn_id 'x' is not an integer"),
        ("0,sleeper,1,,", "espn_id must not be 0"),
        ("1,yahoo,1,,", "unknown source 'yahoo'; expected one of gsis, sleeper"),
        ("1,gsis,34857,,", "'34857' does not look like a gsis id"),
        ("1,sleeper,phi,,", "'phi' does not look like a sleeper id"),
        ("1,sleeper,1,,,extra", "expected 5 fields, got 6"),
    ],
)
def test_bad_override_lines_name_the_line(line: str, message: str) -> None:
    text = f"# header next\nespn_id,source,source_id,name,note\n{line}\n"
    with pytest.raises(CrosswalkError, match=rf"^overrides\.csv:3: {re.escape(message)}"):
        parse_overrides(text, where="overrides.csv")


def test_duplicate_overrides_are_rejected() -> None:
    header = "espn_id,source,source_id,name,note\n"
    with pytest.raises(CrosswalkError, match="overrides:3: ESPN 1 sleeper already set on line 2"):
        parse_overrides(header + "1,sleeper,10,,\n1,sleeper,11,,\n")
    with pytest.raises(CrosswalkError, match="overrides:3: sleeper id 10 already used on line 2"):
        parse_overrides(header + "1,sleeper,10,,\n2,sleeper,10,,\n")


def test_header_must_carry_the_id_columns() -> None:
    with pytest.raises(CrosswalkError, match="overrides:1: header lacks source_id"):
        parse_overrides("espn_id,source,name\n")


def test_load_overrides_reads_the_file_or_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "ov.csv"
    path.write_text("espn_id,source,source_id,name,note\n7,gsis,00-0000007,Seven,\n", encoding="utf-8")
    assert load_overrides(path) == (Override(7, GSIS, "00-0000007", "Seven"),)
    with pytest.raises(CrosswalkError, match="overrides file not found"):
        load_overrides(tmp_path / "missing.csv")


def test_committed_overrides_file_is_well_formed() -> None:
    assert DEFAULT_OVERRIDES.is_file()
    overrides = load_overrides()
    assert all(override.source in SOURCES for override in overrides)
    build_crosswalk(ff_playerids(), overrides, as_of=AS_OF)  # and applies cleanly


# --- through the nflverse adapter ---


class FakeLoader:
    """Only ``load_ff_playerids``; anything else the adapter might call is a test failure."""

    def __init__(self, data: pl.DataFrame) -> None:
        self.data = data
        self.calls = 0
        self.fail_with: Exception | None = None

    def load_ff_playerids(self) -> pl.DataFrame:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return self.data

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"unexpected loader call {name}")


class FakeClock:
    def __init__(self) -> None:
        self.now = AS_OF

    def __call__(self) -> datetime:
        return self.now


def test_fetch_crosswalk_goes_through_the_nflverse_adapter(tmp_path: Path) -> None:
    loader = FakeLoader(ff_playerids())
    clock = FakeClock()
    source = NflverseSource(
        loader=cast(NflreadLoader, loader), cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock
    )
    overrides = tmp_path / "overrides.csv"
    overrides.write_text(f"espn_id,source,source_id,name,note\n{NOBODY},sleeper,13999,Nobody Special,\n", "utf-8")

    fetched = fetch_crosswalk(source, overrides=overrides)
    assert (fetched.source, fetched.dataset, fetched.as_of, fetched.cached, fetched.stale) == (
        "nflverse",
        "ff_playerids",
        AS_OF,
        False,
        False,
    )
    assert fetched.data.sleeper_id(NOBODY) == "13999" and fetched.data.gsis_id(ALLEN) == "00-0034857"
    assert fetched.warnings == fetched.data.warnings == (NO_ESPN_ID_WARNING,)
    assert all(row.as_of == AS_OF for row in fetched.data.rows)
    assert (tmp_path / "cache" / "nflverse" / "ff_playerids" / "all.parquet").is_file()

    clock.now += timedelta(hours=25)  # past the 24 h TTL
    loader.fail_with = ConnectionError("Failed to download: offline")
    stale = fetch_crosswalk(source, overrides=overrides)
    assert stale.stale is True and stale.as_of == AS_OF and loader.calls == 2
    assert "offline" in stale.warnings[0] and stale.warnings[1:] == fetched.warnings
    assert stale.data.rows == fetched.data.rows


# --- store round trip ---


def test_save_replaces_the_sports_rows_and_from_store_reads_them_back(store: Store, walk: Crosswalk) -> None:
    assert len(Crosswalk.from_store(store)) == 0
    nba = PlayerIdRow(sport="nba", espn_id=3112335, source="nba", source_id="1629029", origin="name_match", as_of=AS_OF)
    store.player_ids.upsert(nba)

    assert walk.save(store) == len(walk)
    assert store.player_ids.lookup("nfl", SLEEPER, "4984") == next(
        row for row in walk.rows if row.espn_id == ALLEN and row.source == SLEEPER
    )
    again = Crosswalk.from_store(store)
    assert again.rows == walk.rows and again.warnings == ()
    assert store.player_ids.for_player("nba", 3112335) == [nba]  # other sports untouched

    # Re-pointing a source id collides with the one-owner-per-id constraint under an upsert; save replaces instead.
    repointed = build_crosswalk(ff_playerids(), [Override(NOBODY, SLEEPER, "9221")], as_of=AS_OF)
    assert repointed.save(store) == len(repointed)
    found = store.player_ids.lookup("nfl", SLEEPER, "9221")
    assert found is not None and found.espn_id == NOBODY and found.origin == ORIGIN_OVERRIDE
    assert store.player_ids.unmapped("nfl", SLEEPER, [GIBBS, ALLEN]) == {GIBBS}


# --- the unmapped-rostered-player gate ---


def seed_roster(store: Store, *espn_ids: int, period: int = 4) -> LeagueRow:
    league = store.leagues.upsert(
        LeagueRow(key="nfl", sport="nfl", espn_league_id=123456, season=2026, team_id=4, as_of=AS_OF)
    )
    store.teams.upsert(TeamRow(league_id=league.row_id, team_id=4, name="Team 4", as_of=AS_OF))
    entries = [
        RosterEntryRow(
            league_id=league.row_id,
            scoring_period_id=period,
            team_id=4,
            espn_id=espn_id,
            lineup_slot_id=20,
            as_of=AS_OF,
        )
        for espn_id in espn_ids
    ]
    store.rosters.replace(league.row_id, period, 4, entries)
    return league


def test_gate_passes_when_every_rostered_player_is_mapped(store: Store, walk: Crosswalk) -> None:
    league = seed_roster(store, ALLEN, GIBBS, EAGLES_DST)
    assert check_rostered(store, league.row_id, crosswalk=walk) == {ALLEN, GIBBS, EAGLES_DST}


def test_gate_raises_naming_the_unmapped_rostered_player(store: Store, walk: Crosswalk) -> None:
    league = seed_roster(store, ALLEN, NOBODY, EAGLES_DST)
    store.players.upsert_many([player(ALLEN, "Josh Allen", "QB", "BUF"), player(NOBODY, "Nobody Special", "WR", "FA")])
    with pytest.raises(UnmappedPlayersError) as info:
        check_rostered(store, league.row_id, crosswalk=walk)
    assert info.value.unmapped == (UnmappedPlayer(NOBODY, (GSIS, SLEEPER), "Nobody Special", "WR", "FA"),)
    assert str(info.value) == (
        f"rostered players in league {league.row_id} (scoring period 4): 1 unmapped; "
        "add rows to id_overrides.csv and sync again:\n"
        f"  Nobody Special (WR FA, ESPN {NOBODY}): no gsis, sleeper id"
    )
    # Two override lines are the fix.
    fixed = build_crosswalk(
        ff_playerids(), [Override(NOBODY, GSIS, "00-0099999"), Override(NOBODY, SLEEPER, "13999")], as_of=AS_OF
    )
    assert check_rostered(store, league.row_id, crosswalk=fixed) == {ALLEN, NOBODY, EAGLES_DST}


def test_gate_uses_the_saved_crosswalk_and_the_latest_period_by_default(store: Store, walk: Crosswalk) -> None:
    league = seed_roster(store, ALLEN, period=3)
    seed_roster(store, ALLEN, NOBODY, period=4)
    walk.save(store)
    with pytest.raises(UnmappedPlayersError, match=r"\(scoring period 4\)") as info:
        check_rostered(store, league.row_id)
    assert info.value.unmapped == (UnmappedPlayer(NOBODY, (GSIS, SLEEPER)),)  # not in players: id only
    assert check_rostered(store, league.row_id, scoring_period_id=3) == {ALLEN}


def test_gate_can_be_narrowed_to_one_source(store: Store) -> None:
    walk = build_crosswalk(ff_playerids(), [Override(NOBODY, SLEEPER, "13999")], as_of=AS_OF)
    league = seed_roster(store, NOBODY)
    with pytest.raises(UnmappedPlayersError, match="no gsis id"):
        check_rostered(store, league.row_id, crosswalk=walk)
    assert check_rostered(store, league.row_id, crosswalk=walk, sources=(SLEEPER,)) == {NOBODY}


def test_gate_is_a_no_op_before_the_first_roster_snapshot(store: Store) -> None:
    league = store.leagues.upsert(
        LeagueRow(key="nfl", sport="nfl", espn_league_id=1, season=2026, team_id=1, as_of=AS_OF)
    )
    assert check_rostered(store, league.row_id) == set()


def test_gate_refuses_a_league_of_another_sport_or_one_it_does_not_know(store: Store, walk: Crosswalk) -> None:
    # The NFL crosswalk knows nothing of NBA players: an NBA league belongs to its own gate (the sync job dispatches on
    # the sport), so here it is refused outright rather than reported as unmapped or passed with no roster yet.
    nba = store.leagues.upsert(
        LeagueRow(key="nba", sport="nba", espn_league_id=654321, season=2027, team_id=2, as_of=AS_OF)
    )
    refusal = rf"^league {nba.row_id} \(nba 2027\) is an nba league; the nfl crosswalk gates nfl leagues only"
    with pytest.raises(CrosswalkError, match=refusal):
        check_rostered(store, nba.row_id)  # no roster snapshot yet: refused, not a vacuous pass
    store.teams.upsert(TeamRow(league_id=nba.row_id, team_id=2, name="Team 2", as_of=AS_OF))
    entry = RosterEntryRow(
        league_id=nba.row_id, scoring_period_id=1, team_id=2, espn_id=JOKIC, lineup_slot_id=0, as_of=AS_OF
    )
    store.rosters.replace(nba.row_id, 1, 2, [entry])
    with pytest.raises(CrosswalkError, match=refusal) as info:
        check_rostered(store, nba.row_id, crosswalk=walk)
    assert not isinstance(info.value, UnmappedPlayersError)  # a wrong dispatch, not an unmapped player

    with pytest.raises(LookupError, match="no league with id 999 in the store"):
        check_rostered(store, 999, crosswalk=walk)
    nfl = seed_roster(store, ALLEN)  # an NFL league beside it is still gated as before
    assert check_rostered(store, nfl.row_id, crosswalk=walk) == {ALLEN}


def test_unmapped_report_is_sorted_by_name_with_ids_as_fallback(walk: Crosswalk) -> None:
    players = [player(NOBODY, "Zed Nobody"), player(123, "Al Nobody", "TE", "DAL")]
    report = walk.unmapped([ALLEN, NOBODY, 123, 7, EAGLES_DST], players=players)
    assert [entry.name for entry in report] == [None, "Al Nobody", "Zed Nobody"]
    assert str(report[0]) == "ESPN 7: no gsis, sleeper id"
    assert str(report[1]) == "Al Nobody (TE DAL, ESPN 123): no gsis, sleeper id"
    assert str(report[2]) == f"Zed Nobody (ESPN {NOBODY}): no gsis, sleeper id"
    assert str(UnmappedPlayer(5, (GSIS,), position="K")) == "ESPN 5 (K): no gsis id"
