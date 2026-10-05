"""NBA player ID crosswalk (ROADMAP #17): the (name, team, position) match, overrides, the store round trip, the gate.

ESPN's side is the real ``fba`` pool capture under tests/fixtures/sources/market (seven players with their pro teams
and ``eligibleSlots``; ESPN has Giannis Antetokounmpo on Miami) plus hand-made rows for the awkward names. The nba.com
side is DARKO's real talent export (tests/fixtures/sources/darko), the hand-built stats.nba.com splits
(tests/fixtures/sources/nba_stats) and ``nba_api``'s bundled player table. The names used for partial matches are
nba.com's own (``Herbert Jones``, ``Nic Claxton``, ``Moritz Wagner``, ``Bub Carrington`` in the live 2025-26 splits).
No network.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import polars as pl
import pytest

from fm.espn.ids import FBA
from fm.model.ids import CrosswalkError, Override, UnmappedPlayer, UnmappedPlayersError
from fm.model.ids_nba import (
    ESPN_TEAM_ALIASES,
    NBA_OVERRIDES,
    NBA_SOURCE,
    NBA_TEAM_CODES,
    ORIGIN_NAME_MATCH,
    ORIGIN_OVERRIDE,
    ORIGIN_PARTIAL_MATCH,
    MatchKind,
    MatchNote,
    NbaCrosswalk,
    NbaPerson,
    UnmappedNbaPlayersError,
    build_nba_crosswalk,
    check_nba_rostered,
    espn_position_groups,
    espn_tricode,
    fetch_nba_crosswalk,
    load_nba_overrides,
    merge_persons,
    nba_tricode,
    normalize_name,
    parse_nba_overrides,
    persons_from_darko,
    persons_from_frame,
    persons_from_static,
    position_groups,
    tricode,
)
from fm.sources.base import RateLimiter, SourceError
from fm.sources.darko import TALENT_URL, DarkoSource, parse_talent
from fm.sources.nba_stats import NbaStatsSource, NbaStatsTransport, parse_result_sets, pick_result_set
from fm.store import LeagueRow, PlayerIdRow, PlayerRow, RosterEntryRow, Sport, Store, TeamRow

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
POOL = FIXTURES / "sources" / "market" / "espn_fba_kona_player_info.json"
TALENT = FIXTURES / "sources" / "darko" / "talent.csv"
SPLITS = FIXTURES / "sources" / "nba_stats" / "leaguedashplayerstats_Base_2025-26.json"
AS_OF = datetime(2026, 10, 5, 15, 0, tzinfo=UTC)

# ESPN id -> nba.com person id for the seven pool players.
WEMBANYAMA, JOKIC, SGA, EDWARDS, DONCIC, GIANNIS, FLAGG = 5104157, 3112335, 4278073, 4594268, 3945274, 3032977, 5041939
NBA_IDS = {
    WEMBANYAMA: 1641705,
    JOKIC: 203999,
    SGA: 1628983,
    EDWARDS: 1630162,
    DONCIC: 1629029,
    GIANNIS: 203507,
    FLAGG: 1642843,
}
HERB, MOE = 9001, 9002  # hand-made ESPN ids for players nba.com spells differently
HERBERT_JONES, NIC_CLAXTON, MORITZ_WAGNER, FRANZ_WAGNER, BUB_CARRINGTON = 1630529, 1629651, 1629021, 1630532, 1642267


def team(abbrev: str) -> int:
    return next(team_id for team_id, label in FBA.pro_teams.items() if label == abbrev)


def player(
    espn_id: int,
    name: str,
    position: str | None = None,
    pro_team: str | None = None,
    *,
    eligible: tuple[str, ...] = (),
    sport: Sport = "nba",
) -> PlayerRow:
    """A ``players`` row as the sync writes it: labels decoded from ESPN's ids."""
    return PlayerRow(
        sport=sport,
        espn_id=espn_id,
        full_name=name,
        default_position_id=FBA.position_id(position) if position else None,
        position=position,
        pro_team_id=team(pro_team) if pro_team else None,
        pro_team=pro_team,
        eligible_slot_ids=[FBA.slot_id(label) for label in eligible],
        as_of=AS_OF,
    )


def pool_players() -> list[PlayerRow]:
    """The real pool capture as ``players`` rows."""
    rows: list[PlayerRow] = []
    for entry in json.loads(POOL.read_text(encoding="utf-8"))["players"]:
        espn = entry["player"]
        rows.append(
            PlayerRow(
                sport="nba",
                espn_id=espn["id"],
                full_name=espn["fullName"],
                default_position_id=espn["defaultPositionId"],
                position=FBA.position_label(espn["defaultPositionId"]),
                pro_team_id=espn["proTeamId"],
                pro_team=FBA.pro_team(espn["proTeamId"]),
                eligible_slot_ids=espn["eligibleSlots"],
                as_of=AS_OF,
            )
        )
    return rows


def splits_frame() -> pl.DataFrame:
    return pick_result_set(parse_result_sets(SPLITS.read_bytes()), "LeagueDashPlayerStats", ("PLAYER_ID",))


def darko_persons() -> list[NbaPerson]:
    return persons_from_darko(parse_talent(TALENT.read_bytes()))


def persons(*extra: NbaPerson) -> list[NbaPerson]:
    """What the sources would list: the splits, DARKO, anything extra, then the bundled table."""
    return merge_persons(persons_from_frame(splits_frame()), darko_persons(), extra, persons_from_static())


def build(players: list[PlayerRow], found: list[NbaPerson], overrides: tuple[Override, ...] = ()) -> NbaCrosswalk:
    return build_nba_crosswalk(players, found, overrides, as_of=AS_OF)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


# --- names, teams, positions ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "normalized"),
    [
        ("Luka Dončić", "luka doncic"),
        ("Kristaps Porziņģis", "kristaps porzingis"),
        ("Yanic Konan Niederhäuser", "yanic konan niederhauser"),
        ("Nikola Đurišić", "nikola durisic"),
        ("Tarık Biberović", "tarik biberovic"),
        ("P.J. Washington", "pj washington"),
        ("PJ Washington", "pj washington"),
        ("De'Aaron Fox", "deaaron fox"),
        ("D’Angelo Russell", "dangelo russell"),
        ("Shai Gilgeous-Alexander", "shai gilgeous alexander"),
        ("Jaren Jackson Jr.", "jaren jackson"),
        ("Jimmy Butler III", "jimmy butler"),
        ("  Kelly   Oubre Jr ", "kelly oubre"),
        ("Jr.", "jr"),
    ],
)
def test_normalize_name(name: str, normalized: str) -> None:
    assert normalize_name(name) == normalized


@pytest.mark.parametrize(
    ("position", "groups"),
    [
        ("PG", {"G"}),
        ("SF", {"F"}),
        ("C", {"C"}),
        ("G-F", {"G", "F"}),
        ("F-C", {"F", "C"}),
        ("Center-Forward", {"C", "F"}),
        ("Guard", {"G"}),
        ("c_pos", {"C"}),
        ("pg_pos", {"G"}),
        ("Coach", set()),
        ("", set()),
        (None, set()),
    ],
)
def test_position_groups(position: str | None, groups: set[str]) -> None:
    assert position_groups(position) == groups


def test_espn_positions_add_every_single_position_slot_he_is_eligible_at() -> None:
    edwards = next(row for row in pool_players() if row.espn_id == EDWARDS)
    assert edwards.position == "SG" and espn_position_groups(edwards) == {"G", "F"}  # SG/SF eligible
    giannis = next(row for row in pool_players() if row.espn_id == GIANNIS)
    assert giannis.position == "PF" and espn_position_groups(giannis) == {"F", "C"}
    utility = player(1, "Anyone", "PG", eligible=("G/F", "F/C", "UTIL"))  # combination slots say nothing
    assert espn_position_groups(utility) == {"G"}
    assert espn_position_groups(player(2, "Nobody")) == set()


def test_team_codes_are_nba_tricodes() -> None:
    assert {code: tricode(code) for code in ESPN_TEAM_ALIASES} == {
        "GS": "GSW",
        "NO": "NOP",
        "NY": "NYK",
        "SA": "SAS",
        "UTAH": "UTA",
        "WSH": "WAS",
    }
    assert tricode(" lal ") == "LAL" and tricode("FA") is None and tricode("") is None and tricode(None) is None
    assert espn_tricode(player(1, "A", pro_team="NYK")) == "NYK"
    from_id = PlayerRow(sport="nba", espn_id=1, full_name="A", pro_team_id=team("SAS"), pro_team="SA", as_of=AS_OF)
    assert espn_tricode(from_id) == "SAS"
    by_label = PlayerRow(sport="nba", espn_id=1, full_name="A", pro_team_id=None, pro_team="UTAH", as_of=AS_OF)
    assert espn_tricode(by_label) == "UTA"
    free_agent = PlayerRow(sport="nba", espn_id=1, full_name="A", pro_team_id=0, pro_team="FA", as_of=AS_OF)
    assert espn_tricode(free_agent) is None
    assert nba_tricode(1610612759) == "SAS" and nba_tricode(None) is None and nba_tricode(1) is None
    assert set(NBA_TEAM_CODES.values()) == set(FBA.pro_teams.values()) - {"FA"}  # ESPN's id map uses the same codes


# --- persons from the sources -----------------------------------------------------------------------------------------


def test_persons_from_the_stats_splits() -> None:
    assert persons_from_frame(splits_frame()) == [
        NbaPerson(203999, "Nikola Jokic", team="DEN"),
        NbaPerson(1628983, "Shai Gilgeous-Alexander", team="OKC"),
        NbaPerson(1641705, "Victor Wembanyama", team="SAS"),
    ]


def test_game_logs_give_the_latest_team_and_every_spelling() -> None:
    logs = pl.DataFrame(
        {
            "PLAYER_ID": [1628991, 1628991, 1628991, None, 7],
            "PLAYER_NAME": ["Jaren Jackson Jr.", "Jaren Jackson Jr", "Jaren Jackson Jr.", "Ghost", None],
            "TEAM_ABBREVIATION": ["UTA", "MEM", None, "MEM", "MEM"],
            "POSITION": [None, "F-C", "F", None, None],
            "GAME_DATE": ["2026-02-12", "2025-10-22", "2026-01-30", "2026-01-01", "2026-01-01"],
        }
    )
    # By date: MEM (F-C), then no team (F), then UTA with no position: each later row fills only what it names.
    assert persons_from_frame(logs) == [
        NbaPerson(1628991, "Jaren Jackson Jr", team="UTA", position="F", aliases=("Jaren Jackson Jr.",))
    ]
    with pytest.raises(CrosswalkError, match="missing columns PLAYER_NAME"):
        persons_from_frame(pl.DataFrame({"PLAYER_ID": [1]}))
    empty = pick_result_set(parse_result_sets(_splits_payload([])), "LeagueDashPlayerStats", ("PLAYER_ID",))
    assert persons_from_frame(empty) == []  # a season before its first game


def test_persons_from_darkos_sheet() -> None:
    by_id = {person.nba_id: person for person in darko_persons()}
    assert len(by_id) == 13
    assert by_id[203999] == NbaPerson(203999, "Nikola Jokic", team=None, position="C-F")  # tm_id -999
    assert by_id[1641705] == NbaPerson(1641705, "Victor Wembanyama", team="SAS", position="F-C")
    assert by_id[1628973].team == "NYK" and by_id[1642907].team == "MEM"
    talent = parse_talent(TALENT.read_bytes())[0].model_copy(update={"position": None})
    assert persons_from_darko([talent])[0].position == "c_pos"  # the modeled bucket when no position is listed


def test_persons_from_the_bundled_table_are_active_players_only() -> None:
    found = persons_from_static()
    by_id = {person.nba_id: person for person in found}
    assert len(found) == len(by_id) > 400
    assert by_id[1642843] == NbaPerson(1642843, "Cooper Flagg")  # a 2025 rookie: name and id only
    assert by_id[203999].name == "Nikola Jokić"
    assert 76001 not in by_id  # Alaa Abdelnaby, retired


def test_merge_persons_keeps_the_first_team_and_position_and_every_spelling() -> None:
    merged = merge_persons(
        [NbaPerson(1, "Nic Claxton", team="BKN")],
        [NbaPerson(1, "Nicolas Claxton", team="NYK", position="C"), NbaPerson(2, "Other")],
        [NbaPerson(1, "Nic Claxton", position="F")],
    )
    assert merged == [
        NbaPerson(1, "Nic Claxton", team="BKN", position="C", aliases=("Nicolas Claxton",)),
        NbaPerson(2, "Other"),
    ]
    assert str(merged[0]) == "nba 1 Nic Claxton (C BKN)" and str(merged[1]) == "nba 2 Other"


# --- the match --------------------------------------------------------------------------------------------------------


def test_real_pool_players_match_by_name_team_and_position() -> None:
    milwaukee = NbaPerson(203507, "Giannis Antetokounmpo", team="MIL", position="F")  # last season's team
    walk = build(pool_players(), persons(milwaukee))
    assert {espn_id: walk.nba_id(espn_id) for espn_id in NBA_IDS} == NBA_IDS
    assert all(row.origin == ORIGIN_NAME_MATCH and row.as_of == AS_OF for row in walk.rows)
    assert walk.notes == (
        MatchNote(
            GIANNIS,
            "Giannis Antetokounmpo",
            MatchKind.DISPUTED,
            "matched nba 203507 Giannis Antetokounmpo (F MIL) by name, but ESPN lists him as PF MIA; the team "
            "disagrees",
            (203507,),
        ),
    )
    assert walk.warnings == ("name_match: 1 matches disagree with ESPN on team or position (Giannis Antetokounmpo)",)


def test_names_match_across_diacritics_punctuation_and_suffixes() -> None:
    found = [
        NbaPerson(1629029, "Luka Dončić", team="LAL"),
        NbaPerson(1629023, "P.J. Washington", team="DAL"),
        NbaPerson(202710, "Jimmy Butler III", team="GSW"),
        NbaPerson(1628991, "Jaren Jackson Jr.", team="UTA"),
    ]
    players = [
        player(1, "Luka Doncic", "PG", "LAL"),
        player(2, "PJ Washington", "PF", "DAL"),
        player(3, "Jimmy Butler", "SF", "GSW"),
        player(4, "Jaren Jackson Jr.", "PF", "UTA"),
    ]
    walk = build(players, found)
    assert [walk.nba_id(espn_id) for espn_id in (1, 2, 3, 4)] == [1629029, 1629023, 202710, 1628991]
    assert walk.notes == () and walk.warnings == ()


def test_partial_match_on_his_team_by_surname_and_a_shortened_first_name() -> None:
    found = [
        NbaPerson(HERBERT_JONES, "Herbert Jones", team="NOP", position="F"),
        NbaPerson(NIC_CLAXTON, "Nic Claxton", team="BKN"),
        NbaPerson(1628964, "Mo Bamba", team="UTA", position="C"),
    ]
    players = [
        player(HERB, "Herb Jones", "SF", "NOP"),
        player(2, "Nicolas Claxton", "C", "BKN"),
        player(3, "Mohamed Bamba", "C", "UTA"),
    ]
    walk = build(players, found)
    assert walk.nba_id(HERB) == HERBERT_JONES and walk.nba_id(2) == NIC_CLAXTON and walk.nba_id(3) == 1628964
    assert {row.origin for row in walk.rows} == {ORIGIN_PARTIAL_MATCH}
    note = walk.note(HERB)
    assert note is not None and note.kind is MatchKind.PARTIAL and note.candidates == (HERBERT_JONES,)
    assert note.reason == (
        "matched nba 1630529 Herbert Jones (F NOP) by surname, team and first name; confirm it with an override"
    )
    assert walk.warnings == (
        "partial_match: 3 matches by surname, team and first name (Herb Jones, Mohamed Bamba, Nicolas Claxton); "
        "confirm",
    )


def test_partial_match_never_pairs_different_first_names() -> None:
    found = [
        NbaPerson(1628998, "Cody Martin", team="CHA", position="F"),
        NbaPerson(MORITZ_WAGNER, "Moritz Wagner", team="ORL", position="C"),
        NbaPerson(FRANZ_WAGNER, "Franz Wagner", team="ORL", position="F"),
        NbaPerson(BUB_CARRINGTON, "Bub Carrington", team="WAS", position="G"),
    ]
    players = [
        player(1, "Caleb Martin", "SF", "CHA"),  # twins share a surname, an initial and once a team
        player(MOE, "Moe Wagner", "C", "ORL"),
        player(3, "Carlton Carrington", "PG", "WAS"),
    ]
    walk = build(players, found)
    assert walk.mapped() == set() and len(walk) == 0
    notes = {note.espn_id: note for note in walk.notes}
    assert all(note.kind is MatchKind.NO_MATCH for note in notes.values())
    assert notes[MOE].reason == (
        "no nba.com player named 'Moe Wagner'; on his team: nba 1629021 Moritz Wagner (C ORL), nba 1630532 Franz "
        "Wagner (F ORL)"
    )
    assert notes[MOE].candidates == (MORITZ_WAGNER, FRANZ_WAGNER)
    assert notes[1].candidates == (1628998,) and notes[3].candidates == (BUB_CARRINGTON,)
    assert walk.warnings == (
        "name_match: 3 of 3 ESPN players have no nba.com match (Caleb Martin, Carlton Carrington, Moe Wagner)",
    )


def test_partial_match_needs_his_team_and_a_compatible_position() -> None:
    found = [NbaPerson(HERBERT_JONES, "Herbert Jones", team="NOP", position="F")]
    free_agent = build([player(HERB, "Herb Jones", "SF")], found)
    assert free_agent.nba_id(HERB) is None
    no_team = MatchNote(HERB, "Herb Jones", MatchKind.NO_MATCH, "no nba.com player named 'Herb Jones'")
    assert free_agent.note(HERB) == no_team
    center = build([player(HERB, "Herb Jones", "C", "NOP")], found)
    assert center.nba_id(HERB) is None and center.note(HERB) is not None


def test_a_person_two_players_partially_fit_goes_to_neither() -> None:
    found = [NbaPerson(1642259, "Alexandre Sarr", team="WAS", position="C")]  # a formal spelling, for the test
    walk = build([player(1, "Alex Sarr", "C", "WAS"), player(2, "Al Sarr", "C", "WAS")], found)
    assert walk.mapped() == set()
    assert {(note.kind, note.candidates) for note in walk.notes} == {(MatchKind.AMBIGUOUS, (1642259,))}
    assert walk.warnings == ("name_match: 2 ESPN players fit more than one way (Al Sarr, Alex Sarr); add overrides",)


def test_namesakes_are_settled_by_team_or_left_for_an_override() -> None:
    found = [NbaPerson(1630188, "Jalen Smith", team="CHI", position="F"), NbaPerson(77, "Jalen Smith", team="PHX")]
    on_a_team = build([player(1, "Jalen Smith", "PF", "CHI")], found)
    assert on_a_team.nba_id(1) == 1630188 and on_a_team.notes == ()
    unsigned = build([player(1, "Jalen Smith", "PF")], found)
    assert unsigned.nba_id(1) is None
    note = unsigned.note(1)
    assert note is not None and note.kind is MatchKind.AMBIGUOUS and note.candidates == (77, 1630188)
    assert note.reason == "several nba.com players fit: nba 77 Jalen Smith (PHX), nba 1630188 Jalen Smith (F CHI)"


def test_a_lone_namesake_on_another_team_at_another_position_is_someone_else() -> None:
    found = [NbaPerson(55, "Marcus Morris", team="LAL", position="G")]
    walk = build([player(1, "Marcus Morris", "PF", "NYK")], found)
    assert walk.nba_id(1) is None
    assert walk.note(1) == MatchNote(
        1,
        "Marcus Morris",
        MatchKind.NO_MATCH,
        "ESPN lists him as PF NYK, but the nba.com player with this name is not: nba 55 Marcus Morris (G LAL)",
        (55,),
    )
    traded = build([player(1, "Marcus Morris", "SG", "NYK")], found)  # one of the two agreeing is enough
    assert traded.nba_id(1) == 55 and traded.note(1) is not None


def test_espn_players_sharing_a_name_wait_for_overrides() -> None:
    found = [NbaPerson(1631114, "Jalen Williams", team="OKC", position="G")]
    twins = [player(1, "Jalen Williams", "SG", "OKC"), player(2, "Jalen Williams", "SF", "DEN")]
    walk = build(twins, found)
    assert walk.mapped() == set()
    assert [(note.espn_id, note.kind, note.reason) for note in walk.notes] == [
        (1, MatchKind.AMBIGUOUS, "2 ESPN players share this name"),
        (2, MatchKind.AMBIGUOUS, "2 ESPN players share this name"),
    ]
    pinned = build(twins, found, (Override(2, NBA_SOURCE, None),))  # the other one is not in the NBA
    assert pinned.nba_id(1) == 1631114


def test_build_takes_nba_players_once_each() -> None:
    with pytest.raises(CrosswalkError, match="players must be nba, got nfl for ESPN 1"):
        build([player(1, "Josh Allen", sport="nfl")], [])
    found = [NbaPerson(203999, "Nikola Jokic", team="DEN")]
    walk = build([player(1, "Nikola Jokic", "C", "DEN"), player(1, "Nikola Jokic", "C", "DEN")], found)
    assert walk.nba_id(1) == 203999 and len(walk) == 1


# --- overrides --------------------------------------------------------------------------------------------------------


def test_parse_nba_overrides_skips_comments_and_blank_lines() -> None:
    text = (
        "# reviewed by hand\n\nespn_id,source,source_id,name,note\n"
        f"{MOE},nba,{MORITZ_WAGNER},Moe Wagner,nba.com spells him Moritz\n"
        f"{JOKIC},NBA,,Nikola Jokic,keep unmapped\n"
    )
    assert parse_nba_overrides(text) == (
        Override(MOE, NBA_SOURCE, str(MORITZ_WAGNER), "Moe Wagner", "nba.com spells him Moritz"),
        Override(JOKIC, NBA_SOURCE, None, "Nikola Jokic", "keep unmapped"),
    )
    assert parse_nba_overrides("# nothing yet\n") == ()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("x,nba,1,,", "espn_id 'x' is not an integer"),
        ("0,nba,1,,", "espn_id must not be 0"),
        ("1,gsis,1,,", "unknown source 'gsis'; expected nba"),
        ("1,nba,0203999,,", "'0203999' does not look like an nba.com person id"),
        ("1,nba,00-0034857,,", "'00-0034857' does not look like an nba.com person id"),
        ("1,nba,1,,,extra", "expected 5 fields, got 6"),
    ],
)
def test_bad_override_lines_name_the_line(line: str, message: str) -> None:
    text = f"# header next\nespn_id,source,source_id,name,note\n{line}\n"
    with pytest.raises(CrosswalkError, match=rf"^id_overrides_nba\.csv:3: {re.escape(message)}"):
        parse_nba_overrides(text, where="id_overrides_nba.csv")


def test_duplicate_overrides_are_rejected() -> None:
    header = "espn_id,source,source_id,name,note\n"
    with pytest.raises(CrosswalkError, match="overrides:3: ESPN 1 nba already set on line 2"):
        parse_nba_overrides(header + "1,nba,10,,\n1,nba,11,,\n")
    with pytest.raises(CrosswalkError, match="overrides:3: nba id 10 already used on line 2"):
        parse_nba_overrides(header + "1,nba,10,,\n2,nba,10,,\n")
    with pytest.raises(CrosswalkError, match="overrides:1: header lacks source_id"):
        parse_nba_overrides("espn_id,source,name\n")


def test_load_nba_overrides_reads_the_file_or_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "ov.csv"
    path.write_text(f"espn_id,source,source_id,name,note\n{MOE},nba,{MORITZ_WAGNER},Moe Wagner,\n", encoding="utf-8")
    assert load_nba_overrides(path) == (Override(MOE, NBA_SOURCE, str(MORITZ_WAGNER), "Moe Wagner"),)
    with pytest.raises(CrosswalkError, match="overrides file not found"):
        load_nba_overrides(tmp_path / "missing.csv")


def test_committed_overrides_file_is_well_formed() -> None:
    assert NBA_OVERRIDES.is_file() and NBA_OVERRIDES.name == "id_overrides_nba.csv"
    overrides = load_nba_overrides()
    assert all(override.source == NBA_SOURCE for override in overrides)
    build(pool_players(), persons(), overrides)  # and applies cleanly


def test_an_override_maps_a_player_the_match_cannot() -> None:
    found = [NbaPerson(MORITZ_WAGNER, "Moritz Wagner", team="ORL"), NbaPerson(FRANZ_WAGNER, "Franz Wagner", team="ORL")]
    pin = Override(MOE, NBA_SOURCE, str(MORITZ_WAGNER), "Moe Wagner")
    walk = build([player(MOE, "Moe Wagner", "C", "ORL"), player(2, "Franz Wagner", "SF", "ORL")], found, (pin,))
    assert walk.nba_id(MOE) == MORITZ_WAGNER and walk.nba_id(2) == FRANZ_WAGNER
    assert walk.rows[0] == PlayerIdRow(
        sport="nba", espn_id=2, source=NBA_SOURCE, source_id=str(FRANZ_WAGNER), origin=ORIGIN_NAME_MATCH, as_of=AS_OF
    )
    assert walk.rows[1].origin == ORIGIN_OVERRIDE and walk.notes == () and walk.warnings == ()


def test_pins_are_out_of_play_for_the_match() -> None:
    found = [
        NbaPerson(1631114, "Jalen Williams", team="OKC", position="G"),
        NbaPerson(1631119, "Jaylin Williams", team="OKC", position="F"),
    ]
    players = [player(1, "Jalen Williams", "SG", "OKC"), player(2, "Jaylin Williams", "PF", "OKC")]
    # Pinning player 1 to the other Williams leaves player 2's namesake taken, and says so; nobody is re-pointed later.
    walk = build(players, found, (Override(1, NBA_SOURCE, "1631119"),))
    assert walk.nba_id(1) == 1631119 and walk.nba_id(2) is None
    assert walk.note(2) == MatchNote(
        2,
        "Jaylin Williams",
        MatchKind.NO_MATCH,
        "the nba.com player with this name is mapped already: nba 1631119 Jaylin Williams (F OKC) to ESPN 1 "
        "(by an override)",
    )
    swapped = build(players, found, (Override(1, NBA_SOURCE, "1631119"), Override(2, NBA_SOURCE, "1631114")))
    assert (swapped.nba_id(1), swapped.nba_id(2)) == (1631119, 1631114) and swapped.notes == ()


def test_an_empty_override_keeps_a_player_unmapped_and_says_why() -> None:
    walk = build(pool_players(), persons(), (Override(JOKIC, NBA_SOURCE, None), Override(424242, NBA_SOURCE, "77")))
    assert walk.nba_id(JOKIC) is None and walk.espn_id(203999) is None
    assert walk.note(JOKIC) == MatchNote(
        JOKIC,
        "Nikola Jokic",
        MatchKind.REMOVED,
        "id_overrides_nba.csv keeps him unmapped (an empty source_id); give him an nba.com id there",
    )
    assert walk.nba_id(424242) == 77  # a pin for a player the build was not given still lands
    assert walk.warnings == ()  # pins are the review, not something to warn about


# --- the crosswalk ----------------------------------------------------------------------------------------------------


def test_crosswalk_lookups_both_ways() -> None:
    walk = build(pool_players(), persons())
    assert walk.espn_id(203999) == JOKIC and walk.espn_id("203999") == JOKIC and walk.espn_id(" 1642843 ") == FLAGG
    assert walk.nba_id(JOKIC) == 203999 and walk.nba_id(1) is None and walk.espn_id(1) is None
    assert walk.mapped() == set(NBA_IDS) and len(walk) == 7
    assert [row.espn_id for row in walk.rows] == sorted(NBA_IDS)
    assert walk.unmapped([JOKIC, 1, 2], players=[player(2, "Zed Nobody", "C", "LAL")]) == [
        UnmappedPlayer(1, (NBA_SOURCE,)),
        UnmappedPlayer(2, (NBA_SOURCE,), "Zed Nobody", "C", "LAL"),
    ]


def test_crosswalk_rejects_inconsistent_rows() -> None:
    def row(espn_id: int, source_id: str, sport: Sport = "nba") -> PlayerIdRow:
        return PlayerIdRow(
            sport=sport, espn_id=espn_id, source=NBA_SOURCE, source_id=source_id, origin="test", as_of=AS_OF
        )

    with pytest.raises(CrosswalkError, match="two nba ids"):
        NbaCrosswalk([row(1, "1"), row(1, "2")])
    with pytest.raises(CrosswalkError, match="maps to ESPN 1 and ESPN 2"):
        NbaCrosswalk([row(1, "1"), row(2, "1")])
    with pytest.raises(CrosswalkError, match="must be nba"):
        NbaCrosswalk([row(1, "1", sport="nfl")])


def test_save_replaces_the_nba_rows_and_from_store_reads_them_back(store: Store) -> None:
    assert len(NbaCrosswalk.from_store(store)) == 0
    nfl = PlayerIdRow(sport="nfl", espn_id=3918298, source="sleeper", source_id="4984", origin="test", as_of=AS_OF)
    store.player_ids.upsert(nfl)
    walk = build(pool_players(), persons())
    assert walk.save(store) == 7
    found = store.player_ids.lookup("nba", NBA_SOURCE, "203999")
    assert found is not None and found.espn_id == JOKIC and found.origin == ORIGIN_NAME_MATCH
    again = NbaCrosswalk.from_store(store)
    assert again.rows == walk.rows and again.warnings == () and again.notes == ()
    assert store.player_ids.for_player("nfl", 3918298) == [nfl]  # the other sport is untouched
    # A pin that moves an nba.com id to another player would collide under an upsert; save replaces.
    moved = build(pool_players(), persons(), (Override(424242, NBA_SOURCE, "203999"),))
    assert moved.save(store) == 7
    owner = store.player_ids.lookup("nba", NBA_SOURCE, "203999")
    assert owner is not None and owner.espn_id == 424242 and owner.origin == ORIGIN_OVERRIDE
    assert store.player_ids.unmapped("nba", NBA_SOURCE, [JOKIC, SGA]) == {JOKIC}


# --- the unmapped-rostered-player gate --------------------------------------------------------------------------------


def seed_roster(store: Store, *espn_ids: int, period: int = 12, sport: Sport = "nba") -> LeagueRow:
    league = store.leagues.upsert(
        LeagueRow(key=sport, sport=sport, espn_league_id=654321, season=2027, team_id=2, as_of=AS_OF)
    )
    store.teams.upsert(TeamRow(league_id=league.row_id, team_id=2, name="Team 2", as_of=AS_OF))
    entries = [
        RosterEntryRow(
            league_id=league.row_id,
            scoring_period_id=period,
            team_id=2,
            espn_id=espn_id,
            lineup_slot_id=12,
            as_of=AS_OF,
        )
        for espn_id in espn_ids
    ]
    store.rosters.replace(league.row_id, period, 2, entries)
    return league


def test_gate_passes_when_every_rostered_player_is_mapped(store: Store) -> None:
    league = seed_roster(store, JOKIC, SGA, FLAGG)
    assert check_nba_rostered(store, league.row_id, crosswalk=build(pool_players(), persons())) == {JOKIC, SGA, FLAGG}


def test_gate_names_each_unmapped_rostered_player_and_why(store: Store) -> None:
    moe = player(MOE, "Moe Wagner", "C", "ORL")
    found = persons(NbaPerson(MORITZ_WAGNER, "Moritz Wagner", team="ORL", position="C"))
    walk = build([*pool_players(), moe], found)
    league = seed_roster(store, JOKIC, MOE)
    store.players.upsert_many([moe, *pool_players()])
    with pytest.raises(UnmappedNbaPlayersError) as info:
        check_nba_rostered(store, league.row_id, crosswalk=walk)
    error = info.value
    assert isinstance(error, UnmappedPlayersError) and isinstance(error, CrosswalkError)
    assert error.unmapped == (UnmappedPlayer(MOE, (NBA_SOURCE,), "Moe Wagner", "C", "ORL"),)
    assert str(error) == (
        f"rostered players in league {league.row_id} (scoring period 12): 1 unmapped; add rows to "
        "id_overrides_nba.csv and sync again:\n"
        f"  Moe Wagner (C ORL, ESPN {MOE}): no nba id (no nba.com player named 'Moe Wagner'; on his team: "
        "nba 1629021 Moritz Wagner (C ORL))"
    )
    assert set(error.reasons) == {MOE}
    # One override line is the fix.
    fixed = build([*pool_players(), moe], found, (Override(MOE, NBA_SOURCE, str(MORITZ_WAGNER)),))
    assert check_nba_rostered(store, league.row_id, crosswalk=fixed) == {JOKIC, MOE}


def test_gate_uses_the_saved_crosswalk_and_the_latest_period_by_default(store: Store) -> None:
    league = seed_roster(store, JOKIC, period=11)
    seed_roster(store, JOKIC, MOE, period=12)
    build(pool_players(), persons()).save(store)
    with pytest.raises(UnmappedNbaPlayersError, match=r"\(scoring period 12\)") as info:
        check_nba_rostered(store, league.row_id)
    assert info.value.unmapped == (UnmappedPlayer(MOE, (NBA_SOURCE,)),) and info.value.reasons == {}
    assert str(info.value).endswith(f"  ESPN {MOE}: no nba id")  # no name stored, no build notes saved
    assert check_nba_rostered(store, league.row_id, scoring_period_id=11) == {JOKIC}


def test_gate_is_a_no_op_before_the_first_roster_snapshot(store: Store) -> None:
    league = store.leagues.upsert(
        LeagueRow(key="nba", sport="nba", espn_league_id=1, season=2027, team_id=1, as_of=AS_OF)
    )
    assert check_nba_rostered(store, league.row_id) == set()


def test_gate_refuses_a_league_of_another_sport_or_one_it_does_not_know(store: Store) -> None:
    walk = build(pool_players(), persons())
    nfl = store.leagues.upsert(
        LeagueRow(key="nfl", sport="nfl", espn_league_id=123456, season=2026, team_id=4, as_of=AS_OF)
    )
    refusal = rf"^league {nfl.row_id} \(nfl 2026\) is an nfl league; the nba crosswalk gates nba leagues only"
    with pytest.raises(CrosswalkError, match=refusal):
        check_nba_rostered(store, nfl.row_id)  # no roster yet: refused, not a vacuous pass
    store.teams.upsert(TeamRow(league_id=nfl.row_id, team_id=4, name="Team 4", as_of=AS_OF))
    entry = RosterEntryRow(
        league_id=nfl.row_id, scoring_period_id=4, team_id=4, espn_id=3918298, lineup_slot_id=0, as_of=AS_OF
    )
    store.rosters.replace(nfl.row_id, 4, 4, [entry])
    with pytest.raises(CrosswalkError, match=refusal) as info:
        check_nba_rostered(store, nfl.row_id, crosswalk=walk)
    assert not isinstance(info.value, UnmappedPlayersError)  # a wrong dispatch, not an unmapped player
    with pytest.raises(LookupError, match="no league with id 999 in the store"):
        check_nba_rostered(store, 999, crosswalk=walk)
    nba = seed_roster(store, JOKIC)  # an NBA league beside it is gated as usual
    assert check_nba_rostered(store, nba.row_id, crosswalk=walk) == {JOKIC}


# --- through the adapters ---------------------------------------------------------------------------------------------


def _splits_payload(rows: list[tuple[int, str, str]]) -> bytes:
    """The recorded splits payload with its rows replaced: (PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION) on a copy of
    the first recorded row, so every other column keeps a plausible value."""
    payload = json.loads(SPLITS.read_text(encoding="utf-8"))
    result = payload["resultSets"][0]
    headers: list[str] = result["headers"]
    template: list[Any] = result["rowSet"][0]
    built: list[list[Any]] = []
    for nba_id, name, code in rows:
        row = list(template)
        row[headers.index("PLAYER_ID")], row[headers.index("PLAYER_NAME")] = nba_id, name
        row[headers.index("TEAM_ABBREVIATION")] = code
        row[headers.index("TEAM_ID")] = next(team_id for team_id, label in NBA_TEAM_CODES.items() if label == code)
        built.append(row)
    result["rowSet"] = built
    return json.dumps(payload).encode()


class SplitsTransport:
    """Serves ``leaguedashplayerstats`` per season; a season mapped to an exception fails, anything else is a test
    failure."""

    def __init__(self, seasons: Mapping[str, bytes | Exception]) -> None:
        self.seasons = dict(seasons)
        self.calls: list[str] = []

    def get(self, endpoint: str, parameters: Mapping[str, object]) -> bytes:
        assert endpoint == "leaguedashplayerstats" and parameters["MeasureType"] == "Base"
        season = str(parameters["Season"])
        self.calls.append(season)
        served = self.seasons[season]
        if isinstance(served, Exception):
            raise served
        return served


class FakeClock:
    def __init__(self, now: datetime = AS_OF) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def make_stats(seasons: Mapping[str, bytes | Exception], clock: FakeClock, tmp_path: Path) -> NbaStatsSource:
    transport = cast(NbaStatsTransport, SplitsTransport(seasons))
    return NbaStatsSource(transport=transport, cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock)


def make_darko(handler: Callable[[httpx.Request], httpx.Response], clock: FakeClock, tmp_path: Path) -> DarkoSource:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return DarkoSource(
        client=client, cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock, sleep=lambda _: None
    )


def serve_talent(request: httpx.Request) -> httpx.Response:
    assert str(request.url) == TALENT_URL
    return httpx.Response(200, content=TALENT.read_bytes(), headers={"content-type": "text/csv"})


@pytest.fixture
def overrides_file(tmp_path: Path) -> Path:
    path = tmp_path / "id_overrides_nba.csv"
    path.write_text("espn_id,source,source_id,name,note\n", encoding="utf-8")
    return path


def test_fetch_builds_from_stats_darko_and_the_bundled_table(tmp_path: Path, overrides_file: Path) -> None:
    # Giannis is on Miami this season and was on Milwaukee last season; this season's team must win.
    current = _splits_payload([(203507, "Giannis Antetokounmpo", "MIA")])
    previous = _splits_payload([(203999, "Nikola Jokic", "DEN"), (203507, "Giannis Antetokounmpo", "MIL")])
    stats = make_stats({"2026-27": current, "2025-26": previous}, FakeClock(), tmp_path)
    darko = make_darko(serve_talent, FakeClock(AS_OF + timedelta(minutes=5)), tmp_path)  # a fresher pull

    fetched = fetch_nba_crosswalk(pool_players(), stats, ("2026-27", "2025-26"), darko=darko, overrides=overrides_file)
    walk = fetched.data
    assert {espn_id: walk.nba_id(espn_id) for espn_id in NBA_IDS} == NBA_IDS
    assert walk.notes == () and fetched.warnings == walk.warnings == ()  # Miami on both sides: no dispute
    assert (fetched.source, fetched.dataset, fetched.key) == ("nba_stats+darko", "crosswalk", "2026-27+2025-26")
    assert fetched.as_of == AS_OF and all(row.as_of == AS_OF for row in walk.rows)  # the oldest input's stamp
    assert (fetched.cached, fetched.stale, fetched.degraded) == (False, False, False)
    assert fetched.raw_path is not None and fetched.raw_path.is_file()
    assert cast(SplitsTransport, stats.transport).calls == ["2026-27", "2025-26"]

    calls = cast(SplitsTransport, stats.transport).calls
    again = fetch_nba_crosswalk(pool_players(), stats, ("2026-27", "2025-26"), darko=darko, overrides=overrides_file)
    assert again.cached and again.data.rows == walk.rows and len(calls) == 2  # served from the adapters' caches
    forced = fetch_nba_crosswalk(pool_players(), stats, "2026-27", darko=darko, overrides=overrides_file, force=True)
    assert not forced.cached and calls == ["2026-27", "2025-26", "2026-27"]  # fetch options reach the adapters


def test_fetch_reads_the_overrides_file(tmp_path: Path) -> None:
    clock = FakeClock()
    stats = make_stats({"2025-26": _splits_payload([(203999, "Nikola Jokic", "DEN")])}, clock, tmp_path)
    path = tmp_path / "ov.csv"
    path.write_text(f"espn_id,source,source_id,name,note\n{MOE},nba,{MORITZ_WAGNER},Moe Wagner,\n", encoding="utf-8")
    fetched = fetch_nba_crosswalk([player(MOE, "Moe Wagner", "C", "ORL")], stats, "2025-26", overrides=path)
    assert fetched.data.nba_id(MOE) == MORITZ_WAGNER and fetched.key == "2025-26" and fetched.source == "nba_stats"
    with pytest.raises(ValueError, match="at least one season"):
        fetch_nba_crosswalk([], stats, (), overrides=path)
    with pytest.raises(CrosswalkError, match="overrides file not found"):
        fetch_nba_crosswalk([], stats, "2025-26", overrides=tmp_path / "missing.csv")


def test_fetch_degrades_rather_than_fails_when_a_source_is_down(tmp_path: Path, overrides_file: Path) -> None:
    clock = FakeClock()
    stats = make_stats(
        {"2026-27": SourceError("Akamai read timeout"), "2025-26": SourceError("Akamai read timeout")}, clock, tmp_path
    )
    darko = make_darko(lambda request: httpx.Response(500), clock, tmp_path)

    fetched = fetch_nba_crosswalk(pool_players(), stats, ("2026-27", "2025-26"), darko=darko, overrides=overrides_file)
    assert fetched.degraded and not fetched.cached and fetched.raw_path is None
    assert (fetched.source, fetched.as_of) == ("nba_stats", AS_OF)  # nothing fetched: the adapter's clock
    assert [warning.split(" (")[0] for warning in fetched.warnings[:3]] == [
        "nba_stats 2026-27: unavailable, matching without it",
        "nba_stats 2025-26: unavailable, matching without it",
        "darko: unavailable, matching without it",
    ]
    # The bundled table still names everyone in the pool, without teams to check.
    assert fetched.data.mapped() == set(NBA_IDS) and fetched.warnings[3:] == fetched.data.warnings == ()
