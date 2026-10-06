"""News relevance: the relevant players of a league, storing news with player matching, and the relevance filter.

Most tests seed a store by hand: an NFL league (the PPR settings fixture) in week 5 with three rosters, ESPN season
projections for the wire, and open and closed trade proposals. The news is the trimmed real NFL captures in
tests/fixtures/sources/news (2026-10-05), whose players the seeded rosters and wire use. The last test syncs the
fixture NFL league (tests/fixtures/espn, week 4) through ``fm.jobs.sync.sync_league`` and finds its roster's news.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from fm.config import League
from fm.espn.client import FILTER_HEADER, EspnClient
from fm.espn.models import MatchupsView
from fm.espn.settings import LeagueSettings, load_league_settings
from fm.jobs.sync import sync_league
from fm.model.relevance import (
    DEFAULT_FREE_AGENTS,
    DUPLICATE_WINDOW,
    PlayerIndex,
    RelevanceReason,
    ingest_news,
    opponent_from,
    relevance_for,
    relevant_news,
)
from fm.model.scoring import Scorer
from fm.proposals.payloads import TradePayload, WaiverPayload
from fm.sources.news import ESPN_NEWS, ROTOWIRE, NewsItem, parse_espn_news, parse_rotowire_rss
from fm.store import (
    LeagueRow,
    LeagueSettingsRow,
    NewsItemRow,
    PlayerRow,
    ProjectionRow,
    ProposalRow,
    RosterEntryRow,
    Sport,
    Store,
    TeamRow,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NEWS = FIXTURES / "sources" / "news"
PPR = FIXTURES / "espn" / "ffl_settings_ppr.json"
NINE_CAT = FIXTURES / "espn" / "fba_settings_9cat.json"
AS_OF = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
FETCHED = datetime(2026, 10, 5, 23, 0, tzinfo=UTC)
SEASON, WEEK, BENCH = 2026, 5, 20

ROSTER, OPPONENT, FREE_AGENT, TRADE_TARGET = (
    RelevanceReason.ROSTER,
    RelevanceReason.OPPONENT,
    RelevanceReason.FREE_AGENT,
    RelevanceReason.TRADE_TARGET,
)

# Ours (team 1), the opponent (team 2), a third team, and the wire. Positions are ESPN's default position ids.
DANIELS, ALLGEIER, CHASE = 4426348, 4373626, 4362628
MCCONKEY, ALLEN = 4612826, 3918298
WHITE, MCLAURIN, COLEMAN = 4697815, 3121422, 4635008
COUSINS, MIXON, WILSON, DEPTH = 14880, 3116385, 4887558, 9_000_001
ROSTERS = {1: (DANIELS, ALLGEIER, CHASE), 2: (MCCONKEY, ALLEN), 3: (WHITE, MCLAURIN, COLEMAN)}
PLAYERS: dict[int, tuple[str, int]] = {
    DANIELS: ("Jayden Daniels", 1),
    ALLGEIER: ("Tyler Allgeier", 2),
    CHASE: ("Ja'Marr Chase", 3),
    MCCONKEY: ("Ladd McConkey", 3),
    ALLEN: ("Josh Allen", 1),
    WHITE: ("Rachaad White", 2),
    MCLAURIN: ("Terry McLaurin", 3),
    COLEMAN: ("Keon Coleman", 3),
    COUSINS: ("Kirk Cousins", 1),
    MIXON: ("Joe Mixon", 2),
    WILSON: ("Emanuel Wilson", 2),
    DEPTH: ("Depth Back", 2),
}
# ESPN season projections (PPR points: Cousins 228, Mixon 149, Wilson 99, Depth Back 5); Chase is rostered.
SEASON_LINES: dict[int, dict[str, float]] = {
    COUSINS: {"PY": 3800.0, "PTD": 24.0, "INTT": 10.0},
    MIXON: {"RY": 700.0, "RTD": 6.0, "REC": 25.0, "REY": 180.0},
    WILSON: {"RY": 500.0, "RTD": 4.0, "REC": 15.0, "REY": 100.0},
    DEPTH: {"RY": 50.0},
    CHASE: {"REC": 100.0, "REY": 1400.0, "RETD": 10.0},
}


def player(espn_id: int, name: str, position_id: int, sport: Sport = "nfl") -> PlayerRow:
    return PlayerRow(sport=sport, espn_id=espn_id, full_name=name, default_position_id=position_id, as_of=AS_OF)


def line(
    espn_id: int,
    stats: Mapping[str, float],
    *,
    period: int = 0,
    source: str = "espn",
    sport: Sport = "nfl",
    season: int = SEASON,
) -> ProjectionRow:
    return ProjectionRow(
        sport=sport,
        espn_id=espn_id,
        source=source,
        season=season,
        scoring_period_id=period,
        stats=dict(stats),
        as_of=AS_OF,
    )


def seed_league(
    store: Store,
    *,
    sport: Sport = "nfl",
    settings: LeagueSettings | None = None,
    rosters: Mapping[int, tuple[int, ...]] | None = None,
    season: int = SEASON,
    period: int = WEEK,
) -> LeagueRow:
    """A league with ``rosters`` (team -> players) for ``period``; our team is 1. No rosters means no sync yet."""
    league = store.leagues.upsert(
        LeagueRow(key=sport, sport=sport, espn_league_id=1234567, season=season, team_id=1, as_of=AS_OF)
    )
    if settings is not None:
        store.settings.upsert(
            LeagueSettingsRow(league_id=league.row_id, settings=settings.model_dump(mode="json"), as_of=AS_OF)
        )
    for team_id, espn_ids in (rosters or {}).items():
        store.teams.upsert(TeamRow(league_id=league.row_id, team_id=team_id, name=f"Team {team_id}", as_of=AS_OF))
        store.rosters.replace(
            league.row_id,
            period,
            team_id,
            [
                RosterEntryRow(
                    league_id=league.row_id,
                    scoring_period_id=period,
                    team_id=team_id,
                    espn_id=espn_id,
                    lineup_slot_id=BENCH,
                    as_of=AS_OF,
                )
                for espn_id in espn_ids
            ],
        )
    return league


def propose_trade(
    store: Store, league: LeagueRow, *, get: tuple[int, ...], give: tuple[int, ...] = (), status: str = "proposed"
) -> ProposalRow:
    payload = TradePayload(other_team_id=3, give_espn_ids=give, get_espn_ids=get).model_dump(mode="json")
    return store.proposals.insert(
        ProposalRow.model_validate(
            {
                "league_id": league.row_id,
                "kind": "trade_propose",
                "status": status,
                "policy": "approve",
                "payload": payload,
                "created_by": "test",
                "created_at": AS_OF,
            }
        )
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


@pytest.fixture
def league(store: Store) -> LeagueRow:
    """The NFL league: rosters, players, the wire's ESPN lines, one open and one rejected trade, a waiver claim."""
    league = seed_league(store, settings=load_league_settings(PPR), rosters=ROSTERS)
    store.players.upsert_many(player(espn_id, name, position) for espn_id, (name, position) in PLAYERS.items())
    store.projections.upsert_many(line(espn_id, stats) for espn_id, stats in SEASON_LINES.items())
    store.projections.upsert(line(9_000_002, {"RY": 900.0}, source="sleeper"))  # not ESPN's: not on the wire here
    propose_trade(store, league, give=(CHASE,), get=(MCLAURIN,))
    propose_trade(store, league, get=(WHITE,), status="rejected")
    store.proposals.insert(
        ProposalRow(
            league_id=league.row_id,
            kind="waiver",
            policy="approve",
            payload=WaiverPayload(add_espn_id=DEPTH).model_dump(mode="json"),
            created_by="test",
            created_at=AS_OF,
        )
    )
    return league


def news_items() -> list[NewsItem]:
    """The NFL captures as the feeds hand them over: ESPN's and RotoWire's items, stamped with the poll time."""
    espn = parse_espn_news((NEWS / "espn_news_nfl.json").read_bytes(), "nfl")
    roto = parse_rotowire_rss((NEWS / "rotowire_nfl.xml").read_bytes(), "nfl")
    return [replace(entry, fetched_at=FETCHED) for entry in (*espn, *roto)]


def by_id(rows: Iterable[NewsItemRow]) -> dict[str, NewsItemRow]:
    return {row.external_id: row for row in rows}


# --- who is relevant --------------------------------------------------------------------------------------------------


def test_every_reason_is_tagged(store: Store, league: LeagueRow) -> None:
    relevance = relevance_for(store, league.row_id, opponent_team_id=2, free_agents=3, trade_targets=[COLEMAN])

    assert (relevance.league_id, relevance.key, relevance.sport, relevance.team_id) == (league.row_id, "nfl", "nfl", 1)
    assert (relevance.scoring_period_id, relevance.opponent_team_id, relevance.warnings) == (WEEK, 2, ())
    assert relevance.ids(ROSTER) == tuple(sorted(ROSTERS[1]))
    assert relevance.ids(OPPONENT) == tuple(sorted(ROSTERS[2]))
    assert relevance.ids(FREE_AGENT) == tuple(sorted((COUSINS, MIXON, WILSON)))  # Depth Back is fourth of four
    assert relevance.ids(TRADE_TARGET) == tuple(sorted((MCLAURIN, COLEMAN)))
    assert relevance.why(CHASE) == {ROSTER}  # offered in the open trade, but ours
    assert relevance.why(MCLAURIN) == {TRADE_TARGET} and relevance.why(COLEMAN) == {TRADE_TARGET}
    for outsider in (WHITE, DEPTH, 9_000_002):  # a rejected trade's target, the fourth free agent, a Sleeper-only line
        assert outsider not in relevance and relevance.why(outsider) == frozenset()
    assert set(relevance.ids()) == set(relevance.players) and relevance.name(DANIELS) == "Jayden Daniels"
    assert relevance.name(123) == "ESPN 123"
    assert relevance.describe() == (
        "nfl (scoring period 5): 10 relevant players: 3 roster, 2 opponent (team 2), 3 free agents, 2 trade targets"
    )


def test_reasons_accumulate(store: Store, league: LeagueRow) -> None:
    relevance = relevance_for(store, league.row_id, opponent_team_id=3, free_agents=0, trade_targets=[DANIELS])
    assert relevance.why(DANIELS) == {ROSTER, TRADE_TARGET}
    assert relevance.why(MCLAURIN) == {OPPONENT, TRADE_TARGET}  # on the opponent's roster and in our open trade
    assert relevance.ids(FREE_AGENT) == ()


def test_free_agents_are_the_best_unrostered_players_by_league_points(store: Store, league: LeagueRow) -> None:
    relevance = relevance_for(store, league.row_id, free_agents=DEFAULT_FREE_AGENTS)
    assert relevance.ids(FREE_AGENT) == tuple(sorted((COUSINS, MIXON, WILSON, DEPTH)))  # every one of them fits
    assert CHASE not in relevance.ids(FREE_AGENT)  # projected far above them, but rostered
    assert relevance_for(store, league.row_id, free_agents=1).ids(FREE_AGENT) == (COUSINS,)
    assert relevance_for(store, league.row_id, free_agents=2).ids(FREE_AGENT) == tuple(sorted((COUSINS, MIXON)))


def test_the_ranking_follows_the_league_scoring(store: Store) -> None:
    catcher, runner = 9_100_001, 9_100_002
    ppr = load_league_settings(PPR)
    no_ppr = ppr.model_copy(
        update={
            "scoring_items": tuple(
                item.model_copy(update={"points": 0.0}) if item.stat == "REC" else item for item in ppr.scoring_items
            )
        }
    )
    lines = {catcher: {"REC": 60.0, "REY": 500.0}, runner: {"RY": 900.0}}  # PPR 110 v 90; without PPR 50 v 90
    for settings, best in ((ppr, catcher), (no_ppr, runner)):
        with Store.open() as fresh:
            league = seed_league(fresh, settings=settings, rosters={1: (DANIELS,)})
            fresh.projections.upsert_many(line(espn_id, stats) for espn_id, stats in lines.items())
            assert relevance_for(fresh, league.row_id, free_agents=1).ids(FREE_AGENT) == (best,)
    assert Scorer(ppr).points(lines[catcher]) == pytest.approx(110.0)


def test_a_period_line_stands_in_for_a_missing_season_line(store: Store, league: LeagueRow) -> None:
    week_only = 9_000_003
    store.projections.upsert(line(week_only, {"RY": 5000.0}, period=WEEK))  # huge, but a week is not a season
    relevance = relevance_for(store, league.row_id, free_agents=5)
    assert week_only in relevance.ids(FREE_AGENT)
    assert relevance_for(store, league.row_id, free_agents=4).ids(FREE_AGENT) == tuple(
        sorted((COUSINS, MIXON, WILSON, DEPTH))
    )


def test_without_settings_free_agents_are_not_ranked(store: Store) -> None:
    league = seed_league(store, rosters={1: (DANIELS,)})
    store.projections.upsert_many(line(espn_id, stats) for espn_id, stats in SEASON_LINES.items())
    relevance = relevance_for(store, league.row_id, free_agents=2)
    assert relevance.ids(FREE_AGENT) == tuple(sorted((COUSINS, MIXON, WILSON, DEPTH, CHASE)))[:2]  # by id
    assert relevance.warnings == ("nfl: league settings are not synced; free agents are not ranked",)


def test_category_leagues_rank_by_volume_weighted_z_scores(store: Store) -> None:
    volume, tiny, brick, careful = 9_200_001, 9_200_002, 9_200_003, 9_200_004
    base = {"PTS": 1000.0, "REB": 400.0, "AST": 200.0, "STL": 60.0, "BLK": 40.0, "3PM": 80.0, "FTM": 150.0}
    base |= {"FTA": 200.0, "TO": 150.0}
    lines = {
        volume: base | {"FGM": 500.0, "FGA": 1000.0},  # 50% on a thousand shots
        tiny: base | {"FGM": 10.0, "FGA": 10.0},  # 100%, on ten
        brick: base | {"FGM": 300.0, "FGA": 700.0},  # 43% on seven hundred
        careful: base | {"FGM": 500.0, "FGA": 1000.0, "TO": 140.0},  # the volume shooter, with fewer turnovers
    }
    league = seed_league(store, sport="nba", settings=load_league_settings(NINE_CAT), rosters={1: (1,)}, season=2027)
    store.projections.upsert_many(line(espn_id, stats, sport="nba", season=2027) for espn_id, stats in lines.items())

    def best(count: int) -> tuple[int, ...]:
        return relevance_for(store, league.row_id, free_agents=count).ids(FREE_AGENT)

    assert best(1) == (careful,)  # TO is a reverse category: fewer is better, all else equal
    # Not the perfect shooter: ten shots move a team's FG% far less than a thousand at 50% (raw percentages would
    # rank him second here).
    assert best(2) == tuple(sorted((careful, volume)))
    assert best(3) == tuple(sorted((careful, volume, tiny)))  # 43% on seven hundred shots hurts most


def test_before_the_first_sync_only_trade_targets_are_relevant(store: Store) -> None:
    league = seed_league(store, settings=load_league_settings(PPR))
    propose_trade(store, league, get=(MCLAURIN,))
    relevance = relevance_for(store, league.row_id, opponent_team_id=2, trade_targets=[COLEMAN])
    assert relevance.scoring_period_id is None
    assert dict(relevance.reasons) == {MCLAURIN: {TRADE_TARGET}, COLEMAN: {TRADE_TARGET}}
    assert relevance.warnings == ("nfl: no roster snapshot yet (run fm sync); only trade targets are relevant",)
    assert relevance.players == {}  # neither has a stored row yet
    assert relevance.describe().startswith("nfl (no roster yet): 2 relevant players")


def test_missing_rosters_and_unreadable_proposals_are_reported(store: Store, league: LeagueRow) -> None:
    store.proposals.insert(
        ProposalRow(
            league_id=league.row_id,
            kind="trade_propose",
            policy="approve",
            payload={"other_team_id": 3},  # a trade with no players fails its payload model
            created_by="test",
            created_at=AS_OF,
        )
    )
    relevance = relevance_for(store, league.row_id, opponent_team_id=7, free_agents=0)
    assert relevance.ids(TRADE_TARGET) == (MCLAURIN,)
    assert relevance.warnings[0] == "nfl: team 7 has no roster in scoring period 5"
    assert relevance.warnings[1].startswith("proposal ") and "payload does not load" in relevance.warnings[1]
    other_period = relevance_for(store, league.row_id, scoring_period_id=4, free_agents=0)
    assert other_period.ids(ROSTER) == () and "team 1 has no roster in scoring period 4" in other_period.warnings[0]


def test_bad_arguments(store: Store, league: LeagueRow) -> None:
    with pytest.raises(LookupError, match="no league with id 999"):
        relevance_for(store, 999)
    with pytest.raises(ValueError, match="free_agents"):
        relevance_for(store, league.row_id, free_agents=-1)
    with pytest.raises(ValueError, match="our own team"):
        relevance_for(store, league.row_id, opponent_team_id=1)


def test_opponent_from_matchups() -> None:
    matchups = MatchupsView.model_validate(json.loads((FIXTURES / "espn" / "ffl_matchups.json").read_text("utf-8")))
    assert matchups.status is not None and matchups.status.current_matchup_period == 4
    assert opponent_from(matchups, 1) == 2 and opponent_from(matchups, 2) == 1  # week 4, ESPN's current matchup
    assert opponent_from(matchups, 3, matchup_period=3) == 4
    assert opponent_from(matchups, 2, matchup_period=15) is None  # a bye: the matchup has no away side
    assert opponent_from(matchups, 11) is None  # no matchup for that team
    assert opponent_from(matchups.model_copy(update={"status": None}), 1) is None  # no current period known
    assert opponent_from(matchups.model_copy(update={"status": None}), 1, matchup_period=4) == 2


# --- names ------------------------------------------------------------------------------------------------------------


def test_player_index_matches_names_as_sources_spell_them() -> None:
    index = PlayerIndex(
        "nfl",
        [
            player(1, "Paris Johnson Jr.", 1),
            player(2, "D.J. Moore", 3),
            player(3, "Mike Williams", 3),
            player(4, "Mike Williams", 3),
            player(5, "Amon-Ra St. Brown", 3),
        ],
    )
    assert index.resolve("Paris Johnson") == (1,) and index.resolve("paris johnson jr") == (1,)
    assert index.resolve("DJ Moore") == (2,)
    assert index.resolve("Mike Williams") == (3, 4)  # both, for the triage to tell apart
    assert index.resolve("Amon-Ra St. Brown") == (5,)
    assert index.resolve("Nobody Here") == ()
    item = NewsItem(
        source=ROTOWIRE, external_id="x", sport="nfl", title="t", published_at=AS_OF, player_names=("DJ Moore", "x")
    )
    assert index.players_in(item) == (2,)
    assert index.players_in(replace(item, espn_ids=(9,))) == (9,)  # a source's own tags win
    with pytest.raises(ValueError, match="is nba, not nfl"):
        PlayerIndex("nfl", [player(6, "Nikola Jokic", 5, sport="nba")])


def test_player_index_from_store_knows_every_synced_player(store: Store, league: LeagueRow) -> None:
    store.players.upsert(player(9_000_009, "Never Synced Here", 2))  # no roster, no ESPN line: not on the wire
    index = PlayerIndex.from_store(store, "nfl")
    for espn_id, (name, _) in PLAYERS.items():
        assert index.resolve(name) == (espn_id,)
    assert index.resolve("Never Synced Here") == ()
    assert PlayerIndex.from_store(store, "nba").resolve("Jayden Daniels") == ()


# --- storing ----------------------------------------------------------------------------------------------------------


def test_ingest_stores_each_item_once_with_its_players(store: Store, league: LeagueRow) -> None:
    result = ingest_news(store, news_items(), now=FETCHED + timedelta(hours=1))
    assert len(result.stored) == 10 and (result.known, result.repeats) == (0, 0)
    published = [row.published_at for row in result.stored]
    assert published == sorted(published) and all(row.id is not None for row in result.stored)
    rows = by_id(result.stored)
    assert rows["50108832"].espn_ids == [COUSINS, WILSON, COLEMAN, ALLGEIER]  # ESPN's tags, whoever they are
    assert rows["50112664"].espn_ids == [4568652]  # a lineman no sync wrote: ESPN tagged him, so he stays
    assert rows["nfl640904"].espn_ids == [DANIELS]  # RotoWire's title, matched by name
    assert rows["nfl640907"].espn_ids == [MCCONKEY]
    assert rows["nfl640903"].espn_ids == []  # Marcus Mariota: no sync wrote him
    assert {row.external_id for row in result.unmatched} == {"nfl640903", "50112291"}  # and McVay's team news
    assert all(row.fetched_at == FETCHED and row.triaged_at is None for row in result.stored)
    assert rows["nfl640904"].body is not None and rows["nfl640904"].url is not None
    assert result.describe() == "news: 10 stored (2 about no known player), 0 already stored, 0 repeats"

    again = ingest_news(store, news_items())
    assert (again.stored, again.known, again.repeats) == ((), 10, 0)
    assert len(store.news.untriaged()) == 10


def test_ingest_stamps_items_without_a_fetch_time(store: Store, league: LeagueRow) -> None:
    unstamped = replace(news_items()[0], fetched_at=None)
    (row,) = ingest_news(store, [unstamped], now=FETCHED + timedelta(hours=1)).stored
    assert row.fetched_at == FETCHED + timedelta(hours=1)
    assert ingest_news(store, []) == ingest_news(store, ())


def test_ingest_drops_a_repeated_story(store: Store, league: LeagueRow) -> None:
    original = next(entry for entry in news_items() if entry.external_id == "nfl640904")
    ingest_news(store, [original])
    reposted = replace(original, external_id="nfl640999", published_at=original.published_at + timedelta(hours=5))
    elsewhere = replace(original, source=ESPN_NEWS, external_id="1", published_at=original.published_at)
    later = replace(original, external_id="nfl641000", published_at=original.published_at + DUPLICATE_WINDOW * 2)
    nba = replace(original, sport="nba", external_id="nba1")
    result = ingest_news(store, [reposted, elsewhere, later, nba])
    assert result.repeats == 2 and result.known == 0
    assert [row.external_id for row in result.stored] == ["nba1", "nfl641000"]  # another sport; outside the window


def test_ingest_counts_repeats_and_twins_within_a_batch(store: Store, league: LeagueRow) -> None:
    original = next(entry for entry in news_items() if entry.external_id == "nfl640905")
    repost = replace(original, external_id="nfl640999", published_at=original.published_at + timedelta(minutes=20))
    edited = replace(original, title="Terry McLaurin: Edited")  # the same id again: the first copy given wins
    result = ingest_news(store, [repost, original, edited])
    assert [row.external_id for row in result.stored] == ["nfl640905"]  # the earlier of the two copies
    assert (result.known, result.repeats) == (1, 1)
    assert result.stored[0].title == "Terry McLaurin: Wednesday activity TBD"


def test_shared_names_get_every_namesake(store: Store, league: LeagueRow) -> None:
    store.players.upsert_many([player(9_300_001, "Mike Williams", 3), player(9_300_002, "Mike Williams", 3)])
    store.projections.upsert_many([line(9_300_001, {"REY": 1.0}), line(9_300_002, {"REY": 2.0})])
    item = NewsItem(
        source=ROTOWIRE,
        external_id="nfl1",
        sport="nfl",
        title="Mike Williams: Limited Wednesday",
        published_at=AS_OF,
        player_names=("Mike Williams",),
    )
    (row,) = ingest_news(store, [item]).stored
    assert row.espn_ids == [9_300_001, 9_300_002]


# --- filtering --------------------------------------------------------------------------------------------------------


def test_relevant_news_keeps_items_about_relevant_players(store: Store, league: LeagueRow) -> None:
    ingest_news(store, news_items())
    relevance = relevance_for(store, league.row_id, opponent_team_id=2, free_agents=3, trade_targets=[COLEMAN])
    found = relevant_news(store.news.untriaged(), relevance)

    assert [match.item.external_id for match in found] == [
        "50108832",  # the free-agent pickups story: Cousins and Wilson on the wire, Coleman, our Allgeier
        "nfl640904",  # Jayden Daniels, ours
        "50112493",  # Joe Mixon, on the wire
        "nfl640905",  # Terry McLaurin, whom our open trade asks for
        "nfl640907",  # Ladd McConkey, this week's opponent
    ]
    pickups = found[0]
    assert dict(pickups.players) == {
        COUSINS: {FREE_AGENT},
        WILSON: {FREE_AGENT},
        COLEMAN: {TRADE_TARGET},
        ALLGEIER: {ROSTER},
    }
    assert pickups.espn_ids == (COUSINS, WILSON, COLEMAN, ALLGEIER)
    assert pickups.reasons == {FREE_AGENT, TRADE_TARGET, ROSTER}
    assert [match.reasons for match in found[1:]] == [{ROSTER}, {FREE_AGENT}, {TRADE_TARGET}, {OPPONENT}]

    without_opponent = relevant_news(store.news.untriaged(), relevance_for(store, league.row_id, free_agents=3))
    assert "nfl640907" not in [match.item.external_id for match in without_opponent]


def test_rotowire_items_stored_before_their_player_was_known_match_by_title(store: Store, league: LeagueRow) -> None:
    ingest_news(store, news_items(), index=PlayerIndex("nfl", []))  # nobody known at ingest time
    rows = by_id(store.news.untriaged())
    assert rows["nfl640904"].espn_ids == [] and rows["50108832"].espn_ids == [COUSINS, WILSON, COLEMAN, ALLGEIER]
    relevance = relevance_for(store, league.row_id, opponent_team_id=2, free_agents=0)
    found = {match.item.external_id: dict(match.players) for match in relevant_news(store.news.untriaged(), relevance)}
    assert found == {
        "50108832": {ALLGEIER: {ROSTER}},  # ESPN's tags were stored either way
        "nfl640904": {DANIELS: {ROSTER}},
        "nfl640905": {MCLAURIN: {TRADE_TARGET}},
        "nfl640907": {MCCONKEY: {OPPONENT}},
    }


def test_relevant_news_skips_other_sports(store: Store, league: LeagueRow) -> None:
    nba = NewsItemRow(
        source=ESPN_NEWS,
        external_id="1",
        sport="nba",
        title="Same id, other sport",
        espn_ids=[DANIELS],
        published_at=AS_OF,
        fetched_at=AS_OF,
    )
    assert relevant_news([nba], relevance_for(store, league.row_id)) == []


# --- end to end over a fixture sync -----------------------------------------------------------------------------------


def espn_json(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / "espn" / name).read_text(encoding="utf-8"))


def fixture_league_transport() -> httpx.MockTransport:
    """ESPN's views of the fixture NFL league (1234567, 2026 week 4); player cards answer exactly the ids asked."""
    views = {
        "mSettings": "ffl_settings_ppr.json",
        "mTeam+mStandings": "ffl_teams.json",
        "mRoster": "ffl_rosters_week4.json",
    }
    wire = espn_json("ffl_free_agents_week4.json")["players"]
    cards: dict[int, dict[str, Any]] = {entry["id"]: entry for entry in wire}
    for team in espn_json("ffl_rosters_week4.json")["teams"]:
        for entry in team["roster"]["entries"]:
            cards[entry["playerId"]] = {**entry["playerPoolEntry"], "id": entry["playerId"], "onTeamId": team["id"]}
    cards |= {entry["id"]: entry for entry in espn_json("ffl_player_cards_week4.json")["players"]}
    envelope = {"gameId": 1, "id": 1234567, "seasonId": SEASON, "scoringPeriodId": 4}

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.host == "lm-api-reads.fantasy.espn.com"
        view = "+".join(request.url.params.get_list("view"))
        if view == "kona_player_info":
            return httpx.Response(200, json={**envelope, "players": wire})
        if view == "kona_playercard":
            wanted = json.loads(request.headers[FILTER_HEADER])["players"]["filterIds"]["value"]
            return httpx.Response(200, json={**envelope, "players": [cards[i] for i in wanted if i in cards]})
        return httpx.Response(200, json=espn_json(views[view]))

    return httpx.MockTransport(handle)


def test_a_synced_league_finds_its_rosters_news(store: Store) -> None:
    config = League(key="nfl", sport="nfl", espn_league_id=1234567, season=SEASON, team_id=1)
    with (
        httpx.Client(transport=fixture_league_transport()) as http,
        EspnClient.for_league(
            config, None, client=http, capture=False, sleep=lambda _: None, min_interval_s=0.0
        ) as espn,
    ):
        synced = sync_league(store, config, espn)
    matchups = MatchupsView.model_validate(espn_json("ffl_matchups.json"))

    relevance = relevance_for(store, synced.league_id, opponent_team_id=opponent_from(matchups, 1), free_agents=2)
    roster = {entry["playerId"] for entry in espn_json("ffl_rosters_week4.json")["teams"][0]["roster"]["entries"]}
    assert relevance.scoring_period_id == 4 and relevance.warnings == ()
    assert set(relevance.ids(ROSTER)) == roster and len(roster) == 11
    assert relevance.ids(OPPONENT) == (15847, 3916387, 3929630)  # Kelce, Lamar Jackson, Saquon Barkley
    settings = load_league_settings(PPR)
    wire = {
        entry["id"]: Scorer(settings).points(row.stats)
        for entry in espn_json("ffl_free_agents_week4.json")["players"]
        if (row := store.projections.get("nfl", entry["id"], "espn", SEASON, 0)) is not None
    }
    assert len(wire) == 4  # every pooled player has an ESPN season projection
    assert relevance.ids(FREE_AGENT) == tuple(sorted(sorted(wire, key=wire.__getitem__, reverse=True)[:2]))

    ingest_news(store, news_items())
    (match,) = relevant_news(store.news.untriaged(), relevance)
    assert match.item.external_id == "50108832" and dict(match.players) == {ALLGEIER: {ROSTER}}  # on our bench
