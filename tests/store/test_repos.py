"""Repositories: a round trip per table (write, read back equal), upsert semantics, and the targeted queries."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fm.store import (
    REPOSITORIES,
    Availability,
    DecisionEval,
    Execution,
    League,
    LeagueSettingsRecord,
    LlmUsage,
    MarketValue,
    NewsItem,
    NewsSignal,
    Player,
    PlayerId,
    Projection,
    Proposal,
    RawSnapshot,
    RosterEntry,
    Sport,
    Store,
    Team,
    format_timestamp,
)

EDT = timezone(timedelta(hours=-4))
AS_OF = datetime(2026, 10, 4, 9, 30, 15, 123456, tzinfo=EDT)
LATER = AS_OF + timedelta(hours=1)
V1_TABLES = frozenset(
    {
        "leagues",
        "league_settings",
        "teams",
        "players",
        "player_ids",
        "roster_snapshots",
        "projections",
        "availability",
        "news_items",
        "news_signals",
        "market_values",
        "proposals",
        "executions",
        "decision_evals",
        "llm_usage",
        "raw_snapshots",
    }
)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def count(store: Store, table: str) -> int:
    row = store.db.one(f"SELECT COUNT(*) AS n FROM {table}")
    assert row is not None
    return row["n"]


def seed_league(
    store: Store, *, key: str = "nfl", sport: Sport = "nfl", espn_league_id: int = 123456, season: int = 2026
) -> League:
    return store.leagues.upsert(
        League(
            key=key,
            sport=sport,
            espn_league_id=espn_league_id,
            season=season,
            team_id=4,
            name="Test League",
            as_of=AS_OF,
        )
    )


def seed_team(store: Store, league: League, team_id: int = 4) -> Team:
    return store.teams.upsert(Team(league_id=league.row_id, team_id=team_id, name=f"Team {team_id}", as_of=AS_OF))


def seed_proposal(store: Store, league: League) -> Proposal:
    return store.proposals.insert(
        Proposal(
            league_id=league.row_id,
            kind="lineup",
            policy="approve",
            payload={"moves": []},
            created_by="decide.lineup",
            created_at=AS_OF,
        )
    )


def seed_news(store: Store, external_id: str = "rw-1001") -> NewsItem:
    item = store.news.ingest(
        NewsItem(
            source="rotowire",
            external_id=external_id,
            sport="nfl",
            title="Mahomes limited in practice",
            body="Listed as limited with an ankle issue.",
            url="https://www.rotowire.com/football/news/1001",
            espn_ids=[3139477],
            published_at=AS_OF,
            fetched_at=LATER,
        )
    )
    assert item is not None
    return item


def test_every_table_has_a_repository_whose_model_matches_its_columns(store: Store) -> None:
    assert {cls.table for cls in REPOSITORIES} == V1_TABLES
    for cls in REPOSITORIES:
        columns = [row["name"] for row in store.db.all(f"PRAGMA table_info({cls.table})")]
        assert columns == list(cls.columns()), cls.table
        assert set(cls.key) <= set(columns), cls.table


def test_league_round_trip_upsert_and_lookup(store: Store) -> None:
    league = seed_league(store)
    assert league.id == 1
    assert league.as_of == AS_OF and league.as_of.tzinfo == UTC
    assert store.leagues.get(league.row_id) == league
    assert store.leagues.get(99) is None

    renamed = store.leagues.upsert(league.model_copy(update={"id": 999, "name": "Renamed", "team_id": 7}))
    assert (renamed.id, renamed.name, renamed.team_id) == (league.id, "Renamed", 7)
    assert count(store, "leagues") == 1

    nba = seed_league(store, key="nba", sport="nba", espn_league_id=55, season=2027)
    assert store.leagues.by_key("nba") == nba
    assert store.leagues.by_key("nfl") == renamed
    assert store.leagues.by_key("mlb") is None
    assert store.leagues.all() == [nba, renamed]

    next_season = seed_league(store, key="nfl", season=2027)
    assert store.leagues.by_key("nfl") == next_season
    assert count(store, "leagues") == 3


def test_league_settings_round_trip(store: Store) -> None:
    league = seed_league(store)
    raw = store.raw_snapshots.insert(
        RawSnapshot(
            source="espn", kind="mSettings", league_id=league.row_id, path="espn/mSettings.json", fetched_at=AS_OF
        )
    )
    record = LeagueSettingsRecord(
        league_id=league.row_id,
        settings={"scoring": {"53": 1.0, "42": 0.04}, "slots": {"0": 1, "2": 2, "23": 1}, "faab": True},
        raw_snapshot_id=raw.row_id,
        as_of=AS_OF,
    )
    assert store.settings.upsert(record) == record
    assert store.settings.get(league.row_id) == record
    assert store.settings.get(99) is None

    updated = store.settings.upsert(record.model_copy(update={"settings": {"scoring": {}}, "as_of": LATER}))
    assert store.settings.get(league.row_id) == updated
    assert (updated.settings, updated.as_of) == ({"scoring": {}}, LATER)
    assert count(store, "league_settings") == 1

    store.db.execute("DELETE FROM raw_snapshots WHERE id = ?", (raw.row_id,))
    detached = store.settings.get(league.row_id)
    assert detached is not None and detached.raw_snapshot_id is None


def test_team_round_trip(store: Store) -> None:
    league = seed_league(store)
    team = Team(
        league_id=league.row_id,
        team_id=4,
        name="Jon's Team",
        abbrev="JT",
        division_id=1,
        wins=3,
        losses=1,
        ties=0,
        points_for=412.52,
        points_against=380.1,
        playoff_seed=2,
        waiver_rank=5,
        acquisition_budget_spent=37,
        as_of=AS_OF,
    )
    assert store.teams.upsert(team) == team
    rival = store.teams.upsert(Team(league_id=league.row_id, team_id=1, name="Rival", as_of=AS_OF))
    assert store.teams.get(league.row_id, 4) == team
    assert store.teams.get(league.row_id, 99) is None
    assert store.teams.for_league(league.row_id) == [rival, team]

    assert store.teams.upsert_many([team.model_copy(update={"wins": 4}), rival]) == 2
    refreshed = store.teams.get(league.row_id, 4)
    assert refreshed is not None and refreshed.wins == 4
    assert count(store, "teams") == 2


def test_player_round_trip_and_bulk_lookup(store: Store) -> None:
    player = Player(
        sport="nfl",
        espn_id=3139477,
        full_name="Patrick Mahomes",
        default_position_id=1,
        position="QB",
        pro_team_id=12,
        pro_team="KC",
        eligible_slot_ids=[0, 7, 20, 21],
        injury_status="ACTIVE",
        injured=False,
        active=True,
        as_of=AS_OF,
    )
    assert store.players.upsert(player) == player
    assert store.players.get("nfl", 3139477) == player
    assert store.players.get("nba", 3139477) is None

    many = [Player(sport="nfl", espn_id=i, full_name=f"Player {i}", as_of=AS_OF) for i in range(1, 1202)]
    assert store.players.upsert_many(many) == 1201
    assert [p.espn_id for p in store.players.many("nfl", range(1, 1202))] == list(range(1, 1202))
    assert store.players.many("nfl", [1, 999_999]) == [many[0]]
    assert store.players.many("nfl", []) == []

    hurt = store.players.upsert(player.model_copy(update={"injury_status": "QUESTIONABLE", "injured": True}))
    assert hurt.injured is True
    assert store.players.get("nfl", 3139477) == hurt
    assert count(store, "players") == 1202


def test_player_ids_crosswalk(store: Store) -> None:
    rows = [
        PlayerId(sport="nfl", espn_id=1, source="gsis", source_id="00-0033873", origin="ff_playerids", as_of=AS_OF),
        PlayerId(sport="nfl", espn_id=1, source="sleeper", source_id="4046", origin="ff_playerids", as_of=AS_OF),
        PlayerId(sport="nfl", espn_id=2, source="sleeper", source_id="6794", origin="override", as_of=AS_OF),
    ]
    assert store.player_ids.upsert_many(rows) == 3
    assert store.player_ids.for_player("nfl", 1) == [rows[0], rows[1]]
    assert store.player_ids.lookup("nfl", "sleeper", "4046") == rows[1]
    assert store.player_ids.lookup("nfl", "sleeper", "nope") is None
    assert store.player_ids.unmapped("nfl", "sleeper", [1, 2, 3, 4]) == {3, 4}
    assert store.player_ids.unmapped("nfl", "gsis", [1, 2]) == {2}
    assert store.player_ids.unmapped("nfl", "gsis", []) == set()
    assert store.player_ids.unmapped("nba", "sleeper", [1]) == {1}

    with pytest.raises(sqlite3.IntegrityError):  # a source id maps to exactly one ESPN player
        store.player_ids.upsert(
            PlayerId(sport="nfl", espn_id=3, source="sleeper", source_id="4046", origin="name_match", as_of=AS_OF)
        )
    moved = store.player_ids.upsert(rows[1].model_copy(update={"source_id": "4047", "origin": "override"}))
    assert store.player_ids.for_player("nfl", 1) == [rows[0], moved]
    assert count(store, "player_ids") == 3


def test_roster_snapshot_replace(store: Store) -> None:
    league = seed_league(store)
    team = seed_team(store, league)

    def entry(espn_id: int, slot: int, **overrides: object) -> RosterEntry:
        fields: dict[str, object] = {
            "league_id": league.row_id,
            "scoring_period_id": 4,
            "team_id": team.team_id,
            "espn_id": espn_id,
            "lineup_slot_id": slot,
            "as_of": AS_OF,
        }
        fields.update(overrides)
        return RosterEntry.model_validate(fields)

    first = [
        entry(10, 0, acquisition_type="DRAFT", acquisition_date=AS_OF - timedelta(days=30)),
        entry(11, 2),
        entry(12, 20, lineup_locked=True),
    ]
    assert store.rosters.replace(league.row_id, 4, team.team_id, first) == 3
    assert store.rosters.team(league.row_id, 4, team.team_id) == first

    second = [entry(10, 0), entry(13, 2)]
    assert store.rosters.replace(league.row_id, 4, team.team_id, second) == 2
    assert store.rosters.team(league.row_id, 4, team.team_id) == second
    assert store.rosters.rostered_ids(league.row_id, 4) == {10, 13}
    assert store.rosters.league(league.row_id, 4) == second

    store.rosters.replace(league.row_id, 5, team.team_id, [entry(10, 0, scoring_period_id=5)])
    assert store.rosters.latest_period(league.row_id) == 5
    assert store.rosters.latest_period(99) is None
    assert store.rosters.rostered_ids(league.row_id, 4) == {10, 13}

    with pytest.raises(ValueError, match="belongs to"):
        store.rosters.replace(league.row_id, 4, 2, [entry(10, 0)])
    with pytest.raises(sqlite3.IntegrityError):  # unknown team: foreign key, nothing half-written
        store.rosters.replace(league.row_id, 4, 2, [entry(10, 0, team_id=2)])
    assert store.rosters.team(league.row_id, 4, team.team_id) == second
    assert count(store, "roster_snapshots") == 3


def test_projection_round_trip(store: Store) -> None:
    espn = Projection(
        sport="nfl",
        espn_id=1,
        source="espn",
        season=2026,
        scoring_period_id=4,
        stats={"pass_yd": 285.4, "pass_td": 2.1, "int": 0.6},
        as_of=AS_OF,
    )
    sleeper = espn.model_copy(update={"source": "sleeper", "stats": {"pass_yd": 270.0, "pass_td": 1.9}})
    actual = Projection(
        sport="nfl",
        espn_id=1,
        source="espn",
        kind="actual",
        season=2026,
        scoring_period_id=4,
        stats={"pass_yd": 301, "pass_td": 3},
        as_of=LATER,
    )
    season = espn.model_copy(update={"scoring_period_id": 0, "stats": {"pass_yd": 4500.0}})
    assert store.projections.upsert_many([espn, sleeper, actual, season]) == 4

    assert store.projections.get("nfl", 1, "espn", 2026, 4) == espn
    got_actual = store.projections.get("nfl", 1, "espn", 2026, 4, kind="actual")
    assert got_actual == actual
    assert got_actual is not None and got_actual.stats == {"pass_yd": 301.0, "pass_td": 3.0}
    assert store.projections.get("nfl", 1, "darko", 2026, 4) is None

    assert store.projections.for_period("nfl", 2026, 4) == [espn, sleeper]
    assert store.projections.for_period("nfl", 2026, 4, source="sleeper") == [sleeper]
    assert store.projections.for_period("nfl", 2026, 4, kind="actual") == [actual]
    assert store.projections.for_period("nba", 2026, 4) == []
    assert store.projections.for_player("nfl", 1, 2026) == [season, espn, sleeper]

    revised = store.projections.upsert(espn.model_copy(update={"stats": {"pass_yd": 290.0}, "as_of": LATER}))
    assert store.projections.get("nfl", 1, "espn", 2026, 4) == revised
    assert count(store, "projections") == 4


def test_availability_round_trip(store: Store) -> None:
    row = Availability(
        sport="nba",
        espn_id=3917376,
        season=2027,
        scoring_period_id=12,
        designation="QUESTIONABLE",
        p_active=0.65,
        has_game=True,
        game_time=AS_OF + timedelta(hours=10),
        inputs={"designation_source": "espn", "practice": ["DNP", "LP", "FP"], "signals": [3]},
        as_of=AS_OF,
    )
    assert store.availability.upsert(row) == row
    assert store.availability.get("nba", 3917376, 2027, 12) == row
    assert store.availability.get("nba", 3917376, 2027, 13) is None

    no_game = Availability(
        sport="nba", espn_id=1, season=2027, scoring_period_id=12, p_active=0.0, has_game=False, as_of=AS_OF
    )
    assert store.availability.upsert_many([no_game]) == 1
    assert store.availability.for_period("nba", 2027, 12) == [no_game, row]
    assert store.availability.for_period("nba", 2027, 13) == []

    out = store.availability.upsert(row.model_copy(update={"designation": "OUT", "p_active": 0.0}))
    assert store.availability.get("nba", 3917376, 2027, 12) == out
    assert count(store, "availability") == 2

    with pytest.raises(sqlite3.IntegrityError, match="p_active"):
        store.db.execute(
            "INSERT INTO availability (sport, espn_id, season, scoring_period_id, p_active, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("nba", 2, 2027, 12, 1.5, format_timestamp(AS_OF)),
        )


def test_news_ingest_dedupes_and_tracks_triage(store: Store) -> None:
    item = seed_news(store)
    assert item.id == 1 and item.published_at == AS_OF and item.espn_ids == [3139477]
    assert store.news.ingest(item.model_copy(update={"id": None, "title": "Same story, later fetch"})) is None
    assert count(store, "news_items") == 1

    other = store.news.ingest(
        NewsItem(
            source="espn",
            external_id="rw-1001",
            sport="nba",
            title="Out tonight",
            published_at=AS_OF - timedelta(hours=2),
            fetched_at=LATER,
        )
    )
    assert other is not None and other.id == 2
    assert store.news.get(2) == other
    assert store.news.get(3) is None

    assert store.news.untriaged() == [other, item]
    assert store.news.untriaged(limit=1) == [other]
    assert store.news.mark_triaged([1, 2, 42], at=LATER) == 2
    assert store.news.untriaged() == []
    triaged = store.news.get(1)
    assert triaged is not None and triaged.triaged_at == LATER

    assert store.news.published_since(AS_OF - timedelta(hours=1)) == [triaged]
    assert store.news.published_since(AS_OF - timedelta(days=1), sport="nba") == [store.news.get(2)]
    assert store.news.published_since(LATER) == []


def test_news_signal_round_trip_and_cascade(store: Store) -> None:
    item = seed_news(store)
    signal = NewsSignal(
        news_item_id=item.row_id,
        sport="nfl",
        espn_id=3139477,
        kind="injury",
        severity="medium",
        games_out=0,
        p_active_delta=-0.2,
        confidence=0.8,
        summary="Limited practice; expected to play",
        source_url="https://www.rotowire.com/football/news/1001",
        published_at=AS_OF,
        created_at=LATER,
    )
    stored = store.news_signals.insert(signal)
    assert stored == signal.model_copy(update={"id": 1})
    older = store.news_signals.insert(
        signal.model_copy(update={"kind": "role", "published_at": AS_OF - timedelta(days=3)})
    )
    assert store.news_signals.for_player("nfl", 3139477) == [older, stored]
    assert store.news_signals.for_player("nfl", 3139477, since=AS_OF - timedelta(days=1)) == [stored]
    assert store.news_signals.for_player("nfl", 1) == []
    assert store.news_signals.for_item(item.row_id) == [stored, older]

    with pytest.raises(ValidationError):
        NewsSignal.model_validate({**signal.model_dump(), "kind": "vibes"})
    with pytest.raises(sqlite3.IntegrityError):
        store.news_signals.insert(signal.model_copy(update={"news_item_id": 999}))

    store.db.execute("DELETE FROM news_items WHERE id = ?", (item.row_id,))
    assert store.news_signals.for_player("nfl", 3139477) == []


def test_market_value_round_trip(store: Store) -> None:
    fantasycalc = MarketValue(
        sport="nfl",
        espn_id=1,
        source="fantasycalc",
        value=7012.0,
        rank=3,
        position_rank=1,
        trend=-120.5,
        details={"redraft": True, "numQbs": 1},
        as_of=AS_OF,
    )
    espn = MarketValue(
        sport="nfl", espn_id=1, source="espn", rank=5, percent_owned=99.8, percent_started=97.1, as_of=AS_OF
    )
    second = MarketValue(sport="nfl", espn_id=2, source="fantasycalc", value=5000.0, rank=9, as_of=AS_OF)
    assert store.market_values.upsert_many([fantasycalc, espn, second]) == 3
    assert store.market_values.get("nfl", 1, "fantasycalc") == fantasycalc
    assert store.market_values.get("nfl", 1, "espn") == espn
    assert store.market_values.get("nfl", 1, "ktc") is None
    assert store.market_values.for_source("nfl", "fantasycalc") == [fantasycalc, second]

    bumped = store.market_values.upsert(fantasycalc.model_copy(update={"value": 7100.0, "trend": 88.0}))
    assert store.market_values.get("nfl", 1, "fantasycalc") == bumped
    assert count(store, "market_values") == 3


def test_proposal_round_trip_lifecycle_and_queries(store: Store) -> None:
    league = seed_league(store)
    proposal = Proposal(
        league_id=league.row_id,
        kind="lineup",
        policy="approve",
        scoring_period_id=4,
        payload={"moves": [{"playerId": 1, "fromLineupSlotId": 20, "toLineupSlotId": 2}]},
        engine_numbers={"delta_points": 3.4, "p_win_before": 0.52},
        deadline=AS_OF + timedelta(days=3),
        created_by="decide.lineup",
        created_at=AS_OF,
        dedupe_key="lineup:4:1",
    )
    stored = store.proposals.insert(proposal)
    assert stored == proposal.model_copy(update={"id": 1})
    assert store.proposals.get(1) == stored
    assert store.proposals.get(2) is None

    approved = store.proposals.update(
        stored.model_copy(
            update={"status": "approved", "decided_by": "telegram", "decided_at": LATER, "execution_token": "tok-abc"}
        )
    )
    assert store.proposals.get(1) == approved
    assert (approved.status, approved.decided_at) == ("approved", LATER)

    trade = store.proposals.insert(
        Proposal(
            league_id=league.row_id,
            kind="trade_propose",
            policy="approve",
            payload={"give": [1], "get": [2], "team_id": 7},
            created_by="decide.trades",
            created_at=LATER,
        )
    )
    rejected = store.proposals.insert(
        Proposal(
            league_id=league.row_id,
            kind="add_drop",
            status="rejected",
            policy="approve",
            scoring_period_id=4,
            payload={"add": 9, "drop": 8},
            created_by="decide.waivers",
            created_at=LATER + timedelta(hours=1),
            decided_by="cli",
            decided_at=LATER + timedelta(hours=2),
        )
    )
    assert store.proposals.find() == [approved, trade, rejected]
    assert store.proposals.find(league_id=league.row_id, statuses=["approved"]) == [approved]
    assert store.proposals.find(kinds=["trade_propose", "add_drop"]) == [trade, rejected]
    assert store.proposals.find(scoring_period_id=4) == [approved, rejected]
    assert store.proposals.find(league_id=99) == []
    assert store.proposals.open(league.row_id) == [approved, trade]
    assert store.proposals.open() == [approved, trade]

    assert store.proposals.consume_execution_token(1, "wrong", LATER) is False
    assert store.proposals.consume_execution_token(1, "tok-abc", LATER) is True
    assert store.proposals.consume_execution_token(1, "tok-abc", LATER) is False
    consumed = store.proposals.get(1)
    assert consumed is not None and consumed.token_consumed_at == LATER

    with pytest.raises(sqlite3.IntegrityError):  # tokens are unique
        store.proposals.insert(proposal.model_copy(update={"execution_token": "tok-abc"}))
    with pytest.raises(ValueError, match="needs a row with an id"):
        store.proposals.update(proposal)
    with pytest.raises(LookupError, match="no row with id 404"):
        store.proposals.update(stored.model_copy(update={"id": 404}))
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store.proposals.insert(proposal.model_copy(update={"league_id": 999}))
    assert count(store, "proposals") == 3


def test_execution_round_trip(store: Store) -> None:
    league = seed_league(store)
    proposal = seed_proposal(store, league)
    execution = Execution(
        proposal_id=proposal.row_id,
        mode="api",
        started_at=AS_OF,
        request={"type": "ROSTER", "items": [{"playerId": 1, "fromLineupSlotId": 20, "toLineupSlotId": 2}]},
    )
    running = store.executions.insert(execution)
    assert running == execution.model_copy(update={"id": 1})
    assert running.status == "running" and running.artifacts == []

    done = store.executions.update(
        running.model_copy(
            update={
                "status": "verified",
                "finished_at": LATER,
                "response": {"ok": True},
                "verification": {"slots_match": True},
                "artifacts": ["audit/1/request.json", "audit/1/response.json"],
                "espn_transaction_id": "abc123",
            }
        )
    )
    assert store.executions.get(1) == done
    unknown = store.executions.insert(
        Execution(proposal_id=proposal.row_id, mode="ui", status="unknown", started_at=LATER, error="timeout after 30s")
    )
    assert store.executions.for_proposal(proposal.row_id) == [done, unknown]
    assert store.executions.for_proposal(99) == []

    store.db.execute("DELETE FROM proposals WHERE id = ?", (proposal.row_id,))
    assert store.executions.for_proposal(proposal.row_id) == []


def test_decision_eval_round_trip(store: Store) -> None:
    league = seed_league(store)
    proposal = seed_proposal(store, league)
    pending = store.decision_evals.insert(
        DecisionEval(
            league_id=league.row_id,
            kind="lineup",
            season=2026,
            scoring_period_id=4,
            proposal_id=proposal.row_id,
            decided_at=AS_OF,
            inputs={"projections": {"source": "blend", "as_of": format_timestamp(AS_OF)}},
            decision={"starters": [1, 2, 3]},
        )
    )
    assert pending.id == 1 and pending.outcome is None
    assert store.decision_evals.unevaluated(league.row_id) == [pending]

    scored = store.decision_evals.update(
        pending.model_copy(
            update={
                "outcome": {"actual": 112.4, "optimal": 118.0},
                "metrics": {"efficiency": 0.953},
                "evaluated_at": LATER,
            }
        )
    )
    assert store.decision_evals.get(1) == scored
    assert store.decision_evals.unevaluated(league.row_id) == []

    waiver = store.decision_evals.insert(
        DecisionEval(
            league_id=league.row_id,
            kind="waiver",
            season=2026,
            scoring_period_id=5,
            decided_at=LATER,
            inputs={},
            decision={"add": 9, "drop": 8},
        )
    )
    assert store.decision_evals.find(league.row_id) == [scored, waiver]
    assert store.decision_evals.find(league.row_id, kind="waiver") == [waiver]
    assert store.decision_evals.find(league.row_id, season=2026, scoring_period_id=4) == [scored]
    assert store.decision_evals.find(league.row_id, season=2025) == []

    store.db.execute("DELETE FROM proposals WHERE id = ?", (proposal.row_id,))
    kept = store.decision_evals.get(1)
    assert kept is not None and kept.proposal_id is None


def test_llm_usage_round_trip_and_budget(store: Store) -> None:
    league = seed_league(store)
    calls = [
        LlmUsage(
            called_at=AS_OF - timedelta(days=1),
            worker="news_triage",
            model="claude-opus-5-5",
            input_tokens=1200,
            output_tokens=300,
            cache_read_input_tokens=1000,
            cost_usd=0.021,
            batch=True,
            stop_reason="end_turn",
            request_id="req_1",
        ),
        LlmUsage(
            called_at=AS_OF,
            worker="explain",
            model="claude-opus-5-5",
            input_tokens=800,
            output_tokens=120,
            cost_usd=0.012,
            league_id=league.row_id,
        ),
        LlmUsage(called_at=LATER, worker="close_call", model="claude-opus-5-5", cost_usd=0.5, stop_reason="refusal"),
    ]
    stored = [store.llm_usage.insert(call) for call in calls]
    assert stored == [call.model_copy(update={"id": i}) for i, call in enumerate(calls, start=1)]
    assert store.llm_usage.cost_since(AS_OF) == pytest.approx(0.512)
    assert store.llm_usage.cost_since(AS_OF - timedelta(days=2)) == pytest.approx(0.533)
    assert store.llm_usage.cost_since(LATER + timedelta(seconds=1)) == 0.0
    assert store.llm_usage.since(AS_OF) == stored[1:]


def test_raw_snapshot_round_trip(store: Store) -> None:
    league = seed_league(store)
    snapshot = RawSnapshot(
        source="espn",
        kind="mRoster",
        league_id=league.row_id,
        scoring_period_id=4,
        url="https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2026/segments/0/leagues/123456",
        params={"view": "mRoster", "scoringPeriodId": 4},
        path="espn/123456/2026/mRoster-4-a.json",
        sha256="ab" * 32,
        size_bytes=48213,
        status_code=200,
        fetched_at=AS_OF,
    )
    first = store.raw_snapshots.insert(snapshot)
    assert first == snapshot.model_copy(update={"id": 1})
    second = store.raw_snapshots.insert(
        snapshot.model_copy(update={"path": "espn/123456/2026/mRoster-4-b.json", "fetched_at": LATER})
    )
    sleeper = store.raw_snapshots.insert(
        RawSnapshot(source="sleeper", kind="players", path="sleeper/players.json", fetched_at=AS_OF)
    )

    assert store.raw_snapshots.get(1) == first
    assert store.raw_snapshots.get(9) is None
    assert store.raw_snapshots.latest("espn", "mRoster") == second
    assert store.raw_snapshots.latest("espn", "mRoster", league_id=league.row_id, scoring_period_id=4) == second
    assert store.raw_snapshots.latest("espn", "mRoster", scoring_period_id=5) is None
    assert store.raw_snapshots.latest("sleeper", "players") == sleeper
    assert store.raw_snapshots.find("espn") == [first, second]
    assert store.raw_snapshots.find("espn", "mSettings") == []

    with pytest.raises(sqlite3.IntegrityError):  # one index row per file
        store.raw_snapshots.insert(snapshot)
    store.db.execute("DELETE FROM leagues WHERE id = ?", (league.row_id,))
    kept = store.raw_snapshots.get(1)
    assert kept is not None and kept.league_id is None


def test_writes_inside_a_caller_transaction_are_atomic(store: Store) -> None:
    league = seed_league(store)
    with pytest.raises(RuntimeError, match="abort"):
        with store.db.transaction():
            store.teams.upsert(Team(league_id=league.row_id, team_id=1, name="A", as_of=AS_OF))
            store.players.upsert(Player(sport="nfl", espn_id=1, full_name="P", as_of=AS_OF))
            raise RuntimeError("abort")
    assert store.teams.for_league(league.row_id) == []
    assert store.players.get("nfl", 1) is None
    assert not store.db.in_transaction
