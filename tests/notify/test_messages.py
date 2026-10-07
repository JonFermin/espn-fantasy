"""What the phone shows (``fm.notify.messages``): proposals as one line per action in player names and slot labels, the
rationale and the deadline (never the raw engine numbers); confirmations, alerts and reports.

Leagues come from ``fixtures/config.sample.toml``; players and teams are seeded with placeholder names.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from fm.config import Config, load_config
from fm.notify.base import DecisionResult
from fm.notify.messages import (
    alert,
    decision_message,
    describe_payload,
    local_time,
    moves_text,
    player_names,
    proposal_message,
    proposal_name,
    relative,
    report,
)
from fm.proposals import (
    AddDropPayload,
    LineupMove,
    LineupPayload,
    Payload,
    ProposalKind,
    TradePayload,
    TradeResponsePayload,
    TransactionCancelPayload,
    WaiverPayload,
    propose,
)
from fm.store import LeagueRow, PlayerRow, Store, TeamRow

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
DEADLINE = NOW + timedelta(hours=2)
SWAP = LineupPayload(
    moves=(LineupMove(espn_id=10, from_slot_id=20, to_slot_id=0), LineupMove(espn_id=11, from_slot_id=0, to_slot_id=20))
)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


@pytest.fixture
def config() -> Config:
    return load_config(FIXTURES / "config.sample.toml", environ={})


def league_row(store: Store, config: Config, key: str) -> LeagueRow:
    league = config.league(key)
    return store.leagues.upsert(
        LeagueRow(
            key=league.key,
            sport=league.sport,
            espn_league_id=league.espn_league_id,
            season=league.season,
            team_id=league.team_id,
            as_of=NOW,
        )
    )


@pytest.fixture
def nfl(store: Store, config: Config) -> LeagueRow:
    row = league_row(store, config, "nfl")
    for espn_id, name in (
        (10, "Sample Quarterback"),
        (11, "Backup Passer"),
        (12, "Waiver Runner"),
        (13, "Bench Tight End"),
    ):
        store.players.upsert(PlayerRow(sport="nfl", espn_id=espn_id, full_name=name, as_of=NOW))
    store.teams.upsert(TeamRow(league_id=row.row_id, team_id=3, name="Sample Rivals", as_of=NOW))
    return row


class TestProposalMessage:
    def test_a_lineup_proposal_reads_as_moves_rationale_and_deadline_without_engine_numbers(
        self, store: Store, config: Config, nfl: LeagueRow
    ) -> None:
        row = propose(
            store,
            config,
            nfl,
            ProposalKind.LINEUP,
            SWAP,
            created_by="test",
            engine_numbers={"delta_points": 3.24, "p_win": 0.613, "flagged": True, "games": 2},
            rationale="  Backup Passer has the better matchup.  ",
            deadline=DEADLINE,
            now=NOW,
        )
        message = proposal_message(store, nfl, row, now=NOW, tz=UTC)
        assert message.title == f"🏈 NFL: Lineup change (#{row.row_id})"
        assert message.priority == "high"
        assert message.body.splitlines() == [
            "Bench: Backup Passer",
            "Start: Sample Quarterback (QB)",
            "Backup Passer has the better matchup.",
            "⏰ Decide by Sun 04 Oct 14:00 UTC (in 2h 00m)",
            "No answer: nothing happens",
        ]
        assert "delta_points" not in message.body

    def test_an_auto_proposal_says_it_goes_ahead_and_the_deadline_is_local(
        self, store: Store, config: Config, nfl: LeagueRow
    ) -> None:
        bench = LineupPayload(moves=(LineupMove(espn_id=10, from_slot_id=0, to_slot_id=20),))
        row = propose(
            store, config, nfl, ProposalKind.BENCH_INACTIVE, bench, created_by="test", deadline=DEADLINE, now=NOW
        )
        mountain = timezone(timedelta(hours=-6))
        message = proposal_message(store, nfl, row, now=NOW, tz=mountain)
        assert message.title == f"🏈 NFL: Bench inactive starters (#{row.row_id})"
        assert message.body.splitlines() == [
            "Bench: Sample Quarterback",
            "⏰ Decide by Sun 04 Oct 08:00 UTC-06:00 (in 2h 00m)",
            "No answer: it goes ahead automatically 15 min before",
        ]

    def test_a_long_windows_zone_name_shows_as_its_initials(self, store: Store, config: Config, nfl: LeagueRow) -> None:
        row = propose(store, config, nfl, ProposalKind.LINEUP, SWAP, created_by="t", deadline=DEADLINE, now=NOW)
        mountain = timezone(timedelta(hours=-6), "Mountain Daylight Time")
        lines = proposal_message(store, nfl, row, now=NOW, tz=mountain).body.splitlines()
        assert "Sun 04 Oct 08:00 MDT" in lines[-2]

    def test_a_slot_to_slot_move_shows_both_slots(self, store: Store, config: Config, nfl: LeagueRow) -> None:
        swap = LineupPayload(moves=(LineupMove(espn_id=10, from_slot_id=0, to_slot_id=23),))
        row = propose(store, config, nfl, ProposalKind.LINEUP, swap, created_by="t", now=NOW)
        assert proposal_message(store, nfl, row, now=NOW).body.splitlines()[0] == (
            "Move: Sample Quarterback (QB → RB/WR/TE)"
        )

    def test_an_add_drop_without_a_deadline(self, store: Store, config: Config, nfl: LeagueRow) -> None:
        payload = AddDropPayload(add_espn_id=12, drop_espn_id=13)
        row = propose(store, config, nfl, ProposalKind.ADD_DROP, payload, created_by="t", now=NOW)
        assert proposal_message(store, nfl, row, now=NOW).body.splitlines() == [
            "➕ Add Waiver Runner",
            "➖ Drop Bench Tight End",
            "⏰ No deadline",
        ]

    def test_a_trade_reads_as_give_and_get(self, store: Store, config: Config, nfl: LeagueRow) -> None:
        payload = TradePayload(other_team_id=3, give_espn_ids=(13,), get_espn_ids=(12,))
        row = propose(store, config, nfl, ProposalKind.TRADE_PROPOSE, payload, created_by="t", now=NOW)
        assert proposal_message(store, nfl, row, now=NOW).body.splitlines()[:3] == [
            "With Sample Rivals",
            "Give: Bench Tight End",
            "Get: Waiver Runner",
        ]


class TestAlertHelpers:
    def test_moves_text_is_the_push_lines_on_one_line(self, store: Store, config: Config, nfl: LeagueRow) -> None:
        payload = AddDropPayload(add_espn_id=12, drop_espn_id=13)
        row = propose(store, config, nfl, ProposalKind.ADD_DROP, payload, created_by="t", now=NOW)
        assert moves_text(store, nfl, row) == "➕ Add Waiver Runner; ➖ Drop Bench Tight End"
        assert moves_text(store, None, row) == payload.summary()  # no league: the ids

    def test_proposal_name_and_local_time(self) -> None:
        assert proposal_name("nfl", "add_drop", 3) == "NFL free-agent add/drop #3"
        assert proposal_name("nba", "something_new", 4) == "NBA something_new #4"
        mountain = timezone(timedelta(hours=-6), "Mountain Daylight Time")
        assert local_time(DEADLINE, mountain) == "Sun 04 Oct 08:00 MDT"


class TestDescribePayload:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (AddDropPayload(add_espn_id=12, drop_espn_id=13), "add Waiver Runner, drop Bench Tight End"),
            (AddDropPayload(drop_espn_id=13), "drop Bench Tight End"),
            (
                WaiverPayload(add_espn_id=12, drop_espn_id=13, bid_amount=12),
                "claim Waiver Runner, drop Bench Tight End, bid $12",
            ),
            (WaiverPayload(add_espn_id=12), "claim Waiver Runner"),
            (
                TradePayload(other_team_id=3, give_espn_ids=(10, 11), get_espn_ids=(12,)),
                "with Sample Rivals: give Sample Quarterback, Backup Passer; get Waiver Runner",
            ),
            (TradePayload(other_team_id=8, get_espn_ids=(12,)), "with team 8: give nothing; get Waiver Runner"),
            (
                TradeResponsePayload(
                    espn_transaction_id="tx-7", other_team_id=3, give_espn_ids=(13,), get_espn_ids=(12,)
                ),
                "offer tx-7 with Sample Rivals: give Bench Tight End; get Waiver Runner",
            ),
            (TransactionCancelPayload(espn_transaction_id="tx-9"), "cancel transaction tx-9"),
        ],
    )
    def test_moves_read_in_names(self, store: Store, nfl: LeagueRow, payload: Payload, expected: str) -> None:
        assert describe_payload(store, nfl, payload) == expected

    def test_an_unseen_player_shows_as_its_espn_id(self, store: Store, nfl: LeagueRow) -> None:
        assert describe_payload(store, nfl, AddDropPayload(add_espn_id=4046, drop_espn_id=13)) == (
            "add player 4046, drop Bench Tight End"
        )
        assert player_names(store, "nfl", AddDropPayload(add_espn_id=4046)) == {}

    def test_nba_slots_use_the_nba_labels(self, store: Store, config: Config) -> None:
        nba = league_row(store, config, "nba")
        store.players.upsert(PlayerRow(sport="nba", espn_id=10, full_name="Sample Center", as_of=NOW))
        payload = LineupPayload(moves=(LineupMove(espn_id=10, from_slot_id=12, to_slot_id=11),))
        assert describe_payload(store, nba, payload) == "Sample Center: BE -> UTIL"

    def test_names_come_from_the_league_sport(self, store: Store, config: Config, nfl: LeagueRow) -> None:
        nba = league_row(store, config, "nba")  # the NFL player 10 is not an NBA player
        assert describe_payload(store, nba, AddDropPayload(add_espn_id=10)) == "add player 10"


class TestPlainPushes:
    def test_an_alert_is_urgent_and_can_link_to_the_fix(self) -> None:
        message = alert("Lineup not set", "start A over B", link="https://fantasy.espn.com/football/team")
        assert (message.priority, message.tags, message.link) == (
            "high",
            ("warning",),
            "https://fantasy.espn.com/football/team",
        )

    def test_a_report_is_quiet(self) -> None:
        message = report("Weekly report", "body")
        assert (message.title, message.body, message.priority, message.link) == ("Weekly report", "body", "low", None)

    def test_a_confirmation_says_what_was_recorded_or_refused(self) -> None:
        ok = decision_message(DecisionResult(4, "approve", ok=True, detail="nfl lineup change: A: BE -> QB"))
        refused = decision_message(DecisionResult(4, "approve", ok=False, detail="it expired"))
        assert (ok.title, ok.body, ok.tags) == ("#4 approved", "nfl lineup change: A: BE -> QB", ("white_check_mark",))
        assert (refused.title, refused.tags) == ("#4 not approved", ("x",))


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (timedelta(days=3, hours=4, minutes=30), "in 3d 4h"),
        (timedelta(hours=2, minutes=5), "in 2h 05m"),
        (timedelta(minutes=12, seconds=59), "in 12m"),
        (timedelta(seconds=30), "in 0m"),
        (timedelta(0), "passed"),
        (timedelta(minutes=-5), "passed"),
    ],
)
def test_relative(delta: timedelta, expected: str) -> None:
    assert relative(delta) == expected
