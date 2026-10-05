"""Kinds and their settings (trade kinds are approve-only), payload models, and every guardrail in ``evaluate``.

Leagues come from ``tests/fixtures/config.sample.toml`` (nfl: 3 transactions/week, 35% FAAB cap, untouchables
``"Sample Player"`` and ``4242424``) and league settings from the hand-built ``mSettings`` fixtures (ffl: $100 FAAB,
one scoring period per matchup; fba: daily scoring periods grouped into weekly matchups).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from fm.config import Config, League, Policy, load_config
from fm.espn.settings import LeagueSettings, load_league_settings
from fm.proposals import (
    ACQUISITION_KINDS,
    KINDS,
    TRADE_KINDS,
    AddDropPayload,
    LineupMove,
    LineupPayload,
    PolicyError,
    ProposalKind,
    TradePayload,
    TradeResponsePayload,
    TransactionCancelPayload,
    WaiverPayload,
    default_setting,
    effective_setting,
    evaluate,
    faab_bid_cap,
    find_untouchables,
    kind_spec,
    parse_payload,
    pause,
    propose,
    reject,
    resume,
    validate_setting,
)
from fm.store import LeagueRow, LeagueSettingsRow, PlayerRow, ProposalRow, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
DEADLINE = NOW + timedelta(hours=2)
TRADE_DEADLINE = datetime(2026, 11, 25, 17, 0, tzinfo=UTC)  # tradeSettings.deadlineDate in ffl_settings_ppr.json
LINEUP = LineupPayload(moves=(LineupMove(espn_id=10, from_slot_id=20, to_slot_id=0),))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


@pytest.fixture
def config() -> Config:
    return load_config(FIXTURES / "config.sample.toml", environ={})


def seed_league(store: Store, league: League, **overrides: object) -> LeagueRow:
    """A store row matching a configured league (the allowlist), unless ``overrides`` make it disagree."""
    fields: dict[str, object] = {
        "key": league.key,
        "sport": league.sport,
        "espn_league_id": league.espn_league_id,
        "season": league.season,
        "team_id": league.team_id,
        "as_of": NOW,
    }
    fields.update(overrides)
    return store.leagues.upsert(LeagueRow.model_validate(fields))


def seed_settings(store: Store, league: LeagueRow, name: str, **updates: object) -> LeagueSettings:
    settings = load_league_settings(FIXTURES / "espn" / name)
    if updates:
        settings = settings.model_copy(update=updates)
    store.settings.upsert(
        LeagueSettingsRow(league_id=league.row_id, settings=settings.model_dump(mode="json"), as_of=NOW)
    )
    return settings


def seed_player(store: Store, espn_id: int, name: str, sport: str = "nfl") -> PlayerRow:
    return store.players.upsert(
        PlayerRow.model_validate({"sport": sport, "espn_id": espn_id, "full_name": name, "as_of": NOW})
    )


def add_drop(
    store: Store, config: Config, league: LeagueRow, add: int, *, period: int | None, now: datetime = NOW
) -> ProposalRow:
    return propose(
        store,
        config,
        league,
        ProposalKind.ADD_DROP,
        AddDropPayload(add_espn_id=add),
        created_by="test",
        scoring_period_id=period,
        now=now,
    )


class TestKinds:
    def test_every_kind_has_a_spec_and_round_trips_through_its_string(self) -> None:
        assert set(KINDS) == set(ProposalKind)
        for kind in ProposalKind:
            assert kind_spec(kind) is kind_spec(kind.value) is KINDS[kind]
            assert kind_spec(kind).kind is kind

    def test_trade_kinds_are_hard_coded_approve_only(self) -> None:
        assert TRADE_KINDS == (
            ProposalKind.TRADE_PROPOSE,
            ProposalKind.TRADE_ACCEPT,
            ProposalKind.TRADE_DECLINE,
            ProposalKind.TRADE_CANCEL,
        )
        for kind in TRADE_KINDS:
            spec = kind_spec(kind)
            assert spec.is_trade and spec.policy_field is None
            assert spec.allowed == ("approve",) and spec.default == "approve"
            assert validate_setting(kind, "approve") == "approve"
            for setting in ("auto", "off"):
                with pytest.raises(
                    PolicyError, match=rf"{kind.value}: policy '{setting}' is not allowed; trades are approval-only"
                ):
                    validate_setting(kind, setting)

    def test_trade_kinds_ignore_the_config_and_stay_approve(self) -> None:
        permissive = Policy(bench_inactive="auto", lineup="auto", add_drop="off", waiver="off")
        for kind in TRADE_KINDS:
            assert effective_setting(kind, permissive) == "approve"
            assert effective_setting(kind, Policy()) == "approve"

    def test_configurable_kinds_follow_their_policy_field(self) -> None:
        policy = Policy(bench_inactive="off", lineup="auto", add_drop="approve", waiver="off")
        assert effective_setting(ProposalKind.BENCH_INACTIVE, policy) == "off"
        assert effective_setting(ProposalKind.LINEUP, policy) == "auto"
        assert effective_setting(ProposalKind.ADD_DROP, policy) == "approve"
        assert effective_setting(ProposalKind.WAIVER, policy) == "off"
        assert effective_setting(ProposalKind.WAIVER_CANCEL, policy) == "off"  # follows the waiver field

    def test_auto_is_a_lineup_only_setting(self) -> None:
        for kind in (ProposalKind.BENCH_INACTIVE, ProposalKind.LINEUP):
            assert validate_setting(kind, "auto") == "auto"
            assert kind_spec(kind).allowed == ("off", "approve", "auto")
        for kind in (ProposalKind.ADD_DROP, ProposalKind.WAIVER, ProposalKind.WAIVER_CANCEL):
            assert kind_spec(kind).allowed == ("off", "approve")
            with pytest.raises(
                PolicyError, match=rf"{kind.value}: policy 'auto' is not allowed; allowed: off, approve"
            ):
                validate_setting(kind, "auto")

    def test_defaults_match_the_config_defaults(self) -> None:
        defaults = Policy()
        assert default_setting(ProposalKind.BENCH_INACTIVE) == defaults.bench_inactive == "auto"
        assert default_setting(ProposalKind.LINEUP) == defaults.lineup == "approve"
        assert default_setting(ProposalKind.ADD_DROP) == defaults.add_drop == "approve"
        assert default_setting(ProposalKind.WAIVER) == defaults.waiver == "approve"
        assert default_setting("trade_propose") == "approve"

    def test_only_adds_and_claims_count_as_transactions(self) -> None:
        assert ACQUISITION_KINDS == (ProposalKind.ADD_DROP, ProposalKind.WAIVER)

    def test_unknown_kind_names_the_known_ones(self) -> None:
        with pytest.raises(
            PolicyError, match="unknown proposal kind 'draft'; known kinds: bench_inactive, lineup, add_drop"
        ):
            kind_spec("draft")


class TestPayloads:
    def test_models_validate_their_invariants(self) -> None:
        with pytest.raises(ValidationError, match="TRAN_ROSTER_SAME_SLOT"):
            LineupMove(espn_id=1, from_slot_id=0, to_slot_id=0)
        with pytest.raises(ValidationError, match="at least 1"):
            LineupPayload(moves=())
        with pytest.raises(ValidationError, match="needs add_espn_id, drop_espn_id or both"):
            AddDropPayload()
        with pytest.raises(ValidationError, match="cannot be both added and dropped"):
            AddDropPayload(add_espn_id=1, drop_espn_id=1)
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            WaiverPayload(add_espn_id=1, bid_amount=-1)
        with pytest.raises(ValidationError, match="players on at least one side"):
            TradePayload(other_team_id=2)
        with pytest.raises(ValidationError, match=r"players \[7\] appear on both sides"):
            TradePayload(other_team_id=2, give_espn_ids=(7,), get_espn_ids=(7, 8))
        with pytest.raises(ValidationError, match="extra_forbidden"):
            AddDropPayload.model_validate({"add_espn_id": 1, "bid": 3})

    def test_guardrail_hooks(self) -> None:
        assert LINEUP.outgoing_players == () and LINEUP.bid is None
        assert AddDropPayload(add_espn_id=1, drop_espn_id=2).outgoing_players == (2,)
        assert AddDropPayload(add_espn_id=1).outgoing_players == ()
        claim = WaiverPayload(add_espn_id=1, drop_espn_id=2, bid_amount=12)
        assert claim.outgoing_players == (2,) and claim.bid == 12
        assert WaiverPayload(add_espn_id=1).bid is None
        offer = TradePayload(other_team_id=3, give_espn_ids=(1, 2), get_espn_ids=(9,))
        assert offer.outgoing_players == (1, 2)
        assert TransactionCancelPayload(espn_transaction_id="t1").outgoing_players == ()

    def test_summaries_are_one_line(self) -> None:
        assert LINEUP.summary() == "10: slot 20 -> 0"
        assert AddDropPayload(add_espn_id=1, drop_espn_id=2).summary() == "add 1, drop 2"
        assert WaiverPayload(add_espn_id=1, drop_espn_id=2, bid_amount=12).summary() == "claim 1, drop 2, bid $12"
        assert (
            TradePayload(other_team_id=3, give_espn_ids=(1,), get_espn_ids=(9,)).summary()
            == "with team 3: give 1; get 9"
        )
        response = TradeResponsePayload(other_team_id=3, get_espn_ids=(9,), espn_transaction_id="t1")
        assert response.summary() == "offer t1 with team 3: give nothing; get 9"
        assert TransactionCancelPayload(espn_transaction_id="t1").summary() == "cancel transaction t1"

    def test_stored_payload_parses_back_by_kind(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        claim = WaiverPayload(add_espn_id=1, drop_espn_id=2, bid_amount=12)
        seed_settings(store, league, "ffl_settings_ppr.json")
        row = propose(
            store, config, league, ProposalKind.WAIVER, claim, created_by="test", scoring_period_id=4, now=NOW
        )
        assert row.payload == {"add_espn_id": 1, "drop_espn_id": 2, "bid_amount": 12}
        fetched = store.proposals.get(row.row_id)
        assert fetched is not None and parse_payload(fetched) == claim

    def test_wrong_payload_type_is_a_programming_error(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        with pytest.raises(TypeError, match="waiver takes a WaiverPayload, not LineupPayload"):
            evaluate(store, config, league, ProposalKind.WAIVER, LINEUP, now=NOW)


class TestAllowlistAndSettings:
    def test_a_configured_league_with_an_allowed_kind_passes(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        verdict = evaluate(store, config, league, ProposalKind.LINEUP, LINEUP, deadline=DEADLINE, now=NOW)
        assert verdict.allowed and verdict.reasons == () and verdict.setting == "approve"
        verdict.raise_if_blocked()

    def test_a_league_missing_from_config_is_blocked(self, store: Store, config: Config) -> None:
        stray = seed_league(store, config.league("nfl"), key="dynasty", espn_league_id=999)
        verdict = evaluate(store, config, stray, ProposalKind.LINEUP, LINEUP, now=NOW)
        assert verdict.reasons == ("league 'dynasty' is not in config.toml (league allowlist)",)
        with pytest.raises(PolicyError, match="lineup blocked: league 'dynasty' is not in config.toml"):
            verdict.raise_if_blocked()

    def test_a_league_row_pointing_elsewhere_is_blocked(self, store: Store, config: Config) -> None:
        moved = seed_league(store, config.league("nfl"), espn_league_id=999)
        (reason,) = evaluate(store, config, moved, ProposalKind.LINEUP, LINEUP, now=NOW).reasons
        assert reason.startswith("league 'nfl' in the store is ESPN nfl league 999 season 2026, but config.toml says")

    def test_off_kinds_are_blocked_but_still_report_their_setting(self, store: Store) -> None:
        config = Config.model_validate(
            {
                "league": [
                    {
                        "key": "nfl",
                        "sport": "nfl",
                        "espn_league_id": 1,
                        "season": 2026,
                        "team_id": 1,
                        "policy": {"add_drop": "off"},
                    }
                ]
            }
        )
        league = seed_league(store, config.league("nfl"))
        verdict = evaluate(store, config, league, ProposalKind.ADD_DROP, AddDropPayload(add_espn_id=1), now=NOW)
        assert verdict.setting == "off"
        assert verdict.reasons == ("add_drop is off for league 'nfl' ([league.policy] in config.toml)",)

    def test_auto_needs_a_deadline(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))  # bench_inactive = auto in the sample
        verdict = evaluate(store, config, league, ProposalKind.BENCH_INACTIVE, LINEUP, now=NOW)
        assert verdict.setting == "auto"
        assert verdict.reasons == (
            "an auto proposal needs a deadline: auto fires at T-15 when the proposal is unanswered",
        )
        assert evaluate(store, config, league, ProposalKind.BENCH_INACTIVE, LINEUP, deadline=DEADLINE, now=NOW).allowed

    def test_a_deadline_already_passed_is_blocked(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        verdict = evaluate(store, config, league, ProposalKind.LINEUP, LINEUP, deadline=NOW, now=NOW)
        assert verdict.reasons == ("deadline 2026-10-04 12:00 UTC has passed",)
        assert evaluate(
            store, config, league, ProposalKind.LINEUP, LINEUP, deadline=NOW + timedelta(seconds=1), now=NOW
        ).allowed

    def test_naive_times_are_refused(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        with pytest.raises(ValueError, match="timezone-aware"):
            evaluate(store, config, league, ProposalKind.LINEUP, LINEUP, now=datetime(2026, 10, 4, 12, 0))
        with pytest.raises(ValueError, match="timezone-aware"):
            evaluate(store, config, league, ProposalKind.LINEUP, LINEUP, deadline=datetime(2026, 10, 4, 14, 0), now=NOW)

    def test_pause_blocks_every_kind(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        pause("travelling", now=NOW)
        try:
            verdict = evaluate(store, config, league, ProposalKind.LINEUP, LINEUP, deadline=DEADLINE, now=NOW)
            assert verdict.reasons == ("paused since 2026-10-04 12:00 UTC (travelling); run fm resume",)
        finally:
            resume()
        assert evaluate(store, config, league, ProposalKind.LINEUP, LINEUP, deadline=DEADLINE, now=NOW).allowed

    def test_every_reason_is_reported_at_once(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        seed_player(store, 4242424, "Someone Else")
        payload = WaiverPayload(add_espn_id=1, drop_espn_id=4242424, bid_amount=50)  # untouchable drop, no settings
        reasons = evaluate(
            store, config, league, ProposalKind.WAIVER, payload, deadline=NOW - timedelta(minutes=1), now=NOW
        ).reasons
        assert [reason.split(" ")[0] for reason in reasons] == ["deadline", "Someone", "FAAB"]


class TestUntouchables:
    def test_matched_by_id_or_by_name(self, store: Store, config: Config) -> None:
        seed_player(store, 4242424, "Someone Else")
        seed_player(store, 100, "Sample Player")
        seed_player(store, 101, "Free Agent")
        policy = config.league("nfl").policy
        assert find_untouchables(store, "nfl", policy, [4242424, 100, 101, 102]) == {
            4242424: "Someone Else",
            100: "Sample Player",
        }
        assert find_untouchables(store, "nfl", policy, []) == {}
        assert find_untouchables(store, "nfl", Policy(), [4242424]) == {}
        # An id on the list protects a player the store has not seen yet; a name cannot.
        assert find_untouchables(store, "nba", policy, [4242424, 100]) == {4242424: "4242424"}

    def test_names_match_loosely_and_digit_strings_are_ids(self, store: Store) -> None:
        seed_player(store, 1, "A.J. Brown")
        seed_player(store, 2, "Ja'Marr Chase")
        seed_player(store, 3, "Other Guy")
        policy = Policy(untouchables=("aj  brown", "JA'MARR CHASE", "3"))
        assert find_untouchables(store, "nfl", policy, [1, 2, 3, 4]) == {
            1: "A.J. Brown",
            2: "Ja'Marr Chase",
            3: "Other Guy",
        }

    def test_drops_and_outgoing_trade_players_are_protected(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        seed_settings(store, league, "ffl_settings_ppr.json")
        seed_player(store, 100, "Sample Player")
        blocked = "Sample Player (100) is untouchable"
        cases = (
            (ProposalKind.ADD_DROP, AddDropPayload(add_espn_id=1, drop_espn_id=100)),
            (ProposalKind.WAIVER, WaiverPayload(add_espn_id=1, drop_espn_id=100, bid_amount=5)),
            (ProposalKind.TRADE_PROPOSE, TradePayload(other_team_id=2, give_espn_ids=(100,), get_espn_ids=(9,))),
            (
                ProposalKind.TRADE_ACCEPT,
                TradeResponsePayload(other_team_id=2, give_espn_ids=(100,), espn_transaction_id="t1"),
            ),
        )
        for kind, payload in cases:
            assert evaluate(store, config, league, kind, payload, scoring_period_id=4, now=NOW).reasons == (blocked,), (
                kind
            )
        with pytest.raises(PolicyError, match="add_drop blocked: Sample Player \\(100\\) is untouchable"):
            propose(
                store,
                config,
                league,
                ProposalKind.ADD_DROP,
                AddDropPayload(drop_espn_id=100),
                created_by="test",
                now=NOW,
            )

    def test_declining_or_receiving_an_untouchable_is_fine(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        seed_player(store, 100, "Sample Player")
        decline = TradeResponsePayload(
            other_team_id=2, give_espn_ids=(100,), get_espn_ids=(9,), espn_transaction_id="t1"
        )
        assert evaluate(store, config, league, ProposalKind.TRADE_DECLINE, decline, now=NOW).allowed
        receive = TradePayload(other_team_id=2, give_espn_ids=(7,), get_espn_ids=(100,))
        assert evaluate(store, config, league, ProposalKind.TRADE_PROPOSE, receive, now=NOW).allowed
        assert evaluate(
            store, config, league, ProposalKind.ADD_DROP, AddDropPayload(add_espn_id=100), scoring_period_id=4, now=NOW
        ).allowed


class TestTradeDeadline:
    OFFER = TradePayload(other_team_id=2, give_espn_ids=(7,), get_espn_ids=(9,))
    ACCEPT = TradeResponsePayload(other_team_id=2, give_espn_ids=(7,), get_espn_ids=(9,), espn_transaction_id="t1")

    def test_proposing_or_accepting_after_the_deadline_is_blocked(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        assert seed_settings(store, league, "ffl_settings_ppr.json").trade.deadline == TRADE_DEADLINE
        blocked = "the league's trade deadline 2026-11-25 17:00 UTC has passed"
        for kind, payload in ((ProposalKind.TRADE_PROPOSE, self.OFFER), (ProposalKind.TRADE_ACCEPT, self.ACCEPT)):
            before = TRADE_DEADLINE - timedelta(seconds=1)
            assert evaluate(store, config, league, kind, payload, now=before).allowed, kind
            assert evaluate(store, config, league, kind, payload, now=TRADE_DEADLINE).reasons == (blocked,), kind
        with pytest.raises(PolicyError, match=f"trade_propose blocked: {blocked}"):
            propose(
                store, config, league, ProposalKind.TRADE_PROPOSE, self.OFFER, created_by="test", now=TRADE_DEADLINE
            )

    def test_declining_or_withdrawing_stays_allowed_and_an_unknown_or_absent_deadline_does_not_block(
        self, store: Store, config: Config
    ) -> None:
        league = seed_league(store, config.league("nfl"))
        settings = seed_settings(store, league, "ffl_settings_ppr.json")
        after = TRADE_DEADLINE + timedelta(days=1)
        decline = TradeResponsePayload(other_team_id=2, get_espn_ids=(9,), espn_transaction_id="t1")
        cancel = TransactionCancelPayload(espn_transaction_id="t1")
        assert evaluate(store, config, league, ProposalKind.TRADE_DECLINE, decline, now=after).allowed
        assert evaluate(store, config, league, ProposalKind.TRADE_CANCEL, cancel, now=after).allowed
        # A league without a deadline never closes; before the first sync the deadline is unknown and the executor's
        # precondition check against the live league is the gate.
        no_deadline = settings.trade.model_copy(update={"deadline": None})
        seed_settings(store, league, "ffl_settings_ppr.json", trade=no_deadline)
        assert evaluate(store, config, league, ProposalKind.TRADE_PROPOSE, self.OFFER, now=after).allowed
        unsynced = seed_league(store, config.league("nba"))
        assert evaluate(store, config, unsynced, ProposalKind.TRADE_PROPOSE, self.OFFER, now=after).allowed


class TestWeeklyCap:
    def test_counts_committed_and_pending_acquisitions_in_the_matchup_period(
        self, store: Store, config: Config
    ) -> None:
        league = seed_league(store, config.league("nfl"))  # max_transactions_per_week = 3
        seed_settings(store, league, "ffl_settings_ppr.json")  # matchup period == scoring period
        rows = [add_drop(store, config, league, add, period=4) for add in (1, 2, 3)]
        with pytest.raises(
            PolicyError,
            match=r"add_drop blocked: 3 of 3 transactions already used this week \(max_transactions_per_week\)",
        ):
            add_drop(store, config, league, 4, period=4)
        claim = WaiverPayload(add_espn_id=4, bid_amount=1)
        assert not evaluate(store, config, league, ProposalKind.WAIVER, claim, scoring_period_id=4, now=NOW).allowed
        # Another week, a lineup move, and a cancel do not touch the count.
        assert add_drop(store, config, league, 4, period=5).scoring_period_id == 5
        assert evaluate(store, config, league, ProposalKind.LINEUP, LINEUP, scoring_period_id=4, now=NOW).allowed
        cancel = TransactionCancelPayload(espn_transaction_id="t1")
        assert evaluate(store, config, league, ProposalKind.WAIVER_CANCEL, cancel, scoring_period_id=4, now=NOW).allowed
        # Rejecting one gives the slot back.
        reject(store, rows[0].row_id, decided_by="cli", now=NOW)
        assert add_drop(store, config, league, 5, period=4).status == "proposed"

    def test_nba_daily_periods_share_their_matchup_week(self, store: Store, config: Config) -> None:
        nba = config.league("nba").model_copy(update={"policy": Policy(max_transactions_per_week=1)})
        config = config.model_copy(update={"leagues": (config.league("nfl"), nba)})
        league = seed_league(store, nba)
        seed_settings(store, league, "fba_settings_points.json")  # matchup 2 = scoring periods 7..13
        add_drop(store, config, league, 1, period=7)
        with pytest.raises(PolicyError, match="1 of 1 transactions already used this week"):
            add_drop(store, config, league, 2, period=13)
        assert add_drop(store, config, league, 2, period=14).status == "proposed"

    def test_without_settings_or_period_the_window_is_the_trailing_week(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        add_drop(store, config, league, 1, period=None, now=NOW - timedelta(days=8))  # outside the window
        add_drop(store, config, league, 2, period=None, now=NOW - timedelta(days=6))
        add_drop(store, config, league, 3, period=None, now=NOW - timedelta(days=1))
        assert add_drop(store, config, league, 4, period=None, now=NOW).status == "proposed"
        with pytest.raises(PolicyError, match="3 of 3 transactions already used this week"):
            add_drop(store, config, league, 5, period=None, now=NOW)

    def test_rows_without_a_period_still_count_toward_the_matchup_week(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))  # max_transactions_per_week = 3
        seed_settings(store, league, "ffl_settings_ppr.json")
        add_drop(store, config, league, 1, period=None, now=NOW - timedelta(days=8))  # older than the trailing week
        add_drop(store, config, league, 2, period=None)  # a producer that does not know the period, e.g. the MCP
        add_drop(store, config, league, 3, period=None)
        assert add_drop(store, config, league, 4, period=4).status == "proposed"  # the week's third slot
        for period in (4, None):
            with pytest.raises(PolicyError, match="3 of 3 transactions already used this week"):
                add_drop(store, config, league, 5, period=period)

    def test_a_claim_past_its_deadline_gives_its_slot_back(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))  # max_transactions_per_week = 3
        seed_settings(store, league, "ffl_settings_ppr.json")
        for add in (1, 2, 3):
            claim = WaiverPayload(add_espn_id=add, bid_amount=1)
            propose(
                store,
                config,
                league,
                ProposalKind.WAIVER,
                claim,
                created_by="test",
                scoring_period_id=4,
                deadline=DEADLINE,
                now=NOW,
            )
        with pytest.raises(PolicyError, match="3 of 3 transactions already used this week"):
            add_drop(store, config, league, 4, period=4)
        # The waiver run passes with the claims unanswered: proposing expires them, and the week is free again.
        assert add_drop(store, config, league, 4, period=4, now=DEADLINE).status == "proposed"
        assert all(row.status == "expired" for row in store.proposals.find(kinds=["waiver"]))

    def test_a_cap_of_zero_blocks_every_transaction(self, store: Store, config: Config) -> None:
        nfl = config.league("nfl").model_copy(update={"policy": Policy(max_transactions_per_week=0)})
        config = config.model_copy(update={"leagues": (nfl, config.league("nba"))})
        league = seed_league(store, nfl)
        with pytest.raises(PolicyError, match="0 of 0 transactions already used this week"):
            add_drop(store, config, league, 1, period=4)
        assert evaluate(store, config, league, ProposalKind.LINEUP, LINEUP, now=NOW).allowed


class TestFaabCap:
    def test_cap_is_the_policy_share_of_the_season_budget_rounded_down(self, config: Config) -> None:
        ffl = load_league_settings(FIXTURES / "espn" / "ffl_settings_ppr.json")  # $100
        nine_cat = load_league_settings(FIXTURES / "espn" / "fba_settings_9cat.json")  # $200
        policy = config.league("nfl").policy  # 35%
        assert faab_bid_cap(policy, ffl) == 35
        assert faab_bid_cap(policy, nine_cat) == 70
        assert faab_bid_cap(Policy(max_faab_pct_per_bid=0.333), ffl) == 33
        assert faab_bid_cap(Policy(max_faab_pct_per_bid=1.0), ffl) == 100
        no_faab = ffl.acquisition.model_copy(update={"uses_faab": False, "budget": None})
        assert faab_bid_cap(policy, ffl.model_copy(update={"acquisition": no_faab})) is None

    def test_bids_above_the_cap_are_blocked(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        seed_settings(store, league, "ffl_settings_ppr.json")

        def verdict(bid: int | None):
            return evaluate(
                store,
                config,
                league,
                ProposalKind.WAIVER,
                WaiverPayload(add_espn_id=1, bid_amount=bid),
                scoring_period_id=4,
                now=NOW,
            )

        assert verdict(35).allowed and verdict(0).allowed and verdict(None).allowed
        assert verdict(36).reasons == ("bid $36 exceeds 35% of the $100 FAAB budget ($35)",)
        with pytest.raises(
            PolicyError, match=r"waiver blocked: bid \$99 exceeds 35% of the \$100 FAAB budget \(\$35\)"
        ):
            propose(
                store,
                config,
                league,
                ProposalKind.WAIVER,
                WaiverPayload(add_espn_id=1, bid_amount=99),
                created_by="test",
                now=NOW,
            )

    def test_a_bid_needs_synced_settings_and_a_faab_league(self, store: Store, config: Config) -> None:
        league = seed_league(store, config.league("nfl"))
        claim = WaiverPayload(add_espn_id=1, bid_amount=5)
        assert evaluate(store, config, league, ProposalKind.WAIVER, claim, now=NOW).reasons == (
            "FAAB budget unknown: league settings are not synced (run fm sync)",
        )
        assert evaluate(store, config, league, ProposalKind.WAIVER, WaiverPayload(add_espn_id=1), now=NOW).allowed
        ffl = load_league_settings(FIXTURES / "espn" / "ffl_settings_ppr.json")
        no_faab = ffl.acquisition.model_copy(update={"uses_faab": False, "budget": None})
        seed_settings(store, league, "ffl_settings_ppr.json", acquisition=no_faab)
        assert evaluate(store, config, league, ProposalKind.WAIVER, claim, now=NOW).reasons == (
            "a bid of $5 was given but the league does not use FAAB",
        )
