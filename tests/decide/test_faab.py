"""The FAAB bid model (ROADMAP #34): winning bids read from a league's transaction history, a bid monotone in the
claim's value by construction, held to the policy cap and the budget left, and a fallback to the heuristic when the
history is too thin to learn from.

The fit runs on synthetic histories built here (the recorded NFL league's claims all bid $0: it claims by priority)
and on the recorded files for the cases the real leagues give. The store and cache dir are the test's own.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fm import paths
from fm.decide.faab import (
    MIN_WINNING_BIDS,
    SHRINKAGE,
    TRANSACTIONS_KIND,
    BidModel,
    bid_strength,
    fit_bid_model,
    heuristic_bid,
    load_bid_history,
    modeled_bid,
    winning_bids,
)
from fm.espn.models import Transaction, TransactionsView
from fm.model.projections import ESPN
from fm.store import LeagueRow, RawSnapshotRow, Store

FIXTURES = Path(__file__).parent.parent / "fixtures" / "espn"
BUDGET = 100
NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def record(
    ident: str,
    bid: int | None,
    *,
    kind: str = "WAIVER",
    status: str = "EXECUTED",
    related: str | None = None,
    execution: str = "PROCESS",
) -> Transaction:
    raw: dict[str, Any] = {"id": ident, "type": kind, "status": status, "executionType": execution, "teamId": 1}
    if bid is not None:
        raw["bidAmount"] = bid
    if related is not None:
        raw["relatedTransactionId"] = related
    return Transaction.model_validate(raw)


def history(bids: Iterable[int]) -> list[Transaction]:
    return [record(f"w{index}", bid) for index, bid in enumerate(bids)]


def fitted(bids: Iterable[int], budget: int = BUDGET) -> BidModel:
    model = fit_bid_model(history(bids), budget=budget)
    assert model.fitted, model.reason
    return model


# --- the history ------------------------------------------------------------------------------------------------------


def test_only_executed_waiver_claims_with_a_bid_are_prices() -> None:
    mixed = [
        record("won", 17),
        record("outbid", 40, status="FAILED_INVALIDPLAYERSOURCE"),
        record("pending", 12, status="PENDING"),
        record("free agent", 0, kind="FREEAGENT", execution="EXECUTE"),
        record("trade", 30, kind="TRADE_ACCEPT", execution="EXECUTE"),
        record("cancelled", 22, status="EXECUTED", execution="CANCEL", related="gone"),
        record("no bid", None),
        record("zero", 0),
    ]
    assert winning_bids(mixed) == (17, 0)


def test_a_claim_and_its_processing_record_are_one_price() -> None:
    """ESPN's processing record names the claim behind it in ``relatedTransactionId``: one win, counted once."""
    claim = record("claim", 14, execution="EXECUTE")
    processed = record("processed", 14, related="claim")
    assert winning_bids([claim, processed]) == (14,)
    assert winning_bids([processed, claim, processed]) == (14,)  # the same record twice is one record


def test_the_recorded_leagues_history_has_no_prices() -> None:
    """Both real leagues claim by priority: the executed claims bid $0, so there is nothing to fit."""
    for league in ("ffl", "fba"):
        view = TransactionsView.model_validate_json((FIXTURES / "real" / league / "mTransactions2.json").read_bytes())
        assert set(winning_bids(view.transactions)) <= {0}
        assert not fit_bid_model(view.transactions, budget=BUDGET).fitted


def test_the_synthetic_fixture_has_one_winning_bid() -> None:
    view = TransactionsView.model_validate_json((FIXTURES / "ffl_transactions_week4.json").read_bytes())
    assert winning_bids(view.transactions) == (17,)  # the outbid $9 claim and the pending $12 one are not prices
    model = fit_bid_model(view.transactions, budget=BUDGET)
    assert not model.fitted and model.reason == f"only 1 winning bids in the history (need {MIN_WINNING_BIDS})"


# --- fitting and falling back -----------------------------------------------------------------------------------------


def test_the_fit_is_the_sorted_shares_of_the_season_budget() -> None:
    model = fitted([30, 5, 12, 1, 20, 8])
    assert model.n == 6 and model.fractions == (0.01, 0.05, 0.08, 0.12, 0.2, 0.3)
    assert model.weight == pytest.approx(6 / (6 + SHRINKAGE))
    assert model.reason is None and model.numbers()["winning_bids"] == 6
    assert fitted([500] * MIN_WINNING_BIDS).fractions == (1.0,) * MIN_WINNING_BIDS  # beyond the budget counts as it


@pytest.mark.parametrize(
    ("bids", "budget", "reason"),
    [
        ([5, 9, 12], BUDGET, f"only 3 winning bids in the history (need {MIN_WINNING_BIDS})"),
        ([], BUDGET, f"only 0 winning bids in the history (need {MIN_WINNING_BIDS})"),
        ([0] * 9, BUDGET, "all 9 winning bids were $0: the league does not appear to bid"),
        ([5] * 9, None, "the league does not use FAAB"),
        ([5] * 9, 0, "the league does not use FAAB"),
    ],
)
def test_too_little_history_or_no_faab_fits_no_model_and_says_why(
    bids: list[int], budget: int | None, reason: str
) -> None:
    model = fit_bid_model(history(bids), budget=budget)
    assert not model.fitted and model.reason == reason
    assert model.weight == 0.0 and model.quantile(0.5) == 0.0
    assert model.numbers() == {"fitted": False, "reason": reason, "winning_bids": len(bids) if budget else 0}
    with pytest.raises(ValueError, match="not fitted"):
        modeled_bid(model, 0.5, budget_left=80, cap=35)


def test_the_history_needed_is_adjustable() -> None:
    assert fit_bid_model(history([3, 4]), budget=BUDGET, min_bids=2).fitted
    assert not fit_bid_model(history([3, 4]), budget=BUDGET, min_bids=3).fitted


# --- the bid ----------------------------------------------------------------------------------------------------------


def test_the_quantile_runs_from_the_cheapest_to_the_dearest_winning_bid() -> None:
    model = fitted([10, 20, 30, 40, 50])
    assert [model.quantile(u) for u in (0.0, 0.25, 0.5, 0.75, 1.0)] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert model.quantile(0.125) == pytest.approx(0.15)  # linear between neighbours
    assert model.quantile(-1) == model.quantile(0.0) and model.quantile(7) == model.quantile(1.0)
    assert fitted([25] * 6).quantile(0.3) == pytest.approx(0.25)  # one price everywhere


def test_a_league_that_pays_more_bids_more_and_one_that_pays_less_bids_less() -> None:
    gain, roster = 40.0, 400.0  # strength 0.4
    strength = bid_strength(gain, roster)
    base = heuristic_bid(gain, roster_value=roster, budget_left=BUDGET, cap=BUDGET)
    dear = modeled_bid(fitted([60] * 20), strength, budget_left=BUDGET, cap=BUDGET)
    cheap = modeled_bid(fitted([2] * 20), strength, budget_left=BUDGET, cap=BUDGET)
    assert base == 40 and dear is not None and cheap is not None
    assert cheap < base < dear


def test_a_thin_history_moves_the_bid_less_than_a_long_one() -> None:
    strength = 0.4
    long_history = modeled_bid(fitted([60] * 80), strength, budget_left=BUDGET, cap=BUDGET)
    thin_history = modeled_bid(fitted([60] * MIN_WINNING_BIDS), strength, budget_left=BUDGET, cap=BUDGET)
    assert thin_history is not None and long_history is not None
    assert 40 < thin_history < long_history <= 60  # the heuristic's $40, then the model's $60 as the history grows


def test_no_strength_bids_the_minimum_and_an_unaffordable_claim_bids_nothing() -> None:
    model = fitted([10] * 9)
    assert modeled_bid(model, 0.0, budget_left=80, cap=35, minimum_bid=2) == 2
    assert modeled_bid(model, -3.0, budget_left=80, cap=35, minimum_bid=2) == 2
    assert modeled_bid(model, 0.5, budget_left=1, cap=35, minimum_bid=2) is None  # the budget left is below the minimum
    assert modeled_bid(model, 0.5, budget_left=80, cap=1, minimum_bid=2) is None  # and so is the cap
    assert modeled_bid(model, 9.0, budget_left=80, cap=35) == modeled_bid(model, 1.0, budget_left=80, cap=35)


def test_bad_inputs_are_refused() -> None:
    model = fitted([10] * 9)
    with pytest.raises(ValueError, match="finite"):
        modeled_bid(model, float("nan"), budget_left=80, cap=35)
    with pytest.raises(ValueError, match=">= 0"):
        modeled_bid(model, 0.5, budget_left=-1, cap=35)
    with pytest.raises(ValueError, match=">= 0"):
        modeled_bid(model, 0.5, budget_left=80, cap=-1)


prices = st.lists(st.integers(min_value=0, max_value=200), min_size=MIN_WINNING_BIDS, max_size=60).filter(
    lambda bids: max(bids) > 0
)
gains = st.lists(st.floats(min_value=-50, max_value=5_000, allow_nan=False), min_size=2, max_size=30)


@settings(max_examples=150, deadline=None)
@given(
    bids=prices,
    values=gains,
    roster_value=st.floats(min_value=0, max_value=20_000, allow_nan=False),
    budget=st.integers(min_value=1, max_value=300),
    spent=st.integers(min_value=0, max_value=300),
    cap_pct=st.floats(min_value=0.0, max_value=1.0),
    minimum_bid=st.integers(min_value=0, max_value=3),
)
def test_the_bid_is_monotonic_in_value_and_inside_the_cap_and_the_budget(
    bids: list[int], values: list[float], roster_value: float, budget: int, spent: int, cap_pct: float, minimum_bid: int
) -> None:
    model = fit_bid_model(history(bids), budget=budget)
    assert model.fitted
    budget_left = max(0, budget - spent)
    cap = int(budget * cap_pct)
    modeled = [
        modeled_bid(model, bid_strength(gain, roster_value), budget_left=budget_left, cap=cap, minimum_bid=minimum_bid)
        for gain in sorted(values)
    ]
    if min(cap, budget_left) < minimum_bid:
        assert modeled == [None] * len(modeled)  # nothing can be bid
        return
    placed = [bid for bid in modeled if bid is not None]
    assert len(placed) == len(modeled)
    assert placed == sorted(placed)  # non-decreasing in the claim's value
    assert all(minimum_bid <= bid <= min(cap, budget_left) for bid in placed)  # inside the cap and the budget left


@settings(max_examples=100, deadline=None)
@given(bids=prices, strengths=st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=2, max_size=20))
def test_the_quantile_is_non_decreasing_and_stays_inside_the_prices(bids: list[int], strengths: list[float]) -> None:
    model = fit_bid_model(history(bids), budget=200)
    assert model.fitted
    quantiles = [model.quantile(strength) for strength in sorted(strengths)]
    assert quantiles == sorted(quantiles)
    assert model.fractions[0] <= min(quantiles) and max(quantiles) <= model.fractions[-1]


def test_the_strength_is_the_heuristics_ratio_capped_at_one() -> None:
    assert bid_strength(50, 1000) == pytest.approx(0.2)  # 50 / (0.25 * 1000)
    assert bid_strength(500, 1000) == 1.0 and bid_strength(0, 1000) == 0.0 and bid_strength(-5, 1000) == 0.0
    assert bid_strength(10, 0) == 1.0  # a worthless roster bids it all, as the heuristic does
    assert heuristic_bid(50, roster_value=1000, budget_left=80, cap=35) == 16  # 80 * 0.2


# --- the captured history ---------------------------------------------------------------------------------------------


def league_row(store: Store) -> LeagueRow:
    return store.leagues.upsert(LeagueRow(key="nfl", sport="nfl", espn_league_id=7, season=2026, team_id=1, as_of=NOW))


def capture(store: Store, league: LeagueRow, name: str, transactions: Iterable[dict[str, Any]], when: datetime) -> str:
    """An ``mTransactions2`` page under the cache dir, indexed in ``raw_snapshots`` as a capture would be."""
    path = paths.cache_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"transactions": list(transactions)}), encoding="utf-8")
    store.raw_snapshots.insert(
        RawSnapshotRow(source=ESPN, kind=TRANSACTIONS_KIND, league_id=league.row_id, path=name, fetched_at=when)
    )
    return name


def claim_json(ident: str, bid: int, status: str = "EXECUTED") -> dict[str, Any]:
    return {"id": ident, "type": "WAIVER", "status": status, "bidAmount": bid, "teamId": 2, "executionType": "PROCESS"}


def test_the_history_is_read_back_from_the_captured_pages_the_later_page_winning(store: Store) -> None:
    league = league_row(store)
    early = datetime(2026, 9, 30, tzinfo=UTC)
    capture(store, league, "tx/a.json", [claim_json("one", 7, "PENDING"), claim_json("two", 9)], early)
    capture(store, league, "tx/b.json", [claim_json("one", 7), claim_json("three", 12)], NOW)
    transactions, warnings = load_bid_history(store, league)
    assert warnings == ()
    assert {t.id: t.status for t in transactions} == {"one": "EXECUTED", "two": "EXECUTED", "three": "EXECUTED"}
    assert sorted(winning_bids(transactions)) == [7, 9, 12]


def test_a_league_with_no_captured_history_has_an_empty_one(store: Store) -> None:
    assert load_bid_history(store, league_row(store)) == ((), ())


def test_a_page_gone_or_unreadable_is_reported_and_the_rest_is_still_read(store: Store) -> None:
    league = league_row(store)
    capture(store, league, "tx/good.json", [claim_json("one", 7)], NOW)
    store.raw_snapshots.insert(
        RawSnapshotRow(
            source=ESPN, kind=TRANSACTIONS_KIND, league_id=league.row_id, path="tx/gone.json", fetched_at=NOW
        )
    )
    (paths.cache_dir() / "tx" / "bad.json").write_text("[1, 2]", encoding="utf-8")
    store.raw_snapshots.insert(
        RawSnapshotRow(source=ESPN, kind=TRANSACTIONS_KIND, league_id=league.row_id, path="tx/bad.json", fetched_at=NOW)
    )
    transactions, warnings = load_bid_history(store, league)
    assert [t.id for t in transactions] == ["one"]
    assert warnings == (
        "nfl: transactions page tx/gone.json is gone from the cache",
        "nfl: transactions page tx/bad.json could not be read (ValidationError)",
    )
