"""FAAB bids (DESIGN section 9.2, ROADMAP #34): what a waiver claim should bid, from what the league's managers have
paid and, with too little of that, from a heuristic.

**The heuristic** (:func:`heuristic_bid`) bids the share of the budget left that the claim's gain is of a quarter
(:data:`DEFAULT_BID_SHARE`) of the roster's remaining value. That ratio, capped at one, is the claim's *strength*
(:func:`bid_strength`): how much of the most a claim could be worth this one is.

**The model** (:func:`fit_bid_model`) learns what winning costs in this league: every executed ``WAIVER`` record in
the league's ``mTransactions2`` history (:func:`load_bid_history`) is a winning bid, taken as a fraction of the
season's FAAB budget. A claim's strength is read as a percentile of those prices, so a claim that is worth most of what
a claim can be worth bids what the league's dearest winners paid and a modest one bids what its cheap winners paid
(:meth:`BidModel.quantile`: the empirical quantile function, linearly interpolated). Failed claims are not prices (an
outbid claim only says the price was higher) and are left out. The model's bid is shrunk toward the heuristic's with
the weight ``n / (n + SHRINKAGE)`` for ``n`` winning bids, so a thin history moves the bid a little and a long one
leads it (:func:`modeled_bid`).

**Monotone by construction.** The strength is non-decreasing in the claim's gain, a quantile function built from the
sorted prices is non-decreasing in the strength, the heuristic's amount is non-decreasing in the strength, a convex
combination of two non-decreasing functions is non-decreasing, and the rounding and the clamp to ``[minimum_bid,
min(cap, budget_left)]`` preserve the order. Nothing here relies on the data being well behaved.

**It degrades.** With fewer than :data:`MIN_WINNING_BIDS` winning bids, or none above $0 (a league that does not
bid reports ``bidAmount: 0``, and both real leagues' claims go by waiver priority), no model is fitted: the
:class:`BidModel` says why (``reason``), :func:`modeled_bid` is never asked for, and the claim bids the heuristic's
amount, which :mod:`fm.decide.waivers` reports in the decision's warnings and the proposal's engine numbers.

Nothing here writes to ESPN (CLAUDE.md: workers propose, the executor acts).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final

from pydantic import ValidationError

from fm import paths
from fm.espn.client import View
from fm.espn.models import Transaction, TransactionsView
from fm.model.projections import ESPN
from fm.store import LeagueRow, Store

DEFAULT_BID_SHARE: Final = 0.25
"""The share of the roster's remaining value the FAAB budget left is taken to buy: a claim worth that much bids all
of it."""
MIN_WINNING_BIDS: Final = 5
"""Winning bids below which no model is fitted: a handful of prices say little about how this league bids."""
SHRINKAGE: Final = 8
"""Winning bids that weigh the model and the heuristic equally: the model's weight is ``n / (n + SHRINKAGE)``."""
TRANSACTIONS_KIND: Final = View.TRANSACTIONS.value
"""``raw_snapshots.kind`` of the ``mTransactions2`` pages a capture indexes."""
WAIVER_TYPE: Final = "WAIVER"
"""``transactions[].type`` of a waiver claim, and of ESPN's record of its processing."""


# --- the heuristic ----------------------------------------------------------------------------------------------------


def _check(gain: float, roster_value: float, budget_left: int, cap: int, minimum_bid: int, share: float) -> None:
    if not (math.isfinite(gain) and math.isfinite(roster_value)):
        raise ValueError(f"gain and roster_value must be finite, got {gain!r} and {roster_value!r}")
    if budget_left < 0 or cap < 0 or minimum_bid < 0:
        raise ValueError(f"budget_left, cap and minimum_bid must be >= 0, got {budget_left}, {cap}, {minimum_bid}")
    if not (math.isfinite(share) and share > 0):
        raise ValueError(f"share must be a positive number, got {share!r}")


def bid_strength(gain: float, roster_value: float, share: float = DEFAULT_BID_SHARE) -> float:
    """How much of the most a claim can be worth a claim of ``gain`` points to a roster worth ``roster_value`` is, in
    ``[0, 1]``: ``gain / (share * roster_value)`` capped at one, ``0`` for a gain that is not positive, and ``1`` for a
    worthless roster (or one so small the product underflows). Non-decreasing in ``gain``."""
    if gain <= 0:
        return 0.0
    scale = share * roster_value
    return 1.0 if scale <= 0 else min(1.0, gain / scale)


def heuristic_bid(
    gain: float,
    *,
    roster_value: float,
    budget_left: int,
    cap: int,
    minimum_bid: int = 0,
    share: float = DEFAULT_BID_SHARE,
) -> int | None:
    """A FAAB bid for a claim worth ``gain`` rest-of-season points to a roster worth ``roster_value``.

    The bid is ``budget_left`` times the claim's :func:`bid_strength` (all of it at most), rounded down, then raised to
    ``minimum_bid`` and held to ``cap`` and ``budget_left``. It is monotonic in the gain, and as the season runs down
    the same share of the remaining value buys more of what is left: the weeks left enter through both values, which
    shrink together. ``None`` when no bid is possible because the cap or the budget left is below the minimum. Raises
    ``ValueError`` for a non-finite value, a negative budget, cap or minimum, or a ``share`` that is not positive.
    """
    _check(gain, roster_value, budget_left, cap, minimum_bid, share)
    ceiling = min(cap, budget_left)
    if ceiling < minimum_bid:
        return None
    if gain <= 0:
        return minimum_bid
    return max(minimum_bid, min(math.floor(budget_left * bid_strength(gain, roster_value, share)), ceiling))


# --- the model --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BidModel:
    """What the league's winning bids look like: ``fractions`` are the winning bids as shares of the season's FAAB
    ``budget``, sorted. ``reason`` is why no model was fitted (``None`` when :attr:`fitted`)."""

    budget: int
    fractions: tuple[float, ...] = ()
    reason: str | None = None
    seen: int = 0
    """The winning bids the history had, whether or not enough to fit."""

    @property
    def n(self) -> int:
        """The winning bids the model is drawn from (none when it is not fitted)."""
        return len(self.fractions)

    @property
    def fitted(self) -> bool:
        return self.reason is None and self.n > 0

    @property
    def weight(self) -> float:
        """How far the model leads the heuristic: ``n / (n + SHRINKAGE)``, ``0`` when nothing is fitted."""
        return self.n / (self.n + SHRINKAGE) if self.fitted else 0.0

    def quantile(self, strength: float) -> float:
        """The winning bid, as a share of the season budget, at percentile ``strength`` (clamped to ``[0, 1]``) of the
        league's winning bids: the sorted prices, linearly interpolated. Non-decreasing in ``strength``. ``0`` when
        nothing is fitted."""
        if not self.fitted:
            return 0.0
        position = min(1.0, max(0.0, strength)) * (self.n - 1)
        low = math.floor(position)
        high = min(low + 1, self.n - 1)
        return self.fractions[low] + (self.fractions[high] - self.fractions[low]) * (position - low)

    def numbers(self) -> dict[str, Any]:
        """What the model is drawn from, for a proposal's engine numbers."""
        if not self.fitted:
            return {"fitted": False, "reason": self.reason, "winning_bids": self.seen}
        return {
            "fitted": True,
            "winning_bids": self.n,
            "weight": round(self.weight, 4),
            "median_fraction": round(self.quantile(0.5), 4),
            "max_fraction": round(self.fractions[-1], 4),
        }


def winning_bids(transactions: Iterable[Transaction]) -> tuple[int, ...]:
    """The bids that won in ``transactions``: executed ``WAIVER`` records with a bid, once each. A record another
    names in ``relatedTransactionId`` is the claim behind ESPN's record of its processing, which is counted instead.
    Failed claims, free-agent adds, trades and lineup records are not prices."""
    records = {transaction.id: transaction for transaction in transactions}
    behind = {t.related_transaction_id for t in records.values() if t.related_transaction_id is not None}
    return tuple(
        t.bid_amount
        for t in records.values()
        if t.type == WAIVER_TYPE
        and t.executed
        and not t.is_cancellation
        and t.id not in behind
        and t.bid_amount is not None
        and t.bid_amount >= 0
    )


def fit_bid_model(
    transactions: Iterable[Transaction], *, budget: int | None, min_bids: int = MIN_WINNING_BIDS
) -> BidModel:
    """The league's :class:`BidModel` from its transaction history, or the reason none could be fitted: the league has
    no FAAB ``budget``, fewer than ``min_bids`` winning bids, or none above $0 (it does not bid; ESPN reports
    ``bidAmount: 0`` without FAAB). A bid above the budget counts as the whole budget."""
    if budget is None or budget <= 0:
        return BidModel(budget=max(budget or 0, 0), reason="the league does not use FAAB")
    bids = winning_bids(transactions)
    if len(bids) < max(1, min_bids):
        return BidModel(
            budget, reason=f"only {len(bids)} winning bids in the history (need {min_bids})", seen=len(bids)
        )
    if max(bids) <= 0:
        return BidModel(
            budget, reason=f"all {len(bids)} winning bids were $0: the league does not appear to bid", seen=len(bids)
        )
    return BidModel(budget, tuple(sorted(min(1.0, bid / budget) for bid in bids)), seen=len(bids))


def modeled_bid(
    model: BidModel,
    strength: float,
    *,
    budget_left: int,
    cap: int,
    minimum_bid: int = 0,
) -> int | None:
    """A FAAB bid for a claim of ``strength`` (:func:`bid_strength`): the model's price at that percentile of the
    league's winning bids, shrunk toward the heuristic's ``budget_left * strength`` with the model's
    :attr:`~BidModel.weight`, rounded down and held to ``[minimum_bid, min(cap, budget_left)]``. Non-decreasing in
    ``strength`` (the module docs). A claim of no strength bids the minimum, as the heuristic does; ``None`` when the
    cap or the budget left is below the minimum. Raises ``ValueError`` for a negative budget, cap or minimum, a
    ``strength`` that is not a finite number, or a model that is not fitted."""
    if not model.fitted:
        raise ValueError(f"the bid model is not fitted: {model.reason}")
    if not math.isfinite(strength):
        raise ValueError(f"strength must be finite, got {strength!r}")
    if budget_left < 0 or cap < 0 or minimum_bid < 0:
        raise ValueError(f"budget_left, cap and minimum_bid must be >= 0, got {budget_left}, {cap}, {minimum_bid}")
    ceiling = min(cap, budget_left)
    if ceiling < minimum_bid:
        return None
    if strength <= 0:
        return minimum_bid
    strength = min(1.0, strength)
    weight = model.weight
    amount = weight * model.budget * model.quantile(strength) + (1.0 - weight) * budget_left * strength
    return max(minimum_bid, min(math.floor(amount), ceiling))


# --- the history ------------------------------------------------------------------------------------------------------


def load_bid_history(store: Store, league: LeagueRow) -> tuple[tuple[Transaction, ...], tuple[str, ...]]:
    """(the league's transactions, warnings) from the ``mTransactions2`` pages captured for it and indexed in
    ``raw_snapshots``, read back from under ``fm.paths.cache_dir()``. A transaction seen on two pages takes the later
    one (its status may have moved on). With no page, or a page that is gone or unreadable, the history is empty or
    partial and a warning says so; the cache is deletable and a league need not have captured any."""
    pages = store.raw_snapshots.find(ESPN, TRANSACTIONS_KIND, league_id=league.row_id)
    root = paths.cache_dir()
    merged: dict[str, Transaction] = {}
    warnings: list[str] = []
    for page in pages:
        try:
            view = TransactionsView.model_validate_json((root / page.path).read_bytes())
        except FileNotFoundError:
            warnings.append(f"{league.key}: transactions page {page.path} is gone from the cache")
            continue
        except (OSError, ValidationError) as exc:
            warnings.append(f"{league.key}: transactions page {page.path} could not be read ({type(exc).__name__})")
            continue
        merged.update((transaction.id, transaction) for transaction in view.transactions)
    return tuple(merged.values()), tuple(warnings)
