"""``claim_waiver`` and ``cancel_waiver``: a waiver claim with its bid, and the withdrawal of our pending claim
(DESIGN 6.3, docs/espn-api.md section 4 "claim_waiver" and "cancel").

:class:`ClaimWaiver` serves the ``waiver`` proposals of both sports (a :class:`fm.proposals.WaiverPayload`: the player
to claim off waivers, the drop if any, the FAAB bid where the league bids). :class:`CancelWaiverClaim` serves
``waiver_cancel`` (a :class:`fm.proposals.TransactionCancelPayload`: the ESPN id of our open claim). The executor
(:func:`fm.executor.execute`) drives both and owns the write-safety rules.

A claim is processed at the league's waiver run, not when it is sent, so its preconditions are about the run:

- The proposal's scoring period is ESPN's current one or a later one (the period the run lands in); a past period
  means the run it was timed to has passed.
- The player is on waivers now (a free agent is added, not claimed; ``add_drop``) and his ``waiverProcessDate`` is
  still ahead.
- No open claim of ours already names him: ``EspnClient.pending_offers`` lists only claims and offers that are
  ``PENDING``, not expired and not named by a ``CANCEL`` record (docs/espn-api.md section 1 #2).
- The drop, if any, is on our roster, not on ESPN's undroppable list, and (when the claim executes in the current
  period) not past the league's roster lock, the same ``transaction_cutoff`` rule as ``add_drop``.
- Where the league bids FAAB (``acquisitionSettings.isUsingAcquisitionBudget``), the claim carries a bid of at least
  the league's minimum and no more than the budget left (``transactionCounter.acquisitionBudgetSpent``); where it
  does not, it carries none.
- Without a drop, the roster has room, or the claim would fail at the run (``FAILED_ROSTERLIMIT``, as the real
  NBA league's records show).

API mode sends the web client's ``WAIVER`` envelope: ``ADD`` (and ``DROP``) items and ``bidAmount``, ``null`` without
a bid as the captured claim sent it (``ffl/write_WAIVER_1.json``). UI mode is the captured path, the roster-fix page
(``type=claim``): a claim with a drop and no bid; a claim without a drop (what follows the list's ``Claim`` button
with roster room was not captured) or with a bid (no bid field was seen) is left to API mode. Verification re-reads
the open claims: ours for the player, with the drop and the bid, is pending.

A cancel needs the claim to be open and ours; it sends ``{type: WAIVER, executionType: CANCEL,
relatedTransactionId}`` (API only, as documented) and is verified when the claim is no longer open.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from fm.browser.flows import (
    Flow,
    FlowContext,
    Mode,
    ModeUnavailableError,
    Preconditions,
    UiDriver,
    Verification,
    WriteRequest,
    WriteResponse,
    register_flow,
)
from fm.browser.flows.add_drop import (
    LeagueRead,
    MoveFacts,
    PlayerFacts,
    RosterFixType,
    drop_facts,
    drop_failures,
    not_on_roster,
    player_facts,
    pool_entry,
    read_league,
    require_unchanged,
    roster_fix_walk,
    stamp,
)
from fm.browser.transactions import (
    Envelope,
    ExecutionType,
    TransactionType,
    add_item,
    drop_item,
    transaction_request,
    ui_may_follow,
)
from fm.espn.models import POOL_FREE_AGENT, POOL_WAIVERS, Transaction
from fm.proposals import ProposalKind, TransactionCancelPayload, WaiverPayload


def claim_summary(transaction: Transaction) -> dict[str, Any]:
    """A pending claim as the audit folder records it."""
    return {
        "id": transaction.id,
        "type": transaction.type,
        "status": transaction.status,
        "team_id": transaction.team_id,
        "scoring_period_id": transaction.scoring_period_id,
        "adds": list(transaction.adds),
        "drops": list(transaction.drops),
        "bid_amount": transaction.bid_amount,
    }


def our_open_claims[P: WaiverPayload | TransactionCancelPayload](ctx: FlowContext[P]) -> tuple[Transaction, ...]:
    """Our team's waiver claims still open at the run's clock, oldest first."""
    return tuple(
        transaction
        for transaction in ctx.reader.pending_offers(now=ctx.now)
        if transaction.is_waiver and transaction.team_id == ctx.team_id
    )


# --- preconditions ----------------------------------------------------------------------------------------------------


def claim_failures(add: PlayerFacts, team_id: int, now: datetime) -> list[str]:
    """Every reason the player cannot be claimed now."""
    failures: list[str] = []
    if add.status == POOL_FREE_AGENT:
        failures.append(f"{add.label} has cleared waivers: add him with an add_drop proposal instead of claiming him")
    elif add.on_team_id is not None:
        whose = "already on our team" if add.on_team_id == team_id else f"on team {add.on_team_id}"
        failures.append(f"{add.label} is {whose}, not on waivers")
    elif add.status != POOL_WAIVERS:
        failures.append(f"{add.label} is not on waivers (ESPN status {add.status!r})")
    if add.waiver_process_date is not None and add.waiver_process_date <= now:
        failures.append(
            f"{add.label}'s waiver run ({stamp(add.waiver_process_date)}) has passed; run fm sync and decide again"
        )
    return failures


def duplicate_failure(add_id: int, label: str, claims: Sequence[Transaction]) -> str | None:
    """A claim of ours for the same player is already open."""
    for claim in claims:
        if add_id in claim.adds:
            return f"a claim for {label} is already pending (transaction {claim.id}); cancel it first or leave it"
    return None


def bid_failures[P: WaiverPayload](
    ctx: FlowContext[P], read: LeagueRead, bid: int | None
) -> tuple[list[str], dict[str, Any]]:
    """The bid against the league's FAAB settings, and the facts read for the audit folder."""
    acquisition = read.settings.acquisition
    facts: dict[str, Any] = {"uses_faab": acquisition.uses_faab, "bid": bid}
    failures: list[str] = []
    if not acquisition.uses_faab:
        if bid:
            failures.append(
                f"this league does not bid FAAB (claims go by priority), and the claim carries a bid of ${bid}"
            )
        return failures, facts
    facts["minimum_bid"] = acquisition.minimum_bid
    facts["budget"] = acquisition.budget
    if bid is None:
        failures.append("this league bids FAAB for claims: the claim needs a bid")
        return failures, facts
    if bid < acquisition.minimum_bid:
        failures.append(f"the bid ${bid} is below the league's minimum bid ${acquisition.minimum_bid}")
    if acquisition.budget is not None:
        spent = ctx.reader.teams().data.team(ctx.team_id).transaction_counter.acquisition_budget_spent
        left = acquisition.budget - spent
        facts["spent"] = spent
        facts["left"] = left
        if bid > left:
            failures.append(
                f"the bid ${bid} is more than the ${left} FAAB left (${acquisition.budget} budget, ${spent} spent)"
            )
    return failures, facts


# --- the claim --------------------------------------------------------------------------------------------------------


class ClaimWaiver(Flow[WaiverPayload]):
    """A waiver claim with its bid (``waiver``) in both sports."""

    name = "claim_waiver"
    kinds = (ProposalKind.WAIVER,)
    payload_type = WaiverPayload
    modes = (Mode.API, Mode.UI)

    # --- preconditions ------------------------------------------------------------------------------------------------

    def check(self, ctx: FlowContext[WaiverPayload]) -> Preconditions:
        read = read_league(ctx)
        payload = ctx.payload
        failures: list[str] = []
        if read.target < read.latest:
            failures.append(
                f"scoring period {read.target} is over: ESPN is on scoring period {read.latest}, so the waiver run "
                "the claim was timed to has passed"
            )
        executes_now = read.target == read.latest
        roster = read.our_roster(ctx.team_id)
        if roster is None:
            failures.append(f"ESPN's rosters for scoring period {read.target} have no team {ctx.team_id} (our team_id)")
            return Preconditions(
                tuple(failures), {"scoring_period_id": read.target, "latest_scoring_period": read.latest}
            )
        claims = our_open_claims(ctx)
        add: PlayerFacts | None = None
        entry = pool_entry(ctx, payload.add_espn_id)
        if entry is None:
            failures.append(f"ESPN has no player {payload.add_espn_id}")
            label = f"player {payload.add_espn_id}"
        else:
            add = player_facts(entry, read, ctx.game)
            label = add.label
            failures.extend(claim_failures(add, ctx.team_id, ctx.now))
        duplicate = duplicate_failure(payload.add_espn_id, label, claims)
        if duplicate is not None:
            failures.append(duplicate)
        drop: PlayerFacts | None = None
        if payload.drop_espn_id is not None:
            drop = drop_facts(roster, read, ctx.game, payload.drop_espn_id)
            if drop is None:
                failures.append(not_on_roster(payload.drop_espn_id, ctx.team_id, read.target))
            else:
                if executes_now and read.unmapped_lock is not None:
                    failures.append(read.unmapped_lock)
                failures.extend(drop_failures(drop, read, ctx.now, locks_apply=executes_now))
        bid_problems, bidding = bid_failures(ctx, read, payload.bid_amount)
        failures.extend(bid_problems)
        facts = MoveFacts(
            scoring_period_id=read.target,
            latest_scoring_period=read.latest,
            roster_lock_type=read.settings.roster_lock_type.value,
            roster_count=read.roster_count(roster),
            roster_size=read.settings.roster_size,
            add=add,
            drop=drop,
        )
        if drop is None and facts.roster_count >= facts.roster_size:
            failures.append(
                f"the roster is full ({facts.roster_count} of {facts.roster_size}): the claim needs a drop, or it "
                "fails at the run (FAILED_ROSTERLIMIT)"
            )
        observed = facts.to_json()
        observed["bidding"] = bidding
        observed["open_claims"] = [claim_summary(claim) for claim in claims]
        return Preconditions(tuple(failures), observed)

    # --- API mode -----------------------------------------------------------------------------------------------------

    def build_request(self, ctx: FlowContext[WaiverPayload], pre: Preconditions) -> WriteRequest:
        facts = MoveFacts.from_observed(pre.observed)
        payload = ctx.payload
        items: list[dict[str, Any]] = [add_item(payload.add_espn_id, ctx.team_id)]
        if payload.drop_espn_id is not None:
            items.append(drop_item(payload.drop_espn_id, ctx.team_id))
        envelope = Envelope(
            team_id=ctx.team_id,
            type=TransactionType.WAIVER,
            member_id=ctx.member_id,
            scoring_period_id=facts.latest_scoring_period,
            items=tuple(items),
            bid_amount=payload.bid_amount,
        )
        return transaction_request(ctx.league, envelope)

    def ui_may_follow(self, response: WriteResponse) -> bool:
        return ui_may_follow(response)

    # --- UI mode ------------------------------------------------------------------------------------------------------

    def run_ui(self, ctx: FlowContext[WaiverPayload], ui: UiDriver, pre: Preconditions) -> None:
        facts = MoveFacts.from_observed(pre.observed)
        if facts.add is None:
            raise ValueError("the preconditions found no player to claim")
        if facts.drop is None:
            raise ModeUnavailableError(
                "a claim without a drop: what follows the player list's Claim button with roster room is not "
                "captured; API mode sends it"
            )
        if ctx.payload.bid_amount:
            raise ModeUnavailableError(
                "a claim with a FAAB bid: the roster-fix page the capture saw has no bid field (the real leagues "
                "claim by priority); API mode sends it"
            )
        require_unchanged(ctx, facts)
        roster_fix_walk(
            ctx,
            ui,
            RosterFixType.CLAIM,
            facts.add,
            facts.drop,
            what=f"claim {facts.add.name} and drop {facts.drop.name}",
        )

    # --- verification -------------------------------------------------------------------------------------------------

    def verify(self, ctx: FlowContext[WaiverPayload], pre: Preconditions) -> Verification:
        payload = ctx.payload
        claims = our_open_claims(ctx)
        expected = {"add": payload.add_espn_id, "drop": payload.drop_espn_id, "bid": payload.bid_amount}
        observed = {"open_claims": [claim_summary(claim) for claim in claims]}
        for claim in claims:
            if payload.add_espn_id not in claim.adds:
                continue
            if payload.drop_espn_id is not None and payload.drop_espn_id not in claim.drops:
                continue
            if payload.bid_amount is not None and claim.bid_amount != payload.bid_amount:
                continue
            return Verification(True, f"claim {claim.id} is pending", expected, observed)
        detail = f"no open claim of team {ctx.team_id} for player {payload.add_espn_id}"
        return Verification(False, detail, expected, observed)


# --- the cancel -------------------------------------------------------------------------------------------------------


class CancelWaiverClaim(Flow[TransactionCancelPayload]):
    """Withdraw our open waiver claim (``waiver_cancel``) in both sports; API mode only, as documented."""

    name = "cancel_waiver"
    kinds = (ProposalKind.WAIVER_CANCEL,)
    payload_type = TransactionCancelPayload
    modes = (Mode.API,)

    def check(self, ctx: FlowContext[TransactionCancelPayload]) -> Preconditions:
        settings = ctx.reader.settings().data
        wanted = ctx.payload.espn_transaction_id
        offers = ctx.reader.pending_offers(now=ctx.now)
        claim = next((transaction for transaction in offers if transaction.id == wanted), None)
        failures: list[str] = []
        if claim is None:
            failures.append(
                f"transaction {wanted} is not an open claim or offer (processed, cancelled or expired already); "
                "run fm sync"
            )
        else:
            if not claim.is_waiver:
                failures.append(f"transaction {wanted} is a {claim.type}, not a waiver claim")
            if claim.team_id != ctx.team_id:
                failures.append(f"transaction {wanted} is team {claim.team_id}'s, not team {ctx.team_id}'s (ours)")
        observed = {
            "transaction_id": wanted,
            "claim": None if claim is None else claim_summary(claim),
            "scoring_period_id": settings.current_scoring_period,
        }
        return Preconditions(tuple(failures), observed)

    def build_request(self, ctx: FlowContext[TransactionCancelPayload], pre: Preconditions) -> WriteRequest:
        period = pre.observed.get("scoring_period_id")
        envelope = Envelope(
            team_id=ctx.team_id,
            type=TransactionType.WAIVER,
            member_id=ctx.member_id,
            scoring_period_id=None if period is None else int(period),
            execution_type=ExecutionType.CANCEL,
            related_transaction_id=ctx.payload.espn_transaction_id,
        )
        return transaction_request(ctx.league, envelope)

    def verify(self, ctx: FlowContext[TransactionCancelPayload], pre: Preconditions) -> Verification:
        wanted = ctx.payload.espn_transaction_id
        offers = ctx.reader.pending_offers(now=ctx.now)
        still_open = next((transaction for transaction in offers if transaction.id == wanted), None)
        expected = {"open": False, "transaction_id": wanted}
        observed = {"open": still_open is not None, "open_ids": [transaction.id for transaction in offers]}
        if still_open is None:
            return Verification(True, f"claim {wanted} is no longer open", expected, observed)
        return Verification(False, f"claim {wanted} is still open", expected, observed)


CLAIM_WAIVER = register_flow(ClaimWaiver())
"""The process-wide ``claim_waiver`` flow."""
CANCEL_WAIVER = register_flow(CancelWaiverClaim())
"""The process-wide ``cancel_waiver`` flow."""
