"""Proposal queue, policy evaluation, approvals, and expiry (DESIGN section 11).

Everything the engine or Claude wants done to a league is a proposal; only the executor acts on one, and only after it
cleared policy and was approved (or, for the lineup kinds set to ``auto``, went unanswered until T-15). Usage::

    from fm.proposals import ProposalKind, WaiverPayload, approve, begin_execution, propose

    row = propose(store, config, league_row, ProposalKind.WAIVER,
                  WaiverPayload(add_espn_id=4046, drop_espn_id=3917315, bid_amount=12),
                  created_by="decide.waivers", scoring_period_id=5, deadline=waiver_run)
    approved = approve(store, row.row_id, decided_by="cli")           # issues the execution token
    executing = begin_execution(store, row.row_id, approved.execution_token)  # executor only; consumes it

- :mod:`fm.proposals.payloads`: one payload model per kind, the contract with the decision modules and the executor.
- :mod:`fm.proposals.policy`: kinds, their allowed settings (trade kinds are approve-only), and the guardrails.
- :mod:`fm.proposals.queue`: the status lifecycle, execution tokens, expiry at the deadline, T-15 auto-approval.
- :mod:`fm.proposals.pause`: the ``fm pause`` / ``fm resume`` kill switch.
"""

from __future__ import annotations

from fm.proposals.pause import PAUSE_FILE, PauseState, is_paused, pause, pause_file, pause_state, resume
from fm.proposals.payloads import (
    AddDropPayload,
    LineupMove,
    LineupPayload,
    Payload,
    TradePayload,
    TradeResponsePayload,
    TransactionCancelPayload,
    WaiverPayload,
)
from fm.proposals.policy import (
    ACQUISITION_KINDS,
    COUNTED_STATUSES,
    KINDS,
    TRADE_KINDS,
    KindSpec,
    PolicyError,
    ProposalError,
    ProposalKind,
    Verdict,
    acquisitions_this_week,
    default_setting,
    effective_setting,
    evaluate,
    faab_bid_cap,
    find_untouchables,
    kind_spec,
    parse_payload,
    stored_settings,
    validate_setting,
)
from fm.proposals.queue import (
    AUTO_LEAD,
    DECIDED_BY_AUTO,
    DECIDED_BY_EXPIRY,
    TRANSITIONS,
    LifecycleError,
    Outcome,
    PausedError,
    approve,
    auto_approve_due,
    begin_execution,
    expire_due,
    finish_execution,
    get_proposal,
    is_open,
    propose,
    reject,
)

__all__ = [
    "ACQUISITION_KINDS",
    "AUTO_LEAD",
    "COUNTED_STATUSES",
    "DECIDED_BY_AUTO",
    "DECIDED_BY_EXPIRY",
    "KINDS",
    "PAUSE_FILE",
    "TRADE_KINDS",
    "TRANSITIONS",
    "AddDropPayload",
    "KindSpec",
    "LifecycleError",
    "LineupMove",
    "LineupPayload",
    "Outcome",
    "PauseState",
    "PausedError",
    "Payload",
    "PolicyError",
    "ProposalError",
    "ProposalKind",
    "TradePayload",
    "TradeResponsePayload",
    "TransactionCancelPayload",
    "Verdict",
    "WaiverPayload",
    "acquisitions_this_week",
    "approve",
    "auto_approve_due",
    "begin_execution",
    "default_setting",
    "effective_setting",
    "evaluate",
    "expire_due",
    "faab_bid_cap",
    "find_untouchables",
    "finish_execution",
    "get_proposal",
    "is_open",
    "is_paused",
    "kind_spec",
    "parse_payload",
    "pause",
    "pause_file",
    "pause_state",
    "propose",
    "reject",
    "resume",
    "stored_settings",
    "validate_setting",
]
