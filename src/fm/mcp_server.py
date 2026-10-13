"""The MCP server (DESIGN section 12, ROADMAP #40): read tools plus ``create_proposal``, which never acts.

It lets Claude Code or Claude Desktop answer "who should I start at FLEX?" from the store ``fm sync`` filled. Run it
with ``fm mcp`` (stdio). Tools, all returning JSON-able dicts (no ANSI text):

- ``status``, ``lineup``, ``waivers``, ``trade_eval`` and ``list_proposals``: the read views of :mod:`fm.views`
  (:func:`~fm.views.status`, :func:`~fm.views.lineup`, :func:`~fm.views.waivers`, :func:`~fm.views.trade_eval` and
  :func:`~fm.views.list_proposals`), which the dashboard shows too. Each tool is a thin wrapper: it declares the
  parameters the client sees, pins the clock to :func:`_now` unless ``as_of`` is given, and turns a
  :class:`fm.views.ViewError` into a ``ToolError``. What each view returns, and that it stores nothing, is documented
  there.
- ``create_proposal``: :func:`fm.proposals.propose` with the policy verdict. It stores one ``proposed`` row, tagged
  ``created_by = "mcp"``, and nothing else.

No write path
-------------
Workers propose, the executor acts (CLAUDE.md). This module is a worker, so it has no way to act:

- It imports nothing from ``fm.executor`` or ``fm.browser`` and never opens an ESPN session: the reads are
  :mod:`fm.views`, which answer from the store and the cache dir the sync filled and say when a capture is missing. The
  test suite pins this with an AST check of this file's imports and a subprocess walk of ``sys.modules``, and pins
  :mod:`fm.views` the same way.
- It imports no approve, reject, execute, pause or resume function from :mod:`fm.proposals`, and registers no tool for
  one; the tool list is an allowlist (:data:`TOOL_NAMES`) the tests compare against, and they also scan every tool name
  and description for words that would suggest otherwise.
- ``create_proposal`` stores a row in status ``proposed`` and stops. A trade kind is approve-only by policy and a lineup
  proposal that policy would run as ``auto`` (approved on its own at T-15) is stored as ``approve`` instead: only the
  engine's own drafts may be auto-approved, never a move Claude suggested.
- Policy refusals come back as a result (``status: "blocked"`` with every reason), never as an exception.

Every tool opens the store and config per call (``FM_CONFIG_DIR`` / ``FM_CACHE_DIR`` as for the CLI), so a long-lived
server sees what ``fm sync`` stored since it started.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field, ValidationError

from fm import views
from fm.config import Config, ConfigError, load_config
from fm.decide.trades import DEFAULT_SEED, trade_etiquette
from fm.proposals import (
    PolicyError,
    ProposalKind,
    TradePayload,
    effective_setting,
    evaluate,
    kind_spec,
    propose,
)
from fm.proposals.policy import as_utc
from fm.store import LeagueRow, Store

SERVER_NAME = "espn-fantasy"
MCP_CREATED_BY = "mcp"
"""``proposals.created_by`` of every proposal this server stores."""
TOOL_NAMES = ("status", "lineup", "waivers", "trade_eval", "list_proposals", "create_proposal")
"""The whole tool list. Adding a tool means changing this tuple and the test that guards it."""
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
STORES_PROPOSAL = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
INSTRUCTIONS = (
    "Personal ESPN fantasy manager for NFL (ffl) and NBA (fba) leagues. The read tools (status, lineup, waivers, "
    "trade_eval, list_proposals) answer from the local store and change nothing. create_proposal records a suggestion "
    "for the user to review in the fm CLI or on their phone; it never changes a league, and trades always need the "
    "user's explicit review. Names are looked up for display, but every move is made of ESPN ids."
)
MAX_RATIONALE = 2000

LeagueArg = Annotated[
    str | None,
    Field(description="League key from config.toml (for example 'ffl' or 'fba'). Default: every configured league."),
]
AsOfArg = Annotated[
    str | None,
    Field(
        description="Pin the clock (ISO 8601, e.g. 2026-10-04T15:00Z; a naive time is UTC): locks, deadlines and "
        "availability are judged at this instant. Default: now."
    ),
]


def _now() -> datetime:
    """The wall clock; tests replace it."""
    return datetime.now(UTC)


def _config() -> Config:
    try:
        return load_config()
    except ConfigError as exc:
        raise ToolError(str(exc)) from None


def _answer[**P](view: Callable[P, dict[str, Any]], *args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
    """Run a read view, reporting its :class:`fm.views.ViewError` to the client as a ``ToolError``."""
    try:
        return view(*args, **kwargs)
    except views.ViewError as exc:
        raise ToolError(str(exc)) from None


# --- the read tools (fm.views) ----------------------------------------------------------------------------------------


def _status(league: LeagueArg = None) -> dict[str, Any]:
    return _answer(views.status, league)


def _lineup(
    league: LeagueArg = None,
    as_of: AsOfArg = None,
    opponent_team_id: Annotated[
        int | None,
        Field(
            ge=1,
            description="This period's opponent (ESPN team id); his synced roster gives the win probability. "
            "Needs a single league.",
        ),
    ] = None,
) -> dict[str, Any]:
    return _answer(views.lineup, league, as_of=as_of, now=_now(), opponent_team_id=opponent_team_id)


def _waivers(
    league: LeagueArg = None,
    as_of: AsOfArg = None,
    top: Annotated[int, Field(ge=0, le=100, description="Ranked (add, drop) pairs to return.")] = 10,
    max_moves: Annotated[int | None, Field(ge=1, description="Plan at most this many moves.")] = None,
) -> dict[str, Any]:
    return _answer(views.waivers, league, as_of=as_of, now=_now(), top=top, max_moves=max_moves)


def _trade_eval(
    deal: Annotated[str, Field(description='The deal, as in "give A, B get C": players by name or ESPN id.')],
    league: LeagueArg = None,
    as_of: AsOfArg = None,
    runs: Annotated[
        int, Field(ge=100, le=50_000, description="Simulated seasons (only when the league's schedule is captured).")
    ] = views.DEFAULT_TRADE_RUNS,
    seed: Annotated[
        int, Field(description="The simulation's seed; the same seed gives the same numbers.")
    ] = DEFAULT_SEED,
) -> dict[str, Any]:
    return _answer(views.trade_eval, deal, league, as_of=as_of, now=_now(), runs=runs, seed=seed)


def _list_proposals(
    league: LeagueArg = None,
    include_closed: Annotated[
        bool, Field(description="Include proposals already decided or finished (rejected, expired, verified, failed).")
    ] = False,
    limit: Annotated[int, Field(ge=1, le=200, description="At most this many, newest first.")] = 50,
) -> dict[str, Any]:
    return _answer(views.list_proposals, league, include_closed=include_closed, limit=limit)


# --- create_proposal --------------------------------------------------------------------------------------------------


def _dedupe_key(league: LeagueRow, kind: ProposalKind, payload: Mapping[str, Any]) -> str:
    """One open proposal per league, kind and payload: asking twice finds the first."""
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"{MCP_CREATED_BY}:{league.key}:{league.season}:{kind.value}:{digest}"


def _create_proposal(
    league: Annotated[str, Field(description="League key from config.toml (for example 'ffl' or 'fba').")],
    kind: Annotated[
        str,
        Field(
            description="Proposal kind: bench_inactive, lineup, add_drop, waiver, waiver_cancel, trade_propose, "
            "trade_accept, trade_decline or trade_cancel."
        ),
    ],
    payload: Annotated[
        dict[str, Any],
        Field(
            description="The move, as ESPN ids. lineup and bench_inactive: {moves: [{espn_id, from_slot_id, "
            "to_slot_id}]}. add_drop: {add_espn_id, drop_espn_id}. waiver: {add_espn_id, drop_espn_id, bid_amount}. "
            "trade_propose: {other_team_id, give_espn_ids, get_espn_ids}. trade_accept and trade_decline add "
            "espn_transaction_id. waiver_cancel and trade_cancel: {espn_transaction_id}."
        ),
    ],
    rationale: Annotated[str, Field(min_length=1, max_length=MAX_RATIONALE, description="Why, in a few sentences.")],
    scoring_period_id: Annotated[
        int | None, Field(ge=1, description="The scoring period the move is for (counts toward that week's cap).")
    ] = None,
    deadline: Annotated[
        str | None,
        Field(description="ISO 8601 time after which the move stops making sense (lock, waiver run). Default: none."),
    ] = None,
) -> dict[str, Any]:
    config = _config()
    try:
        spec = kind_spec(kind)
    except PolicyError as exc:
        return _refused("invalid", [str(exc)])
    try:
        model = spec.payload_type.model_validate(payload)
    except ValidationError as exc:
        reasons = [f"payload.{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        return _refused("invalid", reasons)
    try:
        due = None if deadline is None else as_utc(views.parse_as_of(deadline, now=_now()))
    except views.ViewError as exc:
        return _refused("invalid", [str(exc).replace("as_of", "deadline")])
    known = {configured.key for configured in config.leagues}
    if league not in known:
        return _refused(
            "invalid", [f"no league {league!r} in config.toml; known: {', '.join(sorted(known)) or 'none'}"]
        )
    at = _now()
    with Store.open() as store:
        row = store.leagues.by_key(league)
        if row is None:
            return _refused("invalid", [f"{league}: not synced; run fm sync"])
        open_before = {found.row_id for found in store.proposals.open(row.row_id)}
        key = _dedupe_key(row, spec.kind, model.model_dump(mode="json"))
        if isinstance(model, TradePayload) and spec.kind is ProposalKind.TRADE_PROPOSE:
            # The engine's own etiquette (DESIGN 9.4): one open offer a team and the weekly cap. Asking again for an
            # offer that is already open still returns it.
            etiquette = trade_etiquette(store, row, at=at)
            if key not in etiquette.open_keys and (blocker := etiquette.blocker(model.other_team_id)) is not None:
                return _refused("blocked", [blocker])
        try:
            with store.db.transaction():
                # Only the engine's own drafts may be approved on their own at T-15, never a move Claude suggested:
                # the policy ceiling is approve, applied before the auto checks (an auto lineup needs a deadline).
                stored = propose(
                    store,
                    config,
                    row,
                    spec.kind,
                    model,
                    created_by=MCP_CREATED_BY,
                    scoring_period_id=scoring_period_id,
                    engine_numbers={"origin": MCP_CREATED_BY},
                    rationale=rationale,
                    deadline=due,
                    dedupe_key=key,
                    max_setting="approve",
                    now=at,
                )
                note = None
                if stored.row_id not in open_before and _configured_setting(config, league, spec.kind) == "auto":
                    note = (
                        "policy auto was lowered to approve: a suggestion made through MCP is never approved on its own"
                    )
        except PolicyError as exc:
            verdict = evaluate(
                store,
                config,
                row,
                spec.kind,
                model,
                scoring_period_id=scoring_period_id,
                deadline=due,
                max_setting="approve",
                now=at,
            )
            return _refused("blocked", list(verdict.reasons) or [str(exc)], policy=views.verdict_dict(verdict))
        existing = stored.row_id in open_before
        return {
            "status": "existing" if existing else "proposed",
            "proposal": views.proposal_dict(stored, league),
            "reasons": [],
            "note": note,
            "message": (f"already open as #{stored.row_id}" if existing else f"stored as proposal #{stored.row_id}")
            + f" (policy {stored.policy}); it waits for the user's review and nothing has been changed in ESPN",
        }


def _configured_setting(config: Config, league: str, kind: ProposalKind) -> str | None:
    """The setting ``config.toml`` gives ``kind`` in ``league`` before the MCP ceiling; ``None`` for an unknown one."""
    try:
        return effective_setting(kind, config.league(league).policy)
    except KeyError:
        return None


def _refused(status: str, reasons: list[str], *, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "proposal": None,
        "policy": policy,
        "reasons": reasons,
        "note": None,
        "message": "nothing was stored" + (": " + "; ".join(reasons) if reasons else ""),
    }


# --- the server -------------------------------------------------------------------------------------------------------


def build_server() -> FastMCP:
    """A fresh server with exactly the tools in :data:`TOOL_NAMES`."""
    server = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)
    server.tool(
        _status,
        name="status",
        description="Per league: sync state, our record and FAAB, the captured pro schedule, the kill-switch state "
        "and the open proposals.",
        annotations=READ_ONLY,
    )
    server.tool(
        _lineup,
        name="lineup",
        description="Plan this scoring period's lineup from the synced roster: the recommended starters per slot with "
        "projections, expected points and win probability, the moves from the current lineup and the proposals the "
        "engine would suggest. A plan only; nothing is stored.",
        annotations=READ_ONLY,
    )
    server.tool(
        _waivers,
        name="waivers",
        description="Rank (add, drop) pairs for the waiver wire and free agents by rest-of-season value, with the "
        "moves the engine would suggest, the bid, replacement levels and every skipped candidate. A ranking only; "
        "nothing is stored.",
        annotations=READ_ONLY,
    )
    server.tool(
        _trade_eval,
        name="trade_eval",
        description='Judge one trade given as "give A, B get C": both sides\' rest-of-season lineup value and title '
        "odds, legality, P(accept) and a recommendation. Returns the payload to hand to create_proposal.",
        annotations=READ_ONLY,
    )
    server.tool(
        _list_proposals,
        name="list_proposals",
        description="The proposal queue with each proposal's kind, status, policy, payload and rationale.",
        annotations=READ_ONLY,
    )
    server.tool(
        _create_proposal,
        name="create_proposal",
        description="Record a suggested move for the user to review (CLI or phone). It goes through the league's "
        "policy and guardrails and is stored as proposed, or refused with every reason; duplicates return the "
        "existing one. It never changes a league: trades always need the user's explicit review.",
        annotations=STORES_PROPOSAL,
    )
    return server
