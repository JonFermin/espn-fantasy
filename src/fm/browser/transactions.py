"""ESPN transaction envelopes and error codes: what API mode sends to the write host, and what ESPN's answers mean.

ESPN's web app writes every move through one endpoint, ``POST https://lm-api-writes.fantasy.espn.com/apis/v3/games/
{ffl|fba}/seasons/{season}/segments/0/leagues/{id}/transactions/``, with one JSON envelope (DESIGN 6.3,
docs/espn-api.md section 4). This module builds that request the way the web client's own code does. The code is
saved verbatim in ``tests/fixtures/espn/real/webclient.json``, and ``tests/fixtures/espn/real/test_real_fixtures.py``
pins each rule below against it:

- :class:`Envelope` is the transaction model; :meth:`Envelope.body` is its serializer (``get()``). The body always
  starts ``{isLeagueManager: false, teamId, type}``, then ``memberId`` (the signed-in SWID) and ``scoringPeriodId``
  when set, ``executionType`` (``EXECUTE``, or ``CANCEL`` to withdraw a claim or offer) and ``items`` when there are
  any. Type-specific keys follow: ``bidAmount`` with ``WAIVER``, ``expirationDate`` and ``comment`` with
  ``TRADE_PROPOSAL``, ``comment`` with ``TRADE_DECLINE``, and ``relatedTransactionId`` with a cancel or a trade
  response. ``isLeagueManager`` is always false, so the league-manager keys never appear.
- The item builders (:func:`lineup_item`, :func:`add_item`, :func:`drop_item`, :func:`trade_item`) follow the
  client's: ``{playerId, type}``, then the team ids when they are set (0 is free agency, and the client leaves it
  out), then the lineup slots. ``LINEUP`` items carry slots and no team ids.
- :func:`lineup_type` and :func:`lineup_envelope` encode ``movePlayers``. A move for ESPN's current scoring period
  (``status.latestScoringPeriod``) is ``ROSTER``; a move for a later one is ``FUTURE_ROSTER`` with that period. Real
  records match this in both games (``ROSTER`` for NBA day 1, ``FUTURE_ROSTER`` for days 2, 3 and 5).
- :func:`transaction_request` wraps an envelope in a :class:`fm.browser.flows.WriteRequest` for this league's
  endpoint, with the headers the client adds (:data:`WEB_CLIENT_HEADERS`). The client's ``platformVersion`` query
  parameter names its own build, which we cannot know, so it is left out.

Sending is the executor's job. It sends a request exactly once from inside the logged-in browser context
(``fm.executor.transport.PlaywrightTransport`` over ``BrowserContext.request``, so it carries the same cookies and
client as the web app), with a hard timeout and never a retry. Nothing in this module sends anything.

ESPN's answers: a failed write returns ``details[{type, message, metaData.teamid, resolution}]``, and transaction
records carry ``FAILED_*`` statuses. :data:`ERROR_CODES` maps every code the web client knows (the ``errorCodes`` of
``tests/fixtures/espn/real/*/calendar.json``) plus the ones DESIGN 6.3 lists to an :class:`ErrorKind` and a plain
reading of the code. :func:`ui_may_follow` is the flows' rule for a UI click-through after a rejected request. It is
stricter than the executor's default (:func:`fm.browser.flows.rejection_allows_ui`) and never looser.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from fm.browser.flows import WriteRequest, WriteResponse, rejection_allows_ui, transactions_url
from fm.espn.auth import AUTH_FAILURE_CODE
from fm.espn.ids import Game
from fm.espn.models import EXECUTION_CANCEL, ITEM_ADD, ITEM_DROP, ITEM_LINEUP, ITEM_TRADE
from fm.proposals.payloads import LineupMove
from fm.store import LeagueRow

WEB_CLIENT_HEADERS: Mapping[str, str] = MappingProxyType(
    {"X-Fantasy-Source": "kona", "X-Fantasy-Platform": "espn-fantasy-web"}
)
"""Headers the web client adds to every request (``requestDefaults``, ``requestConfig``; docs/espn-api.md section 2).
The transport adds ``Content-Type`` and ``Accept``, and the browser session adds the cookies."""
HTTP_TOO_MANY_REQUESTS = 429


class TransactionType(StrEnum):
    """The envelope ``type`` values this tool sends (``typeNames`` in ``webclient.json``)."""

    ROSTER = "ROSTER"  # lineup moves for the current scoring period, or a bare drop
    FUTURE_ROSTER = "FUTURE_ROSTER"  # lineup moves for a later scoring period
    FREEAGENT = "FREEAGENT"  # a free-agent add, with a drop when the roster is full
    WAIVER = "WAIVER"  # a waiver claim, or its cancellation
    TRADE_PROPOSAL = "TRADE_PROPOSAL"  # a trade offer, or its cancellation
    TRADE_ACCEPT = "TRADE_ACCEPT"
    TRADE_DECLINE = "TRADE_DECLINE"


class ExecutionType(StrEnum):
    """``executionType``: the serializer defaults it to ``EXECUTE``; ``CANCEL`` withdraws a pending claim or offer."""

    EXECUTE = "EXECUTE"
    CANCEL = EXECUTION_CANCEL


CANCELLABLE: frozenset[TransactionType] = frozenset({TransactionType.WAIVER, TransactionType.TRADE_PROPOSAL})
"""Types the client cancels (``cancelWaiverClaim``, ``cancelTrade``): the same type with ``executionType: CANCEL``."""
_COMMENTED = frozenset({TransactionType.TRADE_PROPOSAL, TransactionType.TRADE_DECLINE})
_TRADE_RESPONSES = frozenset({TransactionType.TRADE_ACCEPT, TransactionType.TRADE_DECLINE})


def _require_id(value: object, what: str, *, positive: bool = True) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or (positive and value <= 0):
        kind = "a positive integer" if positive else "an integer"
        raise ValueError(f"{what} must be {kind}, got {value!r}")
    return value


# --- items ------------------------------------------------------------------------------------------------------------


def _item(
    player_id: int,
    item_type: str,
    *,
    from_team_id: int | None = None,
    to_team_id: int | None = None,
    from_slot_id: int | None = None,
    to_slot_id: int | None = None,
) -> dict[str, Any]:
    """The web client's item builder: ``playerId`` and ``type``, team ids only when truthy, slots when given."""
    item: dict[str, Any] = {"playerId": _require_id(player_id, "playerId", positive=False)}
    if item_type:
        item["type"] = item_type
    if from_team_id:
        item["fromTeamId"] = _require_id(from_team_id, "fromTeamId")
    if to_team_id:
        item["toTeamId"] = _require_id(to_team_id, "toTeamId")
    if from_slot_id is not None:
        item["fromLineupSlotId"] = _require_id(from_slot_id, "fromLineupSlotId", positive=False)
    if to_slot_id is not None:
        item["toLineupSlotId"] = _require_id(to_slot_id, "toLineupSlotId", positive=False)
    return item


def lineup_item(player_id: int, from_slot_id: int, to_slot_id: int) -> dict[str, Any]:
    """``{playerId, type: LINEUP, fromLineupSlotId, toLineupSlotId}``: one side of a lineup move. D/ST ids are negative.

    Raises ``ValueError`` for a move into the slot the player is already in (ESPN: ``TRAN_ROSTER_SAME_SLOT``).
    """
    if from_slot_id == to_slot_id:
        raise ValueError(f"player {player_id} is already in slot {to_slot_id} (TRAN_ROSTER_SAME_SLOT)")
    return _item(player_id, ITEM_LINEUP, from_slot_id=from_slot_id, to_slot_id=to_slot_id)


def add_item(player_id: int, to_team_id: int) -> dict[str, Any]:
    """``{playerId, type: ADD, toTeamId}``: a player joining ``to_team_id`` (ours)."""
    return _item(player_id, ITEM_ADD, to_team_id=_require_id(to_team_id, "toTeamId"))


def drop_item(player_id: int, from_team_id: int) -> dict[str, Any]:
    """``{playerId, type: DROP, fromTeamId}``: a player leaving ``from_team_id`` (ours) for free agency."""
    return _item(player_id, ITEM_DROP, from_team_id=_require_id(from_team_id, "fromTeamId"))


def trade_item(player_id: int, from_team_id: int, to_team_id: int) -> dict[str, Any]:
    """``{playerId, type: TRADE, fromTeamId, toTeamId}``: one player changing teams in a trade."""
    return _item(
        player_id,
        ITEM_TRADE,
        from_team_id=_require_id(from_team_id, "fromTeamId"),
        to_team_id=_require_id(to_team_id, "toTeamId"),
    )


# --- envelopes --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Envelope:
    """One transaction as the web client's model holds it. :meth:`body` is what goes over the wire.

    There is no ``is_league_manager`` field: league-manager powers are never used (DESIGN 6.3), so the body always says
    ``isLeagueManager: false`` and never carries ``isActingAsTeamOwner`` or ``skipTransactionCounters``. Construction
    refuses combinations the client never sends (a bid outside a waiver claim, a cancel of a lineup move, a trade
    response without the offer's id).
    """

    team_id: int
    type: TransactionType
    member_id: str | None = None
    """The account's SWID (``{...}``); the serializer leaves ``memberId`` out when it is empty."""
    scoring_period_id: int | None = None
    """The period the move is for; left out when unset, as the serializer does."""
    items: tuple[Mapping[str, Any], ...] = ()
    execution_type: ExecutionType = ExecutionType.EXECUTE
    bid_amount: int | None = None
    """``WAIVER`` only: the FAAB bid (0 in a league without FAAB). ``None`` leaves it out, as a cancel does."""
    related_transaction_id: str | None = None
    """The claim or offer a cancel or trade response refers to."""
    expiration_date: int | None = None
    """``TRADE_PROPOSAL`` only: when the offer expires, in epoch milliseconds like ESPN's records."""
    comment: str | None = None
    """``TRADE_PROPOSAL`` / ``TRADE_DECLINE`` only."""

    def __post_init__(self) -> None:
        _require_id(self.team_id, "teamId")
        kind = TransactionType(self.type)
        execution = ExecutionType(self.execution_type)
        if self.scoring_period_id is not None:
            _require_id(self.scoring_period_id, "scoringPeriodId")
        if self.bid_amount is not None:
            if kind is not TransactionType.WAIVER:
                raise ValueError(f"bidAmount travels only with WAIVER, not {kind.value}")
            if _require_id(self.bid_amount, "bidAmount", positive=False) < 0:
                raise ValueError(f"bidAmount cannot be negative, got {self.bid_amount}")
        if self.expiration_date is not None and kind is not TransactionType.TRADE_PROPOSAL:
            raise ValueError(f"expirationDate travels only with TRADE_PROPOSAL, not {kind.value}")
        if self.comment is not None and kind not in _COMMENTED:
            raise ValueError(f"a comment travels only with TRADE_PROPOSAL or TRADE_DECLINE, not {kind.value}")
        if execution is ExecutionType.CANCEL:
            if kind not in CANCELLABLE:
                raise ValueError(f"only WAIVER and TRADE_PROPOSAL are cancelled, not {kind.value}")
            if not self.related_transaction_id:
                raise ValueError("a cancel needs the relatedTransactionId of the claim or offer it withdraws")
        if kind in _TRADE_RESPONSES and not self.related_transaction_id:
            raise ValueError(f"{kind.value} needs the relatedTransactionId of the offer it answers")
        if self.related_transaction_id and kind not in CANCELLABLE | _TRADE_RESPONSES:
            raise ValueError(f"relatedTransactionId does not travel with {kind.value}")

    def body(self) -> dict[str, Any]:
        """The JSON body, key for key and in the order of the web client's serializer (``get()``)."""
        kind = TransactionType(self.type)
        body: dict[str, Any] = {"isLeagueManager": False, "teamId": self.team_id, "type": kind.value}
        if self.member_id:
            body["memberId"] = self.member_id
        if self.scoring_period_id:
            body["scoringPeriodId"] = self.scoring_period_id
        body["executionType"] = ExecutionType(self.execution_type).value
        if self.items:
            body["items"] = [dict(item) for item in self.items]
        if kind is TransactionType.WAIVER:
            if self.bid_amount is not None:
                body["bidAmount"] = self.bid_amount
            if self.related_transaction_id:
                body["relatedTransactionId"] = self.related_transaction_id
        if kind is TransactionType.TRADE_PROPOSAL and self.expiration_date is not None:
            body["expirationDate"] = self.expiration_date
        if kind in _COMMENTED and self.comment is not None:
            body["comment"] = self.comment
        if (kind is TransactionType.TRADE_PROPOSAL and self.related_transaction_id) or kind in _TRADE_RESPONSES:
            body["relatedTransactionId"] = self.related_transaction_id
        return body


def lineup_type(scoring_period_id: int, latest_scoring_period: int) -> TransactionType:
    """``ROSTER`` for ESPN's current scoring period, ``FUTURE_ROSTER`` for a later one (the client's ``movePlayers``).

    Raises ``ValueError`` for a period before ``latest_scoring_period``: a past lineup cannot change.
    """
    target = _require_id(scoring_period_id, "scoringPeriodId")
    latest = _require_id(latest_scoring_period, "latestScoringPeriod")
    if target < latest:
        raise ValueError(f"scoring period {target} is over: ESPN is on scoring period {latest}")
    return TransactionType.FUTURE_ROSTER if target > latest else TransactionType.ROSTER


def lineup_envelope(
    *,
    team_id: int,
    member_id: str | None,
    scoring_period_id: int,
    latest_scoring_period: int,
    moves: Iterable[LineupMove],
) -> Envelope:
    """The ``set_lineup`` envelope: one ``LINEUP`` item per move, both sides of every swap, in the order given.

    ``scoring_period_id`` is the period the lineup is for; ``latest_scoring_period`` is ESPN's current one
    (``status.latestScoringPeriod``), which decides between ``ROSTER`` and ``FUTURE_ROSTER`` (:func:`lineup_type`).
    """
    items = tuple(lineup_item(move.espn_id, move.from_slot_id, move.to_slot_id) for move in moves)
    if not items:
        raise ValueError("a lineup transaction needs at least one move")
    players = [item["playerId"] for item in items]
    repeated = sorted({player for player in players if players.count(player) > 1})
    if repeated:
        raise ValueError(f"players {repeated} are moved more than once in one transaction")
    return Envelope(
        team_id=team_id,
        type=lineup_type(scoring_period_id, latest_scoring_period),
        member_id=member_id,
        scoring_period_id=scoring_period_id,
        items=items,
    )


def transaction_request(league: LeagueRow, envelope: Envelope) -> WriteRequest:
    """The API-mode write for ``envelope`` in ``league``: a POST to its ``transactions/`` endpoint on ESPN's write host
    with the web client's headers. The executor checks it (``fm.executor.transport.check_write_request``) and sends it
    once."""
    url = transactions_url(Game.from_sport(league.sport), league.season, league.espn_league_id)
    return WriteRequest(url=url, body=envelope.body(), headers=dict(WEB_CLIENT_HEADERS))


# --- error codes ------------------------------------------------------------------------------------------------------


class ErrorKind(StrEnum):
    """What an ESPN error code says about a move."""

    LOCKED = "locked"  # a lock has passed: a lineup slot, adds and drops, trading, or the whole team
    ROSTER_LIMIT = "roster_limit"  # the roster, a position or a lineup slot would be over its limit
    INELIGIBLE = "ineligible"  # a slot the player cannot fill, or a move that changes nothing
    UNAVAILABLE = "unavailable"  # a player is not where the move assumes: taken, on waivers, gone, protected
    LIMIT = "limit"  # an acquisition, trade or budget limit
    NOT_ALLOWED = "not_allowed"  # the team may not do this
    INVALID = "invalid"  # ESPN could not accept the request as sent
    AUTH = "auth"  # the ESPN session is gone: run fm login
    UNKNOWN = "unknown"  # ESPN failed, or a code this tool does not know yet


@dataclass(frozen=True, slots=True)
class ErrorCode:
    """One ESPN error code (``details[].type`` of a failed write, or a ``FAILED_*`` record status)."""

    code: str
    kind: ErrorKind
    meaning: str
    """A plain reading of the code for logs, the CLI and notifications."""

    @property
    def known(self) -> bool:
        return self.code in ERROR_CODES

    def describe(self) -> str:
        """``TRAN_LINEUP_LOCKED: a player in the move is locked (his game has started)``."""
        return f"{self.code}: {self.meaning}"


def _codes(kind: ErrorKind, entries: Mapping[str, str]) -> dict[str, ErrorCode]:
    return {code: ErrorCode(code, kind, meaning) for code, meaning in entries.items()}


_ROSTER_FULL = "the roster would be over its size limit, so the move needs a drop"
_ROSTER_FULL_RESERVED = "the roster would be over its size limit counting players held by a pending trade"
ERROR_CODES: Mapping[str, ErrorCode] = MappingProxyType(
    {
        **_codes(ErrorKind.AUTH, {AUTH_FAILURE_CODE: "the ESPN session is missing or expired; run fm login"}),
        **_codes(
            ErrorKind.LOCKED,
            {
                "TRAN_LINEUP_LOCKED": "a player in the move is locked (his game has started)",
                "FAILED_LINEUPLOCK": "a player's lineup slot is locked (his game has started)",
                "FAILED_ROSTERLOCK": "adds and drops are locked for a player in the move (the roster lock has passed)",
                "FAILED_TRADELOCK": "a player in the trade is locked for trading",
                "FAILED_TRANSACTIONLOCKED": "the team's transactions are locked",
            },
        ),
        **_codes(
            ErrorKind.ROSTER_LIMIT,
            {
                "TRAN_ROSTER_LIMIT_EXCEEDED_ONE": _ROSTER_FULL,
                "TRAN_ROSTER_LIMIT_EXCEEDED_ONE_LM": _ROSTER_FULL,
                "TRAN_ROSTER_LIMIT_EXCEEDED_PLURAL": _ROSTER_FULL,
                "TRAN_ROSTER_LIMIT_EXCEEDED_PLURAL_LM": _ROSTER_FULL,
                "TRAN_ROSTER_LIMIT_EXCEEDED_TRADE_RESERVED_ONE": _ROSTER_FULL_RESERVED,
                "TRAN_ROSTER_LIMIT_EXCEEDED_TRADE_RESERVED_PLURAL": _ROSTER_FULL_RESERVED,
                "TRAN_ROSTER_POSITION_LIMIT_EXCEEDED": "the roster would be over a position limit",
                "TRAN_ROSTER_POSITION_LIMIT_EXCEEDED_LM": "the roster would be over a position limit",
                "TRAN_ROSTER_SLOT_LIMIT_EXCEEDED": "a lineup slot would hold more players than the league allows",
                "TRAN_ROSTER_SLOT_LIMIT_EXCEEDED_LM": "a lineup slot would hold more players than the league allows",
                "FAILED_ROSTERLIMIT": _ROSTER_FULL,
                "FAILED_POSITIONLIMIT": "the roster would be over a position limit",
                "FAILED_SLOTLIMIT": "a lineup slot would hold more players than the league allows",
                "FAILED_INVALIDROSTER": "the roster would not be legal after the move",
            },
        ),
        **_codes(
            ErrorKind.INELIGIBLE,
            {
                "TRAN_ROSTER_SAME_SLOT": "a player is already in the slot the move puts him in",
                "FAILED_INELIGIBLESLOT": "a player cannot fill the slot the move puts him in",
                "FAILED_IRSLOT": "a player cannot go to (or stay in) an IR slot",
                "FAILED_TRADEPLAYERTOIR": "the trade would put a player in an IR slot",
            },
        ),
        **_codes(
            ErrorKind.UNAVAILABLE,
            {
                "FAILED_INVALIDPLAYERSOURCE": "the player is no longer available that way (taken, or the claim lost)",
                "FAILED_NOTCLEAREDWAIVERS": "the player is on waivers: claim him instead of adding him",
                "FAILED_PLAYERALREADYDROPPED": "the player to drop is no longer on the roster",
                "FAILED_PLAYERNOTONROSTER": "a player in the move is not on the roster",
                "FAILED_UNDROPPABLEPLAYER": "the player to drop is on ESPN's undroppable list",
                "FAILED_DROPRESERVEDPLAYER": "the player to drop is held by a pending trade",
                "FAILED_TRADE_RESERVED": "a player is already held by another pending trade",
                "FAILED_DEPENDENCY": "a move this one depends on did not go through",
            },
        ),
        **_codes(
            ErrorKind.LIMIT,
            {
                "FAILED_ACQUISITIONLIMIT": "the season's acquisition limit is reached",
                "FAILED_MATCHUPACQUISITIONLIMIT": "the matchup's acquisition limit is reached",
                "FAILED_TRADELIMIT": "the league's trade limit is reached",
                "FAILED_AUCTIONBUDGETEXCEEDED": "the bid is more than the FAAB budget left",
                "FAILED_MINIMUMBID": "the bid is below the league's minimum",
            },
        ),
        **_codes(
            ErrorKind.NOT_ALLOWED,
            {
                "FAILED_NOPERMISSION": "this account may not make the move for this team",
                "FAILED_CANCELACCEPTEDTRADE": "an accepted trade cannot be cancelled",
                "FAILED_CANCELTRANSACTION": "the transaction could not be cancelled",
            },
        ),
        **_codes(
            ErrorKind.INVALID,
            {
                "FAILED_INVALID_FORMAT": "ESPN could not read the request",
                "FAILED_TRANSVALIDATIONFAILED": "ESPN's validation rejected the transaction",
                "FAILED_BADUNIVERSE": "a player is not in this game's player pool",
            },
        ),
        **_codes(
            ErrorKind.UNKNOWN,
            {
                "FAILED_UNKNOWN": "ESPN failed without saying why",
                "FAILED_PULL_FROM_PREBUFFER": "ESPN failed while processing the transaction",
            },
        ),
    }
)
"""Every ESPN error code the web client knows (``errorCodes`` in the calendar fixtures, 42 of them), plus the ones
DESIGN 6.3 lists (``TRAN_LINEUP_LOCKED``, ``TRAN_ROSTER_SAME_SLOT``, ``AUTH_MISSING_CREDENTIALS``). The meanings read
the code names; ESPN documents none of them."""


def error_code(code: str) -> ErrorCode:
    """What ``code`` means; a code missing from :data:`ERROR_CODES` is ``AUTH`` for an ``AUTH_`` prefix, else
    ``UNKNOWN``, so a new ESPN code shows up in output instead of breaking anything."""
    known = ERROR_CODES.get(code)
    if known is not None:
        return known
    if code.startswith("AUTH_"):
        return ErrorCode(code, ErrorKind.AUTH, "an ESPN sign-in error; run fm login")
    return ErrorCode(code, ErrorKind.UNKNOWN, "an ESPN code this tool does not know yet")


def explain(response: WriteResponse) -> tuple[ErrorCode, ...]:
    """The error codes in ESPN's answer to a write (``details[].type`` and any code in the text), explained."""
    return tuple(error_code(code) for code in response.error_codes)


def ui_may_follow(response: WriteResponse) -> bool:
    """Whether a UI click-through may follow this answer to an API-mode write.

    Only when :func:`fm.browser.flows.rejection_allows_ui` allows it (a definite rejection without a league-rule or
    sign-in code) and ESPN did not answer 429: the UI posts the same transaction from the same session, so it would be
    throttled too. No code in :data:`ERROR_CODES` lets the UI follow: each one is a league rule or a sign-in problem,
    which the UI meets the same way.
    """
    if not rejection_allows_ui(response) or response.status == HTTP_TOO_MANY_REQUESTS:
        return False
    return not any(code.known for code in explain(response))
