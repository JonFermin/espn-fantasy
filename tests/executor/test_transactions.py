"""Transaction envelopes and ESPN error codes (ROADMAP #25): ``fm.browser.transactions`` against docs/espn-api.md
section 4.

The reference for a write is what ESPN's own web app sent: the requests the guarded UI capture aborted and saved as
``tests/fixtures/espn/real/{ffl,fba}/write_*.json`` (ROADMAP #14), and behind them the web client's serializer, item
builders and service methods, saved verbatim in ``tests/fixtures/espn/real/webclient.json`` and pinned to the rules of
docs/espn-api.md section 4 by ``tests/fixtures/espn/real/test_real_fixtures.py``. These tests hold the envelopes to
those sources:

- each captured flow, built from a proposal's own ids the way its flow builds it, is the captured body key for key;
- the keys each body may carry come from section 4's serializer table;
- their order comes from the serializer's ``get()``, and the items' order from the item builder;
- the lineup transactions on record in both real leagues are rebuilt from their own items, and come out with the type
  (``ROSTER`` or ``FUTURE_ROSTER``), period and items ESPN recorded.

Offline: nothing here sends a request.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from fm.browser.flows import WriteRefusedError, WriteResponse, rejection_allows_ui
from fm.browser.transactions import (
    ERROR_CODES,
    WEB_CLIENT_HEADERS,
    Envelope,
    ErrorKind,
    ExecutionType,
    TransactionType,
    add_item,
    drop_item,
    error_code,
    explain,
    lineup_envelope,
    lineup_item,
    lineup_type,
    trade_item,
    transaction_request,
    ui_may_follow,
)
from fm.executor import check_write_request
from fm.proposals import LineupMove
from fm.store import LeagueRow

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "tests" / "fixtures" / "espn" / "real"
API_DOC = ROOT / "docs" / "espn-api.md"
SWID = "{00000000-0000-0000-0000-000000000001}"
NOW = datetime(2026, 10, 5, 18, 0, tzinfo=UTC)
LATEST = {"ffl": 4, "fba": 1}
"""``status.latestScoringPeriod`` of each real league when its fixtures were captured."""
LEAGUE_MANAGER_KEYS = {"isActingAsTeamOwner", "skipTransactionCounters"}
DOCUMENTED_TYPES = {*TransactionType, "TRADE_UPHOLD", "TRADE_VETO"}
"""The envelope types section 4's serializer table names (the last two are league votes this tool never casts)."""
NFL = LeagueRow(key="nfl", sport="nfl", espn_league_id=1010101, season=2026, team_id=1, as_of=NOW)
NBA = LeagueRow(key="nba", sport="nba", espn_league_id=2020202, season=2027, team_id=1, as_of=NOW)


def load(relative: str) -> Any:
    return json.loads((REAL / relative).read_text(encoding="utf-8"))


def webclient(part: str) -> str:
    return load("webclient.json")["transactionCode"]["excerpts"][part]


def ordered(keys: list[str], order: list[str]) -> bool:
    """``keys`` are a subsequence of ``order``: each one found after the one before it."""
    position = 0
    for key in keys:
        try:
            position = order.index(key, position) + 1
        except ValueError:
            return False
    return True


def serializer_order() -> list[str]:
    """Every assignment the web client's serializer ``get()`` makes, in order. A key may appear twice:
    ``relatedTransactionId`` is written in the ``WAIVER`` branch and again, after ``comment``, for trade answers."""
    model = webclient("model")
    body = model[model.index("get(){") :]
    first = re.search(r"const \w+=\{((?:\w+:this\.\w+,?)+)\}", body)
    assert first is not None
    keys = re.findall(r"(\w+):this\.\1", first.group(1))
    return keys + re.findall(r"\b\w+\.(\w+)=this\.\1\b", body[first.end() :])


def item_order() -> list[str]:
    """The item keys in the order the web client's item builder writes them."""
    builder = re.search(r"let (\w)=\{playerId:\w\};(.*?)return \1\}", webclient("model"))
    assert builder is not None
    return ["playerId", *re.findall(r"\b\w\.(\w+)=\w\)", builder.group(2))]


def documented_keys() -> tuple[set[str], dict[str, set[str]], set[str]]:
    """docs/espn-api.md section 4's serializer table: keys any body may carry, keys per type, league-manager keys.

    Each line is ``keys  condition``; a condition naming transaction types limits its keys to those types.
    """
    text = API_DOC.read_text(encoding="utf-8")
    section = text[text.index("## 4. Write flows") :]
    block = section[section.index("```") + 3 :]
    block = block[: block.index("```")]
    general: set[str] = set()
    by_type: dict[str, set[str]] = {}
    manager: set[str] = set()
    for line in block.strip().splitlines():
        left, _, right = line.strip().partition("  ")
        keys = set(re.findall(r"\b[a-z][A-Za-z]+\b", left.replace("when set", "")))
        types = [word for word in re.findall(r"\b[A-Z][A-Z_]+\b", right) if word in DOCUMENTED_TYPES]
        if "isLeagueManager" in right:
            manager |= keys
        elif types:
            for kind in types:
                by_type.setdefault(kind, set()).update(keys)
        else:
            general |= keys
    return general, by_type, manager


def swap(a: int, b: int, a_slot: int, b_slot: int) -> list[LineupMove]:
    return [
        LineupMove(espn_id=a, from_slot_id=a_slot, to_slot_id=b_slot),
        LineupMove(espn_id=b, from_slot_id=b_slot, to_slot_id=a_slot),
    ]


EVERY_KIND = [
    lineup_envelope(
        team_id=1, member_id=SWID, scoring_period_id=4, latest_scoring_period=4, moves=swap(4430807, 4685382, 2, 20)
    ),
    lineup_envelope(
        team_id=1, member_id=SWID, scoring_period_id=3, latest_scoring_period=1, moves=swap(3059319, 4432848, 11, 12)
    ),
    Envelope(1, TransactionType.ROSTER, SWID, 4, items=(drop_item(4360078, 1),)),
    Envelope(1, TransactionType.FREEAGENT, SWID, 4, items=(add_item(4426515, 1), drop_item(4360078, 1))),
    Envelope(1, TransactionType.WAIVER, SWID, 4, items=(add_item(4426515, 1), drop_item(4360078, 1)), bid_amount=0),
    Envelope(1, TransactionType.WAIVER, SWID, 4, execution_type=ExecutionType.CANCEL, related_transaction_id="c-1"),
    Envelope(
        1,
        TransactionType.TRADE_PROPOSAL,
        SWID,
        4,
        items=(trade_item(4430807, 1, 3), trade_item(4241416, 3, 1)),
        expiration_date=1791218400000,
        comment="",
    ),
    Envelope(
        1, TransactionType.TRADE_PROPOSAL, SWID, 4, execution_type=ExecutionType.CANCEL, related_transaction_id="o-1"
    ),
    Envelope(1, TransactionType.TRADE_ACCEPT, SWID, 4, items=(drop_item(4360078, 1),), related_transaction_id="o-2"),
    Envelope(1, TransactionType.TRADE_DECLINE, SWID, 4, related_transaction_id="o-3", comment="no thanks"),
]


# --- envelopes --------------------------------------------------------------------------------------------------------


def test_a_lineup_swap_is_the_documented_roster_envelope() -> None:
    envelope = lineup_envelope(
        team_id=1, member_id=SWID, scoring_period_id=4, latest_scoring_period=4, moves=swap(4430807, 4685382, 2, 20)
    )
    body = envelope.body()
    assert body == {
        "isLeagueManager": False,
        "teamId": 1,
        "type": "ROSTER",
        "memberId": SWID,
        "scoringPeriodId": 4,
        "executionType": "EXECUTE",
        "items": [
            {"playerId": 4430807, "type": "LINEUP", "fromLineupSlotId": 2, "toLineupSlotId": 20},
            {"playerId": 4685382, "type": "LINEUP", "fromLineupSlotId": 20, "toLineupSlotId": 2},
        ],
    }
    assert list(body) == ["isLeagueManager", "teamId", "type", "memberId", "scoringPeriodId", "executionType", "items"]


def test_the_current_period_is_roster_a_later_one_future_roster_and_a_past_one_refused() -> None:
    assert lineup_type(1, 1) is TransactionType.ROSTER
    assert lineup_type(2, 1) is TransactionType.FUTURE_ROSTER and lineup_type(153, 1) is TransactionType.FUTURE_ROSTER
    with pytest.raises(ValueError, match="scoring period 3 is over: ESPN is on scoring period 4"):
        lineup_type(3, 4)
    future = lineup_envelope(
        team_id=1, member_id=SWID, scoring_period_id=3, latest_scoring_period=1, moves=swap(3059319, 4432848, 11, 12)
    ).body()
    assert (future["type"], future["scoringPeriodId"]) == ("FUTURE_ROSTER", 3)


@pytest.mark.parametrize("game", ["ffl", "fba"])
def test_recorded_lineup_transactions_rebuild_from_their_own_items(game: str) -> None:
    """Every ``ROSTER`` / ``FUTURE_ROSTER`` record of the real league, rebuilt from its items, is what ESPN recorded."""
    records = load(f"{game}/mTransactions2.json")["transactions"]
    lineups = [t for t in records if t["type"] in ("ROSTER", "FUTURE_ROSTER")]
    assert {t["type"] for t in lineups} == ({"ROSTER", "FUTURE_ROSTER"} if game == "fba" else {"ROSTER"})
    for record in lineups:
        moves = [
            LineupMove(
                espn_id=item["playerId"], from_slot_id=item["fromLineupSlotId"], to_slot_id=item["toLineupSlotId"]
            )
            for item in record["items"]
        ]
        body = lineup_envelope(
            team_id=record["teamId"],
            member_id=record["memberId"],
            scoring_period_id=record["scoringPeriodId"],
            latest_scoring_period=LATEST[game],
            moves=moves,
        ).body()
        for key in ("isLeagueManager", "teamId", "type", "memberId", "scoringPeriodId", "executionType"):
            assert body[key] == record[key], (record["id"], key)
        assert body["items"] == [
            {key: item[key] for key in ("playerId", "type", "fromLineupSlotId", "toLineupSlotId")}
            for item in record["items"]
        ]


@pytest.mark.parametrize("envelope", EVERY_KIND, ids=lambda e: f"{e.type.value}-{e.execution_type.value}")
def test_bodies_carry_the_keys_section_4_lists_in_the_serializers_order(envelope: Envelope) -> None:
    general, by_type, manager = documented_keys()
    assert manager == LEAGUE_MANAGER_KEYS
    body = envelope.body()
    assert set(body) <= general | by_type.get(envelope.type.value, set()), envelope
    assert body["isLeagueManager"] is False and not set(body) & LEAGUE_MANAGER_KEYS
    assert ordered(list(body), serializer_order())
    for item in body.get("items", []):
        assert ordered(list(item), item_order())


def test_type_specific_keys_follow_the_serializer() -> None:
    by_kind = {(e.type, e.execution_type): e.body() for e in EVERY_KIND}
    claim = by_kind[TransactionType.WAIVER, ExecutionType.EXECUTE]
    assert claim["bidAmount"] == 0 and "relatedTransactionId" not in claim and "comment" not in claim
    assert by_kind[TransactionType.WAIVER, ExecutionType.CANCEL] == {
        "isLeagueManager": False,
        "teamId": 1,
        "type": "WAIVER",
        "memberId": SWID,
        "scoringPeriodId": 4,
        "executionType": "CANCEL",
        "relatedTransactionId": "c-1",
    }
    offer = by_kind[TransactionType.TRADE_PROPOSAL, ExecutionType.EXECUTE]
    assert offer["expirationDate"] == 1791218400000 and offer["comment"] == "" and "relatedTransactionId" not in offer
    assert offer["items"] == [
        {"playerId": 4430807, "type": "TRADE", "fromTeamId": 1, "toTeamId": 3},
        {"playerId": 4241416, "type": "TRADE", "fromTeamId": 3, "toTeamId": 1},
    ]
    withdrawn = by_kind[TransactionType.TRADE_PROPOSAL, ExecutionType.CANCEL]
    assert (withdrawn["executionType"], withdrawn["relatedTransactionId"]) == ("CANCEL", "o-1")
    assert "items" not in withdrawn and "expirationDate" not in withdrawn
    accept = by_kind[TransactionType.TRADE_ACCEPT, ExecutionType.EXECUTE]
    assert accept["relatedTransactionId"] == "o-2" and accept["items"] == [
        {"playerId": 4360078, "type": "DROP", "fromTeamId": 1}
    ]
    decline = by_kind[TransactionType.TRADE_DECLINE, ExecutionType.EXECUTE]
    assert (decline["comment"], decline["relatedTransactionId"]) == ("no thanks", "o-3")
    add = by_kind[TransactionType.FREEAGENT, ExecutionType.EXECUTE]
    assert "bidAmount" not in add and add["items"] == [
        {"playerId": 4426515, "type": "ADD", "toTeamId": 1},
        {"playerId": 4360078, "type": "DROP", "fromTeamId": 1},
    ]
    bare = Envelope(1, TransactionType.ROSTER).body()
    assert bare == {"isLeagueManager": False, "teamId": 1, "type": "ROSTER", "executionType": "EXECUTE"}
    assert "memberId" not in Envelope(1, TransactionType.ROSTER, member_id="").body()  # the serializer's `&&`


def test_items_follow_the_web_client_builders() -> None:
    assert item_order() == ["playerId", "type", "fromTeamId", "toTeamId", "fromLineupSlotId", "toLineupSlotId"]
    assert lineup_item(-16005, 16, 20) == {
        "playerId": -16005,  # a D/ST's id is negative
        "type": "LINEUP",
        "fromLineupSlotId": 16,
        "toLineupSlotId": 20,
    }
    assert add_item(1, 4) == {"playerId": 1, "type": "ADD", "toTeamId": 4}
    assert drop_item(1, 4) == {"playerId": 1, "type": "DROP", "fromTeamId": 4}
    assert trade_item(1, 4, 7) == {"playerId": 1, "type": "TRADE", "fromTeamId": 4, "toTeamId": 7}
    with pytest.raises(ValueError, match="TRAN_ROSTER_SAME_SLOT"):
        lineup_item(1, 20, 20)
    for bad in (0, True, -3):
        with pytest.raises(ValueError, match="toTeamId"):
            add_item(1, bad)


def test_envelopes_refuse_what_the_web_client_never_sends() -> None:
    refused: list[tuple[dict[str, Any], str]] = [
        ({"type": TransactionType.ROSTER, "bid_amount": 5}, "bidAmount travels only with WAIVER"),
        ({"type": TransactionType.WAIVER, "bid_amount": -1}, "cannot be negative"),
        ({"type": TransactionType.WAIVER, "expiration_date": 1}, "expirationDate travels only with TRADE_PROPOSAL"),
        ({"type": TransactionType.FREEAGENT, "comment": "hi"}, "a comment travels only with"),
        (
            {"type": TransactionType.ROSTER, "execution_type": ExecutionType.CANCEL, "related_transaction_id": "x"},
            "only WAIVER and TRADE_PROPOSAL are cancelled",
        ),
        ({"type": TransactionType.WAIVER, "execution_type": ExecutionType.CANCEL}, "a cancel needs"),
        ({"type": TransactionType.TRADE_ACCEPT}, "TRADE_ACCEPT needs the relatedTransactionId"),
        ({"type": TransactionType.FREEAGENT, "related_transaction_id": "x"}, "does not travel with FREEAGENT"),
        ({"type": TransactionType.ROSTER, "scoring_period_id": 0}, "scoringPeriodId must be a positive integer"),
    ]
    for changes, message in refused:
        with pytest.raises(ValueError, match=message):
            Envelope(**{"team_id": 1, **changes})
    with pytest.raises(ValueError, match="teamId must be a positive integer"):
        Envelope(False, TransactionType.ROSTER)  # a bool is not a team id
    twice = [
        LineupMove(espn_id=7, from_slot_id=11, to_slot_id=12),
        LineupMove(espn_id=7, from_slot_id=12, to_slot_id=11),
    ]
    with pytest.raises(ValueError, match="moved more than once"):
        lineup_envelope(team_id=1, member_id=SWID, scoring_period_id=1, latest_scoring_period=1, moves=twice)
    with pytest.raises(ValueError, match="at least one move"):
        lineup_envelope(team_id=1, member_id=SWID, scoring_period_id=1, latest_scoring_period=1, moves=[])


# --- the request ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("league", "path"), [(NFL, "ffl/seasons/2026"), (NBA, "fba/seasons/2027")])
def test_requests_post_to_the_league_endpoint_on_the_write_host(league: LeagueRow, path: str) -> None:
    envelope = lineup_envelope(
        team_id=1, member_id=SWID, scoring_period_id=1, latest_scoring_period=1, moves=swap(3059319, 4432848, 11, 12)
    )
    request = transaction_request(league, envelope)
    assert request.method == "POST" and request.body == envelope.body()
    assert request.url == (
        f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/{path}/segments/0/leagues/{league.espn_league_id}"
        "/transactions/"
    )
    check_write_request(request, league)  # the executor's gate lets it through
    assert dict(request.headers) == dict(WEB_CLIENT_HEADERS)
    assert not {name.lower() for name in request.headers} & {"cookie", "authorization"}
    other_team = transaction_request(league, Envelope(2, TransactionType.ROSTER, SWID, 1))
    with pytest.raises(WriteRefusedError, match="teamId must be our team 1"):
        check_write_request(other_team, league)


def test_the_headers_are_the_web_clients() -> None:
    source = re.search(r'"X-Fantasy-Source":"([\w-]+)"', webclient("requestDefaults"))
    assert source is not None and WEB_CLIENT_HEADERS["X-Fantasy-Source"] == source.group(1)
    assert '"X-Fantasy-Platform":' in webclient("requestConfig")
    platform = WEB_CLIENT_HEADERS["X-Fantasy-Platform"]
    assert f"X-Fantasy-Platform: {platform}" in API_DOC.read_text(encoding="utf-8")  # the value the capture saw


# --- the guarded UI captures ------------------------------------------------------------------------------------------

WRITE_CAPTURES = sorted(path.relative_to(REAL).as_posix() for path in REAL.glob("*/write_*.json"))
"""``{ffl,fba}/write_*.json``: the requests the guard aborted while the UI flows were driven by hand on 2026-10-06
(ROADMAP #14): a bench swap in each league, an NFL waiver claim with a drop, and two NBA free-agent adds (one-click from
the player list, and add-plus-drop through the roster-fix page). They are the reference for every envelope."""
CAPTURED_FLOWS = {
    "ffl/write_ROSTER_1.json": "set_lineup",
    "ffl/write_WAIVER_1.json": "claim_waiver",
    "fba/write_FREEAGENT_1.json": "add_drop (one-click Add, no memberId)",
    "fba/write_FREEAGENT_2.json": "add_drop (roster-fix add plus drop)",
    "fba/write_ROSTER_1.json": "set_lineup",
}
LATEST_AT_CAPTURE = {"ffl": 5, "fba": 1}
"""``status.latestScoringPeriod`` on 2026-10-06, the Tuesday of NFL week 5 and still NBA preseason day 1."""
_REBUILD_ITEM = {
    "LINEUP": lambda item: lineup_item(item["playerId"], item["fromLineupSlotId"], item["toLineupSlotId"]),
    "ADD": lambda item: add_item(item["playerId"], item["toTeamId"]),
    "DROP": lambda item: drop_item(item["playerId"], item["fromTeamId"]),
    "TRADE": lambda item: trade_item(item["playerId"], item["fromTeamId"], item["toTeamId"]),
}


def test_the_captures_are_the_five_the_guard_saved() -> None:
    assert WRITE_CAPTURES == sorted(CAPTURED_FLOWS)


@pytest.mark.parametrize("relative", WRITE_CAPTURES, ids=WRITE_CAPTURES)
def test_envelopes_rebuild_the_captured_web_client_requests(relative: str) -> None:
    """Each captured request, rebuilt from its own fields by this module, is byte for byte what the web client sent."""
    rebuild_capture(relative.split("/")[0], load(relative))


def flow_built(relative: str) -> tuple[Envelope, LeagueRow]:
    """The captured transaction built the way its flow would: ``set_lineup`` through ``lineup_envelope`` from
    ``LineupMove``s, a claim or an add through ``Envelope`` with the item builders, nothing copied from the capture
    but the player and slot ids a proposal would carry."""
    game = relative.split("/")[0]
    league = NFL if game == "ffl" else NBA
    body = load(relative)["body"]
    match relative:
        case "ffl/write_ROSTER_1.json":  # bench swap: Omarion Hampton (FLEX 23) and the bench back (20)
            envelope = lineup_envelope(
                team_id=1,
                member_id=SWID,
                scoring_period_id=5,
                latest_scoring_period=LATEST_AT_CAPTURE["ffl"],
                moves=swap(15847, 4685382, 23, 20),
            )
        case "fba/write_ROSTER_1.json":  # bench swap: Andrew Wiggins (UTIL 11) and Jaime Jaquez Jr. (bench 12)
            envelope = lineup_envelope(
                team_id=1,
                member_id=SWID,
                scoring_period_id=1,
                latest_scoring_period=LATEST_AT_CAPTURE["fba"],
                moves=swap(3059319, 4432848, 11, 12),
            )
        case "ffl/write_WAIVER_1.json":  # a claim with a drop in a league without FAAB: bidAmount is null
            envelope = Envelope(
                1, TransactionType.WAIVER, SWID, 5, items=(add_item(4723086, 1), drop_item(4360078, 1)), bid_amount=None
            )
        case "fba/write_FREEAGENT_1.json":  # the player list's one-click Add: no drop and no memberId
            envelope = Envelope(1, TransactionType.FREEAGENT, None, 1, items=(add_item(6589, 1),))
        case "fba/write_FREEAGENT_2.json":  # the roster-fix page: add plus drop, with memberId
            envelope = Envelope(1, TransactionType.FREEAGENT, SWID, 1, items=(add_item(6589, 1), drop_item(4397136, 1)))
        case _:
            raise AssertionError(f"no builder for {relative}: add one when a new capture lands")
    assert envelope.type.value == body["type"]
    return envelope, league


@pytest.mark.parametrize("relative", WRITE_CAPTURES, ids=WRITE_CAPTURES)
def test_each_flow_builds_its_captured_body(relative: str) -> None:
    """A transaction built from a proposal's own ids (not from the capture) is the captured body, key for key and in
    order; the placeholder ids (team 1, our SWID) are the scrubber's, the player and slot ids are ESPN's."""
    envelope, _league = flow_built(relative)
    captured = load(relative)["body"]
    assert envelope.body() == captured
    assert list(envelope.body()) == list(captured)
    assert json.dumps(envelope.body()) == json.dumps(captured)


def test_a_claim_without_a_bid_sends_bid_amount_null_and_a_cancel_sends_none() -> None:
    claim = Envelope(1, TransactionType.WAIVER, SWID, 5, items=(add_item(4723086, 1),)).body()
    assert "bidAmount" in claim and claim["bidAmount"] is None  # ffl/write_WAIVER_1.json
    assert list(claim)[-1] == "bidAmount"
    cancel = Envelope(
        1, TransactionType.WAIVER, SWID, 5, execution_type=ExecutionType.CANCEL, related_transaction_id="c"
    )
    assert "bidAmount" not in cancel.body()  # cancelWaiverClaim never sets one, so the serializer writes undefined


def test_member_id_is_optional_on_the_wire() -> None:
    one_click = load("fba/write_FREEAGENT_1.json")["body"]
    roster_fix = load("fba/write_FREEAGENT_2.json")["body"]
    assert "memberId" not in one_click and roster_fix["memberId"] == SWID
    assert all("memberId" in load(name)["body"] for name in WRITE_CAPTURES if name != "fba/write_FREEAGENT_1.json")
    assert "memberId" not in Envelope(1, TransactionType.FREEAGENT, None, 1, items=(add_item(6589, 1),)).body()


@pytest.mark.parametrize("relative", WRITE_CAPTURES, ids=WRITE_CAPTURES)
def test_the_request_matches_the_capture_but_for_platform_version(relative: str) -> None:
    """URL (minus the web client's build sha), method, and the headers the executor sends from the browser session
    (:data:`WEB_CLIENT_HEADERS` plus the transport's ``Content-Type`` and ``Accept``) are what the web client sent."""
    envelope, league = flow_built(relative)
    capture = load(relative)
    request = transaction_request(league, envelope)
    path, _, query = capture["url"].partition("?")
    assert (request.method, request.url) == (capture["method"], path)
    assert re.fullmatch(r"platformVersion=[0-9a-f]{40}", query)  # the client's own build; ours has none to send
    sent = {"Content-Type": "application/json", "Accept": "application/json", **dict(request.headers)}
    assert {name.lower(): value for name, value in sent.items()} == capture["headers"]
    check_write_request(request, league)


def rebuild_capture(game: str, capture: dict[str, Any]) -> None:
    body = capture["body"]
    items = tuple(_REBUILD_ITEM.get(item.get("type"), dict)(item) for item in body.get("items", ()))
    rebuilt = Envelope(
        team_id=body["teamId"],
        type=TransactionType(body["type"]),
        member_id=body.get("memberId"),
        scoring_period_id=body.get("scoringPeriodId"),
        items=items,
        execution_type=ExecutionType(body["executionType"]),
        bid_amount=body.get("bidAmount"),
        related_transaction_id=body.get("relatedTransactionId"),
        expiration_date=body.get("expirationDate"),
        comment=body.get("comment"),
    )
    assert json.dumps(rebuilt.body()) == json.dumps(body)  # same keys, same values, same order
    league = NFL if game == "ffl" else NBA
    assert capture["url"].split("?")[0] == transaction_request(league, rebuilt).url


def test_every_capture_is_in_the_fixture_index() -> None:
    """A capture the index does not know would be loaded by nothing but the tests above."""
    index = load("index.json")["files"]
    assert set(WRITE_CAPTURES) == {name for name, entry in index.items() if entry["kind"] == "write"}


# --- error codes ------------------------------------------------------------------------------------------------------


def test_every_code_the_web_client_knows_is_mapped() -> None:
    known = {code for game in ("ffl", "fba") for code in load(f"{game}/calendar.json")["errorCodes"]}
    assert len(known) == 42
    design = {"TRAN_LINEUP_LOCKED", "TRAN_ROSTER_SAME_SLOT", "FAILED_ROSTERLOCK", "AUTH_MISSING_CREDENTIALS"}
    assert known | design <= set(ERROR_CODES)
    for code, entry in ERROR_CODES.items():
        assert entry.code == code and entry.meaning and entry.known
    assert error_code("TRAN_LINEUP_LOCKED").kind is ErrorKind.LOCKED
    assert error_code("FAILED_ROSTERLOCK").kind is ErrorKind.LOCKED
    assert error_code("TRAN_ROSTER_LIMIT_EXCEEDED_ONE").kind is ErrorKind.ROSTER_LIMIT
    assert error_code("FAILED_NOTCLEAREDWAIVERS").kind is ErrorKind.UNAVAILABLE
    assert error_code("FAILED_MATCHUPACQUISITIONLIMIT").kind is ErrorKind.LIMIT
    assert error_code("AUTH_MISSING_CREDENTIALS").kind is ErrorKind.AUTH
    assert error_code("TRAN_LINEUP_LOCKED").describe() == (
        "TRAN_LINEUP_LOCKED: a player in the move is locked (his game has started)"
    )


def test_unknown_codes_still_explain_themselves() -> None:
    new = error_code("FAILED_SOMETHINGNEW")
    assert (new.kind, new.known) == (ErrorKind.UNKNOWN, False) and "does not know" in new.meaning
    assert error_code("AUTH_TOKEN_EXPIRED").kind is ErrorKind.AUTH


@pytest.mark.parametrize("game", ["ffl", "fba"])
def test_failed_statuses_on_record_are_known_codes(game: str) -> None:
    statuses = {
        record["status"]
        for name in ("mTransactions2.json", "mTransactions2_waiver_trade.json")
        for record in load(f"{game}/{name}")["transactions"]
        if record["status"].startswith("FAILED")
    }
    for status in statuses:
        assert error_code(status).known, status


def test_explain_reads_the_codes_in_an_answer() -> None:
    locked = WriteResponse.from_text(
        409, '{"messages":["Lineup is locked"],"details":[{"type":"TRAN_LINEUP_LOCKED","message":"Lineup is locked"}]}'
    )
    assert [code.kind for code in explain(locked)] == [ErrorKind.LOCKED]
    assert explain(WriteResponse(400, {"messages": ["Bad request"]})) == ()


@pytest.mark.parametrize(
    ("response", "allowed"),
    [
        (WriteResponse(400, {"messages": ["Bad request"]}), True),  # the envelope changed: what the UI is for
        (WriteResponse(404, None, "Not Found"), True),  # the endpoint moved
        (WriteResponse.from_text(409, '{"details":[{"type":"TRAN_LINEUP_LOCKED"}]}'), False),  # a league rule
        (WriteResponse.from_text(400, '{"details":[{"type":"FAILED_INVALID_FORMAT"}]}'), False),
        (WriteResponse.from_text(403, '{"details":[{"type":"FAILED_NOPERMISSION"}]}'), False),
        (WriteResponse(401, None, "AUTH_MISSING_CREDENTIALS"), False),  # signed out: the page is too
        (WriteResponse(429, None, "Too Many Requests"), False),  # the UI would be throttled the same way
        (WriteResponse(503, None, "Service Unavailable"), False),  # unknown: never followed
        (WriteResponse(200, {"id": "t-1"}), False),  # accepted: a UI walk would be a second write
    ],
)
def test_ui_may_follow_only_a_definite_rejection_the_ui_could_get_past(response: WriteResponse, allowed: bool) -> None:
    assert ui_may_follow(response) is allowed
    assert ui_may_follow(response) <= rejection_allows_ui(
        response
    )  # stricter than the executor's default, never looser
