"""The real-league capture tools in ``scripts/capture/`` (ROADMAP #14), on synthetic inputs: the write guard's
classification and routing (``guard.py``), the snapshot comparison that backs it up (``reads.py``), and the
web-client extractors (``webclient.py``). No browser, no network: requests, routes and bundles are fakes.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

CAPTURE_DIR = Path(__file__).resolve().parents[4] / "scripts" / "capture"


def load_capture(name: str) -> ModuleType:
    """``scripts/capture/<name>.py`` (the capture scripts are not a package), loaded once as ``capture_<name>``."""
    module_name = f"capture_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, CAPTURE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


guard = load_capture("guard")
reads = load_capture("reads")
webclient = load_capture("webclient")

LEAGUE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2026/segments/0/leagues/1"
WRITE = "https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl/seasons/2026/segments/0/leagues/1/transactions/"


# --- guard: what is aborted -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "url"),
    [
        ("POST", WRITE),
        ("GET", "https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl/seasons/2026"),  # the write host, any method
        ("PUT", "https://lm-api-writes.fantasy.espn.com./apis/v3/x"),  # trailing dot
        (
            "POST",
            "https://fantasy.espn.com/apis/v3/games/ffl/seasons/2026/segments/0/leagues/1/teams/1/pendingTransactions",
        ),
        ("PUT", "https://fantasy.espn.com/apis/v3/games/ffl/seasons/2026/segments/0/leagues/1/teams/1"),
        ("DELETE", f"{LEAGUE}/teams/1"),
        ("PATCH", "https://www.espn.com/x"),
        ("POST", "https://ESPN.com/x"),
        ("post", "https://site.api.espn.com/apis/x"),
        ("POST", "https://sw88.espn.com/b/ss/x"),  # analytics
        ("GET", f"{LEAGUE}/transactions/"),  # the marker, even on the read host
        ("GET", "https://example.com/transactions"),  # the marker, any host
    ],
)
def test_guard_aborts(method: str, url: str) -> None:
    assert guard.block_reason(method, url) is not None


@pytest.mark.parametrize(
    ("method", "url"),
    [
        ("GET", f"{LEAGUE}?view=mRoster"),
        ("GET", f"{LEAGUE}?view=mPendingTransactions&view=mTransactions2"),  # the app's own reads: capital T
        ("GET", "https://fantasy.espn.com/football/team?leagueId=1&teamId=1"),
        ("POST", "https://registerdisney.go.com/jgc/v8/client/ESPN-ONESITE.WEB-PROD/guest/login"),  # Disney sign-in
        ("POST", "https://espn.com.example.net/x"),  # not an ESPN host
        ("POST", "https://notespn.com/x"),
    ],
)
def test_guard_lets_through(method: str, url: str) -> None:
    assert guard.block_reason(method, url) is None


@pytest.mark.parametrize(
    ("method", "url", "aborted"),
    [
        ("POST", WRITE, True),
        ("GET", "https://lm-api-writes.fantasy.espn.com/x", True),
        ("POST", "https://fantasy.espn.com/apis/v3/x", True),
        ("PUT", "https://lm-api-communication.fantasy.espn.com/x", True),
        ("GET", f"{LEAGUE}/transactions/", True),
        ("POST", "https://www.espn.com/x", False),
        ("POST", "https://registerdisney.go.com/x", False),
        ("GET", f"{LEAGUE}?view=mRoster", False),
    ],
)
def test_sign_in_mode_still_aborts_every_league_write(method: str, url: str, aborted: bool) -> None:
    assert (guard.block_reason(method, url, sign_in=True) is not None) is aborted


@pytest.mark.parametrize(
    ("method", "url", "expected"),
    [
        ("GET", f"{LEAGUE}?view=mRoster", True),
        ("GET", "https://fantasy.espn.com/apis/v3/games/ffl/seasons/2026", True),
        ("GET", "https://site.web.api.espn.com/apis/site/v2/content", True),
        ("GET", "https://fantasy.espn.com/football/team", False),
        ("POST", f"{LEAGUE}?view=mRoster", False),
        ("GET", "https://api.espn.com.example.net/apis/x", False),
        ("GET", "https://cdn1.espn.net/kona/main.js", False),
    ],
)
def test_api_reads(method: str, url: str, expected: bool) -> None:
    assert guard.is_api_read(method, url) is expected


# --- guard: routing and records ---------------------------------------------------------------------------------------


class FakeRequest:
    def __init__(self, method: str, url: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> None:
        self.method = method
        self.url = url
        self.post_data_buffer = body
        self.resource_type = "fetch"
        self.headers = headers or {}


class FakeRoute:
    def __init__(self) -> None:
        self.outcome: tuple[str, str | None] | None = None

    def abort(self, error_code: str | None = None) -> None:
        self.outcome = ("abort", error_code)

    def continue_(self) -> None:
        self.outcome = ("continue", None)


class FakeResponse:
    def __init__(self, request: FakeRequest, status: int) -> None:
        self.request = request
        self.status = status


class Unreadable:
    """A request whose fields cannot be read (its frame is gone)."""

    url = WRITE

    @property
    def method(self) -> str:
        raise RuntimeError("detached")


def route(write_guard: Any, request: Any) -> tuple[str, str | None] | None:
    fake = FakeRoute()
    write_guard._on_route(fake, request)
    return fake.outcome


def test_a_write_is_aborted_and_recorded_without_secrets() -> None:
    write_guard = guard.WriteGuard(pace_reads=False)
    body = json.dumps({"type": "FREEAGENT", "teamId": 1}).encode()
    headers = {"content-type": "application/json", "cookie": "espn_s2=secret", "x-fantasy-source": "kona"}
    assert route(write_guard, FakeRequest("POST", WRITE, body, headers)) == ("abort", guard.ABORT_ERROR)
    (record,) = write_guard.captures
    assert record.reason == "write host" and record.body == {"type": "FREEAGENT", "teamId": 1}
    assert record.headers == {"content-type": "application/json", "x-fantasy-source": "kona"}
    assert not record.probe
    write_guard.check()


def test_a_request_the_guard_cannot_classify_is_aborted() -> None:
    write_guard = guard.WriteGuard(pace_reads=False)
    assert route(write_guard, Unreadable()) == ("abort", guard.ABORT_ERROR)
    with pytest.raises(guard.GuardError):
        write_guard.check()


def test_api_reads_are_paced_and_recorded() -> None:
    clock = [100.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += seconds

    write_guard = guard.WriteGuard(sleep=sleep, monotonic=lambda: clock[0])
    first = FakeRequest("GET", f"{LEAGUE}?view=mRoster", headers={"x-fantasy-filter": '{"players":{}}'})
    assert route(write_guard, first) == ("continue", None)
    clock[0] += 0.25
    assert route(write_guard, FakeRequest("GET", f"{LEAGUE}?view=mTeam")) == ("continue", None)
    assert slept == [pytest.approx(0.75)]
    assert [read.filter for read in write_guard.reads] == [{"players": {}}, None]
    write_guard._on_response(FakeResponse(first, 200))
    assert write_guard.reads[0].status == 200


def test_a_response_to_an_aborted_request_is_a_leak() -> None:
    write_guard = guard.WriteGuard(pace_reads=False)
    write_guard._on_response(FakeResponse(FakeRequest("OPTIONS", WRITE), 204))  # a preflight is no leak
    write_guard.check()
    write_guard._on_response(FakeResponse(FakeRequest("POST", WRITE), 200))
    with pytest.raises(guard.GuardLeakError):
        write_guard.check()


def test_sign_in_mode_records_what_it_lets_through() -> None:
    write_guard = guard.WriteGuard(pace_reads=False, sign_in=True)
    assert route(write_guard, FakeRequest("POST", "https://www.espn.com/x")) == ("continue", None)
    assert route(write_guard, FakeRequest("POST", "https://fantasy.espn.com/apis/v3/x")) == ("abort", guard.ABORT_ERROR)
    assert route(write_guard, FakeRequest("POST", "https://registerdisney.go.com/x")) == ("continue", None)
    assert write_guard.passed == ["POST https://www.espn.com/x"]
    write_guard._on_response(FakeResponse(FakeRequest("POST", "https://www.espn.com/x"), 200))
    write_guard.check()  # allowed in sign-in mode, so no leak


# --- snapshots: what a leaked write would leave -----------------------------------------------------------------------

OURS = 3


def record(kind: str, status: str, team: int, teams: list[int] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "type": kind,
        "status": status,
        "execution_type": extra.get("execution_type", "EXECUTE"),
        "team_id": team,
        "team_ids": sorted(teams or [team]),
        "scoring_period": 5,
        "related": extra.get("related"),
    }


BASE: dict[str, Any] = {
    "version": reads.SNAPSHOT_VERSION,
    "taken_at": "2026-10-05T19:00:00+00:00",
    "team_id": OURS,
    "periods": [5, 6],
    "rosters": {
        "5": {"3": [[101, 0], [102, 20]], "4": [[201, 0]]},
        "6": {"3": [[101, 0], [102, 20]], "4": [[201, 0]]},
    },
    "team": {"transactionCounter": {"acquisitions": 1, "drops": 1, "moveToActive": 0}, "tradeBlock": {}},
    "transactions": {
        "lineup": record("ROSTER", "EXECUTED", OURS),
        "offer-in": record("TRADE_PROPOSAL", "PENDING", 4, [3, 4]),
        "offer-others": record("TRADE_PROPOSAL", "PENDING", 4, [4, 5]),
        "claim": record("WAIVER", "PENDING", OURS),
    },
    "pending": {
        "offer-in": record("TRADE_PROPOSAL", "PENDING", 4, [3, 4]),
        "offer-others": record("TRADE_PROPOSAL", "PENDING", 4, [4, 5]),
        "claim": record("WAIVER", "PENDING", OURS),
    },
}


def compare(snap: dict[str, Any]) -> tuple[list[str], list[str]]:
    return reads.compare_snapshots(BASE, snap, OURS)


def test_identical_snapshots_compare_clean() -> None:
    assert compare(copy.deepcopy(BASE)) == ([], [])


def test_a_lineup_move_for_a_later_period_is_a_problem() -> None:
    snap = copy.deepcopy(BASE)
    snap["rosters"]["6"]["3"] = [[101, 20], [102, 0]]  # a FUTURE_ROSTER swap: the current period is unchanged
    problems, notes = compare(snap)
    assert len(problems) == 1 and problems[0].startswith("period 6, team 3") and not notes


def test_other_managers_moves_are_notes() -> None:
    snap = copy.deepcopy(BASE)
    snap["rosters"]["5"]["4"] = [[202, 0]]
    snap["transactions"]["theirs"] = record("FREEAGENT", "EXECUTED", 4)
    problems, notes = compare(snap)
    assert problems == [] and len(notes) == 2


def test_an_answered_offer_is_a_problem_even_though_its_id_existed() -> None:
    snap = copy.deepcopy(BASE)
    snap["transactions"]["offer-in"] = record("TRADE_PROPOSAL", "DECLINED", 4, [3, 4])
    snap["transactions"]["decline"] = record("TRADE_DECLINE", "EXECUTED", OURS, [3, 4], related="offer-in")
    del snap["pending"]["offer-in"]
    problems, notes = compare(snap)
    assert not notes
    assert any(line.startswith("transaction changed (status)") and "offer-in" in line for line in problems)
    assert any(line.startswith("new transaction TRADE_DECLINE decline") for line in problems)
    assert any(line.startswith("pending gone (now DECLINED)") and "offer-in" in line for line in problems)


def test_a_cancel_of_an_existing_claim_is_a_problem() -> None:
    snap = copy.deepcopy(BASE)
    snap["transactions"]["cancel"] = record("WAIVER", "EXECUTED", OURS, related="claim", execution_type="CANCEL")
    del snap["pending"]["claim"]
    snap["transactions"]["claim"] = record("WAIVER", "CANCELED", OURS)
    problems, _ = compare(snap)
    assert any("new transaction WAIVER/CANCEL cancel" in line for line in problems)
    assert any(line.startswith("pending gone (now CANCELED)") for line in problems)


def test_a_cancel_record_alone_is_a_problem() -> None:
    """An offer's cancel (or expiry) leaves the offer PENDING and adds a CANCEL record: the record gives it away."""
    snap = copy.deepcopy(BASE)
    snap["transactions"]["cancel"] = record(
        "TRADE_PROPOSAL", "CANCELED", 4, [3, 4], related="offer-in", execution_type="CANCEL"
    )
    problems, notes = compare(snap)
    assert len(problems) == 1 and "TRADE_PROPOSAL/CANCEL cancel" in problems[0] and not notes


def test_a_pending_item_of_other_teams_that_vanished_is_a_note() -> None:
    snap = copy.deepcopy(BASE)
    del snap["pending"]["offer-others"]
    del snap["transactions"]["offer-others"]
    problems, notes = compare(snap)
    assert problems == [] and len(notes) == 2
    assert any(line.startswith("pending gone (no longer listed)") for line in notes)


def test_counter_and_trade_block_changes_are_problems() -> None:
    snap = copy.deepcopy(BASE)
    snap["team"] = {"transactionCounter": {"acquisitions": 2, "drops": 1, "moveToActive": 0}, "tradeBlock": {"x": 1}}
    problems, _ = compare(snap)
    assert [line.split(" changed")[0] for line in problems] == ["our transactionCounter", "our tradeBlock"]


def test_a_new_pending_move_of_ours_is_a_problem() -> None:
    snap = copy.deepcopy(BASE)
    snap["pending"]["new-claim"] = record("WAIVER", "PENDING", OURS)
    snap["transactions"]["new-claim"] = record("WAIVER", "PENDING", OURS)
    problems, _ = compare(snap)
    assert sorted(line.split(" ")[1] for line in problems) == ["pending", "transaction"]


def test_snapshots_from_the_old_format_are_refused() -> None:
    old = {"scoring_period": 4, "rosters": {}, "pending": []}
    with pytest.raises(reads.SnapshotError):
        reads.compare_snapshots(old, BASE, OURS)


@pytest.mark.parametrize(
    ("current", "expected"),
    [(1, [1, 2, 3, 4, 5, 6]), (4, [4, 5, 6]), (13, [13]), (20, [20])],
)
def test_periods_left_in_matchup(current: int, expected: list[int]) -> None:
    matchups = {1: [1, 2, 3, 4, 5, 6], 2: list(range(7, 14))}
    assert reads.periods_left_in_matchup(matchups, current) == expected


# --- web client: calendars --------------------------------------------------------------------------------------------

DAY = 86_400_000


def scoring_periods(ends: list[int], *, post_from: int) -> str:
    """A minified ``scoringPeriods`` constant: period ``i`` runs from ``ends[i - 1]`` to ``ends[i]`` (epoch ms)."""
    items = [
        f"{{endDate:{end},id:{index},postSeason:{'!0' if index >= post_from else '!1'},preSeason:!1,"
        f"startDate:{ends[index - 1] if index else 0}}}"
        for index, end in enumerate(ends)
    ]
    return "e.exports=[" + ",".join(items) + "]"


def segments(periods: list[tuple[int, int, int]]) -> str:
    inner = ",".join(f"{{id:{i},scoringPeriodStart:{s},scoringPeriodEnd:{e}}}" for i, s, e in periods)
    return f"e.exports=[{{id:0,periodTypes:[{{id:2,periods:[{inner}]}},{{id:0,seasonLong:!0,periods:[]}}]}}]"


def day_ends(count: int) -> list[int]:
    return [DAY * (10 + day) for day in range(count + 1)]


def bundle_with_calendars() -> str:
    first = scoring_periods(day_ends(4), post_from=4)  # last regular period 3
    second = scoring_periods(day_ends(6), post_from=6)  # last regular period 5
    return ",".join(
        [
            f"function(e,t){{{first}}}",
            f"function(e,t){{{segments([(1, 1, 3)])}}}",
            f"function(e,t){{{second}}}",
            f"function(e,t){{{segments([(1, 1, 2), (2, 3, 5)])}}}",
        ]
    )


def test_js_literal() -> None:
    assert webclient._js_literal('{a:1,b:"x",c:!0,d:!1,e:[1e3,2.5e2]}') == {
        "a": 1,
        "b": "x",
        "c": True,
        "d": False,
        "e": [1000, 250],
    }


def test_extract_and_match_calendars() -> None:
    calendars = webclient.extract_calendars(bundle_with_calendars())
    assert [calendar.last_regular_period for calendar in calendars] == [3, 5]
    games = {"5": [{"id": 9, "scoringPeriodId": 5, "date": DAY * 15 - 1}], "1": [{"id": 8, "scoringPeriodId": 1}]}
    games["1"][0]["date"] = DAY * 11 - 1  # inside day 1: the day before its end
    schedule = {"settings": {"proTeams": [{"proGamesByScoringPeriod": games}]}}
    matched = webclient.match_calendar(calendars, schedule, weekly_game=False)
    assert matched is calendars[1]
    assert matched.matchup_days({"1": [1], "2": [2]}, 2) == {1: [1, 2], 2: [3, 4, 5]}
    games["5"][0]["date"] = DAY * 16  # after the last period: no calendar fits
    with pytest.raises(webclient.CalendarError):
        webclient.match_calendar(calendars, schedule, weekly_game=False)


def test_calendar_round_trips_and_period_one_starts_one_period_before_its_end() -> None:
    calendar = webclient.extract_calendars(bundle_with_calendars())[1]
    assert webclient.Calendar.from_json(calendar.as_json()) == calendar
    assert calendar.window(1, weekly_game=False) == (DAY * 10, DAY * 11)
    assert calendar.window(1, weekly_game=True) == (DAY * 4, DAY * 11)


# --- web client: stat settings and transaction code -------------------------------------------------------------------


def test_scan_skips_strings_and_templates() -> None:
    text = "x{a:\"}\",b:`${{c:1}}}`,d:['{']}y"
    assert text[webclient._scan(text, 1) :] == "y"
    assert webclient._scan("const a={b:1};c", 0, statement=True) == len("const a={b:1};")


def test_extract_stat_settings() -> None:
    bundle = (
        'function(e,t){e.exports={showSeasonProjections:true,sources:[{description:"Real",id:0}],'
        'splitTypes:[{description:"Season",gameSplit:false,id:0},{description:"Game",gameSplit:true,id:1}],'
        'stats:[{abbrev:"PA",apiIdentifier:"passing.passingAttempts",id:0}]}}'
    )
    assert webclient.extract_stat_settings(bundle) == [
        {
            "sources": [{"description": "Real", "id": 0}],
            "splitTypes": [
                {"description": "Season", "gameSplit": False, "id": 0},
                {"description": "Game", "gameSplit": True, "id": 1},
            ],
            "firstStat": "passing.passingAttempts",
        }
    ]


TYPES = (
    'function(e,t,n){"use strict";const r="drop";t["j"]=r;const o="ROSTER";t["v"]=o;'
    'const a="TRADE_PROPOSAL";t["G"]=a;const i="WAIVER";t["N"]=i;const s="CANCEL";t["n"]=s}'
)
MODEL = (
    'function(e,t,n){"use strict";var r=n(44);var o=n(10);var a=n.n(o);'
    "const i=({playerId:n,type:a})=>{let i={playerId:n};a&&(i.type=a);return i};"
    'const l=(e,t)=>e.map(e=>i({playerId:e.id,type:"ADD",toTeamId:t}));t["a"]=l;'
    "class b{constructor({type:d}){this.items=[];this.type=d}pushItems(e){this.items.push(...e)}"
    'get(){const e={type:this.type};if(this.type===r["N"]){e.bidAmount=this.bidAmount}return e}}t["h"]=b}'
)
SERVICE = (
    'function(e,t,n){"use strict";var r=n(1959);var o=n(44);var a=n(14);'
    'class M{createTransaction({type:f}){return new r["h"]({type:f})}'
    'static cancelWaiverClaim(e){return this.prototype.createTransaction({type:o["N"],executionType:o["n"]})}'
    'static dropPlayers(e){return e[o["j"]]}}t["a"]=M}'
)
API = (
    'function(e,t,n){"use strict";(function(e){'
    'const Mt=(e,t)=>"https://lm-api-writes.".concat(e,".").concat(t,".com");'
    'const ft="HOST_TYPE_FANTASY_WRITE_API";'
    'const zt={headers:{Accept:"application/json","X-Fantasy-Source":"kona"},timeout:5e4};'
    'const Yt=e=>{const i={withCredentials:true};return Object.assign({},zt,i,{headers:{"X-Fantasy-Platform":"w"}})};'
    "function Zt(e){return en.apply(this,arguments)}"
    "function en(){en=nt(function*({config:e={},headers:t={},params:n={},path:o}){"
    't["Content-Type"]="application/json";const l=yield Object(r["post"])({config:Yt({headers:t}),data:n,'
    "url:St({options:{hostType:ft},path:o})});return l});return en.apply(this,arguments)}"
    "function co(){co=nt(function*(e,t,n,r){return yield Zt({config:e,params:JSON.stringify(r),"
    'path:"games/".concat(e.uri_nextgen_api,"/seasons/").concat(t,"/segments/0/leagues/").concat(n,"/transactions/")'
    "})});return co.apply(this,arguments)}"
    "function po(){po=nt(function*(e,t,n,r,o){return yield Zt({config:e,params:JSON.stringify(o),"
    'path:"games/".concat(n,"/teams/").concat(r,"/pendingTransactions")})});return po.apply(this,arguments)}'
    "function vo(){vo=nt(function*(e,t,n,r,o,a){return yield Zt({config:e,params:JSON.stringify({bidAmount:a}),"
    'path:"games/".concat(r,"/pendingTransactions/").concat(o)})});return vo.apply(this,arguments)}'
    "}).call(this,n(1))}"
)


def test_extract_transaction_code() -> None:
    code = webclient.extract_transaction_code("[" + ",".join([TYPES, MODEL, SERVICE, API]) + "]")
    excerpts = code["excerpts"]
    assert (excerpts["types"], excerpts["model"], excerpts["service"]) == (TYPES, MODEL, SERVICE)
    assert code["typeNames"] == {"j": "drop", "v": "ROSTER", "G": "TRADE_PROPOSAL", "N": "WAIVER", "n": "CANCEL"}
    assert code["constantsAlias"] == {"model": "r", "service": "o"}
    assert (
        excerpts["saveTransaction"].startswith("function co(){") and '"/transactions/"' in excerpts["saveTransaction"]
    )
    assert excerpts["reorderPendingTransactions"].startswith("function po(){")
    assert excerpts["updatePendingBid"].startswith("function vo(){")
    assert excerpts["post"].startswith("function Zt(e){return en.apply(this,arguments)}function en(){")
    assert excerpts["post"].endswith("return en.apply(this,arguments)}")
    assert excerpts["writeHost"].startswith("const Mt=") and excerpts["writeHost"].endswith('".com");')
    assert excerpts["requestConfig"].startswith("const Yt=e=>{") and excerpts["requestConfig"].endswith("})};")
    resolved = webclient.resolve_constants(excerpts["model"], "r", code["typeNames"])
    assert 'if(this.type==="WAIVER")' in resolved
    service = webclient.resolve_constants(excerpts["service"], "o", code["typeNames"])
    assert 'type:"WAIVER",executionType:"CANCEL"' in service and 'new r["h"]' in service  # other imports untouched


def test_a_missing_piece_of_transaction_code_is_an_error() -> None:
    with pytest.raises(webclient.WebClientError, match="transaction model"):
        webclient.extract_transaction_code("[" + ",".join([TYPES, SERVICE, API]) + "]")
