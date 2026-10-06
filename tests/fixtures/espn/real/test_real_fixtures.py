"""The real-league fixtures (ROADMAP #14) load through the model that serves each view, stay scrubbed, and back the
facts ``docs/espn-api.md`` states from them.

Every JSON file in this directory is a scrubbed, trimmed capture of one of the two configured leagues, written by
``scripts/capture/capture.py fixtures``; ``index.json`` names its game, view and model kind. ``webclient.json`` holds
verbatim excerpts of ESPN's web client (the code that builds and sends every write, and each game's stat split labels),
which the write-payload tests read. The last tests exercise the scrubber that wrote the fixtures
(``scripts/capture/scrub.py``) on a synthetic league. Offline: no network.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fm.espn.client import stat_entry_id
from fm.espn.models import (
    LeagueEnvelope,
    MatchupsView,
    PlayersView,
    ProSchedule,
    RostersView,
    TeamsView,
    Transaction,
    TransactionsView,
)
from fm.espn.settings import LeagueSettings, LockType, ScoringKind, parse_league_settings

HERE = Path(__file__).resolve().parent
CAPTURE_DIR = HERE.parents[3] / "scripts" / "capture"
INDEX: dict[str, Any] = json.loads((HERE / "index.json").read_text(encoding="utf-8"))
FILES: dict[str, dict[str, Any]] = INDEX["files"]
LEAGUE_PLACEHOLDERS = {"ffl": 1010101, "fba": 2020202}
SEASONS = {"ffl": 2026, "fba": 2027}
PLACEHOLDER_SWID = re.compile(r"\{00000000-0000-0000-0000-\d{12}\}")
BRACED_GUID = re.compile(r"\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}")


def load(relative: str) -> Any:
    return json.loads((HERE / relative).read_text(encoding="utf-8"))


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


def settings(game: str) -> LeagueSettings:
    return parse_league_settings(load(f"{game}/mSettings.json"), game=game)


def our_team(game: str) -> int:
    """The scrubber gives our team the smallest id of the league's id set."""
    return min(team.id for team in TeamsView.model_validate(load(f"{game}/mTeam+mStandings.json")).teams)


# --- every fixture loads ----------------------------------------------------------------------------------------------


def _calendar(data: Any, game: str) -> None:
    assert data["game"] == game and data["scoringPeriods"] and data["periodTypes"]


def _probes(data: Any, game: str) -> None:
    assert {"filter_slot_ids", "top_scoring_period_zero", "stat_entries"} <= set(data)


def _game_state(data: Any, game: str) -> None:
    assert isinstance(data["currentScoringPeriod"]["id"], int)


def _write(data: Any, game: str) -> None:
    body = data["body"]
    assert data["method"] == "POST" and body["isLeagueManager"] is False
    assert {"teamId", "type", "scoringPeriodId", "executionType"} <= set(body)
    # The player list's one-click Add sends no memberId; every other captured flow does.
    assert body.get("memberId", "{00000000-0000-0000-0000-000000000001}").startswith("{00000000-")
    path, _, query = data["url"].partition("?")
    assert path == (
        f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/{game}/seasons/{SEASONS[game]}/segments/0/leagues/"
        f"{LEAGUE_PLACEHOLDERS[game]}/transactions/"
    )
    assert re.fullmatch(r"platformVersion=[0-9a-f]{40}", query)  # the web client's build sha
    assert data["headers"] == {
        "accept": "application/json",
        "content-type": "application/json",
        "x-fantasy-platform": "espn-fantasy-web",
        "x-fantasy-source": "kona",
    }


DERIVED = "derived, not observed"


def _write_derived(data: Any, game: str) -> None:
    """A request built from ESPN's code (``capture.py derive-trade``): it says so, and has a capture's shape."""
    assert data["provenance"].startswith(DERIVED)
    _write(data, game)
    assert data["sources"] and data["unknowns"] and data["inputs"]["players"]


def _trade_review(data: Any, game: str) -> None:
    assert data["game"] == game and data["status"] in ("review reached", "review not reached", "login required")
    assert data["canary"] and all(record["reason"].startswith("trade lock") for record in data["canary"])


def _webclient(data: Any, game: None) -> None:
    code = data["transactionCode"]
    assert data["webClientBuild"] and re.fullmatch(r"[0-9a-f]{64}", data["bundleSha256"])
    assert set(code["excerpts"]) >= {"types", "model", "service", "saveTransaction", "post", "writeHost"}
    assert {"ROSTER", "FUTURE_ROSTER", "FREEAGENT", "WAIVER", "TRADE_PROPOSAL", "CANCEL"} <= set(
        code["typeNames"].values()
    )
    assert all(entry["splitTypes"] for entry in data["statSettings"])


LOADERS: dict[str, Callable[[Any, Any], object]] = {
    "settings": lambda data, game: parse_league_settings(data, game=game),
    "teams": lambda data, game: TeamsView.model_validate(data),
    "rosters": lambda data, game: RostersView.model_validate(data),
    "matchups": lambda data, game: MatchupsView.model_validate(data),
    "players": lambda data, game: PlayersView.model_validate(data),
    "transactions": lambda data, game: TransactionsView.model_validate(data),
    "pro_schedule": lambda data, game: ProSchedule.model_validate(data),
    "envelope": lambda data, game: LeagueEnvelope.model_validate(data),
    "game_state": _game_state,
    "calendar": _calendar,
    "probes": _probes,
    "write": _write,
    "write_derived": _write_derived,
    "trade_review": _trade_review,
    "webclient": _webclient,
}


def test_index_lists_every_fixture() -> None:
    on_disk = {path.relative_to(HERE).as_posix() for path in HERE.rglob("*.json")} - {"index.json"}
    assert on_disk == set(FILES)
    assert {entry["kind"] for entry in FILES.values()} <= set(LOADERS)


@pytest.mark.parametrize("relative", sorted(FILES))
def test_fixture_loads(relative: str) -> None:
    entry = FILES[relative]
    LOADERS[entry["kind"]](load(relative), entry["game"])


@pytest.mark.parametrize("relative", sorted(FILES))
def test_fixture_is_scrubbed(relative: str) -> None:
    text = (HERE / relative).read_text(encoding="utf-8")
    assert all(PLACEHOLDER_SWID.fullmatch(guid) for guid in BRACED_GUID.findall(text))
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("id"), int) and "gameId" in data:
        assert data["id"] == LEAGUE_PLACEHOLDERS[FILES[relative]["game"]]
    for member in data.get("members", []) if isinstance(data, dict) else []:
        assert re.fullmatch(r"Manager \d+", member["displayName"])
    for team in data.get("teams", []) if isinstance(data, dict) else []:
        if "name" in team:
            assert team["name"] == f"Team {team['id']}" and team["abbrev"] == f"T{team['id']}"


# --- league settings (fm.espn.settings) -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("game", "roster_lock"),
    [("ffl", LockType.INDIVIDUAL_GAME), ("fba", LockType.FIRSTGAME_SCORINGPERIOD)],
)
def test_lock_type_keys(game: str, roster_lock: LockType) -> None:
    parsed = settings(game)
    assert parsed.lineup_lock_type is LockType.INDIVIDUAL_GAME
    assert parsed.roster_lock_type is roster_lock and parsed.roster_lock_type_raw == roster_lock.value
    # The literal: the #4 guess no league carries is no longer a LockType member (it parses as UNKNOWN).
    assert "FIRST_GAME_OF_WEEK" not in json.dumps(load(f"{game}/mSettings.json"))


@pytest.mark.parametrize("game", ["ffl", "fba"])
def test_settings_shapes(game: str) -> None:
    raw = load(f"{game}/mSettings.json")["settings"]
    parsed = settings(game)
    assert raw["rosterSettings"]["lineupSlotStatLimits"] == {}
    assert parsed.slot_stat_limits == {}
    assert parsed.acquisition.waiver_hours == raw["acquisitionSettings"]["waiverHours"] == 24
    assert parsed.scoring_kind is ScoringKind.POINTS
    assert not parsed.acquisition.uses_faab and parsed.acquisition.budget is None


def test_nba_acquisition_limit_is_a_daily_rate() -> None:
    acquisition = settings("fba").acquisition
    assert acquisition.matchup_limit_per_scoring_period
    assert acquisition.matchup_limit_rate == pytest.approx(3 / 7)
    assert acquisition.matchup_limit_for(7) == 3
    assert acquisition.matchup_limit_for(14) == 6


# --- calendars: matchup periods to scoring periods --------------------------------------------------------------------


def matchup_days(game: str) -> dict[int, list[int]]:
    """``scheduleSettings.matchupPeriods`` resolved by ``webclient.Calendar.matchup_days`` (the code the capture's
    snapshot uses) through the web client's calendar for the league's period type."""
    schedule = load(f"{game}/mSettings.json")["settings"]["scheduleSettings"]
    calendar = load_capture("webclient").Calendar.from_json(load(f"{game}/calendar.json"))
    return calendar.matchup_days(schedule["matchupPeriods"], schedule["periodTypeId"])


def test_nba_matchups_map_to_weeks_of_days() -> None:
    days = matchup_days("fba")
    final = settings("fba").final_scoring_period
    assert final is not None
    assert days[1] == list(range(1, 7))  # Tue Oct 20 - Sun Oct 25: a six-day opening week
    assert days[2] == list(range(7, 14))
    assert days[18] == list(range(119, 133))  # the All-Star break folds two weeks into one matchup
    assert days[21][-1] == final == 153
    assert [day for matchup in sorted(days) for day in days[matchup]] == list(range(1, final + 1))
    assert {len(days[m]) for m in days if m not in (1, 18)} == {7}


def test_nba_day_one_is_opening_night() -> None:
    calendar = load("fba/calendar.json")
    day_one = next(p for p in calendar["scoringPeriods"] if p["id"] == 1)
    end = datetime.fromtimestamp(day_one["endDate"] / 1000, UTC)
    assert end - timedelta(days=1) == datetime(2026, 10, 20, 7, tzinfo=UTC)  # 3 a.m. ET, ESPN's day boundary
    schedule = ProSchedule.model_validate(load("fba/proTeamSchedules_wl.json"))
    first = schedule.first_game(1)
    assert first is not None and end - timedelta(days=1) <= first.date < end


def test_nfl_playoff_matchups_span_two_weeks() -> None:
    days = matchup_days("ffl")
    assert days[1] == [1] and days[13] == [13]
    assert days[14] == [14, 15] and days[15] == [16, 17]
    assert days[15][-1] == settings("ffl").final_scoring_period


# --- pending offers and transactions ----------------------------------------------------------------------------------


def test_pending_view_list_key() -> None:
    assert load("ffl/mPendingTransactions.json")["pendingTransactions"] == []
    assert "pendingTransactions" not in load("fba/mPendingTransactions.json")
    for game in ("ffl", "fba"):
        assert TransactionsView.model_validate(load(f"{game}/mPendingTransactions.json")).transactions == ()


def test_expired_trade_offers_still_say_pending() -> None:
    """ESPN leaves an expired offer at status PENDING and records the expiry as a separate CANCEL transaction."""
    view = TransactionsView.model_validate(load("fba/mTransactions2_waiver_trade.json"))
    captured = datetime.fromisoformat(FILES["fba/mTransactions2_waiver_trade.json"]["fetched_at"])
    offers = [t for t in view.transactions if t.type == "TRADE_PROPOSAL" and t.status == "PENDING"]
    cancels = {t.related_transaction_id: t for t in view.transactions if t.execution_type == "CANCEL"}
    assert offers
    for offer in offers:
        assert offer.expiration_date is not None and offer.proposed_date is not None
        lifetime = (offer.expiration_date - offer.proposed_date).total_seconds()
        assert abs(lifetime - timedelta(hours=48).total_seconds()) < 1
        assert offer.expiration_date < captured
        cancel = cancels[offer.id]
        assert cancel.type == "TRADE_PROPOSAL" and cancel.status == "CANCELED"
        assert cancel.player_ids == offer.player_ids


def open_offers(view: TransactionsView, now: datetime) -> list[Transaction]:
    """docs/espn-api.md section 1 #2: an offer is open while it is PENDING, its expirationDate is in the future and no
    CANCEL record points at it. ``status``/``isPending`` alone are not enough."""
    cancelled = {t.related_transaction_id for t in view.transactions if t.execution_type == "CANCEL"}
    return [
        t
        for t in view.of_type("TRADE_PROPOSAL")
        if t.status == "PENDING"
        and t.execution_type != "CANCEL"
        and t.expiration_date is not None
        and t.expiration_date > now
        and t.id not in cancelled
    ]


def test_no_offer_was_open_although_six_records_say_pending() -> None:
    view = TransactionsView.model_validate(load("fba/mTransactions2_waiver_trade.json"))
    captured = datetime.fromisoformat(FILES["fba/mTransactions2_waiver_trade.json"]["fetched_at"])
    flagged = [t for t in view.transactions if t.is_pending]
    assert len(flagged) == 6  # the three expired offers and their three CANCEL records all carry isPending: true
    assert {(t.status, t.execution_type) for t in flagged} == {("PENDING", "EXECUTE"), ("CANCELED", "CANCEL")}
    assert open_offers(view, captured) == []
    # The models apply the same rule: the expired offers are still PENDING, but nothing is open.
    assert len(view.pending()) == 3 and all(t.type == "TRADE_PROPOSAL" for t in view.pending())
    assert view.open(captured) == ()
    before_expiry = min(t.expiration_date for t in view.pending() if t.expiration_date is not None) - timedelta(hours=1)
    assert view.open(before_expiry) == ()  # each one is also closed by its CANCEL record


def test_recorded_lineup_moves_list_both_sides_of_a_swap() -> None:
    view = TransactionsView.model_validate(load("ffl/mTransactions2.json"))
    swaps = [t for t in view.of_type("ROSTER") if len(t.items_of("LINEUP")) == 2]
    assert swaps
    first, second = swaps[0].items_of("LINEUP")
    assert first.from_lineup_slot_id == second.to_lineup_slot_id
    assert first.to_lineup_slot_id == second.from_lineup_slot_id


def test_nba_moves_for_later_days_are_future_roster() -> None:
    view = TransactionsView.model_validate(load("fba/mTransactions2.json"))
    future = view.of_type("FUTURE_ROSTER")
    assert future and all(t.scoring_period_id and t.scoring_period_id > 1 for t in future)


# --- stat entries and filters -----------------------------------------------------------------------------------------


def stat_index(game: str, name: str) -> dict[str, dict[str, Any]]:
    return load(f"{game}/probes.json")[name]["returned"]


def test_nfl_stat_entry_ids() -> None:
    returned = stat_index("ffl", "stat_entries")
    assert returned["002026"]["statSplitTypeId"] == 0 and returned["102026"]["statSourceId"] == 1
    assert stat_entry_id(2026, 4, projected=True) in returned  # "1120264": projections are keyed by period
    assert stat_entry_id(2026, 4) not in returned  # actual weekly lines are keyed by the pro game id instead
    actual_weeks = {k: v for k, v in returned.items() if v["statSourceId"] == 0 and v["scoringPeriodId"] > 0}
    assert actual_weeks and all(k.startswith("01") and len(k) == 11 for k in actual_weeks)


def test_player_stat_entry_finds_each_games_single_period_lines() -> None:
    """``Player.stat_entry`` matches on fields: each game's "Game" split (1 in ffl, 5 in fba), never the id, and never
    one of fba's rolling windows."""
    for game in SEASONS:
        cards = PlayersView.model_validate(load(f"{game}/kona_playercard_stat_entries.json"))
        lines = [
            (entry.player, line) for entry in cards.players for line in entry.player.stats if line.scoring_period_id
        ]
        assert lines
        for player, line in lines:
            season, period, projected = line.season_id, line.scoring_period_id, line.is_projection
            assert player.stat_entry(season=season, scoring_period=period, projected=projected, game=game) is line
            assert player.stat_entry(season=season, scoring_period=period, projected=projected) is line
            other = "fba" if game == "ffl" else "ffl"  # the other game's split number finds nothing
            assert player.stat_entry(season=season, scoring_period=period, projected=projected, game=other) is None
            assert line.is_single_period
        if game == "fba":
            windows = [line for entry in cards.players for line in entry.player.stats if line.stat_split_type_id < 4]
            assert {line.stat_split_type_id for line in windows} == {0, 1, 2, 3}  # the season and the three windows
            assert not any(line.is_single_period for line in windows)
    banchero = PlayersView.model_validate(load("fba/kona_playercard_stat_entries.json")).players[0].player
    assert banchero.actual(2026, 174, game="fba") is not None  # a played day of last season, split 5
    assert banchero.actual(2027, 0) is not None and banchero.actual(2027, 0) is banchero.stat_entry(season=2027)


def test_nba_daily_lines_use_split_five_and_windows_use_one_to_three() -> None:
    returned = stat_index("fba", "stat_entries")
    daily = [v for v in returned.values() if v["scoringPeriodId"] > 0]
    assert daily and {v["statSplitTypeId"] for v in daily} == {5}
    assert all(k.startswith("05") and len(k) == 11 for k, v in returned.items() if v["scoringPeriodId"] > 0)
    windows = {k: v for k, v in returned.items() if k in ("012027", "022027", "032027")}
    assert {v["statSplitTypeId"] for v in windows.values()} == {1, 2, 3}
    assert all(v["scoringPeriodId"] == 0 for v in windows.values())
    assert stat_entry_id(2027, 1, projected=True) not in returned  # ESPN publishes no daily NBA projections


def split_labels(first_stat: str) -> list[dict[int, tuple[str, bool]]]:
    """``{statSplitTypeId: (label, gameSplit)}`` of every web-client stat settings block whose first stat is
    ``first_stat`` (``webclient.json``; both basketball games share ``offensive.points``)."""
    return [
        {split["id"]: (split["description"], split["gameSplit"]) for split in entry["splitTypes"]}
        for entry in load("webclient.json")["statSettings"]
        if entry["firstStat"] == first_stat
    ]


def test_web_client_labels_the_stat_splits() -> None:
    """ESPN's own labels behind docs/espn-api.md section 1 #10-#11: the single-period actual lines are each game's
    "Game" split (1 in football, 5 in basketball), and 1/2/3 are basketball's last 7/15/30 days."""
    (football,) = split_labels("passing.passingAttempts")
    assert football == {0: ("Season", False), 1: ("Game", True), 2: ("Rest Of Season", False)}
    basketball = split_labels("offensive.points")
    assert basketball
    for labels in basketball:
        assert labels == {
            0: ("Season", False),
            1: ("Last 7 Days", False),
            2: ("Last 15 Days", False),
            3: ("Last 30 Days", False),
            4: ("Average", False),
            5: ("Game", True),
        }
    for game, split in (("ffl", 1), ("fba", 5)):
        played = [v for v in stat_index(game, "stat_entries").values() if v["scoringPeriodId"] > 0]
        assert {v["statSplitTypeId"] for v in played if v["statSourceId"] == 0} == {split}


def test_nfl_sends_a_second_season_projection() -> None:
    """``122026`` (projected, split 2: "Rest Of Season") beside ``102026`` (projected, split 0). Their totals do not
    fit the labels: ``122026`` covers more games (appliedTotal / appliedAverage), so which line is rest-of-season is
    unsettled (docs/espn-api.md section 1 #10)."""
    lines = {line["id"]: line for line in load("ffl/kona_player_info.json")["players"][0]["player"]["stats"]}
    season, second = lines["102026"], lines["122026"]
    assert (season["statSourceId"], season["statSplitTypeId"], season["scoringPeriodId"]) == (1, 0, 0)
    assert (second["statSourceId"], second["statSplitTypeId"], second["scoringPeriodId"]) == (1, 2, 0)

    def games(line: dict[str, Any]) -> int:
        return round(line["appliedTotal"] / line["appliedAverage"])

    assert games(season) < games(second)


def test_top_scoring_periods_filter() -> None:
    for game in ("ffl", "fba"):
        probes = load(f"{game}/probes.json")
        assert probes["top_scoring_period_zero"]["status"] == 400
        assert "FILTER_INVALID_VALUE" in probes["top_scoring_period_zero"]["detail"]
        assert probes["top_scoring_periods_two"]["status"] == 200
        assert probes["filter_slot_ids"]["same_players_same_order"] is True


def test_nba_has_season_projections_only() -> None:
    assert load("fba/kona_game_state.json")["settings"]["statSettings"]["hasGameStatProjections"] is False
    assert load("ffl/kona_game_state.json")["settings"]["statSettings"]["hasGameStatProjections"] is True


# --- write payloads: ESPN's own code (webclient.json; docs/espn-api.md section 4) -------------------------------------


def transaction_code(part: str) -> str:
    """One excerpt of the web client's write path, with the type constants it reads spelled out
    (``this.type===r["N"]`` becomes ``this.type==="WAIVER"``)."""
    code = load("webclient.json")["transactionCode"]
    source = code["excerpts"][part]
    alias = code["constantsAlias"].get(part)
    return load_capture("webclient").resolve_constants(source, alias, code["typeNames"]) if alias else source


def service_method(name: str) -> str:
    service = transaction_code("service")
    start = service.index(f"static {name}(")
    end = service.find("static ", start + 1)
    return service[start : end if end > 0 else len(service)]


def types_in(condition: str) -> set[str]:
    return set(re.findall(r'this\.type[!=]=="(\w+)"', condition))


def test_serializer_builds_the_documented_envelope() -> None:
    model = transaction_code("model")
    body = model[model.index("get(){") :]
    assert re.search(r"\{isLeagueManager:this\.isLeagueManager,teamId:this\.teamId,type:this\.type\}", body)
    for key in ("memberId", "scoringPeriodId", "executionType"):
        assert re.search(rf"this\.{key}&&\(\w+\.{key}=this\.{key}\)", body)
    assert re.search(r'this\.executionType=\w+\|\|"EXECUTE"', model)
    assert re.search(r"this\.items\.length>0&&\(\w+\.items=this\.items\)", body)
    waiver = re.search(r'if\(this\.type==="WAIVER"\)\{(\w+)\.bidAmount=this\.bidAmount;([^}]*)\}', body)
    assert waiver and "relatedTransactionId" in waiver.group(2)
    assert re.search(r'this\.type==="TRADE_PROPOSAL"&&\(\w+\.expirationDate=this\.expirationDate\)', body)
    comment = re.search(r"((?:this\.type!==\"\w+\"&&)*this\.type!==\"\w+\")\|\|\(\w+\.comment=this\.comment\)", body)
    assert comment and types_in(comment.group(1)) == {"TRADE_PROPOSAL", "TRADE_DECLINE", "TRADE_VETO"}
    related = re.search(r"\(([^()]*)\)&&\(\w+\.relatedTransactionId=this\.relatedTransactionId\)", body)
    assert related and types_in(related.group(1)) == {
        "TRADE_PROPOSAL",
        "TRADE_ACCEPT",
        "TRADE_DECLINE",
        "TRADE_UPHOLD",
        "TRADE_VETO",
    }
    assert re.search(
        r"if\(this\.isLeagueManager\)\{\w+\.isActingAsTeamOwner=this\.isActingAsTeamOwner;"
        r"\w+\.skipTransactionCounters=this\.skipTransactionCounters\}",
        body,
    )


def test_item_builders() -> None:
    model = transaction_code("model")
    assert re.search(r'playerId:\w+\.id,type:"ADD",toTeamId:\w+\}', model)
    assert re.search(r'playerId:\w+\.id,type:"DROP",fromTeamId:\w+\}', model)
    assert re.search(r'fromSlotId:\w+\.lineupSlotId,playerId:\w+\.id,toSlotId:\w+\.toSlotId,type:"LINEUP"\}', model)
    assert len(re.findall(r'playerId:\w+\.id,type:"TRADE",fromTeamId:\w+,toTeamId:\w+\}', model)) == 2
    for key in ("fromLineupSlotId", "toLineupSlotId"):  # LINEUP items carry slots and no team ids
        assert re.search(rf'"undefined"!==typeof \w+&&\(\w+\.{key}=\w+\)', model)


def test_each_flow_picks_its_transaction_type() -> None:
    service = transaction_code("service")
    create = service[service.index("createTransaction({") : service.index("createOfflineDraftTransaction")]
    assert '"profile.swid"' in create  # memberId: the signed-in SWID
    assert re.search(r"\w+\|\|\w+\.status\.latestScoringPeriod", create)  # scoringPeriodId default
    move = service_method("movePlayers")
    assert '="ROSTER"' in move and re.search(r'if\(!\w+&&(\w+)>(\w+)\)\{\2=\1;\w+="FUTURE_ROSTER"\}', move)
    assert '==="FREEAGENT"?"add":"claim"' in service_method("addPlayers")
    assert 'type:"ROSTER"' in service_method("dropPlayers")
    trade = service_method("proposeTrade")
    assert 'type:"TRADE_PROPOSAL"' in trade and 'type:"ACQUISITION_BUDGET_TRADE"' in trade
    assert re.search(r'type:"TRADE_ACCEPT",relatedTransactionId:\w+', service_method("acceptTrade"))
    assert re.search(r'type:"TRADE_DECLINE",comment:\w+,relatedTransactionId:\w+', service_method("declineTrade"))
    assert re.search(
        r'type:"TRADE_PROPOSAL",relatedTransactionId:\w+,executionType:"CANCEL"', service_method("cancelTrade")
    )
    assert re.search(
        r'type:"WAIVER",relatedTransactionId:\w+,executionType:"CANCEL"', service_method("cancelWaiverClaim")
    )


def test_writes_post_json_to_the_transactions_path_on_the_write_host() -> None:
    save = transaction_code("saveTransaction")
    assert '"/segments/0/leagues/"' in save and '"/transactions/"' in save and "JSON.stringify(" in save
    post = transaction_code("post")
    host_type = re.search(r'const (\w+)="HOST_TYPE_FANTASY_WRITE_API"', transaction_code("writeHostType"))
    assert host_type and f"hostType:{host_type.group(1)}" in post
    assert '["Content-Type"]="application/json"' in post and '["post"]' in post
    assert '"https://lm-api-writes."' in transaction_code("writeHost")
    assert '"X-Fantasy-Source":"kona"' in transaction_code("requestDefaults")
    config = transaction_code("requestConfig")
    assert "withCredentials:true" in config and '"X-Fantasy-Platform":' in config
    assert '"/pendingTransactions"' in transaction_code("reorderPendingTransactions")
    assert "bidAmount" in transaction_code("updatePendingBid")


# --- trade offers: observed, reviewed and derived ---------------------------------------------------------------------
#
# ffl/write_TRADE_PROPOSAL_1.json is observed: the web app's own request, aborted by the guard on the write host on
# 2026-10-06 when a guarded `writes` session went as far as Send Trade Proposal. {ffl,fba}/trade_review.json are
# observed: `trade-review` took each builder to its review step and stopped. {ffl,fba}/write_TRADE_PROPOSAL_derived.json
# are derived, not observed: ESPN's saved code run offline with the send stubbed, for the offer the review showed.

ISO_MILLIS_Z = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")
OFFER_KEYS = [
    "isLeagueManager",
    "teamId",
    "type",
    "memberId",
    "scoringPeriodId",
    "executionType",
    "items",
    "expirationDate",
    "comment",
]
OFFERS = [
    "ffl/write_TRADE_PROPOSAL_1.json",
    "ffl/write_TRADE_PROPOSAL_derived.json",
    "fba/write_TRADE_PROPOSAL_derived.json",
]


@pytest.mark.parametrize("relative", OFFERS)
def test_an_offer_has_the_serializers_keys_and_an_iso_expiry(relative: str) -> None:
    body = load(relative)["body"]
    assert list(body) == OFFER_KEYS and body["type"] == "TRADE_PROPOSAL" and body["executionType"] == "EXECUTE"
    assert ISO_MILLIS_Z.fullmatch(body["expirationDate"])  # a string on the wire; ESPN's records hold epoch ms
    assert body["comment"] == ""  # the empty textarea is sent, not left out
    ours = body["teamId"]
    assert all(list(item) == ["playerId", "type", "fromTeamId", "toTeamId"] for item in body["items"])
    sides = [item["fromTeamId"] == ours for item in body["items"]]
    assert sides == sorted(sides, reverse=True) and True in sides and False in sides  # ours first, then theirs
    assert all(ours in (item["fromTeamId"], item["toTeamId"]) for item in body["items"])


def test_the_observed_offer_expires_two_days_after_it_was_sent() -> None:
    capture = load("ffl/write_TRADE_PROPOSAL_1.json")
    sent = datetime.fromisoformat(capture["at"])
    expires = datetime.fromisoformat(capture["body"]["expirationDate"].replace("Z", "+00:00"))
    assert timedelta(days=2) - timedelta(seconds=1) < expires - sent <= timedelta(days=2)  # the builder's "2 Days"


@pytest.mark.parametrize("game", ["ffl", "fba"])
def test_the_derived_offer_is_the_reviewed_one(game: str) -> None:
    derived, review = load(f"{game}/write_TRADE_PROPOSAL_derived.json"), load(f"{game}/trade_review.json")
    selection = review["selection"]
    assert derived["inputs"]["players"] == selection["players"]
    assert [(item["playerId"], item["fromTeamId"], item["toTeamId"]) for item in derived["body"]["items"]] == [
        (player["id"], player["teamId"], selection["toTeamId" if player["side"] == "ours" else "fromTeamId"])
        for player in selection["players"]
    ]
    assert derived["body"]["teamId"] == selection["fromTeamId"] == our_team(game)
    assert derived["body"]["scoringPeriodId"] == selection["latestScoringPeriod"]
    expiry = next(select for select in review["review"]["selects"] if select["options"])
    assert derived["inputs"]["expiry_days"] == int(expiry["value"])


def test_the_derived_offer_matches_the_observed_one_but_for_its_players() -> None:
    observed, derived = load("ffl/write_TRADE_PROPOSAL_1.json"), load("ffl/write_TRADE_PROPOSAL_derived.json")
    assert derived["url"] == observed["url"] and derived["headers"] == observed["headers"]
    assert derived["method"] == observed["method"] == "POST"
    for key in ("isLeagueManager", "teamId", "type", "memberId", "scoringPeriodId", "executionType", "comment"):
        assert derived["body"][key] == observed["body"][key]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the saved client code")
@pytest.mark.parametrize("game", ["ffl", "fba"])
def test_the_derivation_reproduces_from_the_saved_client_code(game: str) -> None:
    """Re-run ``derive_trade.cjs`` on the committed ``webclient.json``: ESPN's own code, offline, send stubbed."""
    derived = load(f"{game}/write_TRADE_PROPOSAL_derived.json")
    body = derived["body"]
    inputs = {
        "game": game,
        "seasonId": SEASONS[game],
        "leagueId": LEAGUE_PLACEHOLDERS[game],
        "latestScoringPeriod": body["scoringPeriodId"],
        "swid": body["memberId"],
        "fromTeamId": body["teamId"],
        "toTeamId": next(item["toTeamId"] for item in body["items"] if item["fromTeamId"] == body["teamId"]),
        "trade": [{"id": player["id"], "teamId": player["teamId"]} for player in derived["inputs"]["players"]],
        "expirationDate": body["expirationDate"],
        "comment": body["comment"],
    }
    done = subprocess.run(
        ["node", str(CAPTURE_DIR / "derive_trade.cjs"), str(HERE / "webclient.json")],
        input=json.dumps(inputs),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    result = json.loads(done.stdout)
    assert json.dumps(json.loads(result["data"])) == json.dumps(body)  # same keys, values and order
    assert derived["url"].split("?")[0].endswith("/apis/v3/" + result["path"])


@pytest.mark.parametrize("game", ["ffl", "fba"])
def test_the_review_step_was_reached_and_nothing_tried_to_send(game: str) -> None:
    review = load(f"{game}/trade_review.json")
    assert review["status"] == "review reached" and review["send_button_present"]
    assert review["review"]["trapHits"] == 0  # nothing even tried to press Send
    assert len(review["canary"]) == 8  # four probes before the page opened, four more before the first click
    for record in review["review_blocked"]:  # analytics beacons only: no league write, no transaction
        assert "lm-api-writes" not in record["url"] and "/transactions" not in record["url"].lower()
    dialog = review["review"]["dialogs"][0]["text"]
    assert dialog[0] == "Confirm Transaction" and "Receiving:" in dialog and "Offering:" in dialog
    assert "Keep trade open for:" in dialog
    assert [line for line in dialog if line.endswith(" Days")] == [f"{n} Days" for n in range(1, 8)]
    assert any(line.startswith("An email will be sent to all managers of") for line in dialog)


@pytest.mark.parametrize("game", ["ffl", "fba"])
def test_continue_is_client_side(game: str) -> None:
    """Pressing Continue read no league view and wrote nothing: only the presence heartbeat went out."""
    reads = load(f"{game}/trade_review.json")["review_reads"]
    assert [read["url"].split("?")[0] for read in reads] == ["https://presence.fantasy.espn.com/apis/v1/heartbeat"]


# --- rosters ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("game", ["ffl", "fba"])
def test_our_roster_is_whole(game: str) -> None:
    league = settings(game)
    roster = RostersView.model_validate(load(f"{game}/mRoster.json")).roster(our_team(game))
    allowed = {slot.slot_id for slot in league.lineup_slots}
    assert roster.entries and {entry.lineup_slot_id for entry in roster.entries} <= allowed
    for slot in league.active_slots:
        assert len(roster.in_slot(slot.slot_id)) <= slot.count
    assert len(roster.entries) <= league.roster_size + league.ir_count


# --- the scrubber that wrote these fixtures (scripts/capture/scrub.py) ------------------------------------------------


REAL_LEAGUE = 987654321
OUR_SWID = "{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE}"
THEIR_SWID = "{11111111-2222-3333-4444-555555555555}"
PLACEHOLDER_ONE = "{00000000-0000-0000-0000-000000000001}"  # ours
PLACEHOLDER_TWO = "{00000000-0000-0000-0000-000000000002}"
TEAMS_VIEW: dict[str, Any] = {
    "id": REAL_LEAGUE,
    "gameId": 1,
    "members": [
        {"id": THEIR_SWID, "displayName": "Pat Example", "firstName": "Pat", "lastName": "Example"},
        {"id": OUR_SWID, "displayName": "Jo Sample", "firstName": "Jo", "lastName": "Sample"},
    ],
    "teams": [
        {"id": 4, "name": "Example Elite", "abbrev": "EXEL", "owners": [THEIR_SWID], "primaryOwner": THEIR_SWID},
        {"id": 9, "name": "Sample Squad", "abbrev": "SMPL", "owners": [OUR_SWID], "logo": "https://x.invalid/a.png"},
    ],
}
WRITE_RECORD: dict[str, Any] = {
    "method": "POST",
    "url": f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl/seasons/2026/segments/0/leagues/{REAL_LEAGUE}/transactions/",
    "body": {
        "isLeagueManager": False,
        "teamId": 9,
        "type": "FREEAGENT",
        "memberId": OUR_SWID,
        "scoringPeriodId": 4,
        "executionType": "EXECUTE",
        "items": [
            {"playerId": 3918298, "type": "ADD", "toTeamId": 9},
            {"playerId": 4429795, "type": "DROP", "fromTeamId": 9},
        ],
    },
}


def make_scrubber() -> Any:
    return load_capture("scrub").Scrubber.for_league(
        game="ffl",
        league_id=REAL_LEAGUE,
        our_team_id=9,
        teams_view=TEAMS_VIEW,
        settings={"settings": {"name": "Sample Friends League"}},
    )


def test_scrubber_replaces_every_identity() -> None:
    scrubber = make_scrubber()
    teams = scrubber.scrub(TEAMS_VIEW)
    assert teams["id"] == LEAGUE_PLACEHOLDERS["ffl"]
    ours, theirs = (next(t for t in teams["teams"] if t["owners"] == [s]) for s in (PLACEHOLDER_ONE, PLACEHOLDER_TWO))
    assert ours["id"] == 4 and theirs["id"] == 9  # ours takes the smallest id of the league's own set
    assert (ours["name"], ours["abbrev"], ours["logo"]) == (
        "Team 4",
        "T4",
        "https://example.invalid/fixture-team-logo.png",
    )
    assert [m["displayName"] for m in teams["members"]] == ["Manager 2", "Manager 1"]
    assert scrubber.leaks(teams) == []

    write = scrubber.scrub(WRITE_RECORD)
    assert f"/leagues/{LEAGUE_PLACEHOLDERS['ffl']}/transactions/" in write["url"]
    assert write["body"]["teamId"] == 4 and write["body"]["memberId"] == PLACEHOLDER_ONE
    assert [item.get("toTeamId") or item.get("fromTeamId") for item in write["body"]["items"]] == [4, 4]
    assert [item["playerId"] for item in write["body"]["items"]] == [3918298, 4429795]
    assert scrubber.leaks(write) == []


def test_scrubber_flags_leftovers_but_not_public_player_names() -> None:
    scrubber = make_scrubber()
    assert scrubber.leaks(TEAMS_VIEW)  # unscrubbed: names, SWIDs and the league id are all still there
    assert scrubber.leaks(WRITE_RECORD)
    assert scrubber.leaks({"note": "trade with Sample Squad"})
    assert scrubber.leaks({"players": [{"player": {"fullName": "Pat Example"}}]}) == []
