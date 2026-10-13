"""League discovery, offline: the season rule, the fan-API GET and its two parser paths over the synthetic fixtures in
``tests/fixtures/espn/fan_api/`` (structured, URL-only, other sports, malformed), league-URL parsing for both sports,
``resolve_team`` against the scrubbed real ``mTeam`` capture, and ``confirm_league`` over a mocked reads host.

Every HTTP exchange goes through respx or an ``httpx.MockTransport``; the conftest network guard would fail anything
else. The fan-API fixtures are hand-written stand-ins (see their README), never captures.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
import pytest
import respx

from fm.espn.auth import EspnSession
from fm.espn.client import READS_HOST, EspnClient, View
from fm.espn.discover import (
    DISCOVERY_ATTEMPTS,
    DISCOVERY_TIMEOUT,
    FAN_API_HOST,
    FAN_API_URL,
    DiscoveredLeague,
    DiscoveryError,
    LeagueRef,
    confirm_league,
    current_season,
    discover_leagues,
    parse_fan_profile,
    parse_league_url,
    resolve_team,
)
from fm.espn.ids import Game
from fm.espn.models import TeamsView

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "espn"
FAN_API = FIXTURES / "fan_api"
REAL_FFL = FIXTURES / "real" / "ffl"

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)  # NFL 2026 and NBA 2027 are the current seasons
SWID = "{00000000-0000-0000-0000-000000000001}"  # owns team 1 in the real ffl capture
S2 = "s2-secret-value"
SESSION = EspnSession(espn_s2=S2, swid=SWID, expires_at=NOW + timedelta(days=200))

LEAGUE_A = DiscoveredLeague("nfl", 1010101, 2026, 7, "League A", "Fixture Team 7")
LEAGUE_B = DiscoveredLeague("nba", 2020202, 2027, 3, "League B", "Fixture Team Three")
LEAGUE_C = DiscoveredLeague("nfl", 3030303, 2025, 2, "League C", "Fixture Team 2")
NAMELESS_A = DiscoveredLeague("nfl", 1010101, 2026, team_hint=7)  # a URL's teamId is a hint, never the team
NAMELESS_B = DiscoveredLeague("nba", 2020202, 2027, team_hint=3)
NAMELESS_C = DiscoveredLeague("nfl", 3030303, 2025)

REAL_LEAGUE_ID = 1010101
REAL_SEASON = 2026


def fixture(name: str) -> bytes:
    return (FAN_API / name).read_bytes()


def ok(body: bytes) -> httpx.Response:
    return httpx.Response(200, content=body, headers={"content-type": "application/json;charset=UTF-8"})


def espn_error(status: int, kind: str, message: str) -> httpx.Response:
    body = {"messages": [message], "details": [{"message": message, "shortMessage": message, "type": kind}]}
    return httpx.Response(status, json=body)


class FakeFanApi:
    """Answers the fan profile GET from a queue (the last answer repeats) and records every request."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.requests: list[httpx.Request] = []
        self.answers: list[httpx.Response | Exception] = [ok(fixture("structured.json"))]
        self.route = router.get(host=FAN_API_HOST, path__startswith="/apis/v2/fans/").mock(side_effect=self._answer)

    def serve(self, *answers: httpx.Response | Exception) -> None:
        self.answers = list(answers)

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def fan_api() -> Iterator[FakeFanApi]:
    with respx.mock(assert_all_called=False) as router:
        yield FakeFanApi(router)


# --- current_season ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sport", "when", "season"),
    [
        ("nfl", datetime(2026, 1, 15, tzinfo=UTC), 2025),
        ("nfl", datetime(2026, 7, 31, tzinfo=UTC), 2025),
        ("nfl", datetime(2026, 8, 1, tzinfo=UTC), 2026),
        ("nfl", datetime(2026, 12, 31, tzinfo=UTC), 2026),
        ("nba", datetime(2026, 7, 31, tzinfo=UTC), 2026),
        ("nba", datetime(2026, 8, 1, tzinfo=UTC), 2027),
        ("nba", datetime(2027, 1, 15, tzinfo=UTC), 2027),
        ("nba", datetime(2027, 6, 30, tzinfo=UTC), 2027),
    ],
)
def test_current_season_rolls_over_in_august(sport: str, when: datetime, season: int) -> None:
    assert current_season(sport, when) == season


def test_current_season_accepts_game_keys_and_defaults_to_today() -> None:
    assert current_season(Game.FFL, NOW) == current_season("ffl", NOW) == 2026
    assert current_season(Game.FBA, NOW) == current_season("fba", NOW) == 2027
    assert current_season("nfl") in (datetime.now(UTC).year - 1, datetime.now(UTC).year)


# --- discover_leagues: request shape ----------------------------------------------------------------------------------


def test_fan_profile_request_carries_swid_cookies_and_timeouts(fan_api: FakeFanApi) -> None:
    discover_leagues(SESSION, now=NOW)
    assert len(fan_api.requests) == 1
    request = fan_api.requests[0]
    assert request.method == "GET"
    assert unquote(str(request.url)) == f"{FAN_API_URL}/{SWID}"
    assert request.headers["cookie"] == f"espn_s2={S2}; SWID={SWID}"
    assert request.headers["accept"] == "application/json"
    assert request.headers["user-agent"].startswith("espn-fantasy/")
    assert request.extensions["timeout"] == DISCOVERY_TIMEOUT.as_dict()


def test_bare_or_encoded_swid_is_sent_with_braces(fan_api: FakeFanApi) -> None:
    bare = EspnSession(espn_s2=S2, swid=SWID.strip("{}"), expires_at=None)
    discover_leagues(bare, now=NOW)
    encoded = EspnSession(espn_s2=S2, swid="%7B" + SWID.strip("{}") + "%7D", expires_at=None)
    discover_leagues(encoded, now=NOW)
    for request in fan_api.requests:
        assert unquote(request.url.path) == f"/apis/v2/fans/{SWID}"
        assert request.headers["cookie"].endswith(f"SWID={SWID}")


def test_injected_client_is_used_and_left_open() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok(fixture("structured.json"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        leagues = discover_leagues(SESSION, client=client, now=NOW)
        assert not client.is_closed
        assert leagues == [LEAGUE_A, LEAGUE_B]
    assert len(seen) == 1 and seen[0].url.host == FAN_API_HOST


def test_transport_error_is_retried_once(fan_api: FakeFanApi) -> None:
    fan_api.serve(httpx.ConnectError("boom"), ok(fixture("structured.json")))
    assert discover_leagues(SESSION, now=NOW) == [LEAGUE_A, LEAGUE_B]
    assert len(fan_api.requests) == 2 == DISCOVERY_ATTEMPTS


def test_repeated_transport_errors_raise(fan_api: FakeFanApi) -> None:
    fan_api.serve(httpx.ConnectError("boom"), httpx.ReadTimeout("slow"))
    with pytest.raises(DiscoveryError, match="could not reach fan.api.espn.com.*ReadTimeout"):
        discover_leagues(SESSION, now=NOW)
    assert len(fan_api.requests) == DISCOVERY_ATTEMPTS


def test_error_statuses_are_not_retried(fan_api: FakeFanApi) -> None:
    fan_api.serve(httpx.Response(500, text="oops"))
    with pytest.raises(DiscoveryError, match="HTTP 500"):
        discover_leagues(SESSION, now=NOW)
    assert len(fan_api.requests) == 1


def test_rejected_session_says_to_log_in(fan_api: FakeFanApi) -> None:
    fan_api.serve(espn_error(401, "AUTH_MISSING_CREDENTIALS", "no credentials"))
    with pytest.raises(DiscoveryError, match=r"rejected the session.*fm login"):
        discover_leagues(SESSION, now=NOW)
    fan_api.serve(espn_error(403, "AUTH_MISSING_CREDENTIALS", "no credentials"))
    with pytest.raises(DiscoveryError, match="fm login"):
        discover_leagues(SESSION, now=NOW)


# --- discover_leagues: parser paths -----------------------------------------------------------------------------------


def test_structured_shape(fan_api: FakeFanApi) -> None:
    assert discover_leagues(SESSION, now=NOW) == [LEAGUE_A, LEAGUE_B]


def test_structured_shape_keeps_every_season_on_request(fan_api: FakeFanApi) -> None:
    assert discover_leagues(SESSION, now=NOW, all_seasons=True) == [LEAGUE_A, LEAGUE_B, LEAGUE_C]


def test_structured_shape_drops_a_league_that_is_no_longer_current(fan_api: FakeFanApi) -> None:
    later = datetime(2027, 9, 1, tzinfo=UTC)  # NFL 2027 and NBA 2028
    assert discover_leagues(SESSION, now=later) == []


def test_url_only_shape(fan_api: FakeFanApi) -> None:
    fan_api.serve(ok(fixture("url_only.json")))
    assert discover_leagues(SESSION, now=NOW) == [NAMELESS_A, NAMELESS_B]
    assert discover_leagues(SESSION, now=NOW, all_seasons=True) == [NAMELESS_A, NAMELESS_B, NAMELESS_C]


def test_other_sport_or_non_fantasy_profile_has_no_leagues(fan_api: FakeFanApi) -> None:
    fan_api.serve(ok(fixture("other_sport.json")))
    assert discover_leagues(SESSION, now=NOW) == []
    assert discover_leagues(SESSION, now=NOW, all_seasons=True) == []


@pytest.mark.parametrize(
    "body",
    [
        b"<!doctype html><html><body>Sign in</body></html>",
        b"[1, 2, 3]",
        b"null",
        b'"a string"',
        b"{not json",
        b"",
    ],
)
def test_malformed_body_raises(fan_api: FakeFanApi, body: bytes) -> None:
    fan_api.serve(ok(body))
    with pytest.raises(DiscoveryError, match="not JSON|not a JSON object"):
        discover_leagues(SESSION, now=NOW)


def test_entry_without_groups_falls_back_to_its_url() -> None:
    profile: dict[str, Any] = {
        "preferences": [
            {
                "metaData": {
                    "entry": {
                        "abbrev": "FBA",
                        "entryUrl": "https://fantasy.espn.com/basketball/team?leagueId=2020202&teamId=3&seasonId=2027",
                        "entryMetadata": {"teamName": "Fixture Team Three"},
                    }
                }
            }
        ]
    }
    expected = DiscoveredLeague("nba", 2020202, 2027, None, None, "Fixture Team Three", team_hint=3)
    assert parse_fan_profile(profile, now=NOW) == [expected]


def test_entry_fields_may_be_strings_and_keys_may_differ_in_case() -> None:
    profile: dict[str, Any] = {
        "Preferences": [
            {
                "metadata": {
                    "Entry": {
                        "gameid": "1",
                        "seasonid": "2026",
                        "entryid": "7",
                        "groups": [{"groupid": "1010101", "groupname": " League A "}],
                    }
                }
            }
        ]
    }
    assert parse_fan_profile(profile, now=NOW) == [DiscoveredLeague("nfl", 1010101, 2026, 7, "League A", None)]


def test_an_unsupported_game_id_or_abbrev_settles_the_sport_before_the_url_does() -> None:
    """A baseball entry stays dropped even when its URL says football; with a structured NBA league beside it the deep
    walk never runs, so the football URL is not picked up either."""
    baseball = {"gameId": 2, "entryURL": "https://fantasy.espn.com/football/team?leagueId=1010101&teamId=7"}
    hockey = {"abbrev": "FHL", "entryURL": "https://fantasy.espn.com/basketball/team?leagueId=4040404&teamId=5"}
    nba = {"gameId": 3, "seasonId": 2027, "entryId": 3, "groups": [{"groupId": 2020202}]}
    profile: dict[str, Any] = {
        "preferences": [{"metaData": {"entry": entry}} for entry in (baseball, hockey, nba)],
    }
    assert parse_fan_profile(profile, now=NOW, all_seasons=True) == [DiscoveredLeague("nba", 2020202, 2027, 3)]


def test_the_deep_walk_runs_when_no_entry_is_an_nfl_or_nba_league() -> None:
    profile: dict[str, Any] = {
        "preferences": [{"metaData": {"entry": {"gameId": 4, "groups": [{"groupId": 4040404}]}}}],
        "promo": "https://fantasy.espn.com/basketball/league?leagueId=2020202&seasonId=2027",
    }
    assert parse_fan_profile(profile, now=NOW) == [DiscoveredLeague("nba", 2020202, 2027)]


def test_structured_entries_win_over_stray_urls() -> None:
    profile: dict[str, Any] = {
        "preferences": [
            {"metaData": {"entry": {"gameId": 1, "seasonId": 2026, "entryId": 7, "groups": [{"groupId": 1010101}]}}}
        ],
        "promo": "https://fantasy.espn.com/basketball/league?leagueId=2020202&seasonId=2027",
    }
    assert parse_fan_profile(profile, now=NOW) == [DiscoveredLeague("nfl", 1010101, 2026, 7)]


def test_deep_walk_merges_duplicates_and_fills_missing_fields() -> None:
    profile: dict[str, Any] = {
        "a": "https://fantasy.espn.com/football/league?leagueId=1010101&seasonId=2026",
        "b": ["https://fantasy.espn.com/football/team?leagueId=1010101&teamId=7&seasonId=2026"],
        "c": {"d": "https://fantasy.espn.com/football/team?leagueId=1010101&teamId=9&seasonId=2026"},
    }
    assert parse_fan_profile(profile, now=NOW) == [DiscoveredLeague("nfl", 1010101, 2026, team_hint=7)]


def test_the_deep_walk_never_takes_a_urls_team_as_the_accounts_team() -> None:
    """A fan profile links to an opponent's team page before ours: the first URL wins the merge, so its teamId must
    stay a hint and the team be settled by SWID in confirm_league."""
    profile: dict[str, Any] = {
        "news": "Your opponent: https://fantasy.espn.com/football/team?leagueId=1010101&teamId=3&seasonId=2026",
        "mine": "https://fantasy.espn.com/football/team?leagueId=1010101&teamId=1&seasonId=2026",
    }
    [league] = parse_fan_profile(profile, now=NOW)
    assert league.team_id is None and league.team_hint == 3
    assert league.ref == LeagueRef("nfl", 1010101, 2026, None, team_hint=3)


def test_a_structured_entry_id_is_trusted_and_its_url_team_is_not_needed() -> None:
    entry = {
        "gameId": 1,
        "seasonId": 2026,
        "entryId": 7,
        "entryURL": "https://fantasy.espn.com/football/team?leagueId=1010101&teamId=9",
        "groups": [{"groupId": 1010101}],
    }
    profile: dict[str, Any] = {"preferences": [{"metaData": {"entry": entry}}]}
    assert parse_fan_profile(profile, now=NOW) == [DiscoveredLeague("nfl", 1010101, 2026, 7)]


def _seasons_profile(*rows: tuple[int, int, int]) -> dict[str, Any]:
    """Structured entries ``(gameId, seasonId, leagueId)``, entry 1 in each."""
    entries = [
        {"gameId": game, "seasonId": season, "entryId": 1, "groups": [{"groupId": league}]}
        for game, season, league in rows
    ]
    return {"preferences": [{"metaData": {"entry": entry}} for entry in entries]}


def test_in_july_a_league_renewed_for_the_coming_season_is_the_one_kept() -> None:
    july = datetime(2026, 7, 15, tzinfo=UTC)  # NFL current is still 2025, NBA current is 2026
    assert (current_season("nfl", july), current_season("nba", july)) == (2025, 2026)
    profile = _seasons_profile((1, 2025, 1010101), (1, 2026, 1010101), (3, 2026, 2020202), (3, 2027, 2020202))
    assert parse_fan_profile(profile, now=july) == [
        DiscoveredLeague("nfl", 1010101, 2026, 1),
        DiscoveredLeague("nba", 2020202, 2027, 1),
    ]


def test_a_league_not_yet_renewed_keeps_its_current_season_and_older_ones_are_dropped() -> None:
    july = datetime(2026, 7, 15, tzinfo=UTC)
    profile = _seasons_profile((1, 2024, 1010101), (1, 2025, 1010101), (1, 2024, 3030303))
    assert parse_fan_profile(profile, now=july) == [DiscoveredLeague("nfl", 1010101, 2025, 1)]
    assert len(parse_fan_profile(profile, now=july, all_seasons=True)) == 3


@pytest.mark.parametrize("data", [[], None, "text", 7])
def test_parse_fan_profile_rejects_non_objects(data: object) -> None:
    with pytest.raises(DiscoveryError, match="not a JSON object"):
        parse_fan_profile(data, now=NOW)


def test_parse_fan_profile_defaults_to_today(fan_api: FakeFanApi) -> None:
    profile: dict[str, Any] = {"preferences": [{"metaData": {"entry": {"gameId": 1, "groups": [{"groupId": 1}]}}}]}
    [league] = parse_fan_profile(profile)
    assert league.season == current_season("nfl")


# --- parse_league_url -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "ref"),
    [
        (
            "https://fantasy.espn.com/football/team?leagueId=1010101&teamId=7&seasonId=2026",
            LeagueRef("nfl", 1010101, 2026, 7),
        ),
        (
            "https://fantasy.espn.com/basketball/team?leagueId=2020202&teamId=3&seasonId=2027",
            LeagueRef("nba", 2020202, 2027, 3),
        ),
        ("https://fantasy.espn.com/football/league?leagueId=1010101", LeagueRef("nfl", 1010101)),
        ("https://fantasy.espn.com/basketball/league?leagueId=2020202&seasonId=2027", LeagueRef("nba", 2020202, 2027)),
        (
            "https://fantasy.espn.com/football/league/scoreboard?leagueId=1010101&matchupPeriodId=5",
            LeagueRef("nfl", 1010101),
        ),
        (
            "https://fantasy.espn.com/basketball/league/scoreboard?leagueId=2020202&seasonId=2027&matchupPeriodId=2",
            LeagueRef("nba", 2020202, 2027),
        ),
        (
            "https://fantasy.espn.com/football/league/standings?leagueId=1010101&seasonId=2026",
            LeagueRef("nfl", 1010101, 2026),
        ),
        ("fantasy.espn.com/basketball/team?teamId=3&leagueId=2020202", LeagueRef("nba", 2020202, None, 3)),
        (
            "here it is: https://fantasy.espn.com/football/league?leagueId=1010101&seasonId=2026 thanks",
            LeagueRef("nfl", 1010101, 2026),
        ),
        (
            "https://fantasy.espn.com/football/team?leagueId=1010101&teamId=7#roster",
            LeagueRef("nfl", 1010101, None, 7),
        ),
    ],
)
def test_parse_league_url(text: str, ref: LeagueRef) -> None:
    assert parse_league_url(text) == ref


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not a url",
        "https://fantasy.espn.com/football/league?seasonId=2026",
        "https://fantasy.espn.com/football/league?leagueId=abc",
        "https://fantasy.espn.com/football/league?leagueId=0",
        "https://fantasy.espn.com/hockey/league?leagueId=4040404",
        "https://www.espn.com/nfl/scoreboard",
        "https://fantasy.espn.com/football/",
    ],
)
def test_parse_league_url_rejects_other_text(text: str) -> None:
    with pytest.raises(DiscoveryError, match="not an ESPN fantasy league URL"):
        parse_league_url(text)


def test_refs_and_discovered_leagues_know_their_game() -> None:
    assert LeagueRef("nfl", 1).game is Game.FFL
    assert LeagueRef("nba", 1).game is Game.FBA
    assert LEAGUE_B.game is Game.FBA
    assert LEAGUE_A.ref == LeagueRef("nfl", 1010101, 2026, 7)


# --- resolve_team -----------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_teams() -> TeamsView:
    return TeamsView.model_validate(json.loads((REAL_FFL / "mTeam+mStandings.json").read_text(encoding="utf-8")))


def test_resolve_team_by_swid(real_teams: TeamsView) -> None:
    assert resolve_team(real_teams, SWID) == 1
    assert resolve_team(real_teams, "{00000000-0000-0000-0000-000000000002}") == 3


@pytest.mark.parametrize(
    "swid",
    [
        "00000000-0000-0000-0000-000000000001",
        "%7B00000000-0000-0000-0000-000000000001%7D",
        "{00000000-0000-0000-0000-000000000001}".lower(),
        "  {00000000-0000-0000-0000-000000000001}  ",
    ],
)
def test_resolve_team_normalizes_the_swid(real_teams: TeamsView, swid: str) -> None:
    assert resolve_team(real_teams, swid) == 1


def test_resolve_team_with_no_match(real_teams: TeamsView) -> None:
    with pytest.raises(DiscoveryError, match=f"no team in league {REAL_LEAGUE_ID} is owned by this account"):
        resolve_team(real_teams, "{00000000-0000-0000-0000-0000000000FF}")


def test_resolve_team_with_several_matches() -> None:
    shared = {"id": REAL_LEAGUE_ID, "teams": [{"id": 1, "owners": [SWID]}, {"id": 4, "owners": ["{X}", SWID]}]}
    with pytest.raises(DiscoveryError, match=r"teams \[1, 4\] in league 1010101 all list this account"):
        resolve_team(TeamsView.model_validate(shared), SWID)


def test_a_url_hint_breaks_a_tie_only_between_the_accounts_own_teams(real_teams: TeamsView) -> None:
    shared = TeamsView.model_validate(
        {"id": REAL_LEAGUE_ID, "teams": [{"id": 1, "owners": [SWID]}, {"id": 4, "owners": ["{X}", SWID]}, {"id": 5}]}
    )
    assert resolve_team(shared, SWID, hint=4) == 4
    with pytest.raises(DiscoveryError, match="all list this account"):
        resolve_team(shared, SWID, hint=5)  # someone else's team never wins the tie
    assert resolve_team(real_teams, SWID, hint=3) == 1  # one owned team: the hint is not even asked


# --- confirm_league ---------------------------------------------------------------------------------------------------


class FakeReads:
    """Serves the league view GET on the reads host from one body and records the requests."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.requests: list[httpx.Request] = []
        self.answer: httpx.Response = ok(merged_team_and_settings())
        self.route = router.get(
            host=READS_HOST, path__regex=r"^/apis/v3/games/(?P<game>ffl|fba)/seasons/\d+/segments/0/leagues/\d+$"
        ).mock(side_effect=self._respond)

    def _respond(self, request: httpx.Request, game: str) -> httpx.Response:
        self.requests.append(request)
        return self.answer


def merged_team_and_settings(*, with_settings: bool = True) -> bytes:
    """The real ``mTeam+mStandings`` capture with the real ``mSettings`` block beside it, as one multi-view GET
    returns them."""
    body = json.loads((REAL_FFL / "mTeam+mStandings.json").read_text(encoding="utf-8"))
    if with_settings:
        settings = json.loads((REAL_FFL / "mSettings.json").read_text(encoding="utf-8"))
        body["settings"] = settings["settings"]
    return json.dumps(body).encode()


@pytest.fixture
def reads() -> Iterator[FakeReads]:
    with respx.mock(assert_all_called=False) as router:
        yield FakeReads(router)


def test_confirm_league_fills_season_team_and_names(reads: FakeReads) -> None:
    confirmed = confirm_league(LeagueRef("nfl", REAL_LEAGUE_ID), SESSION, now=NOW)
    assert confirmed == DiscoveredLeague("nfl", REAL_LEAGUE_ID, REAL_SEASON, 1, "Fixture ffl League", "Team 1")
    [request] = reads.requests
    assert request.url.path == f"/apis/v3/games/ffl/seasons/{REAL_SEASON}/segments/0/leagues/{REAL_LEAGUE_ID}"
    assert set(request.url.params.get_list("view")) == {View.TEAM, View.STANDINGS, View.SETTINGS}
    assert request.headers["cookie"] == f"espn_s2={S2}; SWID={SWID}"


def test_confirm_league_keeps_a_given_team_the_account_owns_and_the_season(reads: FakeReads) -> None:
    owner_of_3 = EspnSession(espn_s2=S2, swid="{00000000-0000-0000-0000-000000000002}", expires_at=None)
    confirmed = confirm_league(LeagueRef("nfl", REAL_LEAGUE_ID, REAL_SEASON, 3), owner_of_3, now=NOW)
    assert confirmed.team_id == 3 and confirmed.team_name == "Team 3" and confirmed.season == REAL_SEASON
    assert reads.requests[0].url.path.startswith(f"/apis/v3/games/ffl/seasons/{REAL_SEASON}/")


def test_confirm_league_refuses_a_pasted_team_the_account_does_not_own(reads: FakeReads) -> None:
    """A pasted opponent's team URL must never make the tool manage the opponent's team."""
    with pytest.raises(DiscoveryError, match=r"team 3 is not this account's team \(this account owns team 1\)"):
        confirm_league(LeagueRef("nfl", REAL_LEAGUE_ID, REAL_SEASON, 3), SESSION, now=NOW)
    stranger = EspnSession(espn_s2=S2, swid="{00000000-0000-0000-0000-0000000000FF}", expires_at=None)
    with pytest.raises(DiscoveryError, match="team 3 is not this account's team.*no team in league 1010101 is owned"):
        confirm_league(LeagueRef("nfl", REAL_LEAGUE_ID, REAL_SEASON, 3), stranger, now=NOW)


def test_confirm_league_settles_a_deep_walk_league_by_swid_not_by_its_url_hint(reads: FakeReads) -> None:
    walked = DiscoveredLeague("nfl", REAL_LEAGUE_ID, REAL_SEASON, team_hint=3)  # an opponent's page came first
    confirmed = confirm_league(walked.ref, SESSION, now=NOW)
    assert confirmed.team_id == 1 and confirmed.team_name == "Team 1"


def test_confirm_league_prefers_the_season_espn_reports(reads: FakeReads) -> None:
    confirmed = confirm_league(LeagueRef("nfl", REAL_LEAGUE_ID, 2025, 1), SESSION, now=NOW)
    assert reads.requests[0].url.path.startswith("/apis/v3/games/ffl/seasons/2025/")
    assert confirmed.season == REAL_SEASON  # the body says 2026


def test_confirm_league_without_settings_leaves_the_league_name_unset(reads: FakeReads) -> None:
    reads.answer = ok(merged_team_and_settings(with_settings=False))
    confirmed = confirm_league(LeagueRef("nfl", REAL_LEAGUE_ID), SESSION, now=NOW)
    assert confirmed.league_name is None and confirmed.team_name == "Team 1"


def test_confirm_league_uses_the_client_factory(reads: FakeReads, tmp_path: Path) -> None:
    calls: list[tuple[Game, int, int, EspnSession]] = []

    def factory(game: Game, league_id: int, season: int, session: EspnSession) -> EspnClient:
        calls.append((game, league_id, season, session))
        return EspnClient(game, league_id, season, session, cache_root=tmp_path / "cache", capture=False)

    confirmed = confirm_league(LeagueRef("nba", REAL_LEAGUE_ID, 2027), SESSION, client_factory=factory, now=NOW)
    assert calls == [(Game.FBA, REAL_LEAGUE_ID, 2027, SESSION)]
    assert confirmed.sport == "nba" and confirmed.team_id == 1
    assert not (tmp_path / "cache").exists()


def test_confirm_league_rejects_a_team_not_in_the_league(reads: FakeReads) -> None:
    with pytest.raises(DiscoveryError, match="no team 42 in the league"):
        confirm_league(LeagueRef("nfl", REAL_LEAGUE_ID, REAL_SEASON, 42), SESSION, now=NOW)


def test_confirm_league_needs_a_team_for_an_account_that_owns_none(reads: FakeReads) -> None:
    stranger = EspnSession(espn_s2=S2, swid="{00000000-0000-0000-0000-0000000000FF}", expires_at=None)
    with pytest.raises(DiscoveryError, match="no team in league 1010101 is owned by this account"):
        confirm_league(LeagueRef("nfl", REAL_LEAGUE_ID), stranger, now=NOW)


def test_confirm_league_wraps_http_and_auth_failures(reads: FakeReads) -> None:
    reads.answer = espn_error(404, "NOT_FOUND", "League not found")
    with pytest.raises(DiscoveryError, match=r"could not read NFL league 1010101 \(2026\).*HTTP 404"):
        confirm_league(LeagueRef("nfl", REAL_LEAGUE_ID), SESSION, now=NOW)
    reads.answer = espn_error(401, "AUTH_MISSING_CREDENTIALS", "no credentials")
    with pytest.raises(DiscoveryError, match="fm login"):
        confirm_league(LeagueRef("nfl", REAL_LEAGUE_ID), SESSION, now=NOW)


def test_confirm_league_wraps_a_body_that_does_not_parse(reads: FakeReads) -> None:
    reads.answer = ok(json.dumps({"id": REAL_LEAGUE_ID, "teams": [{"id": "not-a-number"}]}).encode())
    with pytest.raises(DiscoveryError, match="team view did not parse"):
        confirm_league(LeagueRef("nfl", REAL_LEAGUE_ID), SESSION, now=NOW)
