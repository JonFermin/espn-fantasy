"""NBA injuries adapter against a trimmed ESPN capture and a hand-built official-report PDF (respx; no network).

tests/fixtures/sources/nba_injuries/espn_injuries.json was captured on 2026-10-04 (preseason) and cut to three teams and
eight players, with link lists, logos and news trimmed. The PDF carries the league report's layout (one text line per
printed row) so the line parser is exercised through pdfplumber.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from fm.espn.ids import InjuryStatus
from fm.sources.base import RateLimiter, SourceSchemaError, SourceUnavailable
from fm.sources.nba_injuries import (
    ESPN_INJURIES_URL,
    NbaInjuriesSource,
    NbaInjury,
    OfficialInjuryEntry,
    espn_athlete_id,
    parse_espn_injuries,
    parse_report_lines,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sources" / "nba_injuries"
T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
REPORT_DAY = date(2026, 10, 21)
REPORT_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-10-21_05PM.pdf"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def ok(name: str, content_type: str) -> httpx.Response:
    return httpx.Response(200, content=fixture(name), headers={"content-type": content_type})


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


class Routes:
    def __init__(self, router: respx.MockRouter) -> None:
        self.espn = router.get(ESPN_INJURIES_URL).mock(return_value=ok("espn_injuries.json", "application/json"))
        self.report = router.get(REPORT_URL).mock(
            return_value=ok("Injury-Report_2026-10-21_05PM.pdf", "application/pdf")
        )


@pytest.fixture
def routes() -> Iterator[Routes]:
    with respx.mock(assert_all_called=False) as router:
        yield Routes(router)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def source(routes: Routes, clock: FakeClock, tmp_path: Path) -> Iterator[NbaInjuriesSource]:
    with NbaInjuriesSource(
        cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock, sleep=lambda _: None
    ) as src:
        yield src


# --- ESPN injuries ---


def test_espn_injuries_are_typed(source: NbaInjuriesSource) -> None:
    result = source.espn()
    injuries = result.data
    assert len(injuries) == 8 and result.degraded is False
    assert [injury.name for injury in injuries][:2] == ["Henri Veesaar", "Mouhamed Gueye"]

    veesaar = injuries[0]
    assert (veesaar.espn_id, veesaar.short_name, veesaar.position) == (5105571, "H. Veesaar", "C")
    assert (veesaar.espn_team_id, veesaar.team_abbr, veesaar.team_name, veesaar.tricode) == (
        1,
        "ATL",
        "Atlanta Hawks",
        "ATL",
    )
    assert (veesaar.status, veesaar.status_code, veesaar.fantasy_status) == ("Out", "O", "OFS")
    assert veesaar.normalized is InjuryStatus.OUT and veesaar.out_for_season is True
    assert (veesaar.injury, veesaar.location, veesaar.side, veesaar.detail) == ("Knee", "Leg", "Right", None)
    assert veesaar.return_date == date(2027, 7, 1) and veesaar.news_id == "-57577"
    assert veesaar.updated == datetime(2026, 9, 21, 19, 50, tzinfo=UTC)
    assert veesaar.short_comment is not None and veesaar.short_comment.startswith("Veesaar was diagnosed")
    assert veesaar.long_comment is not None and len(veesaar.long_comment) > len(veesaar.short_comment)

    edey = next(injury for injury in injuries if injury.name == "Zach Edey")
    assert (edey.espn_id, edey.position, edey.espn_team_id, edey.tricode) == (4600663, "C", 29, "MEM")
    assert (edey.status, edey.status_code, edey.fantasy_status) == ("Day-To-Day", "DD", "GTD")
    assert edey.normalized is InjuryStatus.DAY_TO_DAY and edey.out_for_season is False
    assert (edey.injury, edey.location, edey.side, edey.detail) == ("Ankle", "Leg", "Left", "Surgery")
    assert edey.return_date == date(2026, 10, 5) and edey.updated == datetime(2026, 10, 2, 20, 16, tzinfo=UTC)


def test_athlete_team_wins_over_the_grouping_team(source: NbaInjuriesSource) -> None:
    # ESPN files Peavy under Memphis, but the athlete record says New Orleans; the player's own team is the truth,
    # and the nba.com tricode comes from the ESPN team id (ESPN abbreviates New Orleans as "NO").
    peavy = next(injury for injury in source.espn().data if injury.name == "Micah Peavy")
    assert (peavy.espn_team_id, peavy.team_abbr, peavy.team_name, peavy.tricode) == (
        3,
        "NO",
        "New Orleans Pelicans",
        "NOP",
    )


def test_espn_id_comes_from_the_link_when_there_is_no_headshot(source: NbaInjuriesSource) -> None:
    lopez = next(injury for injury in source.espn().data if injury.name == "Karim Lopez")
    assert lopez.espn_id == 5231919 and lopez.injury == "Undisclosed" and lopez.side is None
    brown = source.espn().data[-1]
    assert (brown.name, brown.espn_id, brown.tricode, brown.position) == ("Mikel Brown Jr.", 5101761, "BKN", "G")


def test_espn_athlete_id_fallbacks() -> None:
    link = {"links": [{"href": "https://www.espn.com/nba/player/_/id/4600663/zach-edey"}]}
    app_link = {"links": [{"href": "sportscenter://x-callback-url/showClubhouse?uid=s:40~l:46~a:4600663&section=bio"}]}
    headshot = {"links": [], "headshot": {"href": "https://a.espncdn.com/i/headshots/nba/players/full/4600663.png"}}
    uid = {"uid": "s:40~l:46~a:4600663"}
    assert espn_athlete_id(link) == espn_athlete_id(app_link) == espn_athlete_id(headshot) == espn_athlete_id(uid)
    assert espn_athlete_id(link) == 4600663
    assert espn_athlete_id({"links": [{"href": "https://www.espn.com/nba/team/_/name/mem"}]}) is None
    assert espn_athlete_id({"displayName": "Nobody"}) is None


def test_status_normalization_never_guesses() -> None:
    def status(raw: str) -> InjuryStatus:
        return NbaInjury(name="X", status=raw).normalized

    assert status("Day-To-Day") is InjuryStatus.DAY_TO_DAY and status("Out") is InjuryStatus.OUT
    assert status("Questionable") is InjuryStatus.QUESTIONABLE and status("Doubtful") is InjuryStatus.DOUBTFUL
    assert status("Probable") is InjuryStatus.PROBABLE and status("Suspension") is InjuryStatus.SUSPENSION
    assert status("Game Time Decision") is InjuryStatus.UNKNOWN
    blank = NbaInjury(name="X", status="Out", fantasy_status="", position=" ", team_abbr=None)
    assert (blank.fantasy_status, blank.position, blank.tricode, blank.out_for_season) == (None, None, None, False)


def test_espn_is_cached_for_fifteen_minutes_with_a_browser_ua(
    source: NbaInjuriesSource, routes: Routes, clock: FakeClock, tmp_path: Path
) -> None:
    first = source.espn()
    clock.advance(minutes=14)
    assert source.espn().cached is True and routes.espn.call_count == 1
    clock.advance(minutes=2)
    assert source.espn().cached is False and routes.espn.call_count == 2
    headers = routes.espn.calls.last.request.headers
    assert "Chrome" in headers["user-agent"] and "espn-fantasy" not in headers["user-agent"]
    raw = tmp_path / "cache" / "nba_injuries" / "espn" / "nba.json"
    assert first.raw_path == raw and raw.read_bytes() == fixture("espn_injuries.json")


def test_espn_failure_raises_without_a_copy_and_serves_stale_with_one(
    source: NbaInjuriesSource, routes: Routes, clock: FakeClock
) -> None:
    routes.espn.mock(return_value=httpx.Response(500))
    with pytest.raises(SourceUnavailable, match="HTTP 500"):
        source.espn()
    assert routes.espn.call_count == 3  # retried, then gave up
    routes.espn.mock(return_value=ok("espn_injuries.json", "application/json"))
    good = source.espn()
    clock.advance(minutes=16)
    routes.espn.mock(return_value=httpx.Response(503))
    stale = source.espn()
    assert (stale.stale, stale.as_of) == (True, T0) and stale.data == good.data and "HTTP 503" in stale.warnings[0]


def test_parse_rejects_bad_shapes_and_skips_bad_entries(caplog: pytest.LogCaptureFixture) -> None:
    for payload in (b"[]", b'{"injuries": "x"}', b'{"timestamp": "t"}'):
        with pytest.raises(SourceSchemaError, match="expected an object with an injuries list"):
            parse_espn_injuries(payload)
    assert parse_espn_injuries(b'{"injuries": []}') == []
    assert parse_espn_injuries(b'{"injuries": [{"id": "1", "displayName": "Atlanta Hawks", "injuries": []}]}') == []
    with pytest.raises(SourceSchemaError, match="none of 2 entries validated"):
        parse_espn_injuries(b'{"injuries": [{"id": "1", "injuries": [{"status": "Out"}, "junk"]}]}')
    good = {"status": "Out", "athlete": {"displayName": "A B", "team": {"id": "1", "abbreviation": "ATL"}}}
    payload = {"injuries": [{"id": "1", "displayName": "Atlanta Hawks", "injuries": [good, {"status": None}]}, 5]}
    with caplog.at_level(logging.WARNING, logger="fm.sources.nba_injuries"):
        (injury,) = parse_espn_injuries(httpx.Response(200, json=payload).content)
    assert (injury.name, injury.espn_id, injury.tricode, injury.team_name) == ("A B", None, "ATL", "Atlanta Hawks")
    assert "skipped 2 entries" in caplog.text


# --- official report PDF ---


def test_official_report_pdf_parses_player_rows(source: NbaInjuriesSource, routes: Routes, tmp_path: Path) -> None:
    result = source.official_report(REPORT_DAY)
    report = result.data
    assert result.degraded is False and result.dataset == "official" and result.key == "2026-10-21_05PM"
    assert len(report.entries) == 6
    smith = report.entries[0]
    assert smith == OfficialInjuryEntry(
        REPORT_DAY,
        "07:30 (ET)",
        "HOU@OKC",
        "Houston Rockets",
        "Smith Jr., Jabari",
        "Out",
        "Injury/Illness - Left Ankle; Sprain",
    )
    assert smith.name == "Jabari Smith Jr."
    thunder = report.for_team("Oklahoma City Thunder")
    assert [(entry.player, entry.status) for entry in thunder] == [
        ("Caruso, Alex", "Questionable"),
        ("Hartenstein, Isaiah", "Available"),
        ("Gilgeous-Alexander, Shai", "Probable"),
    ]
    assert {(entry.game_day, entry.game_time, entry.matchup) for entry in thunder} == {
        (REPORT_DAY, "07:30 (ET)", "HOU@OKC")
    }
    lakers = report.for_team("Los Angeles Lakers")
    assert [(entry.player, entry.status, entry.reason) for entry in lakers] == [
        ("Doncic, Luka", "Doubtful", "Injury/Illness - Left Calf; Strain"),
        ("James, LeBron", "Out", "Not With Team"),
    ]
    assert {(entry.game_time, entry.matchup) for entry in lakers} == {("10:00 (ET)", "GSW@LAL")}
    assert report.not_submitted == (("GSW@LAL", "Golden State Warriors"),)
    assert [entry.name for entry in lakers] == ["Luka Doncic", "LeBron James"]

    raw = tmp_path / "cache" / "nba_injuries" / "official" / "2026-10-21_05PM.pdf"
    assert result.raw_path == raw and raw.read_bytes() == fixture("Injury-Report_2026-10-21_05PM.pdf")
    assert routes.report.calls.last.request.headers["accept"] == "application/pdf"


def test_report_lines_forward_fill_and_join_wrapped_reasons() -> None:
    lines = [
        "Injury Report: 10/21/26 05:00 PM",
        "Game Date Game Time Matchup Team Player Name Current Status Reason",
        "10/21/2026 07:30 (ET) HOU@OKC Houston Rockets Smith Jr., Jabari Out Injury/Illness - Left Ankle;",
        "Sprain",
        "Sengun, Alperen Available",
        "Oklahoma City Thunder NOT YET SUBMITTED",
        "10/21/2026 10:00 (ET) GSW@LAL Los Angeles Lakers James, LeBron Out Not With Team",
        "Page 1 of 1",
    ]
    report = parse_report_lines(lines)
    assert [(e.team, e.player, e.status, e.reason) for e in report.entries] == [
        ("Houston Rockets", "Smith Jr., Jabari", "Out", "Injury/Illness - Left Ankle; Sprain"),
        ("Houston Rockets", "Sengun, Alperen", "Available", ""),
        ("Los Angeles Lakers", "James, LeBron", "Out", "Not With Team"),
    ]
    assert report.entries[2].game_time == "10:00 (ET)" and report.entries[2].matchup == "GSW@LAL"
    assert report.not_submitted == (("HOU@OKC", "Oklahoma City Thunder"),)
    assert parse_report_lines([]).entries == ()


def test_official_report_degrades_when_missing(
    source: NbaInjuriesSource, routes: Routes, caplog: pytest.LogCaptureFixture
) -> None:
    routes.report.mock(return_value=httpx.Response(404))
    with caplog.at_level(logging.WARNING, logger="fm.sources.nba_injuries"):
        result = source.official_report(REPORT_DAY)
    assert result.degraded is True and result.data.entries == () and result.data.not_submitted == ()
    assert "HTTP 404" in result.warnings[0] and "continuing without it" in caplog.text
    assert result.as_of == T0 and routes.report.call_count == 1


def test_official_report_rejects_a_non_pdf_and_keeps_the_good_copy(
    source: NbaInjuriesSource, routes: Routes, clock: FakeClock, tmp_path: Path
) -> None:
    routes.report.mock(return_value=httpx.Response(200, text="<html>maintenance</html>"))
    degraded = source.official_report(REPORT_DAY)
    assert degraded.degraded is True and "did not parse" in degraded.warnings[0]
    cache = tmp_path / "cache" / "nba_injuries" / "official"
    assert (cache / "2026-10-21_05PM.rejected.pdf").read_bytes() == b"<html>maintenance</html>"
    assert not (cache / "2026-10-21_05PM.pdf").exists()

    routes.report.mock(return_value=ok("Injury-Report_2026-10-21_05PM.pdf", "application/pdf"))
    good = source.official_report(REPORT_DAY)
    clock.advance(minutes=16)
    routes.report.mock(return_value=httpx.Response(200, text="<html>maintenance</html>"))
    stale = source.official_report(REPORT_DAY)
    assert (stale.stale, stale.degraded, stale.as_of) == (True, False, T0) and stale.data == good.data


def test_official_report_time_label_is_validated(source: NbaInjuriesSource) -> None:
    with pytest.raises(ValueError, match="time_label"):
        source.official_report(REPORT_DAY, time_label="5 PM")


def test_accidental_live_calls_are_not_swallowed(source: NbaInjuriesSource, routes: Routes) -> None:
    routes.report.mock(side_effect=RuntimeError("outbound network is disabled in unit tests"))
    with pytest.raises(RuntimeError, match="outbound network"):
        source.official_report(REPORT_DAY)


def test_ttls_follow_the_refresh_cadence() -> None:
    assert NbaInjuriesSource.ttl["espn"] == timedelta(minutes=15)
    assert NbaInjuriesSource.ttl["official"] == timedelta(minutes=15)
