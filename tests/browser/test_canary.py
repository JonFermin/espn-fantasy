"""The selector canary (ROADMAP #28): every registered selector resolves on its page, every read view still parses,
and drift becomes an alert payload.

Everything runs offline. Pages are :class:`fm.browser.fakes.FakePage` trees built from the selector registry's own
shape (a table of rows, each a slot cell, a player cell and MOVE); read views come from the real-league fixtures in
``tests/fixtures/espn/real`` through the fake read API. Nothing sends: the alert tests inspect the message and the
payload, and the command tests swap the phone channel for a recorder.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from fm import paths
from fm.browser import canary, selectors
from fm.browser.canary import (
    DRIFT_KINDS,
    PAGE_ADDRESSES,
    CanaryReport,
    FindingKind,
    check_selectors,
    check_views,
    drift_alert,
    drift_payload,
    run_canary,
)
from fm.browser.fakes import FakeBrowser, FakeElement, FakeEspnApi, FakePage
from fm.browser.selectors import Selector, WebPage
from fm.commands import canary as canary_cmd
from fm.config import League, load_config
from fm.espn.client import EspnClient
from fm.executor import Runtime
from fm.executor.transport import RefusingTransport
from fm.notify import Message

REAL = Path(__file__).resolve().parents[1] / "fixtures" / "espn" / "real"
SAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "fixtures" / "config.sample.toml"
LEAGUE = League(key="nfl", sport="nfl", espn_league_id=1010101, season=2026, team_id=1)
ROSTER_URL = selectors.team_page_url("ffl", 1010101, 1, 2026)
VIEWS = (
    "mSettings",
    "mTeam+mStandings",
    "mRoster",
    "mMatchup",
    "mMatchupScore+mScoreboard",
    "kona_player_info",
    "mPendingTransactions",
    "mTransactions2",
    "proTeamSchedules_wl",
)
runner = CliRunner()


# --- fake pages -------------------------------------------------------------------------------------------------------


def roster_row(label: str, player: str | None, *, move: bool = True) -> FakeElement:
    """One lineup row as the registry describes it: the slot cell, the player cell (or Empty), MOVE when unlocked."""
    row = FakeElement(role="row").add(
        FakeElement(role="cell", name=label, text=label),
        FakeElement(role="cell", text=player or "Empty"),
    )
    if move and player is not None:
        row.add(FakeElement(role="button", name=f"Select {player} to move"))
    return row


def healthy_screen() -> list[FakeElement]:
    return [FakeElement(role="table").add(roster_row("QB", "Josh Allen"), roster_row("RB", "Bijan Robinson"))]


def page_showing(*screen: FakeElement, url: str = ROSTER_URL) -> FakePage:
    return FakePage(screens={url: list(screen)})


def players_screen(*, add: bool = True, status_filter: bool = True) -> list[FakeElement]:
    """The player list: the status filter, and an Add button for a free agent when there is one on the page."""
    if status_filter:
        screen = [FakeElement(role="combobox", name="Status")]
    else:
        screen = [FakeElement(role="heading", name="Players")]
    if add:
        screen.append(FakeElement(role="button", name="Add Some Player QB for Team"))
    return screen


def roster_fix_screen(*, cancel: bool = True, proceed: bool = True) -> list[FakeElement]:
    """The roster-fix page: Cancel, Continue and a DROP button per droppable player."""
    screen = [FakeElement(role="button", name="Drop Player Some Starter")]
    if cancel:
        screen.append(FakeElement(role="button", name="Cancel"))
    if proceed:
        screen.append(FakeElement(role="button", name="Continue"))
    return screen


def trade_screen() -> list[FakeElement]:
    """The trade builder: its heading, a checkbox per rostered player, Continue and Cancel Trade."""
    return [
        FakeElement(role="heading", name="Propose Trade Team 2"),
        FakeElement(role="checkbox", name="Trade Some Starter"),
        FakeElement(role="button", name="Continue"),
        FakeElement(role="button", name="Cancel Trade"),
    ]


TRADE_URL = selectors.trade_builder_url("ffl", 1010101, 2026, 2, 1)


def fix_url(player_id: int, kind: str) -> str:
    return selectors.roster_fix_url("ffl", 1010101, 2026, 1, player_id, kind)


FIRST_POOL_PLAYER = 4241478  # the first entry of the recorded NFL pool, who is on waivers
PLAYERS_URL = selectors.players_page_url("ffl", 1010101, 1, 2026)
CLAIM_PROBE = canary.PageProbe(FIRST_POOL_PLAYER, selectors.RosterFixType.CLAIM)
ROSTER_ONLY = selectors.selectors_for(WebPage.ROSTER)


def site(**screens: list[FakeElement]) -> FakePage:
    """Every page of the registry, healthy unless a screen is replaced (``roster=``, ``players=``, ``fix=``,
    ``trade=``)."""
    return FakePage(
        screens={
            ROSTER_URL: screens.get("roster", healthy_screen()),
            PLAYERS_URL: screens.get("players", players_screen()),
            TRADE_URL: screens.get("trade", trade_screen()),
            fix_url(FIRST_POOL_PLAYER, "claim"): screens.get("fix", roster_fix_screen()),
        }
    )


def check(page: FakePage, **options: Any) -> canary.SelectorCheck:
    """The roster page's selectors (the registry's other pages have their own tests), unless a registry is given."""
    options.setdefault("registry", ROSTER_ONLY)
    return check_selectors(page, LEAGUE, **options)


def report_of(page: FakePage) -> CanaryReport:
    return run_canary(LEAGUE, reader=None, page=page, registry=ROSTER_ONLY)


# --- selectors --------------------------------------------------------------------------------------------------------


def test_a_healthy_roster_page_raises_nothing() -> None:
    page = page_showing(*healthy_screen())
    result = check(page)

    assert result.findings == ()
    assert result.clear == ("roster.login_required",)
    assert {"roster.table", "roster.row", "roster.slot_cell", "roster.move"} <= set(result.resolved)
    assert result.absent_sometimes == ("roster.empty_slot",)  # no open slot today: not a finding
    assert page.did("goto") == [ROSTER_URL]


def test_every_registered_selector_is_judged_or_explained() -> None:
    """Iterating the registry means a selector added later is covered, or skipped with a reason, never forgotten."""
    result = check(site(), registry=selectors.registered_selectors(), probe=CLAIM_PROBE)
    judged = set(result.resolved) | set(result.absent_sometimes) | set(result.clear)
    judged |= {skip.key for skip in result.skipped}
    judged |= {finding.subject for finding in result.findings}
    assert judged == {item.key for item in selectors.registered_selectors()}


def test_every_registered_page_has_an_address() -> None:
    """A page added to ``WebPage`` (free agents in #27, trades in #44) must be given an address in the canary, or its
    selectors would go unwatched."""
    registered = {item.page for item in selectors.registered_selectors()}
    assert registered <= set(PAGE_ADDRESSES), f"add an address to fm.browser.canary.PAGE_ADDRESSES for {registered}"
    assert set(WebPage) <= set(PAGE_ADDRESSES)


def test_a_missing_always_selector_is_drift_with_an_alert_payload() -> None:
    page = page_showing(FakeElement(role="heading", name="Roster"))  # ESPN stopped rendering a roster table
    report = report_of(page)

    assert not report.ok
    table = next(finding for finding in report.drift if finding.subject == "roster.table")
    assert table.kind is FindingKind.MISSING and table.is_drift
    assert (table.page, table.expected, table.url) == (WebPage.ROSTER, "always", ROSTER_URL)
    assert ROSTER_URL in table.detail and "the roster table" in table.detail

    message = drift_alert([report])
    assert isinstance(message, Message)
    assert message.priority == "high" and message.link == ROSTER_URL
    assert message.title.startswith("ESPN drift: ")
    assert "nfl: [roster] roster.table (always): missing:" in message.body

    payload = drift_payload([report])
    assert json.loads(json.dumps(payload)) == payload  # plain JSON
    assert payload["drift"] is True and payload["ok"] is False
    [league] = payload["leagues"]
    assert league["league"] == "nfl"
    assert {
        "kind": "missing",
        "drift": True,
        "subject": "roster.table",
        "page": "roster",
        "expected": "always",
        "detail": table.detail,
        "url": ROSTER_URL,
    } in league["findings"]


def test_a_missing_inner_selector_is_named_on_its_own() -> None:
    page = page_showing(FakeElement(role="table").add(FakeElement(role="row").add(FakeElement(text="Josh Allen"))))
    report = report_of(page)

    assert [finding.subject for finding in report.drift] == ["roster.slot_cell"]  # the table and rows are fine
    assert drift_alert([report]) is not None


def test_a_sometimes_selector_that_is_absent_raises_no_alert() -> None:
    # An all-locked lineup with no open slot: no MOVE button, no Empty text, and that is a healthy page.
    locked = FakeElement(role="table").add(roster_row("QB", "Josh Allen", move=False))
    report = report_of(page_showing(locked))

    assert report.ok and report.findings == ()
    assert report.selector_check is not None
    assert {"roster.move", "roster.empty_slot"} <= set(report.selector_check.absent_sometimes)
    assert drift_alert([report]) is None


def test_a_sometimes_selector_that_is_present_is_only_observed() -> None:
    table = FakeElement(role="table").add(roster_row("QB", None), roster_row("RB", "Bijan Robinson"))
    result = check(page_showing(table))

    assert result.findings == () and {"roster.empty_slot", "roster.move"} <= set(result.resolved)


def test_a_never_marker_is_drift_and_does_not_flood_the_report() -> None:
    page = page_showing(FakeElement(role="heading", name="Log in Required"))
    result = check(page)

    assert [(finding.kind, finding.subject) for finding in result.findings] == [
        (FindingKind.WARNING, "roster.login_required")
    ]
    assert result.findings[0].expected == "never" and result.findings[0].is_drift
    # The other selectors are not blamed for a page that is not the team page.
    skipped = {skip.key: skip.reason for skip in result.skipped}
    assert "roster.table" in skipped and "warning marker" in skipped["roster.table"]


def test_selectors_that_need_a_click_are_skipped_and_nothing_is_clicked() -> None:
    page = page_showing(*healthy_screen())
    result = check(page)

    skipped = {skip.key: skip.reason for skip in result.skipped}
    assert "roster.here" in skipped and "never clicks" in skipped["roster.here"]
    assert page.did("click") == [] and page.did("fill") == [] and page.did("press") == []
    assert {kind for kind, _ in page.actions} <= {"goto"}


def test_an_always_selector_that_needs_a_click_is_not_judged() -> None:
    registry = (
        Selector("roster.table", WebPage.ROSTER, "the table", role="table"),
        Selector("roster.confirm", WebPage.ROSTER, "a confirm", role="button", name="Confirm", after="roster.table"),
    )
    result = check(page_showing(*healthy_screen()), registry=registry)

    assert result.findings == ()
    assert [skip.key for skip in result.skipped] == ["roster.confirm"]


def test_an_always_selector_inside_an_absent_container_is_not_blamed_twice() -> None:
    registry = (
        Selector("roster.table", WebPage.ROSTER, "the table", role="table"),
        Selector("roster.row", WebPage.ROSTER, "a row", role="row", within="roster.table"),
        Selector("roster.cell", WebPage.ROSTER, "a cell", role="cell", within="roster.row"),
    )
    result = check(page_showing(FakeElement(role="heading", name="Nothing here")), registry=registry)

    assert [finding.subject for finding in result.findings] == ["roster.table"]
    assert [skip.key for skip in result.skipped] == ["roster.row", "roster.cell"]


def test_a_page_that_does_not_load_could_not_be_checked() -> None:
    page = page_showing(*healthy_screen())
    page.goto_errors[ROSTER_URL] = TimeoutError("net::ERR_TIMED_OUT")
    report = report_of(page)

    [finding] = report.findings
    assert finding.kind is FindingKind.PAGE_FAILED and not finding.is_drift
    assert report.drift == () and report.failures == (finding,)
    message = drift_alert([report])
    assert message is not None and "could not check" in message.title and message.link == ROSTER_URL


def test_a_registered_page_without_an_address_is_a_finding() -> None:
    result = check(page_showing(*healthy_screen()), addresses={})

    [finding] = result.findings
    assert finding.kind is FindingKind.NO_ADDRESS and finding.page is WebPage.ROSTER
    assert "PAGE_ADDRESSES" in finding.detail
    assert FindingKind.NO_ADDRESS not in DRIFT_KINDS


def test_a_locator_that_raises_is_reported_not_raised() -> None:
    page = page_showing(*healthy_screen())
    original = page.goto

    def close_after_goto(url: str, **kwargs: Any) -> None:
        original(url, **kwargs)
        page.closed = True  # the page dies under the locators

    page.goto = close_after_goto  # type: ignore[method-assign]
    result = check(page)

    assert result.findings and {finding.kind for finding in result.findings} == {FindingKind.LOCATOR_ERROR}


def test_the_player_list_selectors_that_are_always_there_are_checked() -> None:
    players = selectors.selectors_for(WebPage.PLAYERS)
    healthy = check(site(), registry=players)
    assert healthy.findings == () and "players.status_filter" in healthy.resolved

    broken = site(players=players_screen(status_filter=False))  # ESPN dropped the status filter
    report = run_canary(LEAGUE, reader=None, page=broken, registry=players)
    [finding] = report.drift
    assert (finding.kind, finding.subject, finding.page, finding.expected) == (
        FindingKind.MISSING,
        "players.status_filter",
        WebPage.PLAYERS,
        "always",
    )
    assert finding.url == PLAYERS_URL and broken.did("goto") == [PLAYERS_URL]
    message = drift_alert([report])
    assert message is not None and message.link == PLAYERS_URL and "[players] players.status_filter" in message.body


def test_the_player_list_without_an_add_button_is_still_healthy() -> None:
    result = check(site(players=players_screen(add=False)), registry=selectors.selectors_for(WebPage.PLAYERS))

    assert result.findings == ()
    assert {"players.add", "players.claim"} <= set(result.absent_sometimes)


def test_the_roster_fix_page_is_opened_for_a_player_the_pool_showed() -> None:
    registry = selectors.selectors_for(WebPage.ROSTERFIX)
    page = site()
    result = check(page, registry=registry, probe=CLAIM_PROBE)

    assert page.did("goto") == [fix_url(FIRST_POOL_PLAYER, "claim")]
    assert result.findings == () and {"rosterfix.cancel", "rosterfix.continue"} <= set(result.resolved)
    skipped = {skip.key for skip in result.skipped}
    assert {"rosterfix.confirm_dialog", "rosterfix.confirm"} <= skipped  # they appear only after Continue is clicked
    assert page.did("click") == []

    free_agent = canary.PageProbe(77, selectors.RosterFixType.ADD)
    other = FakePage(screens={fix_url(77, "add"): roster_fix_screen(proceed=False)})
    broken = check(other, registry=registry, probe=free_agent)
    assert [(f.kind, f.subject) for f in broken.findings] == [(FindingKind.MISSING, "rosterfix.continue")]
    assert other.did("goto") == [fix_url(77, "add")]


def test_the_roster_fix_page_needs_a_player_to_open_it_with() -> None:
    page = site()
    result = check(page, registry=selectors.selectors_for(WebPage.ROSTERFIX))

    [finding] = result.findings
    assert finding.kind is FindingKind.PAGE_FAILED and not finding.is_drift and finding.page is WebPage.ROSTERFIX
    assert page.did("goto") == []


def test_the_probe_is_the_first_free_agent_else_the_first_waiver_player() -> None:
    assert check_views(reader(api_for())).probe == CLAIM_PROBE  # the recorded NFL pool is all waivers
    api = api_for()
    pool = json.loads((REAL / "ffl" / "kona_player_info.json").read_text(encoding="utf-8"))
    pool["players"][2]["status"] = "FREEAGENT"
    api.serve("kona_player_info", pool)
    assert check_views(reader(api)).probe == canary.PageProbe(pool["players"][2]["id"], selectors.RosterFixType.ADD)


# --- read views -------------------------------------------------------------------------------------------------------


def api_for(game: str = "ffl") -> FakeEspnApi:
    api = FakeEspnApi()
    for view in VIEWS:
        api.serve_file(view, REAL / game / f"{view}.json")
    return api


def reader(api: FakeEspnApi, game: str = "ffl", league_id: int = 1010101, season: int = 2026) -> EspnClient:
    return api.client(game, league_id, season)


def test_every_recorded_read_view_parses() -> None:
    api = api_for()
    result = check_views(reader(api))

    assert result.findings == () and result.skipped == ()
    assert result.parsed == VIEWS
    assert {view_key for view_key in VIEWS} == {request_view(request) for request in api.requests}


def request_view(request: Any) -> str:
    return "+".join(request.url.params.get_list("view"))


def test_a_read_view_that_fails_to_parse_is_drift() -> None:
    api = api_for()
    broken = json.loads((REAL / "ffl" / "mRoster.json").read_text(encoding="utf-8"))
    broken["teams"] = "ESPN renamed this"
    api.serve("mRoster", broken)
    result = check_views(reader(api))

    [finding] = result.findings
    assert (finding.kind, finding.subject, finding.expected, finding.page) == (
        FindingKind.UNPARSEABLE,
        "mRoster",
        "parses",
        None,
    )
    assert finding.is_drift and "teams" in finding.detail
    assert "mRoster" not in result.parsed and len(result.parsed) == len(VIEWS) - 1  # the rest still ran

    report = run_canary(LEAGUE, reader=reader(api), page=None)
    assert not report.ok and report.drift == (finding,)
    message = drift_alert([report])
    assert message is not None and "nfl: [view] mRoster (parses): unparseable:" in message.body
    assert message.link is None  # a view has no page to open
    assert drift_payload([report])["leagues"][0]["findings"][0]["page"] is None


def test_a_league_of_the_wrong_sport_is_drift_in_settings() -> None:
    result = check_views(reader(api_for("fba")))  # NBA answers read as an NFL league

    assert FindingKind.UNPARSEABLE in {finding.kind for finding in result.findings}
    assert any(finding.subject == "mSettings" for finding in result.findings)


def test_a_view_ESPN_will_not_answer_is_a_failure_not_drift() -> None:
    api = api_for()
    api.fail(500, view="mMatchup")
    result = check_views(reader(api))

    [finding] = result.findings
    assert finding.kind is FindingKind.VIEW_FAILED and finding.subject == "mMatchup" and not finding.is_drift
    assert "mMatchup" not in result.parsed
    api.fail(500, view="mMatchup")
    report = run_canary(LEAGUE, reader=reader(api), page=None)
    assert report.drift == () and not report.ok and [f.kind for f in report.failures] == [FindingKind.VIEW_FAILED]


def test_a_season_with_no_matchup_period_skips_the_scoreboard_with_a_reason() -> None:
    api = api_for()
    roster = json.loads((REAL / "ffl" / "mRoster.json").read_text(encoding="utf-8"))
    roster.get("status", {}).pop("currentMatchupPeriod", None)
    api.serve("mRoster", roster)
    api.serve("mMatchup", {**json.loads((REAL / "ffl" / "mMatchup.json").read_text(encoding="utf-8")), "schedule": []})
    result = check_views(reader(api))

    assert [skip.key for skip in result.skipped] == ["mMatchupScore+mScoreboard"]
    assert result.findings == ()


# --- the run and the alert --------------------------------------------------------------------------------------------


def test_a_clean_run_has_no_alert() -> None:
    report = run_canary(LEAGUE, reader=reader(api_for()), page=site())

    assert report.ok and report.findings == ()
    assert drift_alert([report]) is None
    payload = drift_payload([report])
    assert payload["ok"] is True and payload["drift"] is False
    assert payload["leagues"][0]["selectors"]["resolved"] and payload["leagues"][0]["views"]["parsed"] == list(VIEWS)


def test_one_alert_lists_every_league_and_caps_the_lines() -> None:
    nba = League(key="nba", sport="nba", espn_league_id=2020202, season=2027, team_id=1)
    reports = [
        report_of(page_showing(FakeElement(role="heading", name="Roster"))),
        run_canary(nba, reader=None, page=page_showing(url=selectors.team_page_url("fba", 2020202, 1, 2027))),
    ]
    message = drift_alert(reports)

    assert message is not None
    assert "nfl: [roster] roster.table" in message.body and "nba: [roster] roster.table" in message.body

    many = [report_of(page_showing(FakeElement(role="heading", name="Roster"))) for _ in range(12)]
    capped = drift_alert(many)
    assert capped is not None and len(capped.body.splitlines()) == canary.ALERT_LINES + 1
    assert capped.body.splitlines()[-1].startswith("... and ")
    assert len(drift_payload(many)["leagues"]) == 12  # the payload keeps everything


# --- fm canary --------------------------------------------------------------------------------------------------------


class Recorder:
    """A phone channel that records what it is asked to send."""

    name = "recorder"

    def __init__(self) -> None:
        self.sent: list[Message] = []
        self.closed = False

    def send(self, message: Message) -> None:
        self.sent.append(message)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def configured() -> None:
    directory = paths.config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.toml").write_text(SAMPLE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    (directory / ".env").write_text("ANTHROPIC_API_KEY=placeholder\nNTFY_TOPIC=a\nNTFY_REPLY_TOPIC=b\n", "utf-8")


@pytest.fixture
def phone(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(canary_cmd, "open_channel", lambda config: recorder)
    return recorder


def cli() -> typer.Typer:
    root = typer.Typer()
    root.callback()(lambda: None)  # keep it a group, as fm.cli does
    canary_cmd.register(root)
    return root


def sample_league(key: str) -> League:
    return load_config().league(key)


def live_runtime(monkeypatch: pytest.MonkeyPatch, page: FakePage, api: FakeEspnApi) -> None:
    """Swap the live browser opener for a fake page and a fake read API."""

    @contextmanager
    def runtime(row: Any, *, dry_run: bool) -> Iterator[Runtime]:
        assert dry_run, "the canary must open its runtime as a dry run: every non-GET to ESPN is aborted"
        yield Runtime(
            reader=api.client("ffl", row.espn_league_id, row.season),
            transport=RefusingTransport("dry run"),
            browser=FakeBrowser(page),
        )

    monkeypatch.setattr(canary_cmd, "live_opener", lambda options: runtime)


def nfl_site() -> FakePage:
    """Every page of the sample config's NFL league (not the ids the helpers above use)."""
    league = sample_league("nfl")
    league_id, team, season = league.espn_league_id, league.team_id, league.season
    return FakePage(
        screens={
            selectors.team_page_url("ffl", league_id, team, season): healthy_screen(),
            selectors.players_page_url("ffl", league_id, team, season): players_screen(),
            selectors.roster_fix_url("ffl", league_id, season, team, FIRST_POOL_PLAYER, "claim"): roster_fix_screen(),
            selectors.trade_builder_url("ffl", league_id, season, 1 if team != 1 else 2, team): trade_screen(),
        }
    )


@pytest.mark.usefixtures("configured")
def test_fm_canary_help() -> None:
    result = runner.invoke(cli(), ["canary", "--help"])
    assert result.exit_code == 0 and "--fixtures" in result.output and "--no-alert" in result.output


@pytest.mark.usefixtures("configured")
def test_fm_canary_reads_recorded_views_without_a_browser_or_an_alert(phone: Recorder) -> None:
    result = runner.invoke(cli(), ["canary", "--league", "nfl", "--fixtures", str(REAL / "ffl")])

    assert result.exit_code == 0, result.output
    assert "nfl: selectors not checked (no browser)" in result.output
    assert f"nfl: {len(VIEWS)} views parsed, 0 skipped" in result.output and "nfl: ok" in result.output
    assert phone.sent == []


@pytest.mark.usefixtures("configured")
def test_fm_canary_fixtures_that_do_not_parse_fail_but_never_alert(phone: Recorder) -> None:
    result = runner.invoke(cli(), ["canary", "--league", "nba", "--fixtures", str(REAL / "ffl")])

    assert result.exit_code == 1
    assert "nba: DRIFT [view] mSettings (parses): unparseable:" in result.output
    assert phone.sent == []  # recorded views never alert


@pytest.mark.usefixtures("configured")
def test_a_live_run_alerts_on_drift(monkeypatch: pytest.MonkeyPatch, phone: Recorder) -> None:
    page = page_showing(FakeElement(role="heading", name="Roster"), url=nfl_page_url())
    live_runtime(monkeypatch, page, api_for())
    result = runner.invoke(cli(), ["canary", "--league", "nfl"])

    assert result.exit_code == 1, result.output
    assert "nfl: DRIFT [roster] roster.table (always): missing:" in result.output
    [message] = phone.sent
    assert message.title.startswith("ESPN drift: ") and "roster.table (always)" in message.body
    assert phone.closed and "alert sent" in result.output
    assert page.closed  # the canary's page is closed afterwards
    assert page.did("click") == []


def nfl_page_url() -> str:
    league = sample_league("nfl")
    return selectors.team_page_url("ffl", league.espn_league_id, league.team_id, league.season)


@pytest.mark.usefixtures("configured")
def test_a_clean_live_run_sends_nothing(monkeypatch: pytest.MonkeyPatch, phone: Recorder) -> None:
    live_runtime(monkeypatch, nfl_site(), api_for())
    result = runner.invoke(cli(), ["canary", "--league", "nfl"])

    assert result.exit_code == 0, result.output
    assert "nfl: ok" in result.output and "selectors resolved" in result.output
    assert phone.sent == []


@pytest.mark.usefixtures("configured")
def test_no_alert_keeps_a_failing_run_quiet(monkeypatch: pytest.MonkeyPatch, phone: Recorder) -> None:
    page = page_showing(FakeElement(role="heading", name="Roster"), url=nfl_page_url())
    live_runtime(monkeypatch, page, api_for())
    result = runner.invoke(cli(), ["canary", "--league", "nfl", "--no-alert"])

    assert result.exit_code == 1 and phone.sent == []


@pytest.mark.usefixtures("configured")
def test_an_unknown_league_key_is_refused(phone: Recorder) -> None:
    result = runner.invoke(cli(), ["canary", "--league", "nope", "--fixtures", str(REAL / "ffl")])

    assert result.exit_code == 1 and "error:" in result.output and phone.sent == []


def test_the_command_is_discovered_by_the_root_cli() -> None:
    from fm.cli import app

    result = runner.invoke(app, ["canary", "--help"])
    assert result.exit_code == 0 and "selector" in result.output


def test_the_canary_module_has_no_write_path() -> None:
    """The canary only loads pages and reads views: nothing it imports can click or send."""
    source = Path(canary.__file__).read_text(encoding="utf-8")
    for forbidden in (
        ".click(",
        ".fill(",
        ".press(",
        ".check(",
        ".select_option(",
        "WriteTransport",
        "PlaywrightTransport",
    ):
        assert forbidden not in source
