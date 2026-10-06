"""The UI fallback drill (ROADMAP #41): each UI flow is walked on a page up to its final save and never past it.

Everything runs offline. The league is the recorded real NBA league (``tests/fixtures/espn/real/fba``, day 1: nobody
locked, a free-agent pool) served to the real read client by ``fm.browser.fakes``, and :class:`Site` is a fake of ESPN's
lineup editor and roster-fix page built from the accessible names the #14 capture saw. Every control that saves
(HERE, ``Confirm add ...``) counts its clicks on the :class:`Site`, so the tests can say that none was ever clicked.
Nothing sends: the runtime is the dry-run kind (a refusing transport), and the alert tests inspect the message and
the payload or swap the phone channel for a recorder.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import typer
from typer.testing import CliRunner

from fm import paths
from fm.browser import drills, selectors
from fm.browser.drills import (
    DrillError,
    DrillFailureKind,
    DrillPlan,
    DrillReport,
    DrillStatus,
    DrillUi,
    FinalSaveClickError,
    GuardedPage,
    NoDrillTargetError,
    RequestWatch,
    add_drop_target,
    claim_target,
    could_not_run_alert,
    drill_alert,
    drill_payload,
    final_save_probes,
    observed_names,
    read_views,
    run_drills,
    swap_target,
)
from fm.browser.fakes import FAKE_SWID, FakeBrowser, FakeElement, FakeEspnApi, FakePage, FakeTransport
from fm.browser.flows import (
    WRITES_HOST,
    DryRunStop,
    Flow,
    FlowContext,
    FlowRegistry,
    Mode,
    Preconditions,
    UiDriver,
    Verification,
    registered_flows,
)
from fm.browser.flows.add_drop import AddDrop
from fm.browser.flows.lineup import SetLineup
from fm.browser.flows.waiver import CancelWaiverClaim, ClaimWaiver
from fm.commands import drill as drill_cmd
from fm.config import League
from fm.espn.auth import AuthError
from fm.espn.ids import Game
from fm.executor import AuditLog, Runtime, UiSession, dry_run_block_reason
from fm.executor.transport import RefusingTransport
from fm.notify import Message
from fm.proposals import AddDropPayload, ProposalKind

REAL = Path(__file__).resolve().parents[1] / "fixtures" / "espn" / "real"
NOW = datetime(2026, 10, 19, 20, 0, tzinfo=UTC)
TEAM = 1
LEAGUES = {
    "fba": League(key="nba", sport="nba", espn_league_id=2020202, season=2027, team_id=TEAM),
    "ffl": League(key="nfl", sport="nfl", espn_league_id=1010101, season=2026, team_id=TEAM),
}
VIEWS = (
    "mRoster",
    "mSettings",
    "kona_player_info",
    "proTeamSchedules_wl",
    "mPendingTransactions",
    "mTransactions2",
    "mTeam+mStandings",
)
FAR_FUTURE_MS = int(datetime(2027, 1, 1, tzinfo=UTC).timestamp() * 1000)
WRITE_URL = f"https://{WRITES_HOST}/apis/v3/games/fba/seasons/2027/segments/0/leagues/2020202/transactions/"
runner = CliRunner()


def load(game: str, name: str) -> Any:
    return json.loads((REAL / game / f"{name}.json").read_text(encoding="utf-8"))


# --- the fake league and its site -------------------------------------------------------------------------------------


class Site(FakePage):
    """ESPN's team page (the lineup editor) and roster-fix page over the recorded league's roster and pool.

    ``MOVE`` (named ``Select <Player> to move``) opens the editor; the rows the picked player may go to then show
    ``HERE`` (``Confirm move of <Player> to <Slot>``, or ``Move`` on an empty row), which saves. The roster-fix page
    lists ``Drop Player <Name>`` per droppable player and ``Continue``, renamed ``Continue to add <A> and drop <D>``
    once one is picked, which opens the ``Confirm Transaction`` dialog with the button that saves. Every save lands in
    ``saves`` and every control that would save is in ``final_controls``."""

    def __init__(self, game: str) -> None:
        super().__init__()
        self.game = Game(game)
        self.league = LEAGUES[game]
        self.rosters: dict[str, Any] = load(game, "mRoster")
        self.settings: dict[str, Any] = load(game, "mSettings")
        self.cards: dict[int, dict[str, Any]] = {card["id"]: card for card in load(game, "kona_player_info")["players"]}
        for entry in self.entries():
            self.cards[entry["playerId"]] = entry["playerPoolEntry"]
        self.signed_in = True
        self.lacks: set[str] = set()
        """Controls the site fails to show: ``move``, ``here``, ``drop``, ``continue``, ``dialog``, ``confirm``."""
        self.picked: dict[str, Any] | None = None
        self.drop: int | None = None
        self.saves: list[str] = []
        self.final_controls: list[FakeElement] = []
        self.handlers: list[Any] = []

    # --- the league ---

    def entries(self) -> list[dict[str, Any]]:
        team = next(team for team in self.rosters["teams"] if team["id"] == TEAM)
        return team["roster"]["entries"]

    def entry(self, player_id: int) -> dict[str, Any]:
        return next(entry for entry in self.entries() if entry["playerId"] == player_id)

    def name(self, player_id: int) -> str:
        return self.cards[player_id]["player"]["fullName"]

    def lock_everyone(self) -> None:
        for entry in self.entries():
            entry["playerPoolEntry"]["lineupLocked"] = True

    def put_on_waivers(self, player_id: int) -> None:
        card = self.cards[player_id]
        card["status"], card["onTeamId"], card["waiverProcessDate"] = "WAIVERS", 0, FAR_FUTURE_MS

    def api(self) -> FakeEspnApi:
        api = FakeEspnApi()
        api.serve("mRoster", lambda _request: self.rosters)
        api.serve("mSettings", lambda _request: self.settings)
        api.serve("kona_player_info", lambda _request: {"players": list(self.cards.values())})
        api.serve("kona_playercard", self._serve_cards)
        for view in ("proTeamSchedules_wl", "mPendingTransactions", "mTransactions2", "mTeam+mStandings"):
            api.serve(view, load("fba", view if view != "mTransactions2" else "mTransactions2_waiver_trade"))
        return api

    def _serve_cards(self, request: Any) -> dict[str, Any]:
        header = json.loads(request.headers.get("x-fantasy-filter", "{}"))
        ids = header.get("players", {}).get("filterIds", {}).get("value", [])
        return {"players": [self.cards[player_id] for player_id in ids if player_id in self.cards]}

    def runtime(self, api: FakeEspnApi | None = None, *, transport: Any = None, dry_run: bool = True) -> Runtime:
        reader = (api or self.api()).client("fba", self.league.espn_league_id, self.league.season)
        return Runtime(
            reader=reader,
            transport=transport if transport is not None else RefusingTransport("dry run"),
            browser=FakeBrowser(self),
            member_id=FAKE_SWID,
            dry_run=dry_run,
        )

    # --- what a visit shows ---

    def goto(self, url: str, *, timeout: float | None = None, wait_until: str | None = None) -> None:
        self.log("goto", url)
        self._url = url
        parts = urlsplit(url)
        if not parts.hostname:
            self.elements = []
        elif not self.signed_in:
            self.elements = [FakeElement(role="heading", name="Log in Required")]
        elif parts.path.endswith("/team"):
            self.picked = None
            self.elements = self.render_team()
        elif parts.path.endswith("/rosterfix"):
            self.drop = None
            self.elements = self.render_fix(int(parse_qs(parts.query)["players"][0]), parse_qs(parts.query)["type"][0])
        else:
            self.elements = []

    def on(self, event: str, handler: Any) -> None:
        self.handlers.append((event, handler))

    def emit_request(self, method: str, url: str) -> None:
        for event, handler in self.handlers:
            if event == "request":
                handler(type("Request", (), {"method": method, "url": url})())

    def save(self, what: str) -> None:
        self.saves.append(what)
        self.emit_request("POST", WRITE_URL)

    def final(self, element: FakeElement) -> FakeElement:
        self.final_controls.append(element)
        return element

    @property
    def final_clicks(self) -> int:
        return sum(element.clicks for element in self.final_controls)

    # --- the team page ---

    def lineup_rows(self) -> list[tuple[int, dict[str, Any] | None]]:
        counts = self.settings["settings"]["rosterSettings"]["lineupSlotCounts"]
        rows: list[tuple[int, dict[str, Any] | None]] = []
        for raw_slot, count in counts.items():
            slot = int(raw_slot)
            holders = [entry for entry in self.entries() if entry["lineupSlotId"] == slot]
            rows.extend((slot, entry) for entry in holders)
            rows.extend((slot, None) for _ in range(count - len(holders)))
        return rows

    def render_team(self) -> list[FakeElement]:
        return [FakeElement(role="table").add(*(self.team_row(slot, entry) for slot, entry in self.lineup_rows()))]

    def team_row(self, slot: int, entry: dict[str, Any] | None) -> FakeElement:
        label = selectors.slot_label(self.game, slot)
        player = None if entry is None else entry["playerPoolEntry"]["player"]
        row = FakeElement(role="row").add(
            FakeElement(role="cell", name=label, text=label),
            FakeElement(role="cell", text="Empty" if player is None else player["fullName"]),
        )
        if entry is not None and player is not None and self.picked is None:
            if not entry["playerPoolEntry"].get("lineupLocked") and "move" not in self.lacks:
                row.add(
                    FakeElement(
                        role="button",
                        name=f"Select {player['fullName']} to move",
                        text="MOVE",
                        on_click=functools.partial(self.pick, entry),
                    )
                )
        elif self.picked is not None and entry is not self.picked and self.may_take(slot, entry):
            here = f"Confirm move of {player['fullName']} to {label}" if player is not None else "Move"
            if "here" not in self.lacks:
                row.add(
                    self.final(
                        FakeElement(
                            role="button", name=here, text="HERE", on_click=functools.partial(self.here, slot, entry)
                        )
                    )
                )
        return row

    def may_take(self, slot: int, entry: dict[str, Any] | None) -> bool:
        picked = self.picked
        assert picked is not None
        eligible = picked["playerPoolEntry"]["player"]["eligibleSlots"]
        if slot == picked["lineupSlotId"] or slot not in eligible:
            return False
        return entry is None or picked["lineupSlotId"] in entry["playerPoolEntry"]["player"]["eligibleSlots"]

    def pick(self, entry: dict[str, Any], page: FakePage) -> None:
        self.picked = entry
        page.show(*self.render_team())

    def here(self, slot: int, entry: dict[str, Any] | None, page: FakePage) -> None:
        self.save(f"move {self.picked['playerId'] if self.picked else '?'} to {slot}")

    # --- the roster-fix page ---

    def render_fix(self, add_id: int, kind: str) -> list[FakeElement]:
        elements: list[FakeElement] = []
        if "drop" not in self.lacks:
            for entry in self.entries():
                player = entry["playerPoolEntry"]["player"]
                if player.get("droppable", True):
                    elements.append(
                        FakeElement(
                            role="button",
                            name=f"Drop Player {player['fullName']}",
                            text="DROP",
                            on_click=functools.partial(self.pick_drop, entry["playerId"], add_id, kind),
                        )
                    )
        elements.append(FakeElement(role="button", name="Cancel", text="Cancel"))
        if "continue" in self.lacks:
            return elements
        if self.drop is None:
            elements.append(FakeElement(role="button", name="Continue", text="Continue"))
        else:
            label = f"{kind} {self.name(add_id)} and drop {self.name(self.drop)}"
            elements.append(
                FakeElement(
                    role="button",
                    name=f"Continue to {label}",
                    text="Continue",
                    on_click=functools.partial(self.open_dialog, add_id, kind),
                )
            )
        return elements

    def pick_drop(self, drop_id: int, add_id: int, kind: str, page: FakePage) -> None:
        self.drop = drop_id
        self.emit_request("GET", f"https://lm-api-reads.fantasy.espn.com/{kind}")  # an ordinary page read
        page.show(*self.render_fix(add_id, kind))

    def open_dialog(self, add_id: int, kind: str, page: FakePage) -> None:
        if "dialog" in self.lacks:
            return
        label = f"{kind} {self.name(add_id)} and drop {self.name(self.drop or 0)}"
        confirm = self.final(
            FakeElement(
                role="button",
                name=f"Confirm {label}",
                text="Confirm",
                on_click=lambda _page: self.save(label),
            )
        )
        dialog = FakeElement(role="dialog", name="Confirm Transaction")
        if "confirm" not in self.lacks:
            dialog.add(confirm)
        page.add(dialog.add(FakeElement(role="button", name="Cancel", text="Cancel")))


def nba_site(*, waivers: bool = True) -> Site:
    """The recorded NBA league; ``waivers`` puts the pool's last player on waivers so the claim has a target."""
    site = Site("fba")
    if waivers:
        site.put_on_waivers(list(site.cards)[3])
    return site


def real_registry() -> FlowRegistry:
    registry = FlowRegistry()
    for flow in (SetLineup(), AddDrop(), ClaimWaiver(), CancelWaiverClaim()):
        registry.register(flow)
    return registry


def drill(site: Site, *, tmp_path: Path, flows: list[str] | None = None, **options: Any) -> DrillReport:
    options.setdefault("registry", real_registry())
    return run_drills(
        LEAGUES["fba"],
        runtime=site.runtime(options.pop("api", None)),
        flows=flows,
        audit_root=tmp_path / "drills",
        now=NOW,
        **options,
    )


def only(report: DrillReport, flow: str) -> drills.DrillResult:
    return next(result for result in report.results if result.flow == flow)


# --- a healthy drill --------------------------------------------------------------------------------------------------


def test_a_healthy_drill_walks_every_ui_flow_to_its_confirm_and_no_further(tmp_path: Path) -> None:
    site = nba_site()
    report = drill(site, tmp_path=tmp_path)

    assert report.ok and report.findings == (), [finding.line() for finding in report.findings]
    assert [result.flow for result in report.results] == ["set_lineup", "add_drop", "claim_waiver"]  # cancel: API only
    assert all(result.status is DrillStatus.PASSED for result in report.results)
    lineup, add, claim = (only(report, name) for name in ("set_lineup", "add_drop", "claim_waiver"))
    assert lineup.reached is not None and lineup.reached.startswith("move ") and " for " in lineup.reached
    assert add.reached is not None and add.reached.startswith("add ") and " and drop " in add.reached
    assert claim.reached is not None and claim.reached.startswith("claim ") and " and drop " in claim.reached
    assert [click.split("[")[0] for click in lineup.clicks] == ["button"]  # MOVE, and nothing else
    assert [click.split("[")[0] for click in add.clicks] == ["button", "button"]  # Drop Player, Continue
    assert len(lineup.pages) == 1 and len(add.pages) == 1 and "rosterfix" in add.pages[0]
    assert "type=claim" in claim.pages[0]

    assert site.saves == [] and site.final_clicks == 0 and len(site.final_controls) >= 3
    assert site.final_controls and all(control.clicks == 0 for control in site.final_controls)
    assert drill_alert([report]) is None
    payload = drill_payload([report])
    assert payload["ok"] is True and payload["unsafe"] is False
    assert [item["status"] for item in payload["leagues"][0]["flows"]] == ["passed"] * 3


def test_each_walk_navigates_away_and_closes_its_page(tmp_path: Path) -> None:
    site = nba_site()
    runtime = site.runtime()
    run_drills(LEAGUES["fba"], runtime=runtime, registry=real_registry(), audit_root=tmp_path / "drills", now=NOW)
    browser = runtime.browser
    assert isinstance(browser, FakeBrowser) and len(browser.opened) == 3
    assert all(page.closed for page in browser.opened)
    gotos = site.did("goto")
    assert gotos.count("about:blank") == 3 and gotos[-1] == "about:blank"
    assert site.saves == []


def test_screenshots_go_to_the_drill_audit_folder(tmp_path: Path) -> None:
    site = nba_site()
    report = drill(site, tmp_path=tmp_path)

    names = {Path(name).name for result in report.results for name in result.artifacts}
    assert any(name.startswith("drill-set_lineup-") and "before-move" in name for name in names)
    assert any("drill-add_drop-" in name and "rosterfix" in name for name in names)
    assert all((tmp_path / "drills" / name).exists() for result in report.results for name in result.artifacts)
    assert all(Path(name).parts[0] == "2026-10-19" for result in report.results for name in result.artifacts)


def test_the_flows_the_drill_covers_are_the_registered_ones_with_a_ui_mode() -> None:
    ui_flows = {registration.flow.name for registration in registered_flows() if Mode.UI in registration.flow.modes}
    assert ui_flows == {"set_lineup", "add_drop", "claim_waiver"}
    assert ui_flows <= set(drills.PLANNERS)  # a new UI flow needs a planner here, or the drill reports it


def test_the_default_registry_is_the_process_wide_one(tmp_path: Path) -> None:
    site = nba_site()
    report = run_drills(LEAGUES["fba"], runtime=site.runtime(), audit_root=tmp_path / "drills", now=NOW)

    assert report.ok  # registered in module order: add_drop, set_lineup, claim_waiver
    assert {result.flow for result in report.results} == {"set_lineup", "add_drop", "claim_waiver"}


def test_flows_can_be_picked_by_name_and_an_unknown_name_is_refused(tmp_path: Path) -> None:
    site = nba_site()
    report = drill(site, tmp_path=tmp_path, flows=["add_drop"])
    assert [result.flow for result in report.results] == ["add_drop"] and report.ok

    with pytest.raises(DrillError, match=r"no UI flow cancel_waiver for nba; with a UI mode: set_lineup, add_drop"):
        drill(site, tmp_path=tmp_path, flows=["cancel_waiver"])


# --- targets ----------------------------------------------------------------------------------------------------------


def views_of(site: Site) -> drills.DrillViews:
    return read_views(LEAGUES["fba"], site.api().client("fba", 2020202, 2027))


def test_the_lineup_target_swaps_a_starter_with_an_eligible_bench_player() -> None:
    site = nba_site()
    plan = swap_target(views_of(site))

    first, second = plan.payload.moves  # type: ignore[attr-defined]
    held = {entry["playerId"]: entry["lineupSlotId"] for entry in site.entries()}
    assert (first.from_slot_id, second.from_slot_id) == (held[first.espn_id], held[second.espn_id])
    assert (first.to_slot_id, second.to_slot_id) == (second.from_slot_id, first.from_slot_id)
    assert second.from_slot_id == 12  # the bench: a starter and a bench player, as a real rotation is
    assert plan.names == (site.name(first.espn_id), site.name(second.espn_id))
    assert plan.description.startswith("swap ") and "(Bench)" in plan.description


def test_locked_players_are_never_moved() -> None:
    site = nba_site()
    site.lock_everyone()
    with pytest.raises(NoDrillTargetError, match="no two unlocked players"):
        swap_target(views_of(site))

    open_ones = [entry for entry in site.entries() if entry["lineupSlotId"] in (0, 12)][:2]
    for entry in open_ones:
        entry["playerPoolEntry"]["lineupLocked"] = False
    for entry in site.entries():
        if entry not in open_ones:
            entry["playerPoolEntry"]["lineupLocked"] = True
    try:
        plan = swap_target(views_of(site))
    except NoDrillTargetError:
        return  # these two cannot trade slots: nothing else is open, which is the point
    moved = {move.espn_id for move in plan.payload.moves}  # type: ignore[attr-defined]
    assert moved <= {entry["playerId"] for entry in open_ones}


def test_the_add_and_the_claim_targets_come_from_the_pool_and_the_roster() -> None:
    site = nba_site()
    views = views_of(site)
    add = add_drop_target(views)
    claim = claim_target(views)

    assert isinstance(add.payload, AddDropPayload) and add.payload.add_espn_id == 6589  # the first free agent
    on_roster = {entry["playerId"] for entry in site.entries()}
    assert add.payload.drop_espn_id in on_roster
    assert site.entry(add.payload.drop_espn_id)["lineupSlotId"] == 12  # a bench player first  # type: ignore[arg-type]
    assert claim.payload.add_espn_id == list(site.cards)[3]  # type: ignore[attr-defined]
    assert claim.payload.bid_amount is None  # type: ignore[attr-defined]
    assert claim.description.startswith("claim ") and add.description.startswith("add ")


def test_a_pool_without_the_right_player_skips_the_flow() -> None:
    views = views_of(nba_site(waivers=False))
    with pytest.raises(NoDrillTargetError, match="nobody on waivers"):
        claim_target(views)

    nfl = Site("ffl")  # the recorded NFL league is on the Monday of week 4: only Bijan Robinson has not played
    nfl_views = read_views(LEAGUES["ffl"], nfl_api(nfl).client("ffl", 1010101, 2026))
    with pytest.raises(NoDrillTargetError, match="no free agent"):  # and its recorded pool is all on waivers
        add_drop_target(nfl_views)
    with pytest.raises(NoDrillTargetError, match="no two unlocked players"):  # nobody else can be moved or dropped
        swap_target(nfl_views)
    with pytest.raises(NoDrillTargetError, match="no player on our roster can be dropped"):
        claim_target(nfl_views)


def nfl_api(site: Site) -> FakeEspnApi:
    api = site.api()
    for view in VIEWS:
        api.serve(view, load("ffl", view if view != "mTransactions2" else "mTransactions2_waiver_trade"))
    api.serve("kona_player_info", load("ffl", "kona_player_info"))
    return api


def test_a_flow_without_a_target_is_skipped_with_its_reason_and_raises_no_alert(tmp_path: Path) -> None:
    site = nba_site(waivers=False)
    report = drill(site, tmp_path=tmp_path)

    claim = only(report, "claim_waiver")
    assert claim.status is DrillStatus.SKIPPED and claim.skipped == "the pool read shows nobody on waivers"
    assert report.ok and drill_alert([report]) is None
    assert only(report, "set_lineup").status is DrillStatus.PASSED


def test_a_drill_that_checked_nothing_is_a_failure(tmp_path: Path) -> None:
    site = nba_site(waivers=False)
    site.lock_everyone()
    site.cards = {pid: card for pid, card in site.cards.items() if pid in {e["playerId"] for e in site.entries()}}
    report = drill(site, tmp_path=tmp_path)

    assert all(result.status is DrillStatus.SKIPPED for result in report.results)
    [finding] = report.findings
    assert finding.kind is DrillFailureKind.NOT_DRILLED and "nothing was checked" in finding.detail
    message = drill_alert([report])
    assert message is not None and "not_drilled" in message.body


# --- a missing control ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lacks", "flow", "needle"),
    [
        ("move", "set_lineup", "Timeout"),
        ("here", "set_lineup", "Timeout"),
        ("drop", "add_drop", "offers no Drop Player button"),
        ("continue", "add_drop", "Timeout"),
        ("dialog", "add_drop", "Timeout"),
        ("confirm", "add_drop", "Timeout"),
        ("drop", "claim_waiver", "offers no Drop Player button"),
    ],
)
def test_a_missing_control_fails_that_flow_and_names_it_in_the_alert(
    tmp_path: Path, lacks: str, flow: str, needle: str
) -> None:
    site = nba_site()
    site.lacks.add(lacks)
    report = drill(site, tmp_path=tmp_path)

    assert not report.ok
    [finding] = [item for item in report.findings if item.flow == flow]
    assert finding.kind is DrillFailureKind.UI_FAILED and needle in finding.detail, finding.detail
    assert finding.url is not None and finding.url.startswith("https://fantasy.espn.com/")
    assert only(report, flow).status is DrillStatus.FAILED
    assert site.saves == [] and site.final_clicks == 0

    message = drill_alert([report])
    assert isinstance(message, Message)
    assert message.title.startswith("UI fallback drill failed: ") and message.priority == "high"
    assert f"nba: {flow}: ui_failed: the walk failed before its confirm" in message.body
    assert message.link == report.findings[0].url
    payload = drill_payload([report])
    assert payload["ok"] is False and payload["unsafe"] is False
    [entry] = [item for item in payload["leagues"][0]["findings"] if item["flow"] == flow]
    assert entry["kind"] == "ui_failed" and entry["url"] == finding.url


def test_a_failed_walk_still_navigates_away_and_leaves_a_screenshot(tmp_path: Path) -> None:
    site = nba_site()
    site.lacks.add("continue")
    report = drill(site, tmp_path=tmp_path, flows=["add_drop"])

    assert site.did("goto")[-1] == "about:blank" and site.closed
    assert any("failed" in Path(name).name for name in only(report, "add_drop").artifacts)


def test_a_failure_screenshot_that_raises_does_not_skip_leaving_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: Any, name: str) -> None:
        raise RuntimeError("the page is gone")

    monkeypatch.setattr(UiSession, "screenshot", broken)
    site = nba_site()
    site.lacks.add("continue")
    report = drill(site, tmp_path=tmp_path, flows=["add_drop"])  # does not raise

    assert site.did("goto")[-1] == "about:blank" and site.closed
    assert only(report, "add_drop").status is DrillStatus.FAILED


def test_a_signed_out_profile_is_a_failure_naming_the_login(tmp_path: Path) -> None:
    site = nba_site()
    site.signed_in = False
    report = drill(site, tmp_path=tmp_path)

    assert {finding.flow for finding in report.findings} == {"set_lineup", "add_drop", "claim_waiver"}
    assert all("Log in Required" in finding.detail for finding in report.findings)


def test_the_precondition_notes_of_a_flow_ride_along_in_the_failure(tmp_path: Path) -> None:
    site = nba_site()
    site.lacks.add("drop")
    site.cards[site.entries()[0]["playerId"]]["player"]["droppable"] = False
    report = drill(site, tmp_path=tmp_path, flows=["add_drop"])

    [finding] = report.findings
    assert finding.kind is DrillFailureKind.UI_FAILED
    assert only(report, "add_drop").pages  # it opened the page before it failed


# --- the views and the registry ---------------------------------------------------------------------------------------


def test_a_view_the_drill_cannot_read_is_a_setup_failure_with_no_walk(tmp_path: Path) -> None:
    site = nba_site()
    api = site.api()
    api.fail(500, view="mRoster")
    report = drill(site, tmp_path=tmp_path, api=api)

    [finding] = report.findings
    assert finding.kind is DrillFailureKind.SETUP_FAILED and finding.flow == "*"
    assert "the views could not be read" in finding.detail and report.results == ()
    assert site.did("goto") == []  # no page was opened


def test_a_flows_preconditions_that_cannot_be_read_fail_only_that_flow(tmp_path: Path) -> None:
    site = nba_site()
    api = site.api()
    api.fail(500, view="proTeamSchedules_wl", times=2)  # add_drop and claim_waiver read the pro schedule
    report = drill(site, tmp_path=tmp_path, api=api)

    kinds = {finding.flow: finding.kind for finding in report.findings}
    assert kinds == {"add_drop": DrillFailureKind.SETUP_FAILED, "claim_waiver": DrillFailureKind.SETUP_FAILED}
    assert only(report, "set_lineup").status is DrillStatus.PASSED


def test_a_flow_whose_check_blows_up_fails_only_that_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: Any, ctx: Any) -> Any:
        raise KeyError("no such roster")

    monkeypatch.setattr(SetLineup, "check", broken)
    report = drill(nba_site(), tmp_path=tmp_path)

    [finding] = report.findings
    assert finding.kind is DrillFailureKind.SETUP_FAILED and finding.flow == "set_lineup"
    assert only(report, "add_drop").status is DrillStatus.PASSED


class NewFlow(Flow[AddDropPayload]):
    """A UI flow the drill has never heard of, as a trade flow would be before it gets a planner."""

    name = "new_flow"
    kinds = (ProposalKind.ADD_DROP,)
    payload_type = AddDropPayload
    modes = (Mode.UI,)

    def check(self, ctx: FlowContext[AddDropPayload]) -> Preconditions:
        return Preconditions()

    def run_ui(self, ctx: FlowContext[AddDropPayload], ui: UiDriver, pre: Preconditions) -> None:
        raise AssertionError("no planner: the drill must not run it")

    def verify(self, ctx: FlowContext[AddDropPayload], pre: Preconditions) -> Verification:
        return Verification(True)


def test_a_ui_flow_without_a_planner_is_a_finding_so_it_cannot_go_unwatched(tmp_path: Path) -> None:
    registry = FlowRegistry()
    registry.register(NewFlow())
    report = drill(nba_site(), tmp_path=tmp_path, registry=registry)

    [finding] = report.findings
    assert finding.kind is DrillFailureKind.NO_PLANNER and finding.flow == "new_flow"
    assert "fm.browser.drills.PLANNERS" in finding.detail


def test_an_api_only_flow_is_not_drilled(tmp_path: Path) -> None:
    registry = FlowRegistry()
    registry.register(CancelWaiverClaim())
    report = drill(nba_site(), tmp_path=tmp_path, registry=registry)

    assert report.results == () and report.ok


# --- the final save is never clicked ----------------------------------------------------------------------------------


def test_the_confirm_the_flows_reach_is_waited_for_but_never_clicked(tmp_path: Path) -> None:
    site = nba_site()
    drill(site, tmp_path=tmp_path)

    assert site.final_controls
    assert [control.name for control in site.final_controls if control.clicks] == []
    clicked = site.did("click")
    assert clicked and all("Confirm" not in detail for detail in clicked)
    assert site.saves == []


class SneakyFlow(Flow[AddDropPayload]):
    """A flow that skips ``ui.confirm`` and clicks the save control itself, swallowing what it can."""

    name = "sneaky"
    kinds = (ProposalKind.ADD_DROP,)
    payload_type = AddDropPayload
    modes = (Mode.UI,)

    def __init__(self, site: Site) -> None:
        self.site = site

    def check(self, ctx: FlowContext[AddDropPayload]) -> Preconditions:
        return Preconditions()

    def run_ui(self, ctx: FlowContext[AddDropPayload], ui: UiDriver, pre: Preconditions) -> None:
        assert ctx.payload.add_espn_id is not None and ctx.payload.drop_espn_id is not None
        add, drop = self.site.name(ctx.payload.add_espn_id), self.site.name(ctx.payload.drop_espn_id)
        page = ui.page
        url = selectors.roster_fix_url(
            ctx.game, ctx.league.espn_league_id, ctx.league.season, ctx.team_id, ctx.payload.add_espn_id, "add"
        )
        page.goto(url)
        selectors.drop_player_button(page, drop).click()
        selectors.continue_button(page, add, drop).click()
        dialog = selectors.CONFIRM_DIALOG.locate(page)
        try:
            selectors.confirm_transaction_button(dialog, add, drop).click()
        except Exception:  # a guard that raised an ordinary Exception could be swallowed here
            pass

    def verify(self, ctx: FlowContext[AddDropPayload], pre: Preconditions) -> Verification:
        return Verification(True)


def test_a_flow_that_clicks_a_save_control_itself_is_stopped_before_the_click(tmp_path: Path) -> None:
    site = nba_site()
    add, drop = 6589, site.entries()[-1]["playerId"]
    plan = DrillPlan(
        AddDropPayload(add_espn_id=add, drop_espn_id=drop), "a flow that clicks", (site.name(add), site.name(drop))
    )
    registry = FlowRegistry()
    registry.register(SneakyFlow(site))
    report = drill(site, tmp_path=tmp_path, registry=registry, planners={"sneaky": lambda _views: plan})

    [finding] = report.findings
    assert finding.kind is DrillFailureKind.UNSAFE and finding.flow == "sneaky"
    assert "refused to click button[" in finding.detail and "Confirm" in finding.detail
    assert site.saves == [] and site.final_clicks == 0
    assert len(site.final_controls) == 1  # the dialog opened and its button was within reach
    message = drill_alert([report])
    assert message is not None and message.title == "UI drill SAFETY: 1 save attempt caught"
    assert drill_payload([report])["unsafe"] is True


def test_the_guard_refuses_every_shape_of_save_control_and_what_it_cannot_classify() -> None:
    names = ["Cole Anthony", "Tyus Jones"]
    probes = final_save_probes(names)
    page = FakePage()
    guard = drills._Guard(probes)
    guarded = GuardedPage(page, guard)

    page.add(FakeElement(role="button", name="Anything", text="x"))
    blocked = [
        guarded.get_by_role("button", name=selectors.HERE_NAME),
        guarded.get_by_role("button", name=selectors.ADD_NAME),
        guarded.get_by_role("button", name=selectors.CLAIM_NAME),
        guarded.get_by_role("button", name=selectors.CONFIRM_TRANSACTION_NAME),
        selectors.add_button(guarded, "Cole Anthony"),
        selectors.claim_button(guarded, "Cole Anthony"),
        selectors.confirm_transaction_button(guarded, "Cole Anthony", "Tyus Jones"),
        guarded.get_by_role("button", name="Confirm add Cole Anthony and drop Tyus Jones"),
        guarded.get_by_role("button", name="Confirm"),  # a name that picks any button that begins like a save
        guarded.get_by_role("button"),  # a button with no name could be any of them
        guarded.get_by_text("Move", exact=True),
        guarded.get_by_label("Save"),
        guarded.locator(".save"),
        guarded.get_by_test_id("save"),
    ]
    for locator in blocked:
        with pytest.raises(FinalSaveClickError):
            locator.click()
    assert guard.clicks == [] and len(guard.refused) == len(blocked)

    allowed = [
        guarded.get_by_role("button", name=selectors.MOVE_NAME),
        selectors.drop_player_button(guarded, "Cole Anthony"),
        selectors.continue_button(guarded, "Cole Anthony", "Tyus Jones"),
        guarded.get_by_role("button", name=selectors.CONTINUE_NAME),
        guarded.get_by_role("button", name="Cancel", exact=True),
    ]
    for locator in allowed:
        guard.before_click(locator._spec)
    assert len(guard.clicks) == len(allowed)


def test_a_confirm_like_name_is_refused_whatever_the_player_is() -> None:
    """The probes only spell out the players the drill knows of; a flow builds its locators from ``PlayerFacts.name``,
    which can differ, so a name that starts like a save control is refused on its own."""
    guard = drills._Guard(final_save_probes(["Cole Anthony"]))
    guarded = GuardedPage(FakePage(), guard)
    blocked = [
        selectors.add_button(guarded, "Somebody Else"),
        selectors.claim_button(guarded, "Somebody Else"),
        selectors.confirm_transaction_button(guarded, "Somebody Else", "Another One"),
        guarded.get_by_role("button", name=re.compile(r"^\s*(?:here)\s*$", re.IGNORECASE)),
        guarded.get_by_role("button", name=re.compile(r"(?i)^claim\s+x")),
        guarded.get_by_role("button", name="  CONFIRM the thing"),
        guarded.get_by_role("button", name=re.compile(r"^(?:move this|add a)", re.IGNORECASE)),  # a later alternative
        guarded.get_by_text(re.compile(r"^add", re.IGNORECASE)),
        guarded.get_by_text("Here"),
    ]
    for locator in blocked:
        with pytest.raises(FinalSaveClickError):
            locator.click()
    assert guard.clicks == [] and len(guard.refused) == len(blocked)
    harmless: list[Any] = [
        selectors.drop_player_button(guarded, "Add Smith"),  # a player called Add is still only a drop button
        selectors.continue_button(guarded, "Somebody Else", "Another One"),  # its (?:add|claim) is not a lead
        guarded.get_by_role("button", name="Address book", exact=True),  # not the word Add
        guarded.get_by_role("button", name=selectors.MOVE_NAME),
    ]
    for locator in harmless:
        guard.before_click(locator._spec)
    assert len(guard.clicks) == 4


def test_press_check_and_select_option_are_guarded_like_a_click() -> None:
    """Enter or Space on a focused save control is a click, and so is checking or choosing on one."""
    page = FakePage(
        FakeElement(role="button", name="Add Somebody Guard for Team", on_click=lambda _page: pytest.fail("clicked")),
        FakeElement(role="checkbox", name="Confirm it", options=("a",)),
        FakeElement(role="combobox", name="Status", options=("ALL",)),
    )
    guard = drills._Guard(())
    guarded = GuardedPage(page, guard)
    add = guarded.get_by_role("button", name=re.compile("^add", re.IGNORECASE))
    for key in ("Enter", "Space", " ", "Control+Enter", "NumpadEnter", "Return"):
        with pytest.raises(FinalSaveClickError):
            add.press(key)
    with pytest.raises(FinalSaveClickError):
        guarded.get_by_role("checkbox", name="Confirm it").check()
    with pytest.raises(FinalSaveClickError):
        guarded.locator("select").select_option("a")  # unclassified, as a click on it would be
    assert page.did("press") == [] and page.did("check") == [] and page.did("select") == []
    add.press("Tab")  # a key that activates nothing is not a save
    status = guarded.get_by_role("combobox", name="Status")
    status.select_option("ALL")
    assert page.did("press") == [f"{add.__repr__()[len('GuardedLocator(') : -1]} Tab"] or len(page.did("press")) == 1
    assert page.did("select") != []


def test_the_probes_are_built_from_the_names_the_preconditions_observed_too(tmp_path: Path) -> None:
    observed = {
        "add": {"name": "Cole Anthony", "slot": 3},
        "roster": {"1": {"name": "Tyus Jones"}},
        "moves": [{"name": "A"}],
    }
    assert observed_names(observed) == ("Cole Anthony", "Tyus Jones", "A")
    assert observed_names({"name": 5, "x": [{"name": " "}]}) == ()

    seen: list[tuple[str, ...]] = []

    class Spy(SneakyFlow):
        name = "spy"

        def check(self, ctx: FlowContext[AddDropPayload]) -> Preconditions:
            return Preconditions(observed={"add": {"name": "Observed Name"}})

        def run_ui(self, ctx: FlowContext[AddDropPayload], ui: UiDriver, pre: Preconditions) -> None:
            seen.append(ui.guard.probes)  # type: ignore[attr-defined]
            raise DryRunStop("spied")

    site = nba_site()
    plan = DrillPlan(AddDropPayload(add_espn_id=6589), "spy", ("Plan Name",))
    registry = FlowRegistry()
    registry.register(Spy(site))
    drill(site, tmp_path=tmp_path, registry=registry, planners={"spy": lambda _views: plan})
    [probes] = seen
    assert any("Plan Name" in probe for probe in probes) and any("Observed Name" in probe for probe in probes)


def test_a_derived_locator_keeps_what_it_was_built_from() -> None:
    page = FakePage(FakeElement(role="row").add(FakeElement(role="button", name="Move", text="HERE")))
    guarded = GuardedPage(page, drills._Guard(final_save_probes(["Cole Anthony"])))
    row = selectors.ROSTER_ROW.locate(guarded).filter(has_text="Move").first
    button = selectors.HERE_BUTTON.locate(row).first.nth(0)

    with pytest.raises(FinalSaveClickError):
        button.click()
    assert page.elements[0].children[0].clicks == 0 and page.did("click") == []


def test_ui_confirm_stops_the_walk_and_cannot_click_its_control(tmp_path: Path) -> None:
    button = FakeElement(role="button", name="Confirm add A and drop B", on_click=lambda _page: pytest.fail("clicked"))
    page = FakePage(button)
    ui = DrillUi(page, AuditLog(tmp_path / "a", tmp_path), final_save_probes(["A", "B"]))

    with pytest.raises(DryRunStop):
        ui.confirm(ui.page.get_by_role("button"), what="add A and drop B")
    assert ui.reached == ["add A and drop B"] and button.clicks == 0 and ui.dry_run is True
    assert ui.session.stopped_before == "add A and drop B" and ui.session.confirms == []

    missing = DrillUi(FakePage(), AuditLog(tmp_path / "b", tmp_path), ())
    with pytest.raises(Exception, match="Timeout"):
        missing.confirm(missing.page.get_by_role("button", name="Nope"), what="never there")


def test_a_confirm_that_returns_is_a_safety_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(UiSession, "confirm", lambda self, target, *, what: None)  # a session that would carry on
    site = nba_site()
    report = drill(site, tmp_path=tmp_path, flows=["add_drop"])

    [finding] = report.findings
    assert finding.kind is DrillFailureKind.UNSAFE and "returned instead of stopping the drill" in finding.detail
    assert site.saves == [] and site.final_clicks == 0


def test_an_interrupt_still_leaves_the_page_before_it_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(self: Any, ctx: Any, ui: Any, pre: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(AddDrop, "run_ui", interrupted)
    site = nba_site()
    with pytest.raises(KeyboardInterrupt):
        drill(site, tmp_path=tmp_path, flows=["add_drop"])
    assert site.did("goto")[-1] == "about:blank" and site.closed


# --- non-GET requests are refused -------------------------------------------------------------------------------------


def test_the_drill_refuses_a_runtime_that_could_send(tmp_path: Path) -> None:
    site = nba_site()
    with pytest.raises(DrillError, match="needs a dry-run runtime"):
        run_drills(
            LEAGUES["fba"],
            runtime=site.runtime(transport=FakeTransport()),
            registry=real_registry(),
            audit_root=tmp_path,
            now=NOW,
        )
    assert site.did("goto") == [] and site.saves == []


def test_the_drill_requires_the_runtime_to_say_it_was_opened_as_a_dry_run(tmp_path: Path) -> None:
    """A refusing transport alone is not enough: the opener's own ``dry_run`` flag must be set (it is what makes
    the live context abort every write), so a runtime built by hand with the right transport type is refused."""
    site = nba_site()
    with pytest.raises(DrillError, match="needs a dry-run runtime"):
        run_drills(
            LEAGUES["fba"],
            runtime=site.runtime(dry_run=False),
            registry=real_registry(),
            audit_root=tmp_path,
            now=NOW,
        )
    assert site.did("goto") == [] and site.saves == []


def test_the_dry_run_rule_the_drill_runs_under_aborts_every_write() -> None:
    assert dry_run_block_reason("POST", WRITE_URL) is not None
    assert dry_run_block_reason("GET", WRITE_URL) is not None  # the write host: any verb
    assert dry_run_block_reason("PUT", "https://fantasy.espn.com/apis/v3/anything") is not None
    assert dry_run_block_reason("GET", selectors.team_page_url("fba", 2020202, 1, 2027)) is None


def test_a_write_the_page_makes_is_reported_even_though_the_context_aborts_it(tmp_path: Path) -> None:
    site = nba_site()
    original = site.pick_drop

    def drop_that_posts(drop_id: int, add_id: int, kind: str, page: FakePage) -> None:
        site.emit_request("POST", WRITE_URL)  # what a click past the guard would have sent
        original(drop_id, add_id, kind, page)

    site.pick_drop = drop_that_posts  # type: ignore[method-assign]
    report = drill(site, tmp_path=tmp_path, flows=["add_drop"])

    [finding] = report.findings
    assert finding.kind is DrillFailureKind.UNSAFE and "the dry-run guard aborted it" in finding.detail
    assert f"POST {WRITE_URL}" in finding.detail
    assert only(report, "add_drop").status is DrillStatus.FAILED  # reached its confirm, but a write left: not a pass


def test_the_watch_tells_a_save_attempt_from_the_pages_other_blocked_requests() -> None:
    watch = RequestWatch()

    def request(method: str, url: str) -> Any:
        return type("Request", (), {"method": method, "url": url})()

    watch.record(request("GET", "https://fantasy.espn.com/basketball/team"))  # allowed: not blocked
    watch.record(request("POST", "https://sw88.espn.com/b/ss/x"))  # a beacon: blocked, not a save
    watch.record(request("POST", WRITE_URL))
    watch.record(request("GET", f"https://{WRITES_HOST}/apis/v3/games/fba"))
    watch.record(request("POST", "https://fantasy.espn.com/apis/v3/games/fba/leagues/1/transactions/"))

    assert len(watch.blocked) == 4 and len(watch.save_attempts) == 3
    assert "sw88" not in " ".join(watch.save_attempts)


def test_a_page_without_an_event_hook_is_still_drilled(tmp_path: Path) -> None:
    class Hookless(FakePage):
        pass

    watch = RequestWatch()
    watch.attach(Hookless())  # no `on`: nothing to listen to, and nothing breaks
    assert watch.blocked == []


# --- alerts -----------------------------------------------------------------------------------------------------------


def test_one_alert_lists_every_league_and_caps_the_lines() -> None:
    findings = tuple(
        drills.DrillFinding(DrillFailureKind.UI_FAILED, f"flow{index}", "the walk failed", f"https://x/{index}")
        for index in range(15)
    )
    many = [DrillReport("nfl", (), findings), DrillReport("nba", (), findings[:1]), DrillReport("clean", ())]
    message = drill_alert(many)

    assert message is not None and message.title == "UI fallback drill failed: 16 findings"
    lines = message.body.splitlines()
    assert len(lines) == 13 and lines[-1] == "... and 4 more (fm drill lists them all)"
    assert lines[0] == "nfl: flow0: ui_failed: the walk failed" and message.link == "https://x/0"
    assert len(drill_payload(many)["leagues"][0]["findings"]) == 15  # the payload keeps everything


def test_a_drill_that_could_not_start_has_its_own_alert() -> None:
    message = could_not_run_alert("the saved ESPN session expired\nrun `fm login`")
    assert message.title == "UI fallback drill could not run" and message.priority == "high"
    assert message.body == "the saved ESPN session expired run `fm login`"


# --- fm drill ---------------------------------------------------------------------------------------------------------


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


CONFIG = """
[[league]]
key = "nba"
sport = "nba"
espn_league_id = 2020202
season = 2027
team_id = 1

[[league]]
key = "other"
sport = "nba"
espn_league_id = 2020203
season = 2027
team_id = 1

[llm]
model = "claude-opus-5-5"
daily_budget_usd = 2.0
"""


@pytest.fixture
def configured() -> None:
    directory = paths.config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.toml").write_text(CONFIG, encoding="utf-8")
    (directory / ".env").write_text("ANTHROPIC_API_KEY=placeholder\nNTFY_TOPIC=a\nNTFY_REPLY_TOPIC=b\n", "utf-8")


@pytest.fixture
def phone(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(drill_cmd, "open_channel", lambda config: recorder)
    return recorder


def cli() -> typer.Typer:
    root = typer.Typer()
    root.callback()(lambda: None)  # keep it a group, as fm.cli does
    drill_cmd.register(root)
    return root


class Opened:
    """What the live opener was asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.options: list[Any] = []


def live(monkeypatch: pytest.MonkeyPatch, site: Site, *, error: Exception | None = None) -> Opened:
    """Swap the live browser opener for the fake site and the fake read API."""
    opened = Opened()

    @contextmanager
    def runtime(row: Any, *, dry_run: bool) -> Iterator[Runtime]:
        opened.calls.append((row.key, dry_run))
        if error is not None:
            raise error
        assert dry_run, "the drill must open its runtime as a dry run: every non-GET to ESPN is aborted"
        yield Runtime(
            reader=site.api().client("fba", row.espn_league_id, row.season),
            transport=RefusingTransport("dry run"),
            browser=FakeBrowser(site),
            member_id=FAKE_SWID,
            dry_run=dry_run,
        )

    def opener(options: Any) -> Any:
        opened.options.append(options)
        return runtime

    monkeypatch.setattr(drill_cmd, "live_opener", opener)
    return opened


@pytest.mark.usefixtures("configured")
def test_fm_drill_help() -> None:
    result = runner.invoke(cli(), ["drill", "--help"])
    assert result.exit_code == 0
    for option in ("--league", "--flow", "--no-alert", "--headed", "--channel"):
        assert option in result.output
    assert "--fixtures" not in result.output


@pytest.mark.usefixtures("configured")
def test_a_clean_live_drill_sends_nothing_and_opens_a_dry_run(monkeypatch: pytest.MonkeyPatch, phone: Recorder) -> None:
    site = nba_site()
    opened = live(monkeypatch, site)
    result = runner.invoke(cli(), ["drill", "--league", "nba"])

    assert result.exit_code == 0, result.output
    assert opened.calls == [("nba", True)]  # dry_run=True, as fm canary and fm execute --dry-run open it
    assert "nba: set_lineup: ok, swap " in result.output and "stopped before 'move " in result.output
    assert "nba: add_drop: ok, add " in result.output and "nba: claim_waiver: ok, claim " in result.output
    assert "nba: ok" in result.output
    assert phone.sent == [] and site.saves == [] and site.final_clicks == 0
    assert opened.options[0].headless is True


@pytest.mark.usefixtures("configured")
def test_a_failing_live_drill_alerts_once_for_the_whole_run(monkeypatch: pytest.MonkeyPatch, phone: Recorder) -> None:
    site = nba_site()
    site.lacks.add("drop")
    live(monkeypatch, site)
    result = runner.invoke(cli(), ["drill"])  # both leagues

    assert result.exit_code == 1, result.output
    assert (
        "nba: FAILED add_drop: ui_failed:" in result.output
        and "other: FAILED claim_waiver: ui_failed:" in result.output
    )
    [message] = phone.sent
    assert message.title == "UI fallback drill failed: 4 findings"
    assert "nba: add_drop: ui_failed" in message.body and "other: claim_waiver: ui_failed" in message.body
    assert phone.closed and "alert sent" in result.output
    assert site.saves == [] and site.final_clicks == 0


@pytest.mark.usefixtures("configured")
def test_no_alert_keeps_a_failing_drill_quiet(monkeypatch: pytest.MonkeyPatch, phone: Recorder) -> None:
    site = nba_site()
    site.lacks.add("here")
    live(monkeypatch, site)
    result = runner.invoke(cli(), ["drill", "--league", "nba", "--no-alert"])

    assert result.exit_code == 1 and phone.sent == [] and "set_lineup: ui_failed" in result.output


@pytest.mark.usefixtures("configured")
def test_flow_picks_what_is_walked(monkeypatch: pytest.MonkeyPatch, phone: Recorder) -> None:
    site = nba_site()
    live(monkeypatch, site)
    result = runner.invoke(cli(), ["drill", "--league", "nba", "--flow", "set_lineup", "-f", "claim_waiver"])

    assert result.exit_code == 0, result.output
    assert "add_drop" not in result.output and "set_lineup: ok" in result.output and "claim_waiver: ok" in result.output


@pytest.mark.usefixtures("configured")
def test_an_unknown_flow_or_league_is_refused_before_a_browser_opens(
    monkeypatch: pytest.MonkeyPatch, phone: Recorder
) -> None:
    opened = live(monkeypatch, nba_site())
    flow = runner.invoke(cli(), ["drill", "--flow", "cancel_waiver"])
    league = runner.invoke(cli(), ["drill", "--league", "nope"])

    assert flow.exit_code == 1 and "no UI flow cancel_waiver for nba" in flow.output
    assert league.exit_code == 1 and "error:" in league.output
    assert opened.calls == [] and phone.sent == []


@pytest.mark.usefixtures("configured")
def test_a_drill_that_cannot_start_alerts_too(monkeypatch: pytest.MonkeyPatch, phone: Recorder) -> None:
    live(monkeypatch, nba_site(), error=AuthError("the saved ESPN session expired; run `fm login`"))
    result = runner.invoke(cli(), ["drill", "--league", "nba"])

    assert result.exit_code == 1 and "the saved ESPN session expired" in result.output
    [message] = phone.sent
    assert message.title == "UI fallback drill could not run" and "fm login" in message.body

    phone.sent.clear()
    quiet = runner.invoke(cli(), ["drill", "--league", "nba", "--no-alert"])
    assert quiet.exit_code == 1 and phone.sent == []


def test_the_command_is_discovered_by_the_root_cli() -> None:
    from fm.cli import app

    result = runner.invoke(app, ["drill", "--help"])
    assert result.exit_code == 0 and "final save" in result.output


def test_the_drill_modules_make_no_state_changing_click_of_their_own() -> None:
    """The one ``.click(`` in the drill is the guarded locator's, behind the check that refuses a save."""
    source = Path(drills.__file__).read_text(encoding="utf-8")
    command = Path(drill_cmd.__file__).read_text(encoding="utf-8")

    assert source.count(".click(") == 1 and "self._guard.before_click(self._spec)\n        self._inner.click(" in source
    for forbidden in ("PlaywrightTransport", "WriteTransport", "build_request", ".send(", "propose(", "execute("):
        assert forbidden not in source and forbidden not in command
