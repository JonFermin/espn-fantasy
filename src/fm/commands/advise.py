"""``fm status``, ``fm lineup`` and ``fm waivers``: the advisor commands (DESIGN section 12, ROADMAP #26).

All three read the store ``fm sync`` filled and render through :mod:`fm.render`; none of them touches ESPN's write
path (CLAUDE.md: workers propose, the executor acts).

- ``fm status`` is read-only: per configured league, the sync state, the team's record and FAAB, whether a pro schedule
  is captured, and the open proposals with their rationale.
- ``fm lineup`` runs :func:`fm.decide.lineup.plan_lineup` for the synced scoring period and renders the recommended
  lineup with its expected points and, with ``--opponent``, the win probability. Each draft then goes through
  :func:`fm.proposals.propose`, so the output shows the policy verdict (proposed as #n, already open, or blocked and
  why) instead of raising; ``--dry-run`` evaluates policy without storing anything.
- ``fm waivers`` runs :func:`fm.decide.waivers.decide_waivers`, whose warnings name every skipped candidate and why,
  and renders the moves with their verdicts, the top of the ranking and the replacement levels.

Both decisions need the season's pro schedule (lock times, byes, the waiver run's period). It comes from
``--schedule FILE`` (a recorded ``proTeamSchedules_wl`` view), else the newest capture indexed in ``raw_snapshots``
under the cache dir, else a live read through the ESPN session (``fm login``), which is captured and indexed for the
next run. Without any, ``fm lineup`` reports why and plans nothing; ``fm waivers`` runs with the decision's own
fallbacks. ``--as-of`` pins the clock (locks, deadlines, expiry) for replays; the period planned is the one the
roster was synced for, so a stale sync is reported rather than planned around.

Fixture home: ``FM_CONFIG_DIR=tests/fixtures/home FM_CACHE_DIR=tests/fixtures/home/cache uv run fm lineup --as-of
2026-10-04T15:00Z --opponent 2`` (``tests/fixtures/home/README.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from pydantic import ValidationError

from fm import paths
from fm.browser.session import BrowserError, profile_exists
from fm.config import Config, ConfigError, League, load_config
from fm.decide import registry
from fm.decide.lineup import DECISION_KIND, LineupDecision, LineupDraft, LineupError, plan_lineup
from fm.decide.waivers import WAIVERS_KIND, WaiverDecision, WaiverError, decide_waivers
from fm.espn.auth import AuthError, load_session
from fm.espn.client import EspnClient, EspnClientError, EspnRead, View
from fm.espn.ids import ids_for
from fm.espn.models import ProSchedule
from fm.jobs.sync import ESPN_SOURCE
from fm.model.valuation import ValuationError
from fm.proposals import PolicyError, ProposalKind, evaluate, pause_state, propose
from fm.proposals.policy import stored_settings
from fm.render import stamp
from fm.render.lineup import lineup_lines
from fm.render.status import LeagueStatus, status_lines
from fm.render.waivers import waiver_lines
from fm.sports.base import plugin_for
from fm.store import LeagueRow, RawSnapshotRow, Store

SCHEDULE_KIND = View.PRO_SCHEDULES.value
"""``raw_snapshots.kind`` of a captured pro schedule."""
LINEUP_CREATED_BY = "cli.lineup"
"""``created_by`` of the proposals ``fm lineup`` stores."""
DEFAULT_TOP = 10

LeagueOption = Annotated[
    list[str] | None,
    typer.Option("--league", "-l", help="Only these league keys from config.toml (repeatable). Default: every league."),
]
AsOfOption = Annotated[
    str | None,
    typer.Option(
        "--as-of",
        help="Pin the clock (ISO 8601, e.g. 2026-10-04T15:00Z; a naive time is UTC): locks, deadlines and expiry are "
        "judged at this instant. Default: now.",
    ),
]
ScheduleOption = Annotated[
    Path | None,
    typer.Option(
        "--schedule",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="A recorded proTeamSchedules_wl view to use as the pro schedule instead of the cache or ESPN.",
    ),
]
DryRunOption = Annotated[
    bool, typer.Option("--dry-run", help="Show what would be proposed and the policy verdict; store nothing.")
]
OpponentOption = Annotated[
    int | None,
    typer.Option(
        "--opponent",
        min=1,
        help="This period's opponent (ESPN team id); his synced roster gives the win probability. One league only.",
    ),
]
MaxMovesOption = Annotated[int | None, typer.Option("--max-moves", min=1, help="Propose at most this many moves.")]
TopOption = Annotated[int, typer.Option("--top", min=0, help="Ranked (add, drop) pairs to show.")]


# --- status -----------------------------------------------------------------------------------------------------------


def status(league: LeagueOption = None) -> None:
    """Sync state, team record, FAAB and open proposals per league, from the store alone."""
    config = _config()
    with Store.open() as store:
        statuses = [_status(store, configured) for configured in _select(config, league)]
        paused = pause_state()
        lines = status_lines(statuses, paused=None if paused is None else paused.describe(), profile=profile_exists())
    for line in lines:
        typer.echo(line)


def _status(store: Store, configured: League) -> LeagueStatus:
    row = store.leagues.by_key(configured.key)
    if row is None:
        return LeagueStatus(configured)
    snapshot = _schedule_snapshot(store, configured)
    captured, periods = None, ()
    if snapshot is not None:
        schedule, _ = _read_schedule(paths.cache_dir() / snapshot.path)
        if schedule is not None:
            captured, periods = snapshot.fetched_at, schedule.scoring_periods
    return LeagueStatus(
        configured,
        league=row,
        settings=stored_settings(store, row),
        team=store.teams.get(row.row_id, row.team_id),
        roster_period=store.rosters.latest_period(row.row_id),
        schedule_captured=captured,
        schedule_periods=periods,
        open_proposals=store.proposals.open(row.row_id),
    )


# --- lineup -----------------------------------------------------------------------------------------------------------


def lineup(
    league: LeagueOption = None,
    as_of: AsOfOption = None,
    opponent: OpponentOption = None,
    schedule: ScheduleOption = None,
    dry_run: DryRunOption = False,
) -> None:
    """Recommend this period's lineup: expected points, win probability with --opponent, the moves and deadlines.

    Drafts go through the proposal policy; the verdict is shown, never raised. --dry-run stores nothing.
    """
    config = _config()
    at = _parse_as_of(as_of)
    selected = _select(config, league)
    if opponent is not None and len(selected) != 1:
        _fail("--opponent names one league's opponent; add --league KEY")
    failed = False
    with Store.open() as store:
        for index, configured in enumerate(selected):
            if index:
                typer.echo("")
            failed |= not _lineup_for(store, config, configured, at=at, opponent=opponent, path=schedule, dry=dry_run)
    if failed:
        raise typer.Exit(1)


def _lineup_for(
    store: Store,
    config: Config,
    configured: League,
    *,
    at: datetime,
    opponent: int | None,
    path: Path | None,
    dry: bool,
) -> bool:
    """Plan, propose and print one league's block; False when the league could not be planned."""
    key = configured.key
    row = store.leagues.by_key(key)
    if row is None:
        typer.echo(f"{key}: not synced; run fm sync")
        return True
    if registry.get(row.sport, DECISION_KIND) is None:
        typer.echo(f"{key}: no lineup decision is registered for {row.sport}")
        return True
    period = store.rosters.latest_period(row.row_id)
    if period is None:
        typer.echo(f"{key}: no roster snapshot; run fm sync")
        return True
    loaded = _schedule(store, configured, path=path)
    if loaded.schedule is None:
        typer.echo(f"{key}: no lineup planned")
        typer.echo(f"  warning: {loaded.note}")
        return True
    try:
        decision = plan_lineup(
            store, row, schedule=loaded.schedule, now=at, period=period, opponent_team_id=opponent, weights=None
        )
    except LineupError as exc:
        typer.echo(f"error: {key}: {exc}", err=True)
        return False
    verdicts = [_lineup_verdict(store, config, row, draft, at=at, dry=dry) for draft in decision.drafts]
    warnings = [*decision.warnings, *_period_warning(row, loaded.schedule, period, at), *loaded.warnings]
    decision = _with_warnings(decision, warnings)
    team = store.teams.get(row.row_id, row.team_id)
    rival = store.teams.get(row.row_id, opponent) if opponent is not None else None
    for line in lineup_lines(
        decision,
        league=row,
        ids=ids_for(row.sport),
        team_name=team.name if team is not None else f"team {row.team_id}",
        opponent=rival.name if rival is not None else None if opponent is None else f"team {opponent}",
        verdicts=verdicts,
        as_of=stamp(at),
    ):
        typer.echo(line)
    return True


def _with_warnings(decision: LineupDecision, warnings: Sequence[str]) -> LineupDecision:
    return LineupDecision(
        league_id=decision.league_id,
        inputs=decision.inputs,
        current=decision.current,
        best=decision.best,
        rescue=decision.rescue,
        outlook=decision.outlook,
        drafts=decision.drafts,
        warnings=tuple(warnings),
    )


def _lineup_verdict(
    store: Store, config: Config, row: LeagueRow, draft: LineupDraft, *, at: datetime, dry: bool
) -> str:
    """Draft the proposal through policy: stored (or found open already), or blocked and why."""
    if dry:
        verdict = evaluate(
            store,
            config,
            row,
            draft.kind,
            draft.payload,
            scoring_period_id=draft.scoring_period_id,
            deadline=draft.deadline,
            now=at,
        )
        return _verdict_text(verdict.allowed, verdict.setting, verdict.reasons, dry=True)
    open_before = {existing.row_id for existing in store.proposals.open(row.row_id)}
    try:
        stored = propose(
            store,
            config,
            row,
            draft.kind,
            draft.payload,
            created_by=LINEUP_CREATED_BY,
            scoring_period_id=draft.scoring_period_id,
            engine_numbers=draft.engine_numbers,
            rationale=draft.rationale,
            deadline=draft.deadline,
            dedupe_key=draft.dedupe_key,
            now=at,
        )
    except PolicyError as exc:
        return _blocked(draft.kind, str(exc))
    return _stored_text(stored.row_id, stored.status, stored.policy, existing=stored.row_id in open_before)


def _period_warning(row: LeagueRow, schedule: ProSchedule, period: int, at: datetime) -> list[str]:
    current = plugin_for(row.sport).scoring_period_at(at, schedule)
    if current is None:
        return [f"{row.key}: the pro schedule has no scoring period at {stamp(at)}; planned the synced period {period}"]
    if current != period:
        return [f"{row.key}: roster synced for scoring period {period} but period {current} is current; run fm sync"]
    return []


# --- waivers ----------------------------------------------------------------------------------------------------------


def waivers(
    league: LeagueOption = None,
    as_of: AsOfOption = None,
    schedule: ScheduleOption = None,
    max_moves: MaxMovesOption = None,
    top: TopOption = DEFAULT_TOP,
    dry_run: DryRunOption = False,
) -> None:
    """Rank (add, drop) pairs by rest-of-season value and propose the best claims and free-agent adds.

    Every skipped candidate is named with the reason. Moves go through the proposal policy; the verdict is shown.
    """
    config = _config()
    at = _parse_as_of(as_of)
    failed = False
    with Store.open() as store:
        for index, configured in enumerate(_select(config, league)):
            if index:
                typer.echo("")
            failed |= not _waivers_for(
                store, config, configured, at=at, path=schedule, max_moves=max_moves, top=top, dry=dry_run
            )
    if failed:
        raise typer.Exit(1)


def _waivers_for(
    store: Store,
    config: Config,
    configured: League,
    *,
    at: datetime,
    path: Path | None,
    max_moves: int | None,
    top: int,
    dry: bool,
) -> bool:
    key = configured.key
    row = store.leagues.by_key(key)
    if row is None:
        typer.echo(f"{key}: not synced; run fm sync")
        return True
    if registry.get(row.sport, WAIVERS_KIND) is None:
        typer.echo(f"{key}: no waivers decision is registered for {row.sport}")
        return True
    loaded = _schedule(store, configured, path=path)
    open_before = {existing.row_id for existing in store.proposals.open(row.row_id)}
    try:
        decision = decide_waivers(
            store, config, row, now=at, schedule=loaded.schedule, max_moves=max_moves, store_proposals=not dry
        )
    except (WaiverError, ValuationError) as exc:
        typer.echo(f"error: {key}: {exc}", err=True)
        return False
    verdicts = _waiver_verdicts(store, config, decision, at=at, dry=dry, open_before=open_before)
    if loaded.warnings:
        decision = _waivers_with_warnings(decision, [*decision.warnings, *loaded.warnings])
    for line in waiver_lines(decision, verdicts=verdicts, as_of=stamp(at), top=top):
        typer.echo(line)
    return True


def _waivers_with_warnings(decision: WaiverDecision, warnings: Sequence[str]) -> WaiverDecision:
    return WaiverDecision(
        league=decision.league,
        valuation=decision.valuation,
        wire=decision.wire,
        replacement=decision.replacement,
        ranked=decision.ranked,
        moves=decision.moves,
        proposals=decision.proposals,
        blocked=decision.blocked,
        warnings=tuple(warnings),
    )


def _waiver_verdicts(
    store: Store, config: Config, decision: WaiverDecision, *, at: datetime, dry: bool, open_before: set[int]
) -> list[str]:
    """One verdict per move: the decision stored it (or found it open), policy blocked it, or (dry) what policy says."""
    verdicts: list[str] = []
    blocked = {id(move): reason for move, reason in decision.blocked}
    stored = iter(decision.proposals)
    for move in decision.moves:
        if dry:
            verdict = evaluate(
                store,
                config,
                decision.league,
                move.kind,
                move.payload,
                scoring_period_id=move.scoring_period,
                deadline=move.deadline,
                now=at,
            )
            verdicts.append(_verdict_text(verdict.allowed, verdict.setting, verdict.reasons, dry=True))
        elif id(move) in blocked:
            verdicts.append(_blocked(move.kind, blocked[id(move)]))
        else:
            row = next(stored)
            verdicts.append(_stored_text(row.row_id, row.status, row.policy, existing=row.row_id in open_before))
    return verdicts


# --- verdict wording --------------------------------------------------------------------------------------------------


def _verdict_text(allowed: bool, setting: str, reasons: Sequence[str], *, dry: bool) -> str:
    if not allowed:
        return "blocked: " + "; ".join(reasons)
    return f"would be proposed (policy {setting}); --dry-run stored nothing" if dry else f"proposed (policy {setting})"


def _stored_text(row_id: int | None, status: str, policy: str, *, existing: bool) -> str:
    if existing:
        return f"already open as #{row_id} ({status}, policy {policy})"
    return f"proposed as #{row_id} (policy {policy}); fm proposals approve {row_id}"


def _blocked(kind: ProposalKind, message: str) -> str:
    prefix = f"{kind.value} blocked: "
    return "blocked: " + (message[len(prefix) :] if message.startswith(prefix) else message)


# --- the pro schedule -------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedSchedule:
    """A pro schedule and where it came from, or ``None`` and the note saying why there is none."""

    schedule: ProSchedule | None
    note: str
    warnings: tuple[str, ...] = ()


def _schedule(store: Store, configured: League, *, path: Path | None) -> LoadedSchedule:
    """``--schedule FILE``, else the newest capture under the cache, else a live read (captured for next time)."""
    key = configured.key
    if path is not None:
        schedule, problem = _read_schedule(path)
        if schedule is None:
            _fail(f"--schedule {path}: {problem}")
        return LoadedSchedule(schedule, f"from {path}")
    warnings: list[str] = []
    snapshot = _schedule_snapshot(store, configured)
    if snapshot is not None:
        schedule, problem = _read_schedule(paths.cache_dir() / snapshot.path)
        if schedule is not None:
            return LoadedSchedule(schedule, f"captured {stamp(snapshot.fetched_at)}")
        warnings.append(f"{key}: the captured pro schedule {snapshot.path} {problem}; fetching it again")
    try:
        session = load_session()
    except (AuthError, BrowserError) as exc:
        return LoadedSchedule(
            None,
            f"{key}: no pro schedule for {configured.game} {configured.season} is captured under {paths.cache_dir()} "
            f"(FM_CACHE_DIR) and there is no ESPN session to fetch one ({exc}); run fm login, or pass --schedule FILE",
            tuple(warnings),
        )
    try:
        with EspnClient.for_league(configured, session) as client:
            read = client.pro_schedule()
    except EspnClientError as exc:
        return LoadedSchedule(None, f"{key}: the pro schedule could not be read from ESPN: {exc}", tuple(warnings))
    remember_schedule(store, read)
    return LoadedSchedule(read.data, f"fetched {stamp(read.as_of)}", tuple(warnings))


def _schedule_snapshot(store: Store, configured: League) -> RawSnapshotRow | None:
    """The newest captured pro schedule for the league's game and season (captures are season-level, not per league)."""
    marker = f"/games/{configured.game}/seasons/{configured.season}"
    rows = store.raw_snapshots.find(ESPN_SOURCE, SCHEDULE_KIND)
    return next((row for row in reversed(rows) if row.url is not None and marker in row.url), None)


def _read_schedule(path: Path) -> tuple[ProSchedule | None, str]:
    try:
        return ProSchedule.model_validate_json(path.read_bytes()), ""
    except FileNotFoundError:
        return None, "is gone from the cache"
    except (OSError, ValidationError, ValueError) as exc:
        return None, f"could not be read ({type(exc).__name__})"


def remember_schedule(store: Store, read: EspnRead[ProSchedule]) -> RawSnapshotRow | None:
    """Index a pro-schedule read's capture in ``raw_snapshots`` so later runs read it back from the cache."""
    capture = read.capture
    if capture is None:
        return None
    return store.raw_snapshots.insert(
        RawSnapshotRow(
            source=ESPN_SOURCE,
            kind=capture.kind,
            url=capture.url,
            params=dict[str, Any](capture.params),
            path=capture.relative_path,
            sha256=capture.sha256,
            size_bytes=capture.size_bytes,
            status_code=capture.status_code,
            fetched_at=capture.fetched_at,
        )
    )


# --- shared -----------------------------------------------------------------------------------------------------------


def _config() -> Config:
    try:
        return load_config()
    except ConfigError as exc:
        _fail(str(exc))


def _select(config: Config, keys: Sequence[str] | None) -> list[League]:
    if not keys:
        return list(config.leagues)
    known = {league.key: league for league in config.leagues}
    missing = [key for key in keys if key not in known]
    if missing:
        names = ", ".join(repr(key) for key in missing)
        _fail(f"no league {names} in config.toml; known: {', '.join(known) or 'none'}")
    return [known[key] for key in dict.fromkeys(keys)]


def _parse_as_of(text: str | None) -> datetime:
    if text is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        _fail(f"--as-of {text!r} is not an ISO 8601 timestamp (try 2026-10-04T15:00Z)")
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("status")(status)
    root.command("lineup")(lineup)
    root.command("waivers")(waivers)
