"""``fm status``: one block per configured league, from the store alone."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from fm.config import League
from fm.espn.settings import LeagueSettings
from fm.proposals import parse_payload
from fm.render import INDENT, points, stamp
from fm.store import LeagueRow, ProposalRow, TeamRow


@dataclass(frozen=True, slots=True)
class LeagueStatus:
    """What ``fm status`` knows about one configured league: the store's rows (``None`` before the first sync)."""

    configured: League
    league: LeagueRow | None = None
    settings: LeagueSettings | None = None
    team: TeamRow | None = None
    roster_period: int | None = None
    schedule_captured: datetime | None = None
    schedule_periods: tuple[int, ...] = ()
    open_proposals: Sequence[ProposalRow] = ()


def status_lines(statuses: Sequence[LeagueStatus], *, paused: str | None, profile: bool) -> list[str]:
    """The whole report: the kill switch and the browser profile first, then each league."""
    lines: list[str] = []
    if paused is not None:
        lines.append(f"PAUSED: {paused}; run fm resume")
    lines.append("browser profile: present" if profile else "browser profile: missing; run fm login before fm sync")
    for status in statuses:
        lines.extend(league_lines(status))
    return lines


def league_lines(status: LeagueStatus) -> list[str]:
    configured, league = status.configured, status.league
    head = f"{configured.key}: {configured.sport.upper()} {configured.season}, ESPN league {configured.espn_league_id}"
    if league is None:
        return [f"{head}: not synced; run fm sync"]
    lines = [f"{head}: {league.name or 'unnamed league'}"]
    team = status.team
    if team is not None:
        record = f"{team.wins}-{team.losses}" + (f"-{team.ties}" if team.ties else "")
        lines.append(f"{INDENT}team {team.team_id} {team.name}: {record}, {points(team.points_for)} points for")
    else:
        lines.append(f"{INDENT}team {configured.team_id}: not in the synced standings")
    period = status.roster_period
    roster = "no roster snapshot" if period is None else f"roster for scoring period {period}"
    lines.append(f"{INDENT}synced {stamp(league.as_of)}: {roster}")
    settings = status.settings
    if settings is not None and settings.acquisition.uses_faab and team is not None:
        budget = settings.acquisition.budget
        total = f" of ${budget}" if budget is not None else ""
        lines.append(f"{INDENT}FAAB: ${team.acquisition_budget_spent}{total} spent")
    if status.schedule_captured is None:
        lines.append(f"{INDENT}pro schedule: none captured; fm lineup fetches it (fm login) or takes --schedule FILE")
    else:
        periods = status.schedule_periods
        span = f"periods {periods[0]}-{periods[-1]}" if periods else "no periods"
        lines.append(f"{INDENT}pro schedule: captured {stamp(status.schedule_captured)}, {span}")
    open_rows = status.open_proposals
    if not open_rows:
        lines.append(f"{INDENT}open proposals: none")
        return lines
    lines.append(f"{INDENT}open proposals: {len(open_rows)} (fm proposals list)")
    for row in open_rows:
        lines.append(f"{INDENT * 2}{proposal_line(row)}")
    return lines


def proposal_line(row: ProposalRow) -> str:
    """One open proposal with its rationale, which carries the engine's headline numbers."""
    text = f"#{row.row_id} {row.kind} ({row.status}, policy {row.policy}), due {stamp(row.deadline)}"
    return f"{text}: {row.rationale or parse_payload(row).summary()}"
