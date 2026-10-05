"""Read-side capture for ROADMAP #14: every read view of each configured league, the probes that settle the open
unknowns (DESIGN sections 6.1, 6.3 and 9.3; ROADMAP #9), and league snapshots that back up the write guard by showing
the write flows changed nothing (:func:`snapshot`, :func:`compare_snapshots`).

Every request goes through :class:`fm.espn.client.EspnClient`, the read client under test, with raw bodies captured
under ``<raw>/cache/espn/...`` (unscrubbed: keep that directory out of the repo; ``make_fixtures`` scrubs what becomes a
fixture). The ``httpx`` client underneath carries two hooks from one :class:`RequestLog`: the request hook starts each
request at least :data:`MIN_INTERVAL_S` after the previous one, across every league's client, and the response hook
records each status and the response headers that bear on rate limiting and caching, never cookie values. Nothing here
writes to ESPN: the client only issues GETs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import httpx

from fm.config import League
from fm.espn.auth import EspnSession
from fm.espn.client import (
    DEFAULT_TIMEOUT,
    EspnClient,
    EspnClientError,
    EspnHttpError,
    EspnRead,
    View,
    player_filter,
    stat_entry_id,
)
from fm.espn.models import RostersView, Transaction

MIN_INTERVAL_S = 1.0
"""Seconds between requests during the capture (the spike's safety protocol)."""
MAX_ATTEMPTS = 2
BACKOFF_S = 2.0
CARD_PLAYERS = 6
"""Roster players whose cards are captured (the fixtures keep a few; the view takes up to 40)."""
RATE_HEADER_HINTS = ("rate", "limit", "retry", "quota", "throttl")
KEPT_RESPONSE_HEADERS = ("cache-control", "age", "x-cache", "via", "server", "content-type", "x-amz-cf-pop")


# --- request log ------------------------------------------------------------------------------------------------------


@dataclass
class RequestLog:
    """Paces every request the capture's clients send (one shared clock across leagues) and records each status and
    the response headers that bear on rate limiting."""

    min_interval_s: float = MIN_INTERVAL_S
    sleep: Callable[[float], object] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    entries: list[dict[str, Any]] = field(default_factory=list)
    header_names: set[str] = field(default_factory=set)
    starts: list[float] = field(default_factory=list)

    def on_request(self, request: httpx.Request) -> None:
        """httpx request hook: wait until ``min_interval_s`` has passed since the previous request started."""
        if self.starts:
            wait = self.min_interval_s - (self.monotonic() - self.starts[-1])
            if wait > 0:
                self.sleep(wait)
        self.starts.append(self.monotonic())
        request.extensions["fm_started"] = self.starts[-1]

    def hook(self, response: httpx.Response) -> None:
        names = {name.lower() for name in response.headers}
        self.header_names |= names
        kept = {
            name: value
            for name, value in response.headers.items()
            if name.lower() in KEPT_RESPONSE_HEADERS
            or name.lower().startswith("x-fantasy")
            or any(hint in name.lower() for hint in RATE_HEADER_HINTS)
        }
        cookies = [_cookie_attributes(raw) for raw in response.headers.get_list("set-cookie")]
        self.entries.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "url": str(response.request.url),
                "status": response.status_code,
                "headers_ms": _headers_ms(response, self.monotonic()),
                "headers": kept,
                "set_cookie": cookies,
            }
        )

    def summary(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for entry in self.entries:
            statuses[str(entry["status"])] = statuses.get(str(entry["status"]), 0) + 1
        gaps = [b - a for a, b in zip(self.starts, self.starts[1:], strict=False)]
        return {
            "requests": len(self.entries),
            "statuses": statuses,
            "min_start_gap_s": round(min(gaps), 3) if gaps else None,
            "rate_limit_headers": sorted(n for n in self.header_names if any(h in n for h in RATE_HEADER_HINTS)),
            "header_names": sorted(self.header_names),
            "set_cookie_names": sorted({c["name"] for e in self.entries for c in e["set_cookie"]}),
        }


def _headers_ms(response: httpx.Response, now: float) -> int | None:
    """Milliseconds from the paced send to the response headers (set by :meth:`RequestLog.on_request`)."""
    started = response.request.extensions.get("fm_started")
    return round((now - started) * 1000) if isinstance(started, float) else None


def _cookie_attributes(raw: str) -> dict[str, Any]:
    """A ``Set-Cookie`` header without its value: the name and the attributes only."""
    parts = [part.strip() for part in raw.split(";")]
    name = parts[0].split("=", 1)[0] if parts else ""
    attributes = {}
    for part in parts[1:]:
        key, _, value = part.partition("=")
        attributes[key.lower()] = value or True
    return {"name": name, "attributes": attributes}


def http_client(log: RequestLog) -> httpx.Client:
    """The client EspnClient would build, plus the logging hook."""
    try:
        installed = distribution_version("espn-fantasy")
    except Exception:  # pragma: no cover - the package is installed by uv sync
        installed = "0+unknown"
    return httpx.Client(
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": f"espn-fantasy/{installed}", "Accept": "application/json"},
        follow_redirects=False,
        event_hooks={"request": [log.on_request], "response": [log.hook]},
    )


@contextmanager
def league_client(league: League, session: EspnSession, raw_dir: Path, log: RequestLog) -> Iterator[EspnClient]:
    """An :class:`EspnClient` for one league on the logging, paced HTTP client; closes both afterwards (the read
    client leaves an injected HTTP client open)."""
    http = http_client(log)
    client = EspnClient.for_league(
        league,
        session,
        client=http,
        cache_root=raw_dir / "cache",
        min_interval_s=MIN_INTERVAL_S,
        max_attempts=MAX_ATTEMPTS,
        backoff_s=BACKOFF_S,
    )
    try:
        yield client
    finally:
        client.close()
        http.close()


# --- captures ---------------------------------------------------------------------------------------------------------


def capture_info(read: EspnRead[Any], name: str) -> dict[str, Any]:
    capture = read.capture
    if capture is None:
        return {"name": name, "saved": None}
    return {
        "name": name,
        "kind": capture.kind,
        "saved": capture.relative_path,
        "params": capture.params,
        "status": capture.status_code,
        "bytes": capture.size_bytes,
        "fetched_at": capture.fetched_at.isoformat(),
    }


@dataclass
class LeagueCapture:
    """What one league's read pass saved and learned."""

    league_key: str
    game: str
    captures: list[dict[str, Any]] = field(default_factory=list)
    probes: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def keep(self, name: str, fetch: Callable[[], EspnRead[Any]]) -> EspnRead[Any] | None:
        try:
            read = fetch()
        except EspnClientError as exc:
            self.failures.append(f"{name}: {exc}")
            return None
        self.captures.append(capture_info(read, name))
        return read

    def as_json(self) -> dict[str, Any]:
        return {
            "league_key": self.league_key,
            "game": self.game,
            "captures": self.captures,
            "probes": self.probes,
            "failures": self.failures,
        }


WEB_APP_GAME_VIEWS: tuple[str, ...] = ("kona_game_state",)
"""Season-level views the ESPN web app reads on every page (the current scoring period and its lock state)."""
WEB_APP_LEAGUE_VIEWS: tuple[str, ...] = ("mStatus",)
"""League views the web app's schedule page adds to ``mMatchupScore``/``mSettings``/``mTeam``."""


def capture_league(client: EspnClient, league: League, *, extra_views: Sequence[str] = ()) -> LeagueCapture:
    """Every read view of DESIGN section 6.1 for one league, the web app's extra views, then the probes."""
    result = LeagueCapture(league.key, client.game.value)
    settings = result.keep("settings", client.settings)
    current = settings.data.current_scoring_period if settings is not None else None
    matchup_period = settings.data.current_matchup_period if settings is not None else None
    result.keep("teams", client.teams)
    rosters = result.keep("rosters", client.rosters)
    if current is not None:
        result.keep("rosters_current_period", lambda: client.rosters(current))
    result.keep("matchups", client.matchups)
    if matchup_period is not None:
        result.keep("scoreboard", lambda: client.scoreboard(matchup_period, current))
    result.keep("free_agents", lambda: client.free_agents(current))
    result.keep("waivers", lambda: client.free_agents(current, statuses=("WAIVERS",), limit=25))
    ours: list[int] = []
    if rosters is not None:
        view: RostersView = rosters.data
        ours = list(view.roster(league.team_id).player_ids[:CARD_PLAYERS])
    if ours:
        result.keep("player_cards", lambda: client.player_cards(ours, scoring_period=current))
    result.keep("transactions", lambda: client.transactions(current))
    result.keep("transactions_all_types", lambda: client.transactions(current, types=_ALL_TRANSACTION_TYPES))
    result.keep("pending_transactions", client.pending_transactions)
    result.keep(
        "pending_via_transactions2",
        lambda: client.transactions(current, types=("WAIVER", "TRADE_PROPOSAL"), statuses=None),
    )
    result.keep("pro_schedule", client.pro_schedule)
    for view_name in WEB_APP_GAME_VIEWS:
        result.keep(f"game_{view_name}", lambda v=view_name: client.get_game_view(v))
    for view_name in (*WEB_APP_LEAGUE_VIEWS, *extra_views):
        result.keep(f"view_{view_name}", lambda v=view_name: client.get_view(v))
    result.probes = run_probes(client, current, ours)
    return result


_ALL_TRANSACTION_TYPES: tuple[str, ...] = (
    "FREEAGENT",
    "WAIVER",
    "WAIVER_ERROR",
    "TRADE_PROPOSAL",
    "TRADE_ACCEPT",
    "TRADE_DECLINE",
    "TRADE_VETO",
    "TRADE_UPHOLD",
    "ROSTER",
    "FUTURE_ROSTER",
    "DRAFT",
)


# --- probes -----------------------------------------------------------------------------------------------------------


def run_probes(client: EspnClient, current: int | None, player_ids: Sequence[int]) -> dict[str, Any]:
    """The experiments behind the resolved unknowns in docs/espn-api.md."""
    probes: dict[str, Any] = {}
    probes["filter_slot_ids"] = _probe_slot_filter(client, current)
    if player_ids:
        probes["top_scoring_period_zero"] = _probe_top_zero(client, player_ids)
        probes["top_scoring_periods_two"] = _probe_top_periods(client, current, player_ids, 2)
        probes["stat_entries"] = _probe_stat_entries(client, current, player_ids)
        probes["played_period"] = _probe_played_period(client, player_ids, probes["top_scoring_periods_two"])
    return probes


def _stat_index(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every stat entry in a player-card response by composite id, with its scope fields."""
    seen: dict[str, dict[str, Any]] = {}
    for entry in data.get("players") or []:
        for stat in (entry.get("player") or {}).get("stats") or []:
            seen.setdefault(
                str(stat.get("id")),
                {
                    "seasonId": stat.get("seasonId"),
                    "scoringPeriodId": stat.get("scoringPeriodId"),
                    "statSourceId": stat.get("statSourceId"),
                    "statSplitTypeId": stat.get("statSplitTypeId"),
                    "stat_count": len(stat.get("stats") or {}),
                },
            )
    return dict(sorted(seen.items()))


def _probe_top_periods(client: EspnClient, current: int | None, player_ids: Sequence[int], top: int) -> dict[str, Any]:
    """``filterStatsForTopScoringPeriodIds.value = top`` with no ``additionalValue``: which stat lines come back."""
    filter = player_filter(ids=player_ids[:2], top_scoring_periods=top)
    try:
        read = client.get_view(View.PLAYER_CARD, scoring_period=current, filter=filter, key=f"top{top}")
    except EspnHttpError as exc:
        return {"status": exc.status_code, "detail": exc.detail}
    return {"status": 200, "returned": _stat_index(read.data), "saved": capture_info(read, f"top{top}")["saved"]}


def _probe_slot_filter(client: EspnClient, current: int | None) -> dict[str, Any]:
    """``kona_player_info`` with ``filterSlotIds`` omitted (this client) vs ``[]`` (espn-api): same players?"""
    base = player_filter(statuses=("FREEAGENT", "WAIVERS"), limit=50, sort_percent_owned=True)
    with_empty = json.loads(json.dumps(base))
    with_empty["players"]["filterSlotIds"] = {"value": []}
    outcome: dict[str, Any] = {}
    ids: dict[str, list[int]] = {}
    for label, filter in (("omitted", base), ("empty_list", with_empty)):
        try:
            read = client.get_view(View.PLAYER_INFO, scoring_period=current, filter=filter, key=f"slots_{label}")
        except EspnHttpError as exc:
            outcome[label] = {"status": exc.status_code, "detail": exc.detail}
            continue
        players = read.data.get("players") or []
        ids[label] = [int(p["id"]) for p in players if isinstance(p, Mapping) and "id" in p]
        outcome[label] = {"status": 200, "players": len(ids[label]), "saved": capture_info(read, label)["saved"]}
    if len(ids) == 2:
        outcome["same_players_same_order"] = ids["omitted"] == ids["empty_list"]
    return outcome


def _probe_top_zero(client: EspnClient, player_ids: Sequence[int]) -> dict[str, Any]:
    """``filterStatsForTopScoringPeriodIds.value = 0``: the #9 note says HTTP 400."""
    filter = player_filter(ids=player_ids[:2], top_scoring_periods=0, stat_ids=[stat_entry_id(client.season)])
    try:
        read = client.get_view(View.PLAYER_CARD, filter=filter, key="top0")
    except EspnHttpError as exc:
        return {"status": exc.status_code, "detail": exc.detail}
    players = read.data.get("players") or []
    return {"status": 200, "players": len(players), "saved": capture_info(read, "top0")["saved"]}


def _probe_stat_entries(client: EspnClient, current: int | None, player_ids: Sequence[int]) -> dict[str, Any]:
    """Which composite ids and ``statSplitTypeId`` values come back for the season, a period, and rolling windows."""
    season = client.season
    previous = season - 1
    wanted = [
        f"00{season}",
        f"10{season}",
        f"00{previous}",
        f"10{previous}",
        f"01{previous}",
        f"02{previous}",
        f"03{previous}",
        f"01{season}",
        f"02{season}",
        f"03{season}",
    ]
    if current:
        wanted += [stat_entry_id(season, current), stat_entry_id(season, current, projected=True)]
    filter = player_filter(ids=player_ids[:3], top_scoring_periods=max(current or 1, 1), stat_ids=wanted)
    try:
        read = client.get_view(View.PLAYER_CARD, scoring_period=current, filter=filter, key="stat_entries")
    except EspnHttpError as exc:
        return {"requested": wanted, "status": exc.status_code, "detail": exc.detail}
    return {"requested": wanted, "returned": _stat_index(read.data), "saved": capture_info(read, "cards")["saved"]}


def _probe_played_period(client: EspnClient, player_ids: Sequence[int], top: Mapping[str, Any]) -> dict[str, Any]:
    """Does ESPN serve an already played period's actual line under a period-keyed id? Takes the latest actual line
    the top-2 probe returned (its season, period and split) and asks for ``01{season}{period}`` and the same id with
    that line's split (``05...`` in ``fba``); ``served`` lists which of them came back. The ``stat_entries`` probe
    could not settle this for ``fba``: it asked for a day not yet played."""
    returned = top.get("returned") or {}
    played = [
        (int(line["seasonId"]), int(line["scoringPeriodId"]), int(line["statSplitTypeId"]))
        for line in returned.values()
        if line.get("statSourceId") == 0 and (line.get("scoringPeriodId") or 0) > 0
    ]
    if not played:
        return {"skipped": "the top-2 probe returned no played period"}
    season, period, split = max(played)
    wanted = sorted({f"01{season}{period}", f"0{split}{season}{period}"})
    filter = player_filter(ids=player_ids[:2], top_scoring_periods=1, stat_ids=wanted)
    try:
        read = client.get_view(View.PLAYER_CARD, filter=filter, key="played_period")
    except EspnHttpError as exc:
        return {"requested": wanted, "status": exc.status_code, "detail": exc.detail}
    index = _stat_index(read.data)
    return {
        "requested": wanted,
        "season": season,
        "scoring_period": period,
        "returned": index,
        "served": sorted(set(wanted) & set(index)),
    }


# --- league snapshots -------------------------------------------------------------------------------------------------

SNAPSHOT_VERSION = 2
SNAPSHOT_TRANSACTION_TYPES: tuple[str, ...] = tuple(kind for kind in _ALL_TRANSACTION_TYPES if kind != "DRAFT")
"""Every ``mTransactions2`` type a write can create (the draft is over; its records would only add noise)."""


class SnapshotError(ValueError):
    """A snapshot cannot be compared (taken by an older version of this module)."""


def periods_left_in_matchup(matchup_periods: Mapping[int, Sequence[int]], current: int) -> list[int]:
    """The scoring periods from ``current`` to the end of the matchup that holds it (``[current]`` when none does):
    every period a lineup move made now can still change. ``matchup_periods`` maps each matchup to its scoring
    periods (``webclient.Calendar.matchup_days``)."""
    for periods in matchup_periods.values():
        if current in periods:
            return [period for period in periods if period >= current]
    return [current]


def _record(transaction: Transaction) -> dict[str, Any]:
    return {
        "type": transaction.type,
        "status": transaction.status,
        "execution_type": transaction.execution_type,
        "team_id": transaction.team_id,
        "team_ids": sorted(transaction.team_ids),
        "scoring_period": transaction.scoring_period_id,
        "related": transaction.related_transaction_id,
    }


def snapshot(client: EspnClient, team_id: int, periods: Sequence[int]) -> dict[str, Any]:
    """What a leaked write would change in one league, read before the write flows and again after them:

    - every team's roster (player and slot) in each of ``periods``, the rest of the current matchup, because a lineup
      move can target a later period (``FUTURE_ROSTER``);
    - our team's raw ``transactionCounter`` and ``tradeBlock`` from ``mTeam``;
    - every ``mTransactions2`` record of the types in :data:`SNAPSHOT_TRANSACTION_TYPES`, read for each of ``periods``,
      with its status: lineup moves (``ROSTER``, ``FUTURE_ROSTER``), adds, drops, claims, trade steps and the
      ``CANCEL`` records of offers and claims;
    - every pending item: our ``mPendingTransactions`` plus every ``PENDING`` record above (expired offers included).

    The caller pins ``periods`` (the first is the current one) so the re-read compares like for like.
    """
    if not periods:
        raise ValueError("a snapshot needs at least one scoring period")
    rosters: dict[str, dict[str, list[list[int]]]] = {}
    transactions: dict[str, dict[str, Any]] = {}
    for period in periods:
        view = client.rosters(period).data
        rosters[str(period)] = {
            str(team.team_id): sorted([entry.player_id, entry.lineup_slot_id] for entry in team.entries)
            for team in view.teams
        }
        for transaction in client.transactions(period, types=SNAPSHOT_TRANSACTION_TYPES).data.transactions:
            transactions[transaction.id] = _record(transaction)
    teams = client.get_view(View.TEAM, key="snapshot").data.get("teams") or []
    ours = next((team for team in teams if isinstance(team, Mapping) and team.get("id") == team_id), None)
    if ours is None:
        raise KeyError(f"team {team_id} is not in the league")
    pending = {t.id: _record(t) for t in client.pending_transactions(periods[0]).data.transactions}
    pending.update({tid: record for tid, record in transactions.items() if record["status"] == "PENDING"})
    return {
        "version": SNAPSHOT_VERSION,
        "taken_at": datetime.now(UTC).isoformat(),
        "team_id": team_id,
        "periods": list(periods),
        "rosters": rosters,
        "team": {"transactionCounter": ours.get("transactionCounter"), "tradeBlock": ours.get("tradeBlock")},
        "transactions": transactions,
        "pending": pending,
    }


def _touches(record: Mapping[str, Any] | None, team_id: int) -> bool:
    return record is not None and (record.get("team_id") == team_id or team_id in (record.get("team_ids") or ()))


def _describe(transaction_id: str, record: Mapping[str, Any]) -> str:
    kind = str(record.get("type"))
    if record.get("execution_type") not in (None, "EXECUTE"):
        kind += f"/{record['execution_type']}"
    return (
        f"{kind} {transaction_id} ({record.get('status')}, period {record.get('scoring_period')}, "
        f"by team {record.get('team_id')}, touching teams {record.get('team_ids')})"
    )


def _compare_records(
    label: str,
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    team_id: int,
    now: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    """New, vanished and changed records by id; ``now`` (every record after) says what a vanished one became."""
    problems: list[str] = []
    notes: list[str] = []
    for transaction_id in sorted(set(before) | set(after)):
        old, new = before.get(transaction_id), after.get(transaction_id)
        if old == new:
            continue
        if old is not None and new is not None:
            changed = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
            line = f"{label} changed ({', '.join(changed)}): {_describe(transaction_id, old)} -> {new.get('status')}"
        elif new is not None:
            line = f"new {label} {_describe(transaction_id, new)}"
        elif old is not None:
            became = now.get(transaction_id)
            status = f"now {became.get('status')}" if became is not None else "no longer listed"
            line = f"{label} gone ({status}): {_describe(transaction_id, old)}"
        else:
            continue
        (problems if _touches(old, team_id) or _touches(new, team_id) else notes).append(line)
    return problems, notes


def compare_snapshots(before: Mapping[str, Any], after: Mapping[str, Any], team_id: int) -> tuple[list[str], list[str]]:
    """``(problems, notes)`` between two :func:`snapshot` results of one league.

    A problem is anything of ours that differs: our roster in any snapshot period, our ``transactionCounter`` or
    ``tradeBlock``, and any transaction record or pending item touching our team (as actor or in an item) that is new,
    gone, or changed in status or shape. That covers a leaked lineup move for a later day, an add, drop or claim, a
    proposal, an accept, decline or veto, and a cancel of an existing offer or claim (ESPN records a cancel as a new
    ``CANCEL`` record and the claim leaves ``mPendingTransactions``). Other managers keep playing while the capture
    runs, so changes that do not touch our team are notes. A pending offer that expires during the run also shows up
    (as a new ``CANCEL`` record touching both teams); it is reported as a problem to look at, not filtered out.

    Not covered: periods outside the snapshot (the caller reads the rest of the current matchup) and league-manager
    actions, which this account never takes (``isLeagueManager`` is always false).
    """
    for snap in (before, after):
        if snap.get("version") != SNAPSHOT_VERSION:
            raise SnapshotError("snapshot from an older capture.py; run `capture.py snapshot` again")
    problems: list[str] = []
    notes: list[str] = []
    ours = str(team_id)
    for period in before["periods"]:
        old_teams = before["rosters"].get(str(period), {})
        new_teams = after["rosters"].get(str(period))
        if new_teams is None:
            problems.append(f"period {period}: not re-read")
            continue
        for team in sorted(set(old_teams) | set(new_teams), key=int):
            old = {tuple(item) for item in old_teams.get(team, [])}
            new = {tuple(item) for item in new_teams.get(team, [])}
            if old != new:
                line = f"period {period}, team {team}: removed {sorted(old - new)}, added {sorted(new - old)}"
                (problems if team == ours else notes).append(line)
    for key in ("transactionCounter", "tradeBlock"):
        if before["team"].get(key) != after["team"].get(key):
            problems.append(f"our {key} changed: {before['team'].get(key)} -> {after['team'].get(key)}")
    for label, field_name in (("transaction", "transactions"), ("pending", "pending")):
        found = _compare_records(label, before[field_name], after[field_name], team_id, after["transactions"])
        problems += found[0]
        notes += found[1]
    return problems, notes
