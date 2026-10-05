"""Turn the raw real-league captures into scrubbed, trimmed fixtures under ``tests/fixtures/espn/real/`` (ROADMAP #14).

Inputs live in the raw capture dir (``capture.py --raw``): ``reads.json`` (what ``reads`` saved and learned),
``calendar_<game>.json`` and ``webclient.json`` (from ``webclient``) and ``writes/*.json`` (from ``writes``). Each
fixture keeps ESPN's structure and drops only volume: fewer teams, players, matchups and stat entries, and no long text
blocks (player outlooks and rankings, notification settings). Every file is scrubbed by one :class:`scrub.Scrubber` per
league and checked with :meth:`scrub.Scrubber.leaks` before it is written (``webclient.json``, which belongs to no
league, is checked by every league's); a leak stops the run. ``index.json`` records, per file, the game, the views and
request that produced it, the model that parses it and how it was trimmed.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scrub import Scrubber

from fm.config import League

PLAYER_DROP = frozenset({"rankings", "outlooks", "draftRanksByRankType", "seasonOutlook", "lastVideoDate"})
TEAM_DROP = frozenset({"currentSimulationResults", "draftStrategy"})
ROSTER_KEYS = ("rosterForCurrentScoringPeriod", "rosterForMatchupPeriod", "rosterForMatchupPeriodDelayed")
WEBCLIENT_FILE = "webclient.json"
"""What ``capture.py webclient`` saves from the bundle; the fixture of the same name is shared by both games."""

type Trim = Callable[[dict[str, Any], "TrimContext"], dict[str, Any]]


class FixtureLeakError(RuntimeError):
    """A scrubbed fixture still held a real value; nothing more was written."""


@dataclass(frozen=True)
class TrimContext:
    season: int
    period: int | None
    matchup_period: int | None
    team_id: int
    opponent_id: int | None


@dataclass(frozen=True)
class Spec:
    """One fixture: which read-pass capture it comes from, the file it becomes, and how it is parsed and trimmed."""

    capture: str
    file: str
    kind: str
    trim: Trim
    note: str


# --- trims ------------------------------------------------------------------------------------------------------------


def _stat_rank(stat: Mapping[str, Any], ctx: TrimContext) -> int | None:
    """Which stat lines a trimmed player keeps, best first: this period's projection, this season's projection, this
    period's actual line, this season's actual line. Anything else (other seasons, rolling windows) is dropped."""
    if stat.get("seasonId") != ctx.season:
        return None
    period, source, split = stat.get("scoringPeriodId"), stat.get("statSourceId"), stat.get("statSplitTypeId")
    if ctx.period and period == ctx.period:
        return 0 if source == 1 else 2
    if source == 1 and period == 0 and split == 0:
        return 1
    if source == 0 and period == 0 and split == 0:
        return 3
    return None


def _player(player: Mapping[str, Any], ctx: TrimContext, max_stats: int | None) -> dict[str, Any]:
    out = {key: value for key, value in player.items() if key not in PLAYER_DROP}
    stats = list(player.get("stats") or [])
    if max_stats is not None:
        ranks = [(_stat_rank(stat, ctx), index) for index, stat in enumerate(stats)]
        kept = sorted((rank, index) for rank, index in ranks if rank is not None)[:max_stats]
        stats = [stats[index] for _, index in kept]
    out["stats"] = stats
    return out


def _pool_entry(entry: Mapping[str, Any], ctx: TrimContext, max_stats: int | None) -> dict[str, Any]:
    out = dict(entry)
    if isinstance(entry.get("player"), Mapping):
        out["player"] = _player(entry["player"], ctx, max_stats)
    return out


def _roster(roster: Mapping[str, Any], ctx: TrimContext, *, players: int | None, max_stats: int) -> dict[str, Any]:
    out = dict(roster)
    entries = []
    for entry in list(roster.get("entries") or [])[:players]:
        trimmed = dict(entry)
        if isinstance(entry.get("playerPoolEntry"), Mapping):
            trimmed["playerPoolEntry"] = _pool_entry(entry["playerPoolEntry"], ctx, max_stats)
        entries.append(trimmed)
    out["entries"] = entries
    return out


def trim_none(data: dict[str, Any], ctx: TrimContext) -> dict[str, Any]:
    return data


def trim_teams(data: dict[str, Any], ctx: TrimContext) -> dict[str, Any]:
    out = dict(data)
    out["members"] = [{**member, "notificationSettings": []} for member in data.get("members") or []]
    out["teams"] = [{k: v for k, v in team.items() if k not in TEAM_DROP} for team in data.get("teams") or []]
    out["schedule"] = [m for m in data.get("schedule") or [] if m.get("matchupPeriodId") == 1]
    return out


def trim_rosters(data: dict[str, Any], ctx: TrimContext) -> dict[str, Any]:
    """Our whole roster (lineup tests need every slot) and the first players of this period's opponent."""
    out = dict(data)
    teams = []
    for team in data.get("teams") or []:
        if team.get("id") == ctx.team_id:
            teams.append({**team, "roster": _roster(team.get("roster") or {}, ctx, players=None, max_stats=1)})
        elif team.get("id") == ctx.opponent_id:
            teams.append({**team, "roster": _roster(team.get("roster") or {}, ctx, players=3, max_stats=1)})
    out["teams"] = teams
    return out


def _has_team(matchup: Mapping[str, Any], team_id: int) -> bool:
    return team_id in {(matchup.get(side) or {}).get("teamId") for side in ("home", "away")}


def trim_matchups(data: dict[str, Any], ctx: TrimContext) -> dict[str, Any]:
    """Our matchup in the first, current and last scheduled periods plus one other current matchup; no lineups (the
    scoreboard fixture has those) and no teams block."""
    out = {key: value for key, value in data.items() if key != "teams"}
    schedule = list(data.get("schedule") or [])
    periods = sorted({m.get("matchupPeriodId") for m in schedule})
    current = ctx.matchup_period if ctx.matchup_period in periods else (periods[0] if periods else None)
    wanted = {periods[0], current, periods[-1]} if periods else set()
    kept: list[dict[str, Any]] = []
    other_kept = False
    for matchup in schedule:
        if matchup.get("matchupPeriodId") not in wanted:
            continue
        if not _has_team(matchup, ctx.team_id):
            if other_kept or matchup.get("matchupPeriodId") != current:
                continue
            other_kept = True
        trimmed = dict(matchup)
        for side in ("home", "away"):
            if isinstance(matchup.get(side), Mapping):
                trimmed[side] = {k: v for k, v in matchup[side].items() if k not in ROSTER_KEYS}
        kept.append(trimmed)
    out["schedule"] = kept
    return out


def trim_scoreboard(data: dict[str, Any], ctx: TrimContext) -> dict[str, Any]:
    """Our matchup only: our current lineup whole and the opponent's first 4 players (one stat line each), the
    matchup-period rosters cut to two players each."""
    out = dict(data)
    schedule = []
    for matchup in data.get("schedule") or []:
        if not _has_team(matchup, ctx.team_id):
            continue
        trimmed = dict(matchup)
        for side in ("home", "away"):
            if not isinstance(matchup.get(side), Mapping):
                continue
            values = dict(matchup[side])
            ours = values.get("teamId") == ctx.team_id
            for key in ROSTER_KEYS:
                if isinstance(values.get(key), Mapping):
                    players = (None if ours else 4) if key == "rosterForCurrentScoringPeriod" else 2
                    values[key] = _roster(values[key], ctx, players=players, max_stats=1)
            trimmed[side] = values
        schedule.append(trimmed)
    out["schedule"] = schedule
    return out


def _players(limit: int, *, max_stats: int | None = 3, all_stats_first: bool = False) -> Trim:
    """The first ``limit`` pool entries with at most ``max_stats`` stat lines each; ``all_stats_first`` keeps every
    line of the first entry (the evidence for stat-entry ids that the ranked trim drops, such as ``122026``)."""

    def trim(data: dict[str, Any], ctx: TrimContext) -> dict[str, Any]:
        out = dict(data)
        players = list(data.get("players") or [])[:limit]
        out["players"] = [
            _pool_entry(entry, ctx, None if all_stats_first and index == 0 else max_stats)
            for index, entry in enumerate(players)
        ]
        return out

    return trim


def trim_transactions(data: dict[str, Any], ctx: TrimContext) -> dict[str, Any]:
    out = dict(data)
    seen: Counter[tuple[Any, Any, Any]] = Counter()
    kept = []
    for transaction in data.get("transactions") or []:
        key = (transaction.get("type"), transaction.get("status"), transaction.get("executionType"))
        limit = 2 if transaction.get("type") == "DRAFT" else 3
        if seen[key] < limit:
            kept.append(transaction)
        seen[key] += 1
    out["transactions"] = kept
    return out


def trim_pro_schedule(data: dict[str, Any], ctx: TrimContext) -> dict[str, Any]:
    settings = dict(data.get("settings") or {})
    current = ctx.period or 1
    teams = []
    for team in settings.get("proTeams") or []:
        games = team.get("proGamesByScoringPeriod") or {}
        kept = {key: value for key, value in games.items() if int(key) == current}
        rest = {key: value for key, value in team.items() if key != "teamPlayersByPosition"}
        teams.append({**rest, "proGamesByScoringPeriod": kept})
    settings["proTeams"] = teams
    return {**data, "settings": settings}


def trim_game_state(data: dict[str, Any], ctx: TrimContext) -> dict[str, Any]:
    settings = data.get("settings") or {}
    keep = ("firstMatchupOverrides", "proScheduleAvailable", "statSettings", "teamAutoPilotSettings", "types")
    return {**data, "settings": {key: settings[key] for key in keep if key in settings}}


SPECS: tuple[Spec, ...] = (
    Spec("settings", "mSettings.json", "settings", trim_none, "whole response"),
    Spec(
        "teams",
        "mTeam+mStandings.json",
        "teams",
        trim_teams,
        "notificationSettings emptied; simulation and draft strategy dropped; schedule cut to matchup period 1",
    ),
    Spec(
        "rosters_current_period",
        "mRoster.json",
        "rosters",
        trim_rosters,
        "our whole roster and 3 players of this period's opponent; rankings/outlooks dropped; one stat line per "
        "player (this period's projection, else the season projection)",
    ),
    Spec(
        "matchups",
        "mMatchup.json",
        "matchups",
        trim_matchups,
        "teams block and lineups dropped; our matchup in the first, current and last scheduled periods plus one "
        "other current matchup",
    ),
    Spec(
        "scoreboard",
        "mMatchupScore+mScoreboard.json",
        "matchups",
        trim_scoreboard,
        "our matchup only; our current lineup whole, the opponent's first 4 players, 1 stat line each",
    ),
    Spec(
        "free_agents",
        "kona_player_info.json",
        "players",
        _players(4, all_stats_first=True),
        "first 4 pool players (FREEAGENT + WAIVERS, most owned first): every stat line of the first, at most 3 of "
        "each other",
    ),
    Spec("waivers", "kona_player_info_waivers.json", "players", _players(2), "first 2 players on waivers"),
    Spec("player_cards", "kona_playercard.json", "players", _players(2, max_stats=4), "first 2 cards, 4 stat lines"),
    Spec(
        "transactions_all_types",
        "mTransactions2.json",
        "transactions",
        trim_transactions,
        "every type including ROSTER/FUTURE_ROSTER/DRAFT; at most 3 per (type, status, executionType), 2 drafts",
    ),
    Spec(
        "pending_via_transactions2",
        "mTransactions2_waiver_trade.json",
        "transactions",
        trim_transactions,
        "WAIVER + TRADE_PROPOSAL filter, every status",
    ),
    Spec("pending_transactions", "mPendingTransactions.json", "transactions", trim_none, "whole response"),
    Spec(
        "pro_schedule",
        "proTeamSchedules_wl.json",
        "pro_schedule",
        trim_pro_schedule,
        "games of the current scoring period only; teamPlayersByPosition dropped",
    ),
    Spec(
        "game_kona_game_state",
        "kona_game_state.json",
        "game_state",
        trim_game_state,
        "draft schedule and client flags dropped",
    ),
    Spec("view_mStatus", "mStatus.json", "envelope", trim_none, "whole response"),
)


# --- driver -----------------------------------------------------------------------------------------------------------


def _latest(captures: list[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    matching = [capture for capture in captures if capture.get("name") == name and capture.get("saved")]
    return matching[-1] if matching else None


def _load(cache: Path, capture: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads((cache / str(capture["saved"])).read_text(encoding="utf-8"))


def _opponent(scoreboard: Mapping[str, Any], team_id: int) -> int | None:
    for matchup in scoreboard.get("schedule") or []:
        ids = [(matchup.get(side) or {}).get("teamId") for side in ("home", "away")]
        if team_id in ids:
            return next((team for team in ids if team not in (team_id, None)), None)
    return None


def _request(capture: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(capture.get("params") or {})
    return {key: params[key] for key in ("view", "scoringPeriodId", "filter") if key in params}


@dataclass
class FixtureWriter:
    raw: Path
    dest: Path
    index: dict[str, Any]

    def write(self, scrubber: Scrubber, relative: str, data: Any, entry: dict[str, Any]) -> None:
        scrubbed = scrubber.scrub(data)
        self._save(relative, scrubbed, entry, scrubber.leaks(scrubbed))

    def write_shared(self, scrubbers: list[Scrubber], relative: str, data: Any, entry: dict[str, Any]) -> None:
        """A file that belongs to no league (web-client code): written as is, after every league's leak check."""
        self._save(relative, data, entry, [problem for scrubber in scrubbers for problem in scrubber.leaks(data)])

    def _save(self, relative: str, data: Any, entry: dict[str, Any], problems: list[str]) -> None:
        if problems:
            raise FixtureLeakError(f"{relative}: {'; '.join(sorted(set(problems)))}")
        path = self.dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Compact, as ESPN sends it: the fixtures are machine data and indentation doubles their size.
        path.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
        self.index["files"][relative] = {**entry, "bytes": path.stat().st_size}


def make_fixtures(raw: Path, dest: Path, leagues: list[League]) -> dict[str, Any]:
    """Write every fixture for ``leagues`` and ``index.json``; returns the index."""
    reads = json.loads((raw / "reads.json").read_text(encoding="utf-8"))
    writer = FixtureWriter(raw, dest, {"captured_at": reads.get("captured_at"), "files": {}})
    cache = raw / "cache"
    scrubbers: list[Scrubber] = []
    for league in leagues:
        result = reads["leagues"][league.key]
        captures = result["captures"]
        settings_capture, teams_capture = _latest(captures, "settings"), _latest(captures, "teams")
        if settings_capture is None or teams_capture is None:
            raise FileNotFoundError(f"{league.key}: the read pass has no settings or teams capture")
        settings = _load(cache, settings_capture)
        scrubber = Scrubber.for_league(
            game=league.game,
            league_id=league.espn_league_id,
            our_team_id=league.team_id,
            teams_view=_load(cache, teams_capture),
            settings=settings,
        )
        scrubbers.append(scrubber)
        scoreboard = _latest(captures, "scoreboard")
        period = settings.get("scoringPeriodId")
        matchup_period = (settings.get("status") or {}).get("currentMatchupPeriod")
        ctx = TrimContext(
            season=league.season,
            period=period if isinstance(period, int) else None,
            matchup_period=matchup_period if isinstance(matchup_period, int) else None,
            team_id=league.team_id,
            opponent_id=_opponent(_load(cache, scoreboard), league.team_id) if scoreboard else None,
        )
        for spec in SPECS:
            capture = _latest(captures, spec.capture)
            if capture is None:
                continue
            entry = {
                "game": league.game,
                "kind": spec.kind,
                "view": capture.get("kind"),
                "request": _request(capture),
                "fetched_at": capture.get("fetched_at"),
                "trimmed": spec.note,
            }
            writer.write(scrubber, f"{league.game}/{spec.file}", spec.trim(_load(cache, capture), ctx), entry)
        _write_probes(writer, scrubber, league, result.get("probes") or {}, cache, ctx)
        period_type = (settings.get("settings") or {}).get("scheduleSettings", {}).get("periodTypeId")
        _write_calendar(writer, scrubber, league, raw, period_type)
        _write_captured_writes(writer, scrubber, league, raw)
    _write_webclient(writer, scrubbers, raw)
    index_path = dest / "index.json"
    index_path.write_text(json.dumps(writer.index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return writer.index


def _write_probes(
    writer: FixtureWriter, scrubber: Scrubber, league: League, probes: Mapping[str, Any], cache: Path, ctx: TrimContext
) -> None:
    summary: dict[str, Any] = {}
    slot = probes.get("filter_slot_ids") or {}
    summary["filter_slot_ids"] = {
        "omitted": {k: v for k, v in (slot.get("omitted") or {}).items() if k != "saved"},
        "empty_list": {k: v for k, v in (slot.get("empty_list") or {}).items() if k != "saved"},
        "same_players_same_order": slot.get("same_players_same_order"),
    }
    summary["top_scoring_period_zero"] = probes.get("top_scoring_period_zero")
    for name, outcome in probes.items():
        if name not in ("filter_slot_ids", "top_scoring_period_zero", "top_scoring_periods_two", "stat_entries"):
            summary[name] = {key: value for key, value in dict(outcome or {}).items() if key != "saved"}
    for name in ("top_scoring_periods_two", "stat_entries"):
        probe = dict(probes.get(name) or {})
        saved = probe.pop("saved", None)
        summary[name] = probe
        if saved:
            body = json.loads((cache / saved).read_text(encoding="utf-8"))
            card = _players(1, max_stats=None)(body, ctx)
            file = f"kona_playercard_{'top2' if name.startswith('top') else 'stat_entries'}.json"
            writer.write(
                scrubber,
                f"{league.game}/{file}",
                card,
                {
                    "game": league.game,
                    "kind": "players",
                    "view": "kona_playercard",
                    "request": {"filter": "see probes.json"},
                    "trimmed": "first card, every stat entry kept (the evidence for the stat-entry id and split facts)",
                },
            )
    writer.write(
        scrubber,
        f"{league.game}/probes.json",
        summary,
        {"game": league.game, "kind": "probes", "view": None, "trimmed": "probe outcomes from the read pass"},
    )


def _write_calendar(
    writer: FixtureWriter, scrubber: Scrubber, league: League, raw: Path, period_type: int | None
) -> None:
    """The web client's calendar for the game, keeping the league's own period type and the season-long one."""
    path = raw / f"calendar_{league.game}.json"
    if not path.exists():
        return
    calendar = json.loads(path.read_text(encoding="utf-8"))
    entry = {
        "game": league.game,
        "kind": "calendar",
        "view": "web client constants",
        "trimmed": "period types other than the league's own (scheduleSettings.periodTypeId) and the season-long one "
        "dropped",
    }
    calendar["periodTypes"] = [
        pt for pt in calendar.get("periodTypes") or [] if pt.get("id") == period_type or pt.get("seasonLong")
    ]
    writer.write(scrubber, f"{league.game}/calendar.json", calendar, entry)


def _write_webclient(writer: FixtureWriter, scrubbers: list[Scrubber], raw: Path) -> None:
    """The web client's transaction code and stat split labels (``capture.py webclient``), shared by both games."""
    path = raw / WEBCLIENT_FILE
    if not path.exists():
        return
    entry = {
        "game": None,
        "kind": "webclient",
        "view": "web client code",
        "trimmed": "verbatim excerpts of the bundle (transaction types, model, service, request functions) and each "
        "game's stat sources and split types",
    }
    writer.write_shared(scrubbers, WEBCLIENT_FILE, json.loads(path.read_text(encoding="utf-8")), entry)


def _write_captured_writes(writer: FixtureWriter, scrubber: Scrubber, league: League, raw: Path) -> None:
    folder = raw / "writes"
    if not folder.is_dir():
        return
    for path in sorted(folder.glob(f"{league.game}_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        entry = {"game": league.game, "kind": "write", "view": None, "trimmed": "aborted request, as captured"}
        writer.write(scrubber, f"{league.game}/write_{path.stem.split('_', 1)[1]}.json", record, entry)
