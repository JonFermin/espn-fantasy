"""Rebuild the fixture home (``state.db`` and ``cache/``) from the hand-built ESPN fixtures.

    uv run python tests/fixtures/home/build.py            # rewrites tests/fixtures/home/state.db and cache/
    uv run python tests/fixtures/home/build.py DIR        # builds into DIR instead (tests do this into a temp dir)

The home is what ``FM_CONFIG_DIR=tests/fixtures/home FM_CACHE_DIR=tests/fixtures/home/cache`` points the CLI at: the
hand-built PPR league ``1234567`` (``tests/fixtures/espn/ffl_*.json``, week 4 of 2026, our team 1) synced by the real
sync job (:func:`fm.jobs.sync.sync_league`) through an ``httpx.MockTransport``, so the store, the captured pool pages
and the captured pro schedule (``tests/fixtures/sports/ffl_pro_schedule_2026.json``) are exactly what ``fm sync`` and
``fm lineup`` would have left behind. Every read is stamped :data:`SYNCED` (Sunday 11 a.m. ET, before the early games)
so the build is reproducible. No network, no browser, no real league: the fixtures carry ``Fixture Team N`` and
``Manager N`` only.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

HOME = Path(__file__).resolve().parent
FIXTURES = HOME.parents[0]
ESPN = FIXTURES / "espn"
SCHEDULE = FIXTURES / "sports" / "ffl_pro_schedule_2026.json"
SYNCED = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
"""When the fixture sync read the league: Sunday 11 a.m. ET of week 4, Thursday night played, nothing else started."""
LEAGUE_KEY = "nfl"


def espn_json(name: str) -> dict[str, Any]:
    return json.loads((ESPN / name).read_text(encoding="utf-8"))


def card_pool() -> dict[int, dict[str, Any]]:
    """Every player the NFL fixtures carry, by id, so a player-card request is answered per id."""
    cards: dict[int, dict[str, Any]] = {}
    for team in espn_json("ffl_rosters_week4.json")["teams"]:
        for entry in team["roster"]["entries"]:
            cards[entry["playerId"]] = {
                **entry["playerPoolEntry"],
                "id": entry["playerId"],
                "onTeamId": team["id"],
                "status": "ONTEAM",
            }
    for name in ("ffl_free_agents_week4.json", "ffl_player_cards_week4.json"):
        for entry in espn_json(name)["players"]:
            cards[entry["id"]] = entry
    return cards


def espn_views(request: httpx.Request) -> httpx.Response:
    """The hand-built league's read views and the season's pro schedule, as ESPN would answer them."""
    from fm.espn.client import FILTER_HEADER

    views = "+".join(request.url.params.get_list("view"))
    query = json.loads(request.headers.get(FILTER_HEADER, "{}")).get("players", {})
    if views == "mSettings":
        body = espn_json("ffl_settings_ppr.json")
    elif views == "mTeam+mStandings":
        body = espn_json("ffl_teams.json")
    elif views == "mRoster":
        body = espn_json("ffl_rosters_week4.json")
    elif views == "kona_player_info":
        body = espn_json("ffl_free_agents_week4.json") if not query.get("offset") else {"players": []}
    elif views == "kona_playercard":
        cards = card_pool()
        body = {"players": [cards[espn_id] for espn_id in query["filterIds"]["value"] if espn_id in cards]}
    elif views == "proTeamSchedules_wl":
        body = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    else:
        return httpx.Response(404, json={"details": [{"type": "UNEXPECTED", "message": views}]})
    return httpx.Response(200, json=body)


def build(home: Path) -> Path:
    """Build ``home/state.db`` and ``home/cache`` from the fixtures; returns the state database's path."""
    os.environ["FM_CONFIG_DIR"] = str(home)
    os.environ["FM_CACHE_DIR"] = str(home / "cache")
    # Imported after the environment is set so fm.paths resolves into the home being built.
    from fm.commands.advise import remember_schedule
    from fm.config import load_config
    from fm.espn.client import EspnClient
    from fm.jobs.sync import sync_league
    from fm.store import Store

    home.mkdir(parents=True, exist_ok=True)
    if home != HOME:
        shutil.copy(HOME / "config.toml", home / "config.toml")
    for stale in ("state.db", "state.db-wal", "state.db-shm"):
        (home / stale).unlink(missing_ok=True)
    shutil.rmtree(home / "cache", ignore_errors=True)
    config = load_config(home / "config.toml", environ={})
    league = config.league(LEAGUE_KEY)
    with (
        httpx.Client(transport=httpx.MockTransport(espn_views)) as http,
        EspnClient.for_league(league, None, client=http, min_interval_s=0.0, clock=lambda: SYNCED) as client,
        Store.open(home / "state.db") as store,
    ):
        sync_league(store, league, client)
        row = store.leagues.by_key(LEAGUE_KEY)
        assert row is not None
        remember_schedule(store, client.pro_schedule())
    for leftover in ("state.db-wal", "state.db-shm"):
        (home / leftover).unlink(missing_ok=True)
    return home / "state.db"


if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HOME
    print(build(target))
