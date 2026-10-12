"""League discovery for ``fm setup`` (docs/PLUG_AND_PLAY.md section 2): which ESPN leagues and teams a signed-in
account has, from the fan API or from a pasted league URL, confirmed against the league's own views.

Read-only: every call here is a GET, and nothing here touches ``fm.executor`` or ``fm.browser``. The pieces:

- :func:`current_season` is the season ESPN labels "current" for a sport on a date. NFL seasons are named for the year
  of the September kickoff, so January to July belong to the previous year's season. NBA seasons are named for their
  end year, so from August onward the season is ``year + 1``.
- :func:`discover_leagues` sends one GET to ``https://fan.api.espn.com/apis/v2/fans/{SWID}`` with the session cookies
  and turns the fan profile into :class:`DiscoveredLeague` rows. **The fan API's response shape is not verified
  against a live account** (capturing one would put personal data in the repo), so the parser is defensive and has two
  paths. The structured path reads ``preferences[].metaData.entry``: ``groups[].groupId`` is the league,
  ``entryId`` the team, ``seasonId`` the season, and the sport comes from ``gameId`` (1 = ``ffl``, 3 = ``fba``),
  ``abbrev`` (``FFL`` / ``FBA``) or the ``/football/`` / ``/basketball/`` segment of ``entryURL``; ``groupName`` and
  ``entryMetadata.teamName`` (or ``entryLocation`` + ``entryNickname``) supply the names. Field names are matched
  without regard to case. When that path yields no NFL or NBA league, the deep walk scans every string in the JSON for
  a fantasy URL carrying ``leagueId=`` (plus ``teamId=`` / ``seasonId=`` when present) under ``/football/`` or
  ``/basketball/``, and yields leagues without names. Other ESPN games (hockey, baseball) and non-fantasy preferences
  are dropped, duplicates are merged, and only each sport's current season is kept unless ``all_seasons`` is set. Any
  failure (transport after one retry, an error status, a body that is not a JSON object) raises
  :class:`DiscoveryError`, and the caller falls back to pasted URLs.
- :func:`parse_league_url` turns any ``fantasy.espn.com/{football|basketball}/...`` URL with ``leagueId=`` (team,
  league, scoreboard, standings pages, or text containing one) into a :class:`LeagueRef`.
- :func:`resolve_team` finds the one team in an ``mTeam`` view whose ``owners`` holds the account's SWID.
- :func:`confirm_league` reads the league through :class:`fm.espn.client.EspnClient` (``mTeam`` + ``mStandings`` with
  ``mSettings`` in the same GET, for the league name) and fills in whatever a URL left out: the season, the team and
  both names. The client factory is injectable so tests stay offline.

Cookie values never reach a log line or an error message; SWIDs are account ids, not secrets, but are not logged
either.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl

import httpx
from pydantic import ValidationError

from fm.config import ESPN_GAME, Sport
from fm.espn.auth import EspnSession, is_auth_failure, normalize_swid
from fm.espn.client import EspnClient, EspnClientError, View
from fm.espn.ids import Game
from fm.espn.models import TeamsView

logger = logging.getLogger(__name__)

FAN_API_HOST = "fan.api.espn.com"
FAN_API_URL = f"https://{FAN_API_HOST}/apis/v2/fans"
"""The fan profile endpoint; the account's SWID (braces included) is the last path segment."""
DISCOVERY_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
DISCOVERY_ATTEMPTS = 2
"""One retry on a transport error. Error statuses are not retried: discovery is a convenience with a URL fallback."""
SEASON_ROLLOVER_MONTH = 8
"""August: the NFL season of the coming kickoff and the NBA season of the coming tip-off both start being "current"."""

type LeagueClientFactory = Callable[[Game, int, int, EspnSession], EspnClient]
"""``(game, league_id, season, session) -> EspnClient``; :class:`EspnClient` itself by default."""

_SPORT_OF_GAME: Mapping[Game, Sport] = MappingProxyType({Game(game): sport for sport, game in ESPN_GAME.items()})
_URL_GAMES: Mapping[str, Game] = MappingProxyType({"football": Game.FFL, "basketball": Game.FBA})
_ABBREV_GAMES: Mapping[str, Game] = MappingProxyType({"FFL": Game.FFL, "FBA": Game.FBA})
_LEAGUE_URL = re.compile(r"/(?P<sport>football|basketball)/[^\s<>\"'?#]*\?(?P<query>[^\s<>\"'#]*)", re.IGNORECASE)
_MAX_DEPTH = 64
"""Deep-walk bound: a fan profile is a few levels deep; anything deeper is not worth parsing."""


class DiscoveryError(RuntimeError):
    """Leagues could not be discovered, a URL could not be read, or a league could not be confirmed."""


@dataclass(frozen=True, slots=True)
class LeagueRef:
    """A league named by a URL or by hand. ``season`` and ``team_id`` are filled by :func:`confirm_league`."""

    sport: Sport
    espn_league_id: int
    season: int | None = None
    team_id: int | None = None

    @property
    def game(self) -> Game:
        return Game(ESPN_GAME[self.sport])


@dataclass(frozen=True, slots=True)
class DiscoveredLeague:
    """One league the account plays in, with as much as the source knew; names are ``None`` when it did not say."""

    sport: Sport
    espn_league_id: int
    season: int
    team_id: int | None = None
    league_name: str | None = None
    team_name: str | None = None

    @property
    def game(self) -> Game:
        return Game(ESPN_GAME[self.sport])

    @property
    def ref(self) -> LeagueRef:
        return LeagueRef(self.sport, self.espn_league_id, self.season, self.team_id)


# --- seasons ----------------------------------------------------------------------------------------------------------


def current_season(sport: Sport | Game | str, now: datetime | date | None = None) -> int:
    """ESPN's current season label for ``sport`` (``nfl``/``nba``, or a game key) on ``now`` (today by default).

    NFL: the year of the September kickoff, so January to July belong to the previous year. NBA: the end year, so from
    August onward it is ``year + 1``.
    """
    game = Game.coerce(sport)
    today = now if now is not None else datetime.now(UTC)
    rolled_over = today.month >= SEASON_ROLLOVER_MONTH
    if game is Game.FFL:
        return today.year if rolled_over else today.year - 1
    return today.year + 1 if rolled_over else today.year


# --- fan API ----------------------------------------------------------------------------------------------------------


def discover_leagues(
    session: EspnSession,
    *,
    client: httpx.Client | None = None,
    now: datetime | date | None = None,
    all_seasons: bool = False,
) -> list[DiscoveredLeague]:
    """The NFL and NBA leagues of the signed-in account, from one GET of its fan profile.

    ``client`` may be an injected ``httpx.Client`` (left open afterwards); otherwise one is opened for the call. Only
    the current season per sport is returned unless ``all_seasons`` is set. Raises :class:`DiscoveryError` on any
    failure, including a rejected session (the message says to run ``fm login``).
    """
    normalized = replace(session, swid=normalize_swid(session.swid))
    url = f"{FAN_API_URL}/{normalized.swid}"
    headers = {"Cookie": normalized.cookie_header(), "Accept": "application/json", "User-Agent": _user_agent()}
    response = _get(client, url, headers)
    status = response.status_code
    if status >= 400 and is_auth_failure(status, response.content):
        raise DiscoveryError(f"ESPN rejected the session (HTTP {status}) at {FAN_API_HOST}; run `fm login`")
    if status >= 400:
        raise DiscoveryError(f"{FAN_API_HOST} answered HTTP {status} to the fan profile request")
    try:
        data = json.loads(response.content)
    except ValueError as exc:
        raise DiscoveryError(f"{FAN_API_HOST} answered with a body that is not JSON ({exc})") from exc
    return parse_fan_profile(data, now=now, all_seasons=all_seasons)


def parse_fan_profile(
    data: object, *, now: datetime | date | None = None, all_seasons: bool = False
) -> list[DiscoveredLeague]:
    """Leagues in a decoded fan-profile body: the structured path first, the deep walk when it finds nothing.

    A body that is not a JSON object raises :class:`DiscoveryError`; an object with no NFL or NBA league in it is an
    empty list (a non-fantasy profile, or one with only other sports).
    """
    if not isinstance(data, Mapping):
        raise DiscoveryError(f"fan profile is not a JSON object (got {type(data).__name__})")
    found = [league for entry in _entries(data) for league in _leagues_of_entry(entry, now)]
    source = "structured"
    if not found:
        found = [_from_ref(ref, now) for node in _nodes(data) if isinstance(node, str) for ref in _refs_in(node)]
        source = "deep walk"
    leagues = _merge(found)
    if not all_seasons:
        leagues = [league for league in leagues if league.season == current_season(league.sport, now)]
    logger.info("discover: %d league(s) via the %s path, %d before the season filter", len(leagues), source, len(found))
    return leagues


def _get(client: httpx.Client | None, url: str, headers: Mapping[str, str]) -> httpx.Response:
    owns = client is None
    http = client if client is not None else httpx.Client(follow_redirects=False)
    failure = ""
    try:
        for attempt in range(1, DISCOVERY_ATTEMPTS + 1):
            try:
                return http.get(url, headers=headers, timeout=DISCOVERY_TIMEOUT)
            except httpx.TransportError as exc:
                failure = f"{type(exc).__name__}: {exc}".strip(": ")
                logger.warning("discover: %s (attempt %d/%d)", failure, attempt, DISCOVERY_ATTEMPTS)
    finally:
        if owns:
            http.close()
    raise DiscoveryError(f"could not reach {FAN_API_HOST} after {DISCOVERY_ATTEMPTS} attempts: {failure}")


def _user_agent() -> str:
    try:
        installed = distribution_version("espn-fantasy")
    except PackageNotFoundError:
        installed = "0+unknown"
    return f"espn-fantasy/{installed}"


# --- structured path --------------------------------------------------------------------------------------------------


def _entries(data: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Every ``preferences[].metaData.entry`` object, wherever a ``preferences`` list sits in the tree."""
    for node in _nodes(data):
        preferences = _field(node, "preferences") if isinstance(node, Mapping) else None
        if not isinstance(preferences, list):
            continue
        for preference in preferences:
            metadata = _field(preference, "metaData") if isinstance(preference, Mapping) else None
            entry = _field(metadata, "entry") if isinstance(metadata, Mapping) else None
            if isinstance(entry, Mapping):
                yield entry


def _leagues_of_entry(entry: Mapping[str, Any], now: datetime | date | None) -> Iterator[DiscoveredLeague]:
    url_ref = next(_refs_in(_text(_field(entry, "entryURL")) or ""), None)
    game = _entry_game(entry, url_ref)
    if game is None:
        return
    sport = _SPORT_OF_GAME[game]
    team_id = _positive_int(_field(entry, "entryId"))
    if team_id is None and url_ref is not None:
        team_id = url_ref.team_id
    season = _positive_int(_field(entry, "seasonId"))
    if season is None and url_ref is not None:
        season = url_ref.season
    if season is None:
        season = current_season(game, now)
    team_name = _team_name(entry)
    leagues: list[tuple[int, str | None]] = []
    groups = _field(entry, "groups")
    for group in groups if isinstance(groups, list) else ():
        league_id = _positive_int(_field(group, "groupId")) if isinstance(group, Mapping) else None
        if league_id is not None:
            leagues.append((league_id, _text(_field(group, "groupName"))))
    if not leagues and url_ref is not None:
        leagues.append((url_ref.espn_league_id, None))
    for league_id, league_name in leagues:
        yield DiscoveredLeague(sport, league_id, season, team_id, league_name, team_name)


def _entry_game(entry: Mapping[str, Any], url_ref: LeagueRef | None) -> Game | None:
    """The entry's game from ``gameId``, else ``abbrev``, else its URL. A present but unsupported value (hockey,
    baseball) settles it as another sport rather than falling through."""
    game_id = _positive_int(_field(entry, "gameId"))
    if game_id is not None:
        try:
            return Game.from_game_id(game_id)
        except ValueError:
            return None
    abbrev = _text(_field(entry, "abbrev"))
    if abbrev is not None:
        return _ABBREV_GAMES.get(abbrev.upper())
    return url_ref.game if url_ref is not None else None


def _team_name(entry: Mapping[str, Any]) -> str | None:
    metadata = _field(entry, "entryMetadata")
    if isinstance(metadata, Mapping):
        name = _text(_field(metadata, "teamName"))
        if name is not None:
            return name
    parts = (_text(_field(entry, "entryLocation")), _text(_field(entry, "entryNickname")))
    return " ".join(part for part in parts if part) or None


# --- deep walk --------------------------------------------------------------------------------------------------------


def _nodes(node: object, depth: int = 0) -> Iterator[object]:
    """Every value in a decoded JSON tree, depth-first, down to :data:`_MAX_DEPTH`."""
    yield node
    if depth >= _MAX_DEPTH:
        return
    if isinstance(node, Mapping):
        for value in node.values():
            yield from _nodes(value, depth + 1)
    elif isinstance(node, list | tuple):
        for item in node:
            yield from _nodes(item, depth + 1)


def _refs_in(text: str) -> Iterator[LeagueRef]:
    """Every fantasy league URL in ``text`` that names a league, in order."""
    for match in _LEAGUE_URL.finditer(text):
        params = {key.lower(): value for key, value in parse_qsl(match["query"])}
        league_id = _positive_int(params.get("leagueid"))
        if league_id is None:
            continue
        game = _URL_GAMES[match["sport"].lower()]
        yield LeagueRef(
            sport=_SPORT_OF_GAME[game],
            espn_league_id=league_id,
            season=_positive_int(params.get("seasonid")),
            team_id=_positive_int(params.get("teamid")),
        )


def _from_ref(ref: LeagueRef, now: datetime | date | None) -> DiscoveredLeague:
    season = ref.season if ref.season is not None else current_season(ref.sport, now)
    return DiscoveredLeague(ref.sport, ref.espn_league_id, season, ref.team_id)


def _merge(found: list[DiscoveredLeague]) -> list[DiscoveredLeague]:
    """One row per (sport, league, season), first seen first; later duplicates fill what the first left ``None``."""
    merged: dict[tuple[Sport, int, int], DiscoveredLeague] = {}
    for league in found:
        key = (league.sport, league.espn_league_id, league.season)
        first = merged.get(key)
        merged[key] = (
            league
            if first is None
            else replace(
                first,
                team_id=first.team_id if first.team_id is not None else league.team_id,
                league_name=first.league_name or league.league_name,
                team_name=first.team_name or league.team_name,
            )
        )
    return list(merged.values())


# --- field helpers ----------------------------------------------------------------------------------------------------


def _field(mapping: Mapping[str, Any], name: str) -> Any:
    """``mapping[name]`` matched without regard to key case (``entryURL`` and ``entryUrl`` are the same field)."""
    if name in mapping:
        return mapping[name]
    wanted = name.lower()
    return next((value for key, value in mapping.items() if isinstance(key, str) and key.lower() == wanted), None)


def _positive_int(value: object) -> int | None:
    """``value`` as a positive int when it is one, a whole float, or a digit string; otherwise ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip()) or None
    return None


def _text(value: object) -> str | None:
    """A non-empty stripped string, or ``None``."""
    if isinstance(value, str):
        return value.strip() or None
    return None


# --- URLs, teams and confirmation -------------------------------------------------------------------------------------


def parse_league_url(text: str) -> LeagueRef:
    """The league in a pasted ``fantasy.espn.com/{football|basketball}/...`` URL (or text holding one).

    Team, league, scoreboard and standings pages all carry ``leagueId=``; ``teamId=`` and ``seasonId=`` are kept when
    present. Raises :class:`DiscoveryError` for anything else, including other sports' URLs.
    """
    ref = next(_refs_in(text), None)
    if ref is None:
        raise DiscoveryError(
            "not an ESPN fantasy league URL; expected https://fantasy.espn.com/football/... or "
            "https://fantasy.espn.com/basketball/... with leagueId="
        )
    return ref


def resolve_team(teams: TeamsView, swid: str) -> int:
    """The id of the one team whose ``owners`` holds ``swid`` (braces and case normalized).

    Raises :class:`DiscoveryError` when no team or several teams match (a co-managed league needs the team id).
    """
    wanted = normalize_swid(swid).upper()
    matches = [team.id for team in teams.teams if any(normalize_swid(owner).upper() == wanted for owner in team.owners)]
    league = f"league {teams.league_id}" if teams.league_id is not None else "the league"
    if not matches:
        raise DiscoveryError(f"no team in {league} is owned by this account; pass the team id")
    if len(matches) > 1:
        raise DiscoveryError(f"teams {matches} in {league} all list this account as an owner; pass the team id")
    return matches[0]


def confirm_league(
    ref: LeagueRef,
    session: EspnSession,
    *,
    client_factory: LeagueClientFactory = EspnClient,
    now: datetime | date | None = None,
) -> DiscoveredLeague:
    """Read the league and fill in what ``ref`` left out: the season (current when missing), the team (by SWID when
    missing) and the league and team names.

    One GET of ``mTeam`` + ``mStandings`` + ``mSettings`` through ``client_factory(game, league_id, season, session)``.
    The league name comes from ``settings.name`` and is ``None`` if ESPN leaves the block out. Raises
    :class:`DiscoveryError` when the read fails, the body does not parse, or the team cannot be settled.
    """
    season = ref.season if ref.season is not None else current_season(ref.sport, now)
    label = f"{ref.sport.upper()} league {ref.espn_league_id} ({season})"
    try:
        with client_factory(ref.game, ref.espn_league_id, season, session) as client:
            read = client.get_view(View.TEAM, View.STANDINGS, View.SETTINGS)
    except EspnClientError as exc:
        raise DiscoveryError(f"could not read {label}: {exc}") from exc
    try:
        teams = TeamsView.model_validate(read.data)
    except ValidationError as exc:
        raise DiscoveryError(f"{label}: the team view did not parse ({exc.error_count()} problems)") from exc
    team_id = ref.team_id if ref.team_id is not None else resolve_team(teams, session.swid)
    try:
        team = teams.team(team_id)
    except KeyError as exc:
        raise DiscoveryError(f"{label}: {exc.args[0]}") from exc
    settings = _field(read.data, "settings")
    league_name = _text(_field(settings, "name")) if isinstance(settings, Mapping) else None
    confirmed = DiscoveredLeague(
        sport=ref.sport,
        espn_league_id=ref.espn_league_id,
        season=teams.season_id if teams.season_id is not None else season,
        team_id=team_id,
        league_name=league_name,
        team_name=team.display_name,
    )
    logger.info("discover: confirmed %s, team %d", label, team_id)
    return confirmed
