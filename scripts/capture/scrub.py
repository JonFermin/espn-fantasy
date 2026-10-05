"""Scrub real ESPN captures into committable fixtures (CLAUDE.md: no real manager names, cookies or ids).

One :class:`Scrubber` serves one league and is built from that league's ``mTeam`` and ``mSettings`` captures, so every
fixture of the league uses the same stable placeholders:

- the league id becomes :data:`LEAGUE_PLACEHOLDERS` for the game, also inside strings (a captured request's URL); the
  league name becomes ``Fixture <game> League``;
- team ids are permuted within the league's own id set (ours becomes the smallest id), so cross references and the
  real shape of the id set (NBA leagues have gaps) survive while our team id does not;
- member SWIDs (and any other braced GUID, such as ``creationInfo.source``) become ``{00000000-...-0000000000NN}``,
  ours first; manager names become ``Manager N``; team names ``Team N``, abbreviations ``TN``, logos a placeholder URL;
- free text a manager could have typed (trade ``comment``) is replaced.

Player ids, names and stats are public and stay. Transaction ids are random UUIDs and stay (``relatedTransactionId``
links must keep working). :meth:`Scrubber.leaks` then searches the scrubbed text for every real value it replaced;
``make_fixtures`` refuses to write a file with any hit.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

LEAGUE_PLACEHOLDERS: dict[str, int] = {"ffl": 1010101, "fba": 2020202}
GUID = re.compile(r"\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}")
PLACEHOLDER_GUID = re.compile(r"\{00000000-0000-0000-0000-[0-9]{12}\}")
LOGO_PLACEHOLDER = "https://example.invalid/fixture-team-logo.png"
TEAM_ID_KEYS = frozenset({"teamId", "onTeamId", "fromTeamId", "toTeamId", "rosterForTeamId"})
TEAM_ID_LIST_KEYS = frozenset({"pickOrder", "keeperOrder"})
TEAM_KEYED_MAPS = frozenset({"teamActions"})
SWID_KEYS = frozenset({"memberId", "primaryOwner", "source"})
GENERIC_DIVISION_NAMES = frozenset({"league", "league standings", "division"})


def swid_placeholder(number: int) -> str:
    return f"{{00000000-0000-0000-0000-{number:012d}}}"


@dataclass
class Scrubber:
    """Stable replacements for one league. Build it with :meth:`for_league`."""

    game: str
    league_id: int
    team_map: dict[int, int]
    swid_map: dict[str, str]
    member_numbers: dict[str, int]
    league_name: str | None
    sensitive: set[str] = field(default_factory=set)
    """Every real string replaced (names, abbreviations, SWIDs, the league name): what :meth:`leaks` looks for."""

    @classmethod
    def for_league(
        cls, *, game: str, league_id: int, our_team_id: int, teams_view: Mapping[str, Any], settings: Mapping[str, Any]
    ) -> Scrubber:
        teams = [team for team in teams_view.get("teams") or [] if isinstance(team, Mapping)]
        real_ids = sorted(int(team["id"]) for team in teams)
        if our_team_id not in real_ids:
            raise ValueError(f"team {our_team_id} is not in the league's teams")
        others = [team_id for team_id in real_ids if team_id != our_team_id]
        team_map = dict(zip([our_team_id, *others], real_ids, strict=True))

        ours = next(team for team in teams if int(team["id"]) == our_team_id)
        our_owners = {str(owner).upper() for owner in ours.get("owners") or []}
        members = [member for member in teams_view.get("members") or [] if isinstance(member, Mapping)]
        ordered = sorted(
            members, key=lambda member: (str(member.get("id")).upper() not in our_owners, members.index(member))
        )
        swid_map: dict[str, str] = {}
        numbers: dict[str, int] = {}
        for number, member in enumerate(ordered, start=1):
            swid = str(member["id"]).upper()
            swid_map[swid] = swid_placeholder(number)
            numbers[swid] = number

        sensitive: set[str] = set()
        for member in members:
            sensitive.add(str(member["id"]).upper())
            for key in ("displayName", "firstName", "lastName"):
                if isinstance(member.get(key), str) and member[key].strip():
                    sensitive.add(member[key].strip())
        for team in teams:
            for key in ("name", "abbrev", "location", "nickname"):
                if isinstance(team.get(key), str) and team[key].strip():
                    sensitive.add(team[key].strip())
        league_name = (settings.get("settings") or {}).get("name")
        if isinstance(league_name, str) and league_name.strip():
            sensitive.add(league_name.strip())
        return cls(
            game=game,
            league_id=league_id,
            team_map=team_map,
            swid_map=swid_map,
            member_numbers=numbers,
            league_name=league_name if isinstance(league_name, str) else None,
            sensitive=sensitive,
        )

    # --- replacement ------------------------------------------------------------------------------------------------

    @property
    def league_placeholder(self) -> int:
        return LEAGUE_PLACEHOLDERS[self.game]

    def team(self, value: Any) -> Any:
        """A team id mapped onto its placeholder; 0, -1 and unknown ids pass through."""
        if isinstance(value, bool) or not isinstance(value, int | str):
            return value
        try:
            number = int(value)
        except ValueError:
            return value
        mapped = self.team_map.get(number, number)
        return mapped if isinstance(value, int) else str(mapped)

    def swid(self, value: str) -> str:
        key = value.upper() if value.startswith("{") else "{" + value.upper() + "}"
        if key not in self.swid_map:
            self.swid_map[key] = swid_placeholder(len(self.swid_map) + 1)
            self.sensitive.add(key)
        mapped = self.swid_map[key]
        return mapped if value.startswith("{") else mapped.strip("{}")

    def scrub(self, data: Any) -> Any:
        """A scrubbed deep copy of a response body (or any part of one)."""
        return self._value(data, key=None, parent=None, top=True)

    def _value(self, value: Any, *, key: str | None, parent: str | None, top: bool = False) -> Any:
        if isinstance(value, Mapping):
            return self._object(value, key=key, parent=parent, top=top)
        if isinstance(value, list):
            if key in TEAM_ID_LIST_KEYS:
                return [self.team(item) for item in value]
            if key == "owners":
                return [self.swid(item) if isinstance(item, str) else item for item in value]
            return [self._value(item, key=key, parent=parent) for item in value]
        if isinstance(value, str):
            if GUID.fullmatch(value.strip()):
                return self.swid(value.strip())
            # A league id inside a string: the URL of a captured write request, a link.
            return re.sub(rf"(?<![0-9]){self.league_id}(?![0-9])", str(self.league_placeholder), value)
        return value

    def _object(self, data: Mapping[str, Any], *, key: str | None, parent: str | None, top: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        is_member = key == "members"
        is_team = key == "teams" and parent is None
        for name, value in data.items():
            if top and name == "id" and isinstance(value, int):
                out[name] = self.league_placeholder
            elif name == "leagueId" and isinstance(value, int | str):
                out[name] = self.league_placeholder
            elif is_member:
                out[name] = self._member_field(data, name, value)
            elif is_team and name in ("id", "name", "abbrev", "location", "nickname", "logo"):
                out[name] = self._team_field(data, name, value)
            elif name in TEAM_ID_KEYS:
                out[name] = self.team(value)
            elif name in TEAM_KEYED_MAPS and isinstance(value, Mapping):
                out[name] = {str(self.team(team)): action for team, action in value.items()}
            elif name in SWID_KEYS and isinstance(value, str) and GUID.fullmatch(value.strip()):
                out[name] = self.swid(value.strip())
            elif name == "comment" and isinstance(value, str) and value:
                out[name] = "comment removed by the fixture scrubber"
            elif name == "name" and key == "settings" and parent is None and isinstance(value, str):
                out[name] = f"Fixture {self.game} League"
            elif name == "name" and key == "divisions" and isinstance(value, str):
                out[name] = value if value.strip().lower() in GENERIC_DIVISION_NAMES else f"Division {data.get('id')}"
            else:
                child_parent = parent if parent is not None or top else key
                out[name] = self._value(value, key=name, parent=child_parent)
        return out

    def _member_field(self, member: Mapping[str, Any], name: str, value: Any) -> Any:
        swid = str(member.get("id", "")).upper()
        number = self.member_numbers.get(swid)
        if number is None and isinstance(member.get("id"), str):
            self.swid(member["id"])
            number = int(self.swid_map[swid].strip("{}").split("-")[-1])
        if name == "id" and isinstance(value, str):
            return self.swid(value)
        if name == "displayName":
            return f"Manager {number}"
        if name == "firstName":
            return "Manager"
        if name == "lastName":
            return str(number)
        return self._value(value, key=name, parent="members")

    def _team_field(self, team: Mapping[str, Any], name: str, value: Any) -> Any:
        placeholder = self.team(team.get("id"))
        if name == "id":
            return placeholder
        if name == "logo":
            return LOGO_PLACEHOLDER if value else value
        if not isinstance(value, str) or not value:
            return value
        return {
            "name": f"Team {placeholder}",
            "abbrev": f"T{placeholder}",
            "location": "Team",
            "nickname": str(placeholder),
        }[name]

    # --- verification -----------------------------------------------------------------------------------------------

    def leaks(self, data: Any, *, extra: Iterable[str] = ()) -> list[str]:
        """Where real values survive in ``data``: the league id, a SWID, or a name or abbreviation outside the public
        player and pro-team data. Messages name the JSON path, never the value."""
        text = json.dumps(data, ensure_ascii=False)
        lowered_text = text.lower()
        problems: list[str] = []
        if re.search(rf"(?<![0-9]){self.league_id}(?![0-9])", text):
            problems.append("the real league id appears")
        secrets = sorted({*self.sensitive, *extra})
        for secret in (s for s in secrets if GUID.fullmatch(s)):
            if secret.lower() in lowered_text or secret.strip("{}").lower() in lowered_text:
                problems.append("a real SWID appears")
        names = [s for s in secrets if not GUID.fullmatch(s)]
        for path, value in _strings(data):
            if _is_public(path) or value == LOGO_PLACEHOLDER:
                continue
            folded = value.strip().lower()
            for secret in names:
                if folded == secret.lower() or (len(secret) >= MIN_SUBSTRING_LENGTH and _word(secret).search(value)):
                    problems.append(f"a real name or abbreviation at {path}")
        for match in GUID.finditer(text):
            if not PLACEHOLDER_GUID.fullmatch(match.group(0)):
                problems.append("a braced GUID that is not a placeholder appears")
                break
        return problems


PUBLIC_KEYS = frozenset({"player", "proTeams", "proTeam"})
"""Subtrees of public data (players, pro teams) that the name check skips."""
MIN_SUBSTRING_LENGTH = 5


def _strings(data: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(data, Mapping):
        for key, value in data.items():
            yield from _strings(value, f"{path}.{key}")
    elif isinstance(data, list):
        for index, value in enumerate(data):
            yield from _strings(value, f"{path}[{index}]")
    elif isinstance(data, str):
        yield path, data


def _is_public(path: str) -> bool:
    parts = re.split(r"[.\[]", path)
    return any(part in PUBLIC_KEYS for part in parts)


def _word(value: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", re.IGNORECASE)
