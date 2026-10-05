"""NBA injury status: ESPN's site injuries JSON (status) and the league's official injury report PDFs (pregame detail).

ESPN: ``https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries``, no key, send a browser UA. One entry
per injured player, grouped by team: ``status`` (``Out``, ``Day-To-Day``, ``Questionable``, ...), ``type`` (ESPN's
status code), ``details`` (body part, side, detail, ``returnDate`` and ``fantasyStatus``: ``OUT``, ``GTD`` or ``OFS``
for out for the season), RotoWire ``shortComment``/``longComment``, and the athlete with position and team. The
athlete object has no ``id`` field, so the ESPN player id is read from the player-card link (``/id/<id>/``), the
headshot URL or the ``uid`` (:func:`espn_athlete_id`). This is a public read of ESPN's site API, not the fantasy
league API, so it needs no session and never touches ``lm-api-*``.

Official report: ``https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<HH>(AM|PM).pdf``,
published on game days from early afternoon and refreshed about every 15 minutes, with the "reason" column
(``Injury/Illness - Left Ankle; Sprain``, ``Not With Team``, ``G League - Two-Way``) that ESPN's status lacks. Reading
a layout PDF is best effort (pdfplumber text lines; game date, time, matchup and team forward-filled down the page),
and the source is optional (DESIGN section 7), so :meth:`NbaInjuriesSource.official_report` degrades to an empty
report with the reason in ``warnings`` instead of raising.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, ClassVar, Unpack

import pdfplumber
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from fm.espn.ids import FBA, InjuryStatus
from fm.sources.base import Fetched, FetchOptions, HttpSource, SourceError, SourceSchemaError, parse_json
from fm.sources.nba_schedule import BROWSER_USER_AGENT

logger = logging.getLogger(__name__)

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
OFFICIAL_REPORT_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{day}_{time}.pdf"
OUT_FOR_SEASON = "OFS"
STATUS_WORDS = ("Out", "Doubtful", "Questionable", "Probable", "Available")
NOT_SUBMITTED = "NOT YET SUBMITTED"

TEAM_NAMES: tuple[str, ...] = tuple(
    sorted(
        (
            "Atlanta Hawks",
            "Boston Celtics",
            "Brooklyn Nets",
            "Charlotte Hornets",
            "Chicago Bulls",
            "Cleveland Cavaliers",
            "Dallas Mavericks",
            "Denver Nuggets",
            "Detroit Pistons",
            "Golden State Warriors",
            "Houston Rockets",
            "Indiana Pacers",
            "LA Clippers",
            "Los Angeles Clippers",
            "Los Angeles Lakers",
            "Memphis Grizzlies",
            "Miami Heat",
            "Milwaukee Bucks",
            "Minnesota Timberwolves",
            "New Orleans Pelicans",
            "New York Knicks",
            "Oklahoma City Thunder",
            "Orlando Magic",
            "Philadelphia 76ers",
            "Phoenix Suns",
            "Portland Trail Blazers",
            "Sacramento Kings",
            "San Antonio Spurs",
            "Toronto Raptors",
            "Utah Jazz",
            "Washington Wizards",
        ),
        key=len,
        reverse=True,
    )
)
"""Team names as the official report prints them, longest first so prefix matching never stops short."""

_ID_IN_LINK = re.compile(r"/id/(\d+)(?:/|$)")
_ID_IN_HEADSHOT = re.compile(r"/players/full/(\d+)\.png")
_ID_IN_UID = re.compile(r"~a:(\d+)")


def espn_athlete_id(athlete: Mapping[str, Any]) -> int | None:
    """The ESPN player id from an injuries ``athlete`` object: player-card link, then headshot URL, then ``uid``."""
    for link in athlete.get("links") or ():
        href = link.get("href") if isinstance(link, Mapping) else None
        if isinstance(href, str) and (match := _ID_IN_LINK.search(href) or _ID_IN_UID.search(href)):
            return int(match[1])
    headshot = athlete.get("headshot")
    href = headshot.get("href") if isinstance(headshot, Mapping) else None
    if isinstance(href, str) and (match := _ID_IN_HEADSHOT.search(href)):
        return int(match[1])
    uid = athlete.get("uid")
    if isinstance(uid, str) and (match := _ID_IN_UID.search(uid)):
        return int(match[1])
    return None


def _instant(value: object) -> datetime | None:
    """ESPN writes ``2026-09-21T19:50Z``."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if not isinstance(value, str):
        raise ValueError(f"not a timestamp: {value!r}")
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


class NbaInjury(BaseModel):
    """One ESPN injuries entry. ``status`` is ESPN's wording; :attr:`normalized` is the shared enum."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    espn_id: int | None = None
    name: str
    short_name: str | None = None
    position: str | None = None
    espn_team_id: int | None = None
    team_abbr: str | None = None
    """ESPN's abbreviation (``GS``, ``NY``, ``UTAH``); :attr:`tricode` gives the nba.com one."""
    team_name: str | None = None
    status: str
    status_code: str | None = None
    """``type.abbreviation``: ``O``, ``DD``, ``Q``, ``D``, ``P``."""
    fantasy_status: str | None = None
    """``details.fantasyStatus.abbreviation``: ``OUT``, ``GTD`` (game-time decision) or ``OFS`` (out for season)."""
    injury: str | None = None
    """Body part or illness (``Knee``, ``Ankle``, ``Undisclosed``)."""
    location: str | None = None
    side: str | None = None
    detail: str | None = None
    return_date: date | None = None
    short_comment: str | None = None
    long_comment: str | None = None
    updated: datetime | None = None
    news_id: str | None = None

    @field_validator(
        "short_name",
        "position",
        "team_abbr",
        "team_name",
        "status_code",
        "fantasy_status",
        "injury",
        "location",
        "side",
        "detail",
        "short_comment",
        "long_comment",
        "news_id",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("updated", mode="before")
    @classmethod
    def _updated(cls, value: object) -> datetime | None:
        return _instant(value)

    @property
    def normalized(self) -> InjuryStatus:
        """``Day-To-Day`` becomes ``DAY_TO_DAY``; unrecognized wording is ``UNKNOWN``, never a guess."""
        return FBA.injury_status(re.sub(r"[^A-Za-z0-9]+", "_", self.status))

    @property
    def tricode(self) -> str | None:
        """nba.com tricode through the ESPN pro-team id, so this joins to the schedule and stats sources."""
        return FBA.pro_team(self.espn_team_id) if self.espn_team_id is not None else None

    @property
    def out_for_season(self) -> bool:
        return self.fantasy_status == OUT_FOR_SEASON


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _injury(team: Mapping[str, Any], entry: Mapping[str, Any]) -> NbaInjury:
    athlete = _mapping(entry.get("athlete"))
    details = _mapping(entry.get("details"))
    kind = _mapping(entry.get("type"))
    fantasy = _mapping(details.get("fantasyStatus"))
    athlete_team = _mapping(athlete.get("team"))
    position = _mapping(athlete.get("position"))
    name = athlete.get("displayName") or " ".join(
        part for part in (athlete.get("firstName"), athlete.get("lastName")) if isinstance(part, str) and part
    )
    status = entry.get("status")
    if not isinstance(name, str) or not name.strip() or not isinstance(status, str) or not status.strip():
        raise ValueError("an injury entry needs a player name and a status")
    news_id = entry.get("id")
    return NbaInjury(
        espn_id=espn_athlete_id(athlete),
        name=name.strip(),
        short_name=athlete.get("shortName"),
        position=position.get("abbreviation"),
        espn_team_id=athlete_team.get("id", team.get("id")),
        team_abbr=athlete_team.get("abbreviation"),
        team_name=athlete_team.get("displayName") or team.get("displayName"),
        status=status.strip(),
        status_code=kind.get("abbreviation"),
        fantasy_status=fantasy.get("abbreviation"),
        injury=details.get("type"),
        location=details.get("location"),
        side=details.get("side"),
        detail=details.get("detail"),
        return_date=details.get("returnDate") or None,
        short_comment=entry.get("shortComment"),
        long_comment=entry.get("longComment"),
        updated=entry.get("date"),
        news_id=str(news_id) if news_id is not None else None,
    )


def parse_espn_injuries(payload: bytes) -> list[NbaInjury]:
    """Every entry across teams, in ESPN's order. Entries that fail validation are skipped and counted; a payload with
    entries but no valid one is a schema error. No injuries at all is a legitimate (if unlikely) empty list."""
    raw = parse_json(payload)
    if not isinstance(raw, dict) or not isinstance(raw.get("injuries"), list):
        raise SourceSchemaError("espn injuries: expected an object with an injuries list")
    injuries: list[NbaInjury] = []
    skipped = 0
    total = 0
    for team in raw["injuries"]:
        entries = team.get("injuries") if isinstance(team, dict) else None
        if not isinstance(entries, list):
            skipped += 1
            continue
        for entry in entries:
            total += 1
            if not isinstance(entry, dict):
                skipped += 1
                continue
            try:
                injuries.append(_injury(team, entry))
            except (ValidationError, ValueError, TypeError):
                skipped += 1
    if total and not injuries:
        raise SourceSchemaError(f"espn injuries: none of {total} entries validated")
    if skipped:
        logger.warning("espn injuries: skipped %d entries that did not validate", skipped)
    return injuries


# --- official report PDF ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OfficialInjuryEntry:
    """One player line of the league's report; game columns are forward-filled from the lines above."""

    game_day: date | None
    game_time: str | None
    """As printed: ``07:30 (ET)``."""
    matchup: str | None
    """As printed: ``HOU@OKC``."""
    team: str | None
    player: str
    """As printed: ``Smith Jr., Jabari``."""
    status: str
    """``Out``, ``Doubtful``, ``Questionable``, ``Probable`` or ``Available``."""
    reason: str

    @property
    def name(self) -> str:
        """``First Last`` (``Jabari Smith Jr.``) for matching against other sources."""
        last, sep, first = self.player.partition(", ")
        return f"{first} {last}".strip() if sep else self.player


@dataclass(frozen=True, slots=True)
class OfficialInjuryReport:
    entries: tuple[OfficialInjuryEntry, ...] = ()
    not_submitted: tuple[tuple[str | None, str], ...] = ()
    """``(matchup, team)`` for teams whose report was ``NOT YET SUBMITTED``."""

    def for_team(self, team: str) -> list[OfficialInjuryEntry]:
        return [entry for entry in self.entries if entry.team == team]


_DATE_RE = re.compile(r"^(?P<date>\d{2}/\d{2}/\d{4})\s+")
_TIME_RE = re.compile(r"^(?P<time>\d{1,2}:\d{2}\s*\(ET\))\s+")
_MATCHUP_RE = re.compile(r"^(?P<matchup>[A-Z]{3}@[A-Z]{3})\s+")
_ENTRY_RE = re.compile(rf"^(?P<player>\S.*?, .+?)\s+(?P<status>{'|'.join(STATUS_WORDS)})\b\s*(?P<reason>.*)$")
_SKIP_RE = re.compile(r"^(?:Injury Report:|Game Date\b|Page \d+ of \d+)")


def pdf_text_lines(payload: bytes) -> list[str]:
    """The PDF's text, one entry per printed line, pages in order. Raises when the payload is not a PDF."""
    with pdfplumber.open(io.BytesIO(payload)) as pdf:
        return [line for page in pdf.pages for line in (page.extract_text() or "").splitlines()]


def parse_official_report(payload: bytes) -> OfficialInjuryReport:
    """Best-effort rows from the report PDF. A payload that is not a PDF, or has no text, is a schema error."""
    lines = pdf_text_lines(payload)
    if not any(line.strip() for line in lines):
        raise SourceSchemaError("official injury report: the PDF has no text")
    return parse_report_lines(lines)


def parse_report_lines(lines: Iterable[str]) -> OfficialInjuryReport:
    """The report's text lines to entries. Game date, time, matchup and team carry down from the line that set them
    (the PDF prints each once per group); a line with no status word continues the previous entry's reason."""
    entries: list[OfficialInjuryEntry] = []
    not_submitted: list[tuple[str | None, str]] = []
    day: date | None = None
    game_time: str | None = None
    matchup: str | None = None
    team: str | None = None
    for raw_line in lines:
        text = raw_line.strip()
        if not text or _SKIP_RE.match(text):
            continue
        if match := _DATE_RE.match(text):
            day = datetime.strptime(match["date"], "%m/%d/%Y").date()
            game_time = matchup = team = None
            text = text[match.end() :]
        if match := _TIME_RE.match(text):
            game_time = match["time"]
            text = text[match.end() :]
        if match := _MATCHUP_RE.match(text):
            matchup = match["matchup"]
            team = None
            text = text[match.end() :]
        for name in TEAM_NAMES:
            if text == name or text.startswith(f"{name} "):
                team = name
                text = text[len(name) :].strip()
                break
        if text.startswith(NOT_SUBMITTED):
            if team is not None:
                not_submitted.append((matchup, team))
            continue
        if match := _ENTRY_RE.match(text):
            entries.append(
                OfficialInjuryEntry(
                    day, game_time, matchup, team, match["player"], match["status"], match["reason"].strip()
                )
            )
        elif entries and text:  # a reason wrapped onto the next line
            last = entries[-1]
            entries[-1] = dataclasses.replace(last, reason=f"{last.reason} {text}".strip())
    return OfficialInjuryReport(tuple(entries), tuple(not_submitted))


class NbaInjuriesSource(HttpSource):
    """ESPN status (raises when unavailable with nothing cached) plus the optional official report (degrades)."""

    name: ClassVar[str] = "nba_injuries"
    min_interval: ClassVar[float] = 0.5
    base_headers: ClassVar[Mapping[str, str]] = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "application/json, text/plain, */*",
    }
    ttl: ClassVar[Mapping[str, timedelta]] = {
        "espn": timedelta(minutes=15),
        "official": timedelta(minutes=15),  # the league refreshes it about that often on game days
    }

    def espn(self, **options: Unpack[FetchOptions]) -> Fetched[list[NbaInjury]]:
        return self.fetch(
            "espn",
            "nba",
            download=lambda: self.get_bytes(ESPN_INJURIES_URL),
            parse=parse_espn_injuries,
            meta={"url": ESPN_INJURIES_URL},
            **options,
        )

    def official_report(
        self, day: date, *, time_label: str = "05PM", **options: Unpack[FetchOptions]
    ) -> Fetched[OfficialInjuryReport]:
        """The report published for ``day`` at ``time_label`` (``01PM``, ``05PM``, ...). Degrades; see module docs."""
        if not re.fullmatch(r"[0-9A-Za-z_]+", time_label):
            raise ValueError(f"time_label must look like 05PM, got {time_label!r}")
        url = OFFICIAL_REPORT_URL.format(day=day.isoformat(), time=time_label)
        key = f"{day.isoformat()}_{time_label}"
        try:
            return self.fetch(
                "official",
                key,
                download=lambda: self.get_bytes(url, headers={"Accept": "application/pdf"}),
                parse=parse_official_report,
                ext="pdf",
                meta={"url": url},
                **options,
            )
        except SourceError as exc:
            logger.warning("%s/official[%s]: unavailable (%s); continuing without it", self.name, key, exc)
            return Fetched(
                OfficialInjuryReport(), self.clock(), self.name, "official", key, degraded=True, warnings=(str(exc),)
            )
