"""Player news: ESPN's news API and RotoWire's RSS feed, polled at most every 10 minutes and deduplicated (DESIGN 7).

ESPN: ``https://site.api.espn.com/apis/site/v2/sports/{football/nfl|basketball/nba}/news``, no key, send a browser UA.
A public read of ESPN's site API, never the fantasy league API, so it needs no session. Each article carries an ``id``,
``headline``, ``description``, ``published`` and ``categories``; the ``athlete`` categories hold ESPN athlete ids,
which are the fantasy player ids, so an ESPN item names its players exactly. ESPN's per-player fantasy feed
(``apis/fantasy/v2/games/{game}/news/players?playerId=``, RotoWire's blurbs with the player id) answers one player per
request and HTTP 500 without one, so it is no feed and is not read.

RotoWire: ``https://www.rotowire.com/rss/news.php?sport=NFL|NBA``. Each ``<item>`` has a ``guid`` (``nfl640907``), a
``title`` of the form ``Player Name: headline``, a player-page ``link``, a ``description`` ending in a "Visit
RotoWire.com" line (dropped), and a ``pubDate`` on a 12-hour clock in US Pacific time (``Mon, 05 Oct 2026 3:42:00 PM
PDT``). feedparser reads that date wrongly (03:42 and no offset), so :func:`parse_rss_date` parses the raw string. An
item names its player only by the title, which :func:`news_subject` extracts; ``fm.model.relevance`` matches it to an
ESPN id. The feed carries just the latest few items (five when the fixtures were captured), so a burst bigger than
that between two polls is partly missed; ESPN's feed covers the major news either way.

Polling: both feeds keep their raw payload in the source cache with a 10-minute TTL, and :data:`POLL_INTERVAL` is a
floor, not a default: while the cached copy is younger, every call is served from it, whatever ``force``, ``max_age``
or ``fresh_since`` say (DESIGN section 7 lists RotoWire's RSS as free to poll at most every 10 minutes, and its
channel ``<ttl>`` is 10; ESPN's feed gets the same courtesy). News is an enrichment, so a feed never raises for an
upstream failure: it serves the last good copy as ``stale``, or an empty ``degraded`` result with the reason in
``warnings``.

Dedupe: an item is the pair (``source``, ``external_id``), which is also what ``news_items`` keeps unique, and polls
overlap, so the same item comes back many times. :func:`dedupe` keeps the first of each pair and also drops a repeat
of the same story under another id or source: the same :func:`fingerprint` (title and body, normalized) published
within :data:`DUPLICATE_WINDOW` of the copy kept, which is the earliest. Two reports of one event in different words
(ESPN's "Max Strus (foot) out at least 4 weeks" and RotoWire's "Max Strus: Will be re-evaluated in four weeks") are
different items: a later report can carry news an earlier one lacked, so nothing short of a repeat is dropped.
``fm.model.relevance.ingest_news`` applies the same rule against what is already stored.
"""

from __future__ import annotations

import email.utils
import hashlib
import html
import logging
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Final, Unpack

import feedparser

from fm.config import Sport
from fm.espn.ids import Game
from fm.sources.base import (
    Fetched,
    FetchOptions,
    HttpSource,
    SourceError,
    SourceSchemaError,
    parse_json,
)
from fm.sources.odds import BROWSER_USER_AGENT, SITE_PATHS

logger = logging.getLogger(__name__)

POLL_INTERVAL: Final = timedelta(minutes=10)
"""The shortest time between two downloads of one feed (DESIGN section 7: RotoWire RSS, poll at most every 10 min)."""
DUPLICATE_WINDOW: Final = timedelta(days=3)
"""Two items with the same text are one story when published at most this far apart; further apart, a repeated
headline is new news (RotoWire's "Out for Sunday" in two different weeks)."""
ESPN_NEWS_LIMIT: Final = 50
"""Articles asked of ESPN per poll; ten minutes of news is far fewer."""
ESPN_NEWS_URL: Final = "https://site.api.espn.com/apis/site/v2/sports/{path}/news"
ROTOWIRE_RSS_URL: Final = "https://www.rotowire.com/rss/news.php"
ESPN_NEWS: Final = "espn_news"
ROTOWIRE: Final = "rotowire"
"""``source`` of the items each feed yields, and the cache directory of its raw payloads."""

ESPN_LEAGUE_IDS: Mapping[Sport, int] = {"nfl": 28, "nba": 46}
"""ESPN's league id, which an article's categories carry as ``sportId``: an athlete tagged from another league (an NFL
story about a courtside NBA star) is not one of the feed's players."""

_ROTOWIRE_FOOTER = re.compile(r"\s*Visit RotoWire\.com for more analysis on this update\.?\s*$", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_IN_WORD = re.compile(r"[.'’‘ʼ`]")
_WORD = re.compile(r"[^\W_]+")
_CLOCK_12H = re.compile(r"\b(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?\s*(?P<half>[AaPp][Mm])\b")
_ATHLETE_UID = re.compile(r"~a:(\d+)")
_GUID_SPORT = re.compile(r"^([a-z]+)\d")


def _sport(value: Sport | Game | str) -> Sport:
    """``nfl`` / ``nba`` from a sport, an ESPN game key or a ``Game``. Raises ``ValueError``."""
    return "nfl" if Game.coerce(value) is Game.FFL else "nba"


def _text(value: object) -> str | None:
    """A string with its whitespace collapsed; ``None`` for anything else or nothing left."""
    if not isinstance(value, str):
        return None
    collapsed = _SPACE.sub(" ", value).strip()
    return collapsed or None


# --- items --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NewsItem:
    """One news item from one feed. ``espn_ids`` are the ESPN players the source tags (ESPN only); ``player_names``
    are the players it names (ESPN's athlete tags, RotoWire's title). ``fetched_at`` is the download time of the
    payload it came from, filled in by the feed (``None`` straight out of a parser)."""

    source: str
    external_id: str
    sport: Sport
    title: str
    published_at: datetime
    body: str | None = None
    url: str | None = None
    espn_ids: tuple[int, ...] = ()
    player_names: tuple[str, ...] = ()
    fetched_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.source or not self.external_id or not self.title:
            raise ValueError("a news item needs a source, an id and a title")
        for stamp in (self.published_at, self.fetched_at):
            if stamp is not None and (stamp.tzinfo is None or stamp.utcoffset() is None):
                raise ValueError(f"news item {self.source}/{self.external_id}: timestamps must be timezone-aware")

    @property
    def key(self) -> tuple[str, str]:
        """The item's identity, unique in ``news_items``: (``source``, ``external_id``)."""
        return (self.source, self.external_id)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.title, self.body)


def fingerprint(title: str, body: str | None = None) -> str:
    """A story's text as a short hash: title and body as lower-case words, with accents, punctuation and spacing
    ignored and apostrophes and periods dropped (``won't`` is ``wont``, ``D.J.`` is ``DJ``), so the same story
    re-posted under another id or by another source hashes the same and a reworded one does not."""
    decomposed = unicodedata.normalize("NFKD", _IN_WORD.sub("", f"{title}\n{body or ''}"))
    plain = "".join(char for char in decomposed if not unicodedata.combining(char)).casefold()
    return hashlib.blake2b(" ".join(_WORD.findall(plain)).encode("utf-8"), digest_size=16).hexdigest()


def news_subject(title: str) -> str | None:
    """The player a RotoWire title is about: the text before its first colon (``Ladd McConkey: Termed
    'week-to-week'`` -> ``Ladd McConkey``); ``None`` when the title has no such prefix."""
    head, colon, _ = title.partition(":")
    return _text(head) if colon else None


def dedupe(items: Iterable[NewsItem]) -> tuple[NewsItem, ...]:
    """Each story once, oldest first: the first item of each (``source``, ``external_id``) in the order given; then,
    of a sport's items sharing a :func:`fingerprint`, the earliest published (on a tie, the one given first), with a
    copy published more than :data:`DUPLICATE_WINDOW` after the one kept counting as a story of its own."""
    first: dict[tuple[str, str], NewsItem] = {}
    for item in items:
        first.setdefault(item.key, item)
    kept: list[NewsItem] = []
    last: dict[tuple[Sport, str], datetime] = {}
    for item in sorted(first.values(), key=lambda entry: entry.published_at):  # stable: ties keep the given order
        story = (item.sport, item.fingerprint)
        previous = last.get(story)
        if previous is None or item.published_at - previous > DUPLICATE_WINDOW:
            last[story] = item.published_at
            kept.append(item)
    return tuple(kept)


# --- ESPN ---------------------------------------------------------------------------------------------------------


def _instant(value: object) -> datetime:
    """ESPN writes ``2026-10-05T22:34:13Z``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"not a timestamp: {value!r}")
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp without a time zone: {value!r}")
    return parsed.astimezone(UTC)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _athlete_id(category: Mapping[str, Any]) -> int | None:
    """``athleteId``, else the nested ``athlete.id``, else the id in the ``uid`` (``s:20~l:28~a:4568652``)."""
    for value in (category.get("athleteId"), _mapping(category.get("athlete")).get("id")):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        if isinstance(value, str) and value.strip().isdigit() and int(value) > 0:
            return int(value)
    uid = category.get("uid")
    if isinstance(uid, str) and (match := _ATHLETE_UID.search(uid)):
        return int(match[1])
    return None


def _espn_athletes(categories: object, sport: Sport) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """The tagged athletes' ESPN ids and names, in tag order, once each; athletes of another league are skipped."""
    ids: dict[int, None] = {}
    names: dict[str, None] = {}
    for category in categories if isinstance(categories, list) else ():
        if not isinstance(category, Mapping) or category.get("type") != "athlete":
            continue
        league = category.get("sportId")
        if isinstance(league, int) and league != ESPN_LEAGUE_IDS[sport]:
            continue
        espn_id = _athlete_id(category)
        if espn_id is None:
            continue
        ids.setdefault(espn_id, None)
        name = _text(category.get("description")) or _text(_mapping(category.get("athlete")).get("description"))
        if name is not None:
            names.setdefault(name, None)
    return tuple(ids), tuple(names)


def _espn_item(article: Mapping[str, Any], sport: Sport) -> NewsItem:
    ident = article.get("id")
    if isinstance(ident, bool) or not isinstance(ident, int | str) or not str(ident).strip():
        raise ValueError(f"article without an id: {ident!r}")
    title = _text(article.get("headline"))
    if title is None:
        raise ValueError(f"article {ident} has no headline")
    published = article.get("published") or article.get("lastModified")
    espn_ids, names = _espn_athletes(article.get("categories"), sport)
    return NewsItem(
        source=ESPN_NEWS,
        external_id=str(ident).strip(),
        sport=sport,
        title=title,
        published_at=_instant(published),
        body=_text(article.get("description")),
        url=_text(_mapping(_mapping(article.get("links")).get("web")).get("href")),
        espn_ids=espn_ids,
        player_names=names,
    )


def parse_espn_news(payload: bytes, sport: Sport | Game | str) -> tuple[NewsItem, ...]:
    """ESPN's articles in feed order (newest first). A payload without an ``articles`` list is a schema error (ESPN's
    error bodies are ``{"code": ..., "detail": ...}``), as is a non-empty list in which no article parses; single bad
    articles are skipped and counted. No articles at all is a quiet feed."""
    league = _sport(sport)
    raw = parse_json(payload)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("articles"), list):
        raise SourceSchemaError("espn news: expected a JSON object with an 'articles' list")
    articles: list[object] = raw["articles"]
    items: list[NewsItem] = []
    skipped = 0
    for article in articles:
        try:
            if not isinstance(article, Mapping):
                raise TypeError(f"article is a {type(article).__name__}")
            items.append(_espn_item(article, league))
        except (TypeError, ValueError) as exc:
            skipped += 1
            logger.warning("espn news: skipped an article (%s)", exc)
    if articles and not items:
        raise SourceSchemaError(f"espn news: none of {len(articles)} articles parsed")
    if skipped:
        logger.warning("espn news: skipped %d of %d articles that did not parse", skipped, len(articles))
    return tuple(items)


# --- RotoWire -----------------------------------------------------------------------------------------------------


def parse_rss_date(text: str) -> datetime:
    """An RSS ``pubDate`` as an aware UTC datetime. RFC 822 dates parse as written; RotoWire's 12-hour clock with a
    US zone name (``Mon, 05 Oct 2026 3:42:00 PM PDT`` -> 22:42 UTC) is turned into one first. A date without a
    known zone raises ``ValueError`` rather than being guessed at."""

    def to_24h(match: re.Match[str]) -> str:
        hour = int(match["hour"]) % 12 + (12 if match["half"].lower() == "pm" else 0)
        return f"{hour:02d}:{match['minute']}:{match['second'] or '00'}"

    normalized = _CLOCK_12H.sub(to_24h, text.strip(), count=1)
    try:
        parsed = email.utils.parsedate_to_datetime(normalized)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"not an RSS date: {text!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"RSS date without a known time zone: {text!r}")
    return parsed.astimezone(UTC)


def _body(value: object) -> str | None:
    """RotoWire's description without its "Visit RotoWire.com" footer, tags or entity escapes."""
    if not isinstance(value, str):
        return None
    text = html.unescape(_TAG.sub(" ", value))
    return _text(_ROTOWIRE_FOOTER.sub("", text))


def _rotowire_item(entry: Mapping[str, Any], sport: Sport) -> NewsItem:
    url = _text(entry.get("link"))
    ident = _text(entry.get("id")) or url
    title = _text(entry.get("title"))
    if ident is None or title is None:
        raise ValueError("an RSS item needs a guid (or link) and a title")
    guid_sport = _GUID_SPORT.match(ident.lower())
    if guid_sport is not None and guid_sport[1] != sport:
        raise ValueError(f"item {ident} is {guid_sport[1]} news, not {sport}")
    published = entry.get("published")
    if not isinstance(published, str):
        raise ValueError(f"item {ident} has no pubDate")
    subject = news_subject(title)
    return NewsItem(
        source=ROTOWIRE,
        external_id=ident,
        sport=sport,
        title=title,
        published_at=parse_rss_date(published),
        body=_body(entry.get("summary")),
        url=url,
        player_names=(subject,) if subject is not None else (),
    )


def parse_rotowire_rss(payload: bytes, sport: Sport | Game | str) -> tuple[NewsItem, ...]:
    """RotoWire's items in feed order (newest first). Anything that is not an RSS or Atom document (an HTML block
    page, an error body) is a schema error, as is a non-empty feed in which no item parses; single bad items (no
    title, an unreadable date, another sport's guid) are skipped and counted. An empty channel is a quiet feed."""
    league = _sport(sport)
    feed: Any = feedparser.parse(payload)  # bytes, never a str: feedparser would treat a str as a URL or a path
    version = feed.get("version") or ""
    if not version.startswith(("rss", "atom")):
        problem = feed.get("bozo_exception") if feed.get("bozo") else "no RSS or Atom document"
        raise SourceSchemaError(f"rotowire rss: not a feed ({problem})")
    entries: list[Any] = list(feed.get("entries") or [])
    items: list[NewsItem] = []
    skipped = 0
    for entry in entries:
        try:
            items.append(_rotowire_item(entry, league))
        except (TypeError, ValueError) as exc:
            skipped += 1
            logger.warning("rotowire rss: skipped an item (%s)", exc)
    if entries and not items:
        raise SourceSchemaError(f"rotowire rss: none of {len(entries)} items parsed")
    if skipped:
        logger.warning("rotowire rss: skipped %d of %d items that did not parse", skipped, len(entries))
    return tuple(items)


# --- feeds --------------------------------------------------------------------------------------------------------


class NewsFeedSource(HttpSource):
    """Shared by the news feeds: the :data:`POLL_INTERVAL` floor, degradation, and ``fetched_at`` on every item."""

    base_headers: ClassVar[Mapping[str, str]] = {"User-Agent": BROWSER_USER_AGENT}
    default_ttl: ClassVar[timedelta] = POLL_INTERVAL
    poll_interval: ClassVar[timedelta] = POLL_INTERVAL

    def news(self, sport: Sport | Game | str, **options: Unpack[FetchOptions]) -> Fetched[tuple[NewsItem, ...]]:
        raise NotImplementedError

    def _feed(
        self,
        dataset: str,
        key: str,
        url: str,
        params: Mapping[str, str | int],
        parse: Callable[[bytes], tuple[NewsItem, ...]],
        ext: str,
        options: FetchOptions,
    ) -> Fetched[tuple[NewsItem, ...]]:
        try:
            fetched = self.fetch(
                dataset,
                key,
                download=lambda: self.get_bytes(url, params=params),
                parse=parse,
                ext=ext,
                meta={"url": url, **params},
                **self._polite(dataset, key, ext, options),
            )
        except SourceError as exc:
            logger.warning("%s/%s[%s]: unavailable (%s); continuing without it", self.name, dataset, key, exc)
            return Fetched((), self.clock(), self.name, dataset, key, degraded=True, warnings=(str(exc),))
        return replace(fetched, data=tuple(replace(item, fetched_at=fetched.as_of) for item in fetched.data))

    def _polite(self, dataset: str, key: str, ext: str, options: FetchOptions) -> FetchOptions:
        """``options``, unless the cached copy is younger than the poll interval: then only a ``max_age`` of the
        interval, so that copy is served whatever the dataset's TTL or the caller asked."""
        entry = self.cache.read(self.name, dataset, key, ext)
        if entry is not None and self.clock() - entry.as_of < self.poll_interval:
            return {"max_age": self.poll_interval}
        return options


class EspnNewsSource(NewsFeedSource):
    """ESPN's news API (``site.api.espn.com``): the latest :data:`ESPN_NEWS_LIMIT` articles of one sport."""

    name: ClassVar[str] = ESPN_NEWS
    min_interval: ClassVar[float] = 0.5
    base_headers: ClassVar[Mapping[str, str]] = {"User-Agent": BROWSER_USER_AGENT, "Accept": "application/json"}
    ttl: ClassVar[Mapping[str, timedelta]] = {"news": POLL_INTERVAL}

    def news(self, sport: Sport | Game | str, **options: Unpack[FetchOptions]) -> Fetched[tuple[NewsItem, ...]]:
        """The sport's latest articles, newest first. Degrades; see the module docstring."""
        league = _sport(sport)
        url = ESPN_NEWS_URL.format(path=SITE_PATHS[Game.coerce(league)])
        params: dict[str, str | int] = {"limit": ESPN_NEWS_LIMIT}
        return self._feed(
            "news", league, url, params, lambda payload: parse_espn_news(payload, league), "json", options
        )


class RotowireSource(NewsFeedSource):
    """RotoWire's player-news RSS feed for one sport."""

    name: ClassVar[str] = ROTOWIRE
    min_interval: ClassVar[float] = 1.0
    base_headers: ClassVar[Mapping[str, str]] = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
    }
    ttl: ClassVar[Mapping[str, timedelta]] = {"rss": POLL_INTERVAL}

    def news(self, sport: Sport | Game | str, **options: Unpack[FetchOptions]) -> Fetched[tuple[NewsItem, ...]]:
        """The sport's latest items, newest first. Degrades; see the module docstring."""
        league = _sport(sport)
        params: dict[str, str | int] = {"sport": league.upper()}
        return self._feed(
            "rss", league, ROTOWIRE_RSS_URL, params, lambda payload: parse_rotowire_rss(payload, league), "xml", options
        )


@dataclass(frozen=True, slots=True)
class NewsPoll:
    """One poll of every feed for a sport: each feed's result with its provenance, and their items deduplicated."""

    sport: Sport
    feeds: tuple[Fetched[tuple[NewsItem, ...]], ...]
    items: tuple[NewsItem, ...]
    """Oldest first (:func:`dedupe`)."""

    @property
    def warnings(self) -> tuple[str, ...]:
        """Every feed's warnings, each prefixed with the feed (``rotowire/rss[nfl]: ...``)."""
        return tuple(
            f"{feed.source}/{feed.dataset}[{feed.key}]: {text}" for feed in self.feeds for text in feed.warnings
        )

    def describe(self) -> str:
        """One line: ``nfl news: 12 items (espn_news 7 fresh, rotowire 5 cached)``."""
        parts = []
        for feed in self.feeds:
            state = "DEGRADED" if feed.degraded else "STALE" if feed.stale else "cached" if feed.cached else "fresh"
            parts.append(f"{feed.source} {len(feed.data)} {state}")
        return f"{self.sport} news: {len(self.items)} items ({', '.join(parts)})"


def poll_news(
    sport: Sport | Game | str,
    *,
    espn: EspnNewsSource | None = None,
    rotowire: RotowireSource | None = None,
    **options: Unpack[FetchOptions],
) -> NewsPoll:
    """Both feeds for ``sport`` (adapters over the cache dir unless given; those made here are closed again), their
    items deduplicated. Never raises for a feed that is down: it shows up degraded in ``feeds``."""
    league = _sport(sport)
    sources: list[NewsFeedSource] = [
        espn if espn is not None else EspnNewsSource(),
        rotowire if rotowire is not None else RotowireSource(),
    ]
    owned = [source for source, given in zip(sources, (espn, rotowire), strict=True) if given is None]
    try:
        feeds = tuple(source.news(league, **options) for source in sources)
    finally:
        for source in owned:
            source.close()
    return NewsPoll(league, feeds, dedupe(item for feed in feeds for item in feed.data))
