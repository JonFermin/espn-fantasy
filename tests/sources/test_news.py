"""News feeds: ESPN's news API and RotoWire's RSS against trimmed real captures (respx; no network).

tests/fixtures/sources/news holds ESPN's NFL and NBA news feeds and RotoWire's NFL and NBA RSS feeds, captured on
2026-10-05 (NFL week 5, NBA preseason); see tests/fixtures/sources/README.md. RotoWire writes ``3:42:00 PM PDT``,
which is 22:42 UTC.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from fm.sources.base import FetchOptions, RateLimiter, SourceSchemaError
from fm.sources.news import (
    DUPLICATE_WINDOW,
    ESPN_NEWS,
    ESPN_NEWS_LIMIT,
    POLL_INTERVAL,
    ROTOWIRE,
    EspnNewsSource,
    NewsItem,
    RotowireSource,
    dedupe,
    fingerprint,
    news_subject,
    parse_espn_news,
    parse_rotowire_rss,
    parse_rss_date,
    poll_news,
)
from fm.sources.odds import BROWSER_USER_AGENT

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sources" / "news"
T0 = datetime(2026, 10, 5, 23, 0, tzinfo=UTC)
ESPN_HOST = "site.api.espn.com"
NFL_NEWS = "/apis/site/v2/sports/football/nfl/news"
NBA_NEWS = "/apis/site/v2/sports/basketball/nba/news"
ROTOWIRE_HOST = "www.rotowire.com"
RSS_PATH = "/rss/news.php"
HTML_BLOCK = b"<html><head><title>Access denied</title></head><body>Blocked</body></html>"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def ok(name: str) -> httpx.Response:
    content_type = "application/json;charset=UTF-8" if name.endswith(".json") else "application/xml"
    return httpx.Response(200, content=fixture(name), headers={"content-type": content_type})


def at(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 10, 5, hour, minute, second, tzinfo=UTC)


def item(
    external_id: str,
    title: str,
    published: datetime,
    *,
    source: str = ROTOWIRE,
    body: str | None = None,
    sport: str = "nfl",
) -> NewsItem:
    return NewsItem(
        source=source,
        external_id=external_id,
        sport="nfl" if sport == "nfl" else "nba",
        title=title,
        published_at=published,
        body=body,
    )


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


class Routes:
    def __init__(self, router: respx.MockRouter) -> None:
        self.espn_nfl = router.get(host=ESPN_HOST, path=NFL_NEWS).mock(return_value=ok("espn_news_nfl.json"))
        self.espn_nba = router.get(host=ESPN_HOST, path=NBA_NEWS).mock(return_value=ok("espn_news_nba.json"))
        self.rotowire = router.get(host=ROTOWIRE_HOST, path=RSS_PATH).mock(side_effect=self._rss)

    @staticmethod
    def _rss(request: httpx.Request) -> httpx.Response:
        return ok(f"rotowire_{request.url.params['sport'].lower()}.xml")


@pytest.fixture
def routes() -> Iterator[Routes]:
    with respx.mock(assert_all_called=False) as router:  # unmatched requests raise; nothing leaves the process
        yield Routes(router)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def espn(routes: Routes, clock: FakeClock, tmp_path: Path) -> Iterator[EspnNewsSource]:
    with EspnNewsSource(
        cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock, sleep=lambda _: None
    ) as src:
        yield src


@pytest.fixture
def rotowire(routes: Routes, clock: FakeClock, tmp_path: Path) -> Iterator[RotowireSource]:
    with RotowireSource(
        cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock, sleep=lambda _: None
    ) as src:
        yield src


# --- ESPN news API ----------------------------------------------------------------------------------------------------


def test_espn_articles_carry_their_tagged_athletes() -> None:
    items = parse_espn_news(fixture("espn_news_nfl.json"), "nfl")
    assert [entry.external_id for entry in items] == ["50112664", "50112533", "50112493", "50108832", "50112291"]
    assert all(entry.source == ESPN_NEWS and entry.sport == "nfl" for entry in items)
    assert all(entry.fetched_at is None for entry in items)  # the feed stamps it, not the parser

    smith = items[0]
    assert smith.title == "Cowboys' Tyler Smith hoping to be ready to face Bucs"
    assert smith.body is not None and smith.body.startswith("Cowboys Pro Bowl guard Tyler Smith hopes")
    assert smith.published_at == at(22, 34, 13)
    assert smith.url == "https://www.espn.com/nfl/story/_/id/50112664/cowboys-tyler-smith-hoping-ready-face-bucs"
    assert (smith.espn_ids, smith.player_names) == ((4568652,), ("Tyler Smith",))

    johnson = items[1]  # tagged by the athlete's name as ESPN spells it, without the period
    assert (johnson.espn_ids, johnson.player_names) == ((4428991,), ("Paris Johnson Jr",))

    pickups = items[3]  # a feature story tagging several players, in ESPN's tag order
    assert pickups.espn_ids == (14880, 4887558, 4635008, 4373626)
    assert pickups.player_names == ("Kirk Cousins", "Emanuel Wilson", "Keon Coleman", "Tyler Allgeier")

    mcvay = items[4]  # team news: two team tags, no athlete
    assert (mcvay.espn_ids, mcvay.player_names) == ((), ())


def test_espn_nba_feed() -> None:
    items = parse_espn_news(fixture("espn_news_nba.json"), "fba")
    assert [(entry.external_id, entry.sport, entry.espn_ids) for entry in items] == [
        ("50112557", "nba", (4065778,)),
        ("50112444", "nba", (4683750,)),
        ("50112238", "nba", ()),
    ]
    assert items[0].title == "Sources: Clippers' Max Strus (foot) out at least 4 weeks"
    assert items[0].published_at == at(22, 41, 42)


def test_espn_athlete_tags_are_read_defensively() -> None:
    article = {
        "id": 1,
        "headline": "  Several   tags ",
        "published": "2026-10-05T20:00:00Z",
        "categories": [
            {"type": "athlete", "sportId": 28, "description": "Uid Only", "uid": "s:20~l:28~a:777"},
            {"type": "athlete", "sportId": 28, "athlete": {"id": "888", "description": "Nested Id"}},
            {"type": "athlete", "sportId": 46, "athleteId": 999, "description": "An NBA Star At The Game"},
            {"type": "athlete", "sportId": 28, "athleteId": 777, "description": "Uid Only"},
            {"type": "athlete", "description": "No Id"},
            {"type": "team", "teamId": 6, "description": "Dallas Cowboys"},
            "junk",
        ],
    }
    (parsed,) = parse_espn_news(json.dumps({"articles": [article]}).encode(), "nfl")
    assert parsed.title == "Several tags"
    assert parsed.espn_ids == (777, 888)  # once each; the other league's athlete and the id-less tag are skipped
    assert parsed.player_names == ("Uid Only", "Nested Id")
    assert parsed.body is None and parsed.url is None


def test_espn_news_falls_back_to_last_modified() -> None:
    article = {"id": "abc", "headline": "Undated", "lastModified": "2026-10-05T21:00:00Z"}
    (parsed,) = parse_espn_news(json.dumps({"articles": [article]}).encode(), "nfl")
    assert parsed.published_at == at(21, 0)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"code":1008,"detail":"script error: unexpected exception or error processing script"}',  # ESPN's 500 body
        b"[]",
        b'{"articles": [{"id": 1}, {"headline": "no id"}, "junk"]}',
    ],
)
def test_espn_news_schema_errors(payload: bytes) -> None:
    with pytest.raises(SourceSchemaError):
        parse_espn_news(payload, "nfl")


def test_espn_news_skips_bad_articles(caplog: pytest.LogCaptureFixture) -> None:
    good = {"id": 5, "headline": "Fine", "published": "2026-10-05T20:00:00Z"}
    bad = {"id": 6, "headline": "No date"}
    payload = json.dumps({"articles": [bad, good, {"id": 7, "headline": "Naive", "published": "2026-10-05T20:00"}]})
    with caplog.at_level(logging.WARNING, logger="fm.sources.news"):
        parsed = parse_espn_news(payload.encode(), "nfl")
    assert [entry.external_id for entry in parsed] == ["5"]
    assert "skipped 2 of 3 articles" in caplog.text
    assert parse_espn_news(b'{"articles": []}', "nfl") == ()  # a quiet feed


# --- RotoWire RSS -----------------------------------------------------------------------------------------------------


def test_rotowire_items() -> None:
    items = parse_rotowire_rss(fixture("rotowire_nfl.xml"), "nfl")
    assert [entry.external_id for entry in items] == ["nfl640907", "nfl640906", "nfl640905", "nfl640904", "nfl640903"]
    assert all(entry.source == ROTOWIRE and entry.sport == "nfl" and entry.espn_ids == () for entry in items)

    mcconkey = items[0]
    assert mcconkey.title == "Ladd McConkey: Termed 'week-to-week'"
    assert mcconkey.player_names == ("Ladd McConkey",)
    assert mcconkey.published_at == at(22, 42)  # "Mon, 05 Oct 2026 3:42:00 PM PDT"
    assert mcconkey.url == "https://www.rotowire.com//football/player/ladd-mcconkey-17724"
    assert mcconkey.body == (
        'McConkey (foot), per head coach Jim Harbaugh, is "week-to-week," Alex Insdorf of BoltBeat.com reports.'
    )
    assert not any("Visit RotoWire" in (entry.body or "") for entry in items)
    assert [entry.player_names[0] for entry in items[1:]] == [
        "Rachaad White",
        "Terry McLaurin",
        "Jayden Daniels",
        "Marcus Mariota",
    ]

    nba = parse_rotowire_rss(fixture("rotowire_nba.xml"), "nba")
    assert [(entry.external_id, entry.player_names) for entry in nba][:3] == [
        ("nba533240", ("Zach Edey",)),
        ("nba533238", ("Kingston Flemings",)),
        ("nba533234", ("Max Strus",)),
    ]
    assert nba[2].published_at == at(22, 21)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mon, 05 Oct 2026 3:42:00 PM PDT", datetime(2026, 10, 5, 22, 42, tzinfo=UTC)),
        ("Mon, 05 Oct 2026 12:30:00 PM PDT", datetime(2026, 10, 5, 19, 30, tzinfo=UTC)),  # noon
        ("Tue, 06 Oct 2026 12:05:00 AM PDT", datetime(2026, 10, 6, 7, 5, tzinfo=UTC)),  # just after midnight
        ("Sun, 01 Nov 2026 1:00 PM PST", datetime(2026, 11, 1, 21, 0, tzinfo=UTC)),  # standard time, no seconds
        ("Mon, 05 Oct 2026 22:42:00 GMT", datetime(2026, 10, 5, 22, 42, tzinfo=UTC)),  # plain RFC 822
        ("05 Oct 2026 15:42:00 -0700", datetime(2026, 10, 5, 22, 42, tzinfo=UTC)),
    ],
)
def test_parse_rss_date(text: str, expected: datetime) -> None:
    parsed = parse_rss_date(text)
    assert parsed == expected and parsed.tzinfo is UTC


@pytest.mark.parametrize(
    "text", ["Mon, 05 Oct 2026 3:42:00 PM", "Mon, 05 Oct 2026 3:42:00 PM XYZ", "yesterday", "", "Mon, 05 Oct 2026"]
)
def test_parse_rss_date_refuses_to_guess(text: str) -> None:
    with pytest.raises(ValueError):
        parse_rss_date(text)


def rss(*items: str) -> bytes:
    body = "".join(f"<item>{entry}</item>" for entry in items)
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>{body}</channel></rss>'.encode()


GOOD_ITEM = (
    "<guid>nfl1</guid><title>Josh Allen: Limited Wednesday</title><link>https://example.test/josh-allen-1</link>"
    "<description>Allen (ankle) was limited. &amp; more.\n\n  Visit RotoWire.com for more analysis on this update."
    "</description><pubDate>Wed, 07 Oct 2026 1:15:00 PM PDT</pubDate>"
)


def test_rotowire_skips_bad_items(caplog: pytest.LogCaptureFixture) -> None:
    payload = rss(
        GOOD_ITEM,
        "<guid>nfl2</guid><title>No date</title>",
        "<guid>nfl3</guid><title>Bad date</title><pubDate>sometime</pubDate>",
        "<guid>nba4</guid><title>Wrong sport</title><pubDate>Wed, 07 Oct 2026 1:15:00 PM PDT</pubDate>",
        "<guid>nfl5</guid><pubDate>Wed, 07 Oct 2026 1:15:00 PM PDT</pubDate>",
    )
    with caplog.at_level(logging.WARNING, logger="fm.sources.news"):
        (parsed,) = parse_rotowire_rss(payload, "nfl")
    assert (parsed.external_id, parsed.player_names, parsed.body) == (
        "nfl1",
        ("Josh Allen",),
        "Allen (ankle) was limited. & more.",
    )
    assert parsed.published_at == datetime(2026, 10, 7, 20, 15, tzinfo=UTC)
    assert "skipped 4 of 5 items" in caplog.text
    assert parse_rotowire_rss(rss(), "nfl") == ()  # an empty channel is a quiet feed


@pytest.mark.parametrize(
    "payload",
    [
        HTML_BLOCK,
        b'{"code": 1}',
        b"",
        rss("<guid>nba1</guid><title>A: b</title><pubDate>Wed, 07 Oct 2026 1:15:00 PM PDT</pubDate>"),  # NBA feed
    ],
)
def test_rotowire_schema_errors(payload: bytes) -> None:
    with pytest.raises(SourceSchemaError):
        parse_rotowire_rss(payload, "nfl")


def test_news_subject() -> None:
    assert news_subject("Ladd McConkey: Termed 'week-to-week'") == "Ladd McConkey"
    assert news_subject("Amon-Ra St. Brown:  Big game: again") == "Amon-Ra St. Brown"
    assert news_subject("No colon here") is None
    assert news_subject(":  leading colon") is None


# --- dedupe -----------------------------------------------------------------------------------------------------------


def test_fingerprint_ignores_case_spacing_punctuation_and_accents() -> None:
    base = fingerprint("Luka Dončić: Out Monday", "Dončić (ankle) won’t play.")
    assert fingerprint("luka doncic -- OUT   monday", "Doncic (ankle) wont play") == base
    assert fingerprint("Luka Doncic: Out Monday", "Doncic (ankle) will play.") != base
    assert fingerprint("Luka Doncic: Out Monday") != base
    assert fingerprint("D.J. Moore: Limited") == fingerprint("DJ Moore: limited")


def test_dedupe_keeps_each_story_once() -> None:
    title, body = "Jayden Daniels: Return to action imminent?", "Aiming to start."
    first = item("nfl1", title, at(22, 30), body=body)
    repoll = item("nfl1", "Jayden Daniels: edited later", at(22, 50), body="A changed copy under the same id.")
    repost = item("nfl9", title, at(22, 45), body=body)
    other_source = item("e1", "Jayden Daniels: return to action imminent", at(22, 35), source=ESPN_NEWS, body=body)
    nba_twin = item("nba1", title, at(22, 40), body=body, sport="nba")
    reworded = item("nfl2", "Jayden Daniels: Expected to start Sunday", at(22, 20), body="Full practice week planned.")

    kept = dedupe([first, repoll, repost, other_source, nba_twin, reworded])
    assert kept == (reworded, first, nba_twin)  # oldest first; the repost and the other source's copy are repeats
    assert dedupe([]) == ()


def test_dedupe_ties_keep_the_order_given() -> None:
    espn = item("e1", "Same story", at(22, 0), source=ESPN_NEWS)
    roto = item("r1", "Same story", at(22, 0))
    assert dedupe([espn, roto]) == (espn,)
    assert dedupe([roto, espn]) == (roto,)


def test_the_same_headline_weeks_apart_is_new_news() -> None:
    week5 = item("nfl1", "Jayden Daniels: Out for Sunday", at(20, 0))
    within = item("nfl2", "Jayden Daniels: Out for Sunday", at(20, 0) + DUPLICATE_WINDOW)
    week6 = item("nfl3", "Jayden Daniels: Out for Sunday", at(20, 0) + timedelta(days=7))
    week6_repost = item("nfl4", "Jayden Daniels: Out for Sunday", at(20, 0) + timedelta(days=7, hours=2))
    assert dedupe([week6_repost, week6, within, week5]) == (week5, week6)


def test_reports_of_one_event_in_different_words_are_both_kept() -> None:
    espn = parse_espn_news(fixture("espn_news_nba.json"), "nba")
    roto = parse_rotowire_rss(fixture("rotowire_nba.xml"), "nba")
    strus = [entry for entry in dedupe([*espn, *roto]) if "Max Strus" in entry.player_names]
    assert [(entry.source, entry.external_id) for entry in strus] == [(ROTOWIRE, "nba533234"), (ESPN_NEWS, "50112557")]


def test_news_items_refuse_naive_times_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        NewsItem(source=ROTOWIRE, external_id="x", sport="nfl", title="t", published_at=datetime(2026, 10, 5))
    with pytest.raises(ValueError, match="needs a source"):
        NewsItem(source=ROTOWIRE, external_id="", sport="nfl", title="t", published_at=T0)


# --- feeds ------------------------------------------------------------------------------------------------------------


def test_espn_feed_request_and_fetched_at(espn: EspnNewsSource, routes: Routes, clock: FakeClock) -> None:
    result = espn.news("nfl")
    assert (result.source, result.dataset, result.key) == (ESPN_NEWS, "news", "nfl")
    assert len(result.data) == 5 and result.as_of == T0
    assert not (result.cached or result.stale or result.degraded)
    assert all(entry.fetched_at == T0 for entry in result.data)
    request = routes.espn_nfl.calls.last.request
    assert request.url.params["limit"] == str(ESPN_NEWS_LIMIT)
    assert request.headers["User-Agent"] == BROWSER_USER_AGENT
    assert "cookie" not in request.headers

    nba = espn.news("fba")
    assert nba.key == "nba" and routes.espn_nba.call_count == 1 and nba.data[0].sport == "nba"


def test_rotowire_feed_request(rotowire: RotowireSource, routes: Routes) -> None:
    nfl = rotowire.news("nfl")
    nba = rotowire.news("nba")
    assert [call.request.url.params["sport"] for call in routes.rotowire.calls] == ["NFL", "NBA"]
    assert (nfl.dataset, nfl.key, nba.key) == ("rss", "nfl", "nba")
    assert nfl.raw_path is not None and nfl.raw_path.suffix == ".xml"
    assert [entry.player_names[0] for entry in nba.data][-1] == "Jalen Duren"


def test_feeds_poll_at_most_every_ten_minutes(espn: EspnNewsSource, routes: Routes, clock: FakeClock) -> None:
    espn.news("nfl")
    clock.advance(minutes=9, seconds=59)
    asks: list[FetchOptions] = [{}, {"force": True}, {"max_age": timedelta(0)}, {"fresh_since": clock.now}]
    for options in asks:
        again = espn.news("nfl", **options)
        assert again.cached and again.as_of == T0 and all(entry.fetched_at == T0 for entry in again.data)
    assert routes.espn_nfl.call_count == 1  # the poll interval is a floor that force cannot lower

    clock.advance(seconds=2)
    fresh = espn.news("nfl")
    assert routes.espn_nfl.call_count == 2 and not fresh.cached
    assert fresh.as_of == T0 + POLL_INTERVAL + timedelta(seconds=1)
    assert all(entry.fetched_at == fresh.as_of for entry in fresh.data)


def test_force_works_once_the_interval_has_passed(espn: EspnNewsSource, routes: Routes, clock: FakeClock) -> None:
    espn.news("nfl")
    clock.advance(minutes=10)
    assert not espn.news("nfl", force=True).cached and routes.espn_nfl.call_count == 2


def test_the_floor_holds_under_a_shorter_ttl(routes: Routes, clock: FakeClock, tmp_path: Path) -> None:
    class Eager(RotowireSource):
        ttl = {"rss": timedelta(minutes=1)}

    with Eager(cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock, sleep=lambda _: None) as eager:
        eager.news("nfl")
        clock.advance(minutes=5)
        assert eager.news("nfl").cached and routes.rotowire.call_count == 1
        clock.advance(minutes=5)
        assert not eager.news("nfl").cached and routes.rotowire.call_count == 2


def test_feed_degrades_without_a_cached_copy(espn: EspnNewsSource, routes: Routes) -> None:
    routes.espn_nfl.mock(return_value=httpx.Response(503))
    result = espn.news("nfl")
    assert result.degraded and result.data == () and not result.cached
    assert result.warnings and "503" in result.warnings[0]


def test_feed_serves_the_last_good_copy_when_a_refresh_fails(
    rotowire: RotowireSource, routes: Routes, clock: FakeClock, tmp_path: Path
) -> None:
    rotowire.news("nfl")
    clock.advance(minutes=11)
    routes.rotowire.mock(return_value=httpx.Response(200, content=HTML_BLOCK, headers={"content-type": "text/html"}))
    stale = rotowire.news("nfl")
    assert stale.stale and stale.cached and stale.as_of == T0 and len(stale.data) == 5
    assert all(entry.fetched_at == T0 for entry in stale.data)
    assert "not a feed" in stale.warnings[0]
    rejected = list((tmp_path / "cache" / ROTOWIRE / "rss").glob("nfl.rejected.xml"))
    assert len(rejected) == 1 and rejected[0].read_bytes() == HTML_BLOCK  # kept beside, never over, the good copy

    clock.advance(minutes=11)
    routes.rotowire.mock(side_effect=Routes._rss)
    assert not rotowire.news("nfl").stale


def test_junk_on_the_first_poll_degrades(rotowire: RotowireSource, routes: Routes) -> None:
    routes.rotowire.mock(return_value=httpx.Response(200, content=HTML_BLOCK))
    result = rotowire.news("nfl")
    assert result.degraded and result.data == () and "not a feed" in result.warnings[0]


def test_poll_news_merges_both_feeds(espn: EspnNewsSource, rotowire: RotowireSource) -> None:
    poll = poll_news("nfl", espn=espn, rotowire=rotowire)
    assert poll.sport == "nfl" and [feed.source for feed in poll.feeds] == [ESPN_NEWS, ROTOWIRE]
    assert len(poll.items) == 10 and poll.warnings == ()
    published = [entry.published_at for entry in poll.items]
    assert published == sorted(published) and poll.items[0].external_id == "50108832"  # oldest first
    assert poll.describe() == "nfl news: 10 items (espn_news 5 fresh, rotowire 5 fresh)"

    again = poll_news("nfl", espn=espn, rotowire=rotowire)
    assert again.items == poll.items
    assert again.describe() == "nfl news: 10 items (espn_news 5 cached, rotowire 5 cached)"


def test_poll_news_survives_a_feed_that_is_down(espn: EspnNewsSource, rotowire: RotowireSource, routes: Routes) -> None:
    routes.rotowire.mock(return_value=httpx.Response(500))
    poll = poll_news("nba", espn=espn, rotowire=rotowire)
    assert [entry.source for entry in poll.items] == [ESPN_NEWS] * 3
    assert poll.feeds[1].degraded and len(poll.warnings) == 1
    assert poll.warnings[0].startswith("rotowire/rss[nba]: ") and "500" in poll.warnings[0]
    assert poll.describe() == "nba news: 3 items (espn_news 3 fresh, rotowire 0 DEGRADED)"


def test_poll_news_builds_its_own_feeds_over_the_cache_dir(routes: Routes) -> None:
    poll = poll_news("nfl")  # adapters made (and closed) here, caching under the test's FM_CACHE_DIR
    assert len(poll.items) == 10 and routes.espn_nfl.call_count == 1 and routes.rotowire.call_count == 1
    assert all(feed.raw_path is not None and feed.raw_path.is_file() for feed in poll.feeds)
