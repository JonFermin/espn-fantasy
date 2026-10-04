"""Source base: TTL cache, ``as_of`` stamps, rate limiting, raw capture, stale fallback, HTTP retries."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx
from hypothesis import given
from hypothesis import strategies as st

from fm import paths
from fm.sources.base import (
    META_SUFFIX,
    HttpSource,
    RateLimiter,
    Source,
    SourceError,
    SourceSchemaError,
    SourceUnavailable,
    safe_name,
)

T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
GOOD = b"[1, 2, 3]"
GOOD2 = b"[4, 5]"
BAD = b'{"not": "a list"}'


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


class Downloads:
    """A download callable that yields its payloads in order (the last one repeats) and counts calls."""

    def __init__(self, *payloads: bytes | Exception) -> None:
        self.payloads = list(payloads)
        self.calls = 0

    def __call__(self) -> bytes:
        self.calls += 1
        item = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        if isinstance(item, Exception):
            raise item
        return item


def parse_numbers(payload: bytes) -> list[int]:
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list")
    return data


class DemoSource(Source):
    name = "demo"
    ttl = {"numbers": timedelta(minutes=10)}
    min_interval = 0.0


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def source(clock: FakeClock, cache_root: Path) -> DemoSource:
    return DemoSource(cache_root=cache_root, limiter=RateLimiter(0), clock=clock)


def fetch_numbers(source: Source, download: Downloads, **options: object):
    return source.fetch("numbers", "k", download=download, parse=parse_numbers, **options)  # type: ignore[arg-type]


# --- cache, as_of, raw capture ---


def test_miss_downloads_stamps_as_of_and_captures_raw(source: DemoSource, cache_root: Path) -> None:
    download = Downloads(GOOD)
    result = source.fetch("numbers", "k", download=download, parse=parse_numbers, meta={"url": "u"})

    assert result.data == [1, 2, 3]
    assert (result.as_of, result.cached, result.stale, result.degraded) == (T0, False, False, False)
    assert (result.source, result.dataset, result.key) == ("demo", "numbers", "k")
    raw = cache_root / "demo" / "numbers" / "k.json"
    assert result.raw_path == raw and raw.read_bytes() == GOOD
    meta = json.loads((raw.parent / f"k{META_SUFFIX}").read_text(encoding="utf-8"))
    assert meta == {"source": "demo", "dataset": "numbers", "key": "k", "as_of": T0.isoformat(), "bytes": 9, "url": "u"}


def test_hit_within_ttl_serves_cache_without_downloading(source: DemoSource, clock: FakeClock) -> None:
    download = Downloads(GOOD, GOOD2)
    fetch_numbers(source, download)
    clock.advance(minutes=9)
    again = fetch_numbers(source, download)
    assert download.calls == 1
    assert again.data == [1, 2, 3] and again.cached is True and again.as_of == T0
    assert again.age(clock.now) == timedelta(minutes=9)


def test_ttl_expiry_refreshes_and_restamps(source: DemoSource, clock: FakeClock) -> None:
    download = Downloads(GOOD, GOOD2)
    fetch_numbers(source, download)
    later = clock.advance(minutes=11)
    fresh = fetch_numbers(source, download)
    assert download.calls == 2
    assert fresh.data == [4, 5] and fresh.cached is False and fresh.as_of == later


def test_max_age_tightens_the_ttl(source: DemoSource, clock: FakeClock) -> None:
    download = Downloads(GOOD, GOOD2)
    fetch_numbers(source, download)
    clock.advance(minutes=2)
    assert fetch_numbers(source, download, max_age=timedelta(minutes=5)).cached is True
    assert fetch_numbers(source, download, max_age=timedelta(minutes=1)).cached is False
    assert download.calls == 2


def test_fresh_since_demands_a_copy_fetched_at_or_after_t(source: DemoSource, clock: FakeClock) -> None:
    download = Downloads(GOOD, GOOD2)
    fetch_numbers(source, download)
    clock.advance(minutes=1)
    assert fetch_numbers(source, download, fresh_since=T0).cached is True
    assert fetch_numbers(source, download, fresh_since=T0 + timedelta(seconds=30)).cached is False
    assert download.calls == 2


def test_force_bypasses_a_fresh_cache(source: DemoSource) -> None:
    download = Downloads(GOOD, GOOD2)
    fetch_numbers(source, download)
    assert fetch_numbers(source, download, force=True).data == [4, 5]
    assert download.calls == 2


def test_default_cache_root_is_the_sources_subdir_of_the_cache_dir(clock: FakeClock) -> None:
    demo = DemoSource(limiter=RateLimiter(0), clock=clock)
    assert demo.cache.root == paths.cache_dir() / "sources"


def test_parsed_value_is_memoized_while_the_cache_entry_is_unchanged(source: DemoSource, clock: FakeClock) -> None:
    parses = 0

    def counting_parse(payload: bytes) -> list[int]:
        nonlocal parses
        parses += 1
        return parse_numbers(payload)

    download = Downloads(GOOD)
    for _ in range(3):
        source.fetch("numbers", "k", download=download, parse=counting_parse)
    assert (download.calls, parses) == (1, 1)
    clock.advance(minutes=11)
    source.fetch("numbers", "k", download=download, parse=counting_parse)
    assert (download.calls, parses) == (2, 2)


# --- failure handling ---


def test_failed_refresh_serves_the_stale_copy_with_a_warning(
    source: DemoSource, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    download = Downloads(GOOD, httpx.ConnectError("boom"))
    fetch_numbers(source, download)
    clock.advance(minutes=11)
    with caplog.at_level(logging.WARNING, logger="fm.sources.base"):
        stale = fetch_numbers(source, download)
    assert stale.data == [1, 2, 3]
    assert (stale.cached, stale.stale, stale.as_of) == (True, True, T0)
    assert stale.warnings == ("boom",)
    assert "serving stale copy" in caplog.text


def test_failed_download_without_a_cache_raises_unavailable(source: DemoSource, cache_root: Path) -> None:
    with pytest.raises(SourceUnavailable, match=r"demo/numbers\[k\]: offline"):
        fetch_numbers(source, Downloads(OSError("offline")))
    assert not cache_root.exists()


def test_stale_not_ok_raises_instead_of_serving_old_data(source: DemoSource, clock: FakeClock) -> None:
    download = Downloads(GOOD, OSError("offline"))
    fetch_numbers(source, download)
    clock.advance(minutes=11)
    with pytest.raises(SourceUnavailable):
        source.fetch("numbers", "k", download=download, parse=parse_numbers, stale_ok=False)


def test_unrecoverable_errors_are_never_swallowed(source: DemoSource, clock: FakeClock) -> None:
    # The test harness raises a bare RuntimeError for accidental network calls; a fallback must not hide it.
    download = Downloads(GOOD, RuntimeError("outbound network is disabled"))
    fetch_numbers(source, download)
    clock.advance(minutes=11)
    with pytest.raises(RuntimeError, match="outbound network") as excinfo:
        fetch_numbers(source, download)
    assert not isinstance(excinfo.value, SourceError)


def test_unparseable_payload_is_kept_as_rejected_and_raises(source: DemoSource, cache_root: Path) -> None:
    with pytest.raises(SourceSchemaError, match="did not parse") as excinfo:
        fetch_numbers(source, Downloads(BAD))
    rejected = cache_root / "demo" / "numbers" / "k.rejected.json"
    assert rejected.read_bytes() == BAD
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert not (cache_root / "demo" / "numbers" / "k.json").exists()
    # The source recovers as soon as a good payload arrives.
    assert fetch_numbers(source, Downloads(GOOD)).data == [1, 2, 3]


def test_unparseable_refresh_keeps_the_last_good_copy(source: DemoSource, clock: FakeClock, cache_root: Path) -> None:
    download = Downloads(GOOD, BAD)
    fetch_numbers(source, download)
    clock.advance(minutes=11)
    stale = fetch_numbers(source, download)
    assert stale.data == [1, 2, 3] and stale.stale is True
    assert "did not parse" in stale.warnings[0]
    assert (cache_root / "demo" / "numbers" / "k.json").read_bytes() == GOOD
    assert (cache_root / "demo" / "numbers" / "k.rejected.json").read_bytes() == BAD


def test_corrupted_cache_entry_is_refreshed(source: DemoSource, cache_root: Path) -> None:
    download = Downloads(GOOD, GOOD2)
    fetch_numbers(source, download)
    (cache_root / "demo" / "numbers" / "k.json").write_bytes(b"\x00garbage")
    refreshed = fetch_numbers(source, download)
    assert refreshed.data == [4, 5] and refreshed.cached is False
    assert download.calls == 2


def test_missing_meta_counts_as_a_miss(source: DemoSource, cache_root: Path) -> None:
    download = Downloads(GOOD, GOOD2)
    fetch_numbers(source, download)
    (cache_root / "demo" / "numbers" / f"k{META_SUFFIX}").unlink()
    assert fetch_numbers(source, download).data == [4, 5]


# --- frames, rate limiting, naming ---


def test_fetch_frame_round_trips_parquet(source: DemoSource, cache_root: Path) -> None:
    frame = pl.DataFrame({"player_id": ["00-1", "00-2"], "week": pl.Series([3, 4], dtype=pl.Int32), "pts": [1.5, None]})
    result = source.fetch_frame("frames", "w", load=lambda: frame)
    assert result.data.equals(frame) and result.data.schema == frame.schema
    raw = cache_root / "demo" / "frames" / "w.parquet"
    assert result.raw_path == raw and pl.read_parquet(raw).equals(frame)
    assert source.fetch_frame("frames", "w", load=lambda: pl.DataFrame()).cached is True


def test_rate_limiter_sleeps_only_for_the_remaining_interval() -> None:
    now = [100.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(1.0, monotonic=lambda: now[0], sleep=sleep)
    assert limiter.wait() == 0.0
    assert limiter.wait() == 1.0
    now[0] += 0.4
    assert limiter.wait() == pytest.approx(0.6)
    now[0] += 5
    assert limiter.wait() == 0.0
    assert slept == [1.0, pytest.approx(0.6)]
    assert limiter.calls == 4 and limiter.slept == pytest.approx(1.6)


def test_rate_limiter_rejects_negative_interval() -> None:
    with pytest.raises(ValueError):
        RateLimiter(-1)


def test_fetch_paces_downloads_but_not_cache_hits(clock: FakeClock, cache_root: Path) -> None:
    limiter = RateLimiter(0, monotonic=lambda: 0.0, sleep=lambda _: None)
    demo = DemoSource(cache_root=cache_root, limiter=limiter, clock=clock)
    download = Downloads(GOOD)
    for _ in range(3):
        fetch_numbers(demo, download)
    assert limiter.calls == 1
    clock.advance(minutes=11)
    fetch_numbers(demo, download)
    assert limiter.calls == 2


def test_default_limiter_uses_the_class_interval(clock: FakeClock, cache_root: Path) -> None:
    class Slow(DemoSource):
        min_interval = 2.5

    assert Slow(cache_root=cache_root, clock=clock).limiter.min_interval == 2.5


@given(st.text())
def test_safe_name_is_always_a_plain_path_component(value: str) -> None:
    name = safe_name(value)
    assert re.fullmatch(r"[A-Za-z0-9._-]+", name)
    assert not name.startswith(".")


def test_safe_name_examples() -> None:
    assert safe_name("2026/w4?x=1") == "2026_w4_x_1"
    assert safe_name("regular_2026_w4") == "regular_2026_w4"
    assert safe_name("") == "default"
    assert safe_name("...") == "default"


def test_keys_with_unsafe_characters_still_cache(source: DemoSource, cache_root: Path) -> None:
    download = Downloads(GOOD)
    result = source.fetch("numbers", "a/b c", download=download, parse=parse_numbers)
    assert result.raw_path == cache_root / "demo" / "numbers" / "a_b_c.json"
    assert source.fetch("numbers", "a/b c", download=download, parse=parse_numbers).cached is True


# --- HttpSource ---

URL = "https://example.test/thing"


class HttpDemo(HttpSource):
    name = "httpdemo"
    min_interval = 0.0
    ttl = {"thing": timedelta(minutes=10)}

    def thing(self, **options: object):
        return self.fetch("thing", "x", download=lambda: self.get_bytes(URL), parse=parse_numbers, **options)  # type: ignore[arg-type]


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def http(clock: FakeClock, cache_root: Path, sleeps: list[float]) -> HttpDemo:
    return HttpDemo(cache_root=cache_root, limiter=RateLimiter(0), clock=clock, sleep=sleeps.append)


@respx.mock
def test_get_bytes_sends_a_user_agent_and_returns_the_body(http: HttpDemo) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, content=GOOD))
    assert http.get_bytes(URL) == GOOD
    assert route.calls.last.request.headers["user-agent"].startswith("espn-fantasy/")


@respx.mock
def test_get_bytes_retries_429_honoring_retry_after(http: HttpDemo, sleeps: list[float]) -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(200, content=GOOD)]
    )
    assert http.get_bytes(URL) == GOOD
    assert route.call_count == 2 and sleeps == [2.0]


@respx.mock
def test_get_bytes_backs_off_exponentially_on_5xx_then_gives_up(http: HttpDemo, sleeps: list[float]) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(503))
    with pytest.raises(SourceError, match="HTTP 503"):
        http.get_bytes(URL)
    assert route.call_count == 3 and sleeps == [1.0, 2.0]


@respx.mock
def test_get_bytes_does_not_retry_client_errors(http: HttpDemo, sleeps: list[float]) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    with pytest.raises(SourceError, match="HTTP 404"):
        http.get_bytes(URL)
    assert route.call_count == 1 and sleeps == []


@respx.mock
def test_get_bytes_retries_transport_errors(http: HttpDemo, sleeps: list[float]) -> None:
    route = respx.get(URL).mock(side_effect=[httpx.ConnectError("refused"), httpx.Response(200, content=GOOD)])
    assert http.get_bytes(URL) == GOOD
    assert route.call_count == 2 and sleeps == [1.0]


@respx.mock
def test_http_fetch_falls_back_to_stale_on_server_errors(http: HttpDemo, clock: FakeClock) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, content=GOOD))
    assert http.thing().data == [1, 2, 3]
    clock.advance(minutes=11)
    route.mock(return_value=httpx.Response(500))
    stale = http.thing()
    assert stale.data == [1, 2, 3] and stale.stale is True
    assert "HTTP 500" in stale.warnings[0]


@respx.mock
def test_get_json_parses_the_body(http: HttpDemo) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, content=GOOD))
    assert http.get_json(URL) == [1, 2, 3]


def test_close_releases_an_owned_client(http: HttpDemo) -> None:
    client = http.client
    assert client is http.client
    http.close()
    assert client.is_closed
    with HttpDemo(limiter=RateLimiter(0)) as demo:
        assert isinstance(demo.client, httpx.Client)


def test_injected_client_is_left_open(clock: FakeClock, cache_root: Path) -> None:
    client = httpx.Client()
    demo = HttpDemo(client=client, cache_root=cache_root, limiter=RateLimiter(0), clock=clock)
    demo.close()
    assert not client.is_closed
    client.close()
