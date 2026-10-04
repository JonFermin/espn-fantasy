"""Source adapter base: TTL cache, ``as_of`` stamps, rate limiting, raw capture (DESIGN sections 5 and 7).

An adapter is a :class:`Source` subclass with one method per dataset; each method calls :meth:`Source.fetch` with a
``download`` callable (raw bytes) and a ``parse`` callable (bytes to a typed value). ``fetch``:

1. serves the cached payload while it is younger than the dataset's TTL; callers tighten that with ``max_age`` or
   ``fresh_since`` ("fresh as of T") or bypass it with ``force``;
2. otherwise paces the call through the adapter's :class:`RateLimiter`, downloads, parses, and captures the raw payload
   under ``cache_dir()/sources/<source>/<dataset>/<key>.<ext>`` next to ``<key>.meta.json`` holding the ``as_of`` stamp;
3. returns a :class:`Fetched` whose ``as_of`` is the download time, so decisions can record the age of every input;
4. falls back to the last good payload (``stale=True``) when a refresh fails and one exists.

A payload that fails to parse is kept as ``<key>.rejected.<ext>`` and never replaces the last good copy. Captured raw
files double as fixture material: trim one and drop it under ``tests/fixtures/sources/``.

:class:`HttpSource` adds a shared ``httpx.Client`` with timeouts and bounded retries on 429/5xx for JSON APIs. Adapters
over client libraries (nflreadpy) use :meth:`Source.fetch_frame`, which stores Polars frames as parquet.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, ClassVar, Self, TypedDict, Unpack

import httpx
import polars as pl

from fm import paths

logger = logging.getLogger(__name__)

META_SUFFIX = ".meta.json"
REJECTED_TAG = "rejected"
CACHE_SUBDIR = "sources"

type Primitive = str | int | float | bool | None
type Params = httpx.QueryParams | Mapping[str, Primitive] | list[tuple[str, Primitive]]


def utcnow() -> datetime:
    """Timezone-aware UTC now; every ``as_of`` stamp is UTC."""
    return datetime.now(UTC)


def user_agent() -> str:
    """``espn-fantasy/<version>`` for outbound requests."""
    try:
        installed = distribution_version("espn-fantasy")
    except PackageNotFoundError:
        installed = "0+unknown"
    return f"espn-fantasy/{installed}"


def default_cache_root() -> Path:
    """``cache_dir()/sources``: deletable raw payloads, resolved at call time so ``FM_CACHE_DIR`` overrides apply."""
    return paths.cache_dir() / CACHE_SUBDIR


class SourceError(RuntimeError):
    """A source could not deliver a dataset."""


class SourceUnavailable(SourceError):
    """The download failed and no cached copy could stand in."""


class SourceSchemaError(SourceError):
    """The payload did not have the expected shape (kept as ``<key>.rejected.<ext>``)."""


# Download failures that may fall back to a stale copy. Deliberately not ``Exception``: the test harness raises a bare
# RuntimeError for accidental network calls, and that must never be swallowed by a fallback.
RECOVERABLE: tuple[type[BaseException], ...] = (
    SourceError,
    httpx.HTTPError,
    OSError,
    ValueError,
    pl.exceptions.PolarsError,
)


@dataclass(frozen=True, slots=True)
class Fetched[T]:
    """A dataset plus its provenance. ``as_of`` is when the payload left the source, not when it was read from cache."""

    data: T
    as_of: datetime
    source: str
    dataset: str
    key: str
    cached: bool = False
    """Served from the on-disk cache rather than downloaded in this call."""
    stale: bool = False
    """Older than the TTL, served because the refresh failed; see ``warnings``."""
    degraded: bool = False
    """The adapter could not get the dataset at all and returned an empty stand-in; see ``warnings``."""
    warnings: tuple[str, ...] = ()
    raw_path: Path | None = None
    """Captured raw payload, when one exists."""

    def age(self, now: datetime | None = None) -> timedelta:
        return (now if now is not None else utcnow()) - self.as_of


class FetchOptions(TypedDict, total=False):
    """Freshness knobs accepted by every dataset method."""

    max_age: timedelta | None
    """Accept a cached copy only if it is at most this old (tighter than the dataset TTL)."""
    fresh_since: datetime | None
    """Accept a cached copy only if it was fetched at or after this instant ("fresh as of T")."""
    force: bool
    """Skip the cache and download."""


class RateLimiter:
    """Minimum spacing between remote calls to one source. ``wait()`` blocks until ``min_interval`` has elapsed."""

    def __init__(
        self,
        min_interval: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")
        self.min_interval = min_interval
        self._monotonic = monotonic
        self._sleep = sleep
        self._last: float | None = None
        self.calls = 0
        self.slept = 0.0

    def wait(self) -> float:
        """Sleep for the remainder of the interval since the previous call; returns the seconds slept."""
        now = self._monotonic()
        delay = 0.0
        if self._last is not None:
            delay = max(0.0, self.min_interval - (now - self._last))
            if delay > 0:
                self._sleep(delay)
        self._last = now + delay
        self.calls += 1
        self.slept += delay
        return delay


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(value: str) -> str:
    """Filesystem-safe path component: ``[A-Za-z0-9._-]`` only, never empty, never a dot-file."""
    cleaned = _UNSAFE.sub("_", value).strip("._")
    return cleaned or "default"


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _digest(payload: bytes) -> bytes:
    return hashlib.blake2b(payload, digest_size=16).digest()


@dataclass(frozen=True, slots=True)
class CacheEntry:
    payload: bytes
    as_of: datetime
    path: Path
    meta: dict[str, Any]


class RawCache:
    """Raw payloads with ``as_of`` metadata on disk, keyed by (source, dataset, key)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, source: str, dataset: str, key: str, ext: str) -> Path:
        return self.root / safe_name(source) / safe_name(dataset) / f"{safe_name(key)}.{ext.lstrip('.')}"

    @staticmethod
    def meta_path(path: Path) -> Path:
        return path.with_name(f"{path.stem}{META_SUFFIX}")

    def read(self, source: str, dataset: str, key: str, ext: str) -> CacheEntry | None:
        path = self.path(source, dataset, key, ext)
        meta_path = self.meta_path(path)
        if not path.is_file() or not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            as_of = datetime.fromisoformat(meta["as_of"])
            payload = path.read_bytes()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("unreadable cache entry %s: %s", path, exc)
            return None
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        return CacheEntry(payload=payload, as_of=as_of, path=path, meta=meta)

    def write(
        self,
        source: str,
        dataset: str,
        key: str,
        ext: str,
        payload: bytes,
        *,
        as_of: datetime,
        meta: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Write payload then metadata (so a half-written entry reads as older, never as fresher, than it is)."""
        path = self.path(source, dataset, key, ext)
        record: dict[str, Any] = {
            "source": source,
            "dataset": dataset,
            "key": key,
            "as_of": as_of.isoformat(),
            "bytes": len(payload),
            **(meta or {}),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(path, payload)
            _atomic_write(self.meta_path(path), json.dumps(record, indent=1, sort_keys=True).encode("utf-8"))
        except OSError as exc:
            logger.warning("could not write cache entry %s: %s", path, exc)
            return None
        return path

    def write_rejected(self, source: str, dataset: str, key: str, ext: str, payload: bytes) -> Path | None:
        """Keep a payload that failed to parse, beside (never over) the last good copy."""
        path = self.path(source, dataset, f"{key}.{REJECTED_TAG}", ext)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(path, payload)
        except OSError as exc:
            logger.warning("could not write rejected payload %s: %s", path, exc)
            return None
        return path


def frame_to_parquet(frame: pl.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def parquet_to_frame(payload: bytes) -> pl.DataFrame:
    return pl.read_parquet(io.BytesIO(payload))


def parse_json(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"))


class Source:
    """Base adapter. Subclasses set ``name``, ``ttl`` per dataset, ``min_interval``, and add one method per dataset."""

    name: ClassVar[str] = "source"
    ttl: ClassVar[Mapping[str, timedelta]] = {}
    default_ttl: ClassVar[timedelta] = timedelta(hours=6)
    min_interval: ClassVar[float] = 0.5
    """Seconds between remote calls when no limiter is injected."""

    def __init__(
        self,
        *,
        cache_root: Path | None = None,
        limiter: RateLimiter | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.cache = RawCache(cache_root if cache_root is not None else default_cache_root())
        self.limiter = limiter if limiter is not None else RateLimiter(self.min_interval)
        self.clock = clock
        # Parsed values per (dataset, key), valid while the cached payload is byte-identical. Saves re-parsing a
        # multi-megabyte payload on every call within a process; hashing it is far cheaper than parsing it.
        self._memo: dict[tuple[str, str], tuple[bytes, Any]] = {}

    def ttl_for(self, dataset: str) -> timedelta:
        return self.ttl.get(dataset, self.default_ttl)

    def fetch[T](
        self,
        dataset: str,
        key: str,
        *,
        download: Callable[[], bytes],
        parse: Callable[[bytes], T],
        ext: str = "json",
        meta: Mapping[str, Any] | None = None,
        stale_ok: bool = True,
        **options: Unpack[FetchOptions],
    ) -> Fetched[T]:
        """Return ``parse(payload)`` for ``dataset``/``key``, from cache when fresh enough, else downloaded.

        Raises :class:`SourceUnavailable` when the download fails with nothing to fall back on, and
        :class:`SourceSchemaError` when a fresh payload does not parse (after keeping it as ``.rejected``).
        """
        max_age = options.get("max_age")
        fresh_since = options.get("fresh_since")
        force = options.get("force", False)
        label = f"{self.name}/{dataset}[{key}]"

        entry = self.cache.read(self.name, dataset, key, ext)
        if entry is not None and not force and self._is_fresh(entry, dataset, max_age, fresh_since):
            try:
                data = self._parse_cached(dataset, key, entry, parse)
            except Exception as exc:  # noqa: BLE001 - any parse failure means the cached copy is unusable
                logger.warning("%s: cached payload no longer parses (%s); refreshing", label, exc)
            else:
                return Fetched(data, entry.as_of, self.name, dataset, key, cached=True, raw_path=entry.path)

        self.limiter.wait()
        try:
            payload = download()
        except RECOVERABLE as exc:
            return self._fallback(entry, exc, dataset, key, parse, stale_ok)
        as_of = self.clock()
        try:
            data = parse(payload)
        except Exception as exc:  # noqa: BLE001 - the payload is ours to inspect; anything raised is a schema problem
            rejected = self.cache.write_rejected(self.name, dataset, key, ext, payload)
            schema_error = SourceSchemaError(f"{label}: payload did not parse ({exc!r}); kept {rejected}")
            schema_error.__cause__ = exc
            return self._fallback(entry, schema_error, dataset, key, parse, stale_ok)
        path = self.cache.write(self.name, dataset, key, ext, payload, as_of=as_of, meta=meta)
        self._memo[(dataset, key)] = (_digest(payload), data)
        return Fetched(data, as_of, self.name, dataset, key, raw_path=path)

    def fetch_frame(
        self,
        dataset: str,
        key: str,
        *,
        load: Callable[[], pl.DataFrame],
        meta: Mapping[str, Any] | None = None,
        stale_ok: bool = True,
        **options: Unpack[FetchOptions],
    ) -> Fetched[pl.DataFrame]:
        """``fetch`` for adapters whose client library already returns a Polars frame; the raw capture is parquet."""
        return self.fetch(
            dataset,
            key,
            download=lambda: frame_to_parquet(load()),
            parse=parquet_to_frame,
            ext="parquet",
            meta=meta,
            stale_ok=stale_ok,
            **options,
        )

    def _is_fresh(
        self,
        entry: CacheEntry,
        dataset: str,
        max_age: timedelta | None,
        fresh_since: datetime | None,
    ) -> bool:
        limit = max_age if max_age is not None else self.ttl_for(dataset)
        if self.clock() - entry.as_of > limit:
            return False
        return fresh_since is None or entry.as_of >= fresh_since

    def _parse_cached[T](self, dataset: str, key: str, entry: CacheEntry, parse: Callable[[bytes], T]) -> T:
        digest = _digest(entry.payload)
        memo = self._memo.get((dataset, key))
        if memo is not None and memo[0] == digest:
            return memo[1]
        data = parse(entry.payload)
        self._memo[(dataset, key)] = (digest, data)
        return data

    def _fallback[T](
        self,
        entry: CacheEntry | None,
        exc: BaseException,
        dataset: str,
        key: str,
        parse: Callable[[bytes], T],
        stale_ok: bool,
    ) -> Fetched[T]:
        label = f"{self.name}/{dataset}[{key}]"
        if entry is not None and stale_ok:
            try:
                data = self._parse_cached(dataset, key, entry, parse)
            except Exception as parse_exc:  # noqa: BLE001 - reported below with the refresh failure
                raise SourceUnavailable(
                    f"{label}: refresh failed ({exc}) and the cached copy does not parse ({parse_exc!r})"
                ) from exc
            logger.warning("%s: refresh failed (%s); serving stale copy from %s", label, exc, entry.as_of.isoformat())
            return Fetched(
                data,
                entry.as_of,
                self.name,
                dataset,
                key,
                cached=True,
                stale=True,
                warnings=(str(exc),),
                raw_path=entry.path,
            )
        if isinstance(exc, SourceSchemaError):
            raise exc
        raise SourceUnavailable(f"{label}: {exc}") from exc


def _retry_after(response: httpx.Response) -> float | None:
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        return None


class HttpSource(Source):
    """A :class:`Source` over an HTTP JSON API: one ``httpx.Client``, timeouts, bounded retries on 429 and 5xx."""

    base_headers: ClassVar[Mapping[str, str]] = {}
    timeout: ClassVar[float] = 30.0
    max_attempts: ClassVar[int] = 3
    backoff: ClassVar[float] = 1.0
    """Seconds before the first retry; doubles per attempt unless the server sends ``Retry-After``."""
    retry_statuses: ClassVar[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], object] = time.sleep,
        cache_root: Path | None = None,
        limiter: RateLimiter | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        super().__init__(cache_root=cache_root, limiter=limiter, clock=clock)
        self._client = client
        self._owns_client = client is None
        self._sleep = sleep

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={"User-Agent": user_agent(), **self.base_headers},
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get_bytes(self, url: str, *, params: Params | None = None, headers: Mapping[str, str] | None = None) -> bytes:
        """GET ``url`` and return the body. Retries transport errors and ``retry_statuses``; other errors raise."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.get(url, params=params, headers=headers)
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning("%s: %s (attempt %d/%d)", self.name, exc, attempt, self.max_attempts)
                delay: float | None = None
            else:
                if response.status_code not in self.retry_statuses:
                    if response.is_error:
                        raise SourceError(f"{self.name}: HTTP {response.status_code} for {response.url}")
                    return response.content
                last_error = SourceError(f"{self.name}: HTTP {response.status_code} for {response.url}")
                delay = _retry_after(response)
            if attempt < self.max_attempts:
                self._sleep(delay if delay is not None else self.backoff * 2 ** (attempt - 1))
        raise SourceError(f"{self.name}: {url} failed after {self.max_attempts} attempts: {last_error}") from last_error

    def get_json(self, url: str, *, params: Params | None = None, headers: Mapping[str, str] | None = None) -> Any:
        return parse_json(self.get_bytes(url, params=params, headers=headers))
