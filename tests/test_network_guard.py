"""The autouse guards: non-loopback connects fail loudly while loopback keeps working (sync and asyncio), no real
browser starts, and neither guard nor the private dirs can be undone by a test's own ``monkeypatch``."""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path

import httpx
import pytest

from fm import paths
from fm.browser import session as browser_session

# TEST-NET-1 (RFC 5737) is never routable, so even a broken guard cannot reach a real host.
UNROUTABLE = ("192.0.2.1", 9)
MESSAGE = "outbound network is disabled"


def blocked() -> AbstractContextManager[pytest.ExceptionInfo[RuntimeError]]:
    return pytest.raises(RuntimeError, match=MESSAGE)


def leaf_messages(exc: BaseException) -> list[str]:
    """Messages of ``exc`` and, for exception groups (anyio wraps task failures), of every nested exception."""
    if isinstance(exc, BaseExceptionGroup):
        return [message for child in exc.exceptions for message in leaf_messages(child)]
    return [str(exc)]


@pytest.fixture
def local_server() -> Iterator[tuple[str, int]]:
    server = socket.create_server(("127.0.0.1", 0))
    try:
        yield server.getsockname()[:2]
    finally:
        server.close()


def test_non_loopback_connect_is_blocked() -> None:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        with blocked():
            sock.connect(UNROUTABLE)
        with blocked():
            sock.connect_ex(UNROUTABLE)


def test_hostnames_are_blocked_without_resolving() -> None:
    with socket.socket() as sock, blocked():
        sock.connect(("example.invalid", 80))


def test_loopback_connect_works(local_server: tuple[str, int]) -> None:
    with socket.create_connection(local_server, timeout=2):
        pass
    with socket.create_connection(("localhost", local_server[1]), timeout=2):
        pass


def test_sync_httpx_is_blocked() -> None:
    with blocked():
        httpx.get("http://192.0.2.1:9/", timeout=1.0)


def test_asyncio_loopback_works_and_non_loopback_is_blocked(local_server: tuple[str, int]) -> None:
    async def main() -> None:
        _, writer = await asyncio.open_connection(*local_server)
        writer.close()
        await writer.wait_closed()
        with blocked():
            await asyncio.open_connection(*UNROUTABLE)

    asyncio.run(main())


def test_async_httpx_is_blocked() -> None:
    async def main() -> None:
        async with httpx.AsyncClient(timeout=1.0) as client:
            with pytest.raises((RuntimeError, BaseExceptionGroup)) as info:
                await client.get("http://192.0.2.1:9/")
        assert any(MESSAGE in message for message in leaf_messages(info.value))

    asyncio.run(main())


def test_a_real_browser_launch_fails_the_test() -> None:
    with pytest.raises(AssertionError, match="real browser launch in a unit test"):
        browser_session.sync_playwright()


def test_undoing_the_shared_monkeypatch_keeps_every_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(tmp_path / "elsewhere"))
    monkeypatch.undo()  # drops this test's own patches only
    for name in (paths.CONFIG_DIR_ENV, paths.CACHE_DIR_ENV):
        assert Path(os.environ[name]).is_relative_to(tmp_path)
    assert paths.config_dir().is_relative_to(tmp_path) and paths.state_db().is_relative_to(tmp_path)
    with socket.socket() as sock, blocked():
        sock.connect(UNROUTABLE)
    with pytest.raises(AssertionError, match="real browser launch in a unit test"):
        browser_session.sync_playwright()
