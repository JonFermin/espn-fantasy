"""Shared test setup: every test runs in private config/cache dirs, with no inherited secrets and no network.

Unit tests run offline against recorded fixtures (CLAUDE.md). The network guard refuses any socket connect to a
non-loopback address so an accidental live call fails loudly instead of hitting ESPN or an API. Loopback keeps
working because asyncio on Windows uses a loopback socketpair for its self-pipe and local test servers need it. The
guard covers Python's socket layer and asyncio's ``sock_connect`` (the Windows proactor loop connects through
``ConnectEx``, bypassing ``socket.connect``); clients implemented in Rust or C bypass it.
"""

from __future__ import annotations

import ipaddress
import socket
from asyncio.proactor_events import BaseProactorEventLoop
from asyncio.selector_events import BaseSelectorEventLoop
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

# Every variable fm.config reads from .env (fm.config.SECRET_ENV_VARS) plus the SDK's auth token. Kept as a literal
# rather than imported so a broken fm.config cannot take the whole harness down.
SECRET_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ODDS_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "NTFY_TOPIC",
    "NTFY_REPLY_TOPIC",
)


class NetworkDisabledError(RuntimeError):
    """Raised when a test tries to open a connection to a non-loopback address."""


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point FM_CONFIG_DIR / FM_CACHE_DIR at per-test temp dirs and drop inherited secrets."""
    monkeypatch.setenv("FM_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("FM_CACHE_DIR", str(tmp_path / "cache"))
    for key in SECRET_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _is_loopback(address: object) -> bool:
    """True for loopback or local (AF_UNIX) addresses. Hostnames other than ``localhost`` are not resolved."""
    if isinstance(address, str | bytes):  # AF_UNIX path or abstract socket name
        return True
    if not isinstance(address, tuple) or not address:
        return False
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    if not isinstance(host, str):
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check_outbound(address: object) -> None:
    if not _is_loopback(address):
        raise NetworkDisabledError(
            f"outbound network is disabled in unit tests (connect to {address!r}); "
            "use recorded fixtures or respx, see tests/conftest.py"
        )


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block socket connects (sync and asyncio) to anything but loopback."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def connect(self: socket.socket, address: Any) -> None:
        _check_outbound(address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        _check_outbound(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)

    for loop_cls in (BaseSelectorEventLoop, BaseProactorEventLoop):
        monkeypatch.setattr(loop_cls, "sock_connect", _guarded_sock_connect(loop_cls.sock_connect))


def _guarded_sock_connect(real: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
    async def sock_connect(self: Any, sock: socket.socket, address: Any) -> None:
        _check_outbound(address)
        await real(self, sock, address)

    return sock_connect
