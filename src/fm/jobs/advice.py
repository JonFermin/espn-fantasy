"""The advisor in the tick: a Claude client when the configuration allows one, else the reason there is none (DESIGN
section 10, ROADMAP #58).

:func:`advisor_for` is the tick's ``advisor_factory`` (:func:`fm.jobs.tick.tick`): an
:class:`fm.advisor.client.AdvisorClient` over the configured backend, or ``None`` when :func:`advisor_off` says why:
``ANTHROPIC_API_KEY`` is not set (``[llm] backend = "api"``, the SDK), the ``claude`` CLI is not found
(``backend = "claude_code"``, the subscription; :func:`claude_cli_path`), or ``[llm] daily_budget_usd`` is 0 (a cap
of 0 refuses every call, so a client would only log refusals). Either way the tick goes on: news is stored without
signals, rationales stay templates, and every decision runs on the engine alone, with the reason in the league's
notes. It never raises, so a broken advisor can never cost a league its decisions.

The SDK (``anthropic``) takes over a second to import, so this module loads :mod:`fm.advisor.client` only when it
builds a client: a tick without a key, or one whose news is not due, never pays for it, and ``fm.jobs.tick`` stays as
light to import as before.
"""

from __future__ import annotations

import logging
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from fm.config import Config
from fm.store import Store

if TYPE_CHECKING:
    from fm.advisor.client import AdvisorClient, Transport

logger = logging.getLogger(__name__)


def claude_cli_path(
    *, which: Callable[[str], str | None] = shutil.which, home: Path | None = None, platform: str = sys.platform
) -> str | None:
    """The ``claude`` CLI for ``[llm] backend = "claude_code"``: the native installer's ``~/.local/bin`` first (a
    PATH search can take seconds on Windows, and ``fm run`` under Task Scheduler may not have the user's PATH), else
    ``which``; ``None`` when neither has it."""
    name = "claude.exe" if platform == "win32" else "claude"
    local = (home if home is not None else Path.home()) / ".local" / "bin" / name
    if local.is_file():
        return str(local)
    return which("claude")


def advisor_off(config: Config, *, cli_path: Callable[[], str | None] | None = None) -> str | None:
    """Why the advisor cannot run under ``config``, or ``None`` when it can: for the ``api`` backend the key missing
    from ``.env``, for ``claude_code`` no ``claude`` CLI (``cli_path``), and for both a daily budget of 0. The text is
    what the tick notes for a league whose news went untriaged."""
    if config.llm.backend == "claude_code":
        if (cli_path or claude_cli_path)() is None:
            return (
                'the claude CLI was not found ([llm] backend = "claude_code"), so the advisor is off (install '
                "Claude Code and run claude login)"
            )
    elif config.secrets.anthropic_api_key is None:
        return "ANTHROPIC_API_KEY is not set in .env, so the advisor is off (fm config check)"
    if config.llm.daily_budget_usd <= 0:
        return "[llm] daily_budget_usd is 0 in config.toml, so the advisor is off"
    return None


def advisor_for(store: Store, config: Config, *, transport: Transport | None = None) -> AdvisorClient | None:
    """The tick's advisor: a client for the configured model and cap over the configured backend (the ``.env`` key or
    the ``claude`` CLI) or ``transport``, or
    ``None``, logged with the reason, when :func:`advisor_off` gives one. Never raises: a client that cannot be built
    is logged and ``None`` too."""
    reason = advisor_off(config)
    if reason is not None:
        logger.info("advisor: %s", reason)
        return None
    from fm.advisor.client import AdvisorClient  # the SDK import is heavy; see the module docs

    if transport is None and config.llm.backend == "claude_code":
        from fm.advisor.claude_cli import ClaudeCliTransport

        cli = claude_cli_path()
        if cli is None:  # gone since advisor_off looked
            logger.info("advisor: the claude CLI was not found; running without it")
            return None
        transport = ClaudeCliTransport([cli])
    try:
        return AdvisorClient.from_config(store, config, transport=transport)
    except MemoryError:
        raise
    except Exception:  # the tick runs without the advisor rather than without the league
        logger.exception("advisor: could not build the Claude client; running without it")
        return None


__all__ = ["advisor_for", "advisor_off", "claude_cli_path"]
