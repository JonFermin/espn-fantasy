"""Runtime directories (DESIGN section 14). Everything mutable lives outside the repo.

``config_dir()`` (``~/.config/espn-fantasy``) holds ``config.toml``, ``.env``, the browser profile, ``state.db`` and
audit artifacts. ``cache_dir()`` (``~/.cache/espn-fantasy``) holds raw source responses and parquet snapshots and is
safe to delete. ``FM_CONFIG_DIR`` / ``FM_CACHE_DIR`` override them (tests point both at temp dirs). Both are created
on first use.

Windows deliberately does not use ``%APPDATA%``: packaged (MSIX) apps virtualize AppData writes, so a process started
from such an app and one started by Task Scheduler would see different state. The home directory is not virtualized.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_DIRNAME = "espn-fantasy"
CONFIG_DIR_ENV = "FM_CONFIG_DIR"
CACHE_DIR_ENV = "FM_CACHE_DIR"


def _resolve(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    path = Path(os.path.expandvars(override)).expanduser() if override else default
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_dir() -> Path:
    """Config, state DB, browser profile and audit root: ``$FM_CONFIG_DIR`` or ``~/.config/espn-fantasy``."""
    return _resolve(CONFIG_DIR_ENV, Path.home() / ".config" / APP_DIRNAME)


def cache_dir() -> Path:
    """Deletable cache root: ``$FM_CACHE_DIR`` or ``~/.cache/espn-fantasy``."""
    return _resolve(CACHE_DIR_ENV, Path.home() / ".cache" / APP_DIRNAME)


def config_file() -> Path:
    """``config.toml`` (leagues, policy, llm, notify)."""
    return config_dir() / "config.toml"


def env_file() -> Path:
    """``.env`` with API keys and bot tokens. Nothing ESPN-related goes here; cookies come from the browser profile."""
    return config_dir() / ".env"


def state_db() -> Path:
    """The SQLite state database."""
    return config_dir() / "state.db"


def browser_profile_dir() -> Path:
    """Persistent Playwright profile holding the ESPN session. Never committed, never copied into fixtures."""
    return config_dir() / "browser-profile"


def audit_dir() -> Path:
    """Execution artifacts: requests, responses, screenshots, Playwright traces. Created on first use."""
    path = config_dir() / "audit"
    path.mkdir(parents=True, exist_ok=True)
    return path
