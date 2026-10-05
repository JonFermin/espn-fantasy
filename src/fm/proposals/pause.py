"""The ``fm pause`` kill switch (DESIGN section 11 guardrails).

Pausing writes a small marker file, ``paused.json``, in the config dir. While it exists no proposal clears policy,
nothing auto-approves at T-15, and the executor is refused (``begin_execution`` raises ``PausedError``). Human
decisions on proposals that already exist (``fm proposals approve|reject``) still work, so a queue can be cleaned up
while paused; whatever is approved runs after ``fm resume``, unless its deadline has passed by then.

A file rather than a table because every ``fm`` process (tick, bot, CLI) must see it at once, it needs no schema, and
a person can create or delete it by hand in an emergency. A marker that exists but cannot be read still counts as
paused: the switch fails safe.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fm import paths

PAUSE_FILE = "paused.json"


@dataclass(frozen=True, slots=True)
class PauseState:
    """When and why the switch was thrown."""

    since: datetime
    reason: str | None = None

    def describe(self) -> str:
        text = f"paused since {self.since.astimezone(UTC):%Y-%m-%d %H:%M} UTC"
        return f"{text} ({self.reason})" if self.reason else text


def pause_file() -> Path:
    """``$FM_CONFIG_DIR/paused.json``, present only while paused."""
    return paths.config_dir() / PAUSE_FILE


def pause_state() -> PauseState | None:
    """The current pause, or ``None`` when running normally."""
    path = pause_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        since = datetime.fromisoformat(data["since"])
        if since.tzinfo is None:
            raise ValueError("naive timestamp")
        reason = data.get("reason")
        return PauseState(since=since.astimezone(UTC), reason=reason if isinstance(reason, str) and reason else None)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError):
        # The marker exists but is unreadable: stay paused and say so rather than silently resuming.
        try:
            since = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            since = datetime.now(UTC)
        return PauseState(since=since, reason=f"unreadable marker {path}; fm resume clears it")


def is_paused() -> bool:
    return pause_state() is not None


def pause(reason: str | None = None, *, now: datetime | None = None) -> PauseState:
    """Throw the switch. Pausing again while paused keeps the original state (idempotent)."""
    current = pause_state()
    if current is not None:
        return current
    since = datetime.now(UTC) if now is None else now.astimezone(UTC)
    state = PauseState(since=since, reason=reason.strip() or None if reason else None)
    path = pause_file()
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps({"since": state.since.isoformat(), "reason": state.reason}), encoding="utf-8")
    os.replace(temp, path)
    return state


def resume() -> PauseState | None:
    """Clear the switch. Returns the state that was in force, or ``None`` when it was not paused."""
    current = pause_state()
    if current is None:
        return None
    pause_file().unlink(missing_ok=True)
    return current
