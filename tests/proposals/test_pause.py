"""The ``fm pause`` kill switch: a marker file under the config dir, idempotent, failing safe when unreadable."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from fm import paths
from fm.proposals import PAUSE_FILE, PauseState, is_paused, pause, pause_file, pause_state, resume

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def test_not_paused_by_default() -> None:
    assert pause_state() is None and not is_paused()
    assert resume() is None
    assert pause_file() == paths.config_dir() / PAUSE_FILE and not pause_file().exists()


def test_pause_and_resume_round_trip() -> None:
    state = pause("travelling", now=NOW)
    assert state == PauseState(since=NOW, reason="travelling")
    assert state.describe() == "paused since 2026-10-04 12:00 UTC (travelling)"
    assert pause_file().is_file() and is_paused()
    assert pause_state() == state  # read back from disk

    cleared = resume()
    assert cleared == state
    assert pause_state() is None and not pause_file().exists()
    assert resume() is None


def test_pausing_twice_keeps_the_first_state() -> None:
    first = pause("one", now=NOW)
    assert pause("two", now=NOW + timedelta(hours=1)) == first
    assert pause_state() == first
    resume()


def test_blank_reason_is_none_and_times_are_stored_in_utc() -> None:
    eastern = datetime(2026, 10, 4, 8, 0, tzinfo=timezone(timedelta(hours=-4)))
    state = pause("   ", now=eastern)
    assert state.reason is None and state.since == NOW and state.since.tzinfo == UTC
    assert state.describe() == "paused since 2026-10-04 12:00 UTC"
    assert pause_state() == state
    resume()


def test_an_unreadable_marker_still_counts_as_paused() -> None:
    pause_file().write_text("not json", encoding="utf-8")
    state = pause_state()
    assert state is not None and is_paused()
    assert state.reason is not None and "unreadable marker" in state.reason and "fm resume" in state.reason
    assert resume() == state
    assert not is_paused()


def test_default_clock_is_now() -> None:
    before = datetime.now(UTC)
    state = pause()
    assert before <= state.since <= datetime.now(UTC) and state.reason is None
    resume()
