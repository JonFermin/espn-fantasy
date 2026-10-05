"""Single-use approval nonces (``fm.notify.nonces``): issued per notification, consumed once, bound to one proposal,
expiring a grace period after its deadline, revoked after a decision, and safe against concurrent consumers.

The files live under the per-test ``FM_CONFIG_DIR`` (tests/conftest.py); clocks are pinned with ``now=``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from fm import paths
from fm.notify import nonces
from fm.notify.nonces import DEFAULT_TTL, LATE_GRACE, NONCE_LENGTH, NONCE_PATTERN, IssuedNonce

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
DEADLINE = NOW + timedelta(hours=2)
MIXED_CASE = "AbCdEfGhIjKlMnOpQrStUvWx"


def test_a_nonce_is_one_file_named_by_its_hash_that_never_holds_the_nonce() -> None:
    issued = nonces.issue(7, deadline=DEADLINE, now=NOW)
    assert NONCE_PATTERN.match(issued.nonce) is not None and len(issued.nonce) == NONCE_LENGTH
    directory = paths.config_dir() / "notify" / "nonces"
    (path,) = directory.iterdir()
    assert path.name == f"{hashlib.sha256(issued.nonce.encode()).hexdigest()}.json"
    content = path.read_text(encoding="utf-8")
    assert issued.nonce not in content
    assert json.loads(content)["proposal_id"] == 7
    assert (issued.proposal_id, issued.issued_at) == (7, NOW)


def test_buttons_work_a_grace_period_past_the_deadline_or_a_week_without_one() -> None:
    assert nonces.issue(1, deadline=DEADLINE, now=NOW).expires_at == DEADLINE + LATE_GRACE
    assert nonces.issue(1, now=NOW).expires_at == NOW + DEFAULT_TTL


def test_every_notification_gets_a_fresh_nonce() -> None:
    assert len({nonces.issue(1, now=NOW).nonce for _ in range(50)}) == 50
    assert nonces.live() == 50


def test_a_nonce_is_consumed_exactly_once() -> None:
    issued = nonces.issue(3, deadline=DEADLINE, now=NOW)
    assert nonces.consume(issued.nonce, 3, now=NOW) == issued
    assert nonces.consume(issued.nonce, 3, now=NOW) is None
    assert nonces.live() == 0


def test_a_nonce_presented_for_another_proposal_is_refused_and_burnt() -> None:
    issued = nonces.issue(3, now=NOW)
    assert nonces.consume(issued.nonce, 4, now=NOW) is None
    assert nonces.consume(issued.nonce, 3, now=NOW) is None


def test_a_late_press_still_counts_within_the_grace_and_not_after_it() -> None:
    within = nonces.issue(3, deadline=DEADLINE, now=NOW)
    after = nonces.issue(3, deadline=DEADLINE, now=NOW)
    assert nonces.consume(within.nonce, 3, now=DEADLINE + timedelta(minutes=5)) == within
    assert nonces.consume(after.nonce, 3, now=DEADLINE + LATE_GRACE) is None


@pytest.mark.parametrize(
    "nonce",
    ["", "short", MIXED_CASE + "x", MIXED_CASE[:-1] + "!", "../" * 8, "A" * NONCE_LENGTH, "\x00" * NONCE_LENGTH],
)
def test_unknown_and_malformed_nonces_are_refused_quietly(nonce: str) -> None:
    nonces.issue(1, now=NOW)
    assert nonces.consume(nonce, 1, now=NOW) is None
    assert nonces.discard(nonce) is False
    assert nonces.live() == 1


def test_a_nonce_differing_only_in_case_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nonces.secrets, "token_urlsafe", lambda nbytes: MIXED_CASE)
    issued = nonces.issue(3, now=NOW)
    assert nonces.consume(MIXED_CASE.swapcase(), 3, now=NOW) is None
    assert nonces.consume(MIXED_CASE.lower(), 3, now=NOW) is None
    assert nonces.consume(issued.nonce, 3, now=NOW) == issued


def test_revoke_kills_every_button_of_one_proposal_only() -> None:
    first, second = nonces.issue(5, now=NOW), nonces.issue(5, now=NOW)
    other = nonces.issue(6, now=NOW)
    assert nonces.revoke(5) == 2
    assert nonces.consume(first.nonce, 5, now=NOW) is None
    assert nonces.consume(second.nonce, 5, now=NOW) is None
    assert nonces.consume(other.nonce, 6, now=NOW) == other
    assert nonces.revoke(5) == 0


def test_restore_puts_back_a_nonce_whose_decision_did_not_happen() -> None:
    issued = nonces.issue(5, deadline=DEADLINE, now=NOW)
    consumed = nonces.consume(issued.nonce, 5, now=NOW)
    assert consumed is not None
    nonces.restore(consumed)
    assert nonces.consume(issued.nonce, 5, now=NOW) == issued


def test_discard_removes_a_nonce_whatever_it_was_for() -> None:
    issued = nonces.issue(5, now=NOW)
    assert nonces.discard(issued.nonce) is True
    assert nonces.discard(issued.nonce) is False
    assert nonces.consume(issued.nonce, 5, now=NOW) is None


def test_sweep_removes_expired_and_unreadable_nonces_orphan_markers_and_stale_temporaries() -> None:
    kept = nonces.issue(1, deadline=NOW + timedelta(days=3), now=NOW)
    nonces.issue(2, deadline=DEADLINE, now=NOW)
    used = nonces.issue(3, deadline=DEADLINE, now=NOW)
    assert nonces.consume(used.nonce, 3, now=NOW) == used
    directory = nonces.nonce_dir()
    (directory / f"{'0' * 64}.json").write_text("{not json", encoding="utf-8")
    (directory / f"{'1' * 64}.json").write_text('{"proposal_id": true}', encoding="utf-8")
    orphan = directory / f"{'2' * 64}.used"
    orphan.touch()
    stale, fresh = directory / "a.0001.tmp", directory / "b.0002.tmp"
    for temp in (stale, fresh):
        temp.write_text("{}", encoding="utf-8")
    two_hours_ago = time.time() - 7200
    os.utime(stale, (two_hours_ago, two_hours_ago))
    later = DEADLINE + LATE_GRACE
    assert nonces.sweep(now=later) == 4
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [f"{hashlib.sha256(kept.nonce.encode()).hexdigest()}.json", fresh.name]
    )
    assert nonces.live(now=later) == 1
    assert nonces.consume(kept.nonce, 1, now=later) == kept


def test_a_used_nonce_stays_used_until_it_is_swept() -> None:
    issued = nonces.issue(4, deadline=DEADLINE, now=NOW)
    assert nonces.consume(issued.nonce, 4, now=NOW) == issued
    assert nonces.sweep(now=NOW) == 0
    assert nonces.consume(issued.nonce, 4, now=NOW + timedelta(hours=1)) is None
    assert nonces.discard(issued.nonce) is False


def test_live_counts_unexpired_nonces_per_proposal() -> None:
    nonces.issue(1, deadline=DEADLINE, now=NOW)
    nonces.issue(1, now=NOW)
    nonces.issue(2, now=NOW)
    assert (nonces.live(now=NOW), nonces.live(1, now=NOW), nonces.live(3, now=NOW)) == (3, 2, 0)
    assert nonces.live(1, now=DEADLINE + LATE_GRACE) == 1


@pytest.mark.parametrize("rounds", [20])
def test_concurrent_consumers_get_a_nonce_exactly_once(rounds: int) -> None:
    """Eight presses race for each nonce. On Windows several concurrent deletes of one file can all report success, so
    this is what proves the claim is the exclusive create of the used marker."""
    for _ in range(rounds):
        issued = nonces.issue(9, now=NOW)
        start = threading.Barrier(8)

        def attempt(_: int, nonce: str = issued.nonce, barrier: threading.Barrier = start) -> IssuedNonce | None:
            barrier.wait()
            return nonces.consume(nonce, 9, now=NOW)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(8)))
        assert [result for result in results if result is not None] == [issued]


def test_a_proposal_id_must_be_positive() -> None:
    with pytest.raises(ValueError):
        nonces.issue(0, now=NOW)
