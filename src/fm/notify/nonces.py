"""Single-use approval nonces: what an Approve or Reject button carries (DESIGN section 11).

A proposal's execution token is minted by ``fm.proposals.approve`` and never leaves this machine, so a button sent
before the decision cannot carry it (see the interface note in :mod:`fm.proposals.queue`). Every notification gets a
fresh nonce from :func:`issue` instead, and :mod:`fm.notify.bot` acts on a press only when :func:`consume` accepts it:
the nonce was issued here for that same proposal, has not been used and has not expired. Consuming marks it used, so a
second press, a replayed reply or a forged proposal id is ignored, and after a decision :func:`revoke` kills the
buttons of the proposal's other notifications too.

A nonce outlives its proposal's deadline by :data:`LATE_GRACE`, so a press that arrives just too late still reaches
``fm.proposals``, which expires the proposal and says so, rather than vanishing. Buttons for a proposal without a
deadline work for :data:`DEFAULT_TTL`.

Storage is files under ``<config dir>/notify/nonces/``, because the tick and the CLI issue nonces while ``fm bot``
consumes them: ``<sha256>.json`` holds the record (written under a temporary name and renamed into place, so no reader
sees half of it), and ``<sha256>.used`` marks it consumed. The marker is created with ``O_EXCL``, which exactly one
caller wins, on Windows too (where several concurrent deletes of one file can all report success, so deleting is no
claim). :func:`sweep` removes both files once the nonce has expired. Names are the nonce's SHA-256, so the directory
never holds a usable nonce and a case-insensitive filesystem cannot match a nonce that differs only in case. 144
random bits per nonce mean one is never guessed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from fm import paths
from fm.proposals.policy import as_utc

NONCE_BYTES = 18
NONCE_LENGTH = 24
"""Characters in ``secrets.token_urlsafe(NONCE_BYTES)``: 144 bits as unpadded URL-safe base64."""
NONCE_PATTERN = re.compile(rf"^[A-Za-z0-9_-]{{{NONCE_LENGTH}}}$")
DEFAULT_TTL = timedelta(days=7)
"""How long the buttons of a proposal without a deadline work."""
LATE_GRACE = timedelta(days=1)
"""How long after its proposal's deadline a nonce still reaches ``fm.proposals`` (which then refuses the decision)."""
NONCE_DIR = Path("notify") / "nonces"
STALE_TEMP = timedelta(hours=1)
"""A temporary file this old belongs to a process that died mid-write; :func:`sweep` removes it."""

_RECORD = ".json"
_USED = ".used"
_TEMP = ".tmp"
_FILE_ATTEMPTS = 10
_FILE_RETRY_S = 0.02
"""Windows refuses to replace or delete a file another process has open; such a refusal lasts milliseconds."""


@dataclass(frozen=True, slots=True)
class IssuedNonce:
    """One button token and the proposal it may decide."""

    nonce: str
    proposal_id: int
    issued_at: datetime
    expires_at: datetime

    def is_expired(self, at: datetime) -> bool:
        return self.expires_at <= at


@dataclass(frozen=True, slots=True)
class _Record:
    """A nonce file as read back: everything but the nonce itself, which only the button holds."""

    proposal_id: int
    issued_at: datetime
    expires_at: datetime

    def is_expired(self, at: datetime) -> bool:
        return self.expires_at <= at


def nonce_dir() -> Path:
    """``<config dir>/notify/nonces``, created on first use."""
    path = paths.config_dir() / NONCE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def expiry_for(deadline: datetime | None, now: datetime) -> datetime:
    """When the buttons for a proposal with this deadline stop working: :data:`LATE_GRACE` after the deadline, or
    :data:`DEFAULT_TTL` from ``now`` without one."""
    return now + DEFAULT_TTL if deadline is None else as_utc(deadline) + LATE_GRACE


def issue(proposal_id: int, *, deadline: datetime | None = None, now: datetime | None = None) -> IssuedNonce:
    """Mint a nonce for one notification of ``proposal_id``, whose deadline is ``deadline``."""
    if proposal_id < 1:
        raise ValueError(f"proposal_id must be positive, got {proposal_id!r}")
    at = as_utc(now)
    issued = IssuedNonce(secrets.token_urlsafe(NONCE_BYTES), proposal_id, at, expiry_for(deadline, at))
    _write(_stem(issued.nonce), issued)
    return issued


def consume(nonce: str, proposal_id: int, *, now: datetime | None = None) -> IssuedNonce | None:
    """Use up ``nonce`` for ``proposal_id``. Returns it exactly once, for a nonce issued here for that proposal and not
    yet expired; unknown, used, mismatched, expired and malformed nonces give ``None`` and nothing is raised.

    A real nonce presented with another proposal's id is burnt as well, so it is never honoured afterwards.
    """
    if NONCE_PATTERN.match(nonce) is None:
        return None
    stem = _stem(nonce)
    record = _read(stem.with_suffix(_RECORD))
    if record is None or record.is_expired(as_utc(now)):
        return None
    if not _claim(stem) or record.proposal_id != proposal_id:
        return None  # used before (or by a concurrent press), or a forged proposal id: burnt now either way
    return IssuedNonce(nonce, record.proposal_id, record.issued_at, record.expires_at)


def restore(issued: IssuedNonce) -> None:
    """Make a consumed nonce usable again: its decision did not happen (an unexpected error after :func:`consume`)."""
    stem = _stem(issued.nonce)
    if not stem.with_suffix(_RECORD).exists():
        _write(stem, issued)
    _retry(stem.with_suffix(_USED).unlink, missing_ok=True)


def discard(nonce: str) -> bool:
    """Kill ``nonce`` whatever it was issued for (its push never reached the phone). ``True`` when it was live."""
    if NONCE_PATTERN.match(nonce) is None:
        return False
    stem = _stem(nonce)
    return _read(stem.with_suffix(_RECORD)) is not None and _claim(stem)


def revoke(proposal_id: int) -> int:
    """Kill every live nonce of a proposal, so the buttons of all its notifications are dead. Returns how many."""
    return sum(1 for stem, record in _records() if record.proposal_id == proposal_id and _claim(stem))


def live(proposal_id: int | None = None, *, now: datetime | None = None) -> int:
    """How many unused, unexpired nonces exist, for one proposal or in all."""
    at = as_utc(now)
    return sum(
        1
        for stem, record in _records()
        if not record.is_expired(at)
        and (proposal_id is None or record.proposal_id == proposal_id)
        and not stem.with_suffix(_USED).exists()
    )


def sweep(*, now: datetime | None = None) -> int:
    """Delete the files of expired nonces (used or not) and of corrupt records, plus temporary files left by a crash
    and markers whose record is gone. Returns how many nonces went."""
    at = as_utc(now)
    directory = nonce_dir()
    removed = 0
    for path in directory.glob(f"*{_RECORD}"):
        try:
            record = _load(path)
        except OSError:
            continue  # busy right now; the next sweep will see it
        if record is None or record.is_expired(at):
            # Order does not matter: consume() refuses an expired or corrupt record before it looks at the marker.
            _delete(path.with_suffix(_USED))
            if _delete(path):
                removed += 1
    for marker in directory.glob(f"*{_USED}"):
        if not marker.with_suffix(_RECORD).exists():
            _delete(marker)
    cutoff = time.time() - STALE_TEMP.total_seconds()
    for temp in directory.glob(f"*{_TEMP}"):
        try:
            stale = temp.stat().st_mtime < cutoff
        except OSError:
            continue
        if stale:
            _delete(temp)
    return removed


def _stem(nonce: str) -> Path:
    """The nonce's files without their suffix: ``<nonce dir>/<sha256 hex>``."""
    return nonce_dir() / hashlib.sha256(nonce.encode("ascii")).hexdigest()


def _records() -> list[tuple[Path, _Record]]:
    found: list[tuple[Path, _Record]] = []
    for path in nonce_dir().glob(f"*{_RECORD}"):
        record = _read(path)
        if record is not None:
            found.append((path.with_suffix(""), record))
    return found


def _claim(stem: Path) -> bool:
    """Create the used marker. ``True`` for exactly one caller, however many race for it."""
    try:
        with stem.with_suffix(_USED).open("x", encoding="utf-8"):
            pass
    except OSError:  # FileExistsError: someone used it first
        return False
    return True


def _write(stem: Path, issued: IssuedNonce) -> None:
    """Write the record under a temporary name, then rename it into place, so no reader ever sees half of it."""
    temp = stem.with_name(f"{stem.name}.{secrets.token_hex(4)}{_TEMP}")
    data = {
        "proposal_id": issued.proposal_id,
        "issued_at": issued.issued_at.isoformat(),
        "expires_at": issued.expires_at.isoformat(),
    }
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(data, handle)
    try:
        _retry(os.replace, temp, stem.with_suffix(_RECORD))
    except OSError:
        temp.unlink(missing_ok=True)
        raise


def _read(path: Path) -> _Record | None:
    try:
        return _load(path)
    except OSError:
        return None


def _load(path: Path) -> _Record | None:
    """The record, or ``None`` when the file is corrupt. ``OSError`` when it is missing or cannot be opened now."""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        proposal_id = data["proposal_id"]
        if not isinstance(proposal_id, int) or isinstance(proposal_id, bool) or proposal_id < 1:
            return None
        return _Record(
            proposal_id,
            as_utc(datetime.fromisoformat(data["issued_at"])),
            as_utc(datetime.fromisoformat(data["expires_at"])),
        )
    except (ValueError, TypeError, KeyError):
        return None


def _delete(path: Path) -> bool:
    try:
        _retry(path.unlink)
    except OSError:
        return False
    return True


def _retry[**P](operation: Callable[P, object], *args: P.args, **kwargs: P.kwargs) -> None:
    for attempt in range(_FILE_ATTEMPTS):
        try:
            operation(*args, **kwargs)
        except PermissionError:
            if attempt + 1 == _FILE_ATTEMPTS:
                raise
            time.sleep(_FILE_RETRY_S)
        else:
            return
