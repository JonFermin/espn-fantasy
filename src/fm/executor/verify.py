"""Verification by API re-read (DESIGN 6.3: a write counts only once a re-read of the league confirms it).

:func:`reread` asks the flow's :meth:`fm.browser.flows.Flow.verify` up to ``attempts`` times, ``interval_s`` apart,
until the league shows the change. Re-reads are idempotent, so repeating them is safe, and ESPN can take a moment to
reflect a write. A read that raises is recorded and the next one runs; the last read that answered decides. The
:class:`ReRead` tells the executor which of three things happened: the change is there (``matched``), the league was
read and does not show it (``readable`` and not ``matched``), or nothing could be read at all.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fm.browser.flows import Flow, FlowContext, Preconditions, Verification
from fm.espn.client import EspnClientError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReRead:
    """The outcome of re-reading the league after (or instead of) a write."""

    after: str
    """What prompted it: ``write``, ``ui``, ``unknown`` (a timeout or uncertain answer), ``rejection``."""
    verification: Verification | None
    """The verdict of the last read that answered; ``None`` when every read failed."""
    reads: int
    errors: tuple[str, ...] = ()

    @property
    def readable(self) -> bool:
        return self.verification is not None

    @property
    def matched(self) -> bool:
        return self.verification is not None and self.verification.matched

    @property
    def detail(self) -> str:
        """The verdict's detail, or the read errors when nothing answered."""
        if self.verification is not None:
            return self.verification.detail or ("matched" if self.verification.matched else "mismatch")
        return "; ".join(self.errors) or "no read"

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"after": self.after, "reads": self.reads, "errors": list(self.errors)}
        if self.verification is not None:
            data.update(self.verification.to_json())
        else:
            data["matched"] = None
        return data


def reread(
    flow: Flow[Any],
    ctx: FlowContext[Any],
    pre: Preconditions,
    *,
    after: str,
    attempts: int,
    interval_s: float,
    sleep: Callable[[float], object],
) -> ReRead:
    """Verify up to ``attempts`` times (at least once), sleeping ``interval_s`` between reads, until matched."""
    total = max(1, attempts)
    verification: Verification | None = None
    errors: list[str] = []
    reads = 0
    for attempt in range(1, total + 1):
        reads += 1
        try:
            result = flow.verify(ctx, pre)
            if not isinstance(result, Verification):
                raise TypeError(f"{flow!r}.verify returned {type(result).__name__}, not Verification")
        except Exception as exc:  # a read that fails, or a flow bug, tells us nothing either way; try again
            logger.warning(
                "executor: re-read %d/%d for proposal #%s failed: %s",
                attempt,
                total,
                ctx.proposal.id,
                exc,
                exc_info=not isinstance(exc, EspnClientError),
            )
            errors.append(f"read {attempt}: {type(exc).__name__}: {exc}")
        else:
            verification = result
            if result.matched:
                break
        if attempt < total:
            sleep(interval_s)
    return ReRead(after=after, verification=verification, reads=reads, errors=tuple(errors))
