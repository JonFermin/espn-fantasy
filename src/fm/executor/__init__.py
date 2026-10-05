"""The only package that writes to ESPN: preconditions, run via ``fm.browser``, verify by API re-read (DESIGN 6.3).

Usage::

    from fm.executor import execute, live_opener
    from fm.proposals import get_proposal

    row = get_proposal(store, proposal_id)  # approved: fm proposals approve minted its execution token
    result = execute(store, row.row_id, token=row.execution_token, opener=live_opener())
    result.ok          # verified (or, with dry_run=True, reached its write point and sent nothing)
    result.attempts    # one executions row per mode tried (API, then the UI fallback)
    result.audit_dir   # request, response, verification, screenshots, trace

- :mod:`fm.executor.run`: :func:`execute`, the refusals, the attempts and their write-safety rules, and
  :func:`reconcile_executions`, which ends the runs a killed process left ``executing`` (the tick calls it).
- :mod:`fm.executor.runtime`: what a run works against (read client, write transport, browser) and the live opener.
- :mod:`fm.executor.transport`: the browser-session write transport and the check every request passes.
- :mod:`fm.executor.ui`: the UI-mode session a flow clicks through with; the only gate for state-changing clicks.
- :mod:`fm.executor.verify`: re-reads that decide whether a write happened.
- :mod:`fm.executor.audit`: the per-run audit folder under ``fm.paths.audit_dir()``.

Flows (what to send, how to click, how to check) live in :mod:`fm.browser.flows` and register there.
"""

from __future__ import annotations

from fm.executor.audit import AuditLog
from fm.executor.run import (
    DEFAULT_UI_TIMEOUT_S,
    DEFAULT_VERIFY_ATTEMPTS,
    DEFAULT_VERIFY_INTERVAL_S,
    DEFAULT_WRITE_TIMEOUT_S,
    STALE_EXECUTION,
    ExecutionResult,
    ExecutorError,
    ExecutorOptions,
    NoFlowError,
    PreconditionReadError,
    execute,
    reconcile_executions,
)
from fm.executor.runtime import (
    BrowserLike,
    LiveBrowser,
    Runtime,
    RuntimeOpener,
    dry_run_block_reason,
    live_opener,
    open_live_runtime,
)
from fm.executor.transport import PlaywrightTransport, RefusingTransport, check_write_request
from fm.executor.ui import DEFAULT_MAX_UI_WRITES, UiSession
from fm.executor.verify import ReRead, reread

__all__ = [
    "DEFAULT_MAX_UI_WRITES",
    "DEFAULT_UI_TIMEOUT_S",
    "DEFAULT_VERIFY_ATTEMPTS",
    "DEFAULT_VERIFY_INTERVAL_S",
    "DEFAULT_WRITE_TIMEOUT_S",
    "STALE_EXECUTION",
    "AuditLog",
    "BrowserLike",
    "ExecutionResult",
    "ExecutorError",
    "ExecutorOptions",
    "LiveBrowser",
    "NoFlowError",
    "PlaywrightTransport",
    "PreconditionReadError",
    "ReRead",
    "RefusingTransport",
    "Runtime",
    "RuntimeOpener",
    "UiSession",
    "check_write_request",
    "dry_run_block_reason",
    "execute",
    "live_opener",
    "open_live_runtime",
    "reconcile_executions",
    "reread",
]
