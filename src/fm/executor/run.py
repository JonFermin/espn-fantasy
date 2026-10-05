"""The executor: carry one proposal out on ESPN, then verify it by re-reading (DESIGN 6.3, ROADMAP #19).

:func:`execute` is the only write path to ESPN (CLAUDE.md: workers propose, the executor acts). One call, one proposal:

1. Refusals that cost nothing come first: ``fm pause`` (``PausedError``, a refusal and never a retry), a proposal that
   is not approved, is past its deadline, or whose single-use execution token does not match or was spent
   (``LifecycleError``), and a kind no registered flow serves (:class:`NoFlowError`). Nothing was read or sent.
2. The runtime opens (browser profile, ESPN session) and the flow checks its preconditions through the API. When ESPN
   cannot be read the run is refused too (:class:`PreconditionReadError`, or ``AuthError`` for a dead session) and the
   token is not spent, so the proposal can run once ESPN answers again.
3. ``fm.proposals.begin_execution`` consumes the token atomically and moves the proposal to ``executing``; a second
   caller holding the same token loses there. From this point :func:`execute` always ends the proposal ``verified`` or
   ``failed`` and returns an :class:`ExecutionResult` instead of raising. The one exception is an interruption
   (Ctrl+C mid-write, ``SystemExit``), which still ends the run first: its attempt ``unknown`` when the write may have
   started, the proposal ``failed``. A process killed outright leaves the proposal ``executing``;
   :func:`reconcile_executions` ends those once they are old enough that no live run can own them.
4. A failed precondition ends the run ``failed`` with nothing sent. Otherwise each mode in the flow's order (API, then
   UI) is an attempt with its own ``executions`` row:

   - API: the flow builds the request, :func:`fm.executor.transport.check_write_request` must pass it, it is saved to
     the audit folder and the row, then sent exactly once with a hard timeout.
   - UI: the flow clicks through on a fresh page with a hard default timeout for every step, screenshots and a
     Playwright trace; state-changing clicks go through ``UiDriver.confirm``.

   A 2xx, or a UI walk that confirmed, is verified by re-reading (a few polls, as ESPN may lag): match -> ``verified``,
   mismatch -> ``failed``, nothing readable -> ``unknown``. A timeout, a 5xx, a transport failure after sending, or any
   failure after a UI confirm marks the attempt ``unknown`` at once (saved before anything else happens) and forces a
   re-read: no retry and no fallback follow an unknown, because a write without an idempotency key may have landed. A
   definite rejection (3xx/4xx) or an unavailable mode lets the next mode run, only when the flow allows it
   (``Flow.ui_may_follow``) and a re-read shows the rejected attempt left nothing behind.
5. ``fm.proposals.finish_execution`` records the outcome. Unverified means failed: an ``unknown`` attempt leaves the
   proposal ``failed``, and its row tells whoever looks next to re-read the league before acting again.

``dry_run=True`` takes a proposed or approved proposal, needs no token and sends nothing: preconditions, then the API
request is built, checked and saved (or the UI is walked up to its first confirm, with screenshots and a trace), and
the run stops. It records ``dry_run`` rows and leaves the proposal as it was. Each run's request, response,
verification, screenshots and trace are in the audit folder the result names (:mod:`fm.executor.audit`).
"""

from __future__ import annotations

import contextlib
import hmac
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fm.browser import flows
from fm.browser.flows import (
    DryRunStop,
    Flow,
    FlowContext,
    FlowRegistry,
    Mode,
    ModeUnavailableError,
    Preconditions,
    UnknownFlowError,
    WriteOutcome,
    WriteRefusedError,
    WriteResponse,
    WriteTimeoutError,
    WriteUncertainError,
)
from fm.espn.auth import AuthError
from fm.espn.client import EspnClientError
from fm.executor.audit import AuditLog
from fm.executor.runtime import BrowserLike, Runtime, RuntimeOpener
from fm.executor.transport import check_write_request
from fm.executor.ui import DEFAULT_MAX_UI_WRITES, UiSession
from fm.executor.verify import ReRead, reread
from fm.proposals import (
    LifecycleError,
    PausedError,
    Payload,
    begin_execution,
    expire_due,
    finish_execution,
    get_proposal,
    kind_spec,
    parse_payload,
    pause_state,
)
from fm.proposals.policy import as_utc
from fm.store import ExecutionRow, ExecutionStatus, LeagueRow, ProposalRow, Store, utc_now

logger = logging.getLogger(__name__)

DEFAULT_WRITE_TIMEOUT_S = 30.0
DEFAULT_UI_TIMEOUT_S = 30.0
DEFAULT_VERIFY_ATTEMPTS = 3
DEFAULT_VERIFY_INTERVAL_S = 2.0
DRY_RUN_STATUSES = ("proposed", "approved")
"""Proposals a dry run accepts: still open and not executing."""
STALE_EXECUTION = timedelta(minutes=30)
"""How old a run must be before :func:`reconcile_executions` treats it as abandoned. A run is a few reads, one write
with a 30 s timeout and a few re-reads (a UI walk a few minutes at most), so a run this old is no longer going, while
one in progress in another process (``fm bot`` executing next to ``fm tick``) is never touched."""


@dataclass(frozen=True, slots=True)
class ExecutorOptions:
    """Timeouts and verification pacing. ``sleep`` exists so tests can verify without waiting."""

    write_timeout_s: float = DEFAULT_WRITE_TIMEOUT_S
    """Hard limit on one API-mode write, start to answer. Running out marks the attempt unknown."""
    ui_timeout_s: float = DEFAULT_UI_TIMEOUT_S
    """Default limit on every UI step (navigation, waits, clicks)."""
    verify_attempts: int = DEFAULT_VERIFY_ATTEMPTS
    """Re-reads after a write before calling it a mismatch."""
    verify_interval_s: float = DEFAULT_VERIFY_INTERVAL_S
    max_ui_writes: int = DEFAULT_MAX_UI_WRITES
    sleep: Callable[[float], object] = time.sleep


class ExecutorError(Exception):
    """The executor refused the run before anything was sent; the proposal is unchanged."""


class NoFlowError(ExecutorError):
    """No registered flow carries out this kind of proposal in this sport, or not in the requested mode."""


class PreconditionReadError(ExecutorError):
    """ESPN could not be read to check the preconditions. Nothing was sent and the execution token was not spent."""


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """What one :func:`execute` call did: the proposal as it ended, and one ``executions`` row per attempt."""

    proposal: ProposalRow
    flow: str
    dry_run: bool
    preconditions: Preconditions
    attempts: tuple[ExecutionRow, ...]
    audit_dir: Path

    @property
    def last(self) -> ExecutionRow | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def status(self) -> ExecutionStatus | None:
        """The last attempt's status: ``verified``, ``failed``, ``unknown`` or ``dry_run``."""
        return None if self.last is None else self.last.status

    @property
    def ok(self) -> bool:
        """Verified, or a dry run that reached its write point with every precondition met."""
        if self.dry_run:
            return self.last is not None and self.last.status == "dry_run" and self.last.error is None
        return self.proposal.status == "verified"


def execute(
    store: Store,
    proposal_id: int,
    *,
    opener: RuntimeOpener,
    token: str | None = None,
    dry_run: bool = False,
    mode: Mode | str | None = None,
    registry: FlowRegistry | None = None,
    options: ExecutorOptions | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> ExecutionResult:
    """Carry out proposal ``proposal_id`` (see the module docstring for the steps and guarantees).

    ``token`` is the proposal's execution token (``ProposalRow.execution_token``, minted by ``fm proposals approve``);
    a dry run needs none. ``opener`` provides the browser, session and clients (``fm.executor.live_opener`` for the
    real league; ``fm.browser.fakes`` in tests); it is called only after the free refusals passed. ``mode`` forces one
    mode instead of the flow's API-then-UI order. ``registry`` defaults to the process-wide flows.

    Raises only refusals, before anything is sent and with the proposal unchanged: ``PausedError`` /
    ``LifecycleError`` (``fm.proposals.ProposalError``), :class:`ExecutorError`, ``fm.espn.auth.AuthError`` and
    ``fm.browser.session.BrowserError``. Once the token is spent every outcome is an :class:`ExecutionResult`, except
    an interruption (``KeyboardInterrupt``, ``SystemExit``), which propagates once the run is recorded: the attempt
    ``unknown`` if its write may have started (else ``failed``) and the proposal ``failed``.
    """
    opts = options if options is not None else ExecutorOptions()
    at = clock()
    proposal = get_proposal(store, proposal_id)
    spend: str | None = None
    if dry_run:
        _check_dry_run(proposal, at)
    else:
        spend = _check_executable(store, proposal, token, at)
    league = store.leagues.get(proposal.league_id)
    if league is None:
        raise LifecycleError(f"proposal #{proposal_id} belongs to league {proposal.league_id}, which is not stored")
    kind = kind_spec(proposal.kind).kind
    flow = _find_flow(registry, kind.value, league)
    plan = _plan(flow, mode)
    payload = _payload(proposal)
    with opener(league, dry_run=dry_run) as runtime:
        ctx = FlowContext(
            proposal=proposal,
            league=league,
            kind=kind,
            payload=payload,
            reader=runtime.reader,
            now=at,
            dry_run=dry_run,
            member_id=runtime.member_id,
        )
        pre = _preconditions(flow, ctx)
        audit = AuditLog.create(proposal.row_id, at, dry_run=dry_run)
        run = _Run(store, flow, ctx, pre, runtime, audit, opts, clock)
        if spend is not None:
            begin_execution(store, proposal.row_id, spend, now=clock())
        return run.run(plan)


def reconcile_executions(
    store: Store, *, now: datetime | None = None, older_than: timedelta = STALE_EXECUTION
) -> list[ProposalRow]:
    """End the runs a killed process left behind: proposals still ``executing`` whose latest attempt (or, with none,
    the spending of the token) is at least ``older_than`` old.

    A process killed mid-run (a crash, a power cut, a closed console) never records an outcome, so the proposal stays
    ``executing`` and, being open, blocks every new proposal with its dedupe key. Each attempt still ``running``
    becomes ``unknown`` (its write may have landed, so the league needs a re-read before anything else) and the
    proposal ``failed``. A run younger than ``older_than`` may still be going in another process and is left alone.
    The tick (ROADMAP #29) calls this before it executes anything. Returns the proposals it failed.
    """
    at = as_utc(now)
    failed: list[ProposalRow] = []
    for proposal in store.proposals.find(statuses=("executing",)):
        with store.db.transaction():
            current = store.proposals.get(proposal.row_id)
            if current is None or current.status != "executing":
                continue  # finished in the meantime
            rows = store.executions.for_proposal(current.row_id)
            started = max((row.started_at for row in rows), default=current.token_consumed_at)
            if started is not None and at - started < older_than:
                continue
            for row in rows:
                if row.status == "running":
                    store.executions.update(
                        row.model_copy(
                            update={
                                "status": "unknown",
                                "finished_at": at,
                                "error": "the run stopped mid-attempt (the process ended before it could record an "
                                "outcome); the write may have landed, check the league before acting again",
                            }
                        )
                    )
            failed.append(finish_execution(store, current.row_id, "failed"))
            logger.warning("executor: proposal #%d was left executing by a run that never finished", current.row_id)
    return failed


# --- refusals ---------------------------------------------------------------------------------------------------------


def _check_dry_run(proposal: ProposalRow, at: datetime) -> None:
    if proposal.status not in DRY_RUN_STATUSES:
        raise LifecycleError(
            f"cannot dry-run proposal #{proposal.row_id}: it is {proposal.status}; only proposed and approved "
            "proposals can be"
        )
    if proposal.deadline is not None and proposal.deadline <= at:
        raise LifecycleError(
            f"cannot dry-run proposal #{proposal.row_id}: "
            f"its deadline {proposal.deadline:%Y-%m-%d %H:%M} UTC has passed"
        )


def _check_executable(store: Store, proposal: ProposalRow, token: str | None, at: datetime) -> str:
    """The checks ``begin_execution`` makes atomically later, made here first so a refused run opens no browser and
    reads nothing. Returns the token to spend."""
    pid = proposal.row_id
    paused = pause_state()
    if paused is not None:
        raise PausedError(f"cannot execute proposal #{pid}: {paused.describe()}; run fm resume")
    if proposal.status in DRY_RUN_STATUSES and proposal.deadline is not None and proposal.deadline <= at:
        expire_due(store, now=at, league_id=proposal.league_id)
        raise LifecycleError(
            f"cannot execute proposal #{pid}: it expired at {proposal.deadline:%Y-%m-%d %H:%M} UTC (its deadline)"
        )
    if proposal.status != "approved":
        hint = f"; approve it first (fm proposals approve {pid})" if proposal.status == "proposed" else ""
        raise LifecycleError(f"cannot execute proposal #{pid}: it is {proposal.status}{hint}")
    stored = proposal.execution_token
    if not token or stored is None or proposal.token_consumed_at is not None or not _same(stored, token):
        raise LifecycleError(f"proposal #{pid}: execution token does not match or was already used")
    return token


def _same(stored: str, given: str) -> bool:
    return hmac.compare_digest(stored.encode("utf-8"), given.encode("utf-8"))


def _payload(proposal: ProposalRow) -> Payload:
    try:
        return parse_payload(proposal)
    except ValidationError as exc:
        raise ExecutorError(f"proposal #{proposal.row_id} has an unreadable {proposal.kind} payload: {exc}") from exc


def _find_flow(registry: FlowRegistry | None, kind: str, league: LeagueRow) -> Flow[Any]:
    try:
        if registry is None:
            return flows.flow_for(kind, league.sport)
        return registry.flow_for(kind, league.sport)
    except UnknownFlowError as exc:
        raise NoFlowError(str(exc)) from exc


def _plan(flow: Flow[Any], mode: Mode | str | None) -> tuple[Mode, ...]:
    modes = tuple(Mode(item) for item in flow.modes)
    if mode is None:
        return modes
    wanted = Mode(mode)
    if wanted not in modes:
        raise NoFlowError(f"{flow.name} has no {wanted.value} mode; it runs in {', '.join(m.value for m in modes)}")
    return (wanted,)


def _preconditions(flow: Flow[Any], ctx: FlowContext[Any]) -> Preconditions:
    """The flow's precondition check. Read failures and flow bugs are refusals: the token is not spent yet."""
    try:
        pre = flow.check(ctx)
    except AuthError:  # EspnAuthError included: the session is gone, run fm login
        raise
    except EspnClientError as exc:
        raise PreconditionReadError(
            f"could not read ESPN to check proposal #{ctx.proposal.id}: {exc}; nothing was sent, run it again later"
        ) from exc
    except Exception as exc:
        logger.exception("executor: %r failed checking proposal #%s", flow, ctx.proposal.id)
        raise ExecutorError(f"{flow.name} could not check its preconditions: {type(exc).__name__}: {exc}") from exc
    if not isinstance(pre, Preconditions):
        raise ExecutorError(f"{flow.name}.check returned {type(pre).__name__}, not Preconditions")
    return pre


# --- one run ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Attempt:
    row: ExecutionRow
    fallback: bool = False
    """A definite failure that wrote nothing, after which the next mode may run."""


class _Run:
    """The attempts of one run, past the refusals: rows, audit files, the write, the re-reads."""

    def __init__(
        self,
        store: Store,
        flow: Flow[Any],
        ctx: FlowContext[Any],
        pre: Preconditions,
        runtime: Runtime,
        audit: AuditLog,
        options: ExecutorOptions,
        clock: Callable[[], datetime],
    ) -> None:
        self.store = store
        self.flow = flow
        self.ctx = ctx
        self.pre = pre
        self.runtime = runtime
        self.audit = audit
        self.options = options
        self.clock = clock
        self.rows: list[ExecutionRow] = []
        self._sending = False
        self._ui: UiSession | None = None
        self.shared = self._write("preconditions.json", self._preconditions_record())

    # --- entry points ---------------------------------------------------------------------------------------------

    def run(self, plan: tuple[Mode, ...]) -> ExecutionResult:
        """The attempts, as a dry run or for real. Whatever escapes them (Ctrl+C mid-write, ``SystemExit``, a store
        failure) is recorded by :meth:`_interrupted` before it propagates."""
        try:
            return self.dry_run(plan) if self.ctx.dry_run else self.execute(plan)
        except BaseException as exc:
            self._interrupted(exc)
            raise

    def dry_run(self, plan: tuple[Mode, ...]) -> ExecutionResult:
        if not self.pre.ok:
            self._close(self._insert(plan[0]), "dry_run", error=self._precondition_error())
            return self._result()
        for mode in plan:
            if not self._attempt(mode).fallback:
                break
        return self._result()

    def execute(self, plan: tuple[Mode, ...]) -> ExecutionResult:
        if not self.pre.ok:
            self._close(self._insert(plan[0]), "failed", error=self._precondition_error())
            return self._finish(verified=False)
        for index, mode in enumerate(plan):
            attempt = self._attempt(mode)
            if attempt.row.status == "verified":
                return self._finish(verified=True)
            if attempt.row.status != "failed" or not attempt.fallback or index + 1 == len(plan):
                return self._finish(verified=False)
            # Nothing was written, as far as ESPN said. Re-read before a second path writes, in case it was anyway.
            check = self._reread("rejection", attempts=1)
            row = self._record_check(attempt.row, check)
            if check.matched:
                self._save(row, status="verified", error=f"{row.error}; the re-read shows the change in place anyway")
                return self._finish(verified=True)
            if not check.readable:
                self._save(row, error=f"{row.error}; no fallback: the league could not be re-read ({check.detail})")
                return self._finish(verified=False)
            logger.info("executor: proposal #%d falls back to %s mode", self.ctx.proposal.id, plan[index + 1].value)
        return self._finish(verified=False)

    # --- attempts -------------------------------------------------------------------------------------------------

    def _attempt(self, mode: Mode) -> _Attempt:
        row = self._insert(mode)
        self._sending = False
        self._ui = None
        try:
            return self._api(row) if mode is Mode.API else self._ui_attempt(row)
        except Exception as exc:  # a bug in a flow or here: the attempt still has to end in a definite state
            logger.exception("executor: %s attempt for proposal #%d crashed", mode.value, self.ctx.proposal.id)
            current = self._current(row)
            message = f"{type(exc).__name__}: {exc}"
            if self.ctx.dry_run:
                return self._close(current, "dry_run", error=f"crashed: {message}")
            if self._write_may_have_started():
                return self._unknown(current, f"crashed after the write started: {message}")
            return self._close(current, "failed", error=f"crashed before writing: {message}")

    def _api(self, row: ExecutionRow) -> _Attempt:
        try:
            request = self.flow.build_request(self.ctx, self.pre)
        except ModeUnavailableError as exc:
            return self._close(row, self._no_write, error=f"API mode unavailable: {exc}", fallback=True)
        except Exception as exc:  # a flow bug; nothing was sent, so the UI may still do it
            logger.exception("executor: %r could not build its request", self.flow)
            return self._close(
                row, self._no_write, error=f"could not build the request: {type(exc).__name__}: {exc}", fallback=True
            )
        record = request.to_json()
        row = self._save(row, request=record, artifacts=[*row.artifacts, *self._write("api-request.json", record)])
        try:
            check_write_request(request, self.ctx.league)
        except WriteRefusedError as exc:
            return self._close(row, self._no_write, error=str(exc))
        if self.ctx.dry_run:
            return self._close(row, "dry_run", error=None)
        self._sending = True
        try:
            response = self.runtime.transport.send(request, timeout_s=self.options.write_timeout_s)
        except WriteRefusedError as exc:
            return self._close(row, "failed", error=f"the transport refused to send: {exc}")
        except WriteTimeoutError as exc:
            return self._unknown(row, f"the write timed out after {self.options.write_timeout_s:g} s ({exc})")
        except WriteUncertainError as exc:
            return self._unknown(row, f"the write may have reached ESPN ({exc})")
        except Exception as exc:  # anything else mid-send is as uncertain as a timeout
            logger.exception("executor: transport failed while sending proposal #%d", self.ctx.proposal.id)
            return self._unknown(row, f"the write failed in flight ({type(exc).__name__}: {exc})")
        record = response.to_json()
        row = self._save(
            row,
            response=record,
            artifacts=[*row.artifacts, *self._write("api-response.json", record)],
            espn_transaction_id=self._transaction_id(response),
        )
        if response.outcome is WriteOutcome.ACCEPTED:
            return self._verify(row, after="write")
        if response.outcome is WriteOutcome.UNKNOWN:
            return self._unknown(row, f"ESPN answered {response.describe()}; the write may have been applied")
        return self._close(
            row,
            "failed",
            error=f"ESPN rejected the request: {response.describe()}",
            fallback=self._ui_may_follow(response),
        )

    def _ui_attempt(self, row: ExecutionRow) -> _Attempt:
        browser = self.runtime.browser
        timeout_ms = self.options.ui_timeout_s * 1000
        try:
            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            page.set_default_navigation_timeout(timeout_ms)
        except Exception as exc:
            return self._close(row, self._no_write, error=f"could not open a page: {type(exc).__name__}: {exc}")
        ui = UiSession(page, self.audit, dry_run=self.ctx.dry_run, max_writes=self.options.max_ui_writes)
        self._ui = ui
        tracing = self._start_trace(browser)
        failure: Exception | None = None
        url = ""
        try:
            self.flow.run_ui(self.ctx, ui, self.pre)
        except DryRunStop:
            pass
        except Exception as exc:
            failure = exc
            if not isinstance(exc, ModeUnavailableError):
                logger.warning("executor: UI walk for proposal #%d failed: %s", self.ctx.proposal.id, exc)
        finally:
            ui.screenshot("error" if failure is not None else "end")
            trace = self._stop_trace(browser, tracing)
            with contextlib.suppress(Exception):
                url = page.url
            with contextlib.suppress(Exception):
                page.close()
        record: dict[str, Any] = {"mode": Mode.UI.value, "url": url, "confirms": list(ui.confirms)}
        if ui.stopped_before is not None:
            record["stopped_before"] = ui.stopped_before
        artifacts = [*row.artifacts, *ui.artifacts, *([trace] if trace else [])]
        row = self._save(row, request=record, artifacts=artifacts)
        unavailable = isinstance(failure, ModeUnavailableError)
        if self.ctx.dry_run:
            if ui.stopped_before is not None:
                return self._close(row, "dry_run", error=None)
            if failure is not None:
                return self._close(
                    row,
                    "dry_run",
                    error=f"the UI walk failed before its first confirm: {_describe(failure)}",
                    fallback=unavailable,
                )
            return self._close(row, "dry_run", error="the UI walk ended without reaching a confirm")
        cause = failure if failure is not None else ui.confirm_error
        if ui.write_started and cause is not None:
            return self._unknown(row, f"the UI walk failed after a confirm click ({_describe(cause)})")
        if failure is not None:
            return self._close(
                row,
                "failed",
                error=f"the UI walk failed before any confirm, so nothing was written: {_describe(failure)}",
                fallback=unavailable,
            )
        if not ui.write_started:
            return self._close(row, "failed", error="the UI walk ended without a confirm click; nothing was written")
        return self._verify(row, after="ui")

    # --- outcomes -------------------------------------------------------------------------------------------------

    def _verify(self, row: ExecutionRow, *, after: str) -> _Attempt:
        """After a write ESPN accepted: matched -> verified, mismatch -> failed, nothing readable -> unknown."""
        check = self._reread(after)
        row = self._record_check(row, check)
        if check.matched:
            return self._close(row, "verified", error=None)
        if check.readable:
            return self._close(row, "failed", error=f"the re-read does not show the change: {check.detail}")
        return self._close(
            row,
            "unknown",
            error=f"could not re-read ESPN to verify the write ({check.detail}); check the league before acting again",
        )

    def _unknown(self, row: ExecutionRow, error: str) -> _Attempt:
        """The write may have landed: say so in the store first, then re-read. Never retried, never followed."""
        row = self._save(row, status="unknown", error=error, finished_at=self.clock())
        check = self._reread("unknown")
        row = self._record_check(row, check)
        if check.matched:
            return self._close(row, "verified", error=f"{error}; the re-read shows the change, so it went through")
        return self._close(
            row,
            "unknown",
            error=f"{error}; the re-read does not confirm it ({check.detail}); check the league before acting again",
        )

    def _finish(self, *, verified: bool) -> ExecutionResult:
        final = finish_execution(self.store, self.ctx.proposal.row_id, "verified" if verified else "failed")
        return self._result(final)

    def _interrupted(self, exc: BaseException) -> None:
        """Record a run that something escaped, before it propagates; a failure to record it is logged, not raised.

        The open attempt ends ``unknown`` when its write may have started (the request may have reached ESPN, so the
        league needs a re-read before anything else), else ``failed``, or ``dry_run`` in a dry run; an attempt already
        ``unknown`` whose re-read was cut short says so. An executing proposal then ends ``verified`` if its last
        attempt was, else ``failed``. Left alone it would stay ``executing`` for good: nothing moves a proposal out of
        it, and as an open proposal it would keep every new one with its dedupe key from being made.
        """
        cause = _describe(exc)
        pid = self.ctx.proposal.row_id
        try:
            row = self._latest()
            if row is not None and row.status == "running":
                if self.ctx.dry_run:
                    row = self._close(row, "dry_run", error=f"interrupted ({cause}); nothing was sent").row
                elif self._write_may_have_started():
                    row = self._save(
                        row,
                        status="unknown",
                        finished_at=self.clock(),
                        error=f"interrupted after the write started ({cause}); check the league before acting again",
                    )
                else:
                    row = self._close(row, "failed", error=f"interrupted before anything was written ({cause})").row
            elif row is not None and row.status == "unknown" and row.verification is None:
                row = self._save(row, error=f"{row.error}; interrupted ({cause}) before the re-read finished")
            if not self.ctx.dry_run and get_proposal(self.store, pid).status == "executing":
                verified = row is not None and row.status == "verified"
                finish_execution(self.store, pid, "verified" if verified else "failed")
        except Exception:
            logger.exception("executor: could not record the interruption of proposal #%d", pid)
        logger.warning("executor: proposal #%d was interrupted (%s)", pid, cause)

    def _latest(self) -> ExecutionRow | None:
        """The run's newest attempt as the store has it: an interruption can land between a save and ``self.rows``."""
        if not self.rows:
            return None
        newest = self.rows[-1]
        return self.store.executions.get(newest.row_id) or newest

    def _result(self, proposal: ProposalRow | None = None) -> ExecutionResult:
        return ExecutionResult(
            proposal=proposal if proposal is not None else get_proposal(self.store, self.ctx.proposal.row_id),
            flow=self.flow.name,
            dry_run=self.ctx.dry_run,
            preconditions=self.pre,
            attempts=tuple(self.rows),
            audit_dir=self.audit.folder,
        )

    # --- helpers --------------------------------------------------------------------------------------------------

    @property
    def _no_write(self) -> ExecutionStatus:
        """The status of an attempt that ended before writing: ``dry_run`` in a dry run, else ``failed``."""
        return "dry_run" if self.ctx.dry_run else "failed"

    def _write_may_have_started(self) -> bool:
        return self._sending or (self._ui is not None and self._ui.write_started)

    def _reread(self, after: str, *, attempts: int | None = None) -> ReRead:
        return reread(
            self.flow,
            self.ctx,
            self.pre,
            after=after,
            attempts=self.options.verify_attempts if attempts is None else attempts,
            interval_s=self.options.verify_interval_s,
            sleep=self.options.sleep,
        )

    def _record_check(self, row: ExecutionRow, check: ReRead) -> ExecutionRow:
        record = check.to_json()
        name = f"{row.mode}-verification.json" if check.after != "rejection" else f"{row.mode}-reread.json"
        return self._save(row, verification=record, artifacts=[*row.artifacts, *self._write(name, record)])

    def _transaction_id(self, response: WriteResponse) -> str | None:
        try:
            return self.flow.transaction_id(response)
        except Exception:  # an id is bookkeeping; a flow bug here must not change the outcome
            logger.exception("executor: %r could not read the transaction id", self.flow)
            return None

    def _ui_may_follow(self, response: WriteResponse) -> bool:
        if response.outcome is not WriteOutcome.REJECTED:
            return False
        try:
            return bool(self.flow.ui_may_follow(response))
        except Exception:
            logger.exception("executor: %r could not judge the rejection; no fallback", self.flow)
            return False

    def _start_trace(self, browser: BrowserLike) -> bool:
        try:
            browser.start_trace()
        except Exception as exc:  # the trace is evidence, not a reason to stop
            logger.warning("executor: could not start the Playwright trace: %s", exc)
            return False
        return True

    def _stop_trace(self, browser: BrowserLike, started: bool) -> str | None:
        if not started:
            return None
        target = self.audit.path("ui-trace.zip")
        try:
            browser.stop_trace(target)
        except Exception as exc:  # evidence again: never a reason to change the outcome
            logger.warning("executor: could not save the Playwright trace: %s", exc)
            return None
        return self.audit.relative(target) if target.exists() else None

    def _write(self, name: str, data: Any) -> list[str]:
        """Save an audit file; ``[relative path]``, or ``[]`` when the disk refuses. The row keeps the same data, and
        a write in flight must reach a final state whatever happens to its evidence."""
        try:
            return [self.audit.write_json(name, data)]
        except OSError as exc:
            logger.warning("executor: could not save %s to the audit folder: %s", name, exc)
            return []

    def _insert(self, mode: Mode) -> ExecutionRow:
        row = self.store.executions.insert(
            ExecutionRow(
                proposal_id=self.ctx.proposal.row_id,
                mode="api" if mode is Mode.API else "ui",
                status="running",
                started_at=self.clock(),
                artifacts=list(self.shared),
            )
        )
        self.rows.append(row)
        return row

    def _save(self, row: ExecutionRow, **changes: Any) -> ExecutionRow:
        saved = self.store.executions.update(row.model_copy(update=changes))
        self.rows = [saved if existing.id == saved.id else existing for existing in self.rows]
        return saved

    def _current(self, row: ExecutionRow) -> ExecutionRow:
        return next((existing for existing in self.rows if existing.id == row.id), row)

    def _close(
        self, row: ExecutionRow, status: ExecutionStatus, *, error: str | None, fallback: bool = False
    ) -> _Attempt:
        return _Attempt(self._save(row, status=status, error=error, finished_at=self.clock()), fallback)

    def _precondition_error(self) -> str:
        return "preconditions failed, nothing was sent: " + "; ".join(self.pre.failures)

    def _preconditions_record(self) -> dict[str, Any]:
        """The run's context for the audit folder. The proposal row is not dumped whole: it holds the token."""
        proposal = self.ctx.proposal
        league = self.ctx.league
        return {
            "proposal": {
                "id": proposal.id,
                "kind": proposal.kind,
                "status": proposal.status,
                "policy": proposal.policy,
                "scoring_period_id": proposal.scoring_period_id,
                "deadline": proposal.deadline,
                "decided_by": proposal.decided_by,
                "payload": self.ctx.payload.model_dump(mode="json"),
                "summary": self.ctx.payload.summary(),
            },
            "league": {
                "key": league.key,
                "sport": league.sport,
                "espn_league_id": league.espn_league_id,
                "season": league.season,
                "team_id": league.team_id,
            },
            "flow": self.flow.name,
            "dry_run": self.ctx.dry_run,
            "started_at": self.ctx.now,
            "preconditions": self.pre.to_json(),
        }


def _describe(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
