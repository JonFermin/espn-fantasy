"""UI mode, the executor's side: the :class:`fm.browser.flows.UiDriver` a flow's ``run_ui`` receives (DESIGN 6.3).

A UI flow clicks through on :attr:`UiSession.page` with role and text locators. Navigation, menus and form fields are
its own business; a click that changes the league (the final Submit, a "Here" that saves a lineup move) must go
through :meth:`UiSession.confirm`, which is where the executor's rules live:

- the control must be visible first, so a page that drifted fails before anything was clicked (a definite failure);
- a screenshot before and after every confirm goes to the audit folder;
- once a confirm click starts, the attempt's outcome is unknown until a re-read says otherwise, even if the click
  raises or the flow swallows the error (:attr:`UiSession.write_started`, :attr:`UiSession.confirm_error`);
- at most ``max_writes`` confirms per attempt (a lineup rotation in the ESPN UI saves one move per click);
- in a dry run the first confirm stops the walk with :class:`fm.browser.flows.DryRunStop` before clicking.
"""

from __future__ import annotations

import logging

from fm.browser.flows import DryRunStop, LocatorLike, PageLike, WriteRefusedError
from fm.executor.audit import AuditLog

logger = logging.getLogger(__name__)

DEFAULT_MAX_UI_WRITES = 10
"""Most state-changing clicks one UI attempt may make."""


class UiSession:
    """One UI-mode attempt on one page."""

    def __init__(
        self,
        page: PageLike,
        audit: AuditLog,
        *,
        dry_run: bool,
        max_writes: int = DEFAULT_MAX_UI_WRITES,
        prefix: str = "ui",
    ) -> None:
        self._page = page
        self._audit = audit
        self._dry_run = dry_run
        self.max_writes = max_writes
        self.prefix = prefix
        self.confirms: list[str] = []
        """The confirm clicks started, in order (``what`` of each)."""
        self.stopped_before: str | None = None
        """In a dry run: the confirm the walk stopped at."""
        self.confirm_error: BaseException | None = None
        """What a confirm click raised, kept even if the flow catches it."""
        self.artifacts: list[str] = []
        """Screenshots taken, relative to the audit root."""
        self._shots = 0

    @property
    def page(self) -> PageLike:
        return self._page

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def write_started(self) -> bool:
        """A confirm click has started, so the league may have changed."""
        return bool(self.confirms)

    def screenshot(self, name: str) -> None:
        """Save a full-page screenshot to the audit folder. Best effort: evidence never breaks a run."""
        self._shots += 1
        target = self._audit.path(f"{self.prefix}-{self._shots:02d}-{name}.png")
        try:
            self._page.screenshot(path=target, full_page=True)
        except Exception as exc:  # a screenshot is evidence, never a reason to stop
            logger.warning("executor: screenshot %s failed: %s", target.name, exc)
            return
        if target.exists():
            self.artifacts.append(self._audit.relative(target))

    def confirm(self, target: LocatorLike, *, what: str) -> None:
        """Make one state-changing click on ``target``; ``what`` names it for the audit trail (``submit lineup``).

        Waits for the control to be visible first. In a dry run, stops the walk there with ``DryRunStop`` instead of
        clicking. Raises ``WriteRefusedError`` past ``max_writes`` confirms.
        """
        target.wait_for(state="visible")
        if self._dry_run:
            self.screenshot(f"before-{what}")
            self.stopped_before = what
            raise DryRunStop(what)
        if len(self.confirms) >= self.max_writes:
            raise WriteRefusedError(f"more than {self.max_writes} state-changing clicks in one UI attempt")
        self.screenshot(f"before-{what}")
        self.confirms.append(what)
        try:
            target.click()
        except BaseException as exc:
            self.confirm_error = exc
            raise
        self.screenshot(f"after-{what}")
