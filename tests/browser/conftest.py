"""Browser tests never start a browser.

Every test in this directory drives fakes: a fake launcher handed to ``launch_persistent``, a fake context for the
cookie jar, or a stubbed ``open_browser``. This autouse guard turns a launch that slipped past a monkeypatch into a
test failure instead of a headed Edge opening on the temp profile and visiting ESPN. Tests that exercise
``open_browser`` itself replace ``sync_playwright`` again with their own fake, which wins because it is applied later.
"""

from __future__ import annotations

from typing import NoReturn

import pytest

from fm.browser import session as browser_session


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missed monkeypatch must fail the test, not start a real browser."""

    def refuse() -> NoReturn:
        raise AssertionError("real browser launch in a unit test; stub open_browser or sync_playwright instead")

    monkeypatch.setattr(browser_session, "sync_playwright", refuse)
