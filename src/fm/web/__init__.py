"""The local dashboard (PLUG_AND_PLAY section 7, ROADMAP #54): ``fm ui`` serves it, and ``fm run`` serves it beside
the loop unless ``--no-ui``.

- :mod:`fm.web.app`: the Starlette app. A JSON API over :mod:`fm.views`, :mod:`fm.health`, the proposal queue, the
  kill switch, the reports and :mod:`fm.config_edit`, one static page, and the guard every request passes (the
  ``Host`` check, the per-process token on every POST, no CORS).
- :mod:`fm.web.server`: one socket bound on :data:`LOOPBACK` and uvicorn serving it, either in the foreground
  (``fm ui``) or in a daemon thread (``fm run``).

The dashboard records human decisions and edits settings. It never writes to ESPN: nothing under ``fm.web`` imports
``fm.executor`` or ``fm.browser`` (``tests/web/test_imports.py`` pins it), so approving here only records the decision,
and ``fm run`` or the scheduled tick executes it.

This module holds constants only, so a command can name the default port without loading Starlette, uvicorn or the
read views (every ``fm`` command module is imported at startup).
"""

from __future__ import annotations

DEFAULT_PORT = 8711
"""Where ``fm ui`` and ``fm run`` serve the dashboard unless ``--port`` says otherwise."""
LOOPBACK = "127.0.0.1"
"""The only address the dashboard binds: it is reachable from this machine only."""
