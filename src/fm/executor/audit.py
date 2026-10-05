"""Audit trail for executions (DESIGN 6.3): one folder per run under ``fm.paths.audit_dir()``.

Layout: ``audit/<YYYY-MM-DD>/p<proposal>-<HHMMSSffffff>[-dry]/`` (UTC) holding ``preconditions.json`` and, per attempt,
``<mode>-request.json``, ``<mode>-response.json`` and ``<mode>-verification.json``, plus UI screenshots
(``ui-NN-<step>.png``) and the Playwright trace (``ui-trace.zip``). ``executions.artifacts`` lists these files by path
relative to the audit root.

The folder lives in the config dir and never in the repo: a Playwright trace records the browser's network traffic,
session cookies included. Nothing else written here carries a cookie or the execution token: a saved request masks
credential headers (``WriteRequest.to_json``; ``fm.executor.transport.check_write_request`` refuses them anyway) and
the token stays on the proposal row.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fm import paths
from fm.browser.flows import jsonable

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    """A file name made of letters, digits, ``.``, ``_`` and ``-`` only."""
    return _UNSAFE.sub("-", name).strip(".-") or "artifact"


class AuditLog:
    """One run's audit folder. Paths it hands out are inside the folder; :meth:`relative` turns them into the form
    ``executions.artifacts`` stores."""

    def __init__(self, folder: Path, root: Path) -> None:
        self.folder = folder
        self.root = root

    @classmethod
    def create(cls, proposal_id: int, at: datetime, *, dry_run: bool = False, root: Path | None = None) -> AuditLog:
        """Make a new folder for a run of proposal ``proposal_id`` started at ``at``; never reuses an existing one."""
        base = root if root is not None else paths.audit_dir()
        stamp = at.astimezone(UTC)
        day = base / stamp.strftime("%Y-%m-%d")
        stem = f"p{proposal_id}-{stamp:%H%M%S%f}" + ("-dry" if dry_run else "")
        folder = day / stem
        counter = 1
        while folder.exists():
            folder = day / f"{stem}-{counter}"
            counter += 1
        folder.mkdir(parents=True)
        return cls(folder, base)

    def path(self, name: str) -> Path:
        """Where a file called ``name`` goes in this run's folder."""
        return self.folder / safe_name(name)

    def relative(self, path: Path) -> str:
        """``path`` relative to the audit root, with forward slashes."""
        return path.relative_to(self.root).as_posix()

    def write_json(self, name: str, data: Any) -> str:
        """Save ``data`` as indented JSON and return its path relative to the audit root."""
        target = self.path(name)
        text = json.dumps(jsonable(data), indent=1, sort_keys=True, ensure_ascii=False)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, target)
        return self.relative(target)
