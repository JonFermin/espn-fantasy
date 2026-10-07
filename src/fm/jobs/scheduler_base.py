"""What the tick's scheduler backends share (DESIGN section 13): the spec, the report, the command runner.

One scheduled job, ``fm tick`` every :data:`DEFAULT_INTERVAL_MINUTES` minutes, run by a generated wrapper under the
config dir (``scripts/espn-fantasy-tick.<suffix>``) that ``cd``s to the repo, sets ``FM_CONFIG_DIR`` /
``FM_CACHE_DIR`` when they were set in the shell that installed it (scheduled jobs inherit no shell environment, and
without them a tick would use another state.db than the interactive CLI), runs ``<uv> run --project <repo> fm tick``
and appends stdout and stderr to ``logs/tick.log``.

Each platform renders that wrapper and registers it its own way: :mod:`fm.jobs.scheduler_windows` (Task Scheduler,
``schtasks`` plus a ``.cmd``) and :mod:`fm.jobs.scheduler_macos` (a launchd LaunchAgent plus a ``.sh``). Each exposes
a :class:`Backend`, and ``fm schedule`` (:mod:`fm.commands.schedule`) picks one by ``sys.platform``. Every backend
action takes an injectable :class:`Runner`, so tests render the commands and never touch a real scheduler.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from fm import paths

TASK_NAME = "espn-fantasy-tick"
"""The scheduled job's name (the Windows task name, the launchd label's stem) and the wrapper's file stem."""
DEFAULT_INTERVAL_MINUTES = 10
"""How often the tick runs (DESIGN section 13: one cheap, idempotent tick every 10 minutes)."""
MAX_INTERVAL_MINUTES = 1439
"""``schtasks /SC MINUTE /MO`` accepts 1..1439; launchd takes the same range so ``--every`` means one thing."""
LOG_FILENAME = "tick.log"
SCRIPTS_DIRNAME = "scripts"
LOGS_DIRNAME = "logs"
REPO_ROOT = Path(__file__).resolve().parents[3]
"""The checkout ``uv run --project`` points at (``src/fm/jobs/scheduler_base.py`` -> the repo)."""

_BAD_NAME_CHARS = frozenset('\\/:*?"<>|')


class SchedulerError(RuntimeError):
    """The scheduler could not be driven: its command failed, is missing, or the spec cannot be rendered."""


# --- runner -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunResult:
    """What one scheduler command answered."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        return (self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr).strip()


class Runner(Protocol):
    """Runs one scheduler command (``schtasks``, ``powershell``, ``launchctl`` argv). Tests pass a fake."""

    def __call__(self, argv: Sequence[str]) -> RunResult: ...


def default_runner(argv: Sequence[str], *, timeout: float = 60.0) -> RunResult:
    """Run a scheduler command for real. Never ``shell=True``; raises :class:`SchedulerError` when the executable is
    missing or the command hangs."""
    args = [str(arg) for arg in argv]
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SchedulerError(f"{args[0]} not found on PATH: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SchedulerError(f"{args[0]} timed out after {timeout:.0f}s") from exc
    return RunResult(argv=tuple(args), returncode=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")


# --- spec -------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScheduleSpec:
    """Everything a backend needs, resolved once at install time (the uv path embedded absolute).

    ``config_dir`` and ``cache_dir`` are embedded only when ``FM_CONFIG_DIR`` / ``FM_CACHE_DIR`` were set in the shell
    that installed the job, so the scheduled tick uses the same state as that shell's CLI; unset, the tick uses the
    platform defaults (``fm.paths``), like the CLI. ``wrapper_suffix`` is the backend's script type.
    """

    repo_root: Path
    uv_path: Path
    scripts_dir: Path
    logs_dir: Path
    config_dir: Path | None = None
    cache_dir: Path | None = None
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES
    task_name: str = TASK_NAME
    wrapper_suffix: str = ".cmd"

    def __post_init__(self) -> None:
        if not 1 <= self.interval_minutes <= MAX_INTERVAL_MINUTES:
            raise SchedulerError(f"interval_minutes must be 1..{MAX_INTERVAL_MINUTES}, got {self.interval_minutes}")
        if not self.task_name or any(ch in _BAD_NAME_CHARS for ch in self.task_name):
            raise SchedulerError(f"task name {self.task_name!r} is not a valid scheduled task name")

    @classmethod
    def build(
        cls,
        *,
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
        uv_path: Path | None = None,
        repo_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
        require_uv: bool = True,
        wrapper_suffix: str = ".cmd",
    ) -> ScheduleSpec:
        """The spec for this checkout and the current environment.

        ``uv`` is looked up on PATH (``require_uv`` false lets ``uninstall`` and ``show`` work without it). The scripts
        and logs dirs sit under the config dir in force (:func:`fm.paths.config_dir`).
        """
        resolved_uv = uv_path if uv_path is not None else find_uv()
        if resolved_uv is None:
            if require_uv:
                raise SchedulerError("uv not found on PATH; install uv or pass uv_path explicitly")
            resolved_uv = Path("uv")
        env = os.environ if environ is None else environ
        config_dir = paths.config_dir()
        return cls(
            repo_root=repo_root if repo_root is not None else REPO_ROOT,
            uv_path=resolved_uv,
            scripts_dir=config_dir / SCRIPTS_DIRNAME,
            logs_dir=config_dir / LOGS_DIRNAME,
            config_dir=_env_path(env, paths.CONFIG_DIR_ENV),
            cache_dir=_env_path(env, paths.CACHE_DIR_ENV),
            interval_minutes=interval_minutes,
            wrapper_suffix=wrapper_suffix,
        )

    @property
    def wrapper_path(self) -> Path:
        return self.scripts_dir / f"{self.task_name}{self.wrapper_suffix}"

    @property
    def log_path(self) -> Path:
        return self.logs_dir / LOG_FILENAME

    def argv(self) -> list[str]:
        """What the wrapper runs: ``<uv> run --project <repo> fm tick``."""
        return [str(self.uv_path), "run", "--project", str(self.repo_root), "fm", "tick"]

    def env(self) -> dict[str, str]:
        """The environment the wrapper sets (scheduled jobs inherit none from the shell)."""
        env: dict[str, str] = {}
        if self.config_dir is not None:
            env[paths.CONFIG_DIR_ENV] = str(self.config_dir)
        if self.cache_dir is not None:
            env[paths.CACHE_DIR_ENV] = str(self.cache_dir)
        return env


def find_uv() -> Path | None:
    found = shutil.which("uv")
    return Path(found).resolve() if found else None


def _env_path(environ: Mapping[str, str], name: str) -> Path | None:
    value = environ.get(name)
    if not value:
        return None
    path = Path(os.path.expandvars(value)).expanduser()
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


# --- report and backend -----------------------------------------------------------------------------------------------


@dataclass(slots=True)
class ScheduleReport:
    """What an install / uninstall did: the job touched, files written or removed, argv run, notes."""

    action: str
    task_name: str
    backend: str = "windows"
    unit: str = "task"
    files: list[Path] = field(default_factory=list[Path])
    commands: list[list[str]] = field(default_factory=list[list[str]])
    notes: list[str] = field(default_factory=list[str])
    installed: bool = False

    def lines(self) -> list[str]:
        lines = [f"{self.action} ({self.backend}) {self.unit} {self.task_name}"]
        lines += [f"  file  {path}" for path in self.files]
        lines += [f"  ran   {subprocess.list2cmdline(argv)}" for argv in self.commands]
        lines += [f"  note  {note}" for note in self.notes]
        return lines


@dataclass(frozen=True, slots=True)
class Backend:
    """One platform's scheduler, as ``fm schedule`` drives it.

    ``build_spec`` takes :meth:`ScheduleSpec.build`'s keywords; ``plan`` renders what ``install`` would do without
    running anything; ``show`` never raises.
    """

    name: str
    scheduler: str
    build_spec: Callable[..., ScheduleSpec]
    plan: Callable[[ScheduleSpec], list[str]]
    install: Callable[[ScheduleSpec, Runner], ScheduleReport]
    uninstall: Callable[[ScheduleSpec, Runner], ScheduleReport]
    show: Callable[[ScheduleSpec, Runner], str]


__all__ = [
    "DEFAULT_INTERVAL_MINUTES",
    "LOGS_DIRNAME",
    "LOG_FILENAME",
    "MAX_INTERVAL_MINUTES",
    "REPO_ROOT",
    "SCRIPTS_DIRNAME",
    "TASK_NAME",
    "Backend",
    "RunResult",
    "Runner",
    "ScheduleReport",
    "ScheduleSpec",
    "SchedulerError",
    "default_runner",
    "find_uv",
]
