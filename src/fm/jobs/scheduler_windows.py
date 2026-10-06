"""Windows Task Scheduler backend for the tick (DESIGN section 13): ``schtasks`` plus a generated ``.cmd`` wrapper.

One scheduled job, ``fm tick`` every :data:`DEFAULT_INTERVAL_MINUTES` minutes, ported from software-factory's
``scheduler/windows.py``:

- ``/TR`` is limited to 261 characters and quoting through schtasks is fragile, so the task runs a generated ``.cmd``
  wrapper under the config dir (``scripts/espn-fantasy-tick.cmd``) that ``cd``s to the repo, sets ``FM_CONFIG_DIR`` /
  ``FM_CACHE_DIR`` when they were set in the shell that installed it (scheduled jobs inherit no shell environment, and
  without them a tick would use another state.db than the interactive CLI), runs ``<uv> run --project <repo> fm tick``
  and appends stdout and stderr to ``logs/tick.log``.
- The task is created with ``/RL LIMITED`` and without ``/RU``/``/NP``, so it keeps the default "run only when the
  user is logged on" (interactive token): the browser profile and the ESPN session belong to that user.
- ``schtasks /Create`` has no switch for the power conditions and defaults to "do not start on batteries, stop when
  going on batteries", and never wakes the machine. After the ``/Create`` the task's settings are rewritten with
  ``Set-ScheduledTask`` (Windows PowerShell 5.1, always present) to allow both and to wake the PC for a run
  (``-WakeToRun``): the PC must be awake at locks (DESIGN section 13), and the first tick after a missed window
  reports what was missed.

Every ``render_*`` function is pure and :func:`install`, :func:`uninstall` and :func:`show` take an injectable
:class:`Runner`, so tests render the commands and never touch the real scheduler (nothing here is run for real in a
unit test). ``fm schedule install|uninstall|show`` (:mod:`fm.commands.schedule`) is the CLI over it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from fm import paths

TASK_NAME = "espn-fantasy-tick"
"""The Windows task name (``schtasks /TN``) and the wrapper's file stem."""
DEFAULT_INTERVAL_MINUTES = 10
"""How often the tick runs (DESIGN section 13: one cheap, idempotent tick every 10 minutes)."""
LOG_FILENAME = "tick.log"
SCRIPTS_DIRNAME = "scripts"
LOGS_DIRNAME = "logs"
TASKRUN_MAX_LEN = 261
"""The documented ``schtasks /TR`` limit, the reason the wrapper exists."""
WRAPPER_SUFFIX = ".cmd"
POWERSHELL = "powershell"
"""Windows PowerShell 5.1 ships with Windows; ``pwsh`` may not be installed."""
EXECUTION_TIME_LIMIT_HOURS = 2
"""A tick that runs this long is stuck (an execution is a few reads, one write and a few re-reads)."""
MAX_INTERVAL_MINUTES = 1439
"""``schtasks /SC MINUTE /MO`` accepts 1..1439."""
NOT_FOUND_MARKERS = ("cannot find the file specified", "does not exist", "no such")
"""Substrings (lower-case) that mean "no such task" in schtasks output (English locale)."""
POWER_RESTRICTED_MARKER = "no start on batteries"
"""Appears in ``schtasks /Query /V`` output when a task still has the schtasks default power conditions."""
REPO_ROOT = Path(__file__).resolve().parents[3]
"""The checkout ``uv run --project`` points at (``src/fm/jobs/scheduler_windows.py`` -> the repo)."""

_NEEDS_QUOTES = frozenset(" \t&()^<>|;,=")
"""cmd.exe metacharacters and whitespace that force double quotes around an argument."""
_SET_LINE = re.compile(r'^set "(?P<key>[^=]+)=(?P<value>.*)"$')


class SchedulerError(RuntimeError):
    """The scheduler could not be driven: ``schtasks`` failed, is missing, or the spec cannot be rendered."""


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
    """Runs one scheduler command (``schtasks`` or ``powershell`` argv). Tests pass a fake that records argv."""

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
    """Everything the backend needs, resolved once at install time (the uv path embedded absolute).

    ``config_dir`` and ``cache_dir`` are embedded only when ``FM_CONFIG_DIR`` / ``FM_CACHE_DIR`` were set in the shell
    that installed the task, so the scheduled tick uses the same state as that shell's CLI; unset, the tick uses the
    platform defaults (``fm.paths``), like the CLI.
    """

    repo_root: Path
    uv_path: Path
    scripts_dir: Path
    logs_dir: Path
    config_dir: Path | None = None
    cache_dir: Path | None = None
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES
    task_name: str = TASK_NAME

    def __post_init__(self) -> None:
        if not 1 <= self.interval_minutes <= MAX_INTERVAL_MINUTES:
            raise SchedulerError(f"interval_minutes must be 1..{MAX_INTERVAL_MINUTES}, got {self.interval_minutes}")
        if not self.task_name or any(ch in self.task_name for ch in '\\/:*?"<>|'):
            raise SchedulerError(f"task name {self.task_name!r} is not a valid schtasks task name")

    @classmethod
    def build(
        cls,
        *,
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
        uv_path: Path | None = None,
        repo_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
        require_uv: bool = True,
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
        )

    @property
    def wrapper_path(self) -> Path:
        return self.scripts_dir / f"{self.task_name}{WRAPPER_SUFFIX}"

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


# --- rendering --------------------------------------------------------------------------------------------------------


def cmd_quote(arg: str) -> str:
    """Quote one argument for a ``.cmd`` line. Windows paths cannot contain ``"`` so there is no escaping."""
    if arg == "" or any(ch in _NEEDS_QUOTES for ch in arg):
        return f'"{arg}"'
    return arg


def ps_quote(text: str) -> str:
    """A PowerShell single-quoted literal (only ``'`` needs doubling)."""
    return "'" + text.replace("'", "''") + "'"


def render_wrapper_cmd(spec: ScheduleSpec) -> str:
    """Text of the ``.cmd`` wrapper (CRLF line endings): cd to the repo, set env, run the tick, append to the log."""
    log = cmd_quote(str(spec.log_path))
    log_dir = cmd_quote(str(spec.logs_dir))
    lines = [
        "@echo off",
        "rem Generated by `fm schedule install`; do not edit, re-run install instead.",
        f"cd /d {cmd_quote(str(spec.repo_root))}",
        *[f'set "{key}={value}"' for key, value in spec.env().items()],
        f"if not exist {log_dir} mkdir {log_dir}",
        f"echo [%DATE% %TIME%] tick start>> {log}",
        f"{' '.join(cmd_quote(arg) for arg in spec.argv())} >> {log} 2>&1",
        "exit /b %ERRORLEVEL%",
    ]
    return "\r\n".join(lines) + "\r\n"


def wrapper_env(path: Path) -> dict[str, str]:
    """The ``set "KEY=VALUE"`` lines of a generated wrapper (what the scheduled tick will see)."""
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _SET_LINE.match(line.strip())
        if match:
            env[match["key"]] = match["value"]
    return env


def _taskrun(wrapper: Path) -> str:
    # Embedded quotes survive subprocess's list2cmdline, so the stored action is the quoted path.
    taskrun = f'"{wrapper}"'
    if len(taskrun) > TASKRUN_MAX_LEN:
        raise SchedulerError(f"wrapper path exceeds the schtasks /TR limit ({TASKRUN_MAX_LEN}): {wrapper}")
    return taskrun


def render_schtasks_create(spec: ScheduleSpec) -> list[str]:
    """``schtasks /Create`` argv: the tick every ``interval_minutes`` minutes, interactive token, replace if present."""
    return [
        "schtasks", "/Create",
        "/TN", spec.task_name,
        "/SC", "MINUTE", "/MO", str(spec.interval_minutes),
        "/TR", _taskrun(spec.wrapper_path),
        "/RL", "LIMITED",
        "/F",
    ]  # fmt: skip


def render_power_settings(task_name: str) -> list[str]:
    """``powershell`` argv that lets the task start on battery, keeps it running when the machine switches to
    battery, wakes the PC to run it, skips a run while one is still going, and caps a runaway run."""
    command = (
        f"Set-ScheduledTask -TaskName {ps_quote(task_name)} -Settings (New-ScheduledTaskSettingsSet "
        "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun -MultipleInstances IgnoreNew "
        f"-ExecutionTimeLimit (New-TimeSpan -Hours {EXECUTION_TIME_LIMIT_HOURS}))"
    )
    return [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command]


def render_schtasks_delete(task_name: str) -> list[str]:
    return ["schtasks", "/Delete", "/TN", task_name, "/F"]


def render_schtasks_query(task_name: str) -> list[str]:
    return ["schtasks", "/Query", "/TN", task_name, "/FO", "LIST", "/V"]


def is_not_found(result: RunResult) -> bool:
    text = result.output.lower()
    return not result.ok and any(marker in text for marker in NOT_FOUND_MARKERS)


def render_plan(spec: ScheduleSpec) -> list[str]:
    """What :func:`install` would do, as lines: the wrapper and the commands, for ``fm schedule install --dry-run``."""
    lines = [f"wrapper {spec.wrapper_path}:"]
    lines.extend(f"  {line}" for line in render_wrapper_cmd(spec).splitlines())
    lines.append("commands:")
    for argv in (render_schtasks_create(spec), render_power_settings(spec.task_name)):
        lines.append("  " + subprocess.list2cmdline(argv))
    return lines


# --- actions ----------------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class ScheduleReport:
    """What :func:`install` / :func:`uninstall` did: the task touched, files written or removed, argv run, notes."""

    action: str
    task_name: str
    files: list[Path] = field(default_factory=list[Path])
    commands: list[list[str]] = field(default_factory=list[list[str]])
    notes: list[str] = field(default_factory=list[str])
    installed: bool = False

    def lines(self) -> list[str]:
        lines = [f"{self.action} (windows) task {self.task_name}"]
        lines += [f"  file  {path}" for path in self.files]
        lines += [f"  ran   {subprocess.list2cmdline(argv)}" for argv in self.commands]
        lines += [f"  note  {note}" for note in self.notes]
        return lines


def install(spec: ScheduleSpec, runner: Runner) -> ScheduleReport:
    """Write the wrapper, create the task and relax its power conditions. Raises :class:`SchedulerError` when the
    ``/Create`` fails; a failed power rewrite is a note (the task still runs on AC)."""
    report = ScheduleReport(action="install", task_name=spec.task_name)
    spec.scripts_dir.mkdir(parents=True, exist_ok=True)
    spec.logs_dir.mkdir(parents=True, exist_ok=True)
    wrapper = spec.wrapper_path
    wrapper.write_text(render_wrapper_cmd(spec), encoding="utf-8", newline="")  # keep CRLF as rendered
    report.files.append(wrapper)
    argv = render_schtasks_create(spec)
    report.commands.append(argv)
    result = runner(argv)
    if not result.ok:
        raise SchedulerError(f"schtasks /Create {spec.task_name} failed ({result.returncode}): {result.output}")
    report.installed = True
    _allow_wake_and_battery(spec.task_name, runner, report)
    return report


def _allow_wake_and_battery(task_name: str, runner: Runner, report: ScheduleReport) -> None:
    argv = render_power_settings(task_name)
    report.commands.append(argv)
    try:
        result = runner(argv)
        problem = "" if result.ok else (result.output.strip().splitlines() or [f"exit {result.returncode}"])[0]
    except SchedulerError as exc:
        problem = str(exc)
    if problem:
        report.notes.append(
            f"{task_name}: could not allow waking and running on battery ({problem}); the task keeps the schtasks "
            "default (no start on battery, stop when going on battery, no wake), so a lock during sleep is missed"
        )


def uninstall(spec: ScheduleSpec, runner: Runner) -> ScheduleReport:
    """Delete the task (a missing task is not an error) and remove the wrapper."""
    report = ScheduleReport(action="uninstall", task_name=spec.task_name)
    argv = render_schtasks_delete(spec.task_name)
    report.commands.append(argv)
    result = runner(argv)
    if result.ok:
        report.installed = False
    elif is_not_found(result):
        report.notes.append(f"{spec.task_name}: not installed")
    else:
        raise SchedulerError(f"schtasks /Delete {spec.task_name} failed ({result.returncode}): {result.output}")
    wrapper = spec.wrapper_path
    if wrapper.exists():
        wrapper.unlink()
        report.files.append(wrapper)
    return report


def show(spec: ScheduleSpec, runner: Runner) -> str:
    """The ``schtasks /Query /V`` listing for the task plus what the wrapper under the current config dir sets (the
    query shows the wrapper path but not what it exports). Never raises: a missing task or a missing ``schtasks`` is
    reported in the text, so ``fm schedule show`` always answers."""
    try:
        result = runner(render_schtasks_query(spec.task_name))
    except SchedulerError as exc:
        return f"{spec.task_name}: cannot query the scheduler ({exc})\n{_wrapper_lines(spec)}"
    if result.ok:
        text = result.stdout.strip() + "\n" + _wrapper_lines(spec)
        if POWER_RESTRICTED_MARKER in result.stdout.lower():
            text += (
                "\nNote: the task keeps the schtasks default power conditions (no start on battery); re-run "
                "fm schedule install"
            )
        return text
    if is_not_found(result):
        return f"{spec.task_name}: not installed (fm schedule install)\n{_wrapper_lines(spec)}"
    return f"{spec.task_name}: error ({result.returncode}): {result.output}"


def _wrapper_lines(spec: ScheduleSpec) -> str:
    wrapper = spec.wrapper_path
    if wrapper.is_file():
        env = ", ".join(f"{key}={value}" for key, value in wrapper_env(wrapper).items()) or "(none)"
        return f"Wrapper:                              {wrapper}\nWrapper Env:                          {env}"
    return f"Wrapper:                              {wrapper} (not written; fm schedule install writes it)"


__all__ = [
    "DEFAULT_INTERVAL_MINUTES",
    "EXECUTION_TIME_LIMIT_HOURS",
    "LOG_FILENAME",
    "MAX_INTERVAL_MINUTES",
    "NOT_FOUND_MARKERS",
    "POWERSHELL",
    "POWER_RESTRICTED_MARKER",
    "REPO_ROOT",
    "TASKRUN_MAX_LEN",
    "TASK_NAME",
    "WRAPPER_SUFFIX",
    "RunResult",
    "Runner",
    "ScheduleReport",
    "ScheduleSpec",
    "SchedulerError",
    "cmd_quote",
    "default_runner",
    "find_uv",
    "install",
    "is_not_found",
    "ps_quote",
    "render_plan",
    "render_power_settings",
    "render_schtasks_create",
    "render_schtasks_delete",
    "render_schtasks_query",
    "render_wrapper_cmd",
    "show",
    "uninstall",
    "wrapper_env",
]
