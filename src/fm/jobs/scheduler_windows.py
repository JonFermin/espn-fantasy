"""Windows Task Scheduler backend for the tick (DESIGN section 13): ``schtasks`` plus a generated ``.cmd`` wrapper.

The job, the wrapper's duties and the spec are shared (:mod:`fm.jobs.scheduler_base`); this backend, ported from
software-factory's ``scheduler/windows.py``, renders them for Windows:

- ``/TR`` is limited to 261 characters and quoting through schtasks is fragile, so the task runs the generated
  ``.cmd`` wrapper (``scripts/espn-fantasy-tick.cmd`` under the config dir).
- The task is created with ``/RL LIMITED`` and without ``/RU``/``/NP``, so it keeps the default "run only when the
  user is logged on" (interactive token): the browser profile and the ESPN session belong to that user.
- ``schtasks /Create`` has no switch for the power conditions and defaults to "do not start on batteries, stop when
  going on batteries", and never wakes the machine. After the ``/Create`` the task's settings are rewritten with
  ``Set-ScheduledTask`` (Windows PowerShell 5.1, always present) to allow both and to wake the PC for a run
  (``-WakeToRun``): the PC must be awake at locks (DESIGN section 13), and the first tick after a missed window
  reports what was missed.

Every ``render_*`` function is pure and :func:`install`, :func:`uninstall` and :func:`show` take an injectable
:class:`Runner`, so tests render the commands and never touch the real scheduler (nothing here is run for real in a
unit test). ``fm schedule install|uninstall|show`` (:mod:`fm.commands.schedule`) drives it through :data:`WINDOWS`.

The logon job for ``fm run`` (DESIGN section 13.3, :data:`WINDOWS_AUTOSTART`, driven by ``fm autostart``) is
registered through PowerShell's ``Register-ScheduledTask`` rather than ``schtasks``, because only the cmdlets expose
an at-logon trigger for one user together with restart-on-failure, an unlimited run time and the power settings:

- The action is ``uvw.exe run --project <repo> --env-file <env> fm run`` with the repo as the working directory:
  ``uvw.exe`` is uv's windowless launcher, so no console window stays open for the lifetime of the loop. A ``.cmd``
  wrapper (what the tick uses) would keep a ``cmd.exe`` window open the whole time, which is why the ``FM_*``
  overrides travel in a generated ``.env`` file (``scripts/espn-fantasy-run.env``) that ``uv run --env-file`` loads
  instead. Without ``uvw.exe`` the task runs ``uv.exe`` and the install says so in a note.
- The task keeps the interactive token (no ``-User``/``-Password`` on the registration), like the tick: the browser
  profile belongs to the logged-on user. ``-MultipleInstances IgnoreNew`` keeps a second copy from starting, the
  run lock (:mod:`fm.jobs.runlock`) covers the ``fm tick`` task, which stays installed as the wake timer.
- An at-logon trigger first fires at the next logon, so the install starts the task at once with
  ``Start-ScheduledTask`` (a failure there is a note: the task is registered either way), as launchd's ``RunAtLoad``
  does on macOS. ``fm autostart install --no-start`` registers only (:data:`WINDOWS_AUTOSTART_NO_START`).
"""

from __future__ import annotations

import dataclasses
import functools
import re
import subprocess
from pathlib import Path

from fm.jobs.scheduler_base import (
    DEFAULT_INTERVAL_MINUTES,
    LOG_FILENAME,
    MAX_INTERVAL_MINUTES,
    REPO_ROOT,
    TASK_NAME,
    WAKE_TIMER_NOTE,
    AutostartBackend,
    AutostartSpec,
    Backend,
    Runner,
    RunResult,
    ScheduleReport,
    SchedulerError,
    ScheduleSpec,
    default_runner,
    find_uv,
)

TASKRUN_MAX_LEN = 261
"""The documented ``schtasks /TR`` limit, the reason the wrapper exists."""
WRAPPER_SUFFIX = ".cmd"
POWERSHELL = "powershell"
"""Windows PowerShell 5.1 ships with Windows; ``pwsh`` may not be installed."""
EXECUTION_TIME_LIMIT_HOURS = 2
"""A tick that runs this long is stuck (an execution is a few reads, one write and a few re-reads)."""
NOT_FOUND_MARKERS = ("cannot find the file specified", "does not exist", "no such")
"""Substrings (lower-case) that mean "no such task" in schtasks output (English locale)."""
POWER_RESTRICTED_MARKER = "no start on batteries"
"""Appears in ``schtasks /Query /V`` output when a task still has the schtasks default power conditions."""
TASK_MISSING_MARKERS = ("no msft_scheduledtask objects found", *NOT_FOUND_MARKERS)
"""Substrings (lower-case) that mean "no such task" from the ``*-ScheduledTask`` cmdlets (English locale)."""
RUN_RESTART_COUNT = 99
"""How many times Task Scheduler restarts a crashed ``fm run`` (one attempt per :data:`RUN_RESTART_MINUTES`)."""
RUN_RESTART_MINUTES = 1
NO_TIME_LIMIT = "PT0S"
"""``ExecutionTimeLimit`` for a task that may run until logoff; assigned after ``New-ScheduledTaskSettingsSet``
because the cmdlet's ``-ExecutionTimeLimit`` parameter does not take a zero on every Windows build."""
UVW_FALLBACK_NOTE = (
    "uvw.exe not found beside uv.exe (an older uv); the task runs uv.exe, whose console window stays open while fm "
    "run runs. Update uv (uv self update) and re-run fm autostart install"
)

_NEEDS_QUOTES = frozenset(" \t&()^<>|;,=")
"""cmd.exe metacharacters and whitespace that force double quotes around an argument."""
_SET_LINE = re.compile(r'^set "(?P<key>[^=]+)=(?P<value>.*)"$')
_ENV_LINE = re.compile(r"""^(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>.*)$""")


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


def install(spec: ScheduleSpec, runner: Runner) -> ScheduleReport:
    """Write the wrapper, create the task and relax its power conditions. Raises :class:`SchedulerError` when the
    ``/Create`` fails; a failed power rewrite is a note (the task still runs on AC)."""
    report = ScheduleReport(action="install", task_name=spec.task_name, backend="windows", unit="task")
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
    report = ScheduleReport(action="uninstall", task_name=spec.task_name, backend="windows", unit="task")
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


WINDOWS = Backend(
    name="windows",
    scheduler="Windows Task Scheduler",
    build_spec=functools.partial(ScheduleSpec.build, wrapper_suffix=WRAPPER_SUFFIX),
    plan=render_plan,
    install=install,
    uninstall=uninstall,
    show=show,
)
"""This backend as ``fm schedule`` drives it."""


# --- the logon job: fm run at logon -----------------------------------------------------------------------------------


def dotenv_quote(value: str) -> str:
    """``value`` as a ``.env`` literal ``uv run --env-file`` reads back unchanged: single-quoted (no escape
    processing, which a Windows path full of backslashes needs), or double-quoted with escapes when it holds ``'``."""
    if "'" not in value:
        return f"'{value}'"
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_env_file(spec: AutostartSpec) -> str:
    """Text of the ``.env`` file the logon task loads: the ``FM_*`` overrides, possibly none (an empty file is fine)."""
    lines = [
        "# Generated by `fm autostart install`; do not edit, re-run install instead.",
        *[f"{key}={dotenv_quote(value)}" for key, value in spec.env().items()],
    ]
    return "\n".join(lines) + "\n"


def env_file_env(path: Path) -> dict[str, str]:
    """The ``KEY=value`` lines of a generated env file (what the logon task will see), unquoted."""
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _ENV_LINE.match(line.strip())
        if match is None:
            continue
        value = match["value"]
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            inner = value[1:-1]
            value = inner if value[0] == "'" else inner.replace('\\"', '"').replace("\\\\", "\\")
        env[match["key"]] = value
    return env


def render_autostart_argv(spec: AutostartSpec) -> list[str]:
    """What the task runs: ``<uvw> run --project <repo> --env-file <env> fm run``."""
    uv, *rest = spec.argv()
    return [uv, *rest[:3], "--env-file", str(spec.env_file_path), *rest[3:]]


def render_register_logon_task(spec: AutostartSpec) -> list[str]:
    """``powershell`` argv that registers (or replaces) the logon task: ``fm run`` through the windowless launcher
    from the repo, at this user's logon, restarted when it crashes, with no run-time limit, on battery too, and
    never a second copy."""
    argument = subprocess.list2cmdline(render_autostart_argv(spec)[1:])
    trigger = "New-ScheduledTaskTrigger -AtLogOn" + (f" -User {ps_quote(spec.user)}" if spec.user else "")
    script = (
        f"$action = New-ScheduledTaskAction -Execute {ps_quote(str(spec.uv_path))} -Argument {ps_quote(argument)} "
        f"-WorkingDirectory {ps_quote(str(spec.repo_root))}; "
        f"$trigger = {trigger}; "
        "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
        f"-MultipleInstances IgnoreNew -RestartCount {RUN_RESTART_COUNT} "
        f"-RestartInterval (New-TimeSpan -Minutes {RUN_RESTART_MINUTES}); "
        f"$settings.ExecutionTimeLimit = {ps_quote(NO_TIME_LIMIT)}; "
        f"Register-ScheduledTask -TaskName {ps_quote(spec.task_name)} -Action $action -Trigger $trigger "
        "-Settings $settings -Force -ErrorAction Stop | Out-Null"
    )
    return [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script]


def render_start_task(task_name: str) -> list[str]:
    """``powershell`` argv that starts the registered task now (``Start-ScheduledTask``), as a logon would."""
    script = f"Start-ScheduledTask -TaskName {ps_quote(task_name)} -ErrorAction Stop"
    return [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script]


def render_unregister_task(task_name: str) -> list[str]:
    script = f"Unregister-ScheduledTask -TaskName {ps_quote(task_name)} -Confirm:$false -ErrorAction Stop"
    return [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script]


def render_get_task(task_name: str) -> list[str]:
    """``powershell`` argv that prints the task's state, action and last result as ``Key: value`` lines."""
    script = (
        f"$t = Get-ScheduledTask -TaskName {ps_quote(task_name)} -ErrorAction Stop; "
        "$i = $t | Get-ScheduledTaskInfo; "
        "'TaskName: ' + $t.TaskName; 'State: ' + $t.State; "
        "'Execute: ' + $t.Actions[0].Execute; 'Arguments: ' + $t.Actions[0].Arguments; "
        "'WorkingDirectory: ' + $t.Actions[0].WorkingDirectory; "
        "'LastRunTime: ' + $i.LastRunTime; 'LastTaskResult: ' + $i.LastTaskResult; "
        "'NumberOfMissedRuns: ' + $i.NumberOfMissedRuns"
    )
    return [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script]


def is_task_missing(result: RunResult) -> bool:
    text = result.output.lower()
    return not result.ok and any(marker in text for marker in TASK_MISSING_MARKERS)


def render_autostart_plan(spec: AutostartSpec, *, start: bool = True) -> list[str]:
    """What :func:`install_autostart` would do, as lines, for ``fm autostart install --dry-run``."""
    lines = [f"env file {spec.env_file_path}:"]
    lines.extend(f"  {line}" for line in render_env_file(spec).splitlines())
    lines.append("commands:")
    lines.append("  " + subprocess.list2cmdline(render_register_logon_task(spec)))
    if start:
        lines.append("  " + subprocess.list2cmdline(render_start_task(spec.task_name)))
    if not spec.windowless:
        lines.append(f"note: {UVW_FALLBACK_NOTE}")
    return lines


def install_autostart(spec: AutostartSpec, runner: Runner, *, start: bool = True) -> ScheduleReport:
    """Write the env file, register the logon task (replacing one that exists) and, unless ``start`` is off, start it
    now with ``Start-ScheduledTask``: an at-logon trigger first fires at the next logon, so without the start ``fm run``
    would not run until then (launchd's ``RunAtLoad`` starts the macOS agent at once). Raises :class:`SchedulerError`
    when the registration fails; a start that fails is a note, since the task is installed and the next logon starts
    it. ``-MultipleInstances IgnoreNew`` makes the start a no-op when the task is already running. The tick task is
    left alone: it is the wake timer."""
    report = ScheduleReport(action="install", task_name=spec.task_name, backend="windows", unit="task")
    spec.scripts_dir.mkdir(parents=True, exist_ok=True)
    spec.logs_dir.mkdir(parents=True, exist_ok=True)
    env_file = spec.env_file_path
    env_file.write_text(render_env_file(spec), encoding="utf-8", newline="\n")
    report.files.append(env_file)
    argv = render_register_logon_task(spec)
    report.commands.append(argv)
    result = runner(argv)
    if not result.ok:
        raise SchedulerError(f"Register-ScheduledTask {spec.task_name} failed ({result.returncode}): {result.output}")
    report.installed = True
    if not spec.windowless:
        report.notes.append(UVW_FALLBACK_NOTE)
    report.notes.append(WAKE_TIMER_NOTE)
    if start:
        report.notes.append(_start_now(spec.task_name, runner, report))
    else:
        report.notes.append(f"{spec.task_name}: not started now (--no-start); it starts at the next logon")
    return report


def _start_now(task_name: str, runner: Runner, report: ScheduleReport) -> str:
    """Run ``Start-ScheduledTask`` and say how it went; never raises."""
    argv = render_start_task(task_name)
    report.commands.append(argv)
    later = f"it starts at the next logon, or now with: Start-ScheduledTask -TaskName {task_name}"
    try:
        result = runner(argv)
    except SchedulerError as exc:
        return f"{task_name}: registered but not started now ({exc}); {later}"
    if not result.ok:
        return f"{task_name}: registered but not started now ({result.output or f'exit {result.returncode}'}); {later}"
    return f"{task_name}: started now, so fm run is running (Start-ScheduledTask)"


def uninstall_autostart(spec: AutostartSpec, runner: Runner) -> ScheduleReport:
    """Unregister the logon task (a missing task is not an error) and remove the env file."""
    report = ScheduleReport(action="uninstall", task_name=spec.task_name, backend="windows", unit="task")
    argv = render_unregister_task(spec.task_name)
    report.commands.append(argv)
    result = runner(argv)
    if result.ok:
        report.installed = False
    elif is_task_missing(result):
        report.notes.append(f"{spec.task_name}: not installed")
    else:
        raise SchedulerError(f"Unregister-ScheduledTask {spec.task_name} failed ({result.returncode}): {result.output}")
    env_file = spec.env_file_path
    if env_file.exists():
        env_file.unlink()
        report.files.append(env_file)
    return report


def show_autostart(spec: AutostartSpec, runner: Runner) -> str:
    """The task as ``Get-ScheduledTask`` / ``Get-ScheduledTaskInfo`` report it, plus the env file. Never raises: a
    missing task or a missing PowerShell is reported in the text, so ``fm autostart show`` always answers."""
    try:
        result = runner(render_get_task(spec.task_name))
    except SchedulerError as exc:
        return f"{spec.task_name}: cannot query the scheduler ({exc})\n{_env_file_lines(spec)}"
    if result.ok:
        return result.stdout.strip() + "\n" + _env_file_lines(spec)
    if is_task_missing(result):
        return f"{spec.task_name}: not installed (fm autostart install)\n{_env_file_lines(spec)}"
    return f"{spec.task_name}: error ({result.returncode}): {result.output}"


def _env_file_lines(spec: AutostartSpec) -> str:
    env_file = spec.env_file_path
    if env_file.is_file():
        env = ", ".join(f"{key}={value}" for key, value in env_file_env(env_file).items()) or "(none)"
        return f"Env file:                             {env_file}\nEnv:                                  {env}"
    return f"Env file:                             {env_file} (not written; fm autostart install writes it)"


WINDOWS_AUTOSTART = AutostartBackend(
    name="windows",
    scheduler="Windows Task Scheduler",
    build_spec=functools.partial(AutostartSpec.build, prefer_windowless=True),
    plan=render_autostart_plan,
    install=install_autostart,
    uninstall=uninstall_autostart,
    show=show_autostart,
)
"""The logon job as ``fm autostart`` drives it: registered, then started at once."""

WINDOWS_AUTOSTART_NO_START = dataclasses.replace(
    WINDOWS_AUTOSTART,
    plan=functools.partial(render_autostart_plan, start=False),
    install=functools.partial(install_autostart, start=False),
)
"""``fm autostart install --no-start``: registered only; ``fm run`` first starts at the next logon."""


__all__ = [
    "DEFAULT_INTERVAL_MINUTES",
    "EXECUTION_TIME_LIMIT_HOURS",
    "LOG_FILENAME",
    "MAX_INTERVAL_MINUTES",
    "NOT_FOUND_MARKERS",
    "NO_TIME_LIMIT",
    "POWERSHELL",
    "POWER_RESTRICTED_MARKER",
    "REPO_ROOT",
    "RUN_RESTART_COUNT",
    "RUN_RESTART_MINUTES",
    "TASKRUN_MAX_LEN",
    "TASK_MISSING_MARKERS",
    "TASK_NAME",
    "UVW_FALLBACK_NOTE",
    "WAKE_TIMER_NOTE",
    "WINDOWS",
    "WINDOWS_AUTOSTART",
    "WINDOWS_AUTOSTART_NO_START",
    "WRAPPER_SUFFIX",
    "AutostartBackend",
    "AutostartSpec",
    "Backend",
    "RunResult",
    "Runner",
    "ScheduleReport",
    "ScheduleSpec",
    "SchedulerError",
    "cmd_quote",
    "default_runner",
    "dotenv_quote",
    "env_file_env",
    "find_uv",
    "install",
    "install_autostart",
    "is_not_found",
    "is_task_missing",
    "ps_quote",
    "render_autostart_argv",
    "render_autostart_plan",
    "render_env_file",
    "render_get_task",
    "render_plan",
    "render_power_settings",
    "render_register_logon_task",
    "render_schtasks_create",
    "render_schtasks_delete",
    "render_schtasks_query",
    "render_start_task",
    "render_unregister_task",
    "render_wrapper_cmd",
    "show",
    "show_autostart",
    "uninstall",
    "uninstall_autostart",
    "wrapper_env",
]
