"""``fm autostart`` and the logon-job backends: rendered PowerShell and plist, fake runners, nothing registered.

Every scheduler command goes to a fake runner that records argv and answers like Task Scheduler or launchd would; an
autouse guard turns a real ``subprocess.run`` into a test failure. ``uv`` is pinned to a fake ``uv.exe`` (with a
``uvw.exe`` beside it unless a test removes it), and the uid and ``~/Library/LaunchAgents`` are pinned too, so both
backends render on any platform. Files land under the per-test config dir (conftest) and a temp LaunchAgents dir.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import pytest
import typer
from typer.testing import CliRunner

from fm import paths
from fm.commands import autostart as autostart_cmd
from fm.jobs import scheduler_base
from fm.jobs import scheduler_macos as macos
from fm.jobs.runlock import RunLock
from fm.jobs.scheduler_base import (
    AUTOSTART_TASK_NAME,
    WAKE_TIMER_NOTE,
    AutostartBackend,
    AutostartSpec,
    RunResult,
    SchedulerError,
    current_user,
)
from fm.jobs.scheduler_macos import (
    MACOS_AUTOSTART,
    install_autostart_agent,
    render_autostart_agent_plan,
    render_run_plist,
    show_autostart_agent,
    uninstall_autostart_agent,
)
from fm.jobs.scheduler_windows import (
    NO_TIME_LIMIT,
    POWERSHELL,
    RUN_RESTART_COUNT,
    RUN_RESTART_MINUTES,
    UVW_FALLBACK_NOTE,
    WINDOWS_AUTOSTART,
    WINDOWS_AUTOSTART_NO_START,
    dotenv_quote,
    env_file_env,
    install_autostart,
    render_autostart_argv,
    render_autostart_plan,
    render_env_file,
    render_get_task,
    render_register_logon_task,
    render_start_task,
    render_unregister_task,
    show_autostart,
    uninstall_autostart,
)

REPO = Path("C:/Users/someone/DEVELOP/espn fantasy")
MAC_UV = Path("/Users/someone/.local/bin/uv")
UID = 501
LABEL = f"local.{AUTOSTART_TASK_NAME}"
TASK_MISSING = "Get-ScheduledTask : No MSFT_ScheduledTask objects found with property 'TaskName' equal to 'x'."
LAUNCHD_MISSING = f'Could not find service "{LABEL}" in domain for user gui: {UID}'
USER_ENV = {"USERNAME": "someone", "USERDOMAIN": "PC"}

runner = CliRunner()


@pytest.fixture(autouse=True)
def pinned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    """No real command, a fake uv (and uvw) on disk, a pinned uid and LaunchAgents dir."""

    def refuse(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError(f"a real command ran in a unit test: {args[0]!r}")

    bin_dir = tmp_path / "uv bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv.exe"
    uvw = bin_dir / "uvw.exe"
    uv.write_bytes(b"")
    uvw.write_bytes(b"")
    agents = tmp_path / "LaunchAgents"
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(scheduler_base, "find_uv", lambda: uv)
    monkeypatch.setattr(macos, "current_uid", lambda: UID)
    monkeypatch.setattr(macos, "launch_agents_dir", lambda: agents)
    return {"uv": uv, "uvw": uvw, "agents": agents}


class FakePowerShell:
    """Answers the logon task's ``Register-`` / ``Start-`` / ``Unregister-`` / ``Get-ScheduledTask`` scripts; records
    every argv. ``started`` says whether ``Start-ScheduledTask`` ran the task."""

    def __init__(self, *, installed: bool = False, register_ok: bool = True, start_ok: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.verbs: list[str] = []
        self.installed = installed
        self.register_ok = register_ok
        self.start_ok = start_ok
        self.started = False
        self.raise_on_query = False
        self.raise_on_start = False

    def __call__(self, argv: Sequence[str]) -> RunResult:
        args = [str(arg) for arg in argv]
        self.calls.append(args)
        assert args[0] == POWERSHELL and args[-2] == "-Command", args
        script = args[-1]
        if script.startswith("Unregister-ScheduledTask"):
            self.verbs.append("unregister")
            if not self.installed:
                return RunResult(tuple(args), 1, stderr=TASK_MISSING)
            self.installed = False
            return RunResult(tuple(args), 0)
        if script.startswith("Start-ScheduledTask"):
            self.verbs.append("start")
            if self.raise_on_start:
                raise SchedulerError("powershell timed out after 60s")
            if not self.installed or not self.start_ok:
                return RunResult(tuple(args), 1, stderr="Start-ScheduledTask : The task is disabled.")
            self.started = True
            return RunResult(tuple(args), 0)
        if script.startswith("$t = Get-ScheduledTask"):
            self.verbs.append("query")
            if self.raise_on_query:
                raise SchedulerError("powershell not found on PATH")
            if not self.installed:
                return RunResult(tuple(args), 1, stderr=TASK_MISSING)
            return RunResult(tuple(args), 0, stdout="TaskName: espn-fantasy-run\nState: Running\n")
        if "Register-ScheduledTask" in script:
            self.verbs.append("register")
            self.installed = self.register_ok
            return RunResult(tuple(args), 0 if self.register_ok else 1, stderr="" if self.register_ok else "denied")
        raise AssertionError(f"unexpected script {script}")


class FakeLaunchctl:
    """Answers ``launchctl bootout|enable|bootstrap|print`` for the run agent; records every argv."""

    def __init__(self, *, loaded: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.loaded = loaded

    def __call__(self, argv: Sequence[str]) -> RunResult:
        args = [str(arg) for arg in argv]
        self.calls.append(args)
        verb = args[1]
        if verb == "bootout":
            if not self.loaded:
                return RunResult(tuple(args), 3, stderr="Boot-out failed: 3: No such process")
            self.loaded = False
            return RunResult(tuple(args), 0)
        if verb == "enable":
            return RunResult(tuple(args), 0)
        if verb == "bootstrap":
            self.loaded = True
            return RunResult(tuple(args), 0)
        if verb == "print":
            if not self.loaded:
                return RunResult(tuple(args), 113, stderr=LAUNCHD_MISSING)
            return RunResult(tuple(args), 0, stdout=f"gui/{UID}/{LABEL} = {{\n\tstate = running\n}}\n")
        raise AssertionError(f"unexpected command {args}")

    def verbs(self) -> list[str]:
        return [call[1] for call in self.calls]


def windows_spec(pinned: dict[str, Path], *, windowless: bool = True, **environ: str) -> AutostartSpec:
    if not windowless:
        pinned["uvw"].unlink()
    return WINDOWS_AUTOSTART.build_spec(repo_root=REPO, environ={**USER_ENV, **environ})


def macos_spec(**environ: str) -> AutostartSpec:
    return MACOS_AUTOSTART.build_spec(uv_path=MAC_UV, repo_root=REPO, environ=environ)


# --- the spec ---------------------------------------------------------------------------------------------------------


def test_the_windows_spec_prefers_the_windowless_launcher_and_embeds_overrides_only_when_set(
    pinned: dict[str, Path], tmp_path: Path
) -> None:
    plain = windows_spec(pinned)
    assert plain.uv_path == pinned["uvw"] and plain.windowless
    assert plain.argv() == [str(pinned["uvw"]), "run", "--project", str(REPO), "fm", "run"]
    assert plain.env() == {} and plain.user == "PC\\someone"
    assert plain.env_file_path == paths.config_dir() / "scripts" / "espn-fantasy-run.env"
    assert plain.log_path == paths.config_dir() / "logs" / "run.log"
    custom = windows_spec(pinned, FM_CONFIG_DIR=str(tmp_path / "cfg"))
    assert custom.env() == {"FM_CONFIG_DIR": str((tmp_path / "cfg").resolve())}


def test_without_uvw_the_windows_spec_falls_back_to_uv(pinned: dict[str, Path]) -> None:
    spec = windows_spec(pinned, windowless=False)
    assert spec.uv_path == pinned["uv"] and not spec.windowless


def test_the_current_user_is_domain_qualified_when_windows_says_so() -> None:
    assert current_user({"USERNAME": "someone", "USERDOMAIN": "PC"}) == "PC\\someone"
    assert current_user({"USERNAME": "someone"}) == "someone"
    assert current_user({"USER": "mac"}) == "mac"


# --- Windows: Register-ScheduledTask ----------------------------------------------------------------------------------


def test_the_logon_task_runs_fm_run_windowless_at_this_users_logon_restarted_and_unlimited(
    pinned: dict[str, Path], tmp_path: Path
) -> None:
    spec = windows_spec(pinned, FM_CONFIG_DIR=str(tmp_path / "cfg"))
    argv = render_register_logon_task(spec)
    assert argv[:-1] == [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]
    script = argv[-1]
    arguments = subprocess.list2cmdline(
        ["run", "--project", str(REPO), "--env-file", str(spec.env_file_path), "fm", "run"]
    )
    assert render_autostart_argv(spec) == [
        str(pinned["uvw"]),
        "run",
        "--project",
        str(REPO),
        "--env-file",
        str(spec.env_file_path),
        "fm",
        "run",
    ]
    assert f"New-ScheduledTaskAction -Execute '{pinned['uvw']}' -Argument '{arguments}'" in script
    assert f"-WorkingDirectory '{REPO}'" in script
    assert "$trigger = New-ScheduledTaskTrigger -AtLogOn -User 'PC\\someone';" in script
    assert "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew" in script
    assert f"-RestartCount {RUN_RESTART_COUNT} -RestartInterval (New-TimeSpan -Minutes {RUN_RESTART_MINUTES})" in script
    assert f"$settings.ExecutionTimeLimit = '{NO_TIME_LIMIT}';" in script and NO_TIME_LIMIT == "PT0S"
    assert script.endswith(
        f"Register-ScheduledTask -TaskName '{AUTOSTART_TASK_NAME}' -Action $action -Trigger $trigger "
        "-Settings $settings -Force -ErrorAction Stop | Out-Null"
    )
    assert "-User" not in script.split("Register-ScheduledTask", 1)[1]  # the interactive token, no stored password


def test_a_quote_in_a_path_survives_the_powershell_literal_and_the_env_file(
    pinned: dict[str, Path], tmp_path: Path
) -> None:
    odd = tmp_path / "it's here"
    spec = windows_spec(pinned, FM_CONFIG_DIR=str(odd))
    assert f"-WorkingDirectory '{REPO}'" in render_register_logon_task(spec)[-1]
    assert dotenv_quote("C:\\plain") == "'C:\\plain'"
    text = render_env_file(spec)
    path = tmp_path / "x.env"
    path.write_text(text, encoding="utf-8")
    assert env_file_env(path) == {"FM_CONFIG_DIR": str(odd.resolve())}


def test_install_writes_the_env_file_registers_once_and_keeps_the_tick_task(
    pinned: dict[str, Path], tmp_path: Path
) -> None:
    spec = windows_spec(pinned, FM_CONFIG_DIR=str(tmp_path / "cfg"))
    fake = FakePowerShell()
    report = install_autostart(spec, fake)
    assert fake.calls == [render_register_logon_task(spec), render_start_task(AUTOSTART_TASK_NAME)]
    assert report.installed and fake.started
    assert report.files == [spec.env_file_path]
    assert env_file_env(spec.env_file_path) == {"FM_CONFIG_DIR": str((tmp_path / "cfg").resolve())}
    assert report.notes == [WAKE_TIMER_NOTE, STARTED]
    assert report.lines()[0] == f"install (windows) task {AUTOSTART_TASK_NAME}"


STARTED = f"{AUTOSTART_TASK_NAME}: started now, so fm run is running (Start-ScheduledTask)"


def test_install_starts_fm_run_now_rather_than_at_the_next_logon(pinned: dict[str, Path]) -> None:
    argv = render_start_task(AUTOSTART_TASK_NAME)
    assert argv[:-1] == [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]
    assert argv[-1] == f"Start-ScheduledTask -TaskName '{AUTOSTART_TASK_NAME}' -ErrorAction Stop"
    spec = windows_spec(pinned)
    assert "  " + subprocess.list2cmdline(argv) in render_autostart_plan(spec)
    assert "  " + subprocess.list2cmdline(argv) not in render_autostart_plan(spec, start=False)


@pytest.mark.parametrize("raise_on_start", [False, True])
def test_a_start_that_fails_is_a_note_and_the_task_stays_installed(
    pinned: dict[str, Path], raise_on_start: bool
) -> None:
    fake = FakePowerShell(start_ok=False)
    fake.raise_on_start = raise_on_start
    report = install_autostart(windows_spec(pinned), fake)
    assert report.installed and fake.installed and not fake.started
    assert fake.verbs == ["register", "start"]
    note = report.notes[-1]
    assert note.startswith(f"{AUTOSTART_TASK_NAME}: registered but not started now (")
    assert ("timed out" if raise_on_start else "The task is disabled") in note
    later = f"it starts at the next logon, or now with: Start-ScheduledTask -TaskName {AUTOSTART_TASK_NAME}"
    assert note.endswith(later)


def test_install_without_starting_registers_only(pinned: dict[str, Path]) -> None:
    fake = FakePowerShell()
    report = install_autostart(windows_spec(pinned), fake, start=False)
    assert fake.verbs == ["register"] and not fake.started and report.installed
    assert report.notes[-1] == f"{AUTOSTART_TASK_NAME}: not started now (--no-start); it starts at the next logon"


def test_install_with_uv_exe_says_a_console_window_stays_open(pinned: dict[str, Path]) -> None:
    spec = windows_spec(pinned, windowless=False)
    report = install_autostart(spec, FakePowerShell())
    assert report.notes == [UVW_FALLBACK_NOTE, WAKE_TIMER_NOTE, STARTED]
    assert f"-Execute '{pinned['uv']}'" in render_register_logon_task(spec)[-1]
    assert render_autostart_plan(spec)[-1] == f"note: {UVW_FALLBACK_NOTE}"


def test_a_failed_registration_is_an_error(pinned: dict[str, Path]) -> None:
    with pytest.raises(SchedulerError, match="Register-ScheduledTask espn-fantasy-run failed"):
        install_autostart(windows_spec(pinned), FakePowerShell(register_ok=False))


def test_uninstall_and_show_report_a_missing_task_without_failing(pinned: dict[str, Path]) -> None:
    spec = windows_spec(pinned)
    fake = FakePowerShell()
    report = uninstall_autostart(spec, fake)
    assert report.notes == [f"{AUTOSTART_TASK_NAME}: not installed"] and report.files == []
    assert fake.calls == [render_unregister_task(AUTOSTART_TASK_NAME)]
    assert show_autostart(spec, fake).startswith(f"{AUTOSTART_TASK_NAME}: not installed (fm autostart install)")
    fake.raise_on_query = True
    assert "cannot query the scheduler (powershell not found on PATH)" in show_autostart(spec, fake)
    assert fake.calls[1] == render_get_task(AUTOSTART_TASK_NAME)


def test_uninstall_removes_the_task_and_the_env_file(pinned: dict[str, Path]) -> None:
    spec = windows_spec(pinned)
    fake = FakePowerShell()
    install_autostart(spec, fake)
    assert "State: Running" in show_autostart(spec, fake)
    report = uninstall_autostart(spec, fake)
    assert report.files == [spec.env_file_path] and not spec.env_file_path.exists()
    assert fake.verbs == ["register", "start", "query", "unregister"]


# --- macOS: the KeepAlive LaunchAgent ---------------------------------------------------------------------------------


def test_the_run_agent_starts_fm_run_at_login_and_keeps_it_alive(tmp_path: Path) -> None:
    spec = macos_spec(FM_CONFIG_DIR=str(tmp_path / "cfg"))
    agent = plistlib.loads(render_run_plist(spec))
    assert agent["Label"] == LABEL
    assert agent["ProgramArguments"] == [str(MAC_UV), "run", "--project", str(REPO), "fm", "run"]
    assert agent["WorkingDirectory"] == str(REPO)
    assert agent["RunAtLoad"] is True and agent["KeepAlive"] is True and agent["ProcessType"] == "Interactive"
    assert agent["EnvironmentVariables"] == {"FM_CONFIG_DIR": str((tmp_path / "cfg").resolve())}
    assert agent["StandardOutPath"].endswith("run-launchd.log")
    assert "EnvironmentVariables" not in plistlib.loads(render_run_plist(macos_spec()))


def test_the_run_agent_install_show_uninstall_round_trip(pinned: dict[str, Path]) -> None:
    spec = macos_spec()
    fake = FakeLaunchctl()
    report = install_autostart_agent(spec, fake)
    plist = pinned["agents"] / f"{LABEL}.plist"
    assert report.files == [plist] and plist.is_file() and report.installed
    assert report.notes == [WAKE_TIMER_NOTE]
    assert fake.verbs() == ["bootout", "enable", "bootstrap"]
    assert fake.calls[-1] == ["launchctl", "bootstrap", f"gui/{UID}", str(plist)]
    assert "state = running" in show_autostart_agent(spec, fake)
    report = uninstall_autostart_agent(spec, fake)
    assert report.files == [plist] and not plist.exists()
    assert show_autostart_agent(spec, fake).startswith(f"{LABEL}: not installed (fm autostart install)")
    plan = render_autostart_agent_plan(spec)
    assert (
        plan[0] == f"agent {plist}:" and "  " + shlex.join(["launchctl", "bootstrap", f"gui/{UID}", str(plist)]) in plan
    )


# --- fm autostart -----------------------------------------------------------------------------------------------------


def _root() -> None:
    pass


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    autostart_cmd.register(root)
    return root


def run(*args: str, expect: int = 0) -> str:
    result = runner.invoke(cli(), list(args), catch_exceptions=False)
    assert result.exit_code == expect, result.output
    return result.output


def use(monkeypatch: pytest.MonkeyPatch, backend: AutostartBackend, fake: FakePowerShell | FakeLaunchctl) -> None:
    monkeypatch.setattr(autostart_cmd, "BACKENDS", {sys.platform: backend})
    monkeypatch.setattr(autostart_cmd, "default_runner", fake)


def test_install_dry_run_renders_the_logon_task_and_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePowerShell()
    use(monkeypatch, WINDOWS_AUTOSTART, fake)
    output = run("autostart", "install", "--dry-run")
    assert output.startswith("dry run: nothing is written or installed")
    assert "Register-ScheduledTask" in output and "-AtLogOn" in output and "uvw.exe" in output
    assert fake.calls == []
    assert not (paths.config_dir() / "scripts" / f"{AUTOSTART_TASK_NAME}.env").exists()


def test_install_dry_run_on_macos_renders_the_agent_and_runs_nothing(
    monkeypatch: pytest.MonkeyPatch, pinned: dict[str, Path]
) -> None:
    fake = FakeLaunchctl()
    use(monkeypatch, MACOS_AUTOSTART, fake)
    output = run("autostart", "install", "--dry-run")
    assert output.startswith("dry run: nothing is written or installed")
    assert "<key>KeepAlive</key>" in output and "<key>RunAtLoad</key>" in output and "launchctl bootstrap" in output
    assert fake.calls == [] and not (pinned["agents"] / f"{LABEL}.plist").exists()


def test_install_show_uninstall_round_trip_and_show_says_whether_fm_run_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePowerShell()
    use(monkeypatch, WINDOWS_AUTOSTART, fake)
    output = run("autostart", "install")
    assert f"install (windows) task {AUTOSTART_TASK_NAME}" in output
    assert "installed: fm run at logon (Windows Task Scheduler)" in output and "wake timer" in output
    assert f"note  {STARTED}" in output and "Start-ScheduledTask" in output
    output = run("autostart", "show")
    assert "State: Running" in output and "fm run" in output
    assert "Active now:                           no (the run lock is free)" in output
    with RunLock(command="fm run"):
        assert f"Active now:                           yes, fm run (pid {os.getpid()})" in run("autostart", "show")
    output = run("autostart", "uninstall")
    assert f"uninstall (windows) task {AUTOSTART_TASK_NAME}" in output
    assert fake.verbs == ["register", "start", "query", "query", "unregister"]


def test_install_no_start_registers_only_on_windows_and_is_a_note_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePowerShell()
    use(monkeypatch, WINDOWS_AUTOSTART, fake)
    monkeypatch.setattr(autostart_cmd, "NO_START_BACKENDS", {sys.platform: WINDOWS_AUTOSTART_NO_START})
    output = run("autostart", "install", "--dry-run", "--no-start")
    assert "Register-ScheduledTask" in output and "Start-ScheduledTask" not in output and fake.calls == []
    output = run("autostart", "install", "--no-start")
    assert fake.verbs == ["register"] and not fake.started
    assert "not started now (--no-start); it starts at the next logon" in output

    launchctl = FakeLaunchctl()
    use(monkeypatch, MACOS_AUTOSTART, launchctl)
    monkeypatch.setattr(autostart_cmd, "NO_START_BACKENDS", {})
    output = run("autostart", "install", "--no-start")
    assert "note: --no-start does not apply to launchd" in output
    assert launchctl.verbs() == ["bootout", "enable", "bootstrap"]


def test_a_failed_install_and_an_unsupported_platform_fail_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, WINDOWS_AUTOSTART, FakePowerShell(register_ok=False))
    assert "error: Register-ScheduledTask" in run("autostart", "install", expect=1)
    monkeypatch.setattr(autostart_cmd, "BACKENDS", {})
    assert "fm autostart supports Windows and macOS" in run("autostart", "show", expect=1)
