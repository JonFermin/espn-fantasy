"""The Windows scheduler backend and ``fm schedule``: rendered commands only, nothing installed.

Every ``schtasks`` / ``powershell`` call goes to a fake runner that records argv and answers from a script; an
autouse guard turns a real ``subprocess.run`` from the backend into a test failure. ``fm schedule`` is driven through
a private typer root with the same fake runner. The wrapper is written under the per-test config dir (conftest).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import pytest
import typer
from typer.testing import CliRunner

from fm import paths
from fm.commands import schedule as schedule_cmd
from fm.jobs import scheduler_windows as scheduler
from fm.jobs.scheduler_windows import (
    DEFAULT_INTERVAL_MINUTES,
    EXECUTION_TIME_LIMIT_HOURS,
    TASK_NAME,
    TASKRUN_MAX_LEN,
    RunResult,
    SchedulerError,
    ScheduleSpec,
    cmd_quote,
    install,
    is_not_found,
    ps_quote,
    render_plan,
    render_power_settings,
    render_schtasks_create,
    render_schtasks_delete,
    render_schtasks_query,
    render_wrapper_cmd,
    show,
    uninstall,
    wrapper_env,
)

UV = Path("C:/Tools/uv bin/uv.exe")
REPO = Path("C:/Users/someone/DEVELOP/espn fantasy")
NOT_FOUND = "ERROR: The system cannot find the file specified."
QUERY_LISTING = "HostName: PC\nTaskName: \\espn-fantasy-tick\nStatus: Ready\nPower Management: Stop On Battery Mode\n"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _never_run_for_real(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError(f"the scheduler backend ran a real command in a unit test: {args[0]!r}")

    monkeypatch.setattr(scheduler.subprocess, "run", refuse)
    monkeypatch.setattr(scheduler, "find_uv", lambda: UV)


class FakeRunner:
    """Answers each command by its verb (``/Create``, ``/Delete``, ``/Query``, powershell); records every argv."""

    def __init__(self, *, installed: bool = True, create_ok: bool = True, power_ok: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.installed = installed
        self.create_ok = create_ok
        self.power_ok = power_ok
        self.raise_on: set[str] = set()

    def __call__(self, argv: Sequence[str]) -> RunResult:
        args = [str(a) for a in argv]
        self.calls.append(args)
        verb = args[1] if args[0] == "schtasks" else args[0]
        if verb in self.raise_on:
            raise SchedulerError(f"{args[0]} not found on PATH")
        if verb == "/Create":
            self.installed = self.create_ok
            return RunResult(tuple(args), 0 if self.create_ok else 1, stderr="" if self.create_ok else "ERROR: nope")
        if verb == "/Delete":
            if not self.installed:
                return RunResult(tuple(args), 1, stderr=NOT_FOUND)
            self.installed = False
            return RunResult(tuple(args), 0, stdout="SUCCESS: The scheduled task was deleted.")
        if verb == "/Query":
            if not self.installed:
                return RunResult(tuple(args), 1, stderr=NOT_FOUND)
            return RunResult(tuple(args), 0, stdout=QUERY_LISTING)
        if verb == scheduler.POWERSHELL:
            return RunResult(tuple(args), 0 if self.power_ok else 1, stderr="" if self.power_ok else "Access denied")
        raise AssertionError(f"unexpected command {args}")

    def verbs(self) -> list[str]:
        return [call[1] if call[0] == "schtasks" else call[0] for call in self.calls]


def spec(**changes: object) -> ScheduleSpec:
    base = ScheduleSpec.build(uv_path=UV, repo_root=REPO, environ={})
    return ScheduleSpec(**{**_fields(base), **changes})  # type: ignore[arg-type]


def _fields(value: ScheduleSpec) -> dict[str, object]:
    return {name: getattr(value, name) for name in ScheduleSpec.__dataclass_fields__}


# --- spec -------------------------------------------------------------------------------------------------------------


def test_spec_resolves_dirs_under_the_config_dir_and_embeds_overrides_only_when_set() -> None:
    plain = ScheduleSpec.build(uv_path=UV, repo_root=REPO, environ={})
    assert plain.scripts_dir == paths.config_dir() / "scripts"
    assert plain.logs_dir == paths.config_dir() / "logs"
    assert plain.wrapper_path == paths.config_dir() / "scripts" / f"{TASK_NAME}.cmd"
    assert plain.log_path == paths.config_dir() / "logs" / "tick.log"
    assert plain.env() == {}
    assert plain.argv() == [str(UV), "run", "--project", str(REPO), "fm", "tick"]
    assert plain.interval_minutes == DEFAULT_INTERVAL_MINUTES

    overridden = ScheduleSpec.build(
        uv_path=UV, repo_root=REPO, environ={"FM_CONFIG_DIR": os.environ["FM_CONFIG_DIR"], "FM_CACHE_DIR": "cache x"}
    )
    env = overridden.env()
    assert Path(env["FM_CONFIG_DIR"]) == paths.config_dir().resolve()
    assert env["FM_CACHE_DIR"].endswith("cache x")


def test_spec_rejects_a_bad_interval_or_task_name_and_needs_uv_unless_told_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SchedulerError, match="1..1439"):
        ScheduleSpec.build(uv_path=UV, repo_root=REPO, interval_minutes=0, environ={})
    with pytest.raises(SchedulerError, match="task name"):
        spec(task_name="bad/name")
    monkeypatch.setattr(scheduler, "find_uv", lambda: None)
    with pytest.raises(SchedulerError, match="uv not found"):
        ScheduleSpec.build(repo_root=REPO, environ={})
    assert ScheduleSpec.build(repo_root=REPO, environ={}, require_uv=False).uv_path == Path("uv")


# --- rendering --------------------------------------------------------------------------------------------------------


def test_quoting() -> None:
    assert cmd_quote("plain") == "plain"
    assert cmd_quote("C:/with space/x") == '"C:/with space/x"'
    assert cmd_quote("a&b") == '"a&b"'
    assert cmd_quote("") == '""'
    assert ps_quote("it's") == "'it''s'"


def test_wrapper_cmd_runs_the_tick_from_the_repo_with_the_env_and_the_log() -> None:
    this = spec(config_dir=Path("C:/cfg dir"), cache_dir=Path("C:/cache"))
    text = render_wrapper_cmd(this)
    assert text.endswith("\r\n") and "\n" not in text.replace("\r\n", "")
    lines = text.splitlines()
    assert lines[0] == "@echo off"
    assert lines[2] == f'cd /d "{REPO}"'
    assert 'set "FM_CONFIG_DIR=C:\\cfg dir"' in lines or 'set "FM_CONFIG_DIR=C:/cfg dir"' in lines
    assert any(line.startswith('set "FM_CACHE_DIR=') for line in lines)
    assert any(line.startswith("if not exist ") and "mkdir" in line for line in lines)
    command = next(line for line in lines if line.endswith("2>&1"))
    assert command.startswith(f'"{UV}" run --project "{REPO}" fm tick >> ')
    assert str(this.log_path) in command
    assert lines[-1] == "exit /b %ERRORLEVEL%"


def test_wrapper_env_reads_back_what_the_wrapper_sets(tmp_path: Path) -> None:
    this = spec(config_dir=Path("C:/cfg"), cache_dir=None)
    path = tmp_path / "w.cmd"
    path.write_text(render_wrapper_cmd(this), encoding="utf-8", newline="")
    assert wrapper_env(path) == {"FM_CONFIG_DIR": "C:\\cfg"} or wrapper_env(path) == {"FM_CONFIG_DIR": "C:/cfg"}


def test_schtasks_create_argv_every_n_minutes_limited_interactive_force() -> None:
    this = spec(interval_minutes=7)
    argv = render_schtasks_create(this)
    assert argv == [
        "schtasks", "/Create",
        "/TN", TASK_NAME,
        "/SC", "MINUTE", "/MO", "7",
        "/TR", f'"{this.wrapper_path}"',
        "/RL", "LIMITED",
        "/F",
    ]  # fmt: skip
    assert "/RU" not in argv and "/NP" not in argv  # interactive token: the browser profile is the user's
    assert len(f'"{this.wrapper_path}"') <= TASKRUN_MAX_LEN


def test_a_wrapper_path_past_the_taskrun_limit_is_refused() -> None:
    long = spec(scripts_dir=Path("C:/" + "x" * (TASKRUN_MAX_LEN + 1)))
    with pytest.raises(SchedulerError, match="/TR limit"):
        render_schtasks_create(long)


def test_power_settings_wake_the_pc_and_allow_battery() -> None:
    argv = render_power_settings("espn-fantasy-tick")
    assert argv[:6] == ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]
    command = argv[6]
    assert command.startswith("Set-ScheduledTask -TaskName 'espn-fantasy-tick' -Settings (New-ScheduledTaskSettingsSet")
    for switch in ("-WakeToRun", "-AllowStartIfOnBatteries", "-DontStopIfGoingOnBatteries"):
        assert switch in command
    assert "-MultipleInstances IgnoreNew" in command
    assert f"-ExecutionTimeLimit (New-TimeSpan -Hours {EXECUTION_TIME_LIMIT_HOURS})" in command


def test_delete_query_and_not_found() -> None:
    assert render_schtasks_delete("t") == ["schtasks", "/Delete", "/TN", "t", "/F"]
    assert render_schtasks_query("t") == ["schtasks", "/Query", "/TN", "t", "/FO", "LIST", "/V"]
    assert is_not_found(RunResult(("schtasks",), 1, stderr=NOT_FOUND))
    assert not is_not_found(RunResult(("schtasks",), 0, stdout=NOT_FOUND))
    assert not is_not_found(RunResult(("schtasks",), 1, stderr="ERROR: Access is denied."))


def test_render_plan_lists_the_wrapper_and_both_commands() -> None:
    lines = render_plan(spec())
    assert lines[0].startswith("wrapper ") and lines[0].endswith(f"{TASK_NAME}.cmd:")
    assert "  @echo off" in lines
    assert any(line.strip().startswith("schtasks /Create /TN espn-fantasy-tick /SC MINUTE /MO 10") for line in lines)
    assert any("Set-ScheduledTask" in line and "-WakeToRun" in line for line in lines)


# --- actions ----------------------------------------------------------------------------------------------------------


def test_install_writes_the_wrapper_creates_the_task_then_relaxes_power() -> None:
    this = spec()
    fake = FakeRunner(installed=False)
    report = install(this, fake)
    assert fake.verbs() == ["/Create", "powershell"]
    assert fake.calls[0] == render_schtasks_create(this)
    assert fake.calls[1] == render_power_settings(TASK_NAME)
    assert report.installed and report.files == [this.wrapper_path] and report.notes == []
    raw = this.wrapper_path.read_bytes()
    assert raw == render_wrapper_cmd(this).encode("utf-8") and b"\r\n" in raw
    assert this.logs_dir.is_dir()
    assert any(line.startswith("install (windows) task espn-fantasy-tick") for line in report.lines())


def test_install_notes_a_failed_power_rewrite_and_raises_on_a_failed_create() -> None:
    this = spec()
    report = install(this, FakeRunner(installed=False, power_ok=False))
    assert report.installed and len(report.notes) == 1
    assert "could not allow waking and running on battery" in report.notes[0]
    with pytest.raises(SchedulerError, match="/Create .* failed"):
        install(this, FakeRunner(installed=False, create_ok=False))


def test_uninstall_deletes_the_task_and_the_wrapper_and_tolerates_a_missing_task() -> None:
    this = spec()
    fake = FakeRunner(installed=False)
    install(this, fake)
    assert this.wrapper_path.exists()
    report = uninstall(this, fake)
    assert fake.verbs()[-1] == "/Delete"
    assert report.files == [this.wrapper_path] and not this.wrapper_path.exists() and report.notes == []
    again = uninstall(this, fake)
    assert again.notes == [f"{TASK_NAME}: not installed"] and again.files == []
    broken = FakeRunner(installed=True)
    broken.raise_on.add("/Delete")
    with pytest.raises(SchedulerError):
        uninstall(this, broken)


def test_show_reports_not_installed_installed_with_wrapper_env_and_an_unavailable_scheduler() -> None:
    this = spec(config_dir=Path("C:/cfg"))
    assert show(this, FakeRunner(installed=False)).startswith(f"{TASK_NAME}: not installed")
    assert "not written" in show(this, FakeRunner(installed=False))
    fake = FakeRunner(installed=False)
    install(this, fake)
    text = show(this, fake)
    assert text.startswith("HostName: PC")
    assert "Wrapper Env:" in text and "FM_CONFIG_DIR=" in text
    assert "keeps the schtasks default power conditions" not in text  # the fixture listing has no such marker
    fake.raise_on.add("/Query")
    assert "cannot query the scheduler" in show(this, fake)


def test_show_flags_the_schtasks_default_power_conditions() -> None:
    class Restricted(FakeRunner):
        def __call__(self, argv: Sequence[str]) -> RunResult:
            result = super().__call__(argv)
            if result.ok and "/Query" in argv:
                return RunResult(result.argv, 0, stdout=result.stdout + "Power Management: No Start On Batteries\n")
            return result

    assert "re-run fm schedule install" in show(spec(), Restricted(installed=True))


# --- fm schedule ------------------------------------------------------------------------------------------------------


def _root() -> None:
    pass


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    schedule_cmd.register(root)
    return root


def run(*args: str, expect: int = 0) -> str:
    result = runner.invoke(cli(), list(args), catch_exceptions=False)
    assert result.exit_code == expect, result.output
    return result.output


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    fake = FakeRunner(installed=False)
    monkeypatch.setattr(scheduler, "default_runner", fake)
    return fake


def test_schedule_show_exits_0_when_nothing_is_installed(fake_runner: FakeRunner) -> None:
    output = run("schedule", "show")
    assert f"{TASK_NAME}: not installed" in output
    assert "Runs:" in output and "fm tick" in output and "Log:" in output
    assert fake_runner.verbs() == ["/Query"]


def test_schedule_install_dry_run_renders_and_runs_nothing(fake_runner: FakeRunner) -> None:
    output = run("schedule", "install", "--dry-run", "--every", "5")
    assert output.startswith("dry run: nothing is written or installed")
    assert "/SC MINUTE /MO 5" in output and "-WakeToRun" in output
    assert fake_runner.calls == []
    assert not (paths.config_dir() / "scripts" / f"{TASK_NAME}.cmd").exists()


def test_schedule_install_show_uninstall_round_trip(fake_runner: FakeRunner) -> None:
    output = run("schedule", "install")
    assert "installed: fm tick every 10 minutes" in output
    wrapper = paths.config_dir() / "scripts" / f"{TASK_NAME}.cmd"
    assert wrapper.is_file()
    assert "HostName: PC" in run("schedule", "show")
    output = run("schedule", "uninstall")
    assert f"file  {wrapper}" in output and not wrapper.exists()
    assert fake_runner.verbs() == ["/Create", "powershell", "/Query", "/Delete"]


def test_schedule_install_reports_a_failed_create(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler, "default_runner", FakeRunner(installed=False, create_ok=False))
    result = runner.invoke(cli(), ["schedule", "install"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "error: schtasks /Create" in result.output


def test_default_runner_never_uses_a_shell_and_maps_a_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
    result = scheduler.default_runner(["schtasks", "/Query"])
    assert result.ok and result.stdout == "ok" and seen["args"] == ["schtasks", "/Query"]
    assert "shell" not in seen["kwargs"]  # type: ignore[operator]

    def missing(args: list[str], **kwargs: object) -> NoReturn:
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(scheduler.subprocess, "run", missing)
    with pytest.raises(SchedulerError, match="not found on PATH"):
        scheduler.default_runner(["schtasks", "/Query"])
