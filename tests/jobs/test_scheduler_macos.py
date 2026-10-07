"""The macOS launchd backend and ``fm schedule`` on macOS: rendered files and commands only, nothing loaded.

Every ``launchctl`` call goes to a fake runner that records argv and answers from a script; an autouse guard turns a
real ``subprocess.run`` into a test failure. The uid and ``~/Library/LaunchAgents`` are pinned, so these run on any
platform. The wrapper is written under the per-test config dir (conftest).
"""

from __future__ import annotations

import plistlib
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import pytest
import typer
from typer.testing import CliRunner

from fm import paths
from fm.commands import schedule as schedule_cmd
from fm.jobs import scheduler_base
from fm.jobs import scheduler_macos as macos
from fm.jobs.scheduler_base import TASK_NAME, RunResult, SchedulerError, ScheduleSpec
from fm.jobs.scheduler_macos import (
    MACOS,
    WAKE_NOTE,
    agent_label,
    agent_wrapper_env,
    install_agent,
    is_agent_missing,
    plist_path,
    render_agent_plan,
    render_launchctl_bootout,
    render_launchctl_bootstrap,
    render_launchctl_enable,
    render_launchctl_print,
    render_plist,
    render_wrapper_sh,
    show_agent,
    uninstall_agent,
)
from fm.jobs.scheduler_windows import WINDOWS

UV = Path("/Users/someone/.local/bin/uv")
REPO = Path("/Users/someone/DEVELOP/espn fantasy")
UID = 501
LABEL = f"local.{TASK_NAME}"
MISSING = f'Could not find service "{LABEL}" in domain for user gui: {UID}'
PRINT_LISTING = f"gui/{UID}/{LABEL} = {{\n\tactive count = 0\n\tstate = not running\n}}\n"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _pinned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    def refuse(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError(f"the scheduler backend ran a real command in a unit test: {args[0]!r}")

    agents = tmp_path / "LaunchAgents"
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(scheduler_base, "find_uv", lambda: UV)
    monkeypatch.setattr(macos, "current_uid", lambda: UID)
    monkeypatch.setattr(macos, "launch_agents_dir", lambda: agents)
    return agents


class FakeLaunchctl:
    """Answers ``launchctl bootout|enable|bootstrap|print`` like launchd would; records every argv."""

    def __init__(self, *, loaded: bool = False, bootstrap_ok: bool = True, enable_ok: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.loaded = loaded
        self.bootstrap_ok = bootstrap_ok
        self.enable_ok = enable_ok
        self.raise_on: set[str] = set()

    def __call__(self, argv: Sequence[str]) -> RunResult:
        args = [str(a) for a in argv]
        self.calls.append(args)
        assert args[0] == "launchctl", args
        verb = args[1]
        if verb in self.raise_on:
            raise SchedulerError("launchctl not found on PATH")
        if verb == "bootout":
            if not self.loaded:
                return RunResult(tuple(args), 3, stderr="Boot-out failed: 3: No such process")
            self.loaded = False
            return RunResult(tuple(args), 0)
        if verb == "enable":
            return RunResult(tuple(args), 0 if self.enable_ok else 1, stderr="" if self.enable_ok else "denied")
        if verb == "bootstrap":
            if not self.bootstrap_ok:
                return RunResult(tuple(args), 5, stderr="Bootstrap failed: 5: Input/output error")
            self.loaded = True
            return RunResult(tuple(args), 0)
        if verb == "print":
            if not self.loaded:
                return RunResult(tuple(args), 113, stderr=MISSING)
            return RunResult(tuple(args), 0, stdout=PRINT_LISTING)
        raise AssertionError(f"unexpected command {args}")

    def verbs(self) -> list[str]:
        return [call[1] for call in self.calls]


def spec(**changes: object) -> ScheduleSpec:
    base = MACOS.build_spec(uv_path=UV, repo_root=REPO, environ={})
    fields = {name: getattr(base, name) for name in ScheduleSpec.__dataclass_fields__}
    return ScheduleSpec(**{**fields, **changes})  # type: ignore[arg-type]


# --- spec and rendering -----------------------------------------------------------------------------------------------


def test_the_macos_spec_writes_a_sh_wrapper_and_the_agent_under_launch_agents(_pinned: Path) -> None:
    this = spec()
    assert this.wrapper_path == paths.config_dir() / "scripts" / f"{TASK_NAME}.sh"
    assert agent_label(this) == LABEL
    assert plist_path(this) == _pinned / f"{LABEL}.plist"


def test_wrapper_sh_runs_the_tick_from_the_repo_with_the_env_and_the_log(tmp_path: Path) -> None:
    this = spec(config_dir=Path("/cfg dir"), cache_dir=Path("/cache"))
    text = render_wrapper_sh(this)
    assert "\r" not in text and text.endswith("\n")
    lines = text.splitlines()
    assert lines[0] == "#!/bin/sh"
    assert lines[2] == f"cd {shlex.quote(str(REPO))} || exit 1"
    assert f"export FM_CONFIG_DIR={shlex.quote(str(Path('/cfg dir')))}" in lines
    command = lines[-1]
    assert command.startswith("exec ") and command.endswith(" 2>&1")
    assert shlex.split(command)[1:7] == [str(UV), "run", "--project", str(REPO), "fm", "tick"]
    assert str(this.log_path) in command

    path = tmp_path / "w.sh"
    path.write_text(text, encoding="utf-8", newline="\n")
    assert agent_wrapper_env(path) == {"FM_CONFIG_DIR": str(Path("/cfg dir")), "FM_CACHE_DIR": str(Path("/cache"))}


def test_plist_runs_the_wrapper_every_interval_at_login_unthrottled() -> None:
    this = spec(interval_minutes=7)
    agent = plistlib.loads(render_plist(this))
    assert agent["Label"] == LABEL
    assert agent["ProgramArguments"] == ["/bin/sh", str(this.wrapper_path)]
    assert agent["StartInterval"] == 7 * 60
    assert agent["RunAtLoad"] is True and agent["ProcessType"] == "Interactive"
    assert agent["WorkingDirectory"] == str(REPO)
    assert agent["StandardOutPath"] == agent["StandardErrorPath"] == str(this.logs_dir / "launchd.log")


def test_launchctl_commands_target_the_users_gui_domain() -> None:
    this = spec()
    service = f"gui/{UID}/{LABEL}"
    assert render_launchctl_bootout(this) == ["launchctl", "bootout", service]
    assert render_launchctl_enable(this) == ["launchctl", "enable", service]
    assert render_launchctl_bootstrap(this) == ["launchctl", "bootstrap", f"gui/{UID}", str(plist_path(this))]
    assert render_launchctl_print(this) == ["launchctl", "print", service]
    assert is_agent_missing(RunResult(("launchctl",), 113, stderr=MISSING))
    assert is_agent_missing(RunResult(("launchctl",), 3, stderr="Boot-out failed: 3: No such process"))
    assert not is_agent_missing(RunResult(("launchctl",), 5, stderr="Bootstrap failed: 5: Input/output error"))


def test_render_agent_plan_lists_the_wrapper_the_plist_and_the_commands() -> None:
    lines = render_agent_plan(spec())
    assert lines[0].startswith("wrapper ") and lines[0].endswith(f"{TASK_NAME}.sh:")
    assert "  #!/bin/sh" in lines
    assert any(line.startswith("agent ") and line.endswith(f"{LABEL}.plist:") for line in lines)
    assert any("<key>StartInterval</key>" in line for line in lines)
    commands = lines[lines.index("commands:") + 1 :]
    assert [shlex.split(line)[1] for line in commands] == ["bootout", "enable", "bootstrap"]


# --- actions ----------------------------------------------------------------------------------------------------------


class SlowUnload(FakeLaunchctl):
    """launchd's real behaviour: the job is still listed for a few ``print``s after ``bootout`` answered."""

    def __init__(self, lingering: int) -> None:
        super().__init__(loaded=True)
        self.lingering = lingering

    def __call__(self, argv: Sequence[str]) -> RunResult:
        if argv[1] == "print" and self.lingering:
            self.lingering -= 1
            self.calls.append(list(argv))
            return RunResult(tuple(argv), 0, stdout=PRINT_LISTING)
        return super().__call__(argv)


def test_install_waits_for_a_replaced_agent_to_unload(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(macos.time, "sleep", slept.append)
    fake = SlowUnload(lingering=2)
    report = install_agent(spec(), fake)
    assert fake.verbs() == ["bootout", "print", "print", "print", "enable", "bootstrap"]
    assert len(slept) == 2 and report.notes == [WAKE_NOTE]

    stuck = SlowUnload(lingering=macos.UNLOAD_WAIT_ATTEMPTS)
    report = install_agent(spec(), stuck)
    assert any("still unloading" in note for note in report.notes)


def test_install_writes_both_files_and_loads_the_agent() -> None:
    this = spec()
    fake = FakeLaunchctl()
    report = install_agent(this, fake)
    assert fake.verbs() == ["bootout", "enable", "bootstrap"]
    assert report.installed and report.files == [this.wrapper_path, plist_path(this)]
    assert report.notes == [WAKE_NOTE]
    assert this.wrapper_path.read_bytes() == render_wrapper_sh(this).encode("utf-8")
    assert plist_path(this).read_bytes() == render_plist(this)
    assert this.logs_dir.is_dir()
    assert report.lines()[0] == f"install (macos) agent {LABEL}"


def test_install_replaces_a_loaded_agent_and_raises_on_a_failed_bootstrap() -> None:
    this = spec()
    fake = FakeLaunchctl(loaded=True)
    install_agent(this, fake)
    assert fake.loaded and fake.verbs() == ["bootout", "print", "enable", "bootstrap"]
    with pytest.raises(SchedulerError, match="bootstrap .* failed"):
        install_agent(this, FakeLaunchctl(bootstrap_ok=False))
    noted = install_agent(this, FakeLaunchctl(enable_ok=False))
    assert noted.installed and any("enable failed" in note for note in noted.notes)


def test_uninstall_unloads_and_removes_both_files_and_tolerates_a_missing_agent() -> None:
    this = spec()
    fake = FakeLaunchctl()
    install_agent(this, fake)
    report = uninstall_agent(this, fake)
    assert fake.verbs()[-1] == "bootout" and not fake.loaded
    assert report.files == [plist_path(this), this.wrapper_path]
    assert not plist_path(this).exists() and not this.wrapper_path.exists()
    again = uninstall_agent(this, fake)
    assert again.notes == [f"{LABEL}: not loaded"] and again.files == []
    broken = FakeLaunchctl(loaded=True)
    broken.raise_on.add("bootout")
    with pytest.raises(SchedulerError):
        uninstall_agent(this, broken)


def test_show_reports_not_installed_loaded_with_wrapper_env_and_an_unavailable_launchctl() -> None:
    this = spec(config_dir=Path("/cfg"))
    text = show_agent(this, FakeLaunchctl())
    assert text.startswith(f"{LABEL}: not installed") and "(not written)" in text
    fake = FakeLaunchctl()
    install_agent(this, fake)
    text = show_agent(this, fake)
    assert text.startswith(f"gui/{UID}/{LABEL} = {{")
    assert "Wrapper Env:" in text and "FM_CONFIG_DIR=" in text and "(not written)" not in text
    fake.raise_on.add("print")
    assert "cannot query launchd" in show_agent(this, fake)


# --- backend choice and fm schedule -----------------------------------------------------------------------------------


def test_backend_is_chosen_by_platform() -> None:
    assert schedule_cmd.backend("darwin") is MACOS
    assert schedule_cmd.backend("win32") is WINDOWS
    with pytest.raises(SchedulerError, match="Windows and macOS"):
        schedule_cmd.backend("linux")


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
def fake_launchctl(monkeypatch: pytest.MonkeyPatch) -> FakeLaunchctl:
    fake = FakeLaunchctl()
    monkeypatch.setattr(schedule_cmd, "backend", lambda: MACOS)
    monkeypatch.setattr(schedule_cmd, "default_runner", fake)
    return fake


def test_schedule_install_dry_run_on_macos_renders_and_runs_nothing(fake_launchctl: FakeLaunchctl) -> None:
    output = run("schedule", "install", "--dry-run", "--every", "5")
    assert output.startswith("dry run: nothing is written or installed")
    assert "<integer>300</integer>" in output and "launchctl bootstrap" in output
    assert fake_launchctl.calls == []
    assert not (paths.config_dir() / "scripts" / f"{TASK_NAME}.sh").exists()


def test_schedule_install_show_uninstall_round_trip_on_macos(fake_launchctl: FakeLaunchctl, _pinned: Path) -> None:
    output = run("schedule", "install")
    assert "installed: fm tick every 10 minutes (launchd)" in output and "does not wake a sleeping Mac" in output
    assert (_pinned / f"{LABEL}.plist").is_file()
    assert "state = not running" in run("schedule", "show")
    output = run("schedule", "uninstall")
    assert f"uninstall (macos) agent {LABEL}" in output and not (_pinned / f"{LABEL}.plist").exists()
    assert fake_launchctl.verbs() == ["bootout", "enable", "bootstrap", "print", "bootout"]


def test_schedule_on_an_unsupported_platform_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schedule_cmd, "BACKENDS", {})
    result = runner.invoke(cli(), ["schedule", "show"], catch_exceptions=False)
    assert result.exit_code == 1 and "supports Windows and macOS" in result.output
