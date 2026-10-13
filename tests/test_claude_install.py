"""``fm claude install|uninstall|show`` (ROADMAP #55): Claude Code and Claude Desktop, against a temp home.

Nothing here may reach the real ``~/.claude``, ``~/.claude.json`` or Claude Desktop config, or run the real ``claude``:
an autouse guard turns ``subprocess.run`` into a test failure and points ``Path.home()``, ``HOME``, ``USERPROFILE``,
``APPDATA`` and ``LOCALAPPDATA`` into the test's temp dir, so even a code path that ignored the injected
:class:`ClaudeHome` would land there. ``claude`` is a :class:`FakeClaude` runner that answers ``mcp add-json`` and
``mcp remove`` like Claude Code's user-scope registry, and uv and the repo are fixed fake paths (with a space, so the
quoting is exercised).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, NoReturn

import pytest
from typer.testing import CliRunner

import fm.claude.install as install_module
from fm.claude import (
    CLAUDE_COMMANDS,
    MCP_SERVER_NAME,
    ClaudeError,
    ClaudeHome,
    McpCommand,
    add_json_argv,
    asset_files,
    assets,
    claude_status,
    install_claude,
    remove_argv,
    shell_command,
    uninstall_claude,
)
from fm.cli import app
from fm.commands import claude as claude_cmd
from fm.jobs.scheduler_base import REPO_ROOT, RunResult, SchedulerError

REPO = Path("C:/Users/someone/DEVELOP/espn fantasy")
UV = Path("C:/Users/someone/.local/bin/uv.exe")
CLAUDE = "C:/Users/someone/.local/bin/claude.exe"
MSIX_PACKAGE = "Claude_pzs8sxrjxfjjc"
OTHER_SERVER = {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/notes"]}
DESKTOP_CONFIG = {
    "mcpServers": {"filesystem": OTHER_SERVER},
    "preferences": {"menuBarEnabled": False, "quickEntryShortcut": "off"},
    "globalShortcut": "",
}
SERVER_ARGS = ["run", "--project", str(REPO), "fm", "mcp"]

runner = CliRunner()


@pytest.fixture(autouse=True)
def user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The stand-in for the user's home, and no real command."""

    def refuse(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError(f"a real command ran in a unit test: {args[0]!r}")

    home = tmp_path / "user"
    home.mkdir()
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    return home


class FakeClaude:
    """Claude Code's ``claude mcp`` for the user scope: one registry, every argv recorded."""

    def __init__(self, *, registered: dict[str, Any] | None = None, fail_add: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.registered = registered
        self.fail_add = fail_add

    def __call__(self, argv: Sequence[str]) -> RunResult:
        args = [str(arg) for arg in argv]
        self.calls.append(args)
        assert args[1] == "mcp" and args[3:6] == ["--scope", "user", MCP_SERVER_NAME], args
        if args[2] == "add-json":
            if self.fail_add is not None:
                return RunResult(tuple(args), 1, stderr=self.fail_add)
            if self.registered is not None:
                return RunResult(tuple(args), 1, stderr=f"MCP server {MCP_SERVER_NAME} already exists in user config")
            self.registered = json.loads(args[6])
            return RunResult(tuple(args), 0, stdout=f"Added stdio MCP server {MCP_SERVER_NAME} to user config")
        if args[2] == "remove":
            if self.registered is None:
                missing = f'No user-scoped MCP server found with name: "{MCP_SERVER_NAME}"'
                return RunResult(tuple(args), 1, stderr=missing)
            self.registered = None
            return RunResult(tuple(args), 0, stdout=f"Removed MCP server {MCP_SERVER_NAME} from user config")
        raise AssertionError(f"unexpected claude command {args}")


def on_path(name: str) -> str | None:
    return CLAUDE if name == "claude" else None


def nowhere(name: str) -> str | None:
    return None


def windows(user: Path, **environ: str) -> ClaudeHome:
    return ClaudeHome(home=user, platform="win32", environ=environ)


def command(**environ: str) -> McpCommand:
    return McpCommand.build(uv_path=UV, repo_root=REPO, environ=environ)


def appdata_dir(user: Path) -> Path:
    folder = user / "AppData" / "Roaming" / "Claude"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def msix_dir(user: Path) -> Path:
    folder = user / "AppData" / "Local" / "Packages" / MSIX_PACKAGE / "LocalCache" / "Roaming" / "Claude"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_config(folder: Path, data: dict[str, Any]) -> Path:
    path = folder / "claude_desktop_config.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def frontmatter(text: str) -> tuple[dict[str, str], str]:
    """The ``key: value`` lines between the ``---`` fences, and the body. Values must be plain YAML scalars or quoted
    strings, with no second ``": "`` that would make YAML read them differently."""
    assert text.startswith("---\n"), text[:40]
    head, body = text[4:].split("\n---\n", 1)
    fields: dict[str, str] = {}
    for line in head.splitlines():
        key, value = line.split(": ", 1)
        if value.startswith('"'):
            assert value.endswith('"') and '"' not in value[1:-1], line
            value = value[1:-1]
        else:
            assert ": " not in value and " #" not in value and not value.startswith(("[", "{", "*", "&", "!")), line
        fields[key] = value
    return fields, body


# --- the packaged skill and commands ----------------------------------------------------------------------------------


def test_the_assets_ship_inside_the_package() -> None:
    root = resources.files("fm.claude") / "assets"
    assert root.is_dir()
    assert (root / "SKILL.md").is_file()
    assert all((root / "commands" / f"{name}.md").is_file() for name in CLAUDE_COMMANDS)
    assert str(assets()) == str(root)
    assert set(asset_files()) == {"SKILL.md", *(f"commands/{name}.md" for name in CLAUDE_COMMANDS)}


def test_the_skill_says_when_to_use_it_and_the_rules() -> None:
    fields, body = frontmatter(asset_files()["SKILL.md"])
    assert set(fields) == {"name", "description"} and fields["name"] == MCP_SERVER_NAME
    description = fields["description"]
    assert "ESPN" in description and "fantasy football" in description and "basketball" in description
    assert "Use it whenever" in description
    for tool in ("status", "lineup", "waivers", "trade_eval", "report", "health", "list_proposals", "create_proposal"):
        assert f"`{tool}`" in body, tool
    assert body.index("Start with `status`") < body.index("`lineup`")
    for rule in ("Never claim a move was made", "only when the user asks", "approval-only", "Cite the engine"):
        assert rule in body, rule
    for command_name in ("fm doctor", "fm setup", "fm ui"):
        assert f"`{command_name}`" in body, command_name


@pytest.mark.parametrize("name", CLAUDE_COMMANDS)
def test_each_slash_command_has_a_description_and_a_prompt(name: str) -> None:
    fields, body = frontmatter(asset_files()[f"commands/{name}.md"])
    assert fields["description"] and set(fields) <= {"description", "argument-hint"}
    assert "$ARGUMENTS" in body and "espn-fantasy" in body
    assert "create_proposal" not in body or "approval" in body  # a command never queues a move silently


def test_the_trade_command_takes_the_deal_and_stays_approval_only() -> None:
    fields, body = frontmatter(asset_files()["commands/trade.md"])
    assert fields["argument-hint"].startswith("give ")
    assert "`trade_eval`" in body and "approval-only" in body and "proposal_payload" in body


def test_the_week_command_runs_status_lineup_waivers_report_in_order() -> None:
    _, body = frontmatter(asset_files()["commands/week.md"])
    positions = [body.index(f"`{tool}`") for tool in ("status", "lineup", "waivers", "report")]
    assert positions == sorted(positions)


# --- the server command -----------------------------------------------------------------------------------------------


def test_the_server_command_is_uv_run_fm_mcp_with_the_dirs_only_when_set(tmp_path: Path) -> None:
    bare = command()
    assert bare.desktop_entry() == {"command": str(UV), "args": SERVER_ARGS}
    assert bare.code_entry() == {"type": "stdio", "command": str(UV), "args": SERVER_ARGS}
    pinned = command(FM_CONFIG_DIR=str(tmp_path / "cfg"), FM_CACHE_DIR="")
    assert pinned.desktop_entry()["env"] == {"FM_CONFIG_DIR": str((tmp_path / "cfg").resolve())}
    both = command(FM_CONFIG_DIR=str(tmp_path / "cfg"), FM_CACHE_DIR=str(tmp_path / "cache"))
    assert list(both.code_entry()["env"]) == ["FM_CONFIG_DIR", "FM_CACHE_DIR"]


def test_the_server_command_defaults_to_this_checkout_and_needs_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    assert McpCommand.build(uv_path=UV, environ={}).args[2] == str(REPO_ROOT)
    monkeypatch.setattr(install_module, "find_uv", lambda: None)
    with pytest.raises(ClaudeError, match="uv not found"):
        McpCommand.build(environ={})
    assert McpCommand.build(environ={}, require_uv=False).command == "uv"


def test_the_shell_command_quotes_for_the_platform() -> None:
    argv = add_json_argv(command(), "claude")
    posix = shell_command(argv, "darwin")
    assert posix.startswith("claude mcp add-json --scope user espn-fantasy '{")
    windows_line = shell_command(["claude", "it's here"], "win32")
    assert windows_line == "claude 'it''s here'"


# --- Claude Code ------------------------------------------------------------------------------------------------------


def test_install_runs_the_expected_add_json(user: Path) -> None:
    fake = FakeClaude()
    report = install_claude(windows(user), command(), desktop=False, runner=fake, which=on_path)
    assert not report.failed, report.lines
    payload = json.dumps({"type": "stdio", "command": str(UV), "args": SERVER_ARGS}, separators=(",", ":"))
    assert fake.calls == [[CLAUDE, "mcp", "add-json", "--scope", "user", "espn-fantasy", payload]]
    assert add_json_argv(command(), CLAUDE) == fake.calls[0]
    assert fake.registered == {"type": "stdio", "command": str(UV), "args": SERVER_ARGS}
    assert any("registered MCP server espn-fantasy" in line for line in report.lines)


def test_install_copies_the_skill_and_the_commands(user: Path) -> None:
    install_claude(windows(user), command(), desktop=False, runner=FakeClaude(), which=on_path)
    texts = asset_files()
    skill = user / ".claude" / "skills" / "espn-fantasy" / "SKILL.md"
    assert skill.read_text(encoding="utf-8") == texts["SKILL.md"]
    for name in CLAUDE_COMMANDS:
        installed = user / ".claude" / "commands" / "espn-fantasy" / f"{name}.md"
        assert installed.read_text(encoding="utf-8") == texts[f"commands/{name}.md"]
    assert sorted(path.name for path in (user / ".claude").rglob("*") if path.is_file()) == sorted(
        ["SKILL.md", *(f"{name}.md" for name in CLAUDE_COMMANDS)]
    )


def test_a_reinstall_replaces_the_registration_and_leaves_current_files_alone(user: Path) -> None:
    home = windows(user)
    fake = FakeClaude()
    install_claude(home, command(), desktop=False, runner=fake, which=on_path)
    before = snapshot(user)
    fake.calls.clear()
    moved = McpCommand.build(uv_path=UV, repo_root=Path("D:/elsewhere"), environ={})
    report = install_claude(home, moved, desktop=False, runner=fake, which=on_path)
    assert not report.failed, report.lines
    assert [call[2] for call in fake.calls] == ["add-json", "remove", "add-json"]
    assert fake.registered is not None and fake.registered["args"][2] == str(Path("D:/elsewhere"))
    assert any("re-registered" in line for line in report.lines)
    assert snapshot(user) == before
    assert sum("is up to date" in line for line in report.lines) == 1 + len(CLAUDE_COMMANDS)


def test_without_claude_on_path_it_says_so_and_prints_the_command(user: Path) -> None:
    fake = FakeClaude()
    report = install_claude(windows(user), command(), desktop=False, runner=fake, which=nowhere)
    assert fake.calls == [] and not report.failed
    (line,) = [line for line in report.lines if "not on PATH" in line]
    assert "claude mcp add-json --scope user espn-fantasy '{" in line and "fm claude install again" in line
    assert (user / ".claude" / "skills" / "espn-fantasy" / "SKILL.md").is_file()  # ready for when it is installed


def test_a_failed_add_json_is_a_failure(user: Path) -> None:
    report = install_claude(windows(user), command(), desktop=False, runner=FakeClaude(fail_add="boom"), which=on_path)
    assert report.failed and any("add-json failed" in line and "boom" in line for line in report.lines)


def test_a_claude_that_will_not_start_is_a_failure_not_a_crash(user: Path) -> None:
    def broken(argv: Sequence[str]) -> RunResult:
        raise SchedulerError(f"{argv[0]} not found on PATH")

    report = install_claude(windows(user), command(), desktop=False, runner=broken, which=on_path)
    assert report.failed and any("not found on PATH" in line for line in report.lines)


def test_uninstall_removes_only_our_registration_and_files(user: Path) -> None:
    home = windows(user)
    fake = FakeClaude()
    install_claude(home, command(), desktop=False, runner=fake, which=on_path)
    mine = user / ".claude" / "commands" / "espn-fantasy" / "mine.md"
    mine.write_text("the user's own command", encoding="utf-8")
    elsewhere = user / ".claude" / "skills" / "other-skill" / "SKILL.md"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("another skill", encoding="utf-8")
    fake.calls.clear()
    report = uninstall_claude(home, desktop=False, runner=fake, which=on_path)
    assert not report.failed, report.lines
    assert fake.calls == [remove_argv(CLAUDE)] and fake.registered is None
    assert not (user / ".claude" / "skills" / "espn-fantasy").exists()
    assert sorted(path.name for path in (user / ".claude" / "commands" / "espn-fantasy").iterdir()) == ["mine.md"]
    assert elsewhere.read_text(encoding="utf-8") == "another skill"


def test_uninstall_when_nothing_is_installed_is_not_an_error(user: Path) -> None:
    report = uninstall_claude(windows(user), desktop=False, runner=FakeClaude(), which=on_path)
    assert not report.failed and any("was not registered" in line for line in report.lines)
    assert not (user / ".claude").exists()


# --- Claude Desktop ---------------------------------------------------------------------------------------------------


def test_install_writes_the_desktop_entry_beside_an_existing_server(user: Path) -> None:
    config = write_config(appdata_dir(user), DESKTOP_CONFIG)
    original = config.read_bytes()
    report = install_claude(windows(user), command(), code=False, runner=FakeClaude(), which=on_path)
    assert not report.failed, report.lines
    data = read(config)
    assert list(data) == ["mcpServers", "preferences", "globalShortcut"]  # every key kept, in place
    assert data["preferences"] == DESKTOP_CONFIG["preferences"] and data["globalShortcut"] == ""
    assert data["mcpServers"] == {
        "filesystem": OTHER_SERVER,
        "espn-fantasy": {"command": str(UV), "args": SERVER_ARGS},
    }
    assert (config.parent / "claude_desktop_config.json.bak").read_bytes() == original
    assert sorted(path.name for path in config.parent.iterdir()) == [
        "claude_desktop_config.json",
        "claude_desktop_config.json.bak",
    ]  # no temp file left behind
    assert any("reopen Claude Desktop" in line for line in report.lines)


def test_the_desktop_entry_carries_the_pinned_dirs(user: Path, tmp_path: Path) -> None:
    config = write_config(appdata_dir(user), {})
    install_claude(windows(user), command(FM_CONFIG_DIR=str(tmp_path / "cfg")), code=False)
    assert read(config)["mcpServers"]["espn-fantasy"]["env"] == {"FM_CONFIG_DIR": str((tmp_path / "cfg").resolve())}


def test_a_missing_desktop_config_is_created_in_an_existing_folder(user: Path) -> None:
    folder = appdata_dir(user)
    install_claude(windows(user), command(), code=False)
    assert read(folder / "claude_desktop_config.json") == {
        "mcpServers": {"espn-fantasy": {"command": str(UV), "args": SERVER_ARGS}}
    }
    assert not (folder / "claude_desktop_config.json.bak").exists()


def test_the_msix_folder_is_used_when_it_is_the_one_that_exists(user: Path) -> None:
    config = write_config(msix_dir(user), DESKTOP_CONFIG)
    install_claude(windows(user), command(), code=False)
    assert set(read(config)["mcpServers"]) == {"filesystem", "espn-fantasy"}
    assert not (user / "AppData" / "Roaming" / "Claude").exists()


def test_both_desktop_folders_are_updated_when_both_exist(user: Path) -> None:
    roaming = write_config(appdata_dir(user), DESKTOP_CONFIG)
    packaged = write_config(msix_dir(user), {"mcpServers": {}})
    install_claude(windows(user), command(), code=False)
    assert set(read(roaming)["mcpServers"]) == {"filesystem", "espn-fantasy"}
    assert set(read(packaged)["mcpServers"]) == {"espn-fantasy"}


def test_appdata_comes_from_the_environment(user: Path, tmp_path: Path) -> None:
    roaming = tmp_path / "roaming elsewhere"
    (roaming / "Claude").mkdir(parents=True)
    install_claude(windows(user, APPDATA=str(roaming)), command(), code=False)
    assert (roaming / "Claude" / "claude_desktop_config.json").is_file()


def test_no_desktop_folder_means_not_installed_and_nothing_is_created(user: Path) -> None:
    before = snapshot(user)
    report = install_claude(windows(user), command(), code=False)
    assert not report.failed
    assert any("Claude Desktop is not installed" in line for line in report.lines)
    assert snapshot(user) == before and not (user / "AppData").exists()


def test_the_macos_desktop_config_lives_in_application_support(user: Path) -> None:
    folder = user / "Library" / "Application Support" / "Claude"
    folder.mkdir(parents=True)
    install_claude(ClaudeHome(home=user, platform="darwin"), command(), code=False)
    assert set(read(folder / "claude_desktop_config.json")["mcpServers"]) == {"espn-fantasy"}


def test_linux_has_no_claude_desktop(user: Path) -> None:
    report = install_claude(ClaudeHome(home=user, platform="linux"), command(), code=False)
    assert not report.failed and any("Windows and macOS" in line for line in report.lines)


def test_a_desktop_reinstall_that_changes_nothing_writes_nothing(user: Path) -> None:
    config = write_config(appdata_dir(user), DESKTOP_CONFIG)
    install_claude(windows(user), command(), code=False)
    after_first = snapshot(user)
    report = install_claude(windows(user), command(), code=False)
    assert snapshot(user) == after_first  # neither the config nor its .bak was rewritten
    assert any(f"{config} is up to date" in line for line in report.lines)
    assert not any("reopen" in line for line in report.lines)


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("{ not json", "not valid JSON"),
        ("[1, 2]", "JSON object"),
        ('{"mcpServers": ["x"]}', "mcpServers is not a JSON object"),
    ],
)
def test_a_desktop_config_it_cannot_read_is_left_alone(user: Path, text: str, needle: str) -> None:
    config = appdata_dir(user) / "claude_desktop_config.json"
    config.write_text(text, encoding="utf-8")
    report = install_claude(windows(user), command(), code=False)
    assert report.failed and any(needle in line for line in report.lines)
    assert config.read_text(encoding="utf-8") == text
    assert sorted(path.name for path in config.parent.iterdir()) == ["claude_desktop_config.json"]


def test_a_desktop_config_with_a_byte_order_mark_is_read(user: Path) -> None:
    config = appdata_dir(user) / "claude_desktop_config.json"
    config.write_bytes(b"\xef\xbb\xbf" + json.dumps(DESKTOP_CONFIG).encode())
    report = install_claude(windows(user), command(), code=False)
    assert not report.failed and set(read(config)["mcpServers"]) == {"filesystem", "espn-fantasy"}


def test_uninstall_removes_only_our_desktop_entry(user: Path) -> None:
    config = write_config(appdata_dir(user), DESKTOP_CONFIG)
    install_claude(windows(user), command(), code=False)
    report = uninstall_claude(windows(user), code=False)
    assert not report.failed, report.lines
    assert read(config) == DESKTOP_CONFIG
    again = uninstall_claude(windows(user), code=False)
    assert any("is not in" in line for line in again.lines)


# --- dry run ----------------------------------------------------------------------------------------------------------


def test_dry_run_writes_nothing_and_runs_nothing(user: Path) -> None:
    write_config(appdata_dir(user), DESKTOP_CONFIG)
    write_config(msix_dir(user), {})
    before = snapshot(user)
    fake = FakeClaude()
    report = install_claude(windows(user), command(), dry_run=True, runner=fake, which=on_path)
    assert snapshot(user) == before and fake.calls == [] and not report.failed
    assert report.lines[0].startswith("dry run")
    assert any("would run" in line and "add-json" in line for line in report.lines)
    assert sum("would add espn-fantasy" in line for line in report.lines) == 2
    assert sum("would create" in line for line in report.lines) == 1 + len(CLAUDE_COMMANDS)


def test_uninstall_dry_run_removes_nothing(user: Path) -> None:
    write_config(appdata_dir(user), DESKTOP_CONFIG)
    fake = FakeClaude()
    install_claude(windows(user), command(), runner=fake, which=on_path)
    before = snapshot(user)
    fake.calls.clear()
    report = uninstall_claude(windows(user), dry_run=True, runner=fake, which=on_path)
    assert snapshot(user) == before and fake.calls == [] and fake.registered is not None
    assert any("would remove espn-fantasy" in line for line in report.lines)


# --- show -------------------------------------------------------------------------------------------------------------


def test_show_reports_what_is_installed_where(user: Path) -> None:
    config = write_config(appdata_dir(user), DESKTOP_CONFIG)
    home = windows(user)
    status = claude_status(home, command(), which=nowhere)
    assert status.code_entry is None and status.skill_state == "missing"
    assert [entry.entry for entry in status.desktop] == [None]
    install_claude(home, command(), runner=FakeClaude(), which=on_path)
    (user / ".claude.json").write_text(
        json.dumps({"numStartups": 3, "mcpServers": {MCP_SERVER_NAME: command().code_entry()}}), encoding="utf-8"
    )
    status = claude_status(home, command(), which=on_path)
    assert status.claude_cli == CLAUDE and status.code_entry is not None
    assert status.skill_state == "installed" and {state for _, state in status.commands} == {"installed"}
    assert [state.path for state in status.desktop] == [config] and status.desktop[0].entry is not None
    text = "\n".join(status.lines())
    assert "registered, user scope" in text and f"{config}: registered" in text and "differs" not in text
    moved = McpCommand.build(uv_path=UV, repo_root=Path("D:/elsewhere"), environ={})
    assert "differs from this checkout" in "\n".join(claude_status(home, moved, which=on_path).lines())


def test_show_survives_broken_files(user: Path) -> None:
    (appdata_dir(user) / "claude_desktop_config.json").write_text("{", encoding="utf-8")
    (user / ".claude.json").write_text("nope", encoding="utf-8")
    text = "\n".join(claude_status(windows(user), command(), which=nowhere).lines())
    assert "unreadable" in text and "unknown" in text
    linux = "\n".join(claude_status(ClaudeHome(home=user, platform="linux"), command(), which=nowhere).lines())
    assert "not available on this platform" in linux


# --- the command line -------------------------------------------------------------------------------------------------


def use(monkeypatch: pytest.MonkeyPatch, user: Path, fake: FakeClaude, which: Callable[[str], str | None]) -> None:
    monkeypatch.setattr(claude_cmd, "_claude_home", lambda: windows(user))
    monkeypatch.setattr(claude_cmd, "_mcp_command", lambda require_uv=True: command())
    monkeypatch.setattr(claude_cmd, "_runner", fake)
    monkeypatch.setattr(claude_cmd, "_which", which)


def run(*args: str, expect: int = 0) -> str:
    result = runner.invoke(app, list(args), catch_exceptions=False)
    assert result.exit_code == expect, result.output
    return result.output


def test_fm_claude_is_discovered() -> None:
    assert "claude" in run("--help")
    for verb in ("install", "uninstall", "show"):
        assert verb in run("claude", "--help")
    options = run("claude", "install", "--help")
    assert "--no-code" in options and "--no-desktop" in options and "--dry-run" in options


def test_fm_claude_install_and_uninstall(monkeypatch: pytest.MonkeyPatch, user: Path) -> None:
    config = write_config(appdata_dir(user), DESKTOP_CONFIG)
    fake = FakeClaude()
    use(monkeypatch, user, fake, on_path)
    output = run("claude", "install")
    assert output.startswith(f"server: {UV} run --project {REPO} fm mcp")
    assert fake.calls[0][:6] == [CLAUDE, "mcp", "add-json", "--scope", "user", "espn-fantasy"]
    assert set(read(config)["mcpServers"]) == {"filesystem", "espn-fantasy"}
    assert "registered" in run("claude", "show")
    run("claude", "uninstall")
    assert read(config) == DESKTOP_CONFIG and fake.registered is None
    assert not (user / ".claude" / "skills" / "espn-fantasy").exists()


def test_fm_claude_install_dry_run_and_flags(monkeypatch: pytest.MonkeyPatch, user: Path) -> None:
    write_config(appdata_dir(user), DESKTOP_CONFIG)
    fake = FakeClaude()
    use(monkeypatch, user, fake, on_path)
    before = snapshot(user)
    output = run("claude", "install", "--dry-run")
    assert "dry run" in output and snapshot(user) == before and fake.calls == []
    run("claude", "install", "--no-desktop")
    assert read(appdata_dir(user) / "claude_desktop_config.json") == DESKTOP_CONFIG
    assert fake.registered is not None
    run("claude", "install", "--no-code")
    assert "espn-fantasy" in read(appdata_dir(user) / "claude_desktop_config.json")["mcpServers"]


def test_fm_claude_install_exits_1_when_a_part_fails(monkeypatch: pytest.MonkeyPatch, user: Path) -> None:
    use(monkeypatch, user, FakeClaude(fail_add="denied"), on_path)
    assert "denied" in run("claude", "install", "--no-desktop", expect=1)


def test_fm_claude_install_without_uv_says_so(monkeypatch: pytest.MonkeyPatch, user: Path) -> None:
    use(monkeypatch, user, FakeClaude(), on_path)

    def no_uv(require_uv: bool = True) -> McpCommand:
        raise ClaudeError("uv not found on PATH")

    monkeypatch.setattr(claude_cmd, "_mcp_command", no_uv)
    result = runner.invoke(app, ["claude", "install"], catch_exceptions=False)
    assert result.exit_code == 1 and "uv not found" in result.output
    assert not (user / ".claude").exists()


def test_the_cli_hooks_default_to_the_current_user(monkeypatch: pytest.MonkeyPatch, user: Path) -> None:
    """Unpatched, the hooks resolve to this user's home (the guard's temp home here), ``claude`` on PATH and this
    checkout's server command."""
    monkeypatch.setattr(shutil, "which", lambda name: f"/bin/{name}")
    assert claude_cmd._claude_home().home == user
    assert claude_cmd._which("claude") == "/bin/claude"
    built = claude_cmd._mcp_command(require_uv=False)
    assert built.args == ("run", "--project", str(REPO_ROOT), "fm", "mcp")
