"""Connect Claude Code and Claude Desktop to ``fm`` (docs/PLUG_AND_PLAY.md section 8, ROADMAP #55).

Both clients get the same MCP server, ``espn-fantasy``, started as ``<uv> run --project <repo> fm mcp``
(:class:`McpCommand`). A client starts it with its own environment, so ``FM_CONFIG_DIR`` / ``FM_CACHE_DIR`` go into the
entry when the installing shell set them; otherwise the server would read another ``state.db`` than that shell's CLI.

- **Claude Code.** The server is registered through Claude Code's own CLI, ``claude mcp add-json --scope user
  espn-fantasy '<json>'`` (:func:`add_json_argv`), rather than by editing ``~/.claude.json``, whose layout is Claude
  Code's business. A name that is already registered is removed and added again, so a re-install picks up a moved repo
  or a new uv. Without ``claude`` on PATH the install says so and prints the command. The skill and the slash commands
  are packaged files (``fm/claude/assets``, :func:`asset_files`) copied to ``~/.claude/skills/espn-fantasy/SKILL.md``
  and ``~/.claude/commands/espn-fantasy/<name>.md`` (``/espn-fantasy:week`` and so on). An installed file whose text
  is not a version fm shipped (:data:`SHIPPED_ASSET_SHA256`) holds the user's edits: it is copied to
  ``<name>.user.bak`` before an install replaces it.
- **Claude Desktop.** It has no CLI, so ``mcpServers.espn-fantasy`` is merged into ``claude_desktop_config.json``: the
  file is parsed, only that one key changes, every other server and top-level key is kept as it was, the previous file
  is kept as ``claude_desktop_config.json.bak`` (and, the first time fm changes the file, as
  ``claude_desktop_config.json.fm-orig.bak``, which is never replaced) and the new one replaces it atomically. A file
  that is not a JSON object is reported and left alone. Where the file lives is per platform
  (:meth:`ClaudeHome.desktop_dirs`): on Windows both
  ``%APPDATA%\\Claude`` and the MSIX package's ``%LOCALAPPDATA%\\Packages\\Claude_*\\LocalCache\\Roaming\\Claude``
  (a packaged app reads its virtualized copy first, so every folder that exists is updated); on macOS
  ``~/Library/Application Support/Claude``. No folder means Claude Desktop is not installed (or never started), and
  that part is skipped with a note.

Uninstall undoes exactly that: ``claude mcp remove --scope user espn-fantasy``, the skill and command files this
installs (by name, and only while they hold a version fm shipped: an edited one is left and reported; the folders go
only when nothing else is left in them) and the one ``mcpServers`` key.

Every effect goes through a parameter: the user's folders (:class:`ClaudeHome`), the command runner (the scheduler
backends' :data:`~fm.jobs.scheduler_base.Runner`), the ``claude`` lookup and the uv path, so tests run against a temp
home and a fake runner and never read the real ``~/.claude`` or Desktop config. Nothing here touches ESPN or the
store: the server it registers is the read-only one in :mod:`fm.mcp_server`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fm import paths
from fm.jobs.scheduler_base import REPO_ROOT, Runner, RunResult, SchedulerError, default_runner, find_uv

MCP_SERVER_NAME = "espn-fantasy"
"""The server's name in both clients; ``fm.mcp_server.SERVER_NAME`` is the same string (a test pins it)."""
CLAUDE_SCOPE = "user"
"""Claude Code's user scope: the server is available in every project, not only in this repo."""
SKILL_DIRNAME = "espn-fantasy"
"""The skill's folder under ``~/.claude/skills`` and the slash commands' namespace under ``~/.claude/commands``."""
SKILL_FILENAME = "SKILL.md"
CLAUDE_COMMANDS = ("week", "lineup", "waivers", "trade")
"""The slash commands, one ``<name>.md`` each: ``/espn-fantasy:week`` and so on."""
SHIPPED_ASSET_SHA256: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        SKILL_FILENAME: frozenset({"8e4d477e599dd15e5871974a8680c67e9a7b0642ee8f68c6fa346f9def904691"}),
        "commands/week.md": frozenset({"d8cc3d2afd61f0a94c463b45323d4099cb4970ea1f62fb2ce28b89065ebd0418"}),
        "commands/lineup.md": frozenset({"cba4051a8251546a5aa69027e0ea437880b806eb091c57c1e4a1ca5c90b64d5d"}),
        "commands/waivers.md": frozenset({"b78e9707a86f1507470aa020dc439bc657123ec638e6e5bfa8763e33089cfc58"}),
        "commands/trade.md": frozenset({"7d83a68bf589a232a33cf4d4fd53652bfcea549a8439fba4811dc5d05664711c"}),
    }
)
"""The SHA-256 of every version of each packaged asset fm has ever shipped (its text as ``read_text`` gives it, so line
endings do not matter), by its path under ``assets``. An installed file whose text is one of these is fm's own: an
install may overwrite it and an uninstall may delete it. Anything else is the user's edit, which an install backs up
first (``<file>.user.bak``) and an uninstall leaves in place. When an asset changes, add the new digest and keep the
old ones (``tests/test_claude_install.py`` fails until the current text is listed)."""
USER_BACKUP_SUFFIX = ".user.bak"
"""Where an install keeps a skill or command file the user edited before replacing it: ``SKILL.md.user.bak``."""
DESKTOP_CONFIG_NAME = "claude_desktop_config.json"
DESKTOP_ORIGINAL_SUFFIX = ".fm-orig.bak"
"""``claude_desktop_config.json.fm-orig.bak``: the Desktop config as it was before fm first changed it, written once
and never replaced (``.bak`` beside it is the rolling copy of the file before the latest change)."""
MSIX_PACKAGE_GLOB = "Claude_*"
"""The Claude Desktop MSIX package folder under ``%LOCALAPPDATA%\\Packages`` (the suffix is the publisher hash)."""
DESKTOP_PLATFORMS = ("win32", "darwin")
_REPLACE_RETRY_S = 0.2
"""Windows refuses ``os.replace`` onto a file another process holds open for a moment; one retry after this."""

_ALREADY_EXISTS = re.compile(r"already exists", re.IGNORECASE)
_NOT_FOUND = re.compile(r"not found|no .*server|does not exist|doesn't exist", re.IGNORECASE)
_INSTALLED = "installed"
_SAFE_ARG = re.compile(r"[A-Za-z0-9_@%+=:,./\\-]+")


class ClaudeError(RuntimeError):
    """The install cannot go ahead: uv is missing, or a Claude config file is not what it should be."""


# --- where things go --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClaudeHome:
    """One user's Claude folders. Every path the installer reads or writes comes from here, so a test hands it a temp
    dir and the real ``~/.claude`` and Claude Desktop config are never touched.

    ``environ`` supplies ``APPDATA`` / ``LOCALAPPDATA`` on Windows (falling back to the usual folders under ``home``).
    """

    home: Path
    platform: str = sys.platform
    environ: Mapping[str, str] = field(default_factory=dict[str, str])

    @classmethod
    def current(cls) -> ClaudeHome:
        """This user on this machine."""
        return cls(home=Path.home(), platform=sys.platform, environ=dict(os.environ))

    @property
    def code_dir(self) -> Path:
        """Claude Code's user folder, ``~/.claude``."""
        return self.home / ".claude"

    @property
    def code_config(self) -> Path:
        """``~/.claude.json``, where Claude Code keeps user-scope MCP servers. Only :func:`claude_status` reads it."""
        return self.home / ".claude.json"

    @property
    def skill_path(self) -> Path:
        return self.code_dir / "skills" / SKILL_DIRNAME / SKILL_FILENAME

    @property
    def commands_dir(self) -> Path:
        return self.code_dir / "commands" / SKILL_DIRNAME

    def command_path(self, name: str) -> Path:
        return self.commands_dir / f"{name}.md"

    def code_files(self) -> list[tuple[str, Path]]:
        """Each packaged asset (its path under ``assets``) and where it is installed."""
        return [
            (SKILL_FILENAME, self.skill_path),
            *((f"commands/{name}.md", self.command_path(name)) for name in CLAUDE_COMMANDS),
        ]

    @property
    def desktop_supported(self) -> bool:
        return self.platform in DESKTOP_PLATFORMS

    def desktop_dirs(self) -> list[Path]:
        """Every folder Claude Desktop may keep its config in on this platform, whether or not it exists."""
        if self.platform == "win32":
            roaming = self._env_dir("APPDATA", self.home / "AppData" / "Roaming")
            local = self._env_dir("LOCALAPPDATA", self.home / "AppData" / "Local")
            packages = local / "Packages"
            msix = sorted(packages.glob(MSIX_PACKAGE_GLOB)) if packages.is_dir() else []
            return [roaming / "Claude", *(package / "LocalCache" / "Roaming" / "Claude" for package in msix)]
        if self.platform == "darwin":
            return [self.home / "Library" / "Application Support" / "Claude"]
        return []

    def desktop_configs(self) -> list[Path]:
        """``claude_desktop_config.json`` in every Desktop folder that exists (the file itself may not yet)."""
        return [folder / DESKTOP_CONFIG_NAME for folder in self.desktop_dirs() if folder.is_dir()]

    def _env_dir(self, name: str, default: Path) -> Path:
        value = self.environ.get(name)
        return Path(value) if value else default


# --- what gets registered ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class McpCommand:
    """How a client starts the server: ``<uv> run --project <repo> fm mcp`` with ``env``.

    ``uv`` is embedded absolute because Claude Desktop starts servers with a minimal PATH (a GUI app on macOS does not
    see ``~/.local/bin``).
    """

    command: str
    args: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()

    @classmethod
    def build(
        cls,
        *,
        uv_path: Path | None = None,
        repo_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
        require_uv: bool = True,
    ) -> McpCommand:
        """The command for this checkout: uv from PATH (as ``fm schedule`` finds it), the repo this package runs from,
        and ``FM_CONFIG_DIR`` / ``FM_CACHE_DIR`` from ``environ`` (default ``os.environ``) when set there.
        ``require_uv`` false lets ``show`` describe the command without uv; it then says plain ``uv``."""
        uv = uv_path if uv_path is not None else find_uv()
        if uv is None:
            if require_uv:
                raise ClaudeError("uv not found on PATH; install uv (https://docs.astral.sh/uv/) and run this again")
            uv = Path("uv")
        env = os.environ if environ is None else environ
        pinned: list[tuple[str, str]] = []
        for name in (paths.CONFIG_DIR_ENV, paths.CACHE_DIR_ENV):
            value = env.get(name)
            if value:
                pinned.append((name, str(_absolute(value))))
        root = repo_root if repo_root is not None else REPO_ROOT
        return cls(command=str(uv), args=("run", "--project", str(root), "fm", "mcp"), env=tuple(pinned))

    def desktop_entry(self) -> dict[str, Any]:
        """The ``mcpServers`` entry for ``claude_desktop_config.json``; ``env`` only when something is pinned."""
        entry: dict[str, Any] = {"command": self.command, "args": list(self.args)}
        if self.env:
            entry["env"] = dict(self.env)
        return entry

    def code_entry(self) -> dict[str, Any]:
        """The JSON ``claude mcp add-json`` takes: a stdio server."""
        return {"type": "stdio", **self.desktop_entry()}

    def display(self) -> str:
        line = " ".join([self.command, *self.args])
        return line + "".join(f" ({name}={value})" for name, value in self.env)


def _absolute(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def add_json_argv(command: McpCommand, claude: str = "claude") -> list[str]:
    """``claude mcp add-json --scope user espn-fantasy '<json>'``."""
    payload = json.dumps(command.code_entry(), separators=(",", ":"))
    return [claude, "mcp", "add-json", "--scope", CLAUDE_SCOPE, MCP_SERVER_NAME, payload]


def remove_argv(claude: str = "claude") -> list[str]:
    """``claude mcp remove --scope user espn-fantasy``."""
    return [claude, "mcp", "remove", "--scope", CLAUDE_SCOPE, MCP_SERVER_NAME]


def shell_command(argv: Sequence[str], platform: str = sys.platform) -> str:
    """``argv`` as one line to paste into a shell: POSIX quoting, or single quotes PowerShell reads literally on
    Windows (where ``'`` is escaped by doubling it)."""
    if platform != "win32":
        return shlex.join(argv)
    return " ".join(arg if _SAFE_ARG.fullmatch(arg) else "'" + arg.replace("'", "''") + "'" for arg in argv)


# --- the packaged skill and commands ----------------------------------------------------------------------------------


def assets() -> Traversable:
    """``fm/claude/assets``, read through :mod:`importlib.resources` so it works from a wheel as from the repo."""
    return resources.files("fm.claude") / "assets"


def asset_files() -> dict[str, str]:
    """Every packaged asset's text by its path under ``assets`` (``SKILL.md``, ``commands/week.md``, ...)."""
    root = assets()
    found: dict[str, str] = {}
    for relative in (SKILL_FILENAME, *(f"commands/{name}.md" for name in CLAUDE_COMMANDS)):
        node = root
        for part in relative.split("/"):
            node = node / part
        found[relative] = node.read_text(encoding="utf-8")
    return found


# --- reports ----------------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class ClaudeReport:
    """What an install or uninstall did, one line per effect. ``failed`` is set when a part could not be done (the
    other parts still ran)."""

    lines: list[str] = field(default_factory=list[str])
    failed: bool = False

    def add(self, line: str) -> None:
        self.lines.append(line)

    def fail(self, line: str) -> None:
        self.lines.append(f"error: {line}")
        self.failed = True


# --- install ----------------------------------------------------------------------------------------------------------


def install_claude(
    home: ClaudeHome,
    command: McpCommand,
    *,
    code: bool = True,
    desktop: bool = True,
    dry_run: bool = False,
    runner: Runner = default_runner,
    which: Callable[[str], str | None] = shutil.which,
) -> ClaudeReport:
    """Register the server, the skill and the slash commands with Claude Code (``code``) and the server with Claude
    Desktop (``desktop``). ``dry_run`` reports what would happen and changes nothing, not even through ``claude``."""
    report = ClaudeReport()
    if dry_run:
        report.add("dry run: nothing is written or registered")
    if not code and not desktop:
        report.add("nothing to do: both Claude Code and Claude Desktop were left out")
    if code:
        _install_code(report, home, command, dry_run=dry_run, runner=runner, which=which)
    if desktop:
        changed = False
        for path in _desktop_targets(report, home):
            try:
                line, changed_here = _merge_desktop(path, command.desktop_entry(), dry_run=dry_run)
            except ClaudeError as exc:
                report.fail(str(exc))
                continue
            report.add(line)
            changed = changed or changed_here
        if changed and not dry_run:
            report.add("claude desktop: quit and reopen Claude Desktop to load the server")
    return report


def _install_code(
    report: ClaudeReport,
    home: ClaudeHome,
    command: McpCommand,
    *,
    dry_run: bool,
    runner: Runner,
    which: Callable[[str], str | None],
) -> None:
    claude = which("claude")
    argv = add_json_argv(command, claude or "claude")
    if dry_run:
        report.add(f"claude code: would run {shell_command(argv, home.platform)}")
    elif claude is None:
        report.add(
            "claude code: claude is not on PATH, so the server is not registered. Install Claude Code and run "
            f"fm claude install again, or run: {shell_command(argv, home.platform)}"
        )
    else:
        _register(report, runner, argv, claude)
    texts = asset_files()
    for relative, target in home.code_files():
        text = texts[relative]
        if _file_state(target, text) == _INSTALLED:
            report.add(f"claude code: {target} is up to date")
            continue
        existed = target.exists()
        edited = existed and not _shipped(relative, target)
        if dry_run:
            kept = f" (keeping your edited copy as {_user_backup(target).name})" if edited else ""
            report.add(f"claude code: would {'update' if existed else 'create'} {target}{kept}")
            continue
        try:
            if edited:
                backup = _user_backup(target)
                shutil.copyfile(target, backup)
                report.add(
                    f"claude code: {target} has edits fm did not make; kept them as {backup.name} before updating it"
                )
            _replace(target, text)
        except OSError as exc:
            report.fail(f"claude code: {target}: {exc.strerror or exc}")
            continue
        report.add(f"claude code: {'updated' if existed else 'created'} {target}")


def _register(report: ClaudeReport, runner: Runner, argv: list[str], claude: str) -> None:
    """Run ``add-json``; when the name is already taken, remove it and add again (so a re-install refreshes it)."""
    result = _run(runner, argv)
    replaced = False
    if not result.ok and _ALREADY_EXISTS.search(result.output):
        removed = _run(runner, remove_argv(claude))
        if not removed.ok:
            report.fail(f"claude code: could not replace the registered server: {removed.output or 'no output'}")
            return
        result = _run(runner, argv)
        replaced = True
    if not result.ok:
        output = result.output or "no output"
        report.fail(f"claude code: claude mcp add-json failed (exit {result.returncode}): {output}")
    else:
        verb = "re-registered" if replaced else "registered"
        report.add(f"claude code: {verb} MCP server {MCP_SERVER_NAME} (user scope)")


def _run(runner: Runner, argv: Sequence[str]) -> RunResult:
    """``runner(argv)``, with a command that could not start at all (:class:`SchedulerError`) as a failed result."""
    try:
        return runner(argv)
    except SchedulerError as exc:
        return RunResult(argv=tuple(argv), returncode=-1, stderr=str(exc))


def _desktop_targets(report: ClaudeReport, home: ClaudeHome) -> list[Path]:
    if not home.desktop_supported:
        report.add(f"claude desktop: skipped, Claude Desktop runs on Windows and macOS (this is {home.platform})")
        return []
    targets = home.desktop_configs()
    if not targets:
        looked = ", ".join(str(folder) for folder in home.desktop_dirs())
        report.add(
            f"claude desktop: skipped, Claude Desktop is not installed (no config folder at {looked}); "
            "start it once, then run fm claude install again"
        )
    return targets


def _merge_desktop(path: Path, entry: dict[str, Any], *, dry_run: bool) -> tuple[str, bool]:
    """Set ``mcpServers.espn-fantasy`` in ``path`` to ``entry``; returns the report line and whether the file changed.
    Every other key keeps its value and its place (``mcpServers`` is added last when the file has none)."""
    data = _read_json_object(path)
    servers = _servers(data, path)
    if servers.get(MCP_SERVER_NAME) == entry:
        return f"claude desktop: {path} is up to date", False
    present = MCP_SERVER_NAME in servers
    where = f"{MCP_SERVER_NAME} {'in' if present else 'to'} {path}"
    if dry_run:
        return f"claude desktop: would {'update' if present else 'add'} {where}", False
    data["mcpServers"] = {**servers, MCP_SERVER_NAME: entry}
    kept = _kept(*_write_json(path, data))
    return f"claude desktop: {'updated' if present else 'added'} {where}{kept}", True


# --- uninstall --------------------------------------------------------------------------------------------------------


def uninstall_claude(
    home: ClaudeHome,
    *,
    code: bool = True,
    desktop: bool = True,
    dry_run: bool = False,
    runner: Runner = default_runner,
    which: Callable[[str], str | None] = shutil.which,
) -> ClaudeReport:
    """Remove what :func:`install_claude` adds and nothing else: our ``claude mcp`` registration, our skill and
    command files, our ``mcpServers`` key. Anything already gone is a line, not an error."""
    report = ClaudeReport()
    if dry_run:
        report.add("dry run: nothing is removed")
    if code:
        _uninstall_code(report, home, dry_run=dry_run, runner=runner, which=which)
    if desktop and home.desktop_supported:
        configs = home.desktop_configs()
        if not configs:
            report.add("claude desktop: not installed; nothing to remove")
        for path in configs:
            try:
                report.add(_unmerge_desktop(path, dry_run=dry_run))
            except ClaudeError as exc:
                report.fail(str(exc))
    return report


def _uninstall_code(
    report: ClaudeReport,
    home: ClaudeHome,
    *,
    dry_run: bool,
    runner: Runner,
    which: Callable[[str], str | None],
) -> None:
    claude = which("claude")
    argv = remove_argv(claude or "claude")
    if dry_run:
        report.add(f"claude code: would run {shell_command(argv, home.platform)}")
    elif claude is None:
        manual = shell_command(argv, home.platform)
        report.add(f"claude code: claude is not on PATH; if the server is registered, run: {manual}")
    else:
        result = _run(runner, argv)
        if result.ok:
            report.add(f"claude code: removed MCP server {MCP_SERVER_NAME} (user scope)")
        elif _NOT_FOUND.search(result.output):
            report.add(f"claude code: MCP server {MCP_SERVER_NAME} was not registered")
        else:
            report.fail(f"claude code: claude mcp remove failed (exit {result.returncode}): {result.output}")
    for relative, target in home.code_files():
        if not target.exists():
            continue
        if not _shipped(relative, target):
            report.add(
                f"claude code: left {target} in place: it is not a version fm shipped (your edits?); delete it "
                "yourself if you no longer want it"
            )
            continue
        if dry_run:
            report.add(f"claude code: would remove {target}")
            continue
        try:
            target.unlink()
        except OSError as exc:
            report.fail(f"claude code: {target}: {exc.strerror or exc}")
            continue
        report.add(f"claude code: removed {target}")
    if not dry_run:
        for folder in (home.skill_path.parent, home.commands_dir):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()


def _unmerge_desktop(path: Path, *, dry_run: bool) -> str:
    if not path.exists():
        return f"claude desktop: {path} does not exist; nothing to remove"
    data = _read_json_object(path)
    servers = _servers(data, path)
    if MCP_SERVER_NAME not in servers:
        return f"claude desktop: {MCP_SERVER_NAME} is not in {path}"
    if dry_run:
        return f"claude desktop: would remove {MCP_SERVER_NAME} from {path}"
    data["mcpServers"] = {name: value for name, value in servers.items() if name != MCP_SERVER_NAME}
    kept = _kept(*_write_json(path, data))
    return f"claude desktop: removed {MCP_SERVER_NAME} from {path}{kept}"


# --- show -------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DesktopState:
    """One Claude Desktop config file: our entry in it (``None`` when absent) or why it could not be read."""

    path: Path
    exists: bool
    entry: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ClaudeStatus:
    """What is installed where, read from the files alone (``claude`` is not run, so ``show`` is quick and offline)."""

    claude_cli: str | None
    code_config: Path
    code_entry: dict[str, Any] | None
    code_error: str | None
    skill: Path
    skill_state: str
    commands: tuple[tuple[str, str], ...]
    desktop_supported: bool
    desktop_dirs: tuple[Path, ...]
    desktop: tuple[DesktopState, ...]
    expected: McpCommand

    def lines(self) -> list[str]:
        """The report ``fm claude show`` prints."""
        out = ["Claude Code"]
        out.append(f"  claude CLI:   {self.claude_cli or 'not on PATH'}")
        if self.code_error is not None:
            out.append(f"  MCP server:   unknown ({self.code_error})")
        elif self.code_entry is None:
            out.append(f"  MCP server:   not registered (user scope, {self.code_config})")
        else:
            out.append(f"  MCP server:   registered, user scope: {_entry_line(self.code_entry, self.expected)}")
        out.append(f"  Skill:        {self.skill} ({self.skill_state})")
        for name, state in self.commands:
            out.append(f"  Command:      /{SKILL_DIRNAME}:{name} ({state})")
        out.append("Claude Desktop")
        if not self.desktop_supported:
            out.append("  not available on this platform")
        elif not self.desktop:
            looked = ", ".join(str(folder) for folder in self.desktop_dirs)
            out.append(f"  not installed (no config folder at {looked})")
        for state in self.desktop:
            if state.error is not None:
                verdict = f"unreadable: {state.error}"
            elif not state.exists:
                verdict = "no config file yet: not registered"
            elif state.entry is None:
                verdict = "not registered"
            else:
                verdict = f"registered: {_entry_line(state.entry, self.expected)}"
            out.append(f"  {state.path}: {verdict}")
        out.append(f"This checkout serves: {self.expected.display()}")
        return out


def claude_status(
    home: ClaudeHome, expected: McpCommand, *, which: Callable[[str], str | None] = shutil.which
) -> ClaudeStatus:
    """Read what is installed. Never raises for a missing or broken file: that is part of the answer."""
    code_entry: dict[str, Any] | None = None
    code_error: str | None = None
    if home.code_config.is_file():
        try:
            code_entry = _our_entry(_read_json_object(home.code_config), home.code_config)
        except ClaudeError as exc:
            code_error = str(exc)
    texts = asset_files()
    commands = tuple(
        (name, _file_state(home.command_path(name), texts[f"commands/{name}.md"])) for name in CLAUDE_COMMANDS
    )
    desktop: list[DesktopState] = []
    if home.desktop_supported:
        for path in home.desktop_configs():
            if not path.exists():
                desktop.append(DesktopState(path, exists=False))
                continue
            try:
                desktop.append(DesktopState(path, exists=True, entry=_our_entry(_read_json_object(path), path)))
            except ClaudeError as exc:
                desktop.append(DesktopState(path, exists=True, error=str(exc)))
    return ClaudeStatus(
        claude_cli=which("claude"),
        code_config=home.code_config,
        code_entry=code_entry,
        code_error=code_error,
        skill=home.skill_path,
        skill_state=_file_state(home.skill_path, texts[SKILL_FILENAME]),
        commands=commands,
        desktop_supported=home.desktop_supported,
        desktop_dirs=tuple(home.desktop_dirs()),
        desktop=tuple(desktop),
        expected=expected,
    )


def _file_state(path: Path, text: str) -> str:
    if not path.is_file():
        return "missing"
    try:
        same = path.read_text(encoding="utf-8") == text
    except (OSError, UnicodeDecodeError):
        return "unreadable"
    return _INSTALLED if same else "installed, differs from this version (run fm claude install)"


def asset_digest(text: str) -> str:
    """The SHA-256 :data:`SHIPPED_ASSET_SHA256` lists for an asset's text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shipped(relative: str, path: Path) -> bool:
    """Whether the file at ``path`` holds a version of the asset ``relative`` that fm shipped (so it is fm's to
    replace or remove). A file that cannot be read as text is treated as the user's."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return asset_digest(text) in SHIPPED_ASSET_SHA256.get(relative, frozenset())


def _user_backup(path: Path) -> Path:
    """Where to keep the user's edited ``path``: ``<name>.user.bak``, or, when that already holds something else (an
    earlier edit), the first free ``<name>.user.<n>.bak``, so no earlier backup is overwritten."""
    candidate = path.with_name(path.name + USER_BACKUP_SUFFIX)
    number = 2
    while candidate.exists() and not _same_bytes(candidate, path):
        candidate = path.with_name(f"{path.name}.user.{number}.bak")
        number += 1
    return candidate


def _same_bytes(first: Path, second: Path) -> bool:
    try:
        return first.read_bytes() == second.read_bytes()
    except OSError:
        return False


def _our_entry(data: dict[str, Any], path: Path) -> dict[str, Any] | None:
    entry = _servers(data, path).get(MCP_SERVER_NAME)
    return entry if isinstance(entry, dict) else None


def _entry_line(entry: Mapping[str, Any], expected: McpCommand) -> str:
    args = entry.get("args")
    words = [str(entry.get("command", "?")), *(str(arg) for arg in args)] if isinstance(args, list) else ["?"]
    same = {key: entry.get(key) for key in ("command", "args", "env")} == {
        key: expected.desktop_entry().get(key) for key in ("command", "args", "env")
    }
    return " ".join(words) + ("" if same else " (differs from this checkout: run fm claude install)")


# --- JSON files -------------------------------------------------------------------------------------------------------


def _read_json_object(path: Path) -> dict[str, Any]:
    """The file's JSON object, ``{}`` for a missing or empty file. Anything else is a :class:`ClaudeError`: a file this
    cannot parse is never rewritten."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        raise ClaudeError(f"{path}: cannot read it: {exc}") from None
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClaudeError(f"{path}: not valid JSON ({exc}); fix or remove it, then run this again") from None
    if not isinstance(data, dict):
        raise ClaudeError(f"{path}: expected a JSON object at the top level; left unchanged")
    return data


def _servers(data: Mapping[str, Any], path: Path) -> dict[str, Any]:
    servers = data.get("mcpServers")
    if servers is None:
        return {}
    if not isinstance(servers, dict):
        raise ClaudeError(f"{path}: mcpServers is not a JSON object; left unchanged")
    return dict(servers)


def _write_json(path: Path, data: Mapping[str, Any]) -> tuple[Path | None, Path | None]:
    """Replace ``path`` with ``data`` atomically, keeping the previous file as ``<name>.bak`` and, the first time fm
    changes it, as ``<name>.fm-orig.bak`` too (never replaced afterwards, so the file as it was before fm is always
    there to go back to, however many installs ran since). Returns the backup's path and the original's when it was
    written now (``None`` when there was no previous file, or the original was kept already)."""
    if not path.exists():
        backup = original = None
    else:
        backup = path.with_name(path.name + ".bak")
        original = path.with_name(path.name + DESKTOP_ORIGINAL_SUFFIX)
        if original.exists():
            original = None
    try:
        _replace(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n", backup=backup, original=original)
    except OSError as exc:
        raise ClaudeError(f"{path}: {exc.strerror or exc}") from exc
    return backup, original


def _kept(backup: Path | None, original: Path | None) -> str:
    """What a Desktop report line says about the copies :func:`_write_json` kept."""
    if backup is None:
        return ""
    if original is None:
        return f" (previous file kept as {backup.name})"
    return f" (previous file kept as {backup.name}, and as {original.name}, the copy from before fm changed it)"


def _replace(path: Path, text: str, *, backup: Path | None = None, original: Path | None = None) -> None:
    """Write ``text`` to a temp file beside ``path``, copy the current file to ``backup`` and ``original`` (each when
    given), then move the temp file over ``path``. The temp file is removed on any failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(text.encode("utf-8"))
        if original is not None:
            shutil.copyfile(path, original)
        if backup is not None:
            shutil.copyfile(path, backup)
        try:
            os.replace(temp, path)
        except PermissionError:  # Windows: another process has the target open for a moment
            time.sleep(_REPLACE_RETRY_S)
            os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)
        raise
