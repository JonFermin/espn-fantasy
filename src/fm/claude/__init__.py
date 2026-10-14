"""Claude integration (docs/DESIGN.md section 12.5, ROADMAP #55): ``fm claude install|uninstall|show``.

:mod:`fm.claude.install` registers the read-only MCP server (:mod:`fm.mcp_server`) with Claude Code and Claude
Desktop and installs a skill and slash commands for Claude Code. The skill and commands are packaged files under
``fm/claude/assets`` (``SKILL.md`` and ``commands/<name>.md``), read through :mod:`importlib.resources`.

Like the MCP server, nothing here has a write path to ESPN: it edits the user's Claude configuration and nothing else.
"""

from fm.claude.install import (
    CLAUDE_COMMANDS,
    CLAUDE_SCOPE,
    DESKTOP_CONFIG_NAME,
    DESKTOP_ORIGINAL_SUFFIX,
    MCP_SERVER_NAME,
    SHIPPED_ASSET_SHA256,
    SKILL_DIRNAME,
    SKILL_FILENAME,
    USER_BACKUP_SUFFIX,
    ClaudeError,
    ClaudeHome,
    ClaudeReport,
    ClaudeStatus,
    DesktopState,
    McpCommand,
    add_json_argv,
    asset_digest,
    asset_files,
    assets,
    claude_status,
    install_claude,
    remove_argv,
    shell_command,
    uninstall_claude,
)

__all__ = [
    "CLAUDE_COMMANDS",
    "CLAUDE_SCOPE",
    "DESKTOP_CONFIG_NAME",
    "DESKTOP_ORIGINAL_SUFFIX",
    "MCP_SERVER_NAME",
    "SHIPPED_ASSET_SHA256",
    "SKILL_DIRNAME",
    "SKILL_FILENAME",
    "USER_BACKUP_SUFFIX",
    "ClaudeError",
    "ClaudeHome",
    "ClaudeReport",
    "ClaudeStatus",
    "DesktopState",
    "McpCommand",
    "add_json_argv",
    "asset_digest",
    "asset_files",
    "assets",
    "claude_status",
    "install_claude",
    "remove_argv",
    "shell_command",
    "uninstall_claude",
]
