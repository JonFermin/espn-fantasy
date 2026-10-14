"""Docs that show configuration must show configuration that loads.

DESIGN section 14 tells a reader its ``config.toml`` template "is the same file by hand", so a copy of it has to pass
``load_config`` exactly as printed: placeholder ids included.
"""

from __future__ import annotations

import re
from pathlib import Path

from fm.config import load_config

DESIGN = Path(__file__).resolve().parents[1] / "docs" / "DESIGN.md"
TEMPLATE = re.compile(r"```toml\n(# ~/\.config/espn-fantasy/config\.toml\n.*?)```", re.DOTALL)


def test_design_config_template_loads(tmp_path: Path) -> None:
    match = TEMPLATE.search(DESIGN.read_text(encoding="utf-8"))
    assert match is not None, "docs/DESIGN.md lost its config.toml template (a ```toml block headed by its path)"
    path = tmp_path / "config.toml"
    path.write_text(match.group(1), encoding="utf-8")
    config = load_config(path, env_path=tmp_path / ".env", environ={})
    assert [league.key for league in config.leagues] == ["nfl", "nba"]
    assert all(league.preset == "balanced" for league in config.leagues)
