"""Config and cache directories: env overrides, defaults, creation on demand."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fm import paths


def test_conftest_points_dirs_at_temp(tmp_path: Path) -> None:
    assert paths.config_dir() == tmp_path / "config"
    assert paths.cache_dir() == tmp_path / "cache"


def test_dirs_are_created_on_demand(tmp_path: Path) -> None:
    assert not (tmp_path / "config").exists()
    assert paths.config_dir().is_dir()
    assert not (tmp_path / "cache").exists()
    assert paths.cache_dir().is_dir()


def test_defaults_live_under_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FM_CONFIG_DIR")
    monkeypatch.delenv("FM_CACHE_DIR")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    assert paths.config_dir() == tmp_path / "home" / ".config" / "espn-fantasy"
    assert paths.cache_dir() == tmp_path / "home" / ".cache" / "espn-fantasy"
    assert paths.config_dir().is_dir() and paths.cache_dir().is_dir()


def test_override_expands_user_and_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # ``~`` expands through USERPROFILE on Windows and HOME elsewhere; keep both away from the real home.
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FM_CONFIG_DIR", os.path.join("~", "cfg"))
    monkeypatch.setenv("FM_TEST_BASE", str(tmp_path))
    monkeypatch.setenv("FM_CACHE_DIR", os.path.join("${FM_TEST_BASE}", "cc"))
    assert paths.config_dir() == tmp_path / "home" / "cfg"
    assert paths.cache_dir() == tmp_path / "cc"


def test_named_paths_hang_off_config_dir(tmp_path: Path) -> None:
    config = tmp_path / "config"
    assert paths.config_file() == config / "config.toml"
    assert paths.env_file() == config / ".env"
    assert paths.state_db() == config / "state.db"
    assert paths.browser_profile_dir() == config / "browser-profile"
    assert paths.audit_dir() == config / "audit"
    assert paths.audit_dir().is_dir()
