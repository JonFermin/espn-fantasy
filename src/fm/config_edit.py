"""The one writer of ``config.toml`` (PLUG_AND_PLAY section 1): ``fm setup``, ``fm config set-preset`` and the
dashboard all edit the file through here, and ``fm.config`` only ever reads it.

tomlkit keeps comments, key order and formatting, so a hand-edited file survives a programmatic edit, and a file
written with CRLF line endings keeps them. Every write follows the same rules: the new text is validated with
``fm.config.parse_config`` first (an invalid edit raises ``ConfigError`` and leaves the file untouched, byte for byte),
the replace is atomic (a temp file beside the target plus ``os.replace``), and the previous contents are kept as
``config.toml.bak``. An edit that changes nothing writes nothing, so a re-run of ``fm setup`` leaves the file alone.

League entries are TOML-shaped mappings, one per ``[[league]]`` table (``key``, ``sport``, ``espn_league_id``,
``season``, ``team_id``, an optional ``preset`` and optional ``policy`` / ``trade_finder`` sub-mappings). A rendered
file writes only the keys it is given: a league with a preset and no policy table lets the preset decide, which is
what keeps ``fm config set-preset`` meaningful later, since a key written in the file always wins over the preset.
Secrets never pass through here; ``.env`` is read only to validate.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit import TOMLDocument
from tomlkit.exceptions import ParseError
from tomlkit.items import AoT, Item, Table

from fm.config import Config, ConfigError, Llm, Notify, config_paths, parse_config

COMMENT_COLUMN = 30
"""Inline comments start here when the key and value leave room, matching the DESIGN.md template."""

_HEADER = (
    "espn-fantasy configuration: the leagues this tool manages and how much it may do on its own.",
    "fm config check validates it; fm config set-preset <league> <preset> changes a preset; fm setup adds leagues.",
    "A preset fills every [league.policy] and [league.trade_finder] key a league leaves unset; a written key wins.",
    "Secrets never go in this file: API keys, bot tokens and ntfy topics live in .env next to it.",
    "Trades are approval-only and have no setting anywhere in this file.",
)

# One line per known key, written as an inline comment. Keys are unique across the tables they belong to.
_COMMENTS: Mapping[str, str] = {
    "key": "short name used on the command line (fm sync --league <key>)",
    "sport": "nfl | nba",
    "espn_league_id": "leagueId=... in the league URL",
    "season": "ESPN labels NBA seasons by their end year",
    "team_id": "teamId=... on your team page",
    "preset": "hands_off | balanced | ask_first",
    "bench_inactive": "off | approve | auto (auto fires at T-15 when unanswered)",
    "lineup": "off | approve | auto",
    "add": "off | approve | auto (auto needs auto_add_min_gain)",
    "auto_add_min_gain": "least engine gain, in league points, for add = auto",
    "add_drop": "off | approve (never auto)",
    "waiver": "off | approve (never auto)",
    "max_transactions_per_week": "our cap; ESPN's own limit is read from settings",
    "max_faab_pct_per_bid": "largest single bid as a fraction of the season FAAB budget",
    "untouchables": "player names or ESPN player IDs; resolved at sync",
    "enabled": "drafts outgoing offers for your approval; never sends one",
    "min_accept": "least P(accept) for a deal to be drafted",
    "min_title_gain": "least gain in title odds (0.01 = 1 point)",
    "min_ros_gain": "without a simulation: least gain in starter seasons",
    "max_offers": "drafts a run; the weekly cap of 3 still holds",
    "model": "Claude model for the advisor workers",
    "daily_budget_usd": "daily spend cap across all workers",
    "channel": "telegram | ntfy; the credentials live in .env",
}


def backup_path(path: Path) -> Path:
    """Where a write keeps the previous file: ``config.toml.bak`` beside it."""
    return path.with_name(path.name + ".bak")


def render_config(
    leagues: Iterable[Mapping[str, Any]],
    *,
    llm: Mapping[str, Any] | None = None,
    notify: Mapping[str, Any] | None = None,
) -> str:
    """A fresh, commented ``config.toml`` for these leagues; ``[llm]`` and ``[notify]`` take the defaults when omitted.

    The text is validated before it is returned, so a bad entry raises ``ConfigError`` (located as ``config.toml``)
    rather than producing a file ``load_config`` would reject. ``write_config`` puts it on disk.
    """
    doc = tomlkit.document()
    for line in _HEADER:
        doc.add(tomlkit.comment(line))
    doc.add(tomlkit.nl())
    tables = tomlkit.aot()
    for entry in leagues:
        tables.append(_league_table(entry, appended=False))
    doc.add("league", tables)
    doc.add("llm", _table(Llm().model_dump() if llm is None else llm))
    doc.add("notify", _table(Notify().model_dump() if notify is None else notify))
    text = tomlkit.dumps(doc)
    parse_config(text, Path("config.toml"), secrets={})
    return text


def write_config(path: Path, text: str, *, env_path: Path | None = None) -> Config:
    """Validate ``text`` and make it the contents of ``path`` atomically, keeping the previous file as ``.bak``.

    ``.env`` is read from ``env_path`` (default: beside ``path``) for the validation only. Raises ``ConfigError`` and
    leaves the file untouched when the text is invalid or the write fails.
    """
    config_path, env_path = config_paths(path, env_path)
    config = parse_config(text, config_path, env_path=env_path)
    temp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(text.encode("utf-8"))
        if path.exists():
            shutil.copyfile(path, backup_path(path))
        os.replace(temp, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)
        raise ConfigError(f"{path}: {exc.strerror or exc}") from exc
    return config


def add_leagues(path: Path, entries: Iterable[Mapping[str, Any]], *, env_path: Path | None = None) -> tuple[str, ...]:
    """Append a ``[[league]]`` table for each entry whose ``(sport, espn_league_id)`` the file lacks; the keys added.

    Entries are the mappings ``render_config`` takes. A league already present is never rewritten, and when nothing is
    new the file is left alone (no write, no backup). The result is validated before the file changes, so a duplicate
    key or a bad entry raises ``ConfigError``.
    """
    text = _read(path)
    doc = _parse(text, path)
    tables = doc.get("league")
    if tables is None:
        tables = tomlkit.aot()
        doc["league"] = tables
    elif not isinstance(tables, AoT):
        raise ConfigError(f"{path}: league is not an array of [[league]] tables")
    present = {_identity(table) for table in tables}
    added: list[str] = []
    for entry in entries:
        identity = _identity(entry)
        if identity in present:
            continue
        present.add(identity)
        tables.append(_league_table(entry, appended=True))
        added.append(str(entry.get("key")))
    if not added:
        return ()
    write_config(path, _keep_newlines(tomlkit.dumps(doc), text), env_path=env_path)
    return tuple(added)


def set_value(path: Path, league_key: str, dotted_key: str, value: object, *, env_path: Path | None = None) -> Config:
    """Set one key of the league ``league_key``: ``preset``, ``policy.lineup``, ``trade_finder.enabled`` and the like.

    ``value`` is a TOML value (a string, bool, int, float or a list of them); ``None`` removes the key. A table on the
    way (``policy``, ``trade_finder``) is created when missing. Comments and the other keys stay as they are. The edit
    is validated before the file changes, so an unknown key or a bad value raises ``ConfigError`` with its location.
    """
    names = dotted_key.split(".")
    if not all(names):
        raise ValueError(f"invalid key {dotted_key!r}: use preset, policy.<key> or trade_finder.<key>")

    def mutate(doc: TOMLDocument) -> None:
        container: MutableMapping[str, Any] = _find_league(doc, league_key, path)
        created: Table | None = None
        for name in names[:-1]:
            child = container.get(name)
            if child is None:
                if value is None:
                    return  # nothing to remove
                child = created = tomlkit.table()
                container[name] = child
            elif not isinstance(child, MutableMapping):
                raise ConfigError(f"{path}: league {league_key!r}: {name} is not a table")
            container = child
        if value is None:
            container.pop(names[-1], None)
        else:
            _set(container, names[-1], value)
        if created is not None:
            created.add(tomlkit.nl())  # a blank line after the new table, as the file's own tables have

    return _edit(path, env_path, mutate)


def set_notify_channel(path: Path, channel: str, *, env_path: Path | None = None) -> Config:
    """Set ``[notify] channel`` (``telegram`` or ``ntfy``), creating the table when the file has none."""

    def mutate(doc: TOMLDocument) -> None:
        notify = doc.get("notify")
        if notify is None:
            notify = tomlkit.table()
            doc["notify"] = notify
        elif not isinstance(notify, MutableMapping):
            raise ConfigError(f"{path}: notify is not a table")
        _set(notify, "channel", channel)

    return _edit(path, env_path, mutate)


def _set(container: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Replace or add a value. tomlkit carries the old inline comment over; this keeps it in its column too."""
    old = container.get(key)
    column = _comment_column(key, old) if isinstance(old, Item) else None
    container[key] = value
    new = container[key]
    if column is not None and isinstance(new, Item) and new.trivia.comment:
        new.trivia.comment_ws = " " * max(1, column - len(key) - len(" = ") - len(new.as_string()))


def _comment_column(key: str, item: Item) -> int | None:
    if not item.trivia.comment:
        return None
    return len(key) + len(" = ") + len(item.as_string()) + len(item.trivia.comment_ws)


def _edit(path: Path, env_path: Path | None, mutate: Callable[[TOMLDocument], None]) -> Config:
    """Read, change, validate and replace; an edit that changes no text validates the file and writes nothing."""
    text = _read(path)
    doc = _parse(text, path)
    mutate(doc)
    new_text = _keep_newlines(tomlkit.dumps(doc), text)
    if new_text == text:
        return parse_config(text, path, env_path=env_path)
    return write_config(path, new_text, env_path=env_path)


def _read(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise ConfigError(
            f"{path} not found; fm setup writes it, or copy the template in DESIGN.md section 14"
        ) from None
    except OSError as exc:
        raise ConfigError(f"{path}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc


def _parse(text: str, path: Path) -> TOMLDocument:
    try:
        return tomlkit.parse(text)
    except ParseError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc


def _keep_newlines(new_text: str, old_text: str) -> str:
    """tomlkit writes new lines with ``\\n``; a file that uses CRLF keeps CRLF throughout."""
    if "\r\n" not in old_text:
        return new_text
    return re.sub(r"(?<!\r)\n", "\r\n", new_text)


def _find_league(doc: TOMLDocument, league_key: str, path: Path) -> Table:
    tables = doc.get("league")
    if not isinstance(tables, AoT):
        raise ConfigError(f"{path}: no [[league]] tables")
    known: list[str] = []
    for table in tables:
        key = str(table.get("key"))
        if key == league_key:
            return table
        known.append(key)
    raise ConfigError(f"{path}: no league {league_key!r}; known: {', '.join(known)}")


def _identity(league: Mapping[str, Any]) -> tuple[str, str]:
    """What makes two ``[[league]]`` entries the same league: the sport and the ESPN league id."""
    return str(league.get("sport")), str(league.get("espn_league_id"))


def _league_table(entry: Mapping[str, Any], *, appended: bool) -> Table:
    """A ``[[league]]`` table: the scalar keys, then ``[league.<name>]`` for each sub-mapping; ``None`` values are
    left out. tomlkit spaces the tables of a fresh document itself; one ``appended`` to a parsed file needs its own
    blank line at the end, or the table that follows would start right under it.
    """
    table = tomlkit.table()
    last = table
    for key, value in entry.items():
        if isinstance(value, Mapping):
            last = _table(value)
            table.add(key, last)
        elif value is not None:
            table.add(key, _commented(key, value))
    if appended:
        last.add(tomlkit.nl())
    return table


def _table(values: Mapping[str, Any]) -> Table:
    table = tomlkit.table()
    for key, value in values.items():
        if value is not None:
            table.add(key, _commented(key, value))
    return table


def _commented(key: str, value: Any) -> Item:
    item = tomlkit.item(value)
    note = _COMMENTS.get(key)
    if note:
        item.comment(note)
        item.trivia.comment_ws = " " * max(1, COMMENT_COLUMN - len(key) - len(" = ") - len(item.as_string()))
    return item
