"""The one writer of ``config.toml`` (DESIGN section 14): ``fm setup``, ``fm config set-preset`` and the
dashboard all edit the file through here, and ``fm.config`` only ever reads it.

tomlkit keeps comments, key order and formatting, so a hand-edited file survives a programmatic edit, and a file
written with CRLF line endings keeps them. Every write follows the same rules: the new text is validated with
``fm.config.parse_config`` first (an invalid edit raises ``ConfigError`` and leaves the file untouched, byte for byte),
the replace is atomic (a uniquely named temp file beside the target, ``.config.toml.<random>``, plus ``os.replace``,
retried once after :data:`REPLACE_RETRY_S` when Windows refuses it because another process has the file open), and the
previous contents are kept as ``config.toml.bak``. An edit that changes nothing writes nothing, so a re-run of
``fm setup`` leaves the file alone.

Writers in different processes (``fm setup``, the dashboard, ``fm config set-preset``) can overlap, so every
read-modify-write holds an exclusive lock on ``config.toml.lock`` beside the file for its duration
(:func:`fm.jobs.runlock.exclusive_lock`, the run lock's OS-level primitive) and waits up to :data:`EDIT_LOCK_TIMEOUT_S`
for another writer: an edit is never lost to a concurrent one, and two writers never share a temp file.

League entries are TOML-shaped mappings, one per ``[[league]]`` table (``key``, ``sport``, ``espn_league_id``,
``season``, ``team_id``, an optional ``preset`` and optional ``policy`` / ``trade_finder`` sub-mappings). A rendered
file writes only the keys it is given: a league with a preset and no policy table lets the preset decide, which is
what keeps ``fm config set-preset`` meaningful later, since a key written in the file always wins over the preset.
:func:`preset_overrides` lists the written keys that do override a preset, so ``set-preset`` can say so (and
``--reset-overrides`` remove them). Secrets never pass through here; ``.env`` is read only to validate.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit import TOMLDocument
from tomlkit.exceptions import ParseError
from tomlkit.items import AoT, Item, Table

from fm.config import Config, ConfigError, Llm, Notify, config_paths, parse_config
from fm.jobs.runlock import RunLockHeld, exclusive_lock
from fm.presets import preset_for

COMMENT_COLUMN = 30
"""Inline comments start here when the key and value leave room, matching the DESIGN.md template."""
EDIT_LOCK_TIMEOUT_S = 10.0
"""How long an edit waits for another process's edit of the same file before giving up with ``ConfigError``."""
EDIT_LOCK_POLL_S = 0.1
REPLACE_RETRY_S = 0.2
"""Windows refuses ``os.replace`` onto a file another process holds open (an editor saving, an antivirus scan, a
reader mid-read); one retry after this pause gets past the usual brief hold."""
TEMP_PREFIX = ".config.toml."
"""The temp file of an atomic write: ``tempfile.mkstemp`` adds a random suffix, so no two writers share one."""
PRESET_TABLES = ("policy", "trade_finder")
"""The sub-tables of a ``[[league]]`` a preset fills."""

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
    "channel": "telegram | ntfy | none; the credentials live in .env",
}


@dataclass(frozen=True, slots=True)
class LeagueUpdate:
    """A league :func:`add_leagues` found already in the file under ``key`` whose ``season`` or ``team_id`` differ from
    the entry's. ``fields`` maps each differing name to ``(in the file, in the entry)``."""

    key: str
    fields: Mapping[str, tuple[object, object]]

    @property
    def updates(self) -> dict[str, object]:
        """The entry's values, ready for :func:`set_values` (``set_values(path, update.key, update.updates)``)."""
        return {name: new for name, (_, new) in self.fields.items()}


@dataclass(frozen=True, slots=True)
class AddLeaguesResult:
    """What :func:`add_leagues` did with each entry. ``added`` and ``unchanged`` hold league keys (``unchanged`` the
    file's own key, which may differ from the entry's); ``changed`` lists the present leagues whose season or team
    differ, which ``add_leagues`` leaves alone so the caller decides (``fm setup`` updates them with
    :func:`set_values`)."""

    added: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    changed: tuple[LeagueUpdate, ...] = ()


def backup_path(path: Path) -> Path:
    """Where a write keeps the previous file: ``config.toml.bak`` beside it."""
    return path.with_name(path.name + ".bak")


def edit_lock_path(path: Path) -> Path:
    """The lock every writer of ``path`` holds while it reads, changes and replaces it: ``config.toml.lock``."""
    return path.with_name(path.name + ".lock")


def render_config(
    leagues: Iterable[Mapping[str, Any]],
    *,
    llm: Mapping[str, Any] | None = None,
    notify: Mapping[str, Any] | str | None = None,
) -> str:
    """A fresh, commented ``config.toml`` for these leagues; ``[llm]`` and ``[notify]`` take the defaults when omitted.

    ``notify`` is the ``[notify]`` table, or just the channel name: ``"none"`` writes ``channel = "none"`` for an
    installation without a phone (proposals then wait for ``fm proposals`` or the dashboard). The text is validated
    before it is returned, so a bad entry raises ``ConfigError`` (located as ``config.toml``) rather than producing a
    file ``load_config`` would reject. ``write_config`` puts it on disk.
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
    if isinstance(notify, str):
        notify = {"channel": notify}
    doc.add("notify", _table(Notify().model_dump() if notify is None else notify))
    text = tomlkit.dumps(doc)
    parse_config(text, Path("config.toml"), secrets={})
    return text


def write_config(path: Path, text: str, *, env_path: Path | None = None) -> Config:
    """Validate ``text`` and make it the contents of ``path`` atomically, keeping the previous file as ``.bak``.

    ``.env`` is read from ``env_path`` (default: beside ``path``) for the validation only. Holds the edit lock while it
    replaces the file. Raises ``ConfigError`` and leaves the file untouched when the text is invalid, another writer
    holds the lock past :data:`EDIT_LOCK_TIMEOUT_S`, or the write fails.
    """
    config_path, env_path = config_paths(path, env_path)
    config = parse_config(text, config_path, env_path=env_path)
    with _edit_lock(path):
        _replace(path, text)
    return config


def add_leagues(path: Path, entries: Iterable[Mapping[str, Any]], *, env_path: Path | None = None) -> AddLeaguesResult:
    """Append a ``[[league]]`` table for each entry whose ``(sport, espn_league_id)`` the file lacks.

    Entries are the mappings ``render_config`` takes. A league already present is never rewritten: it is reported as
    ``unchanged``, or as ``changed`` (with the differing ``season`` / ``team_id``) for the caller to update through
    :func:`set_values`. When nothing is new the file is left alone (no write, no backup). The result is validated
    before the file changes, so a duplicate key or a bad entry raises ``ConfigError``.
    """
    _require(path)
    with _edit_lock(path):
        text = _read(path)
        doc = _parse(text, path)
        tables = doc.get("league")
        if tables is None:
            tables = tomlkit.aot()
            doc["league"] = tables
        elif not isinstance(tables, AoT):
            raise ConfigError(f"{path}: league is not an array of [[league]] tables")
        present: dict[tuple[str, str], Table] = {}
        for table in tables:
            present.setdefault(_identity(table), table)
        added: list[str] = []
        unchanged: list[str] = []
        changed: list[LeagueUpdate] = []
        for entry in entries:
            identity = _identity(entry)
            table = present.get(identity)
            if table is None:
                appended = _league_table(entry, appended=True)
                tables.append(appended)
                present[identity] = appended
                added.append(str(entry.get("key")))
                continue
            differ = {
                name: (_plain(table.get(name)), entry[name])
                for name in ("season", "team_id")
                if entry.get(name) is not None and _plain(table.get(name)) != entry[name]
            }
            if differ:
                changed.append(LeagueUpdate(key=str(table.get("key")), fields=differ))
            else:
                unchanged.append(str(table.get("key")))
        if added:
            _validate_and_replace(path, _keep_newlines(tomlkit.dumps(doc), text), env_path)
    return AddLeaguesResult(added=tuple(added), unchanged=tuple(unchanged), changed=tuple(changed))


def set_value(path: Path, league_key: str, dotted_key: str, value: object, *, env_path: Path | None = None) -> Config:
    """Set one key of the league ``league_key``: ``preset``, ``policy.lineup``, ``trade_finder.enabled`` and the like.

    ``value`` is a TOML value (a string, bool, int, float or a list of them); ``None`` removes the key. A table on the
    way (``policy``, ``trade_finder``) is created when missing. Comments and the other keys stay as they are. The edit
    is validated before the file changes, so an unknown key or a bad value raises ``ConfigError`` with its location.
    """
    return set_values(path, league_key, {dotted_key: value}, env_path=env_path)


def set_values(
    path: Path, league_key: str, values: Mapping[str, object], *, env_path: Path | None = None
) -> Config:
    """:func:`set_value` for several keys of one league in a single edit (one validation, one replace, one backup):
    ``{"preset": "balanced", "policy.lineup": None}`` sets the preset and removes the written lineup key."""
    changes = [(_dotted_names(key), value) for key, value in values.items()]

    def mutate(doc: TOMLDocument) -> None:
        league = _find_league(doc, league_key, path)
        for names, value in changes:
            _apply(league, names, value, league_key, path)

    return _edit(path, env_path, mutate)


def remove_keys(path: Path, league_key: str, dotted_keys: Iterable[str], *, env_path: Path | None = None) -> Config:
    """Remove these keys (``policy.lineup``, ``trade_finder.enabled``) from one league in a single edit; a key that is
    not written is skipped. What ``fm config set-preset --reset-overrides`` uses to hand keys back to the preset."""
    return set_values(path, league_key, dict.fromkeys(dotted_keys), env_path=env_path)


def preset_overrides(
    source: Path | Mapping[str, Any], league_key: str | None = None, *, preset: str | None = None
) -> list[str]:
    """The ``policy.*`` / ``trade_finder.*`` keys written in a league's table that override its preset: keys the preset
    supplies, written with a different value. Keys the preset says nothing about (``untouchables``, say) are not
    overrides, and a league without a preset has none.

    ``source`` is ``config.toml`` (``league_key`` names the league) or one ``[[league]]`` table as a mapping (the raw
    TOML: what the file says, not what the preset filled in). ``preset`` checks against another preset than the
    table's own (what ``set-preset`` is about to write). Raises ``ConfigError`` for an unreadable file or an unknown
    league, ``ValueError`` for an unknown preset name.
    """
    if isinstance(source, Path):
        if league_key is None:
            raise ValueError("preset_overrides needs the league key to read a league from config.toml")
        table: Mapping[str, Any] = _raw_league(source, league_key)
    else:
        table = source
    name = preset if preset is not None else table.get("preset")
    if name is None:
        return []
    chosen = preset_for(str(name), str(table.get("sport")))
    overrides: list[str] = []
    for table_name, supplied in zip(PRESET_TABLES, (chosen.policy, chosen.trade_finder), strict=True):
        written = table.get(table_name)
        if not isinstance(written, Mapping):
            continue
        for key, value in written.items():
            if key in supplied and _plain(value) != supplied[key]:
                overrides.append(f"{table_name}.{key}")
    return overrides


def set_notify_channel(path: Path, channel: str, *, env_path: Path | None = None) -> Config:
    """Set ``[notify] channel`` (``telegram``, ``ntfy`` or ``none``), creating the table when the file has none."""

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


def _dotted_names(dotted_key: str) -> list[str]:
    names = dotted_key.split(".")
    if not all(names):
        raise ValueError(f"invalid key {dotted_key!r}: use preset, policy.<key> or trade_finder.<key>")
    return names


def _apply(league: Table, names: list[str], value: object, league_key: str, path: Path) -> None:
    """Set (or, for ``None``, remove) one dotted key under a ``[[league]]`` table, creating a table on the way."""
    container: MutableMapping[str, Any] = league
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


def _edit(path: Path, env_path: Path | None, mutate: Callable[[TOMLDocument], None]) -> Config:
    """Read, change, validate and replace under the edit lock; an edit that changes no text validates the file and
    writes nothing."""
    _require(path)
    with _edit_lock(path):
        text = _read(path)
        doc = _parse(text, path)
        mutate(doc)
        new_text = _keep_newlines(tomlkit.dumps(doc), text)
        if new_text == text:
            return parse_config(text, path, env_path=env_path)
        return _validate_and_replace(path, new_text, env_path)


def _validate_and_replace(path: Path, text: str, env_path: Path | None) -> Config:
    """:func:`write_config` for a caller that already holds the edit lock."""
    config_path, env = config_paths(path, env_path)
    config = parse_config(text, config_path, env_path=env)
    _replace(path, text)
    return config


@contextmanager
def _edit_lock(path: Path) -> Iterator[None]:
    """Hold ``config.toml.lock`` (another process's edit is waited for, up to :data:`EDIT_LOCK_TIMEOUT_S`)."""
    lock = edit_lock_path(path)
    with contextlib.ExitStack() as stack:
        try:
            held = exclusive_lock(
                lock, command=f"an edit of {path.name}", timeout=EDIT_LOCK_TIMEOUT_S, poll_s=EDIT_LOCK_POLL_S
            )
            stack.enter_context(held)
        except RunLockHeld as exc:
            who = exc.holder.describe() if exc.holder is not None else "another process"
            raise ConfigError(f"{path} is being edited by {who}; try again in a moment") from exc
        except OSError as exc:
            raise ConfigError(f"{lock}: {exc.strerror or exc}") from exc
        yield


def _replace(path: Path, text: str) -> None:
    """Write ``text`` to a fresh temp file beside ``path``, back ``path`` up, and move the temp file over it. The temp
    file is removed on any failure, and ``ConfigError`` says what went wrong."""
    temp: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=TEMP_PREFIX)
        temp = Path(name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(text.encode("utf-8"))
        if path.exists():
            shutil.copyfile(path, backup_path(path))
        try:
            os.replace(temp, path)
        except PermissionError:  # Windows: another process has the target open for a moment
            time.sleep(REPLACE_RETRY_S)
            os.replace(temp, path)
    except OSError as exc:
        if temp is not None:
            with contextlib.suppress(OSError):
                temp.unlink(missing_ok=True)
        raise ConfigError(f"{path}: {exc.strerror or exc}") from exc


def _raw_league(path: Path, league_key: str) -> dict[str, Any]:
    """The ``[[league]]`` table ``league_key`` exactly as the file writes it (no preset applied)."""
    try:
        data = tomllib.loads(_read(path))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    tables = data.get("league")
    known: list[str] = []
    for table in tables if isinstance(tables, list) else ():
        if isinstance(table, dict) and table.get("key") == league_key:
            return table
        known.append(str(table.get("key")) if isinstance(table, dict) else "?")
    raise ConfigError(f"{path}: no league {league_key!r}; known: {', '.join(known)}")


def _plain(value: object) -> object:
    """A tomlkit item as the plain Python value it stands for (tomlkit's ints and strings compare equal already, but a
    table or an array does not compare equal to a dict or a list)."""
    unwrap = getattr(value, "unwrap", None)
    return unwrap() if callable(unwrap) else value


def _require(path: Path) -> None:
    """Fail on a missing file before an edit takes its lock, so editing nothing leaves no ``config.toml.lock``."""
    if not path.exists():
        _read(path)


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
