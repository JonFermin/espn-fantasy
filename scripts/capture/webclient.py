"""Facts the ESPN web client ships as code rather than serving from the API (ROADMAP #14).

The fantasy web app (``fantasy.espn.com``, the "kona" Next.js build) bundles each game's season calendar as constants:
a ``scoringPeriods`` list (``{id, startDate, endDate, preSeason, postSeason}`` in epoch ms) and a ``segments`` table
whose ``periodTypes`` group scoring periods into daily, weekly and season-long periods (``{id, scoringPeriodStart,
scoringPeriodEnd}``). A league's ``scheduleSettings.matchupPeriods`` lists period ids of the type named by
``scheduleSettings.periodTypeId``, so this table is what turns an NBA matchup into days; no read view carries it.

:func:`extract_calendars` pulls every calendar out of the shared ``commons/main`` bundle (the page loads it anyway, so
no extra request is needed) and :func:`match_calendar` picks the one that fits a game's ``proTeamSchedules_wl``: its
last regular scoring period is the schedule's last one and every game's start falls inside its period. The bundle also
lists ESPN's transaction error codes (:func:`error_codes`), labels each game's stat splits
(:func:`extract_stat_settings`) and holds the code that builds and sends every write
(:func:`extract_transaction_code`): the transaction type constants, the item builders and serializer, the service that
picks a type per flow, and the request that POSTs the result to the write host. Those excerpts are saved verbatim, so
the write payloads in ``docs/espn-api.md`` can be re-checked against ESPN's own code.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

BUNDLE_MARKER = "/_next/static/commons/main-"
"""The shared web-client bundle that holds the calendars and the transaction model."""
DAY_MS = 86_400_000
_SEGMENTS = re.compile(r"e\.exports=\[\{id:0,periodTypes:")
_SCORING_PERIODS = re.compile(r"e\.exports=\[\{endDate:")
_ERROR_CODE = re.compile(r'"((?:TRAN|FAILED)_[A-Z0-9_]+)"')


class CalendarError(ValueError):
    """The bundle did not hold a calendar that fits the game."""


class WebClientError(ValueError):
    """The bundle did not hold a piece of web-client code where the extractor looks for it."""


# --- calendars --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Calendar:
    """One game's season calendar as the web client ships it."""

    scoring_periods: list[dict[str, Any]]
    period_types: list[dict[str, Any]]

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Calendar:
        """The inverse of :meth:`as_json`, for ``calendar_<game>.json`` and the ``<game>/calendar.json`` fixtures."""
        return cls(scoring_periods=list(data["scoringPeriods"]), period_types=list(data["periodTypes"]))

    @property
    def last_regular_period(self) -> int:
        regular = [p["id"] for p in self.scoring_periods if p["id"] > 0 and not p.get("postSeason")]
        return max(regular) if regular else 0

    def window(self, scoring_period: int, *, weekly_game: bool) -> tuple[int, int]:
        """``[start, end)`` of a scoring period in epoch ms. Period 1's stored start is a preseason placeholder; the
        web client uses one period length before its end instead (7 days for weekly games, 1 day for daily ones)."""
        period = next(p for p in self.scoring_periods if p["id"] == scoring_period)
        start = period["startDate"]
        if scoring_period == 1:
            start = period["endDate"] - (7 if weekly_game else 1) * DAY_MS
        return start, period["endDate"]

    def periods(self, period_type_id: int) -> list[dict[str, Any]]:
        for period_type in self.period_types:
            if period_type["id"] == period_type_id:
                return list(period_type["periods"])
        raise KeyError(f"no period type {period_type_id}; known: {[pt['id'] for pt in self.period_types]}")

    def matchup_days(self, matchup_periods: Mapping[str, Sequence[int]], period_type_id: int) -> dict[int, list[int]]:
        """Scoring periods per matchup period: ``scheduleSettings.matchupPeriods`` resolved through this calendar."""
        ranges = {p["id"]: (p["scoringPeriodStart"], p["scoringPeriodEnd"]) for p in self.periods(period_type_id)}
        result: dict[int, list[int]] = {}
        for matchup, period_ids in matchup_periods.items():
            days: list[int] = []
            for period_id in period_ids:
                start, end = ranges[int(period_id)]
                days.extend(range(start, end + 1))
            result[int(matchup)] = days
        return dict(sorted(result.items()))

    def as_json(self) -> dict[str, Any]:
        return {"scoringPeriods": self.scoring_periods, "periodTypes": self.period_types}


def _balanced(text: str, start: int) -> str:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise CalendarError("unbalanced constant in the bundle")


def _js_literal(source: str) -> Any:
    """A minified JS object/array literal of plain data as JSON (unquoted keys, ``!0``/``!1``, ``1e3`` numbers)."""
    text = re.sub(r"([{,])([A-Za-z_$][\w$]*):", r'\1"\2":', source)
    text = text.replace("!0", "true").replace("!1", "false")
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)e(\d+)", lambda m: str(round(float(m.group(0)))), text)
    return json.loads(text)


def extract_calendars(bundle: str) -> list[Calendar]:
    """Every (scoring periods, segments) pair in the bundle, paired in module order."""
    offset = len("e.exports=")
    periods = [m.start() for m in _SCORING_PERIODS.finditer(bundle)]
    segments = [m.start() for m in _SEGMENTS.finditer(bundle)]
    calendars: list[Calendar] = []
    for index, position in enumerate(periods):
        next_periods = periods[index + 1] if index + 1 < len(periods) else len(bundle)
        following = [seg for seg in segments if position < seg < next_periods]
        if not following:
            continue
        scoring = _js_literal(_balanced(bundle, position + offset))
        table = _js_literal(_balanced(bundle, following[0] + offset))
        calendars.append(Calendar(scoring_periods=scoring, period_types=table[0]["periodTypes"]))
    return calendars


def pro_games(pro_schedule: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every game of a ``proTeamSchedules_wl`` body, once each."""
    games: dict[Any, dict[str, Any]] = {}
    teams = (pro_schedule.get("settings") or {}).get("proTeams") or []
    for team in teams:
        for period_games in (team.get("proGamesByScoringPeriod") or {}).values():
            for game in period_games:
                games.setdefault(game.get("id"), game)
    return list(games.values())


def match_calendar(calendars: Iterable[Calendar], pro_schedule: Mapping[str, Any], *, weekly_game: bool) -> Calendar:
    """The calendar whose last regular period is the schedule's last and that contains every game's start."""
    games = pro_games(pro_schedule)
    if not games:
        raise CalendarError("the pro schedule has no games")
    last = max(int(game["scoringPeriodId"]) for game in games)
    for calendar in calendars:
        if calendar.last_regular_period != last:
            continue
        if all(_inside(calendar, game, weekly_game) for game in games):
            return calendar
    raise CalendarError(f"no bundled calendar ends at scoring period {last} and contains every game")


def _inside(calendar: Calendar, game: Mapping[str, Any], weekly_game: bool) -> bool:
    try:
        start, end = calendar.window(int(game["scoringPeriodId"]), weekly_game=weekly_game)
    except StopIteration:
        return False
    return start <= int(game["date"]) < end


def error_codes(bundle: str) -> list[str]:
    """ESPN's transaction error types and failed statuses named in the bundle (``TRAN_*``, ``FAILED_*``)."""
    return sorted(set(_ERROR_CODE.findall(bundle)))


# --- stat settings ----------------------------------------------------------------------------------------------------

_STAT_SETTINGS = re.compile(r"e\.exports=\{")
_FIRST_API_IDENTIFIER = re.compile(r'apiIdentifier:"([\w.]+)"')


def extract_stat_settings(bundle: str) -> list[dict[str, Any]]:
    """Each game's stat ``sources`` and ``splitTypes`` as the web client labels them (``statSplitTypeId`` 5 is
    "Game" in basketball, 2 is "Rest Of Season" in football), with the first stat's ``apiIdentifier`` to tell the
    sports apart (``passing.*`` is football, ``offensive.points`` basketball). The bundle names no game key here."""
    found: list[dict[str, Any]] = []
    for match in re.finditer(r"splitTypes:\[", bundle):
        start = bundle.rfind("e.exports={", 0, match.start())
        if start < 0:
            continue
        body = bundle[start : _scan(bundle, start + len("e.exports="))]
        entry: dict[str, Any] = {}
        for key in ("sources", "splitTypes"):
            at = body.find(f"{key}:[")
            if at >= 0:
                array_start = at + len(key) + 1
                entry[key] = _js_literal(body[array_start : _scan(body, array_start)])
        stats_at = body.find("stats:[")
        first = _FIRST_API_IDENTIFIER.search(body, stats_at) if stats_at >= 0 else None
        entry["firstStat"] = first.group(1) if first else None
        if entry.get("splitTypes"):
            found.append(entry)
    return found


# --- transaction code -------------------------------------------------------------------------------------------------

_TYPES_ANCHOR = re.compile(r'const (\w+)="TRADE_PROPOSAL";t\["\w+"\]=\1\b')
_MODEL_ANCHOR = re.compile(r"pushItems\((\w+)\)\{this\.items\.push\(\.\.\.\1\)\}get\(\)\{")
_SERVICE_ANCHOR = re.compile(r"class \w+\{createTransaction\(\{")
_SAVE_ANCHOR = re.compile(r'"/transactions/"')
_ORDER_ANCHOR = re.compile(r'"/pendingTransactions"\)')
_BID_ANCHOR = re.compile(r'"/pendingTransactions/"\)')
_MODULE_HEADER = re.compile(r'(?<=[,\[])function\(e,t,n\)\{"use strict";')
_ASYNC_WRAPPER = re.compile(r"function (\w+)\(\)\{\1=")
_CALLEE = re.compile(r"yield (\w+)\(\{config:")
_IMPORT = re.compile(r"var (\w+)=n\((\d+)\);")
_EXPORTED_STRING = re.compile(r'const (\w+)="([A-Za-z_]+)";t\["(\w+)"\]=\1\b')


def _skip_string(text: str, index: int) -> int:
    """The index just past the string literal (quote or template) that starts at ``text[index]``."""
    quote = text[index]
    position = index + 1
    while position < len(text):
        char = text[position]
        if char == "\\":
            position += 2
        elif char == quote:
            return position + 1
        elif quote == "`" and text.startswith("${", position):
            position = _scan(text, position + 1)
        else:
            position += 1
    raise WebClientError("unterminated string literal in the bundle")


def _scan(text: str, start: int, *, statement: bool = False) -> int:
    """The end (exclusive) of the bracketed block that opens at ``text[start]``, or with ``statement``, of the statement
    that starts there (through its ``;`` at depth 0, or up to the brace that closes its block). String and template
    literals are skipped; regex literals are not understood, which the excerpted code does not need."""
    depth = 0
    position = start
    while position < len(text):
        char = text[position]
        if char in "'\"`":
            position = _skip_string(text, position)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0 and not statement:
                return position + 1
            if depth < 0:
                return position
        elif char == ";" and depth == 0 and statement:
            return position + 1
        position += 1
    raise WebClientError("unbalanced code in the bundle")


def _search(pattern: re.Pattern[str], text: str, what: str, lo: int = 0, hi: int | None = None) -> re.Match[str]:
    match = pattern.search(text, lo, len(text) if hi is None else hi)
    if match is None:
        raise WebClientError(f"{what} not found in the bundle ({pattern.pattern!r})")
    return match


def _block_end(text: str, start: int) -> int:
    """The end of the block whose header starts at ``start``: the first ``{`` from there opens its body."""
    return _scan(text, text.index("{", start))


def _enclosing(text: str, position: int, header: re.Pattern[str], lo: int = 0) -> tuple[int, int]:
    """``(start, end)`` of the innermost block whose header matches ``header`` and whose body contains ``position``.
    The first ``{`` of the header match opens the body."""
    for match in reversed(list(header.finditer(text, lo, position))):
        try:
            end = _block_end(text, match.start())
        except WebClientError:
            continue
        if end > position:
            return match.start(), end
    raise WebClientError(f"no block matching {header.pattern!r} encloses offset {position}")


def _statement(text: str, position: int, lo: int) -> str:
    """The ``const`` statement that contains ``position``."""
    for match in reversed(list(re.finditer(r"const \w+=", text[lo:position]))):
        start = lo + match.start()
        end = _scan(text, start, statement=True)
        if end > position:
            return text[start:end]
    raise WebClientError(f"no const statement encloses offset {position}")


def _module(text: str, position: int) -> tuple[int, int]:
    return _enclosing(text, position, _MODULE_HEADER)


def _wrapper(text: str, position: int, lo: int) -> str:
    """The named async wrapper around ``position``: ``function co(){co=nt(function*(...){...});...}``."""
    start, end = _enclosing(text, position, _ASYNC_WRAPPER, lo)
    return text[start:end]


def type_names(constants_source: str) -> dict[str, str]:
    """The string constants a module exports, by export key: ``{"G": "TRADE_PROPOSAL", "N": "WAIVER", ...}``."""
    return {key: value for _, value, key in _EXPORTED_STRING.findall(constants_source)}


def imports(module_source: str) -> dict[str, str]:
    """A module's ``var r=n(44);`` imports as ``{alias: module id}``."""
    return dict(_IMPORT.findall(module_source))


def _keyed_uses(source: str, alias: str) -> list[str]:
    """Every ``KEY`` in ``alias["KEY"]`` in ``source``."""
    return re.findall(rf'(?<![\w$.]){re.escape(alias)}\["(\w+)"\]', source)


def resolve_constants(source: str, alias: str, names: Mapping[str, str]) -> str:
    """``source`` with every ``alias["KEY"]`` that names a constant replaced by its string (``r["N"]`` -> ``"WAIVER"``),
    so the excerpt reads as the payload rules it encodes."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return json.dumps(names[key]) if key in names else match.group(0)

    return re.sub(rf'(?<![\w$.]){re.escape(alias)}\["(\w+)"\]', replace, source)


def _constants_alias(source: str, aliases: Iterable[str], names: Mapping[str, str]) -> str:
    """Of ``aliases`` (imports of a module the model and the service share), the one read as ``alias["KEY"]`` most
    often with every key a type constant."""
    best, best_uses = None, 0
    for alias in aliases:
        keys = _keyed_uses(source, alias)
        if keys and set(keys) <= set(names) and len(keys) > best_uses:
            best, best_uses = alias, len(keys)
    if best is None:
        raise WebClientError("no shared import reads the type constants")
    return best


def extract_transaction_code(bundle: str) -> dict[str, Any]:
    """ESPN's write path, verbatim: the type constants, the model (item builders and the serializer ``get()``), the
    service (one method per flow), the request functions that POST to ``/transactions/`` and the two
    ``pendingTransactions`` paths, the POST helper, the write host and the request defaults. ``constantsAlias`` names
    the variable through which the model and the service read the type constants (the module both import)."""
    types_start, types_end = _module(bundle, _search(_TYPES_ANCHOR, bundle, "the transaction type constants").start())
    model_start, model_end = _module(bundle, _search(_MODEL_ANCHOR, bundle, "the transaction model").start())
    service_start, service_end = _module(bundle, _search(_SERVICE_ANCHOR, bundle, "the transaction service").start())
    excerpts = {
        "types": bundle[types_start:types_end],
        "model": bundle[model_start:model_end],
        "service": bundle[service_start:service_end],
    }
    save_at = _search(_SAVE_ANCHOR, bundle, "the transactions endpoint").start()
    api_start, api_end = _module(bundle, save_at)
    excerpts["saveTransaction"] = _wrapper(bundle, save_at, api_start)
    order_at = _search(_ORDER_ANCHOR, bundle, "the pending-order endpoint", api_start, api_end).start()
    excerpts["reorderPendingTransactions"] = _wrapper(bundle, order_at, api_start)
    bid_at = _search(_BID_ANCHOR, bundle, "the pending-bid endpoint", api_start, api_end).start()
    excerpts["updatePendingBid"] = _wrapper(bundle, bid_at, api_start)
    callee = _search(_CALLEE, excerpts["saveTransaction"], "the POST helper call").group(1)
    entry = _search(
        re.compile(rf"function {re.escape(callee)}\(\w*\)\{{return (\w+)\.apply\(this,arguments\)\}}"),
        bundle,
        f"the POST helper {callee}",
        api_start,
        api_end,
    )
    helper = _search(
        re.compile(rf"function {re.escape(entry.group(1))}\(\)\{{"), bundle, "the POST helper", entry.end(), api_end
    )
    excerpts["post"] = entry.group(0) + bundle[helper.start() : _block_end(bundle, helper.start())]
    for name, anchor in (
        ("writeHost", '"https://lm-api-writes."'),
        ("writeHostType", '"HOST_TYPE_FANTASY_WRITE_API"'),
        ("requestDefaults", '"X-Fantasy-Source":"kona"'),
        ("requestConfig", '"X-Fantasy-Platform":'),
    ):
        at = bundle.find(anchor, api_start, api_end)
        if at < 0:
            raise WebClientError(f"{name} ({anchor}) not found in the request module")
        excerpts[name] = _statement(bundle, at, api_start)
    names = type_names(excerpts["types"])
    model_imports, service_imports = imports(excerpts["model"]), imports(excerpts["service"])
    shared = set(model_imports.values()) & set(service_imports.values())
    aliases = {
        part: _constants_alias(excerpts[part], [alias for alias, module in found.items() if module in shared], names)
        for part, found in (("model", model_imports), ("service", service_imports))
    }
    return {"typeNames": names, "constantsAlias": aliases, "excerpts": excerpts}
