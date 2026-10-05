"""League scoring: a stat line becomes fantasy points under one league's scoring items, and a category league gets
its stats passed through (DESIGN sections 4 and 8.1; CLAUDE.md: league settings are data, projections are stat lines).

A stat line is ``{abbreviation: value}`` keyed by the sport's stat abbreviations (:class:`fm.sports.base.StatSchema`),
the shape of every projection and actual in the store. :class:`Scorer` binds a :class:`fm.espn.settings.LeagueSettings`
and turns a line into points by multiplying each scored stat by the league's ``points`` for it, honouring ESPN's
per-position ``pointsOverrides`` (an interception worth less to a D/ST than to a linebacker), and summing. PPR,
half-PPR, six-point passing touchdowns, yardage bonuses: none of it lives in code. For a category league
(``LeagueSettings.is_categories``) the same object returns the competed stats themselves, with the percentages derived
from makes and attempts when a line carries only those. ESPN's own ``appliedTotal`` for a line is exactly this sum, so
the scorer reproduces ESPN's points for every line ESPN scores under the same items.

Derived stats. ESPN scores a few stats that are functions of others: "every 25 passing yards" (``PY25``), "100-199
yard rushing game" (``RY100``), the points-allowed and yards-allowed brackets of a D/ST, totals such as ``FUML`` and
``TT``. ESPN's own lines carry most of them (its projections hold ``floor(E[yards] / 25)`` and the bracket
probabilities); other sources do not, so a league scoring one of them would silently drop it from a Sleeper line.
:func:`derive_stats` fills a derivable stat a line lacks, never one it has: every-N stats as ``floor(base / N)`` (exact
for one game; ESPN sums per-game floors in a season aggregate, which comes out a little lower), totals as the sum of
the parts the line has, brackets as a 0/1 indicator on the base stat (the single-source stand-in for ESPN's
probability), and aliases for the stats ESPN reports twice (``DPTSA`` is ``PTSA``; ``DPA14`` is ``PA14``, which keeps
ESPN's bracket probability where a league scores the D/ST family). Per-game and rate stats (``PYPG``, ``YPC``) are
never derived.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from fm.espn.ids import Game
from fm.espn.settings import LeagueSettings, ScoringItem

MAX_DERIVATION_DEPTH: Final = 4
"""How deep a derivation may chain (``DPA14`` from ``PA14`` from ``PTSA``); a cycle in the tables stops here."""


class ScoringError(ValueError):
    """A stat line cannot be scored as given: a non-numeric or non-finite value, or an unknown position label."""


# --- derivation rules -------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Every:
    """``floor(base / size)``, never negative: ESPN's "every N yards / attempts / receptions" stats."""

    base: str
    size: int


@dataclass(frozen=True, slots=True)
class Total:
    """The sum of whichever parts the line carries (``FUML`` from passing, rushing and receiving fumbles lost). With
    one part it is an alias: the same quantity under a second ESPN id."""

    parts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Scaled:
    """``base * factor`` (ESPN counts half sacks, two per sack)."""

    base: str
    factor: float


@dataclass(frozen=True, slots=True)
class Bracket:
    """``1.0`` when ``low <= base < high`` (either bound may be open), else ``0.0``: yardage-game bonuses and the
    D/ST points-allowed and yards-allowed brackets."""

    base: str
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True, slots=True)
class Ratio:
    """``(numerator + bonus_weight * bonus) / denominator``; nothing when the denominator is 0 or absent."""

    numerator: str
    denominator: str
    bonus: str | None = None
    bonus_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class Difference:
    """``minuend - subtrahend`` (missed shots from attempts and makes); nothing when the minuend is absent."""

    minuend: str
    subtrahend: str


type Rule = Every | Total | Scaled | Bracket | Ratio | Difference


def rule_inputs(rule: Rule) -> tuple[str, ...]:
    """The stats ``rule`` reads, in field order (a ``Ratio``'s bonus only when it has one)."""
    match rule:
        case Every(base, _) | Scaled(base, _) | Bracket(base, _, _):
            return (base,)
        case Total(parts):
            return parts
        case Ratio(numerator, denominator, bonus, _):
            return (numerator, denominator) if bonus is None else (numerator, denominator, bonus)
        case Difference(minuend, subtrahend):
            return (minuend, subtrahend)


def _every(base: str, *sizes: tuple[str, int]) -> dict[str, Rule]:
    return {stat: Every(base, size) for stat, size in sizes}


def _brackets(base: str, *bounds: tuple[str, float | None, float | None]) -> dict[str, Rule]:
    return {stat: Bracket(base, low, high) for stat, low, high in bounds}


def _aliases(*pairs: tuple[str, str]) -> dict[str, Rule]:
    return {stat: Total((same,)) for stat, same in pairs}


# ESPN's points-allowed brackets (suffix, low, high): ``PA0`` is a shutout, ``PA1`` 1-6 points, ... ``PA46`` 46 or more.
_POINTS_ALLOWED: tuple[tuple[str, float | None, float | None], ...] = (
    ("0", 0, 1),
    ("1", 1, 7),
    ("7", 7, 14),
    ("14", 14, 18),
    ("18", 18, 22),
    ("22", 22, 28),
    ("28", 28, 35),
    ("35", 35, 46),
    ("46", 46, None),
)

FFL_DERIVATIONS: Mapping[str, Rule] = MappingProxyType(
    {
        **_every("PY", ("PY5", 5), ("PY10", 10), ("PY20", 20), ("PY25", 25), ("PY50", 50), ("PY100", 100)),
        **_every("PC", ("PC5", 5), ("PC10", 10)),
        **_every("INC", ("IP5", 5), ("IP10", 10)),
        **_every("RY", ("RY5", 5), ("RY10", 10), ("RY20", 20), ("RY25", 25), ("RY50", 50), ("R100", 100)),
        **_every("RA", ("RA5", 5), ("RA10", 10)),
        **_every("REY", ("REY5", 5), ("REY10", 10), ("REY20", 20), ("REY25", 25), ("REY50", 50), ("RE100", 100)),
        **_every("REC", ("REC5", 5), ("REC10", 10)),
        **_every("TK", ("TK3", 3), ("TK5", 5)),
        **_every("KR", ("KR10", 10), ("KR25", 25)),
        **_every("PR", ("PR10", 10), ("PR25", 25)),
        **_every("FGY", ("FGY5", 5), ("FGY10", 10), ("FGY20", 20), ("FGY25", 25), ("FGY50", 50), ("FGY100", 100)),
        **_every(
            "FGMY", ("FGMY5", 5), ("FGMY10", 10), ("FGMY20", 20), ("FGMY25", 25), ("FGMY50", 50), ("FGMY100", 100)
        ),
        **_every(
            "FGAY", ("FGAY5", 5), ("FGAY10", 10), ("FGAY20", 20), ("FGAY25", 25), ("FGAY50", 50), ("FGAY100", 100)
        ),
        **_brackets("PY", ("P300", 300, 400), ("P400", 400, None)),
        **_brackets("RY", ("RY100", 100, 200), ("RY200", 200, None)),
        **_brackets("REY", ("REY100", 100, 200), ("REY200", 200, None)),
        **_brackets("PTSA", *((f"PA{suffix}", low, high) for suffix, low, high in _POINTS_ALLOWED)),
        **_aliases(*((f"DPA{suffix}", f"PA{suffix}") for suffix, _, _ in _POINTS_ALLOWED)),
        **_aliases(("DPTSA", "PTSA"), ("RECS", "REC")),
        **_brackets(
            "YA",
            ("YA100", None, 100),
            ("YA199", 100, 200),
            ("YA299", 200, 300),
            ("YA349", 300, 350),
            ("YA399", 350, 400),
            ("YA449", 400, 450),
            ("YA499", 450, 500),
            ("YA549", 500, 550),
            ("YA550", 550, None),
        ),
        "PTL": Total(("2PC", "2PR", "2PRE")),
        "FUM": Total(("PFUM", "RFUM", "REFUM")),
        "FUML": Total(("PFUML", "RFUML", "REFUML")),
        "TT": Total(("INTT", "FUML")),
        "TK": Total(("TKS", "TKA")),
        "DEFRETTD": Total(("INTTD", "FRTD")),
        "TRTD": Total(("BLKKRTD", "KRTD", "PRTD", "INTTD", "FRTD")),
        "FG": Total(("FG0", "FG40", "FG50P")),
        "FGA": Total(("FGA0", "FGA40", "FGA50P")),
        "FGM": Total(("FGM0", "FGM40", "FGM50P")),
        "FG50P": Total(("FG50", "FG60")),
        "FGA50P": Total(("FGA50", "FGA60")),
        "FGM50P": Total(("FGM50", "FGM60")),
        "FGAY": Total(("FGY", "FGMY")),
        "2PRET": Total(("O2PRET", "D2PRET")),
        "1PSF": Total(("O1PSF", "D1PSF")),
        "HALFSK": Scaled("SK", 2.0),
    }
)
"""``ffl`` stats derivable from others, by abbreviation (:data:`fm.espn.ids.FFL_STATS`). Checked against ESPN's own
lines: ``HALFSK`` is twice ``SK``, ``TRTD`` and ``DEFRETTD`` are the sums of their return touchdowns, and an actual
line carries ``PTSA`` and ``DPTSA`` (and each ``PA``/``DPA`` bracket) with one value."""

FBA_DERIVATIONS: Mapping[str, Rule] = MappingProxyType(
    {
        "REB": Total(("OREB", "DREB")),
        "FG%": Ratio("FGM", "FGA"),
        "FT%": Ratio("FTM", "FTA"),
        "3PT%": Ratio("3PM", "3PA"),
        "AFG%": Ratio("FGM", "FGA", bonus="3PM", bonus_weight=0.5),
        "FGMI": Difference("FGA", "FGM"),
        "FTMI": Difference("FTA", "FTM"),
        "3PMI": Difference("3PA", "3PM"),
    }
)
"""``fba`` stats derivable from others; the percentages are ratios (``0.487``), as ESPN's player views carry them."""

DERIVATIONS: Mapping[Game, Mapping[str, Rule]] = MappingProxyType(
    {Game.FFL: FFL_DERIVATIONS, Game.FBA: FBA_DERIVATIONS}
)


def _clean(line: Mapping[str, object], game: Game) -> dict[str, float]:
    """A stat line as ``dict[str, float]``. A non-numeric, NaN or infinite value raises :class:`ScoringError`: a
    stat that silently became nothing (or NaN points) would mis-rank every lineup it touches."""
    clean: dict[str, float] = {}
    for stat, value in line.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ScoringError(f"{game.value}: stat {stat!r} has a non-numeric value {value!r}")
        if not math.isfinite(value):
            raise ScoringError(f"{game.value}: stat {stat!r} has a non-finite value {value!r}")
        clean[str(stat)] = float(value)
    return clean


def _resolve(line: Mapping[str, float], stat: str, rules: Mapping[str, Rule], depth: int = 0) -> float | None:
    """The line's value for ``stat``, or the value its rule derives, or ``None`` when neither is available."""
    value = line.get(stat)
    if value is not None:
        return value
    rule = rules.get(stat)
    if rule is None or depth >= MAX_DERIVATION_DEPTH:
        return None

    def part(name: str) -> float | None:
        return _resolve(line, name, rules, depth + 1)

    match rule:
        case Every(base, size):
            amount = part(base)
            return None if amount is None else float(max(0, math.floor(amount / size)))
        case Total(parts):
            present = [value for name in parts if (value := part(name)) is not None]
            return math.fsum(present) if present else None
        case Scaled(base, factor):
            amount = part(base)
            return None if amount is None else amount * factor
        case Bracket(base, low, high):
            amount = part(base)
            if amount is None:
                return None
            inside = (low is None or amount >= low) and (high is None or amount < high)
            return 1.0 if inside else 0.0
        case Ratio(numerator, denominator, bonus, bonus_weight):
            top, bottom = part(numerator), part(denominator)
            if top is None or not bottom:
                return None
            extra = part(bonus) if bonus is not None else None
            return (top + bonus_weight * (extra or 0.0)) / bottom
        case Difference(minuend, subtrahend):
            whole = part(minuend)
            if whole is None:
                return None
            return whole - (part(subtrahend) or 0.0)


def derive_stats(line: Mapping[str, float], game: Game | str, stats: tuple[str, ...] | None = None) -> dict[str, float]:
    """``line`` plus every derivable stat it lacks (or just ``stats``, when given) that its contents allow.

    Values the line already has are never replaced; a stat whose inputs are absent stays absent.
    """
    resolved = Game.coerce(game)
    rules = DERIVATIONS.get(resolved, {})
    result = _clean(line, resolved)
    for stat in stats if stats is not None else tuple(rules):
        if stat in result:
            continue
        value = _resolve(result, stat, rules)
        if value is not None:
            result[stat] = value
    return result


# --- scoring ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Score:
    """What one stat line is worth in one league: ``points`` with their per-stat ``breakdown`` (points leagues) and
    the competed stats in ``categories`` (category leagues; empty otherwise)."""

    points: float
    breakdown: Mapping[str, float]
    categories: Mapping[str, float]


class Scorer:
    """League scoring bound to one league's settings (the common calls also exist as module functions).

    ``derive`` (default on) fills the scored stats a line lacks from the ones it has (see :func:`derive_stats`); the
    line itself is never changed. ``position`` arguments take an ESPN position label (``"D/ST"``) or
    ``defaultPositionId`` and select the ``pointsOverrides`` entry for it; without one every stat earns its base
    points, so pass the position whenever the league overrides any.
    """

    def __init__(self, settings: LeagueSettings, *, derive: bool = True) -> None:
        self.settings = settings
        self.derive = derive
        self._items: dict[str, ScoringItem] = {item.stat: item for item in settings.scoring_items}
        self._rules: Mapping[str, Rule] = DERIVATIONS.get(settings.game, {}) if derive else {}
        self._derivable: tuple[str, ...] = tuple(stat for stat in self._items if stat in self._rules)

    @property
    def game(self) -> Game:
        return self.settings.game

    @property
    def stats(self) -> tuple[str, ...]:
        """Abbreviations the league scores or competes on, by stat id."""
        return tuple(self._items)

    @property
    def derivable(self) -> tuple[str, ...]:
        """The scored stats this scorer fills in when a line lacks them."""
        return self._derivable

    @property
    def is_points(self) -> bool:
        return self.settings.is_points

    @property
    def is_categories(self) -> bool:
        return self.settings.is_categories

    def position_id(self, position: int | str | None) -> int | None:
        """An ESPN position id from a label or id; ``None`` stays ``None``. Raises :class:`ScoringError` for an
        unknown label (an id the league does not override simply gets the base points)."""
        if position is None or isinstance(position, int):
            return position
        try:
            return self.settings.ids.position_id(position)
        except KeyError as exc:
            raise ScoringError(str(exc)) from exc

    def prepare(self, line: Mapping[str, float]) -> dict[str, float]:
        """A copy of ``line`` with the league's derivable stats filled in where absent (see :func:`derive_stats`)."""
        prepared = _clean(line, self.settings.game)
        for stat in self._derivable:
            if stat not in prepared:
                value = _resolve(prepared, stat, self._rules)
                if value is not None:
                    prepared[stat] = value
        return prepared

    def breakdown(self, line: Mapping[str, float], *, position: int | str | None = None) -> dict[str, float]:
        """Points per scored stat, in scoring-item order; stats contributing nothing are left out."""
        position_id = self.position_id(position)
        prepared = self.prepare(line)
        contributions: dict[str, float] = {}
        for stat, item in self._items.items():
            value = prepared.get(stat)
            if value is None:
                continue
            earned = value * item.points_for(position_id)
            if earned != 0.0:
                contributions[stat] = earned
        return contributions

    def points(self, line: Mapping[str, float], *, position: int | str | None = None) -> float:
        """Fantasy points for ``line`` under the league's scoring items (``0.0`` in a category league, whose items
        carry no points)."""
        return math.fsum(self.breakdown(line, position=position).values())

    def categories(self, line: Mapping[str, float]) -> dict[str, float]:
        """The league's category stats from ``line`` (``0.0`` when absent and underivable); empty for points leagues.

        Percentages are derived from makes and attempts when the line has only those; a player with no attempts gets
        ``0.0``, so volume-weight them with the attempts in the line rather than averaging these values.
        """
        prepared = self.prepare(line)
        return {stat: prepared.get(stat, 0.0) for stat in self.settings.categories}

    def score(self, line: Mapping[str, float], *, position: int | str | None = None) -> Score:
        breakdown = self.breakdown(line, position=position)
        return Score(
            points=math.fsum(breakdown.values()),
            breakdown=MappingProxyType(breakdown),
            categories=MappingProxyType(self.categories(line)),
        )

    def many(
        self,
        lines: Mapping[int, Mapping[str, float]],
        positions: Mapping[int, int | str | None] | None = None,
    ) -> dict[int, float]:
        """Points per key (ESPN id, say) for many lines; ``positions`` supplies each key's position when known."""
        return {
            key: self.points(line, position=positions.get(key) if positions else None) for key, line in lines.items()
        }


def points(
    line: Mapping[str, float],
    settings: LeagueSettings,
    *,
    position: int | str | None = None,
    derive: bool = True,
) -> float:
    """Fantasy points for a stat line under ``settings`` (see :meth:`Scorer.points`)."""
    return Scorer(settings, derive=derive).points(line, position=position)


def breakdown(
    line: Mapping[str, float],
    settings: LeagueSettings,
    *,
    position: int | str | None = None,
    derive: bool = True,
) -> dict[str, float]:
    """Points per scored stat (see :meth:`Scorer.breakdown`)."""
    return Scorer(settings, derive=derive).breakdown(line, position=position)


def categories(line: Mapping[str, float], settings: LeagueSettings, *, derive: bool = True) -> dict[str, float]:
    """The league's category stats from a line (see :meth:`Scorer.categories`)."""
    return Scorer(settings, derive=derive).categories(line)


def score(
    line: Mapping[str, float],
    settings: LeagueSettings,
    *,
    position: int | str | None = None,
    derive: bool = True,
) -> Score:
    """Points, breakdown and categories for a line (see :meth:`Scorer.score`)."""
    return Scorer(settings, derive=derive).score(line, position=position)
