"""Decision module registry: ``register(sport, kind, fn)`` and the lookups the tick and the CLI use.

Decision modules (``fm.decide.lineup``, ``waivers``, ``streaming``, ``trades``, ``weekly``, ...) register one callable
per ``(sport, kind)`` here instead of being imported by name, so adding a decision for a sport touches no shared file
(ROADMAP: decision modules register via this module; executor flows have their own registry). ``fm.jobs.tick`` asks
``registered("nfl")`` for what to run and ``lookup("nba", "lineup_daily")`` fetches one.

A ``(sport, kind)`` is bound once: registering it twice raises, so a module imported from two paths, or two modules
claiming one kind, is a startup error rather than a silent shadow (the rule ``fm.cli.register_commands`` applies to
command names). Sports are ``nfl`` / ``nba`` (``fm.config.Sport``); ESPN game keys (``ffl`` / ``fba``) are accepted and
normalised. Kinds are lower-case identifiers (``lineup``, ``lineup_daily``, ``waivers``, ``streaming``, ``trades``,
``offers``, ``weekly``); the registry does not fix the set, the modules that register do.

The registry stores callables and nothing else. Decisions emit proposals; running them, and whether a proposal is
executed, is the tick's and the policy's business (CLAUDE.md: workers propose, the executor acts). ``registry`` is the
process-wide instance behind the module-level functions; tests build their own :class:`DecisionRegistry`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from fm.config import Sport
from fm.espn.ids import Game

type DecisionFn = Callable[..., Any]
"""A registered decision callable. Its signature is the decision module's contract with the tick, not the registry's."""

KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_SPORTS: dict[Game, Sport] = {Game.FFL: "nfl", Game.FBA: "nba"}


class DuplicateDecisionError(ValueError):
    """``(sport, kind)`` is already registered."""


class UnknownDecisionError(LookupError):
    """Nothing is registered under ``(sport, kind)``."""


def normalize_sport(sport: Game | str) -> Sport:
    """``nfl`` / ``nba`` from a sport, an ESPN game key (``ffl`` / ``fba``) or a ``Game``. Raises ``ValueError``."""
    return _SPORTS[Game.coerce(sport)]


def normalize_kind(kind: str) -> str:
    """A decision kind: a lower-case identifier such as ``lineup`` or ``lineup_daily``. Raises ``ValueError``."""
    if not isinstance(kind, str) or not KIND_PATTERN.match(kind):
        raise ValueError(f"decision kind should be a lower-case identifier such as 'lineup_daily', got {kind!r}")
    return kind


@dataclass(frozen=True, slots=True)
class Registration:
    """One registered decision: the sport and kind it serves and the callable that runs it."""

    sport: Sport
    kind: str
    fn: DecisionFn

    @property
    def key(self) -> tuple[Sport, str]:
        return (self.sport, self.kind)

    @property
    def name(self) -> str:
        """``nfl:lineup`` for logs and command output."""
        return f"{self.sport}:{self.kind}"

    @property
    def target(self) -> str:
        """Where the callable lives (``fm.decide.lineup.decide``), for logs; ``repr`` for callables without a name."""
        module = getattr(self.fn, "__module__", None)
        qualname = getattr(self.fn, "__qualname__", None)
        if module and qualname:
            return f"{module}.{qualname}"
        return repr(self.fn)


class DecisionRegistry:
    """Decision callables keyed by ``(sport, kind)``; iteration and ``registered`` follow registration order."""

    def __init__(self) -> None:
        self._entries: dict[tuple[Sport, str], Registration] = {}

    def register(self, sport: Game | str, kind: str, fn: DecisionFn) -> Registration:
        """Bind ``fn`` to ``(sport, kind)``. Raises ``DuplicateDecisionError`` when the pair is taken, ``ValueError``
        for an unsupported sport or malformed kind, ``TypeError`` when ``fn`` is not callable."""
        if not callable(fn):
            raise TypeError(f"decision for {sport!r}/{kind!r} should be callable, got {fn!r}")
        registration = Registration(sport=normalize_sport(sport), kind=normalize_kind(kind), fn=fn)
        existing = self._entries.get(registration.key)
        if existing is not None:
            raise DuplicateDecisionError(
                f"decision {registration.name} is registered twice, by {existing.target} and by {registration.target}"
            )
        self._entries[registration.key] = registration
        return registration

    def decision[F: DecisionFn](self, sport: Game | str, kind: str) -> Callable[[F], F]:
        """Decorator form of :meth:`register`; returns the function unchanged."""

        def decorate(fn: F) -> F:
            self.register(sport, kind, fn)
            return fn

        return decorate

    def get(self, sport: Game | str, kind: str) -> Registration | None:
        """The registration for ``(sport, kind)``, or ``None``."""
        return self._entries.get((normalize_sport(sport), normalize_kind(kind)))

    def lookup(self, sport: Game | str, kind: str) -> DecisionFn:
        """The callable for ``(sport, kind)``. Raises ``UnknownDecisionError`` naming what is registered instead."""
        registration = self.get(sport, kind)
        if registration is None:
            normalized = normalize_sport(sport)
            available = ", ".join(self.kinds(normalized)) or "nothing"
            raise UnknownDecisionError(f"no {kind!r} decision is registered for {normalized}; registered: {available}")
        return registration.fn

    def registered(self, sport: Game | str | None = None) -> tuple[Registration, ...]:
        """Every registration, or those for one sport, in registration order."""
        if sport is None:
            return tuple(self._entries.values())
        normalized = normalize_sport(sport)
        return tuple(entry for entry in self._entries.values() if entry.sport == normalized)

    def kinds(self, sport: Game | str) -> tuple[str, ...]:
        """Kinds registered for a sport, in registration order."""
        return tuple(entry.kind for entry in self.registered(sport))

    def unregister(self, sport: Game | str, kind: str) -> Registration:
        """Remove and return a registration. Raises ``UnknownDecisionError`` when there is none."""
        registration = self.get(sport, kind)
        if registration is None:
            raise UnknownDecisionError(f"no {kind!r} decision is registered for {normalize_sport(sport)}")
        del self._entries[registration.key]
        return registration

    def clear(self) -> None:
        self._entries.clear()

    def __contains__(self, key: object) -> bool:
        """``("nfl", "lineup") in registry``; the sport may be a game key or ``Game``."""
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        sport, kind = key
        if not isinstance(sport, Game | str) or not isinstance(kind, str):
            return False
        try:
            return (normalize_sport(sport), kind) in self._entries
        except ValueError:
            return False

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Registration]:
        return iter(tuple(self._entries.values()))


registry = DecisionRegistry()
"""The process-wide registry that decision modules register on at import."""


def register(sport: Game | str, kind: str, fn: DecisionFn) -> Registration:
    """Register ``fn`` as the ``kind`` decision for ``sport`` on the process-wide registry."""
    return registry.register(sport, kind, fn)


def decision[F: DecisionFn](sport: Game | str, kind: str) -> Callable[[F], F]:
    """``@decision("nfl", "lineup")`` registers the decorated function on the process-wide registry."""
    return registry.decision(sport, kind)


def get(sport: Game | str, kind: str) -> Registration | None:
    return registry.get(sport, kind)


def lookup(sport: Game | str, kind: str) -> DecisionFn:
    return registry.lookup(sport, kind)


def registered(sport: Game | str | None = None) -> tuple[Registration, ...]:
    return registry.registered(sport)


def unregister(sport: Game | str, kind: str) -> Registration:
    return registry.unregister(sport, kind)
