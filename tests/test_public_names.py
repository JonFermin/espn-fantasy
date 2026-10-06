"""Public names mean one thing across ``fm.config``, ``fm.store``, ``fm.espn``, ``fm.sports``, ``fm.decide``,
``fm.proposals``, ``fm.model``, ``fm.sources.news``, ``fm.jobs``, ``fm.notify``, ``fm.browser`` (flows, transactions,
selectors, canary), ``fm.executor``, ``fm.advisor`` and ``fm.eval``.

Store rows carry a ``Row`` suffix (``LeagueRow``) so they never shadow ``fm.config.League`` or a parsed ESPN model,
and a name two modules both expose must be one object (a re-export such as ``Sport``), never two definitions. A module
can then import from all of them without aliasing. Two names are per-module by convention and exempt: each sport
module's ``PLUGIN`` (what :func:`fm.sports.base.plugin_for` looks up) and a module's ``logger``.
"""

from __future__ import annotations

import importlib
import logging
import types

import fm.config
import fm.store
import fm.store.models
from fm.store import Identified, Row

MODULES = (
    "fm.config",
    "fm.store",
    "fm.store.models",
    "fm.store.repos",
    "fm.store.db",
    "fm.espn.ids",
    "fm.espn.settings",
    "fm.espn.auth",
    "fm.espn.client",
    "fm.espn.models",
    "fm.sports.base",
    "fm.sports.nfl",
    "fm.decide.registry",
    "fm.decide.lineup",
    "fm.decide.waivers",
    "fm.proposals",
    "fm.proposals.payloads",
    "fm.proposals.policy",
    "fm.proposals.queue",
    "fm.proposals.pause",
    "fm.sports.nba",
    "fm.model.ids",
    "fm.model.ids_nba",
    "fm.model.scoring",
    "fm.model.projections",
    "fm.model.availability",
    "fm.model.valuation",
    "fm.model.relevance",
    "fm.model.categories",
    "fm.model.value_nba",
    "fm.model.baseline_nfl",
    "fm.sources.news",
    "fm.jobs.sync",
    "fm.notify",
    "fm.notify.base",
    "fm.notify.messages",
    "fm.notify.nonces",
    "fm.notify.telegram",
    "fm.notify.ntfy",
    "fm.notify.send",
    "fm.notify.bot",
    "fm.browser.flows",
    "fm.browser.flows.lineup",
    "fm.browser.transactions",
    "fm.browser.selectors",
    "fm.executor",
    "fm.executor.audit",
    "fm.executor.run",
    "fm.executor.runtime",
    "fm.executor.transport",
    "fm.executor.ui",
    "fm.executor.verify",
    "fm.model.simulate",
    "fm.browser.flows.add_drop",
    "fm.browser.flows.waiver",
    "fm.jobs.tick",
    "fm.jobs.deadlines",
    "fm.jobs.scheduler_windows",
    "fm.advisor.client",
    "fm.advisor.news_triage",
    "fm.advisor.close_call",
    "fm.advisor.explain",
    "fm.browser.canary",
    "fm.browser.drills",
    "fm.decide.faab",
    "fm.decide.rankings",
    "fm.eval.backtest",
    "fm.eval.tune",
    "fm.espn.calendar",
    "fm.decide.lineup_daily",
    "fm.decide.streaming",
    "fm.decide.weekly",
)
PER_MODULE = frozenset({"PLUGIN"})
"""Names each module binds for itself by convention (``fm.sports.<sport>.PLUGIN``)."""


def public_names(module: types.ModuleType) -> dict[str, object]:
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_")
        and not isinstance(value, types.ModuleType | logging.Logger)
        and name not in PER_MODULE
    }


def test_no_public_name_is_bound_to_two_different_objects() -> None:
    bindings: dict[str, dict[str, object]] = {}
    for module_name in MODULES:
        for name, value in public_names(importlib.import_module(module_name)).items():
            bindings.setdefault(name, {})[module_name] = value
    collisions = {
        name: sorted(where) for name, where in bindings.items() if len({id(value) for value in where.values()}) > 1
    }
    assert collisions == {}


def test_store_rows_are_suffixed_and_share_the_config_sport_type() -> None:
    assert fm.store.LeagueRow is not fm.config.League
    assert fm.store.Sport is fm.config.Sport
    rows = [
        value
        for value in vars(fm.store.models).values()
        if isinstance(value, type) and issubclass(value, Row) and value not in (Row, Identified)
    ]
    assert len(rows) == len(fm.store.REPOSITORIES)  # one row model per table
    assert all(row.__name__.endswith("Row") for row in rows)
