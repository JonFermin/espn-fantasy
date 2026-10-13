"""``fm setup``'s logic (docs/PLUG_AND_PLAY.md section 6), kept apart from the prompts in ``fm.commands.setup``.

Usage::

    from fm.setup import AcceptDefaults, SetupOptions, run_setup

    result = run_setup(SetupOptions(preset="balanced"), deps, prompts=AcceptDefaults(), echo=print)

``deps`` (:class:`SetupDeps`) carries every external effect: session load and login, league discovery and
confirmation, phone setup, sync, the scheduler jobs, the ``fm bot`` probe and the doctor.
``fm.commands.setup.live_deps`` builds the real ones; tests pass fakes. See :mod:`fm.setup.wizard` for the steps.
"""

from __future__ import annotations

from fm.setup.wizard import (
    BOT_COMMAND,
    DEFAULT_PHONE,
    DEFAULT_PRESET,
    PHONE_CHANNELS,
    PHONE_DESCRIPTIONS,
    SECRET_PROMPTS,
    SUMMARY_HINT,
    AcceptDefaults,
    ChannelSetup,
    Job,
    PlannedLeague,
    Prompts,
    SecretPrompt,
    SetupDeps,
    SetupError,
    SetupOptions,
    SetupResult,
    StepError,
    describe_league,
    league_key,
    notify_setup,
    parse_bot_pids,
    read_secrets,
    run_setup,
)

__all__ = [
    "BOT_COMMAND",
    "DEFAULT_PHONE",
    "DEFAULT_PRESET",
    "PHONE_CHANNELS",
    "PHONE_DESCRIPTIONS",
    "SECRET_PROMPTS",
    "SUMMARY_HINT",
    "AcceptDefaults",
    "ChannelSetup",
    "Job",
    "PlannedLeague",
    "Prompts",
    "SecretPrompt",
    "SetupDeps",
    "SetupError",
    "SetupOptions",
    "SetupResult",
    "StepError",
    "describe_league",
    "league_key",
    "notify_setup",
    "parse_bot_pids",
    "read_secrets",
    "run_setup",
]
