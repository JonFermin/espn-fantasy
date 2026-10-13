"""The ``fm setup`` wizard (docs/PLUG_AND_PLAY.md section 6): from a fresh checkout to a running installation.

:func:`run_setup` walks the steps in order, printing through an injected ``echo`` and asking through an injected
:class:`Prompts`:

1. **Session.** The browser profile's ESPN session is loaded; none, or an expired one, runs the ``fm login`` flow, and
   an expiring one offers renewal.
2. **Leagues.** :func:`fm.espn.discover.discover_leagues` lists the account's current NFL and NBA leagues, each shown
   as ``NFL  <league name>  -  team 7 "<team name>"``; a league whose team the fan profile did not name is confirmed
   with :func:`fm.espn.discover.confirm_league` (which refuses a team the account does not own). When discovery fails
   or finds nothing, the ``--league-url`` values are confirmed instead, or the wizard asks for URLs. Every new league
   is selected by default.
3. **Presets.** Each new league gets a short key (``nfl``, ``nba``, then ``nfl2``...) and a preset (default
   ``balanced``; ``--preset`` sets it for all), shown with its one-line description.
4. **API keys.** ``ANTHROPIC_API_KEY`` (hidden, skippable: the advisor falls back to template rationales) and the
   optional ``ODDS_API_KEY`` go to ``.env`` through :func:`dotenv.set_key`. A key already set is left alone, and no
   value is ever printed.
5. **Phone.** ntfy (the default: no account), Telegram (asks for the @BotFather token, then captures the chat) or
   none, through :mod:`fm.notify.setup`, the code ``fm notify setup`` runs. A channel whose secrets are already in
   ``.env`` is not set up again (``fm notify setup --replace`` does that).
6. **Config.** A missing ``config.toml`` is written fresh by :func:`fm.config_edit.render_config`. An existing one is
   never rewritten: :func:`fm.config_edit.add_leagues` appends only the new leagues, a known league whose season or
   team differ from ESPN's is shown and updated through :func:`fm.config_edit.set_values` only once confirmed (``--yes``
   confirms), and :func:`fm.config_edit.set_notify_channel` runs only when the channel changed.
7. **First sync** of each new or updated league, one at a time so each shows its progress (``--no-sync`` skips).
8. **Autostart.** ``fm run`` at logon (``fm autostart install``) and the wake timer (``fm schedule install``), each
   installed only when the scheduler does not already have it (``--no-autostart`` skips). A running ``fm bot`` is
   reported, not stopped: ``fm run`` replaces it, and the two together would both poll the phone channel.
9. **Summary.** The ``fm doctor`` lines, then where to go next.

**Re-running is idempotent.** With nothing new, no file is written: the leagues are all present, the keys are set, the
channel is set up, nothing needs a first sync, and the jobs are installed, so ``config.toml`` stays byte-identical.

**Every external effect is injected** through :class:`SetupDeps` (session load and login, discovery, league
confirmation, phone setup, sync, the two scheduler jobs, the ``fm bot`` probe, the doctor), so the tests drive the
whole flow offline; ``fm.commands.setup`` wires the live ones. ``--yes`` swaps the prompts for :class:`AcceptDefaults`,
which answers every question with its default and skips every secret. Nothing here writes to ESPN: discovery,
confirmation and sync only read.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from dotenv import dotenv_values, set_key
from pydantic import ValidationError

from fm.browser.session import BrowserError
from fm.config import SECRET_ENV_VARS, Channel, Config, ConfigError, Secrets, Sport, config_paths, load_config
from fm.config_edit import add_leagues, render_config, set_notify_channel, set_values, write_config
from fm.espn.auth import AuthError, EspnSession, NotLoggedInError, SessionStatus, describe
from fm.espn.discover import DiscoveredLeague, DiscoveryError, LeagueRef, parse_league_url
from fm.notify import NotifyError
from fm.notify.setup import DEFAULT_WAIT_S, Echo, saved_token, saved_topics, setup_ntfy, setup_telegram
from fm.presets import PRESET_DESCRIPTIONS, PRESET_NAMES, PresetName

DEFAULT_PRESET: PresetName = "balanced"
DEFAULT_PHONE: Channel = "ntfy"
"""The phone channel a fresh setup offers first: ntfy needs no account and no token."""
PHONE_CHANNELS: tuple[Channel, ...] = ("ntfy", "telegram", "none")
PHONE_DESCRIPTIONS: Mapping[Channel, str] = {
    "ntfy": "push notifications in the free ntfy app; no account, nothing to create",
    "telegram": "a Telegram bot you create with @BotFather (/newbot)",
    "none": "no phone: approve in fm proposals or the dashboard",
}
SUMMARY_HINT = "Dashboard: fm ui. Ask Claude: fm claude install."
BOT_COMMAND = re.compile(r"""(?:^|[\s\\/"'])fm(?:\.exe)?["']?\s+bot(?=$|[\s"'])""", re.IGNORECASE)
"""``fm bot`` in a process command line: ``...\\fm.exe bot``, ``uv run fm bot``, ``.../bin/fm bot --once``."""


@dataclass(frozen=True, slots=True)
class SecretPrompt:
    """One ``.env`` key the wizard asks for: the hidden question, and what skipping it means."""

    name: str
    question: str
    skipped: str


SECRET_PROMPTS: tuple[SecretPrompt, ...] = (
    SecretPrompt(
        "ANTHROPIC_API_KEY",
        "Anthropic API key for the advisor (hidden; Enter skips)",
        "the advisor writes template rationales until it is in .env",
    ),
    SecretPrompt(
        "ODDS_API_KEY",
        "The Odds API key (the-odds-api.com, free tier), optional (hidden; Enter skips)",
        "it only adds the sportsbooks' posted team totals; ESPN's lines are read either way",
    ),
)


class SetupError(RuntimeError):
    """Setup cannot go on: no session, no league, a config.toml that does not load, or an edit that failed."""


class StepError(RuntimeError):
    """One step failed (a sync, a job install) but the rest of setup goes on; the summary says what to re-run."""


class Prompts(Protocol):
    """How the wizard asks. The CLI prompts on the terminal; ``--yes`` (:class:`AcceptDefaults`) and tests answer
    without one."""

    def confirm(self, question: str, *, default: bool) -> bool: ...

    def choose(self, question: str, choices: Sequence[str], *, default: str) -> str: ...

    def text(self, question: str, *, default: str = "") -> str: ...

    def secret(self, question: str) -> str:
        """A hidden answer; an empty one skips."""
        ...


class AcceptDefaults:
    """``--yes``: every question takes its default and every secret is skipped, so nothing waits for the keyboard."""

    def confirm(self, question: str, *, default: bool) -> bool:
        return default

    def choose(self, question: str, choices: Sequence[str], *, default: str) -> str:
        return default

    def text(self, question: str, *, default: str = "") -> str:
        return default

    def secret(self, question: str) -> str:
        return ""


@dataclass(frozen=True, slots=True)
class SetupOptions:
    """The flags of ``fm setup``. ``preset`` and ``channel`` replace their questions when set."""

    league_urls: tuple[str, ...] = ()
    preset: PresetName | None = None
    channel: Channel | None = None
    sync: bool = True
    autostart: bool = True


@dataclass(frozen=True, slots=True)
class Job:
    """A scheduler job setup installs: ``installed`` asks the scheduler (``None`` when it cannot tell), ``install``
    registers it and returns report lines, raising :class:`StepError` when that fails. ``command`` is the CLI that
    does the same by hand."""

    name: str
    command: str
    installed: Callable[[], bool | None]
    install: Callable[[], Sequence[str]]


class ChannelSetup(Protocol):
    """Set up a phone channel, saving its secrets to ``env_path``; ``found`` is what ``.env`` holds now. Raises
    :class:`fm.notify.NotifyError` when it cannot finish."""

    def __call__(
        self,
        channel: Literal["telegram", "ntfy"],
        env_path: Path,
        found: Secrets,
        *,
        prompts: Prompts,
        echo: Echo,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SetupDeps:
    """Every external effect of the wizard. ``login`` runs the manual ``fm login`` sign-in and returns the new
    session; ``sync`` syncs the given league keys and returns report lines (raising :class:`StepError`);
    ``bot_pids`` lists running ``fm bot`` processes; ``doctor`` returns the ``fm doctor`` lines."""

    load_session: Callable[[], EspnSession]
    login: Callable[[], EspnSession]
    discover: Callable[[EspnSession], Sequence[DiscoveredLeague]]
    confirm_league: Callable[[LeagueRef, EspnSession], DiscoveredLeague]
    setup_channel: ChannelSetup
    sync: Callable[[Config, Sequence[str], EspnSession], Sequence[str]]
    autostart: Job
    wake_timer: Job
    bot_pids: Callable[[], Sequence[int]]
    doctor: Callable[[], Sequence[str]]


@dataclass(frozen=True, slots=True)
class PlannedLeague:
    """A league the run keeps: a new one (with the key and preset it gets) or one already in ``config.toml``."""

    league: DiscoveredLeague
    key: str
    team_id: int
    preset: PresetName | None = None
    present: bool = False

    def entry(self) -> dict[str, object]:
        """The ``[[league]]`` mapping :func:`fm.config_edit.render_config` and ``add_leagues`` take."""
        return {
            "key": self.key,
            "sport": self.league.sport,
            "espn_league_id": self.league.espn_league_id,
            "season": self.league.season,
            "team_id": self.team_id,
            "preset": self.preset,
        }


@dataclass(frozen=True, slots=True)
class SetupResult:
    """What a run did: the config written fresh or edited, the league keys by outcome, the phone channel, the leagues
    synced. ``declined`` lists present leagues whose season or team change the user did not confirm."""

    config_path: Path
    created: bool
    added: tuple[str, ...]
    present: tuple[str, ...]
    updated: tuple[str, ...]
    declined: tuple[str, ...]
    channel: Channel
    synced: tuple[str, ...]


def run_setup(
    options: SetupOptions,
    deps: SetupDeps,
    *,
    prompts: Prompts,
    echo: Echo,
    config_path: Path | None = None,
    env_path: Path | None = None,
    now: datetime | None = None,
) -> SetupResult:
    """Run every step (see the module docstring) and return what changed. ``config_path`` and ``env_path`` default to
    the config dir's files. Raises :class:`SetupError` when setup cannot go on; nothing is written before the leagues
    are settled, so a failure up to then leaves the installation as it was."""
    config_path, env_path = config_paths(config_path, env_path)
    at = now if now is not None else datetime.now(UTC)
    existing = _existing(config_path, env_path)

    _heading(echo, "ESPN session")
    session = _session(deps, prompts, echo, at)

    _heading(echo, "Leagues")
    plan = _plan(_find_leagues(options, deps, session, prompts, echo), existing, options, prompts, echo)
    if existing is None and not plan:
        raise SetupError("no league selected, so there is no config.toml to write; run fm setup again")

    _heading(echo, "API keys")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    _api_keys(env_path, prompts, echo)

    _heading(echo, "Phone")
    channel = _phone(options, existing, env_path, deps, prompts, echo)

    _heading(echo, "Config")
    written = _write_config(config_path, env_path, plan, channel, existing, prompts, echo)

    _heading(echo, "First sync")
    synced = _first_sync(options, config_path, env_path, (*written.added, *written.updated), deps, session, echo)

    _heading(echo, "Autostart")
    _autostart(options, deps, channel, echo)

    _heading(echo, "Health (fm doctor)")
    for line in deps.doctor():
        echo(line)
    echo("")
    echo(SUMMARY_HINT)
    return SetupResult(
        config_path=config_path,
        created=existing is None,
        added=written.added,
        present=written.present,
        updated=written.updated,
        declined=written.declined,
        channel=channel,
        synced=synced,
    )


# --- helpers the CLI and tests share ----------------------------------------------------------------------------------


def league_key(sport: Sport, taken: Iterable[str]) -> str:
    """The first free short key for a league of ``sport``: ``nfl``, then ``nfl2``, ``nfl3``..."""
    used = set(taken)
    if sport not in used:
        return sport
    number = 2
    while f"{sport}{number}" in used:
        number += 1
    return f"{sport}{number}"


def describe_league(league: DiscoveredLeague) -> str:
    """``NFL  <league name>  -  team 7 "<team name>"``; the league id stands in for a name ESPN did not give."""
    name = league.league_name or f"league {league.espn_league_id}"
    team = f"team {league.team_id}" if league.team_id is not None else "team ?"
    if league.team_name:
        team += f' "{league.team_name}"'
    return f"{league.sport.upper()}  {name}  -  {team}"


def read_secrets(env_path: Path, environ: Mapping[str, str] | None = None) -> Secrets:
    """The secrets ``.env`` holds now, overlaid with the process environment as ``fm.config`` reads them. Raises
    :class:`SetupError` naming (never showing) a value that does not validate."""
    env = os.environ if environ is None else environ
    file_values: Mapping[str, str | None] = dotenv_values(env_path) if env_path.is_file() else {}
    merged: dict[str, str] = {}
    for name in SECRET_ENV_VARS:
        value = env.get(name, file_values.get(name))
        if value is not None:
            merged[name] = value
    try:
        return Secrets.model_validate(merged)
    except ValidationError as exc:
        names = ", ".join(sorted({str(error["loc"][0]).upper() for error in exc.errors() if error["loc"]}))
        raise SetupError(f"{env_path}: invalid value for {names}; fix it and run fm setup again") from None


def notify_setup(
    channel: Literal["telegram", "ntfy"],
    env_path: Path,
    found: Secrets,
    *,
    prompts: Prompts,
    echo: Echo,
    timeout_s: float = DEFAULT_WAIT_S,
) -> None:
    """The live :class:`ChannelSetup`: :mod:`fm.notify.setup`, with the Telegram token asked for (hidden) when
    ``.env`` has none, and ``timeout_s`` to wait for the chat to send its code."""
    if channel == "telegram":
        setup_telegram(
            env_path,
            token=saved_token(found),
            chat_id=found.telegram_chat_id,
            echo=echo,
            timeout_s=timeout_s,
            ask_token=lambda: prompts.secret("Telegram bot token from @BotFather (hidden)"),
        )
    else:
        setup_ntfy(env_path, topics=saved_topics(found), echo=echo)


def parse_bot_pids(listing: str, *, exclude: Iterable[int] = ()) -> list[int]:
    """The pids of ``fm bot`` processes in a process listing of ``<pid> <command line>`` lines (``ps -A -o pid= -o
    command=``, or PowerShell's ``Win32_Process``), minus ``exclude`` (this process)."""
    skip = set(exclude)
    found: list[int] = []
    for line in listing.splitlines():
        pid, _, command = line.strip().partition(" ")
        if pid.isdigit() and int(pid) not in skip and BOT_COMMAND.search(command.strip()):
            found.append(int(pid))
    return found


# --- the steps --------------------------------------------------------------------------------------------------------


def _heading(echo: Echo, title: str) -> None:
    echo("")
    echo(f"== {title} ==")


def _existing(config_path: Path, env_path: Path) -> Config | None:
    """The config already in place, ``None`` when there is none; one that does not load stops setup untouched."""
    if not config_path.exists():
        return None
    try:
        return load_config(config_path, env_path=env_path)
    except ConfigError as exc:
        raise SetupError(
            f"{config_path} exists but does not load, so setup leaves it alone:\n{exc}\n"
            "Fix it (fm config check lists the problems) or move it away, then run fm setup again"
        ) from None


def _session(deps: SetupDeps, prompts: Prompts, echo: Echo, now: datetime) -> EspnSession:
    try:
        session = deps.load_session()
    except NotLoggedInError:
        echo("No ESPN session yet: a browser window opens; sign in to ESPN there.")
        return _login(deps)
    except (AuthError, BrowserError) as exc:
        raise SetupError(
            f"could not read the ESPN session ({exc}); if fm run is mid-tick, wait a minute and run fm setup again"
        ) from None
    echo(describe(session, now))
    match session.status(now):
        case SessionStatus.EXPIRED:
            echo("Signing in again: a browser window opens.")
            return _login(deps)
        case SessionStatus.EXPIRING if prompts.confirm("Renew the ESPN session now?", default=False):
            return _login(deps)
        case _:
            return session


def _login(deps: SetupDeps) -> EspnSession:
    try:
        return deps.login()
    except (AuthError, BrowserError) as exc:
        raise SetupError(f"the ESPN sign-in did not finish ({exc}); run fm setup again") from None


def _find_leagues(
    options: SetupOptions, deps: SetupDeps, session: EspnSession, prompts: Prompts, echo: Echo
) -> list[DiscoveredLeague]:
    """The account's leagues, from discovery and the pasted URLs, each with its team settled."""
    found: list[DiscoveredLeague] = []
    try:
        discovered = list(deps.discover(session))
    except DiscoveryError as exc:
        echo(f"Could not discover your leagues: {exc}")
        discovered = None
    else:
        if not discovered:
            echo("ESPN lists no current NFL or NBA league for this account.")
            discovered = None
    for league in discovered or ():
        confirmed = league if league.team_id is not None else _confirm(deps, league.ref, session, echo)
        if confirmed is not None:
            found.append(confirmed)
    urls = list(options.league_urls)
    if discovered is None and not urls:
        echo("Paste the URL of each league instead (your team page, or any league page with leagueId= in it);")
        echo("an empty line ends the list. Next time: fm setup --league-url <url>.")
        while url := prompts.text("League URL", default="").strip():
            urls.append(url)
    for url in urls:
        try:
            ref = parse_league_url(url)
        except DiscoveryError as exc:
            echo(f"Skipped {url}: {exc}")
            continue
        confirmed = _confirm(deps, ref, session, echo)
        if confirmed is not None:
            found.append(confirmed)
    unique: dict[tuple[str, int], DiscoveredLeague] = {}
    for league in found:
        if league.team_id is not None:
            unique.setdefault((league.sport, league.espn_league_id), league)
    if not unique:
        raise SetupError(
            "no league to set up: pass the URL of your team page with --league-url (repeatable), e.g. "
            "fm setup --league-url 'https://fantasy.espn.com/football/team?leagueId=...&teamId=...'"
        )
    return list(unique.values())


def _confirm(deps: SetupDeps, ref: LeagueRef, session: EspnSession, echo: Echo) -> DiscoveredLeague | None:
    try:
        return deps.confirm_league(ref, session)
    except DiscoveryError as exc:
        echo(f"Skipped {ref.sport.upper()} league {ref.espn_league_id}: {exc}")
        return None


def _plan(
    found: Sequence[DiscoveredLeague],
    existing: Config | None,
    options: SetupOptions,
    prompts: Prompts,
    echo: Echo,
) -> list[PlannedLeague]:
    """Keep the leagues already configured and ask about each new one: manage it? with which preset?"""
    present = {(league.sport, league.espn_league_id): league.key for league in existing.leagues} if existing else {}
    taken = set(present.values())
    plan: list[PlannedLeague] = []
    new: list[tuple[DiscoveredLeague, int]] = []
    for league in found:
        team_id = league.team_id
        if team_id is None:  # _find_leagues settles every team; this keeps the type honest
            continue
        key = present.get((league.sport, league.espn_league_id))
        if key is None:
            echo(f"  {describe_league(league)}")
            new.append((league, team_id))
        else:
            echo(f"  {describe_league(league)}  (in config.toml as {key})")
            plan.append(PlannedLeague(league, key, team_id, present=True))
    if not new:
        echo("No new league.")
        return plan
    if options.preset is None:
        echo("Presets decide how much happens without asking you (fm config set-preset changes one later):")
        for name in PRESET_NAMES:
            echo(f"  {name}: {PRESET_DESCRIPTIONS[name]}")
    for league, team_id in new:
        line = describe_league(league)
        if not prompts.confirm(f"Manage {line}?", default=True):
            echo(f"Skipped {line}.")
            continue
        key = league_key(league.sport, taken)
        taken.add(key)
        preset = options.preset
        if preset is None:
            preset = _preset(prompts.choose(f"Preset for {key}", PRESET_NAMES, default=DEFAULT_PRESET))
        echo(f"  {key}: {line}, preset {preset}")
        plan.append(PlannedLeague(league, key, team_id, preset=preset))
    return plan


def _preset(name: str) -> PresetName:
    for preset in PRESET_NAMES:
        if preset == name:
            return preset
    raise SetupError(f"unknown preset {name!r}; choose one of {', '.join(PRESET_NAMES)}")


def _api_keys(env_path: Path, prompts: Prompts, echo: Echo) -> None:
    found = read_secrets(env_path)
    for prompt in SECRET_PROMPTS:
        if found.is_set(prompt.name):
            echo(f"{prompt.name} is already set.")
            continue
        value = prompts.secret(prompt.question).strip()
        if not value:
            echo(f"Skipped {prompt.name}: {prompt.skipped}.")
            continue
        set_key(env_path, prompt.name, value, quote_mode="never")
        echo(f"Saved {prompt.name} to {env_path}.")


def _phone(
    options: SetupOptions, existing: Config | None, env_path: Path, deps: SetupDeps, prompts: Prompts, echo: Echo
) -> Channel:
    """The channel to configure, set up unless ``.env`` already holds its secrets. A setup that does not finish is a
    warning: the channel is still configured, and ``fm doctor`` points at ``fm notify setup``."""
    channel = options.channel
    if channel is None:
        default = existing.notify.channel if existing is not None else DEFAULT_PHONE
        echo("Where should proposals and alerts reach you?")
        for name in PHONE_CHANNELS:
            echo(f"  {name}: {PHONE_DESCRIPTIONS[name]}")
        channel = _channel(prompts.choose("Phone", PHONE_CHANNELS, default=default))
    if channel == "none":
        echo("No phone: proposals wait for fm proposals or the dashboard; fm notify setup --channel ntfy adds one.")
        return channel
    found = read_secrets(env_path)
    if _channel_ready(channel, found):
        echo(f"{channel} is already set up (fm notify setup --replace sets it up afresh).")
        return channel
    try:
        deps.setup_channel(channel, env_path, found, prompts=prompts, echo=echo)
    except NotifyError as exc:
        echo(f"warning: {channel} is not set up yet: {exc}")
        echo(f"Run fm notify setup --channel {channel} to finish it.")
    return channel


def _channel(name: str) -> Channel:
    for channel in PHONE_CHANNELS:
        if channel == name:
            return channel
    raise SetupError(f"unknown phone channel {name!r}; choose one of {', '.join(PHONE_CHANNELS)}")


def _channel_ready(channel: Channel, found: Secrets) -> bool:
    if channel == "telegram":
        return found.telegram_bot_token is not None and found.telegram_chat_id is not None
    if channel == "ntfy":
        return saved_topics(found) is not None
    return True


@dataclass(frozen=True, slots=True)
class _Written:
    added: tuple[str, ...] = ()
    present: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    declined: tuple[str, ...] = ()


def _write_config(
    config_path: Path,
    env_path: Path,
    plan: Sequence[PlannedLeague],
    channel: Channel,
    existing: Config | None,
    prompts: Prompts,
    echo: Echo,
) -> _Written:
    try:
        if existing is None:
            return _fresh_config(config_path, env_path, plan, channel, echo)
        return _edit_config(config_path, env_path, plan, channel, existing, prompts, echo)
    except ConfigError as exc:
        raise SetupError(str(exc)) from None


def _fresh_config(
    config_path: Path, env_path: Path, plan: Sequence[PlannedLeague], channel: Channel, echo: Echo
) -> _Written:
    write_config(config_path, render_config([league.entry() for league in plan], notify=channel), env_path=env_path)
    keys = tuple(league.key for league in plan)
    echo(f"Wrote {config_path}: {', '.join(keys)}; phone {channel}.")
    return _Written(added=keys)


def _edit_config(
    config_path: Path,
    env_path: Path,
    plan: Sequence[PlannedLeague],
    channel: Channel,
    existing: Config,
    prompts: Prompts,
    echo: Echo,
) -> _Written:
    result = add_leagues(config_path, [league.entry() for league in plan], env_path=env_path)
    if result.added:
        echo(f"Added to {config_path}: {', '.join(result.added)}.")
    updated: list[str] = []
    declined: list[str] = []
    for update in result.changed:
        diff = ", ".join(f"{name} {old} -> {new}" for name, (old, new) in update.fields.items())
        echo(f"{update.key} in config.toml differs from ESPN: {diff}.")
        if prompts.confirm(f"Update {update.key} ({diff})?", default=True):
            set_values(config_path, update.key, update.updates, env_path=env_path)
            echo(f"Updated {update.key}.")
            updated.append(update.key)
        else:
            echo(f"Left {update.key} as it is.")
            declined.append(update.key)
    if existing.notify.channel != channel:
        set_notify_channel(config_path, channel, env_path=env_path)
        echo(f'Set channel = "{channel}" under [notify] (was "{existing.notify.channel}").')
    if not result.added and not updated and existing.notify.channel == channel:
        echo(f"{config_path} already has everything; left unchanged.")
    return _Written(added=result.added, present=result.unchanged, updated=tuple(updated), declined=tuple(declined))


def _first_sync(
    options: SetupOptions,
    config_path: Path,
    env_path: Path,
    keys: Sequence[str],
    deps: SetupDeps,
    session: EspnSession,
    echo: Echo,
) -> tuple[str, ...]:
    if not keys:
        echo("Nothing new to sync.")
        return ()
    if not options.sync:
        echo(f"Skipped (--no-sync): fm sync --league <key> syncs {', '.join(keys)}, or the first tick does.")
        return ()
    try:
        config = load_config(config_path, env_path=env_path)
    except ConfigError as exc:
        raise SetupError(str(exc)) from None
    synced: list[str] = []
    for key in keys:
        echo(f"Syncing {key}: settings, rosters, the player pool and projections (the first one takes a while)...")
        try:
            lines = deps.sync(config, [key], session)
        except StepError as exc:
            echo(f"warning: {key} did not sync: {exc}; fm sync --league {key} retries")
            continue
        for line in lines:
            echo(f"  {line}")
        synced.append(key)
    return tuple(synced)


def _autostart(options: SetupOptions, deps: SetupDeps, channel: Channel, echo: Echo) -> None:
    if not options.autostart:
        echo("Skipped (--no-autostart): fm autostart install starts fm run at logon; fm schedule install adds the")
        echo("wake timer.")
        return
    running = list(deps.bot_pids())
    if running:
        pids = ", ".join(str(pid) for pid in running)
        listens = f"the {channel} channel" if channel != "none" else "the phone channel"
        echo(f"note: fm bot is running (pid {pids}). fm run replaces it: it ticks and records approvals itself, and")
        echo(f"the two together would both poll {listens}. Stop fm bot and take it out of what starts at logon;")
        echo("setup leaves it running.")
    for job in (deps.autostart, deps.wake_timer):
        if job.installed():
            echo(f"{job.name}: already installed ({job.command} reinstalls it).")
            continue
        try:
            lines = job.install()
        except StepError as exc:
            echo(f"warning: {job.name} is not installed: {exc}; run {job.command}")
            continue
        for line in lines:
            echo(line)


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
    "parse_bot_pids",
    "describe_league",
    "league_key",
    "notify_setup",
    "read_secrets",
    "run_setup",
]
