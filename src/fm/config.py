"""Configuration: ``config.toml`` plus ``.env``, validated into frozen pydantic models (DESIGN section 14).

``config.toml`` declares the leagues (``[[league]]``, each with a ``[league.policy]``), ``[llm]`` and ``[notify]``.
``.env`` next to it holds the secrets: ``ANTHROPIC_API_KEY``, ``ODDS_API_KEY`` and the phone channel's credentials
(``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_CHAT_ID``, or ``NTFY_TOPIC`` + ``NTFY_REPLY_TOPIC``). A variable set in the
process environment wins over the file. Nothing ESPN-related is configured here: the session lives in the browser
profile (``fm.paths.browser_profile_dir``).

Validation is strict so a typo cannot silently fall back to a default: unknown keys are errors, league keys are
unique, ``auto`` exists only for lineup kinds, and trades have no policy knob at all because they are approval-only
(CLAUDE.md). ``load_config`` raises ``ConfigError`` listing every problem; ``fm config check`` prints it.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from fm import paths

Sport = Literal["nfl", "nba"]
EspnGame = Literal["ffl", "fba"]
Channel = Literal["telegram", "ntfy"]
Approval = Literal["off", "approve", "auto"]
ApprovalNoAuto = Literal["off", "approve"]

ESPN_GAME: dict[Sport, EspnGame] = {"nfl": "ffl", "nba": "fba"}
DEFAULT_MODEL = "claude-opus-5-5"

# Short name used on the command line (``--league nfl``) and as the store key.
LeagueKey = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")]

# Every TOML table: unknown keys are errors, instances are immutable, Python callers may use field names for aliases.
_TABLE = ConfigDict(extra="forbid", frozen=True, validate_by_name=True)


class ConfigError(Exception):
    """config.toml or .env is missing, unreadable or invalid. The message lists every problem, one per line."""


class Policy(BaseModel):
    """Per-league approval policy and guardrails (DESIGN section 11).

    ``auto`` exists only for the lineup kinds; add/drop and waiver claims are approved or off. Trades are approval-only
    by invariant and deliberately have no field here: any ``trade*`` key is rejected.
    """

    model_config = _TABLE

    bench_inactive: Approval = Field(
        default="auto",
        description="Benching an OUT, bye or no-game starter. auto fires only at T-15 when the proposal is unanswered.",
    )
    lineup: Approval = Field(default="auto", description="Other lineup optimizations.")
    add_drop: ApprovalNoAuto = Field(default="approve", description="Free-agent adds and drops, including streaming.")
    waiver: ApprovalNoAuto = Field(default="approve", description="Waiver claims.")
    max_transactions_per_week: int = Field(default=3, ge=0, description="Our cap; ESPN's own limit is in settings.")
    max_faab_pct_per_bid: float = Field(
        default=0.35, ge=0, le=1, description="Largest single bid as a fraction of the season FAAB budget."
    )
    untouchables: tuple[str | int, ...] = Field(
        default=(), description="Player names or ESPN player IDs never dropped or traded; resolved at sync."
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_trade_policy(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            offending = sorted(str(key) for key in data if str(key).startswith("trade"))
            if offending:
                raise ValueError(f"{', '.join(offending)}: trades are approval-only and not configurable (CLAUDE.md)")
        return data


class League(BaseModel):
    """One ESPN league and the team this tool manages in it."""

    model_config = _TABLE

    key: LeagueKey
    sport: Sport
    espn_league_id: int = Field(ge=1, description="leagueId= in the league URL.")
    season: int = Field(ge=2000, le=2100, description="ESPN labels NBA seasons by their end year.")
    team_id: int = Field(ge=1, description="teamId= on the team page.")
    policy: Policy = Field(default_factory=Policy)

    @property
    def game(self) -> EspnGame:
        """ESPN game key used in API URLs: ``ffl`` for NFL, ``fba`` for NBA."""
        return ESPN_GAME[self.sport]


class Llm(BaseModel):
    """Claude settings (DESIGN section 10). Effort is set per worker in code, not here."""

    model_config = _TABLE

    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    daily_budget_usd: float = Field(default=2.0, ge=0, description="Daily spend cap across all workers.")


class Notify(BaseModel):
    """Phone channel (DESIGN section 11). Credentials and topic names come from ``.env``."""

    model_config = _TABLE

    channel: Channel = "telegram"


class Secrets(BaseModel):
    """Values from ``.env``, addressed by their environment variable names (``Secrets.model_validate(mapping)``).

    Unknown names are ignored, blank values count as unset, and the values never appear in ``repr`` or logs.
    """

    model_config = ConfigDict(extra="ignore", frozen=True, alias_generator=str.upper, validate_by_name=True)

    anthropic_api_key: SecretStr | None = None
    odds_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: int | None = Field(default=None, description="The one chat the bot answers; fm notify setup.")
    ntfy_topic: SecretStr | None = Field(default=None, description="Topic the phone subscribes to; the name is secret.")
    ntfy_reply_topic: SecretStr | None = Field(default=None, description="Topic the PC listens on for button presses.")

    @field_validator("*", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    def is_set(self, env_name: str) -> bool:
        """True when the variable (``TELEGRAM_BOT_TOKEN`` style name) has a value."""
        if env_name not in SECRET_ENV_VARS:
            raise KeyError(f"{env_name} is not a known secret; known: {', '.join(SECRET_ENV_VARS)}")
        return getattr(self, env_name.lower()) is not None


SECRET_ENV_VARS: tuple[str, ...] = tuple(name.upper() for name in Secrets.model_fields)
OPTIONAL_SECRET_ENV_VARS: tuple[str, ...] = ("ODDS_API_KEY",)


class Config(BaseModel):
    """The whole configuration: the ``config.toml`` tables plus the secrets from ``.env``."""

    model_config = _TABLE

    leagues: tuple[League, ...] = Field(default=(), validation_alias="league")  # TOML ``[[league]]``
    llm: Llm = Field(default_factory=Llm)
    notify: Notify = Field(default_factory=Notify)
    secrets: Secrets = Field(default_factory=Secrets)  # filled from .env by load_config, never from the TOML

    @model_validator(mode="after")
    def _leagues_are_distinct(self) -> Self:
        # Runs only once every league validated, so a bad league is reported once rather than also as "too few".
        if not self.leagues:
            raise ValueError("at least one [[league]] table is required")
        keys: set[str] = set()
        espn: dict[tuple[Sport, int], str] = {}
        for league in self.leagues:
            if league.key in keys:
                raise ValueError(f"duplicate league key {league.key!r}")
            keys.add(league.key)
            first = espn.setdefault((league.sport, league.espn_league_id), league.key)
            if first != league.key:
                raise ValueError(
                    f"leagues {first!r} and {league.key!r} point at the same ESPN {league.sport} league "
                    f"{league.espn_league_id}; this tool manages one team per league"
                )
        return self

    def league(self, key: str) -> League:
        """The league configured under ``key``; ``KeyError`` names the known keys."""
        for league in self.leagues:
            if league.key == key:
                return league
        raise KeyError(f"no league {key!r} in config.toml; known: {', '.join(lg.key for lg in self.leagues)}")

    def required_secrets(self) -> tuple[str, ...]:
        """Environment variables this configuration needs: the Anthropic key plus the chosen channel's credentials."""
        channel = (
            ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
            if self.notify.channel == "telegram"
            else ("NTFY_TOPIC", "NTFY_REPLY_TOPIC")
        )
        return ("ANTHROPIC_API_KEY", *channel)

    def missing_secrets(self) -> tuple[str, ...]:
        """Required variables that neither ``.env`` nor the environment provides. Empty means setup is complete."""
        return tuple(name for name in self.required_secrets() if not self.secrets.is_set(name))


def config_paths(path: Path | None = None, env_path: Path | None = None) -> tuple[Path, Path]:
    """Resolve ``(config.toml, .env)``: the config dir's files by default; ``.env`` sits next to an explicit config.

    ``fm.paths`` is only consulted when ``path`` is omitted, so checking an explicit file never creates the config dir.
    """
    config_path = path if path is not None else paths.config_file()
    return config_path, (env_path if env_path is not None else config_path.with_name(".env"))


def load_config(
    path: Path | None = None, *, env_path: Path | None = None, environ: Mapping[str, str] | None = None
) -> Config:
    """Read and validate ``config.toml`` and ``.env``.

    ``path`` defaults to ``$FM_CONFIG_DIR/config.toml`` and ``env_path`` to the ``.env`` beside it. Secret variables
    found in ``environ`` (default ``os.environ``) override the file, so a shell export wins. Raises ``ConfigError``.
    """
    config_path, env_path = config_paths(path, env_path)
    data = _read_toml(config_path)
    if "secrets" in data:
        raise ConfigError(f"{config_path}: [secrets] belongs in {env_path}, not in config.toml")
    data["secrets"] = _read_secrets(env_path, environ)
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_describe(exc, config_path, env_path)) from exc


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError(f"{path} not found; create it from the template in DESIGN.md section 14") from None
    except OSError as exc:
        raise ConfigError(f"{path}: {exc.strerror or exc}") from exc
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc


def _read_secrets(env_path: Path, environ: Mapping[str, str] | None) -> dict[str, str]:
    """The known secret variables from ``.env`` overlaid with the process environment; absent names are left out."""
    environ = os.environ if environ is None else environ
    file_values: Mapping[str, str | None] = dotenv_values(env_path) if env_path.is_file() else {}
    merged: dict[str, str] = {}
    for name in SECRET_ENV_VARS:
        value = environ.get(name, file_values.get(name))
        if value is not None:
            merged[name] = value
    return merged


def _describe(exc: ValidationError, config_path: Path, env_path: Path) -> str:
    errors = exc.errors(include_url=False)
    lines = [f"invalid configuration ({len(errors)} problem{'s' if len(errors) != 1 else ''}):"]
    for error in errors:
        loc = error["loc"]
        if loc and loc[0] == "secrets":
            where, loc, got = env_path, loc[1:], ""  # never echo a secret value back
        else:
            where, got = config_path, _got(error["input"])
        label = _dotted(loc)
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"  {where}: {label + ': ' if label else ''}{message}{got}")
    return "\n".join(lines)


def _dotted(loc: tuple[int | str, ...]) -> str:
    """``("league", 0, "policy", "lineup")`` -> ``league[0].policy.lineup``."""
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else part
    return out


def _got(value: Any) -> str:
    if isinstance(value, str | int | float | bool):
        shown = repr(value)
        return f" (got {shown if len(shown) <= 60 else shown[:57] + '...'})"
    return ""
