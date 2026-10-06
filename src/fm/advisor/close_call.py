"""The ``close_call`` worker: Claude breaks a near-tie between lineup candidates with cited web search (DESIGN §10).

When the lineup engine has scored two or more of our own rostered players within a small margin for one start/sit
decision, :func:`decide_close_call` asks Claude, with the Claude API web search server tool limited to a domain
allowlist, which of them the freshest reporting favors. The answer is a :class:`CloseCallResult`, advisory like every
advisor output: the decision module may use :attr:`CloseCallResult.choice` as the tie-break, and nothing here writes
to ESPN or creates a proposal.

**How the output is bounded** (CLAUDE.md: bounded, cited, logged; a Claude-only signal never triggers a drop or a
trade):

- *Only near-ties are asked.* A :class:`CloseCall` cannot be built unless it has two or more options, all within its
  ``margin`` of the engine's best score (:func:`near_ties` picks them from a larger field). The engine's own top
  option is always one of them.
- *Only the options can be chosen.* The pick must be the ESPN id of one of the options; anything else is
  ``rejected``. There is no way to name a player outside the call, so there is no way to ask for an add, a drop or a
  trade, and :attr:`CloseCallResult.choice` is an option whatever happened.
- *Cited or ignored.* A pick stands only with at least one finding that names an option, has a claim, and cites an
  ``http(s)`` page on an allowlisted domain (the same list the search tool is restricted to, so a page the search
  cannot have returned is dropped here too). A pick with no such finding is ``rejected`` and the engine's choice
  stands. The valid sources come back as :class:`Source` rows for the proposal's rationale.
- *Confident or ignored.* A pick below :data:`CLOSE_CALL_MIN_CONFIDENCE` is ``no_change``, as is a null pick.
- *Logged.* Every outcome is logged with its sources, and the call is recorded in ``llm_usage`` by the client.

The statuses of :class:`fm.advisor.client.WorkerReply` map to ``failed`` (refusal, ``max_tokens`` and the other
non-``ok`` statuses, with the reason: no partial data is read, and a ``pause_turn`` from the search loop is a
failure, not something to resume), and :class:`fm.advisor.client.BudgetExceededError` to ``blocked``. In both the
engine's choice stands, which is the fallback.

**The search tool.** The client's :class:`fm.advisor.client.Prompt` carries no tools, so this module adds
:class:`SearchPrompt` (a prompt that also carries the allowlist and the ``max_uses`` cap, and puts the tool in the
call parameters) and :class:`SearchSdkTransport` (the SDK transport that sends them). :func:`close_call_client` builds
an :class:`~fm.advisor.client.AdvisorClient` over it; :func:`decide_close_call` refuses a client whose transport is
the plain one, which would drop the tool and let the model answer without searching. The allowlist is a parameter
defaulting to :data:`DEFAULT_ALLOWED_DOMAINS` because ``fm.config.Llm`` has no field for it yet.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Final, Literal, NotRequired, cast
from urllib.parse import urlsplit

import anthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fm.advisor.client import (
    FALLBACK_BETA,
    AdvisorClient,
    AdvisorError,
    BudgetExceededError,
    CallParams,
    Prompt,
    RawReply,
    SdkTransport,
    TokenUsage,
    Transport,
    WorkerReply,
    effort_for,
    reply_from_message,
)
from fm.advisor.prompts import prompt_text
from fm.config import Config, Sport
from fm.decide.lineup import LineupCandidate
from fm.store import Store, utc_now

logger = logging.getLogger(__name__)

CLOSE_CALL_WORKER: Final = "close_call"
CLOSE_CALL_MAX_TOKENS: Final = 4096
"""The answer is a pick, a short rationale and a few findings; the search results count as input, not output."""
CLOSE_CALL_MIN_CONFIDENCE: Final = 0.6
"""A pick below this confidence is ignored (``no_change``): 0.5 is a coin flip and the engine is the default."""
CLOSE_CALL_MAX_SEARCHES: Final = 5
"""The search tool's ``max_uses`` cap per call."""
WEB_SEARCH_TOOL: Final = "web_search_20260209"
"""The web search server tool version (DESIGN §10). ``web_search_20250305`` is the older one and
``web_search_20260318`` adds ``response_inclusion`` for code-execution callers, which this worker does not use."""
DEFAULT_ALLOWED_DOMAINS: Final[tuple[str, ...]] = (
    "espn.com",
    "nfl.com",
    "nba.com",
    "rotowire.com",
    "cbssports.com",
)
"""The domains the search is restricted to (DESIGN §10). A domain covers its subdomains (``www.espn.com``)."""

type CloseCallStatus = Literal["pick", "no_change", "rejected", "failed", "blocked"]
"""``pick``: a cited, confident pick among the options; ``no_change``: nothing decisive (null pick or low
confidence); ``rejected``: the answer broke a rule (uncited, not an option); ``failed``: no usable answer;
``blocked``: the daily budget is spent. Only ``pick`` can change the engine's choice."""


# --- the web search tool ----------------------------------------------------------------------------------------------


class WebSearchTool(BaseModel):
    """The ``tools`` entry for the web search server tool, restricted to ``allowed_domains``."""

    model_config = ConfigDict(frozen=True)

    type: str = WEB_SEARCH_TOOL
    name: Literal["web_search"] = "web_search"
    max_uses: int
    allowed_domains: tuple[str, ...]


def normalize_domains(domains: Sequence[str]) -> tuple[str, ...]:
    """Lower-cased bare domains without scheme, path or a leading ``www.``, deduplicated in order. An empty list is
    refused: for the search tool no ``allowed_domains`` means the whole web."""
    seen: dict[str, None] = {}
    for raw in domains:
        text = raw.strip().lower()
        host = urlsplit(text if "//" in text else f"//{text}").hostname or ""
        host = host.removeprefix("www.")
        if not host or "." not in host:
            raise ValueError(f"not a domain: {raw!r}")
        seen.setdefault(host)
    if not seen:
        raise ValueError("the web search needs a non-empty domain allowlist (an empty one would mean the whole web)")
    return tuple(seen)


def domain_allowed(url: str, allowed_domains: Sequence[str]) -> bool:
    """True for an ``http(s)`` URL whose host is an allowlisted domain or a subdomain of one."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if parts.scheme not in ("http", "https") or not host:
        return False
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


class SearchCallParams(CallParams):
    """:class:`~fm.advisor.client.CallParams` plus the server tools of the call."""

    tools: NotRequired[list[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class SearchPrompt(Prompt):
    """A :class:`~fm.advisor.client.Prompt` whose call also enables the web search tool over ``allowed_domains``."""

    allowed_domains: tuple[str, ...] = ()
    max_uses: int = CLOSE_CALL_MAX_SEARCHES

    def __post_init__(self) -> None:
        Prompt.__post_init__(self)
        if self.max_uses < 1:
            raise ValueError(f"max_uses must be positive, got {self.max_uses!r}")
        object.__setattr__(self, "allowed_domains", normalize_domains(self.allowed_domains))

    @property
    def tool(self) -> WebSearchTool:
        return WebSearchTool(max_uses=self.max_uses, allowed_domains=self.allowed_domains)

    def params(self, model: str) -> SearchCallParams:
        base = Prompt.params(self, model)
        return SearchCallParams(**base, tools=[self.tool.model_dump(mode="json")])


class SearchSdkTransport(SdkTransport):
    """:class:`~fm.advisor.client.SdkTransport` that also sends the ``tools`` of a :class:`SearchPrompt`; a call
    without tools goes the plain way."""

    def parse(self, params: CallParams, output: type[BaseModel], *, fallback: bool = False) -> RawReply:
        tools = cast(SearchCallParams, params).get("tools")
        if not tools:
            return super().parse(params, output, fallback=fallback)
        kwargs: dict[str, Any] = {
            "model": params["model"],
            "max_tokens": params["max_tokens"],
            "system": params["system"],
            "messages": params["messages"],
            "output_config": params["output_config"],
            "output_format": output,
            "tools": tools,
        }
        try:
            message: Any
            if fallback:
                message = self._sdk.beta.messages.parse(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                message = self._sdk.messages.parse(**kwargs)
        except ValidationError as exc:
            return RawReply(
                request_id=None,
                model=str(params["model"]),
                stop_reason=None,
                usage=TokenUsage(),
                detail=f"SDK could not parse the answer as {output.__name__} ({exc.error_count()} errors)",
            )
        except anthropic.APIError as exc:
            raise AdvisorError(f"Claude API call failed: {exc}") from exc
        return reply_from_message(message, output)


def close_call_client(store: Store, config: Config, *, transport: Transport | None = None) -> AdvisorClient:
    """An :class:`~fm.advisor.client.AdvisorClient` for the configured model and budget whose transport sends the
    search tool (:class:`SearchSdkTransport` over the ``.env`` key, unless ``transport`` is given)."""
    if transport is None:
        key = config.secrets.anthropic_api_key
        if key is None:
            raise AdvisorError("ANTHROPIC_API_KEY is not set in .env; the advisor cannot run (fm config check)")
        transport = SearchSdkTransport(
            anthropic.Anthropic(api_key=key.get_secret_value(), timeout=120.0, max_retries=2)
        )
    return AdvisorClient(store, config.llm, transport)


# --- the structured output --------------------------------------------------------------------------------------------


class CloseCallFinding(BaseModel):
    """One claim the pick rests on, with its source (the schema the API is held to; the rules are applied in code)."""

    model_config = ConfigDict(extra="forbid")

    espn_id: int = Field(description="The option the claim is about.")
    claim: str = Field(description="One sentence of what the source says.")
    source_url: str = Field(description="The exact URL of the page that says it.")
    source_title: str
    published_at: str | None = Field(default=None, description="The page's own date or time, ISO 8601 when possible.")


class CloseCallOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pick_espn_id: int | None = Field(description="The ESPN id of one option, or null when nothing is decisive.")
    confidence: float = Field(description="0 to 1.")
    rationale: str
    findings: list[CloseCallFinding]


# --- what a call asks -------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CloseCallOption:
    """One tied candidate as the prompt names him: ``score`` is the engine's value (higher is better, in the units
    the engine used, usually expected league points)."""

    espn_id: int
    name: str
    score: float
    position: str | None = None
    pro_team: str | None = None
    opponent: str | None = None
    p_active: float | None = None
    designation: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError(f"option {self.espn_id}: score must be finite, got {self.score!r}")

    def describe(self) -> str:
        parts = [f"{self.espn_id}: {self.name}"]
        if self.position:
            parts.append(self.position)
        if self.pro_team:
            parts.append(self.pro_team)
        if self.opponent:
            parts.append(f"vs {self.opponent}")
        parts.append(f"engine score {self.score:.2f}")
        if self.p_active is not None:
            parts.append(f"chance of playing {self.p_active:.0%}")
        if self.designation:
            parts.append(f"designation {self.designation}")
        if self.note:
            parts.append(self.note)
        return ", ".join(parts)


def option_from_candidate(
    candidate: LineupCandidate, *, pro_team: str | None = None, opponent: str | None = None
) -> CloseCallOption:
    """The option for a lineup candidate: his expected points are the engine's score."""
    return CloseCallOption(
        espn_id=candidate.espn_id,
        name=candidate.label,
        score=candidate.expected,
        position=candidate.position,
        pro_team=pro_team,
        opponent=opponent,
        p_active=candidate.p_active,
        designation=candidate.designation,
    )


def near_ties(options: Sequence[CloseCallOption], margin: float) -> tuple[CloseCallOption, ...]:
    """The options within ``margin`` of the best score, best first (ties keep their order); empty unless at least
    two qualify, since one option is no call."""
    if margin < 0 or not math.isfinite(margin):
        raise ValueError(f"margin must be finite and non-negative, got {margin!r}")
    if not options:
        return ()
    ranked = sorted(options, key=lambda option: -option.score)
    tied = tuple(option for option in ranked if ranked[0].score - option.score <= margin)
    return tied if len(tied) >= 2 else ()


@dataclass(frozen=True, slots=True)
class CloseCall:
    """One start/sit decision the engine cannot separate: ``options`` (two or more, all within ``margin`` of the best
    score) for ``question`` ("Who starts at FLEX in week 5?"). Building one that is not a near-tie raises
    ``ValueError``, which is how the worker stays a tie-breaker."""

    sport: Sport
    league_label: str
    question: str
    options: tuple[CloseCallOption, ...]
    margin: float
    deadline: datetime | None = None
    league_id: int | None = None

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("a close call needs at least two options")
        if len({option.espn_id for option in self.options}) != len(self.options):
            raise ValueError("a close call's options must be distinct players")
        if self.margin < 0 or not math.isfinite(self.margin):
            raise ValueError(f"margin must be finite and non-negative, got {self.margin!r}")
        if not self.question.strip():
            raise ValueError("a close call needs a question")
        gap = self.best.score - min(option.score for option in self.options)
        if gap > self.margin:
            raise ValueError(f"not a close call: the options are {gap:.2f} apart, margin {self.margin:.2f}")
        if self.deadline is not None and (self.deadline.tzinfo is None or self.deadline.utcoffset() is None):
            raise ValueError("deadline must be an aware datetime")

    @property
    def best(self) -> CloseCallOption:
        """The engine's own choice: the highest score, the first listed on a tie."""
        return max(self.options, key=lambda option: option.score)

    @property
    def option_ids(self) -> tuple[int, ...]:
        return tuple(option.espn_id for option in self.options)

    def text(self, now: datetime) -> str:
        """The user block."""
        lines = [
            f"Now: {now:%Y-%m-%dT%H:%M:%SZ}",
            f"League: {self.league_label} ({self.sport})",
            f"Decision: {self.question.strip()}",
        ]
        if self.deadline is not None:
            lines.append(f"Deadline: {self.deadline:%Y-%m-%dT%H:%M:%SZ}")
        lines.append(f"The engine scores these within {self.margin:.2f} of each other. Options:")
        lines.extend(f"- {option.describe()}" for option in self.options)
        return "\n".join(lines)


# --- the answer -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Source:
    """A cited page behind a finding, kept after the allowlist check."""

    url: str
    title: str
    claim: str
    espn_id: int
    published_at: str | None = None


@dataclass(frozen=True, slots=True)
class CloseCallResult:
    """What became of a close call. ``choice`` is what the caller should use: the pick when ``status`` is ``pick``,
    the engine's own choice otherwise, and always one of the call's options. ``reply`` is the client's reply (``None``
    when no request was sent) and has the ``llm_usage`` row."""

    call: CloseCall
    status: CloseCallStatus
    pick: int | None = None
    confidence: float | None = None
    rationale: str | None = None
    sources: tuple[Source, ...] = ()
    detail: str | None = None
    reply: WorkerReply[CloseCallOutput] | None = field(default=None, compare=False)

    @property
    def engine_pick(self) -> int:
        return self.call.best.espn_id

    @property
    def choice(self) -> int:
        return self.pick if self.status == "pick" and self.pick is not None else self.engine_pick

    @property
    def overrides_engine(self) -> bool:
        return self.choice != self.engine_pick

    def describe(self) -> str:
        """One line: ``close_call: pick 123 over the engine's 456 (0.72), 2 sources`` and the like."""
        match self.status:
            case "pick":
                what = (
                    f"pick {self.pick} over the engine's {self.engine_pick}"
                    if self.overrides_engine
                    else f"pick {self.pick}, as the engine"
                )
                what += f" ({self.confidence or 0.0:.2f}), {len(self.sources)} sources"
            case other:
                what = f"{other}, engine's {self.engine_pick} stands"
        return f"{CLOSE_CALL_WORKER}: {what}" + (f": {self.detail}" if self.detail else "")


def judge_close_call(call: CloseCall, output: CloseCallOutput, allowed_domains: Sequence[str]) -> CloseCallResult:
    """The rules of the module docs applied to an answer, as a result without its ``reply``: ``pick``, ``no_change``
    or ``rejected``. Each rejection and clamp is logged."""
    label = f"close call {call.league_label!r}"
    confidence = output.confidence
    if not math.isfinite(confidence):
        return CloseCallResult(call, "rejected", detail="non-finite confidence")
    clamped = min(1.0, max(0.0, confidence))
    if clamped != confidence:
        logger.warning("%s: confidence %.2f outside [0, 1]; clamped", label, confidence)
    rationale = output.rationale.strip() or None

    if output.pick_espn_id is None:
        return CloseCallResult(call, "no_change", confidence=clamped, rationale=rationale, detail="no decisive source")
    if output.pick_espn_id not in call.option_ids:
        options = ", ".join(map(str, call.option_ids))
        detail = f"picked {output.pick_espn_id}, not one of the near-tied options ({options})"
        logger.warning("%s: %s; rejected", label, detail)
        return CloseCallResult(call, "rejected", confidence=clamped, rationale=rationale, detail=detail)

    sources: dict[tuple[str, int], Source] = {}
    for finding in output.findings:
        url = finding.source_url.strip()
        reason = None
        if finding.espn_id not in call.option_ids:
            reason = f"about {finding.espn_id}, not an option"
        elif not finding.claim.strip():
            reason = "no claim"
        elif not domain_allowed(url, allowed_domains):
            reason = f"{url!r} is not an http(s) page on an allowlisted domain"
        if reason is not None:
            logger.warning("%s: finding dropped: %s", label, reason)
            continue
        sources.setdefault(
            (url, finding.espn_id),
            Source(
                url=url,
                title=finding.source_title.strip(),
                claim=finding.claim.strip(),
                espn_id=finding.espn_id,
                published_at=(finding.published_at or "").strip() or None,
            ),
        )
    cited = tuple(sources.values())
    if not cited:
        detail = "the pick has no valid cited finding; ignored"
        logger.warning("%s: %s", label, detail)
        return CloseCallResult(call, "rejected", confidence=clamped, rationale=rationale, detail=detail)
    if rationale is None:
        return CloseCallResult(
            call, "rejected", confidence=clamped, sources=cited, detail="the pick has no rationale; ignored"
        )
    if clamped < CLOSE_CALL_MIN_CONFIDENCE:
        detail = f"confidence {clamped:.2f} is below {CLOSE_CALL_MIN_CONFIDENCE:.2f}"
        return CloseCallResult(call, "no_change", confidence=clamped, rationale=rationale, sources=cited, detail=detail)
    return CloseCallResult(
        call, "pick", pick=output.pick_espn_id, confidence=clamped, rationale=rationale, sources=cited
    )


def close_call_prompt(
    call: CloseCall,
    *,
    allowed_domains: Sequence[str],
    now: datetime,
    max_searches: int = CLOSE_CALL_MAX_SEARCHES,
    max_tokens: int = CLOSE_CALL_MAX_TOKENS,
) -> SearchPrompt:
    """The call's prompt: the worker's instructions and the league as cached system blocks, the question as the
    user block, and the search tool over ``allowed_domains``."""
    return SearchPrompt(
        system=prompt_text(CLOSE_CALL_WORKER),
        context=f"League: {call.league_label} ({call.sport}). Sources are limited to: "
        + ", ".join(normalize_domains(allowed_domains))
        + ".",
        user=call.text(now),
        max_tokens=max_tokens,
        effort=effort_for(CLOSE_CALL_WORKER),
        league_id=call.league_id,
        allowed_domains=tuple(allowed_domains),
        max_uses=max_searches,
    )


def decide_close_call(
    client: AdvisorClient,
    call: CloseCall,
    *,
    allowed_domains: Sequence[str] = DEFAULT_ALLOWED_DOMAINS,
    max_searches: int = CLOSE_CALL_MAX_SEARCHES,
    now: datetime | None = None,
) -> CloseCallResult:
    """Ask Claude to break the tie in ``call`` and apply the module's rules to the answer.

    Never raises for a spent budget or an unreachable API (``blocked`` and ``failed`` leave the engine's choice);
    raises :class:`~fm.advisor.client.AdvisorError` when ``client`` cannot send the search tool (see
    :func:`close_call_client`) and ``ValueError`` for an empty allowlist."""
    transport = getattr(client, "_transport", None)
    if isinstance(transport, SdkTransport) and not isinstance(transport, SearchSdkTransport):
        raise AdvisorError("this client's transport would drop the web search tool; build it with close_call_client")
    when = now if now is not None else utc_now()
    domains = normalize_domains(allowed_domains)
    prompt = close_call_prompt(call, allowed_domains=domains, now=when, max_searches=max_searches)
    try:
        reply = client.ask(CLOSE_CALL_WORKER, prompt, CloseCallOutput, now=when)
    except BudgetExceededError as exc:
        result = CloseCallResult(call, "blocked", detail=str(exc))
        logger.warning("advisor: %s", result.describe())
        return result
    except AdvisorError as exc:
        result = CloseCallResult(call, "failed", detail=str(exc))
        logger.warning("advisor: %s", result.describe())
        return result
    if reply.output is None:
        result = CloseCallResult(call, "failed", detail=reply.detail or reply.status, reply=reply)
        logger.warning("advisor: %s", result.describe())
        return result
    result = replace(judge_close_call(call, reply.output, domains), reply=reply)
    logger.info(
        "advisor: %s; sources: %s",
        result.describe(),
        json.dumps([{"url": s.url, "published_at": s.published_at, "espn_id": s.espn_id} for s in result.sources]),
    )
    return result


def result_numbers(result: CloseCallResult) -> Mapping[str, Any]:
    """The result as JSON for a proposal's ``engine_numbers``: what was asked, what Claude answered, and its sources."""
    return {
        "close_call": {
            "status": result.status,
            "engine_pick": result.engine_pick,
            "pick": result.pick,
            "choice": result.choice,
            "confidence": result.confidence,
            "rationale": result.rationale,
            "margin": result.call.margin,
            "options": {option.espn_id: option.score for option in result.call.options},
            "sources": [
                {
                    "url": s.url,
                    "title": s.title,
                    "claim": s.claim,
                    "espn_id": s.espn_id,
                    "published_at": s.published_at,
                }
                for s in result.sources
            ],
            "detail": result.detail,
        }
    }
