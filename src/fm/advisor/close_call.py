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
  option is always one of them. The margin itself is bounded: at most :data:`CLOSE_CALL_MAX_MARGIN_FRACTION` of the
  best score and never above :data:`CLOSE_CALL_MAX_MARGIN` (:func:`max_margin`), so a caller cannot widen "near" until
  anything qualifies; a larger margin raises ``ValueError``.
- *Only our own players are options.* A :class:`CloseCall` takes the ids of the current roster and refuses an option
  that is not on it, so a free agent can never be among the choices and a "pick" can never turn into an add.
- *Only the options can be chosen.* The pick must be the ESPN id of one of the options; anything else is
  ``rejected``. There is no way to name a player outside the call, so there is no way to ask for an add, a drop or a
  trade, and :attr:`CloseCallResult.choice` is an option whatever happened.
- *Cited or ignored.* A pick stands only with at least one finding that names an option, has a claim, and cites an
  ``http(s)`` page that is on an allowlisted domain (the list the search tool is restricted to) AND was actually
  returned by the search: the URL, normalized by :func:`normalize_url` (scheme and host case, trailing slash,
  fragment), must be among the ``web_search_result`` URLs of the reply (``WorkerReply.search_urls``). A model can
  write a plausible URL it never retrieved; such a finding is logged and dropped. A pick with no valid finding is
  ``rejected`` and the engine's choice stands. The valid sources come back as :class:`Source` rows for the
  proposal's rationale.
- *Confident or ignored.* A pick below :data:`CLOSE_CALL_MIN_CONFIDENCE` is ``no_change``, as is a null pick.
- *Logged.* Every outcome is logged with its sources, and the call is recorded in ``llm_usage`` by the client.

The statuses of :class:`fm.advisor.client.WorkerReply` map to ``failed`` (refusal, ``max_tokens`` and the other
non-``ok`` statuses, with the reason: no partial data is read; a ``pause_turn`` from the search loop is resumed by
the client up to its cap and is a failure if it is still paused then), and
:class:`fm.advisor.client.BudgetExceededError` to ``blocked``. In both the engine's choice stands, which is the
fallback.

**The search tool.** The client's :class:`fm.advisor.client.Prompt` carries ``tools``; :func:`close_call_prompt` puts
the web search tool there (:class:`WebSearchTool`: the allowlist and the ``max_uses`` cap), so any
:class:`~fm.advisor.client.AdvisorClient` can run this worker. The tool entry is the same on every call with the same
allowlist and cap, which keeps the cached prefix warm. The allowlist is a parameter defaulting to
:data:`DEFAULT_ALLOWED_DOMAINS` because ``fm.config.Llm`` has no field for it yet.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Final, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from fm.advisor.client import (
    AdvisorClient,
    AdvisorError,
    BudgetExceededError,
    Prompt,
    WorkerReply,
    effort_for,
)
from fm.advisor.prompts import prompt_text
from fm.config import Sport
from fm.decide.lineup import LineupCandidate
from fm.store import utc_now

logger = logging.getLogger(__name__)

CLOSE_CALL_WORKER: Final = "close_call"
CLOSE_CALL_MAX_TOKENS: Final = 4096
"""The answer is a pick, a short rationale and a few findings; the search results count as input, not output."""
CLOSE_CALL_MIN_CONFIDENCE: Final = 0.6
"""A pick below this confidence is ignored (``no_change``): 0.5 is a coin flip and the engine is the default."""
CLOSE_CALL_MAX_MARGIN_FRACTION: Final = 0.2
"""The widest a margin may be, as a fraction of the best option's score: two options 20% apart are a preference the
engine already has, not a near-tie. A best score at or below 0 allows only an exact tie."""
CLOSE_CALL_MAX_MARGIN: Final = 3.0
"""The widest a margin may be in the engine's score units (usually league points), whatever the best score: no
fraction of a large score turns a clear gap into a tie."""
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
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
    except ValueError:  # e.g. an unclosed IPv6 bracket
        return False
    if parts.scheme not in ("http", "https") or not host:
        return False
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def normalize_url(url: str) -> str:
    """The URL as compared against the search results: scheme and host lower-cased, the fragment dropped and a
    trailing slash on the path removed (``https://ESPN.com/a/#x`` equals ``https://espn.com/a``). The path and the
    query keep their case and order, which are significant. A string that is not a URL comes back stripped."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def web_search_tool(domains: Sequence[str], max_uses: int = CLOSE_CALL_MAX_SEARCHES) -> dict[str, Any]:
    """The ``tools`` entry of a call: the web search tool over ``domains`` with a ``max_uses`` cap."""
    if max_uses < 1:
        raise ValueError(f"max_uses must be positive, got {max_uses!r}")
    tool = WebSearchTool(max_uses=max_uses, allowed_domains=normalize_domains(domains))
    return tool.model_dump(mode="json")


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


def max_margin(best_score: float) -> float:
    """The widest margin a :class:`CloseCall` may have when the best option scores ``best_score``: the smaller of
    :data:`CLOSE_CALL_MAX_MARGIN` and :data:`CLOSE_CALL_MAX_MARGIN_FRACTION` of the score (0 for a score at or below
    0)."""
    return min(CLOSE_CALL_MAX_MARGIN, CLOSE_CALL_MAX_MARGIN_FRACTION * max(best_score, 0.0))


def near_ties(options: Sequence[CloseCallOption], margin: float) -> tuple[CloseCallOption, ...]:
    """The options within ``margin`` of the best score, best first (ties keep their order); empty unless at least
    two qualify, since one option is no call. A margin above :func:`max_margin` of the best score raises
    ``ValueError``, as it would for a :class:`CloseCall`."""
    if margin < 0 or not math.isfinite(margin):
        raise ValueError(f"margin must be finite and non-negative, got {margin!r}")
    if not options:
        return ()
    ranked = sorted(options, key=lambda option: -option.score)
    ceiling = max_margin(ranked[0].score)
    if margin > ceiling:
        raise ValueError(f"margin {margin:.2f} is wider than a near-tie may be (at most {ceiling:.2f})")
    tied = tuple(option for option in ranked if ranked[0].score - option.score <= margin)
    return tied if len(tied) >= 2 else ()


@dataclass(frozen=True, slots=True)
class CloseCall:
    """One start/sit decision the engine cannot separate: ``options`` (two or more, all within ``margin`` of the best
    score) for ``question`` ("Who starts at FLEX in week 5?"). ``roster_ids`` are the ESPN ids of the players on our
    current roster; every option must be one of them. Building one that is not a near-tie, whose margin exceeds
    :func:`max_margin`, or that has an option off the roster raises ``ValueError``, which is how the worker stays a
    tie-breaker among our own players."""

    sport: Sport
    league_label: str
    question: str
    options: tuple[CloseCallOption, ...]
    margin: float
    roster_ids: Collection[int]
    deadline: datetime | None = None
    league_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "roster_ids", frozenset(self.roster_ids))
        if len(self.options) < 2:
            raise ValueError("a close call needs at least two options")
        if len({option.espn_id for option in self.options}) != len(self.options):
            raise ValueError("a close call's options must be distinct players")
        off_roster = [option.espn_id for option in self.options if option.espn_id not in self.roster_ids]
        if off_roster:
            raise ValueError(
                f"options not on our roster: {', '.join(map(str, off_roster))}; only our players can be asked"
            )
        if self.margin < 0 or not math.isfinite(self.margin):
            raise ValueError(f"margin must be finite and non-negative, got {self.margin!r}")
        ceiling = max_margin(self.best.score)
        if self.margin > ceiling:
            raise ValueError(
                f"margin {self.margin:.2f} is wider than a near-tie may be for a best score of "
                f"{self.best.score:.2f} (at most {ceiling:.2f})"
            )
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


def judge_close_call(
    call: CloseCall, output: CloseCallOutput, allowed_domains: Sequence[str], retrieved_urls: Iterable[str]
) -> CloseCallResult:
    """The rules of the module docs applied to an answer, as a result without its ``reply``: ``pick``, ``no_change``
    or ``rejected``. ``retrieved_urls`` are the ``web_search_result`` URLs the search returned: a finding citing any
    other page is dropped, however plausible. Each rejection and clamp is logged."""
    retrieved = {normalize_url(url) for url in retrieved_urls}
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
        elif normalize_url(url) not in retrieved:
            reason = f"{url!r} was not among the pages the search returned"
        if reason is not None:
            logger.warning("%s: finding dropped: %s", label, reason)
            continue
        sources.setdefault(
            (normalize_url(url), finding.espn_id),
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
) -> Prompt:
    """The call's prompt: the worker's instructions and the league as cached system blocks, the question as the
    user block, and the search tool over ``allowed_domains``."""
    domains = normalize_domains(allowed_domains)
    return Prompt(
        system=prompt_text(CLOSE_CALL_WORKER),
        context=f"League: {call.league_label} ({call.sport}). Sources are limited to: " + ", ".join(domains) + ".",
        user=call.text(now),
        max_tokens=max_tokens,
        effort=effort_for(CLOSE_CALL_WORKER),
        league_id=call.league_id,
        tools=(web_search_tool(domains, max_searches),),
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
    raises ``ValueError`` for an empty allowlist."""
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
    result = replace(judge_close_call(call, reply.output, domains, reply.search_urls), reply=reply)
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
