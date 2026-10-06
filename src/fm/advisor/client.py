"""The advisor's Claude client: every Messages API call the workers make goes through here (DESIGN section 10).

A worker is one structured call: a :class:`Prompt` (its instructions, optional league context, the question, the
effort the worker runs at and a token limit) and a Pydantic output type. :meth:`AdvisorClient.ask` sends it through
``client.messages.parse`` with the output type as the structured-output schema and the effort in ``output_config``,
and returns a :class:`WorkerReply` whose ``status`` says whether the output may be read:

- ``ok``: ``stop_reason`` was ``end_turn`` and the output parsed into the type;
- ``refusal``: the whole chain declined (``stop_reason`` ``refusal``, the API's own refusal classifier included); the
  reply carries the explanation when the API gave one and no output. The worker falls back to the engine without
  Claude;
- ``max_tokens``: the answer was cut off. Treated as a failure, never as partial data: the schema makes a truncated
  answer invalid anyway, and the SDK refuses to hand it over;
- ``unparseable``: the answer came back whole but did not match the output type, or the SDK could not parse it;
- ``stopped``: any other stop reason (``tool_use``, ``pause_turn``, ``stop_sequence``, ...), which no worker here
  asks for.

The API's ``stop_reason`` is checked before the parsed output is read (CLAUDE.md), whatever the SDK attached to it.

**Refusal fallback.** By default (``AdvisorClient(refusal_fallback=True)``) a live call opts into the server-side
refusal fallback for ``claude-opus-5-5``: it goes through ``client.beta.messages.parse`` with
``betas=[FALLBACK_BETA]`` and ``fallbacks="default"``, so a refusal by the requested model is retried by the API on
the fallback model for that refusal category instead of coming back empty. The served model can then differ from
the requested one: a ``fallback_message`` entry in ``usage.iterations`` says so, and the call is recorded and priced
under the model that answered (``response.model``), logged. A final ``stop_reason`` of ``refusal`` still means the
whole chain refused and takes the refusal path above. The Batches API rejects ``fallbacks``, so batch requests never
carry it; a batch refusal is logged and skipped.

**Cost.** Every call that reached the API is recorded in ``llm_usage`` (:class:`fm.store.LlmUsageRow`) with its
tokens, the cost at the model's :class:`Pricing` and whether it ran in a batch (half price). A call is refused with
:class:`BudgetExceededError`, before any request is sent, once the calls recorded since 00:00 UTC have spent
``[llm] daily_budget_usd``: the cap is on the spend so far, so one call may finish the day above it, by at most its
own cost (a batch by the batch's). A cap of 0 blocks every call. ``spent_today`` and ``budget_left`` read the same
numbers.

**Caching.** The prompt's instructions and league context are sent as system blocks with ``cache_control``
(``ephemeral``, the 5-minute default), so a run that asks the same worker several questions about one league pays
for that prefix once; a prefix below the model's minimum cacheable size is simply sent uncached.

**Batches.** Overnight work goes through the Message Batches API: :meth:`AdvisorClient.submit_batch` sends several
prompts of one worker as one batch, keyed by the caller's ids, and :meth:`AdvisorClient.collect_batch` reads the
results once the batch has ended (``None`` while it is still processing), one :class:`WorkerReply` per request with the
same status rules; each succeeded request is recorded in ``llm_usage`` as it is collected, at the batch price.

The SDK is reached only through a :class:`Transport`: :class:`SdkTransport` wraps ``anthropic.Anthropic`` in
production, and tests pass their own, so no unit test holds a key or opens a socket (``tests/conftest.py``). The
transport raises :class:`AdvisorError` for a transport or API failure; nothing is recorded for a call that never
got an answer. The advisor has no write path to ESPN: this module imports nothing from ``fm.browser`` or
``fm.executor``, and the SDK's tools are never enabled.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Final, Literal, TypedDict, cast

import anthropic
from anthropic import transform_schema
from anthropic.types import JSONOutputFormatParam, MessageParam, OutputConfigParam, ParsedMessage, TextBlockParam, Usage
from anthropic.types.beta import BetaUsage
from anthropic.types.beta.parsed_beta_message import ParsedBetaMessage
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatchIndividualResponse
from anthropic.types.messages.batch_create_params import Request as BatchRequestParam
from pydantic import BaseModel, SecretStr, TypeAdapter, ValidationError

from fm.config import DEFAULT_MODEL, Config, Llm
from fm.store import LlmUsageRow, Store, utc_now

logger = logging.getLogger(__name__)

type Effort = Literal["low", "medium", "high", "xhigh", "max"]
"""The ``output_config.effort`` levels the API accepts."""

type ReplyStatus = Literal["ok", "refusal", "max_tokens", "unparseable", "stopped"]
"""What became of a call: only ``ok`` has an output (see the module docs)."""

type BatchState = Literal["in_progress", "canceling", "ended"]
"""A batch's ``processing_status``."""

EFFORT_BY_WORKER: Mapping[str, Effort] = MappingProxyType(
    {
        "news_triage": "low",
        "close_call": "medium",
        "explain": "low",
        "trade_pitch": "medium",
        "weekly_strategist": "high",
    }
)
"""Each worker's effort (DESIGN section 10, the advisor table)."""

MTOK: Final = 1_000_000
"""Prices are per million tokens."""

MAX_BATCH_REQUESTS: Final = 10_000
"""The Message Batches API's limit on requests per batch."""

FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
"""The beta header the ``fallbacks="default"`` scalar form requires (the ``-2026-06-01`` header is the older array
form; the two are never mixed)."""


class AdvisorError(RuntimeError):
    """A call could not be made or answered: no API key, a transport failure, an API error."""


class BudgetExceededError(AdvisorError):
    """The daily spend cap is reached; the call was refused before any request was sent."""

    def __init__(self, worker: str, spent: float, budget: float) -> None:
        super().__init__(
            f"{worker}: daily Claude budget reached (${spent:.2f} of ${budget:.2f} spent since 00:00 UTC); call refused"
        )
        self.worker = worker
        self.spent = spent
        self.budget = budget


def effort_for(worker: str) -> Effort:
    """The worker's effort from :data:`EFFORT_BY_WORKER`; ``KeyError`` names the known workers."""
    try:
        return EFFORT_BY_WORKER[worker]
    except KeyError:
        raise KeyError(f"no effort set for worker {worker!r}; known: {', '.join(EFFORT_BY_WORKER)}") from None


# --- pricing ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """The token counts of one answer, as the API's ``usage`` reports them (cache fields 0 when absent)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @classmethod
    def from_sdk(cls, usage: Usage | BetaUsage) -> TokenUsage:
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        )


@dataclass(frozen=True, slots=True)
class Pricing:
    """A model's list price in USD per million tokens. Cache writes cost ``cache_write_multiplier`` times the input
    price and cache reads ``cache_read_multiplier`` times it; a batch pays ``batch_multiplier`` of everything."""

    input_per_mtok: float
    output_per_mtok: float
    cache_write_multiplier: float = 1.25
    cache_read_multiplier: float = 0.1
    batch_multiplier: float = 0.5

    def cost(self, usage: TokenUsage, *, batch: bool = False) -> float:
        """The cost of one answer in USD."""
        total = (
            usage.input_tokens * self.input_per_mtok
            + usage.output_tokens * self.output_per_mtok
            + usage.cache_creation_input_tokens * self.input_per_mtok * self.cache_write_multiplier
            + usage.cache_read_input_tokens * self.input_per_mtok * self.cache_read_multiplier
        ) / MTOK
        return total * self.batch_multiplier if batch else total


PRICING: Mapping[str, Pricing] = MappingProxyType(
    {DEFAULT_MODEL: Pricing(input_per_mtok=4.0, output_per_mtok=20.0, cache_read_multiplier=0.05)}
)
"""List prices by model id, from the Claude API reference as cached on 2026-09-25: ``claude-opus-5-5`` is $4.00 in,
$20.00 out, $0.20 per cache read (0.05x) and 1.25x per 5-minute cache write, per million tokens; Batch is half.
Update this table when the price sheet changes: spend is recorded at these numbers, so the daily cap is only as
right as they are."""


def pricing_for(model: str) -> Pricing:
    """The model's pricing; an unlisted model is priced like :data:`fm.config.DEFAULT_MODEL`, with a warning."""
    pricing = PRICING.get(model)
    if pricing is None:
        logger.warning("advisor: no pricing listed for model %r; costing it like %s", model, DEFAULT_MODEL)
        return PRICING[DEFAULT_MODEL]
    return pricing


# --- prompts and replies ----------------------------------------------------------------------------------------------


class CallParams(TypedDict):
    """The Messages API parameters of one call, before the output format: what :meth:`Prompt.params` builds and the
    :class:`Transport` sends."""

    model: str
    max_tokens: int
    system: list[TextBlockParam]
    messages: list[MessageParam]
    output_config: OutputConfigParam


@dataclass(frozen=True, slots=True)
class Prompt:
    """One worker call. ``system`` is the worker's instructions and ``context`` the league context, both sent as
    cached system blocks (stable across the worker's calls, in that order); ``user`` is the question. ``effort``
    is the worker's (:func:`effort_for`) and ``max_tokens`` bounds the answer; ``league_id`` is recorded with the
    usage."""

    system: str
    user: str
    max_tokens: int
    effort: Effort
    context: str | None = None
    league_id: int | None = None

    def __post_init__(self) -> None:
        if not self.system.strip() or not self.user.strip():
            raise ValueError("a prompt needs non-empty system and user text")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens!r}")

    def params(self, model: str) -> CallParams:
        """The Messages API parameters, without the output format (the transport adds that)."""
        system: list[TextBlockParam] = [
            {"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}},
        ]
        if self.context is not None and self.context.strip():
            system.append({"type": "text", "text": self.context, "cache_control": {"type": "ephemeral"}})
        return {
            "model": model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": self.user}]}],
            "output_config": {"effort": self.effort},
        }


@dataclass(frozen=True, slots=True)
class RawReply:
    """What one request came back with, independent of the SDK's types: the :class:`Transport`'s output.

    ``model`` is the model that answered (a fallback model when ``fallback`` is true); ``output`` is the parsed
    structured output when the SDK could parse one (it is read only when ``stop_reason`` is ``end_turn``);
    ``detail`` is the refusal's explanation or the parse error. A transport that could not parse the answer returns
    ``output=None`` with the ``detail``, and ``stop_reason=None`` with zero usage when the SDK raised before handing
    the message over (it validates the output before returning).
    """

    request_id: str | None
    model: str
    stop_reason: str | None
    usage: TokenUsage
    output: object = None
    detail: str | None = None
    fallback: bool = False


@dataclass(frozen=True, slots=True)
class WorkerReply[T]:
    """A worker's answer. ``output`` is set only when ``status`` is ``ok``; ``usage`` is the ``llm_usage`` row
    recorded for the call (``id`` set when it was stored); ``detail`` explains a refusal or a parse failure."""

    worker: str
    status: ReplyStatus
    output: T | None
    stop_reason: str | None
    usage: LlmUsageRow
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def describe(self) -> str:
        """One line: ``news_triage: ok (end_turn), 1,234 in / 210 out, $0.0124``."""
        usage = self.usage
        tokens = f"{usage.input_tokens:,} in / {usage.output_tokens:,} out"
        if usage.cache_read_input_tokens:
            tokens += f", {usage.cache_read_input_tokens:,} cached"
        what = f"{self.worker}: {self.status} ({self.stop_reason or 'no answer'}), {tokens}, ${usage.cost_usd:.4f}"
        return what if self.detail is None else f"{what}: {self.detail}"


# --- the transport ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """One request of a batch: the caller's id and the API parameters (output format included)."""

    custom_id: str
    params: MessageCreateParamsNonStreaming


@dataclass(frozen=True, slots=True)
class BatchResult:
    """One line of a batch's results: the request's id and its answer, or why there is none (``errored``,
    ``canceled``, ``expired``)."""

    custom_id: str
    reply: RawReply | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BatchStatus:
    """A batch's state and request counts."""

    batch_id: str
    state: BatchState
    succeeded: int = 0
    errored: int = 0
    canceled: int = 0
    expired: int = 0
    processing: int = 0

    @property
    def ended(self) -> bool:
        return self.state == "ended"


class Transport:
    """The SDK surface the client needs. :class:`SdkTransport` implements it over ``anthropic.Anthropic``; a test
    passes a fake. Every method raises :class:`AdvisorError` when the API cannot be reached or answers an error."""

    def parse(self, params: CallParams, output: type[BaseModel], *, fallback: bool = False) -> RawReply:
        """One call; with ``fallback`` the server-side refusal fallback is requested (see the module docs)."""
        raise NotImplementedError

    def create_batch(self, requests: Sequence[BatchRequest]) -> str:
        """Submit the requests; returns the batch id."""
        raise NotImplementedError

    def batch_status(self, batch_id: str) -> BatchStatus:
        raise NotImplementedError

    def batch_results(self, batch_id: str, output: type[BaseModel]) -> list[BatchResult]:
        """The results of an ended batch, parsed into ``output`` where possible."""
        raise NotImplementedError


def output_format(output: type[BaseModel]) -> JSONOutputFormatParam:
    """The ``output_config.format`` for a Pydantic type, as ``messages.parse`` builds it."""
    return {"type": "json_schema", "schema": transform_schema(TypeAdapter(output).json_schema())}


def _first_text(content: Iterable[Any]) -> str | None:
    for block in content:
        if getattr(block, "type", None) == "text":
            return str(block.text)
    return None


def _refusal_detail(message: Any) -> str | None:
    details = getattr(message, "stop_details", None)
    if details is None:
        return None
    parts = [part for part in (getattr(details, "category", None), getattr(details, "explanation", None)) if part]
    return ": ".join(str(part) for part in parts) if parts else None


def served_by_fallback(message: Any) -> bool:
    """True when a fallback model served the answer: a ``fallback_message`` entry in ``usage.iterations`` (the
    API's signal; ``fallback`` content blocks only mark the hops that declined)."""
    iterations = getattr(message.usage, "iterations", None) or ()
    return any(getattr(entry, "type", None) == "fallback_message" for entry in iterations)


def reply_from_message(message: Any, output: type[BaseModel]) -> RawReply:
    """A :class:`RawReply` from an SDK message, plain or parsed, beta or not. A plain message's first text block is
    parsed into ``output`` here (batch results come back unparsed); a parse failure becomes the ``detail``. ``model``
    is the model that answered, which a refusal fallback can change."""
    parsed: object = getattr(message, "parsed_output", None)
    detail = _refusal_detail(message)
    already_parsed = isinstance(message, ParsedMessage | ParsedBetaMessage)
    if parsed is None and message.stop_reason == "end_turn" and not already_parsed:
        text = _first_text(message.content)
        if text is not None:
            try:
                parsed = TypeAdapter(output).validate_json(text)
            except ValidationError as exc:
                detail = f"output does not match {output.__name__} ({exc.error_count()} errors)"
    return RawReply(
        request_id=message.id,
        model=str(message.model),
        stop_reason=message.stop_reason,
        usage=TokenUsage.from_sdk(message.usage),
        output=parsed,
        detail=detail,
        fallback=served_by_fallback(message),
    )


class SdkTransport(Transport):
    """The real thing: ``anthropic.Anthropic`` (the SDK retries 429s and 5xx itself)."""

    def __init__(self, sdk: anthropic.Anthropic) -> None:
        self._sdk = sdk

    @classmethod
    def with_key(cls, api_key: SecretStr | str, *, timeout: float = 120.0, max_retries: int = 2) -> SdkTransport:
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not key.strip():
            raise AdvisorError("ANTHROPIC_API_KEY is blank")
        return cls(anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=max_retries))

    def parse(self, params: CallParams, output: type[BaseModel], *, fallback: bool = False) -> RawReply:
        try:
            message: Any
            if fallback:
                # The beta messages surface: the same parameters (the beta param types are structurally the same
                # TypedDicts) plus the fallback header and the "default" fallback chain, which only this form takes.
                message = self._sdk.beta.messages.parse(
                    model=params["model"],
                    max_tokens=params["max_tokens"],
                    system=cast(Any, params["system"]),
                    messages=cast(Any, params["messages"]),
                    output_config=cast(Any, params["output_config"]),
                    output_format=output,
                    betas=[FALLBACK_BETA],
                    fallbacks="default",
                )
            else:
                message = self._sdk.messages.parse(
                    model=params["model"],
                    max_tokens=params["max_tokens"],
                    system=params["system"],
                    messages=params["messages"],
                    output_config=params["output_config"],
                    output_format=output,
                )
        except ValidationError as exc:
            # The SDK validates the text against the schema before returning the message, so a truncated or
            # off-schema answer surfaces here, without the message (and its usage) it came in.
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

    def create_batch(self, requests: Sequence[BatchRequest]) -> str:
        body: list[BatchRequestParam] = [{"custom_id": r.custom_id, "params": r.params} for r in requests]
        try:
            return self._sdk.messages.batches.create(requests=body).id
        except anthropic.APIError as exc:
            raise AdvisorError(f"Claude batch submission failed: {exc}") from exc

    def batch_status(self, batch_id: str) -> BatchStatus:
        try:
            batch = self._sdk.messages.batches.retrieve(batch_id)
        except anthropic.APIError as exc:
            raise AdvisorError(f"Claude batch {batch_id} could not be read: {exc}") from exc
        counts = batch.request_counts
        return BatchStatus(
            batch_id=batch.id,
            state=batch.processing_status,
            succeeded=counts.succeeded,
            errored=counts.errored,
            canceled=counts.canceled,
            expired=counts.expired,
            processing=counts.processing,
        )

    def batch_results(self, batch_id: str, output: type[BaseModel]) -> list[BatchResult]:
        try:
            lines: Iterable[MessageBatchIndividualResponse] = self._sdk.messages.batches.results(batch_id)
            return [_batch_result(line, output) for line in lines]
        except anthropic.APIError as exc:
            raise AdvisorError(f"Claude batch {batch_id} results could not be read: {exc}") from exc


def _batch_result(line: MessageBatchIndividualResponse, output: type[BaseModel]) -> BatchResult:
    result = line.result
    if result.type == "succeeded":
        return BatchResult(line.custom_id, reply_from_message(result.message, output))
    if result.type == "errored":
        return BatchResult(line.custom_id, None, f"errored: {result.error.error.message}")
    return BatchResult(line.custom_id, None, result.type)


# --- the client -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubmittedBatch:
    """A batch :meth:`AdvisorClient.submit_batch` sent: what to pass to :meth:`AdvisorClient.collect_batch`."""

    batch_id: str
    worker: str
    custom_ids: tuple[str, ...]
    submitted_at: datetime
    league_id: int | None = None


@dataclass(frozen=True, slots=True)
class CollectedBatch:
    """An ended batch's replies by the caller's id, plus the requests that got no answer and why."""

    batch_id: str
    status: BatchStatus
    replies: Mapping[str, WorkerReply[Any]]
    failed: Mapping[str, str]

    def describe(self) -> str:
        ok = sum(1 for reply in self.replies.values() if reply.ok)
        return (
            f"batch {self.batch_id}: {ok} ok, {len(self.replies) - ok} not usable, {len(self.failed)} unanswered, "
            f"${math.fsum(reply.usage.cost_usd for reply in self.replies.values()):.4f}"
        )


def day_start(now: datetime) -> datetime:
    """00:00 UTC of ``now``'s day: the daily budget's window opens here."""
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


class AdvisorClient:
    """The workers' way to Claude (see the module docs). ``llm`` gives the model and the daily cap; ``transport``
    is the SDK or a fake; ``pricing`` defaults to the model's listed price; ``refusal_fallback`` (on by default)
    asks the API for the server-side refusal fallback on live calls."""

    def __init__(
        self,
        store: Store,
        llm: Llm,
        transport: Transport,
        *,
        pricing: Pricing | None = None,
        refusal_fallback: bool = True,
    ) -> None:
        self.store = store
        self.model: str = llm.model
        self.daily_budget_usd: float = llm.daily_budget_usd
        self.pricing: Pricing = pricing if pricing is not None else pricing_for(llm.model)
        self.refusal_fallback: bool = refusal_fallback
        self._transport = transport

    @classmethod
    def from_config(cls, store: Store, config: Config, *, transport: Transport | None = None) -> AdvisorClient:
        """A client for the configured model and cap over the real SDK with the ``.env`` key, unless ``transport``
        is given. Raises :class:`AdvisorError` when no key is configured."""
        if transport is None:
            key = config.secrets.anthropic_api_key
            if key is None:
                raise AdvisorError("ANTHROPIC_API_KEY is not set in .env; the advisor cannot run (fm config check)")
            transport = SdkTransport.with_key(key)
        return cls(store, config.llm, transport)

    # --- budget ---

    def spent_today(self, now: datetime | None = None) -> float:
        """USD recorded in ``llm_usage`` since 00:00 UTC today."""
        return self.store.llm_usage.cost_since(day_start(now if now is not None else utc_now()))

    def budget_left(self, now: datetime | None = None) -> float:
        """What the day's cap still allows, never negative."""
        return max(0.0, self.daily_budget_usd - self.spent_today(now))

    def check_budget(self, worker: str, now: datetime | None = None) -> float:
        """The spend so far today; raises :class:`BudgetExceededError` when it has reached the cap."""
        spent = self.spent_today(now)
        if spent >= self.daily_budget_usd:
            error = BudgetExceededError(worker, spent, self.daily_budget_usd)
            logger.warning("advisor: %s", error)
            raise error
        return spent

    # --- one call ---

    def ask[T: BaseModel](
        self, worker: str, prompt: Prompt, output: type[T], *, now: datetime | None = None
    ) -> WorkerReply[T]:
        """Send ``prompt`` as ``worker`` and parse the answer into ``output`` (see the module docs for the statuses).
        Raises :class:`BudgetExceededError` before sending once the day's cap is reached and :class:`AdvisorError`
        when the API could not be reached; the call is recorded in ``llm_usage`` whenever it got an answer."""
        self.check_budget(worker, now)
        params = prompt.params(self.model)
        raw = self._transport.parse(params, output, fallback=self.refusal_fallback)
        return self._reply(worker, raw, output, league_id=prompt.league_id, batch=False, now=now)

    # --- batches ---

    def submit_batch(
        self,
        worker: str,
        prompts: Mapping[str, Prompt],
        output: type[BaseModel],
        *,
        league_id: int | None = None,
        now: datetime | None = None,
    ) -> SubmittedBatch:
        """Send ``prompts`` (by the caller's ids) as one batch for ``worker``; nothing is recorded until the results
        are collected. The requests never carry ``fallbacks`` (the Batches API rejects it), so a batch refusal is
        just a refusal. Raises :class:`BudgetExceededError` when the cap is already reached, ``ValueError`` for an
        empty or oversized batch."""
        if not prompts:
            raise ValueError("a batch needs at least one prompt")
        if len(prompts) > MAX_BATCH_REQUESTS:
            raise ValueError(f"a batch holds at most {MAX_BATCH_REQUESTS} requests, got {len(prompts)}")
        self.check_budget(worker, now)
        schema = output_format(output)
        requests: list[BatchRequest] = []
        for custom_id, prompt in prompts.items():
            params = prompt.params(self.model)
            requests.append(
                BatchRequest(
                    custom_id,
                    MessageCreateParamsNonStreaming(
                        model=params["model"],
                        max_tokens=params["max_tokens"],
                        system=params["system"],
                        messages=params["messages"],
                        output_config={**params["output_config"], "format": schema},
                    ),
                )
            )
        batch_id = self._transport.create_batch(requests)
        logger.info("advisor: %s submitted batch %s with %d requests", worker, batch_id, len(requests))
        return SubmittedBatch(
            batch_id=batch_id,
            worker=worker,
            custom_ids=tuple(prompts),
            submitted_at=now if now is not None else utc_now(),
            league_id=league_id,
        )

    def batch_status(self, batch_id: str) -> BatchStatus:
        return self._transport.batch_status(batch_id)

    def collect_batch[T: BaseModel](
        self, submitted: SubmittedBatch, output: type[T], *, now: datetime | None = None
    ) -> CollectedBatch | None:
        """The batch's replies once it has ended, each recorded in ``llm_usage`` at the batch price; ``None`` while
        it is still processing. A request the batch did not answer is listed in ``failed`` with the reason."""
        status = self._transport.batch_status(submitted.batch_id)
        if not status.ended:
            return None
        replies: dict[str, WorkerReply[Any]] = {}
        failed: dict[str, str] = {}
        for result in self._transport.batch_results(submitted.batch_id, output):
            if result.custom_id not in submitted.custom_ids:
                logger.warning("advisor: batch %s answered unknown request %r", submitted.batch_id, result.custom_id)
                continue
            if result.reply is None:
                failed[result.custom_id] = result.error or "no result"
                continue
            replies[result.custom_id] = self._reply(
                submitted.worker, result.reply, output, league_id=submitted.league_id, batch=True, now=now
            )
        for custom_id in submitted.custom_ids:
            if custom_id not in replies and custom_id not in failed:
                failed[custom_id] = "missing from the results"
        collected = CollectedBatch(submitted.batch_id, status, MappingProxyType(replies), MappingProxyType(failed))
        logger.info("advisor: %s collected %s", submitted.worker, collected.describe())
        return collected

    # --- recording ---

    def _reply[T: BaseModel](
        self,
        worker: str,
        raw: RawReply,
        output: type[T],
        *,
        league_id: int | None,
        batch: bool,
        now: datetime | None,
    ) -> WorkerReply[T]:
        status, detail = _status_of(raw, output)
        # Recorded and priced under the model that answered: a refusal fallback can serve another model.
        pricing = self.pricing if raw.model == self.model else pricing_for(raw.model)
        usage = LlmUsageRow(
            called_at=now if now is not None else utc_now(),
            worker=worker,
            model=raw.model,
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            cache_creation_input_tokens=raw.usage.cache_creation_input_tokens,
            cache_read_input_tokens=raw.usage.cache_read_input_tokens,
            cost_usd=pricing.cost(raw.usage, batch=batch),
            batch=batch,
            stop_reason=raw.stop_reason,
            request_id=raw.request_id,
            league_id=league_id,
        )
        usage = self.store.llm_usage.insert(usage)
        if raw.fallback:
            logger.info(
                "advisor: %s: %s declined; served by the fallback model %s (request %s)",
                worker,
                self.model,
                raw.model,
                raw.request_id,
            )
        parsed = raw.output if status == "ok" and isinstance(raw.output, output) else None
        reply = WorkerReply(worker, status, parsed, raw.stop_reason, usage, detail)
        level = logging.INFO if reply.ok else logging.WARNING
        logger.log(level, "advisor: %s", reply.describe())
        return reply


def _status_of(raw: RawReply, output: type[BaseModel]) -> tuple[ReplyStatus, str | None]:
    """The reply's status from the stop reason first, then whether the output is there and of the right type."""
    match raw.stop_reason:
        case "end_turn":
            if isinstance(raw.output, output):
                return "ok", None
            return "unparseable", raw.detail or f"no {output.__name__} in the answer"
        case "refusal":
            return "refusal", raw.detail or "the model declined to answer"
        case "max_tokens":
            return "max_tokens", raw.detail or "the answer was cut off at max_tokens; not used"
        case None:
            return "unparseable", raw.detail or "no answer was parsed"
        case other:
            return "stopped", raw.detail or f"stopped with {other}; no output read"


# --- reporting --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpendReport:
    """The day's Claude spend against the cap, for ``fm`` output and logs."""

    since: datetime
    budget_usd: float
    calls: int
    spent_usd: float
    by_worker: Mapping[str, float]

    @property
    def left_usd(self) -> float:
        return max(0.0, self.budget_usd - self.spent_usd)

    def describe(self) -> str:
        """One line: ``Claude since 2026-10-06 00:00 UTC: 7 calls, $0.31 of $2.00 (news_triage $0.31)``."""
        workers = ", ".join(f"{worker} ${cost:.2f}" for worker, cost in sorted(self.by_worker.items()))
        return (
            f"Claude since {self.since:%Y-%m-%d %H:%M} UTC: {self.calls} calls, ${self.spent_usd:.2f} of "
            f"${self.budget_usd:.2f}" + (f" ({workers})" if workers else "")
        )


def spend_report(client: AdvisorClient, now: datetime | None = None) -> SpendReport:
    """Today's recorded spend by worker (00:00 UTC to ``now``)."""
    since = day_start(now if now is not None else utc_now())
    rows = client.store.llm_usage.since(since)
    by_worker: dict[str, float] = {}
    for row in rows:
        by_worker[row.worker] = by_worker.get(row.worker, 0.0) + row.cost_usd
    return SpendReport(
        since=since,
        budget_usd=client.daily_budget_usd,
        calls=len(rows),
        spent_usd=math.fsum(row.cost_usd for row in rows),
        by_worker=MappingProxyType(by_worker),
    )
