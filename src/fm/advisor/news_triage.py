"""The ``news_triage`` worker: Claude reads the new news about relevant players and stores signals (DESIGN section 10).

:func:`triage_news` takes the untriaged items (``store.news.untriaged()``, or the items given), keeps the ones about a
player who matters to the leagues given (:func:`fm.model.relevance.relevant_news` over their :class:`Relevance`,
all of one sport), and asks Claude, a chunk of items at a time, what each item changes for each candidate player.
The answer is a :class:`TriageOutput`: per item, zero or more :class:`TriageSignal` (the DESIGN table's
``{player_id, kind, severity, games_out, p_active_delta, confidence}``). Each is stored as a ``news_signals`` row
(:class:`fm.store.NewsSignalRow`), the shape :func:`fm.model.availability.assess` reads:

- the player must be one of the item's candidates and the item one of the chunk's; anything else is logged and
  dropped (Claude never adds a player);
- ``p_active_delta`` must be finite, and is clamped to ``±NEWS_BOUND`` (0.3) here, logged when it was outside; the
  availability model clamps it again and weighs it by ``confidence`` (clamped to [0, 1] here) when it applies it, so
  the stored number is the proposal, not the weighted effect;
- ``source_url`` and ``published_at`` are the item's, never Claude's: a signal is cited by construction, and an item
  with no URL yields no signal (the model would ignore an uncited one);
- one signal per (item, player, kind): a second reading of a kind in one answer replaces the first. Across items,
  the latest signal of a kind supersedes earlier ones when the model reads them, by ``published_at``.

Every item read is marked triaged, relevant or not, so the next page of ``untriaged()`` is new (an item of another
sport is left alone). The statuses of :class:`fm.advisor.client.WorkerReply` map to:

- ``ok``: signals stored, the chunk's items marked triaged;
- ``refusal``: no signal stored, logged, the items marked triaged anyway: the engine runs on designations alone for
  them, which is the fallback (DESIGN: a Claude-only signal is an adjustment, never a requirement);
- ``max_tokens``: no partial data is read; a chunk of several items is split in two and asked again, a single item
  is reported as failed and left untriaged;
- ``unparseable`` or ``stopped``: the chunk is reported as failed and left for the next run;
- :class:`fm.advisor.client.BudgetExceededError`: the run stops there (``blocked``) and the rest waits for tomorrow.

**Overnight.** :func:`submit_triage` sends the same chunks as one batch (half price) and returns a
:class:`TriageBatch`, which :func:`collect_triage` turns into the same stored signals and report once the batch has
ended. The batch remembers each chunk's items and candidates (``TriageBatch.save`` writes it under the cache dir, so
a later process can ``load`` it); nothing is marked triaged until collection, so a lost batch just means the items
are triaged again. A batch chunk that was cut off is not retried: its items stay untriaged for the next live run.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from fm import paths
from fm.advisor.client import (
    AdvisorClient,
    AdvisorError,
    BudgetExceededError,
    Prompt,
    SubmittedBatch,
    WorkerReply,
    effort_for,
)
from fm.advisor.prompts import prompt_text
from fm.config import Sport
from fm.model.availability import NEWS_BOUND
from fm.model.relevance import Relevance, RelevanceReason, relevant_news
from fm.store import NewsItemRow, NewsSignalRow, SignalKind, Store, utc_now

logger = logging.getLogger(__name__)

WORKER: Final = "news_triage"
CHUNK_SIZE: Final = 20
"""Items per call: enough to share the cached prefix, few enough that an answer fits ``MAX_TOKENS``."""
MAX_TOKENS: Final = 8192
"""Room for several signals per item (each is a short JSON object)."""
BODY_LIMIT: Final = 2000
"""Characters of an item's body sent; a RotoWire blurb is a few hundred, an ESPN story can run long."""

type Severity = Literal["minor", "moderate", "major", "season_ending"]


# --- the structured output --------------------------------------------------------------------------------------------


class TriageSignal(BaseModel):
    """Claude's reading of one item for one player (the schema the API is held to; bounds are applied in code)."""

    model_config = ConfigDict(extra="forbid")

    espn_id: int = Field(description="One of the item's candidate ESPN ids.")
    kind: SignalKind
    severity: Severity
    games_out: int | None = Field(default=None, description="Games the item says he misses; null when it does not say.")
    p_active_delta: float = Field(description="Change to the chance he plays his next game, -0.3 to 0.3.")
    confidence: float = Field(description="0 to 1.")
    summary: str = Field(description="One sentence citing what the item says.")


class TriagedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    news_item_id: int
    signals: list[TriageSignal]


class TriageOutput(BaseModel):
    """One entry per item given, in order."""

    model_config = ConfigDict(extra="forbid")

    items: list[TriagedItem]


# --- what a chunk asks ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """A relevant player as the prompt names him."""

    espn_id: int
    name: str
    position: str | None
    pro_team: str | None
    reasons: frozenset[RelevanceReason]

    def describe(self) -> str:
        parts = [self.name]
        if self.position:
            parts.append(self.position)
        if self.pro_team:
            parts.append(self.pro_team)
        return f"{self.espn_id}: {', '.join(parts)}; {', '.join(sorted(r.value for r in self.reasons))}"


@dataclass(frozen=True, slots=True)
class TriageChunk:
    """The items of one call and, per item, the candidate ESPN ids it may be about."""

    items: tuple[NewsItemRow, ...]
    candidates: Mapping[int, tuple[int, ...]]
    """Item id -> the relevant players the item is about."""

    @property
    def item_ids(self) -> tuple[int, ...]:
        return tuple(item.row_id for item in self.items)

    def split(self) -> tuple[TriageChunk, TriageChunk]:
        half = len(self.items) // 2
        return self._slice(self.items[:half]), self._slice(self.items[half:])

    def _slice(self, items: tuple[NewsItemRow, ...]) -> TriageChunk:
        return TriageChunk(items, MappingProxyType({item.row_id: self.candidates[item.row_id] for item in items}))


def candidates_of(relevances: Sequence[Relevance]) -> dict[int, Candidate]:
    """The relevant players of the leagues given, merged (one sport; ``ValueError`` otherwise)."""
    if not relevances:
        raise ValueError("news triage needs at least one league's relevance")
    sport = relevances[0].sport
    found: dict[int, Candidate] = {}
    for relevance in relevances:
        if relevance.sport != sport:
            raise ValueError(
                f"news triage reads one sport at a time: {relevance.key} is {relevance.sport}, not {sport}"
            )
        for espn_id, reasons in relevance.reasons.items():
            player = relevance.players.get(espn_id)
            earlier = found.get(espn_id)
            found[espn_id] = Candidate(
                espn_id=espn_id,
                name=player.full_name if player is not None else relevance.name(espn_id),
                position=player.position if player is not None else None,
                pro_team=player.pro_team if player is not None else None,
                reasons=(earlier.reasons if earlier is not None else frozenset()) | reasons,
            )
    return found


def league_context(relevances: Sequence[Relevance], candidates: Mapping[int, Candidate]) -> str:
    """The cached context block: the leagues and every relevant player with why he matters."""
    lines = ["Leagues:"]
    for relevance in relevances:
        opponent = (
            f"opponent team {relevance.opponent_team_id}"
            if relevance.opponent_team_id is not None
            else "no opponent given"
        )
        lines.append(f"- {relevance.key} ({relevance.sport}): our team {relevance.team_id}, {opponent}")
    lines.append("")
    lines.append("Relevant players (ESPN id: name, position, pro team; why they matter):")
    lines.extend(f"- {candidate.describe()}" for candidate in sorted(candidates.values(), key=lambda c: c.espn_id))
    return "\n".join(lines)


def items_text(chunk: TriageChunk, candidates: Mapping[int, Candidate]) -> str:
    """The user block: each item with its id, source, time, URL, candidates, title and body."""
    blocks: list[str] = []
    for item in chunk.items:
        who = ", ".join(
            f"{espn_id} ({candidates[espn_id].name})" if espn_id in candidates else str(espn_id)
            for espn_id in chunk.candidates[item.row_id]
        )
        body = (item.body or "").strip()
        if len(body) > BODY_LIMIT:
            body = body[:BODY_LIMIT].rstrip() + " [...]"
        lines = [
            f"Item {item.row_id} ({item.source}, published {item.published_at:%Y-%m-%dT%H:%M:%SZ}"
            + (f", {item.url}" if item.url else "")
            + ")",
            f"candidates: {who}",
            f"title: {item.title.strip()}",
        ]
        if body:
            lines.append(f"body: {body}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def chunk_items(relevant: Iterable[tuple[NewsItemRow, tuple[int, ...]]], chunk_size: int) -> list[TriageChunk]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")
    pairs = list(relevant)
    chunks: list[TriageChunk] = []
    for start in range(0, len(pairs), chunk_size):
        part = pairs[start : start + chunk_size]
        chunks.append(
            TriageChunk(tuple(item for item, _ in part), MappingProxyType({item.row_id: ids for item, ids in part}))
        )
    return chunks


def _prompt(
    chunk: TriageChunk, context: str, candidates: Mapping[int, Candidate], *, max_tokens: int, league_id: int | None
) -> Prompt:
    return Prompt(
        system=prompt_text(WORKER),
        context=context,
        user=items_text(chunk, candidates),
        max_tokens=max_tokens,
        effort=effort_for(WORKER),
        league_id=league_id,
    )


# --- storing an answer ------------------------------------------------------------------------------------------------


def signals_from(output: TriageOutput, chunk: TriageChunk, *, now: datetime) -> list[NewsSignalRow]:
    """The rows to store for an answer, after the checks in the module docs (each rejection and clamp logged)."""
    kept: dict[tuple[int, int, str], NewsSignalRow] = {}
    items = {item.row_id: item for item in chunk.items}
    for entry in output.items:
        item = items.get(entry.news_item_id)
        if item is None:
            logger.warning("news triage: answer names item %d, which was not asked; dropped", entry.news_item_id)
            continue
        allowed = chunk.candidates[item.row_id]
        for signal in entry.signals:
            label = f"news triage: item {item.row_id}, ESPN {signal.espn_id} ({signal.kind})"
            if signal.espn_id not in allowed:
                logger.warning("%s: not a candidate of the item (%s); dropped", label, ", ".join(map(str, allowed)))
                continue
            if not (item.url or "").strip():
                logger.warning("%s: the item has no URL to cite; dropped", label)
                continue
            if not math.isfinite(signal.p_active_delta) or not math.isfinite(signal.confidence):
                logger.warning("%s: non-finite delta or confidence; dropped", label)
                continue
            delta = min(NEWS_BOUND, max(-NEWS_BOUND, signal.p_active_delta))
            if delta != signal.p_active_delta:
                logger.warning("%s: proposed %+.2f, outside +/-%.2f; clamped", label, signal.p_active_delta, NEWS_BOUND)
            confidence = min(1.0, max(0.0, signal.confidence))
            if confidence != signal.confidence:
                logger.warning("%s: confidence %.2f outside [0, 1]; clamped", label, signal.confidence)
            games_out = signal.games_out if signal.games_out is not None and signal.games_out >= 0 else None
            kept[(item.row_id, signal.espn_id, signal.kind)] = NewsSignalRow(
                news_item_id=item.row_id,
                sport=item.sport,
                espn_id=signal.espn_id,
                kind=signal.kind,
                severity=signal.severity,
                games_out=games_out,
                p_active_delta=delta,
                confidence=confidence,
                summary=signal.summary.strip() or None,
                source_url=item.url,
                published_at=item.published_at,
                created_at=now,
            )
    return list(kept.values())


def _store_answer(
    store: Store, rows: Iterable[NewsSignalRow], item_ids: Iterable[int], now: datetime
) -> list[NewsSignalRow]:
    """Insert the rows and mark the items triaged, in one transaction."""
    stored: list[NewsSignalRow] = []
    with store.db.transaction():
        for row in rows:
            stored.append(store.news_signals.insert(row))
        store.news.mark_triaged(item_ids, now)
    for row in stored:
        logger.info(
            "news triage: stored signal %s for %s %d (%s, %s, %+.2f x %.2f): %s",
            row.id,
            row.sport,
            row.espn_id,
            row.kind,
            row.severity,
            row.p_active_delta,
            row.confidence,
            row.summary,
        )
    return stored


# --- the report -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TriageFailure:
    """A chunk that stored nothing and stays untriaged."""

    item_ids: tuple[int, ...]
    status: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class TriageReport:
    """What a run did: counts of items, the signals stored, what went wrong, and the spend."""

    sport: Sport
    read: int
    """Items considered (the page read, or the items given) of this sport."""
    relevant: int
    """Items about a relevant player."""
    triaged: int
    """Items marked triaged: the relevant ones answered plus the irrelevant ones skipped."""
    skipped: int
    """Items of the sport about no relevant player, marked triaged unread."""
    signals: tuple[NewsSignalRow, ...]
    refused: tuple[int, ...] = ()
    """Items Claude declined to read (marked triaged without signals)."""
    failures: tuple[TriageFailure, ...] = ()
    blocked: str | None = None
    """Why the run stopped early (the budget cap, an API failure); the rest of the items stay untriaged."""
    calls: int = 0
    cost_usd: float = 0.0
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.blocked is None and not self.failures

    @property
    def pending(self) -> int:
        """Relevant items left untriaged for the next run."""
        return self.relevant - (self.triaged - self.skipped)

    def describe(self) -> str:
        """One line: ``news triage (nfl): 12 read, 5 relevant, 11 triaged, 4 signals, 1 refused, 1 call, $0.0310``."""
        calls = f"{self.calls} call" + ("" if self.calls == 1 else "s")
        text = (
            f"news triage ({self.sport}): {self.read} read, {self.relevant} relevant, {self.triaged} triaged, "
            f"{len(self.signals)} signals, {len(self.refused)} refused, {calls}, ${self.cost_usd:.4f}"
        )
        if self.failures:
            text += f"; {sum(len(f.item_ids) for f in self.failures)} items failed ({self.failures[0].status})"
        if self.blocked:
            text += f"; stopped: {self.blocked}"
        return text


class _Run:
    """The mutable tally of one run, live or collected."""

    def __init__(self, store: Store, sport: Sport, now: datetime) -> None:
        self.store = store
        self.sport: Sport = sport
        self.now = now
        self.signals: list[NewsSignalRow] = []
        self.triaged = 0
        self.skipped = 0
        self.refused: list[int] = []
        self.failures: list[TriageFailure] = []
        self.calls = 0
        self.cost = 0.0
        self.warnings: list[str] = []

    def count(self, reply: WorkerReply[TriageOutput]) -> None:
        self.calls += 1
        self.cost += reply.usage.cost_usd

    def answered(self, reply: WorkerReply[TriageOutput], chunk: TriageChunk) -> bool:
        """Apply an answer: stored or refused returns True; a failure that leaves the chunk untriaged, False."""
        if reply.ok and reply.output is not None:
            rows = signals_from(reply.output, chunk, now=self.now)
            self.signals.extend(_store_answer(self.store, rows, chunk.item_ids, self.now))
            self.triaged += len(chunk.items)
            return True
        if reply.status == "refusal":
            logger.warning(
                "news triage: Claude declined items %s (%s); the engine runs without signals for them",
                ", ".join(map(str, chunk.item_ids)),
                reply.detail,
            )
            _store_answer(self.store, (), chunk.item_ids, self.now)
            self.refused.extend(chunk.item_ids)
            self.triaged += len(chunk.items)
            return True
        self.failures.append(TriageFailure(chunk.item_ids, reply.status, reply.detail))
        return False

    def skip(self, item_ids: Sequence[int]) -> None:
        """Mark items about no relevant player triaged, unread."""
        if item_ids:
            self.store.news.mark_triaged(item_ids, self.now)
            self.skipped_before(len(item_ids))

    def skipped_before(self, count: int) -> None:
        """Count items already marked triaged as irrelevant (at submission, for a collected batch)."""
        self.skipped += count
        self.triaged += count

    def report(self, *, read: int, relevant: int, blocked: str | None) -> TriageReport:
        report = TriageReport(
            sport=self.sport,
            read=read,
            relevant=relevant,
            triaged=self.triaged,
            skipped=self.skipped,
            signals=tuple(self.signals),
            refused=tuple(self.refused),
            failures=tuple(self.failures),
            blocked=blocked,
            calls=self.calls,
            cost_usd=self.cost,
            warnings=tuple(self.warnings),
        )
        logger.log(logging.INFO if report.ok else logging.WARNING, "%s", report.describe())
        return report


def _select(
    store: Store, relevances: Sequence[Relevance], items: Iterable[NewsItemRow] | None
) -> tuple[list[NewsItemRow], list[tuple[NewsItemRow, tuple[int, ...]]], list[int]]:
    """(the items of the sport read, the relevant ones with their candidates, the irrelevant ones' ids)."""
    sport = relevances[0].sport
    read = [item for item in (list(items) if items is not None else store.news.untriaged()) if item.sport == sport]
    relevant: dict[int, tuple[NewsItemRow, set[int]]] = {}
    for relevance in relevances:
        for hit in relevant_news(read, relevance):
            entry = relevant.setdefault(hit.item.row_id, (hit.item, set()))
            entry[1].update(hit.espn_ids)
    pairs = [(item, tuple(sorted(ids))) for item, ids in relevant.values()]
    pairs.sort(key=lambda pair: (pair[0].published_at, pair[0].row_id))
    irrelevant = [item.row_id for item in read if item.row_id not in relevant]
    return read, pairs, irrelevant


# --- live -------------------------------------------------------------------------------------------------------------


def triage_news(
    store: Store,
    client: AdvisorClient,
    relevances: Relevance | Sequence[Relevance],
    *,
    items: Iterable[NewsItemRow] | None = None,
    chunk_size: int = CHUNK_SIZE,
    max_tokens: int = MAX_TOKENS,
    now: datetime | None = None,
) -> TriageReport:
    """Triage the untriaged items (or ``items``) for the leagues' relevant players and store the signals (see the
    module docs). Never raises for an answer: refusals, cut-offs, the budget cap and API failures are in the report."""
    leagues = (relevances,) if isinstance(relevances, Relevance) else tuple(relevances)
    candidates = candidates_of(leagues)
    stamp = now if now is not None else utc_now()
    run = _Run(store, leagues[0].sport, stamp)
    read, pairs, irrelevant = _select(store, leagues, items)
    run.skip(irrelevant)
    context = league_context(leagues, candidates)
    league_id = leagues[0].league_id if len(leagues) == 1 else None
    blocked: str | None = None
    queue = chunk_items(pairs, chunk_size)
    while queue:
        chunk = queue.pop(0)
        prompt = _prompt(chunk, context, candidates, max_tokens=max_tokens, league_id=league_id)
        try:
            reply = client.ask(WORKER, prompt, TriageOutput, now=stamp)
        except BudgetExceededError as exc:
            blocked = str(exc)
            break
        except AdvisorError as exc:
            blocked = f"API failure: {exc}"
            logger.error("news triage: %s", blocked)
            break
        run.count(reply)
        if reply.status == "max_tokens" and len(chunk.items) > 1:
            logger.warning("news triage: answer for %d items was cut off; asking in two halves", len(chunk.items))
            queue[:0] = list(chunk.split())
            continue
        run.answered(reply, chunk)
    return run.report(read=len(read), relevant=len(pairs), blocked=blocked)


# --- overnight --------------------------------------------------------------------------------------------------------


class TriageBatch(BaseModel):
    """A submitted triage batch: the chunks sent, by request id, with each item's candidates, so the answers can be
    checked and stored later, in another process (``save`` / ``load``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_id: str
    sport: Sport
    league_id: int | None
    submitted_at: datetime
    chunks: dict[str, dict[int, tuple[int, ...]]]
    """Request id -> item id -> candidate ESPN ids."""
    read: int
    relevant: int
    skipped: int

    @property
    def item_ids(self) -> tuple[int, ...]:
        return tuple(item_id for chunk in self.chunks.values() for item_id in chunk)

    def submitted(self) -> SubmittedBatch:
        return SubmittedBatch(
            batch_id=self.batch_id,
            worker=WORKER,
            custom_ids=tuple(self.chunks),
            submitted_at=self.submitted_at,
            league_id=self.league_id,
        )

    @staticmethod
    def path_for(batch_id: str, directory: Path | None = None) -> Path:
        base = directory if directory is not None else paths.cache_dir() / "advisor" / "batches"
        return base / f"{batch_id}.json"

    def save(self, directory: Path | None = None) -> Path:
        """Write the batch under the cache dir (``advisor/batches/<id>.json``) and return the path."""
        path = self.path_for(self.batch_id, directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, batch_id: str, directory: Path | None = None) -> TriageBatch:
        """Read a saved batch; ``FileNotFoundError`` names the path."""
        path = cls.path_for(batch_id, directory)
        if not path.is_file():
            raise FileNotFoundError(f"no saved triage batch {batch_id!r} at {path}")
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def submit_triage(
    store: Store,
    client: AdvisorClient,
    relevances: Relevance | Sequence[Relevance],
    *,
    items: Iterable[NewsItemRow] | None = None,
    chunk_size: int = CHUNK_SIZE,
    max_tokens: int = MAX_TOKENS,
    now: datetime | None = None,
) -> TriageBatch | None:
    """Send the same chunks :func:`triage_news` would ask as one batch. Irrelevant items are marked triaged now;
    the relevant ones wait for :func:`collect_triage`. ``None`` when there is nothing relevant to ask. Raises
    :class:`BudgetExceededError` and :class:`AdvisorError` like :meth:`AdvisorClient.submit_batch`."""
    leagues = (relevances,) if isinstance(relevances, Relevance) else tuple(relevances)
    candidates = candidates_of(leagues)
    stamp = now if now is not None else utc_now()
    read, pairs, irrelevant = _select(store, leagues, items)
    if irrelevant:
        store.news.mark_triaged(irrelevant, stamp)
    if not pairs:
        logger.info("news triage: nothing relevant to batch (%d items read, %d skipped)", len(read), len(irrelevant))
        return None
    context = league_context(leagues, candidates)
    league_id = leagues[0].league_id if len(leagues) == 1 else None
    chunks = chunk_items(pairs, chunk_size)
    prompts = {
        f"{WORKER}-{index}": _prompt(chunk, context, candidates, max_tokens=max_tokens, league_id=league_id)
        for index, chunk in enumerate(chunks)
    }
    submitted = client.submit_batch(WORKER, prompts, TriageOutput, league_id=league_id, now=stamp)
    return TriageBatch(
        batch_id=submitted.batch_id,
        sport=leagues[0].sport,
        league_id=league_id,
        submitted_at=submitted.submitted_at,
        chunks={custom_id: dict(chunk.candidates) for custom_id, chunk in zip(prompts, chunks, strict=True)},
        read=len(read),
        relevant=len(pairs),
        skipped=len(irrelevant),
    )


def collect_triage(
    store: Store, client: AdvisorClient, batch: TriageBatch, *, now: datetime | None = None
) -> TriageReport | None:
    """Store the batch's answers as :func:`triage_news` would; ``None`` while the batch is still processing. An
    item deleted since submission is skipped; a chunk the batch did not answer is a failure (left untriaged)."""
    stamp = now if now is not None else utc_now()
    collected = client.collect_batch(batch.submitted(), TriageOutput, now=stamp)
    if collected is None:
        return None
    run = _Run(store, batch.sport, stamp)
    run.skipped_before(batch.skipped)
    for custom_id, planned in batch.chunks.items():
        items = tuple(item for item_id in planned if (item := store.news.get(item_id)) is not None)
        if len(items) != len(planned):
            run.warnings.append(f"{custom_id}: {len(planned) - len(items)} items vanished since submission")
        chunk = TriageChunk(items, MappingProxyType({item.row_id: planned[item.row_id] for item in items}))
        reply = collected.replies.get(custom_id)
        if reply is None:
            run.failures.append(TriageFailure(chunk.item_ids, "unanswered", collected.failed.get(custom_id)))
            continue
        run.count(reply)
        run.answered(reply, chunk)
    return run.report(read=batch.read, relevant=batch.relevant, blocked=None)
