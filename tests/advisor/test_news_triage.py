"""News triage over a fake transport: what is asked, what is stored (and in the shape the availability model reads),
and what happens on a refusal, a cut-off answer, a bad answer, the budget cap, and in a batch."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import BaseModel

from fm.advisor.client import (
    AdvisorClient,
    AdvisorError,
    BatchRequest,
    BatchResult,
    BatchStatus,
    CallParams,
    RawReply,
    TokenUsage,
    Transport,
)
from fm.advisor.news_triage import (
    WORKER,
    TriageBatch,
    TriageChunk,
    TriagedItem,
    TriageOutput,
    TriageSignal,
    candidates_of,
    collect_triage,
    signals_from,
    submit_triage,
    triage_news,
)
from fm.advisor.prompts import prompt_text
from fm.config import DEFAULT_MODEL, Llm, Sport
from fm.model.availability import weigh_news
from fm.model.relevance import Relevance, relevance_for
from fm.store import LeagueRow, NewsItemRow, PlayerRow, RosterEntryRow, Store, TeamRow

AS_OF = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
PUBLISHED = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)
WEEK, BENCH = 5, 20
DANIELS, ALLGEIER, MCLAURIN = 4426348, 4373626, 3121422
URL = "https://www.espn.com/nfl/story/_/id/46000001"


class FakeTransport(Transport):
    def __init__(self, *replies: RawReply | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[CallParams] = []
        self.batches: dict[str, list[BatchRequest]] = {}
        self.status: BatchStatus | None = None
        self.results: list[BatchResult] = []

    def parse(self, params: CallParams, output: type[BaseModel]) -> RawReply:
        assert output is TriageOutput
        self.calls.append(params)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def create_batch(self, requests: Sequence[BatchRequest]) -> str:
        batch_id = f"msgbatch_{len(self.batches) + 1}"
        self.batches[batch_id] = list(requests)
        return batch_id

    def batch_status(self, batch_id: str) -> BatchStatus:
        assert self.status is not None
        return self.status

    def batch_results(self, batch_id: str, output: type[BaseModel]) -> list[BatchResult]:
        return list(self.results)


def raw(stop_reason: str | None = "end_turn", output: object = None, *, detail: str | None = None) -> RawReply:
    return RawReply("msg_1", DEFAULT_MODEL, stop_reason, TokenUsage(1000, 200, 0, 0), output, detail)


def signal(espn_id: int, **overrides: Any) -> TriageSignal:
    fields: dict[str, Any] = {
        "espn_id": espn_id,
        "kind": "injury",
        "severity": "major",
        "games_out": 1,
        "p_active_delta": -0.25,
        "confidence": 0.9,
        "summary": "Ruled out for Sunday with a hamstring strain.",
    }
    return TriageSignal(**(fields | overrides))


def answer(*items: tuple[int, list[TriageSignal]]) -> RawReply:
    return raw(output=TriageOutput(items=[TriagedItem(news_item_id=item_id, signals=sig) for item_id, sig in items]))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


@pytest.fixture
def league(store: Store) -> LeagueRow:
    """An NFL league whose team 1 rosters Daniels and Allgeier; McLaurin is a known player on no roster."""
    league = store.leagues.upsert(
        LeagueRow(key="nfl", sport="nfl", espn_league_id=1234567, season=2026, team_id=1, as_of=AS_OF)
    )
    store.teams.upsert(TeamRow(league_id=league.row_id, team_id=1, name="Team 1", as_of=AS_OF))
    store.rosters.replace(
        league.row_id,
        WEEK,
        1,
        [
            RosterEntryRow(
                league_id=league.row_id,
                scoring_period_id=WEEK,
                team_id=1,
                espn_id=espn_id,
                lineup_slot_id=BENCH,
                as_of=AS_OF,
            )
            for espn_id in (DANIELS, ALLGEIER)
        ],
    )
    for espn_id, name, position in (
        (DANIELS, "Jayden Daniels", "QB"),
        (ALLGEIER, "Tyler Allgeier", "RB"),
        (MCLAURIN, "Terry McLaurin", "WR"),
    ):
        store.players.upsert(
            PlayerRow(sport="nfl", espn_id=espn_id, full_name=name, position=position, pro_team="WSH", as_of=AS_OF)
        )
    return league


@pytest.fixture
def relevance(store: Store, league: LeagueRow) -> Relevance:
    return relevance_for(store, league.row_id)


def news(
    store: Store,
    external_id: str,
    espn_ids: Sequence[int],
    *,
    sport: Sport = "nfl",
    url: str | None = URL,
    published: datetime = PUBLISHED,
    source: str = "espn_news",
) -> NewsItemRow:
    row = store.news.ingest(
        NewsItemRow(
            source=source,
            external_id=external_id,
            sport=sport,
            title=f"News {external_id}",
            body=f"Body of {external_id}.",
            url=url,
            espn_ids=list(espn_ids),
            published_at=published,
            fetched_at=published + timedelta(minutes=10),
        )
    )
    assert row is not None
    return row


def make_client(
    store: Store, *replies: RawReply | Exception, budget: float = 2.0
) -> tuple[AdvisorClient, FakeTransport]:
    transport = FakeTransport(*replies)
    return AdvisorClient(store, Llm(daily_budget_usd=budget), transport), transport


def triaged(store: Store, item: NewsItemRow) -> bool:
    row = store.news.get(item.row_id)
    assert row is not None
    return row.triaged_at is not None


# --- the happy path ---


def test_parsed_output_is_stored_as_the_signals_availability_reads(
    store: Store, league: LeagueRow, relevance: Relevance, caplog: pytest.LogCaptureFixture
) -> None:
    about_daniels = news(store, "1", [DANIELS])
    about_nobody = news(store, "2", [MCLAURIN])  # known, but on no roster and not on the wire: irrelevant
    nba_item = news(store, "3", [], sport="nba")
    client, transport = make_client(
        store,
        answer(
            (about_daniels.row_id, [signal(DANIELS, p_active_delta=-0.5, confidence=1.4)]),
            (about_nobody.row_id, [signal(MCLAURIN)]),  # not asked: dropped
        ),
    )

    with caplog.at_level(logging.WARNING, logger="fm.advisor.news_triage"):
        report = triage_news(store, client, relevance, now=NOW)

    assert (report.read, report.relevant, report.triaged, report.skipped) == (2, 1, 2, 1)
    assert report.ok and report.pending == 0 and report.calls == 1 and report.cost_usd > 0
    assert report.describe().startswith(
        "news triage (nfl): 2 read, 1 relevant, 2 triaged, 1 signals, 0 refused, 1 call, $"
    )
    (stored,) = report.signals
    assert stored.id is not None
    assert (stored.news_item_id, stored.sport, stored.espn_id, stored.kind, stored.severity) == (
        about_daniels.row_id,
        "nfl",
        DANIELS,
        "injury",
        "major",
    )
    assert stored.p_active_delta == -0.3  # clamped to the bound, the proposal not the weighted effect
    assert stored.confidence == 1.0
    assert stored.games_out == 1
    assert stored.source_url == URL and stored.published_at == PUBLISHED and stored.created_at == NOW
    assert stored.summary == "Ruled out for Sunday with a hamstring strain."
    assert store.news_signals.for_player("nfl", DANIELS) == [stored]
    assert "proposed -0.50, outside +/-0.30; clamped" in caplog.text
    assert "confidence 1.40 outside [0, 1]; clamped" in caplog.text
    assert f"answer names item {about_nobody.row_id}, which was not asked; dropped" in caplog.text
    assert triaged(store, about_daniels) and triaged(store, about_nobody)
    assert not triaged(store, nba_item)  # another sport's item is left for its own triage
    # The availability model counts it: cited, finite, in the window, clamped and weighted by confidence.
    effect = weigh_news([stored], since=PUBLISHED - timedelta(days=1), until=NOW)
    assert [entry.status for entry in effect.entries] == ["counted"]
    assert effect.delta == pytest.approx(-0.3)
    (row,) = store.llm_usage.since(NOW - timedelta(days=1))
    assert row.worker == WORKER and row.league_id == league.row_id and row.stop_reason == "end_turn"


def test_the_prompt_names_the_candidates_and_the_items(store: Store, relevance: Relevance) -> None:
    item = news(store, "1", [DANIELS, ALLGEIER])
    client, transport = make_client(store, answer((item.row_id, [])))
    triage_news(store, client, relevance, now=NOW)

    (params,) = transport.calls
    assert params["output_config"] == {"effort": "low"}
    assert params["max_tokens"] == 8192
    system, context = params["system"]
    assert system["text"] == prompt_text("news_triage") and system.get("cache_control") == {"type": "ephemeral"}
    assert "- nfl (nfl): our team 1, no opponent given" in context["text"]
    assert "- 4426348: Jayden Daniels, QB, WSH; roster" in context["text"]
    assert "- 4373626: Tyler Allgeier, RB, WSH; roster" in context["text"]
    assert "3121422" not in context["text"]
    (message,) = params["messages"]
    user = message["content"][0]["text"]  # type: ignore[index]
    assert f"Item {item.row_id} (espn_news, published 2026-10-06T13:00:00Z, {URL})" in user
    assert "candidates: 4373626 (Tyler Allgeier), 4426348 (Jayden Daniels)" in user
    assert "title: News 1" in user and "body: Body of 1." in user


def test_the_latest_signal_of_a_kind_supersedes_earlier_ones(store: Store, relevance: Relevance) -> None:
    first = news(store, "1", [DANIELS], published=PUBLISHED - timedelta(hours=3))
    second = news(store, "2", [DANIELS], published=PUBLISHED)
    client, _ = make_client(
        store,
        answer(
            (first.row_id, [signal(DANIELS, p_active_delta=-0.3, severity="major")]),
            (second.row_id, [signal(DANIELS, p_active_delta=0.1, severity="minor", summary="Expected to play.")]),
        ),
    )
    report = triage_news(store, client, relevance, now=NOW)
    assert len(report.signals) == 2
    effect = weigh_news(store.news_signals.for_player("nfl", DANIELS), since=PUBLISHED - timedelta(days=1), until=NOW)
    assert [(entry.signal.news_item_id, entry.status) for entry in effect.entries] == [
        (first.row_id, "superseded"),
        (second.row_id, "counted"),
    ]
    assert effect.delta == pytest.approx(0.1 * 0.9)


def test_signals_are_checked_before_they_are_stored(store: Store, caplog: pytest.LogCaptureFixture) -> None:
    cited = news(store, "1", [DANIELS])
    uncited = news(store, "2", [ALLGEIER], url=None)
    chunk = TriageChunk((cited, uncited), MappingProxyType({cited.row_id: (DANIELS,), uncited.row_id: (ALLGEIER,)}))
    output = TriageOutput(
        items=[
            TriagedItem(
                news_item_id=cited.row_id,
                signals=[
                    signal(MCLAURIN),  # not a candidate of the item
                    signal(DANIELS, p_active_delta=math.nan),  # non-finite
                    signal(DANIELS, kind="role", p_active_delta=-0.1, games_out=-2),
                    signal(DANIELS, kind="role", p_active_delta=-0.2),  # same kind: replaces the one above
                    signal(DANIELS, kind="rest", p_active_delta=0.05, summary="  "),
                ],
            ),
            TriagedItem(news_item_id=uncited.row_id, signals=[signal(ALLGEIER)]),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="fm.advisor.news_triage"):
        rows = signals_from(output, chunk, now=NOW)

    assert [(row.espn_id, row.kind, row.p_active_delta, row.games_out, row.summary) for row in rows] == [
        (DANIELS, "role", -0.2, 1, "Ruled out for Sunday with a hamstring strain."),
        (DANIELS, "rest", 0.05, 1, None),
    ]
    assert all(math.isfinite(row.p_active_delta) and row.source_url for row in rows)
    assert f"ESPN {MCLAURIN} (injury): not a candidate of the item ({DANIELS}); dropped" in caplog.text
    assert "non-finite delta or confidence; dropped" in caplog.text
    assert f"item {uncited.row_id}, ESPN {ALLGEIER} (injury): the item has no URL to cite; dropped" in caplog.text


# --- when Claude does not answer ---


def test_a_refusal_stores_nothing_and_marks_the_items_read(
    store: Store, relevance: Relevance, caplog: pytest.LogCaptureFixture
) -> None:
    item = news(store, "1", [DANIELS])
    client, _ = make_client(store, raw("refusal", detail="general_harms: declined"))
    with caplog.at_level(logging.WARNING, logger="fm.advisor.news_triage"):
        report = triage_news(store, client, relevance, now=NOW)

    assert report.ok and report.signals == () and report.refused == (item.row_id,)
    assert (report.triaged, report.pending, report.calls) == (1, 0, 1)
    assert store.news_signals.for_item(item.row_id) == []
    assert triaged(store, item)
    assert f"Claude declined items {item.row_id} (general_harms: declined)" in caplog.text


def test_a_cut_off_answer_is_asked_again_in_halves_and_never_read(store: Store, relevance: Relevance) -> None:
    a = news(store, "1", [DANIELS])
    b = news(store, "2", [ALLGEIER], published=PUBLISHED + timedelta(minutes=1))
    partial = TriageOutput(items=[TriagedItem(news_item_id=a.row_id, signals=[signal(DANIELS)])])
    client, transport = make_client(
        store,
        raw("max_tokens", partial),  # the SDK may still attach something; it is not data
        answer((a.row_id, [signal(DANIELS)])),
        answer((b.row_id, [signal(ALLGEIER, kind="role")])),
    )
    report = triage_news(store, client, relevance, chunk_size=2, now=NOW)

    assert report.ok and report.calls == 3
    assert [row.espn_id for row in report.signals] == [DANIELS, ALLGEIER]
    users = [str(params["messages"][0]["content"][0]["text"]) for params in transport.calls]  # type: ignore[index]
    assert f"Item {a.row_id}" in users[0] and f"Item {b.row_id}" in users[0]
    assert f"Item {a.row_id}" in users[1] and f"Item {b.row_id}" not in users[1]
    assert f"Item {b.row_id}" in users[2] and f"Item {a.row_id}" not in users[2]
    assert len(store.llm_usage.since(NOW - timedelta(days=1))) == 3  # the cut-off call cost something too


def test_a_single_item_cut_off_is_a_failure_left_for_the_next_run(store: Store, relevance: Relevance) -> None:
    item = news(store, "1", [DANIELS])
    client, _ = make_client(store, raw("max_tokens"))
    report = triage_news(store, client, relevance, now=NOW)
    assert not report.ok and report.pending == 1 and report.signals == ()
    (failure,) = report.failures
    assert failure.item_ids == (item.row_id,) and failure.status == "max_tokens"
    assert not triaged(store, item)
    assert "1 items failed (max_tokens)" in report.describe()


@pytest.mark.parametrize("reply", [raw("end_turn", None), raw("end_turn", "junk"), raw("tool_use")])
def test_an_unusable_answer_leaves_the_items_untriaged(store: Store, relevance: Relevance, reply: RawReply) -> None:
    item = news(store, "1", [DANIELS])
    client, _ = make_client(store, reply)
    report = triage_news(store, client, relevance, now=NOW)
    assert not report.ok and report.failures[0].item_ids == (item.row_id,)
    assert report.failures[0].status in {"unparseable", "stopped"}
    assert not triaged(store, item) and store.news.untriaged() == [store.news.get(item.row_id)]


def test_the_budget_cap_blocks_the_call_and_keeps_the_items(store: Store, relevance: Relevance) -> None:
    relevant = news(store, "1", [DANIELS])
    irrelevant = news(store, "2", [MCLAURIN])
    client, transport = make_client(store, answer((relevant.row_id, [signal(DANIELS)])), budget=0.0)
    report = triage_news(store, client, relevance, now=NOW)

    assert transport.calls == []
    assert report.blocked is not None and "daily Claude budget reached" in report.blocked
    assert not report.ok and report.calls == 0 and report.signals == ()
    assert (report.triaged, report.skipped, report.pending) == (1, 1, 1)
    assert not triaged(store, relevant) and triaged(store, irrelevant)
    assert report.describe().endswith(f"stopped: {report.blocked}")


def test_an_api_failure_stops_the_run(store: Store, relevance: Relevance) -> None:
    item = news(store, "1", [DANIELS])
    client, _ = make_client(store, AdvisorError("boom"))
    report = triage_news(store, client, relevance, now=NOW)
    assert report.blocked == "API failure: boom" and not triaged(store, item)


def test_nothing_relevant_means_no_call(store: Store, relevance: Relevance) -> None:
    item = news(store, "1", [MCLAURIN])
    client, transport = make_client(store)
    report = triage_news(store, client, relevance, now=NOW)
    assert transport.calls == [] and report.calls == 0
    assert (report.read, report.relevant, report.triaged, report.skipped) == (1, 0, 1, 1)
    assert triaged(store, item)


def test_leagues_must_share_a_sport(relevance: Relevance) -> None:
    with pytest.raises(ValueError, match="at least one league"):
        candidates_of([])
    nba = Relevance(
        league_id=9,
        key="nba",
        sport="nba",
        team_id=1,
        scoring_period_id=None,
        opponent_team_id=None,
        reasons=MappingProxyType({}),
        players=MappingProxyType({}),
    )
    with pytest.raises(ValueError, match="one sport at a time: nba is nba, not nfl"):
        candidates_of([relevance, nba])


# --- overnight ---


def test_a_batch_is_submitted_saved_and_collected(
    store: Store, league: LeagueRow, relevance: Relevance, tmp_path: Path
) -> None:
    a = news(store, "1", [DANIELS])
    b = news(store, "2", [ALLGEIER], published=PUBLISHED + timedelta(minutes=1))
    irrelevant = news(store, "3", [MCLAURIN])
    client, transport = make_client(store)

    batch = submit_triage(store, client, relevance, chunk_size=1, now=NOW)
    assert batch is not None
    assert batch.batch_id == "msgbatch_1" and batch.sport == "nfl" and batch.league_id == league.row_id
    assert batch.chunks == {"news_triage-0": {a.row_id: (DANIELS,)}, "news_triage-1": {b.row_id: (ALLGEIER,)}}
    assert (batch.read, batch.relevant, batch.skipped) == (3, 2, 1)
    assert triaged(store, irrelevant) and not triaged(store, a) and not triaged(store, b)
    requests = transport.batches["msgbatch_1"]
    assert [r.custom_id for r in requests] == ["news_triage-0", "news_triage-1"]
    assert requests[0].params["output_config"]["format"]["type"] == "json_schema"  # type: ignore[typeddict-item]
    path = batch.save(tmp_path / "batches")
    assert path == tmp_path / "batches" / "msgbatch_1.json"
    assert TriageBatch.load("msgbatch_1", tmp_path / "batches") == batch
    with pytest.raises(FileNotFoundError, match="no saved triage batch 'msgbatch_2'"):
        TriageBatch.load("msgbatch_2", tmp_path / "batches")

    transport.status = BatchStatus("msgbatch_1", "in_progress", processing=2)
    assert collect_triage(store, client, batch, now=NOW) is None
    assert store.llm_usage.since(NOW - timedelta(days=1)) == []

    transport.status = BatchStatus("msgbatch_1", "ended", succeeded=1, errored=1)
    transport.results = [BatchResult("news_triage-0", answer((a.row_id, [signal(DANIELS)])))]
    report = collect_triage(store, client, batch, now=NOW + timedelta(hours=8))
    assert report is not None
    assert (report.read, report.relevant, report.triaged, report.skipped, report.calls) == (3, 2, 2, 1, 1)
    (stored,) = report.signals
    assert stored.espn_id == DANIELS and stored.news_item_id == a.row_id and stored.source_url == URL
    assert triaged(store, a) and not triaged(store, b)
    (failure,) = report.failures
    assert failure.item_ids == (b.row_id,) and failure.status == "unanswered"
    assert failure.detail == "missing from the results"
    (row,) = store.llm_usage.since(NOW - timedelta(days=1))
    assert row.batch is True and row.worker == WORKER and row.called_at == NOW + timedelta(hours=8)
    assert row.cost_usd == pytest.approx(client.pricing.cost(TokenUsage(1000, 200, 0, 0), batch=True))


def test_nothing_relevant_submits_no_batch(store: Store, relevance: Relevance) -> None:
    item = news(store, "1", [MCLAURIN])
    client, transport = make_client(store)
    assert submit_triage(store, client, relevance, now=NOW) is None
    assert transport.batches == {} and triaged(store, item)
