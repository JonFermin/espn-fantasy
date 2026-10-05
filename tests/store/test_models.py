"""Row models: UTC-only timestamps, JSON columns, bounds, literals, immutability."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from fm.store.models import (
    AvailabilityRow,
    LeagueRow,
    NewsSignalRow,
    ProjectionRow,
    ProposalRow,
    format_timestamp,
    utc_now,
)

EDT = timezone(timedelta(hours=-4))
AS_OF = datetime(2026, 10, 4, 9, 30, 15, 123456, tzinfo=EDT)


def league(**overrides: object) -> LeagueRow:
    fields: dict[str, object] = {
        "key": "nfl",
        "sport": "nfl",
        "espn_league_id": 123456,
        "season": 2026,
        "team_id": 4,
        "as_of": AS_OF,
    }
    fields.update(overrides)
    return LeagueRow.model_validate(fields)


def test_aware_datetimes_are_normalised_to_utc() -> None:
    row = league()
    assert row.as_of.tzinfo == UTC
    assert row.as_of == AS_OF
    assert (row.as_of.hour, row.as_of.microsecond) == (13, 123456)


def test_naive_datetimes_are_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        league(as_of=datetime(2026, 10, 4, 9, 30))


def test_format_timestamp_is_fixed_width_utc_and_sorts_chronologically() -> None:
    assert format_timestamp(AS_OF) == "2026-10-04T13:30:15.123456Z"
    assert format_timestamp(AS_OF + timedelta(microseconds=1)) > format_timestamp(AS_OF)
    # Whole seconds keep their fractional digits, which plain isoformat() would drop and then mis-order.
    whole_second = datetime(2026, 10, 4, 13, 30, 16, tzinfo=UTC)
    assert format_timestamp(whole_second) == "2026-10-04T13:30:16.000000Z"
    assert format_timestamp(whole_second) > format_timestamp(AS_OF)
    with pytest.raises(ValueError, match="timezone-aware"):
        format_timestamp(datetime(2026, 1, 1))


def test_json_mode_dump_matches_the_database_encoding() -> None:
    proposal = ProposalRow(
        league_id=1, kind="lineup", policy="approve", payload={"moves": [1, 2]}, created_by="t", created_at=AS_OF
    )
    dumped = proposal.model_dump(mode="json")
    assert dumped["created_at"] == "2026-10-04T13:30:15.123456Z"
    assert dumped["payload"] == {"moves": [1, 2]}
    assert dumped["deadline"] is None
    assert dumped["status"] == "proposed"


def test_json_columns_parse_database_text() -> None:
    proposal = ProposalRow.model_validate(
        {
            "id": 7,
            "league_id": 1,
            "kind": "lineup",
            "status": "approved",
            "policy": "approve",
            "scoring_period_id": None,
            "payload": '{"moves":[1,2]}',
            "engine_numbers": "{}",
            "rationale": None,
            "deadline": None,
            "created_by": "t",
            "created_at": "2026-10-04T13:30:15.123456Z",
            "decided_by": None,
            "decided_at": None,
            "execution_token": None,
            "token_consumed_at": None,
            "dedupe_key": None,
        }
    )
    assert proposal.payload == {"moves": [1, 2]}
    assert proposal.engine_numbers == {}
    assert proposal.created_at == AS_OF
    projection = ProjectionRow.model_validate(
        {
            "sport": "nfl",
            "espn_id": 1,
            "source": "espn",
            "season": 2026,
            "scoring_period_id": 4,
            "stats": '{"pass_yd":250,"pass_td":2}',
            "as_of": AS_OF,
        }
    )
    assert projection.stats == {"pass_yd": 250.0, "pass_td": 2.0}


def test_probabilities_are_bounded() -> None:
    with pytest.raises(ValidationError):
        AvailabilityRow(sport="nfl", espn_id=1, season=2026, scoring_period_id=4, p_active=1.2, as_of=AS_OF)
    with pytest.raises(ValidationError):
        NewsSignalRow(
            news_item_id=1,
            sport="nfl",
            espn_id=1,
            kind="injury",
            severity="low",
            confidence=-0.1,
            published_at=AS_OF,
            created_at=AS_OF,
        )


def test_literal_columns_are_checked() -> None:
    with pytest.raises(ValidationError):
        league(sport="nhl")
    proposal: dict[str, object] = {
        "league_id": 1,
        "kind": "lineup",
        "policy": "auto",
        "payload": {},
        "created_by": "t",
        "created_at": AS_OF,
    }
    with pytest.raises(ValidationError):
        ProposalRow.model_validate({**proposal, "policy": "maybe"})
    with pytest.raises(ValidationError):
        ProposalRow.model_validate({**proposal, "status": "done"})


def test_rows_are_frozen_and_reject_unknown_fields() -> None:
    row = league()
    with pytest.raises(ValidationError):
        setattr(row, "key", "nba")  # noqa: B010 - the point is that assignment is refused
    assert row.model_copy(update={"key": "nba"}).key == "nba"
    with pytest.raises(ValidationError):
        league(bogus=1)


def test_row_id_requires_a_stored_row() -> None:
    assert league(id=3).row_id == 3
    with pytest.raises(ValueError, match="no id yet"):
        _ = league().row_id


def test_utc_now_is_aware() -> None:
    assert utc_now().tzinfo == UTC
