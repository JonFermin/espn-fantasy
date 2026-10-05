"""The channel interface helpers (``fm.notify.base``): the button encoding, text splitting, and secret redaction."""

from __future__ import annotations

import logging

import pytest

from fm.notify.base import (
    BUTTONS,
    DECISIONS,
    Callback,
    Decision,
    DecisionResult,
    Message,
    ProposalNotice,
    decode_callback,
    encode_callback,
    hide_in_logs,
    redact,
    split_text,
)

NONCE = "AbCdEfGhIjKlMnOpQrStUv_-"


class TestCallbacks:
    @pytest.mark.parametrize("decision", DECISIONS)
    def test_round_trip(self, decision: Decision) -> None:
        text = encode_callback(decision, 12, NONCE)
        assert text == f"{decision}:12:{NONCE}"
        assert decode_callback(text) == Callback(decision, 12, NONCE)

    def test_the_longest_callback_fits_telegram_callback_data(self) -> None:
        assert len(encode_callback("approve", 999_999_999_999, NONCE).encode()) <= 64

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        assert decode_callback(f"  reject:3:{NONCE}\n") == Callback("reject", 3, NONCE)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "approve",
            f"approve:12:{NONCE}:extra",
            f"accept:12:{NONCE}",
            f"APPROVE:12:{NONCE}",
            f"approve:0:{NONCE}",
            f"approve:012:{NONCE}",
            f"approve:-1:{NONCE}",
            f"approve:1234567890123:{NONCE}",
            f"approve:12:{NONCE[:-1]}",
            f"approve:12:{NONCE}x",
            f"approve:12:{NONCE[:-1]}!",
            f"approve :12:{NONCE}",
            '{"decision": "approve"}',
        ],
    )
    def test_anything_else_decodes_to_none(self, text: str) -> None:
        assert decode_callback(text) is None

    @pytest.mark.parametrize(("proposal_id", "nonce"), [(0, NONCE), (-3, NONCE), (1, "short"), (1, NONCE + "x")])
    def test_encode_refuses_what_decode_would_not_accept(self, proposal_id: int, nonce: str) -> None:
        with pytest.raises(ValueError):
            encode_callback("approve", proposal_id, nonce)

    def test_a_notice_encodes_both_buttons_with_its_one_nonce(self) -> None:
        notice = ProposalNotice(5, NONCE, Message("nfl: lineup change #5"))
        assert [notice.callback(decision) for _, decision in BUTTONS] == [f"approve:5:{NONCE}", f"reject:5:{NONCE}"]
        assert [label for label, _ in BUTTONS] == ["Approve", "Reject"]


class TestMessages:
    def test_text_joins_title_and_trimmed_body(self) -> None:
        assert Message("Title", "  body line \n\n").text == "Title\nbody line"
        assert Message("Title").text == "Title"
        assert Message("Title", "   ").text == "Title"

    def test_decision_results_describe_themselves(self) -> None:
        approved = DecisionResult(4, "approve", ok=True, detail="nfl lineup change: A: BE -> QB")
        refused = DecisionResult(4, "approve", ok=False, detail="cannot approve proposal #4: it is expired")
        rejected = DecisionResult(9, "reject", ok=True, detail="nba free-agent add/drop: add B")
        assert approved.describe() == "#4 approved: nfl lineup change: A: BE -> QB"
        assert refused.headline == "#4 not approved"
        assert rejected.headline == "#9 rejected"


class TestSplitText:
    def test_short_text_is_one_piece(self) -> None:
        assert split_text("one\ntwo\n") == ["one\ntwo"]
        assert split_text("") == [""]

    def test_long_text_is_cut_at_line_breaks_within_the_limit(self) -> None:
        lines = [f"line {index:02d} " + "x" * 20 for index in range(10)]
        pieces = split_text("\n".join(lines), limit=100)
        assert all(len(piece.encode()) <= 100 for piece in pieces)
        assert "\n".join(pieces) == "\n".join(lines)  # nothing lost, every cut on a line break
        assert len(pieces) == 4

    def test_the_limit_counts_utf8_bytes_and_never_splits_a_character(self) -> None:
        text = "Nikola Jokić " * 40  # no line breaks: hard cuts
        pieces = split_text(text, limit=50)
        assert all(len(piece.encode()) <= 50 for piece in pieces)
        assert "".join(pieces).replace(" ", "") == text.replace(" ", "")
        assert all("�" not in piece for piece in pieces)

    def test_a_limit_below_one_character_is_refused(self) -> None:
        with pytest.raises(ValueError):
            split_text("abc", limit=3)


def test_hidden_secrets_never_reach_httpx_log_lines(caplog: pytest.LogCaptureFixture) -> None:
    secret = "987654:SECRET-redaction-test"
    hide_in_logs(secret)
    caplog.set_level(logging.INFO, logger="httpx")
    url = f"https://api.telegram.org/bot{secret}/getUpdates"
    logging.getLogger("httpx").info('HTTP Request: %s %s "%s"', "POST", url, "HTTP/1.1 200 OK")
    assert "SECRET" not in caplog.text
    assert "https://api.telegram.org/bot***/getUpdates" in caplog.text
    assert redact(f"error at {url}") == "error at https://api.telegram.org/bot***/getUpdates"
