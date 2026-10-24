"""The stored trade pitch (ROADMAP #61): the template matches the advisor's word for word, and the shared module that
the push, the dashboard and ``fm proposals`` read never loads the advisor or the Anthropic SDK."""

from __future__ import annotations

import subprocess
import sys

import pytest

from fm.advisor.pitch import template_pitch as advisor_template
from fm.proposals.payloads import TradePayload
from fm.render.proposals import PITCH_KEY, StoredPitch, pitch_facts, plain_pitch, with_pitch

NAMES = {111: "Alpha Back", 222: "Beta Back", 333: "Gamma Tight"}


@pytest.mark.parametrize(
    ("payload", "names", "their_name", "our_name"),
    [
        (TradePayload(other_team_id=4, give_espn_ids=(111,), get_espn_ids=(222,)), NAMES, "Sam", "Jon"),
        (TradePayload(other_team_id=4, give_espn_ids=(111, 333), get_espn_ids=(222,)), NAMES, None, "Jon"),
        (TradePayload(other_team_id=4, give_espn_ids=(111, 333, 999), get_espn_ids=(222,)), None, "Sam", None),
    ],
)
def test_the_template_is_the_advisors_word_for_word(
    payload: TradePayload, names: dict[int, str] | None, their_name: str | None, our_name: str | None
) -> None:
    ours = plain_pitch(payload, names=names, their_name=their_name, our_name=our_name)
    assert ours == advisor_template(payload, names=names, their_name=their_name, our_name=our_name)


def test_with_pitch_only_adds_the_pitch_key() -> None:
    numbers = {"score": 0.4, "theirs": {"delta_ros": 1.0}}
    stored = with_pitch(numbers, StoredPitch("Hey,", "template", "advisor off"))
    assert stored == {**numbers, PITCH_KEY: {"text": "Hey,", "source": "template", "detail": "advisor off"}}
    assert PITCH_KEY not in numbers  # the input is not changed


def test_pitch_facts_keep_only_what_favors_the_other_side() -> None:
    acceptance = {
        "need_gain": 0.4,
        "market_receive": 5200.0,
        "market_send": 4800.0,
        "surplus": 0.08,
        "rank_type": "redraft",
    }
    numbers = {
        "unit": "points",
        "theirs": {"team_id": 2, "delta_ros": 3.4, "delta_title": 0.01},
        "acceptance": acceptance,
    }
    assert pitch_facts(numbers) == (
        "Their rest-of-season lineup value improves by 3.4 points.",
        "The players they receive fill a need in their lineup (+0.40 starter seasons).",
        "By redraft market value they receive 5,200 for 4,800.",
    )
    unfavourable = {"need_gain": 0.0, "market_receive": 1, "market_send": 2, "surplus": -0.5}
    assert pitch_facts({"theirs": {"delta_ros": -1.0}, "acceptance": unfavourable}) == ()
    assert pitch_facts({}) == () and pitch_facts({"theirs": "junk", "acceptance": {"need_gain": True}}) == ()


def test_the_pitch_readers_never_load_the_sdk() -> None:
    """The push, the dashboard and the CLI read the pitch, and a keyless tick writes the template, without the SDK."""
    code = (
        "import sys, fm.render.proposals, fm.notify.messages, fm.commands.proposals, fm.jobs.tick; "
        "print([n for n in sorted(sys.modules) if n.startswith(('anthropic', 'fm.advisor'))])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]", done.stdout
