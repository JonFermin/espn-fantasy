---
name: espn-fantasy
description: Answer questions about the user's own ESPN fantasy football (NFL) and fantasy basketball (NBA) teams with the espn-fantasy MCP tools. Use it whenever the user asks about their fantasy team, lineup or start/sit choices, matchup, waiver wire or FAAB bids, a trade, standings, playoff odds, the weekly report, or whether their fantasy assistant (fm) is working.
---

# ESPN fantasy manager

The user runs `fm`, a personal assistant GM for their ESPN fantasy leagues. Its MCP server, `espn-fantasy`, answers
from a local store that `fm` keeps in sync with ESPN. The engine does the math (projections, expected points, win
probability, rest-of-season value, playoff and title odds, the chance a trade is accepted). Your job is to explain its
numbers and help the user decide. You cannot change a league, and you never pretend to.

## How to answer

1. **Start with `status`.** It lists the configured leagues and their keys (pass one as `league` to the other tools),
   when each was last synced, the record and FAAB, the open proposals and whether the kill switch is on. If a league
   is not synced or its sync is old, say so before giving advice.
2. **Use the tool that fits the question:**
   - `lineup`: who to start this scoring period. Recommended starters per slot, expected points, win probability and
     the moves from the current lineup.
   - `waivers`: ranked (add, drop) pairs by rest-of-season value, with the suggested FAAB bid and why candidates were
     skipped.
   - `trade_eval`: one deal written as "give A, B get C" (names or ESPN ids). Both sides' rest-of-season value and
     title odds, legality, the chance the other manager accepts, and a recommendation.
   - `report`: the latest weekly report per league (matchup outlook, playoff odds and their trend, moves made,
     upcoming deadlines).
   - `list_proposals`: what is queued and where each item stands.
   - `health`: whether the installation works (config, ESPN session, last tick, sync age, phone channel, deadlines).
     Use it when data looks stale, a tool returns an error, or the user asks whether things are running.
3. **Cite the engine's numbers.** Quote expected points, win probability, value, bids and odds as the tools return
   them, and name the tool they came from. Do not invent projections. When a tool says an input is missing (not
   synced, no pro schedule captured, no simulation), say so instead of guessing. Your own context (news, matchups,
   gut feel) is welcome, but label it as yours and keep the engine's numbers as the basis.

## Rules

- **Never claim a move was made.** Nothing you can call changes a league. `create_proposal` only stores a suggestion:
  the user approves or rejects it on their phone or in the dashboard (`fm ui`), and only then does `fm`'s executor
  make the move on ESPN and verify it. Say "queued for your approval", never "done" or "I dropped him".
- **Call `create_proposal` only when the user asks you to queue a move.** Build it from the payloads the read tools
  return (`lineup`'s `suggested_proposals`, a `waivers` move's `payload`, `trade_eval`'s `proposal_payload`): ESPN
  ids, never names. Report its result as it comes back: `proposed`, `existing`, `blocked` with every reason, or
  `invalid`. Do not try to get around a `blocked` result.
- **Trades are approval-only, always.** A trade proposal waits for the user's explicit decision; no setting changes
  that, so never suggest one does.
- **When `health` shows a problem, say so and give the fix it names.** `fm doctor` shows every check with its fix;
  `fm setup` repairs a missing or broken configuration; `fm login` renews an expired ESPN session; `fm sync` refreshes
  stale data. These are commands for the user to run in a terminal.
