---
description: Judge a fantasy trade with the engine (give A, B get C) and say whether to make it
argument-hint: "give <players> get <players> [in <league key>]"
---

Evaluate this trade with the espn-fantasy MCP tools, following the espn-fantasy skill:

$ARGUMENTS

1. Put the deal in the form "give A, B get C" (player names or ESPN ids). If it names no league and I have more than
   one, use `status` to find the league these players are in, or ask me.
2. Call `trade_eval`. Explain its recommendation, both sides' change in rest-of-season value and title odds, any
   legality problems, and the chance the other manager accepts.
3. If the engine says counter, suggest a counter and evaluate it with `trade_eval` too.

Trades are approval-only. Only if I ask you to queue the offer, call `create_proposal` with kind `trade_propose` and
the `proposal_payload` that `trade_eval` returned; the offer then waits for my approval on the phone or in `fm ui`, and
nothing is sent to the other manager before that.
