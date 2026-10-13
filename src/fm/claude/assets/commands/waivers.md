---
description: Waiver wire and free-agent pickups (add, drop, FAAB bid) for my ESPN fantasy leagues
argument-hint: "[league key]"
---

Rank my waiver options with the espn-fantasy MCP tools, following the espn-fantasy skill.

League (empty means every league): $ARGUMENTS

Call `status` first, then `waivers`. Show the top (add, drop) pairs with each one's rest-of-season value gain, the
suggested FAAB bid against my remaining budget, and the moves the engine would make. Mention notable players it
skipped and why. If I ask you to queue a claim, use `create_proposal` with that move's payload; it waits for my
approval and changes nothing by itself.
