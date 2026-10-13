---
description: Who to start this scoring period in my ESPN fantasy leagues
argument-hint: "[league key]"
---

Plan my lineup with the espn-fantasy MCP tools, following the espn-fantasy skill.

League (empty means every league): $ARGUMENTS

Call `status` first, then `lineup`. Show the recommended starters per slot, the moves from my current lineup, and the
expected points and win probability before and after. Point out locked players, players without a game, and anyone
flagged OUT or questionable. If I ask you to queue the changes, use `create_proposal` with the engine's suggested
proposal; it waits for my approval and changes nothing by itself.
