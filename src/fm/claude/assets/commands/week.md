---
description: Weekly fantasy briefing for my ESPN leagues (status, lineup, waivers, report)
argument-hint: "[league key]"
---

Give me my weekly fantasy briefing with the espn-fantasy MCP tools, following the espn-fantasy skill.

League (empty means every league): $ARGUMENTS

1. `status`: which leagues, how fresh each sync is, record, FAAB, open proposals, kill switch. If a sync is stale or
   missing, run `health` and tell me what to fix before going on.
2. `lineup`: the recommended lineup for this scoring period, the changes from my current one, expected points and win
   probability.
3. `waivers`: the best few (add, drop) pairs with their value and suggested bid.
4. `report`: the matchup outlook, playoff odds and trend, and the deadlines coming up.

End with a short checklist of the decisions I should make this week, each with the engine number behind it. Do not
queue anything unless I ask; nothing here changes my leagues.
