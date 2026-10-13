/*
 * The espn-fantasy dashboard (fm ui). Vanilla JS, no build step, no CDN: it talks to this origin's JSON API only.
 *
 * Every value from the API reaches the page through textContent or an attribute setter; nothing is ever parsed as
 * HTML. POSTs carry the per-process token from <meta name="fm-token"> as X-FM-Token. Approving records a decision;
 * fm run or the scheduled tick executes it.
 */
"use strict";

(() => {
  const TOKEN = document.querySelector('meta[name="fm-token"]')?.getAttribute("content") ?? "";
  const SECTIONS = ["overview", "proposals", "lineup", "waivers", "reports", "settings", "health"];
  const AUTO_REFRESH = new Set(["overview", "proposals", "health"]);
  const REFRESH_MS = 60_000;
  const KIND_LABELS = {
    lineup_lock: "Lineup lock",
    roster_lock: "Roster lock",
    waiver_run: "Waiver run",
    period_turn: "Period turns",
    trade_deadline: "Trade deadline",
  };
  const CHANNEL_LABELS = {
    telegram: "Telegram",
    ntfy: "ntfy (no account needed)",
    none: "None: decide here or with fm proposals",
  };
  const state = { current: "overview", leagues: [], overview: null, report: null };

  // --- DOM helpers ---------------------------------------------------------------------------------------------------

  const $ = (selector) => document.querySelector(selector);

  /** An element with props (class, text, on<event>, boolean properties, attributes) and children (nodes or text). */
  function h(tag, props, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props ?? {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = String(value);
      else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
      else if (typeof value === "boolean" && key in node) node[key] = value;
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    return append(node, children);
  }

  function append(node, children) {
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function clear(node) {
    node.replaceChildren();
    return node;
  }

  const sectionBody = (name) => document.querySelector(`[data-section="${name}"] .body`);
  const dt = (text) => h("dt", { text });
  const dd = (...children) => h("dd", {}, ...children);
  const td = (...children) => h("td", {}, ...children);
  const tdNum = (text) => h("td", { class: "num", text });
  const empty = (text) => h("p", { class: "empty", text });

  function table(headers, rows, numeric = []) {
    const head = h("tr", {}, headers.map((name, i) => h("th", { scope: "col", class: numeric.includes(i) ? "num" : null, text: name })));
    return h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, head), h("tbody", {}, rows)));
  }

  function badge(status, text) {
    return h("span", { class: `badge ${status}`, text: text ?? status });
  }

  function notes(lines) {
    return lines && lines.length ? h("ul", { class: "notes" }, lines.map((line) => h("li", { text: line }))) : null;
  }

  // --- formatting ----------------------------------------------------------------------------------------------------

  const dateTime = new Intl.DateTimeFormat(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
  const clock = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });

  function when(iso) {
    if (!iso) return "none";
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? iso : dateTime.format(date);
  }

  function span(minutes) {
    if (minutes < 1) return "under a minute";
    if (minutes < 60) return `${minutes} min`;
    if (minutes < 48 * 60) return `${Math.round(minutes / 60)} h`;
    return `${Math.round(minutes / 1440)} days`;
  }

  function relative(iso) {
    const ms = new Date(iso).getTime() - Date.now();
    if (Number.isNaN(ms)) return "";
    const text = span(Math.round(Math.abs(ms) / 60_000));
    return ms >= 0 ? `in ${text}` : `${text} ago`;
  }

  function ago(seconds) {
    return seconds === null || seconds === undefined ? "" : `${span(Math.round(seconds / 60))} ago`;
  }

  function num(value, digits = 1) {
    return value === null || value === undefined ? "–" : Number(value).toFixed(digits);
  }

  function pct(value) {
    return value === null || value === undefined ? "–" : `${Math.round(Number(value) * 100)}%`;
  }

  function person(player) {
    if (!player) return "nobody";
    return player.position ? `${player.name} (${player.position})` : player.name;
  }

  function presetName(name) {
    return name ? name.replace("_", " ") : "no preset";
  }

  function record(team) {
    return `${team.wins}-${team.losses}${team.ties ? `-${team.ties}` : ""}`;
  }

  function verdict(policy) {
    if (!policy) return "";
    return policy.allowed ? `allowed (${policy.setting})` : `blocked: ${policy.reasons.join("; ")}`;
  }

  function statusClass(status) {
    return { proposed: "warn", approved: "ok", executing: "ok", verified: "ok", failed: "fail" }[status] ?? "";
  }

  // --- talking to the API --------------------------------------------------------------------------------------------

  async function api(path, { method = "GET", body } = {}) {
    const init = { method, headers: { Accept: "application/json" }, cache: "no-store", credentials: "same-origin" };
    if (method !== "GET") {
      init.headers["X-FM-Token"] = TOKEN;
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body ?? {});
    }
    let response;
    try {
      response = await fetch(path, init);
    } catch {
      throw new Error("The dashboard is not answering. Is fm ui or fm run still running?");
    }
    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    if (!response.ok) throw new Error(data?.error ?? `${response.status} ${response.statusText}`);
    return data;
  }

  let toastTimer = 0;
  function toast(message, bad = false) {
    const node = $("#toast");
    node.textContent = message;
    node.classList.toggle("bad", bad);
    node.hidden = false;
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      node.hidden = true;
    }, bad ? 8000 : 4000);
  }

  function showError(container, error) {
    clear(container).append(h("p", { class: "error", text: error.message ?? String(error) }));
  }

  function setTopStatus(text, bad = false) {
    const node = $("#top-status");
    node.textContent = text;
    node.classList.toggle("error", bad);
  }

  // --- sections ------------------------------------------------------------------------------------------------------

  function currentSection() {
    const name = window.location.hash.replace(/^#/, "");
    return SECTIONS.includes(name) ? name : "overview";
  }

  function show(name) {
    state.current = name;
    for (const section of document.querySelectorAll("[data-section]")) section.hidden = section.dataset.section !== name;
    for (const tab of document.querySelectorAll("[data-tab]")) {
      if (tab.dataset.tab === name) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    }
    return load(name);
  }

  const loaders = {
    overview: loadOverview,
    proposals: loadProposals,
    lineup: (container) => loadPlan(container, "lineup", lineupBlock),
    waivers: (container) => loadPlan(container, "waivers", waiverBlock),
    reports: loadReports,
    settings: loadSettings,
    health: loadHealth,
  };

  async function load(name) {
    const container = sectionBody(name);
    if (!container.hasChildNodes()) container.append(h("p", { class: "muted", text: "Loading…" }));
    try {
      await loaders[name](container);
    } catch (error) {
      showError(container, error);
    }
  }

  async function refreshAll() {
    if (state.current !== "overview") {
      try {
        await fetchOverview();
      } catch (error) {
        setTopStatus(error.message, true);
      }
    }
    await load(state.current);
  }

  // --- overview ------------------------------------------------------------------------------------------------------

  async function fetchOverview() {
    const data = await api("/api/overview");
    state.overview = data;
    state.leagues = data.leagues
      .filter((league) => league.synced)
      .map((league) => ({ key: league.league, label: `${league.sport.toUpperCase()} · ${league.name ?? league.league}` }));
    fillLeagueSelects();
    setPause(data.paused);
    const open = data.leagues.reduce((total, league) => total + (league.open_proposals?.length ?? 0), 0);
    const count = $("#proposal-count");
    count.textContent = String(open);
    count.setAttribute("aria-label", `${open} open`);
    count.hidden = open === 0;
    const session = data.session && data.session.status !== "ok" ? ` · ESPN session: ${data.session.status}` : "";
    setTopStatus(`Updated ${clock.format(new Date())}${session}`);
    return data;
  }

  async function loadOverview(container) {
    const data = await fetchOverview();
    clear(container);
    if (data.config_error) {
      container.append(
        h("p", { class: "error", text: data.config_error }),
        h("p", {}, "Run ", h("code", { text: "fm setup" }), " or ", h("code", { text: "fm config check" }), "."),
      );
    }
    container.append(machineCard(data));
    const grid = h("div", { class: "grid" }, data.leagues.map(leagueCard));
    if (!data.leagues.length && !data.config_error) grid.append(empty("No leagues configured."));
    container.append(grid);
  }

  function machineCard(data) {
    const card = h("div", { class: "card" }, h("div", { class: "card-head" }, h("h3", { text: "This machine" })));
    const list = h("dl", { class: "kv" });
    if (data.session) {
      const fix = data.session.fix ? [" Fix: ", h("code", { text: data.session.fix })] : null;
      list.append(dt("ESPN session"), dd(badge(data.session.status), " ", data.session.detail, fix));
    }
    const { fail, warn } = data.health;
    const summary = fail || warn ? `${fail} failing, ${warn} to look at` : "every check passes";
    list.append(dt("Health"), dd(badge(fail ? "fail" : warn ? "warn" : "ok"), ` ${summary} `, h("a", { href: "#health", text: "details" })));
    list.append(dt("Automation"), dd(data.paused.on ? badge("warn", "paused") : badge("ok", "running")));
    card.append(list);
    if (!data.paused.on) {
      const reason = h("input", { type: "text", id: "pause-reason", maxlength: "200", placeholder: "Why (optional)" });
      card.append(
        h("div", { class: "field" }, h("label", { for: "pause-reason", text: "Pause reason" }), reason),
        h("div", { class: "actions" }, h("button", { type: "button", class: "danger", onclick: () => pauseAll(reason.value) }, "Pause everything")),
        h("p", { class: "muted small", text: "Pausing stops new proposals, automatic approvals and every execution until you resume. Your own approvals are still recorded." }),
      );
    }
    return card;
  }

  function leagueCard(league) {
    const title = `${league.sport.toUpperCase()} · ${league.name ?? league.league}`;
    const card = h("article", { class: "card", "aria-label": title });
    card.append(
      h(
        "div",
        { class: "card-head" },
        h("h3", { text: title }),
        h("span", { class: "muted small", text: `${league.league} · ${league.season}` }),
        h("span", { class: "badge", text: presetName(league.preset) }),
      ),
    );
    if (!league.synced) {
      card.append(h("p", { class: "muted", text: league.note ?? "not synced; run fm sync" }));
      return card;
    }
    const facts = h("dl", { class: "kv" });
    const team = league.team;
    if (team) {
      facts.append(dt("Team"), dd(team.name), dt("Record"), dd(record(team)));
      if (league.standing?.playoff_seed) facts.append(dt("Standing"), dd(`seed ${league.standing.playoff_seed} of ${league.standing.teams}`));
      facts.append(dt("Points"), dd(`${num(team.points_for)} for, ${num(team.points_against)} against`));
    }
    if (league.faab) facts.append(dt("FAAB"), dd(`$${league.faab.budget - league.faab.spent} of $${league.faab.budget} left`));
    facts.append(dt("Synced"), dd(`${when(league.synced_at)} (${ago(league.sync_age_s)})`));
    card.append(facts, h("h4", { text: "Next deadlines" }));
    if (league.deadlines.length) {
      card.append(
        h(
          "ul",
          { class: "plain" },
          league.deadlines.map((deadline) =>
            h(
              "li",
              {},
              h("strong", { text: KIND_LABELS[deadline.kind] ?? deadline.kind }),
              ` ${when(deadline.at)} (${relative(deadline.at)})`,
              deadline.period ? h("span", { class: "muted", text: ` · period ${deadline.period}` }) : null,
              deadline.provisional ? " (provisional)" : null,
            ),
          ),
        ),
      );
    } else {
      card.append(empty("None in the next 7 days."));
    }
    card.append(h("h4", { text: "Open proposals" }));
    if (league.open_proposals.length) {
      card.append(
        h(
          "ul",
          { class: "plain" },
          league.open_proposals.map((proposal) =>
            h("li", {}, h("a", { href: "#proposals", text: `#${proposal.id} ${proposal.label}` }), ` (${proposal.status}): ${proposal.moves}`),
          ),
        ),
      );
    } else {
      card.append(empty("Nothing waiting."));
    }
    card.append(notes(league.notes));
    return card;
  }

  // --- the kill switch -----------------------------------------------------------------------------------------------

  function setPause(paused) {
    $("#pause-banner").hidden = !paused.on;
    $("#pause-detail").textContent = paused.on ? `${paused.detail}.` : "";
  }

  async function pauseAll(reason) {
    if (!window.confirm("Pause everything? No proposal is made, approved automatically or executed until you resume.")) return;
    try {
      const result = await api("/api/pause", { method: "POST", body: { reason: reason.trim() || null } });
      setPause(result.paused);
      toast("Paused.");
    } catch (error) {
      toast(error.message, true);
    }
    await refreshAll();
  }

  async function resumeAll() {
    try {
      const result = await api("/api/resume", { method: "POST" });
      setPause(result.paused);
      toast("Resumed.");
    } catch (error) {
      toast(error.message, true);
    }
    await refreshAll();
  }

  // --- proposals -----------------------------------------------------------------------------------------------------

  async function loadProposals(container) {
    const all = $("#proposals-all").checked;
    const data = await api(`/api/proposals?all=${all ? 1 : 0}`);
    clear(container);
    if (!data.proposals.length) {
      container.append(empty(all ? "No proposals yet." : "Nothing is waiting for you."));
      return;
    }
    if (data.count > data.proposals.length) {
      container.append(h("p", { class: "muted", text: `The newest ${data.proposals.length} of ${data.count}.` }));
    }
    container.append(...data.proposals.map(proposalCard));
  }

  function policyText(proposal) {
    if (proposal.status !== "proposed") return proposal.policy;
    if (proposal.policy === "auto") return "auto: goes ahead on its own 15 min before the deadline unless you reject it";
    return `${proposal.policy}: nothing happens unless you approve`;
  }

  function proposalCard(proposal) {
    const titleId = `proposal-${proposal.id}`;
    const card = h("article", { class: "card", "aria-labelledby": titleId });
    card.append(
      h(
        "div",
        { class: "card-head" },
        h("h3", { id: titleId, text: `#${proposal.id} ${(proposal.league ?? "").toUpperCase()} ${proposal.label}` }),
        badge(statusClass(proposal.status), proposal.status),
      ),
      h("p", { text: proposal.moves }),
    );
    if (proposal.rationale) card.append(h("p", { class: "muted", text: proposal.rationale }));
    const facts = h("dl", { class: "kv small" });
    facts.append(dt("Deadline"), dd(proposal.deadline ? `${when(proposal.deadline)} (${relative(proposal.deadline)})` : "none"));
    facts.append(dt("Policy"), dd(policyText(proposal)));
    facts.append(dt("Proposed"), dd(`${when(proposal.created_at)} by ${proposal.created_by}`));
    if (proposal.decided_at) facts.append(dt("Decided"), dd(`${when(proposal.decided_at)} by ${proposal.decided_by}`));
    card.append(facts);
    const actions = h("div", { class: "actions" });
    if (proposal.can_approve) {
      actions.append(
        h("button", { type: "button", class: "primary", "aria-label": `Approve proposal ${proposal.id}`, onclick: (event) => decide(proposal, "approve", event.currentTarget) }, "Approve"),
      );
    }
    if (proposal.can_reject) {
      const label = proposal.status === "approved" ? "Withdraw approval" : "Reject";
      actions.append(
        h("button", { type: "button", class: "danger", "aria-label": `${label} for proposal ${proposal.id}`, onclick: (event) => decide(proposal, "reject", event.currentTarget) }, label),
      );
    }
    if (actions.hasChildNodes()) card.append(actions);
    return card;
  }

  async function decide(proposal, decision, button) {
    if (decision === "approve" && !window.confirm(`Approve #${proposal.id}?\n\n${proposal.moves}\n\nfm run executes it at its time.`)) return;
    button.disabled = true;
    try {
      const result = await api(`/api/proposals/${proposal.id}/${decision}`, { method: "POST" });
      toast(`#${proposal.id} ${result.proposal.status}.`);
    } catch (error) {
      toast(error.message, true);
    }
    await refreshAll();
  }

  // --- lineup and waivers --------------------------------------------------------------------------------------------

  function fillLeagueSelects() {
    for (const select of [$("#lineup-league"), $("#waivers-league")]) {
      const previous = select.value;
      select.replaceChildren(...state.leagues.map((league) => h("option", { value: league.key, text: league.label })));
      if (state.leagues.some((league) => league.key === previous)) select.value = previous;
    }
  }

  async function loadPlan(container, kind, render) {
    if (!state.overview) await fetchOverview();
    const key = $(`#${kind}-league`).value;
    if (!key) {
      clear(container).append(empty("No synced league yet: run fm sync."));
      return;
    }
    clear(container).append(h("p", { class: "muted", text: "Planning…" }));
    const data = await api(`/api/${kind}/${encodeURIComponent(key)}`);
    clear(container).append(...data.leagues.map(render));
  }

  function planText(plan) {
    const odds = plan.win_probability === null || plan.win_probability === undefined ? "" : `, ${pct(plan.win_probability)} to win`;
    return `${num(plan.expected_points)} expected points${odds}`;
  }

  function lineupBlock(block) {
    const card = h("div", { class: "card" });
    if (!block.ok) return append(card, [h("p", { class: "error", text: block.error })]);
    card.append(
      h("div", { class: "card-head" }, h("h3", { text: `${block.team} · scoring period ${block.scoring_period}` }), h("span", { class: "muted small", text: `as of ${when(block.as_of)}` })),
    );
    const facts = h("dl", { class: "kv" }, dt("Current"), dd(planText(block.current)), dt("Recommended"), dd(planText(block.recommended)));
    if (block.opponent) facts.append(dt("Opponent"), dd(`${block.opponent.name ?? "unknown"}: ${num(block.opponent.projected)} projected`));
    card.append(facts, h("h4", { text: "Moves" }));
    card.append(
      block.moves.length
        ? h("ul", { class: "plain" }, block.moves.map((move) => h("li", { text: `${move.name}: ${move.from_slot} → ${move.to_slot}` })))
        : empty("The current lineup is already the best one."),
    );
    const rows = block.lineup.map((player) => {
      const flags = [player.moved ? `from ${player.from_slot}` : null, player.locked ? "locked" : null, player.designation, player.has_game ? null : "no game"];
      return h(
        "tr",
        { class: player.moved ? "moved" : null },
        td(player.slot),
        td(player.name),
        td(player.position ?? ""),
        tdNum(num(player.projection)),
        tdNum(pct(player.p_active)),
        tdNum(num(player.expected)),
        td(flags.filter(Boolean).join(", ")),
      );
    });
    card.append(table(["Slot", "Player", "Pos", "Proj", "Active", "Exp", "Notes"], rows, [3, 4, 5]));
    if (block.suggested_proposals.length) {
      card.append(
        h("h4", { text: "What the tick would propose" }),
        h("ul", { class: "plain" }, block.suggested_proposals.map((draft) => h("li", { text: `${draft.kind}: ${verdict(draft.policy)}${draft.rationale ? `. ${draft.rationale}` : ""}` }))),
      );
    }
    card.append(notes(block.warnings));
    return card;
  }

  function waiverBlock(block) {
    const card = h("div", { class: "card" });
    if (!block.ok) return append(card, [h("p", { class: "error", text: block.error })]);
    card.append(
      h(
        "div",
        { class: "card-head" },
        h("h3", { text: `${block.league} · scoring period ${block.scoring_period}` }),
        h("span", { class: "muted small", text: `${block.wire.size} free agents, synced ${when(block.wire.synced_at)}` }),
      ),
      h("h4", { text: "Moves the engine would make" }),
    );
    if (block.moves.length) {
      card.append(
        h(
          "ul",
          { class: "plain" },
          block.moves.map((move) =>
            h(
              "li",
              {},
              h("strong", { text: `${move.kind}: add ${person(move.add)}` }),
              move.drop ? `, drop ${person(move.drop)}` : "",
              move.bid !== null && move.bid !== undefined ? `, bid $${move.bid}` : "",
              ` (gain ${num(move.gain, 2)}; ${verdict(move.policy)})`,
              move.rationale ? h("div", { class: "muted small", text: move.rationale }) : null,
            ),
          ),
        ),
      );
    } else {
      card.append(empty("None: nobody on the wire beats your roster by enough."));
    }
    card.append(h("h4", { text: `Best available (top ${block.ranked.length} of ${block.ranked_total})` }));
    const rows = block.ranked.map((move) =>
      h("tr", {}, td(person(move.add)), td(person(move.drop)), tdNum(num(move.gain, 2)), tdNum(num(move.add_value)), tdNum(num(move.drop_value))),
    );
    card.append(rows.length ? table(["Add", "Drop", "Gain", "Add value", "Drop value"], rows, [2, 3, 4]) : empty("Nothing ranked."));
    if (block.replacement_levels.length) {
      card.append(
        h("h4", { text: "Replacement level" }),
        h("p", { class: "small muted", text: block.replacement_levels.map((level) => `${level.slot}: ${level.player ?? "–"} (${num(level.value)})`).join(" · ") }),
      );
    }
    card.append(notes(block.warnings));
    return card;
  }

  // --- reports -------------------------------------------------------------------------------------------------------

  async function loadReports(container) {
    const data = await api("/api/reports");
    clear(container);
    const view = $("#report-view");
    if (!data.reports.length) {
      clear(view);
      container.append(h("p", { class: "empty" }, "No reports yet. ", h("code", { text: "fm report" }), " writes one each week."));
      return;
    }
    if (!data.reports.some((report) => report.name === state.report)) state.report = data.reports[0].name;
    const list = h("div", { class: "report-list", "aria-label": "Reports" });
    for (const report of data.reports) {
      list.append(
        h("button", { type: "button", "data-report": report.name, "aria-pressed": String(report.name === state.report), onclick: () => openReport(report.name) }, report.name.replace(/\.md$/, "")),
      );
    }
    container.append(list);
    await openReport(state.report);
  }

  async function openReport(name) {
    state.report = name;
    for (const button of document.querySelectorAll("[data-report]")) button.setAttribute("aria-pressed", String(button.dataset.report === name));
    const view = $("#report-view");
    try {
      const data = await api(`/api/reports/${encodeURIComponent(name)}`);
      renderMarkdown(data.markdown, clear(view));
    } catch (error) {
      showError(view, error);
    }
  }

  /** A deliberately small markdown reader for fm report's output: headings, lists, tables and paragraphs, as text. */
  function renderMarkdown(text, root) {
    let paragraph = [];
    let list = null;
    let rows = null;
    const plain = (value) => value.replace(/\*\*(.+?)\*\*/g, "$1").replace(/`([^`]+)`/g, "$1");
    const flush = () => {
      if (paragraph.length) root.append(h("p", { text: plain(paragraph.join(" ")) }));
      if (list) root.append(list);
      if (rows && rows.length) root.append(table(rows[0], rows.slice(1).map((cells) => h("tr", {}, cells.map((cell) => td(cell))))));
      paragraph = [];
      list = null;
      rows = null;
    };
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trimEnd();
      const heading = /^(#{1,6})\s+(.*)$/.exec(line);
      const item = /^\s*[-*]\s+(.*)$/.exec(line);
      if (heading) {
        flush();
        root.append(h(`h${Math.min(heading[1].length + 2, 6)}`, { text: plain(heading[2]) }));
      } else if (item) {
        if (!list) {
          flush();
          list = h("ul", { class: "plain" });
        }
        list.append(h("li", { text: plain(item[1]) }));
      } else if (/^\s*\|.*\|$/.test(line)) {
        if (!rows) {
          flush();
          rows = [];
        }
        if (!/^\s*\|[\s:|-]+\|$/.test(line)) rows.push(line.trim().slice(1, -1).split("|").map((cell) => plain(cell.trim())));
      } else if (!line.trim()) {
        flush();
      } else {
        if (list || rows) flush();
        paragraph.push(line.trim());
      }
    }
    flush();
  }

  // --- settings ------------------------------------------------------------------------------------------------------

  async function loadSettings(container) {
    renderSettings(container, await api("/api/settings"));
  }

  function renderSettings(container, data) {
    clear(container);
    if (data.config_error) {
      container.append(h("p", { class: "error", text: data.config_error }));
      return;
    }
    container.append(...data.leagues.map((league) => leagueSettings(league, data.presets)), notifySettings(data), secretsCard(data));
  }

  async function saveSettings(changes, button) {
    button.disabled = true;
    try {
      const data = await api("/api/settings", { method: "POST", body: changes });
      renderSettings(sectionBody("settings"), data);
      toast("Saved to config.toml.");
      fetchOverview().catch(() => {});
    } catch (error) {
      button.disabled = false;
      toast(error.message, true);
    }
  }

  function leagueSettings(league, presets) {
    const id = `settings-${league.league}`;
    const form = h("form", { class: "card", "aria-labelledby": `${id}-title`, novalidate: true });
    form.append(h("h3", { id: `${id}-title`, text: `${league.sport.toUpperCase()} · ${league.league} (${league.season})` }));
    const select = h(
      "select",
      { id: `${id}-preset` },
      h("option", { value: "", text: "No preset" }),
      presets.map((preset) => h("option", { value: preset.name, text: presetName(preset.name) })),
    );
    select.value = league.preset ?? "";
    const description = h("p", { class: "muted small" });
    const describe = () => {
      const chosen = presets.find((preset) => preset.name === select.value);
      description.textContent = chosen ? chosen.description : "Every policy key comes from config.toml or the defaults.";
    };
    select.addEventListener("change", describe);
    describe();
    form.append(h("div", { class: "field" }, h("label", { for: select.id, text: "Preset" }), select), description);
    if (league.overrides.length) {
      form.append(
        h("p", { class: "small" }, "Written in config.toml and overriding the preset: ", league.overrides.map((key, i) => [i ? ", " : "", h("code", { text: key })])),
      );
    }
    const reset = h("input", { type: "checkbox", id: `${id}-reset` });
    const finder = h("input", { type: "checkbox", id: `${id}-finder`, checked: league.trade_finder_enabled });
    form.append(
      h("label", { class: "check", for: reset.id }, reset, "Reset overrides: let the preset decide every key it covers"),
      h("label", { class: "check", for: finder.id }, finder, "Trade finder drafts offers for your approval (it never sends one)"),
    );
    const save = h("button", { type: "submit", class: "primary" }, "Save");
    form.append(h("div", { class: "actions" }, save));
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const changes = { league: league.league };
      if ((select.value || null) !== (league.preset ?? null)) changes.preset = select.value || null;
      if (finder.checked !== league.trade_finder_enabled) changes.trade_finder_enabled = finder.checked;
      if (reset.checked) changes.reset_overrides = true;
      if (Object.keys(changes).length === 1) {
        toast("Nothing changed.");
        return;
      }
      saveSettings(changes, save);
    });
    return form;
  }

  function notifySettings(data) {
    const form = h("form", { class: "card", "aria-labelledby": "notify-title", novalidate: true });
    const group = h("fieldset", {}, h("legend", { text: "Where proposals and alerts go" }));
    for (const channel of data.channels) {
      const input = h("input", { type: "radio", name: "notify-channel", id: `channel-${channel}`, value: channel, checked: channel === data.notify.channel });
      group.append(h("label", { class: "check", for: input.id }, input, CHANNEL_LABELS[channel] ?? channel));
    }
    const save = h("button", { type: "submit", class: "primary" }, "Save");
    form.append(
      h("h3", { id: "notify-title", text: "Phone" }),
      group,
      h("p", { class: "muted small" }, "A channel needs its keys in .env: run ", h("code", { text: "fm notify setup" }), " after switching."),
      h("div", { class: "actions" }, save),
    );
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const chosen = form.querySelector('input[name="notify-channel"]:checked')?.value;
      if (!chosen || chosen === data.notify.channel) {
        toast("Nothing changed.");
        return;
      }
      saveSettings({ notify_channel: chosen }, save);
    });
    return form;
  }

  function secretsCard(data) {
    const rows = Object.entries(data.secrets).map(([name, isSet]) => {
      const missing = data.missing_secrets.includes(name);
      return h("tr", {}, td(h("code", { text: name })), td(isSet ? badge("ok", "set") : badge(missing ? "fail" : "", missing ? "missing" : "not set")));
    });
    return h(
      "div",
      { class: "card" },
      h("h3", { text: "Keys in .env" }),
      h("p", { class: "muted small", text: "Only whether each key is set is shown; the values stay in the file." }),
      table(["Variable", "State"], rows),
      h("p", { class: "muted small" }, "Config file: ", h("code", { text: data.config_path })),
    );
  }

  // --- health --------------------------------------------------------------------------------------------------------

  async function loadHealth(container) {
    const data = await api("/api/health");
    const rows = data.checks.map((check) =>
      h("tr", {}, td(badge(check.status)), td(check.name), td(check.detail, check.fix ? [h("br"), "Fix: ", h("code", { text: check.fix })] : null)),
    );
    clear(container).append(
      h("p", {}, data.has_failure ? badge("fail", "something is failing") : badge("ok", "nothing is failing"), ` Checked ${when(data.as_of)}.`),
      table(["Status", "Check", "Detail"], rows),
    );
  }

  // --- start ---------------------------------------------------------------------------------------------------------

  document.addEventListener("DOMContentLoaded", () => {
    $("#refresh").addEventListener("click", () => refreshAll());
    $("#resume").addEventListener("click", () => resumeAll());
    $("#proposals-all").addEventListener("change", () => load("proposals"));
    $("#lineup-league").addEventListener("change", () => load("lineup"));
    $("#waivers-league").addEventListener("change", () => load("waivers"));
    window.addEventListener("hashchange", () => show(currentSection()));
    window.setInterval(() => {
      if (document.hidden) return;
      if (AUTO_REFRESH.has(state.current)) load(state.current);
      else fetchOverview().catch((error) => setTopStatus(error.message, true));
    }, REFRESH_MS);
    const first = currentSection();
    const ready = first === "overview" ? Promise.resolve() : fetchOverview().catch((error) => setTopStatus(error.message, true));
    ready.then(() => show(first));
  });
})();
