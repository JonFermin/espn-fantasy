// Derive the TRADE_PROPOSAL request by running ESPN's own web-client code offline (capture.py derive-trade).
//
// The saved excerpts in webclient.json are whole webpack modules: `types` (the constants), `model` (item builders
// and the serializer `get()`) and `service` (proposeTrade, createTransaction, saveTransaction). This script evaluates
// them with a stub `require` and calls `proposeTrade` the way the trade builder does. The module that would send the
// request (`a["_1"]`, the `post` excerpt) is replaced by a stub that records its arguments and returns a promise that
// never settles, so nothing after it runs. This process opens no socket: it has no cookies, no fetch, no http module.
//
//   node derive_trade.cjs <webclient.json>  < inputs.json  > derived.json
//
// inputs: {game, seasonId, leagueId, latestScoringPeriod, swid, fromTeamId, toTeamId,
//          trade: [{id, teamId}], expirationDate, comment}
"use strict";

const fs = require("fs");

const webclient = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const inputs = JSON.parse(fs.readFileSync(0, "utf8"));
const excerpts = webclient.transactionCode.excerpts;

function get(object, path, fallback) {
  let value = object;
  for (const key of path.split(".")) value = value == null ? undefined : value[key];
  return value === undefined ? fallback : value;
}

function load(source, modules) {
  const factory = new Function(`return (${source});`)();
  const module = { exports: {} };
  const require = (id) => {
    if (!(id in modules)) throw new Error(`module ${id} is not stubbed`);
    return modules[id];
  };
  require.n = (m) => {
    const getter = () => m;
    getter.a = m;
    return getter;
  };
  factory(module, module.exports, require);
  return module.exports;
}

const sent = [];
const transport = {
  _1: (config, seasonId, leagueId, body) => {
    // What the real post function (`co`) does with these arguments: JSON.stringify(body) as the request data.
    sent.push({ config, seasonId, leagueId, data: JSON.stringify(body) });
    return new Promise(() => {});
  },
};

const types = load(excerpts.types, {});
const model = load(excerpts.model, { 44: types, 10: () => undefined });
const service = load(excerpts.service, {
  1959: model,
  44: types,
  14: transport,
  6: get,
  15: (list, value) => list.includes(value),
}).a;

service.proposeTrade({
  config: { uri_nextgen_api: inputs.game },
  guest: { profile: { swid: inputs.swid } },
  league: { seasonId: inputs.seasonId, id: inputs.leagueId, status: { latestScoringPeriod: inputs.latestScoringPeriod } },
  fromTeamId: inputs.fromTeamId,
  toTeamId: inputs.toTeamId,
  transactionListByAction: { [types.z]: inputs.trade },
  expirationDate: inputs.expirationDate,
  comment: inputs.comment,
});

if (sent.length !== 1) throw new Error(`expected one stubbed send, saw ${sent.length}`);
const [call] = sent;
process.stdout.write(
  JSON.stringify({
    path: `games/${call.config.uri_nextgen_api}/seasons/${call.seasonId}/segments/0/leagues/${call.leagueId}/transactions/`,
    data: call.data,
  }),
);
