/**
 * Build-time rendering: every number on the page is baked into static HTML from the data
 * files, so the story, the results table and every figure read without JavaScript.
 * Client scripts only animate what is already there.
 */
import results from "../src/data/results.json" with { type: "json" };
import { staticBand } from "./band";
import { journeyReplacements } from "./journey";

type Summary = {
  discovery_score: number | null;
  low: number | null;
  high: number | null;
  find: number | null;
  restraint: number | null;
  leak_rate: number | null;
};
type Agent = {
  id: string;
  label: string;
  kind: "frontier" | "reference" | "baseline" | "cheater";
  provenance: "placeholder" | "toy-fixture" | "benchmark" | "final";
  overall: Summary;
  by_mode: Record<string, Partial<Summary>>;
};

// Only agents with scorecards are rendered; frontier rows appear when their runs land.
const agents = (results.agents as unknown as Agent[]).filter((a) => a.provenance !== "placeholder");

const esc = (s: string) =>
  s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] as string);
const fmt = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined ? "—" : (Math.abs(v) < 0.005 ? 0 : v).toFixed(d);

const KIND_LABEL: Record<Agent["kind"], string> = {
  frontier: "Frontier agent",
  reference: "Reference",
  baseline: "Classical baseline",
  cheater: "Gaming strategy",
};

/** The scorecards as a real table. */
export function resultsTable(): string {
  const order: Agent["kind"][] = ["reference", "frontier", "baseline", "cheater"];
  const rows = [...agents]
    .sort(
      (a, b) =>
        order.indexOf(a.kind) - order.indexOf(b.kind) ||
        (b.overall.discovery_score ?? -1) - (a.overall.discovery_score ?? -1),
    )
    .map((a) => {
      const cell = (v: number | null | undefined) => `<td class="num">${fmt(v)}</td>`;
      return `<tr data-kind="${a.kind}">
  <th scope="row">${esc(a.label)}</th>
  <td>${KIND_LABEL[a.kind]}</td>
  ${cell(a.overall.discovery_score)}
  <td class="num">${fmt(a.overall.low)} to ${fmt(a.overall.high)}</td>
  ${cell(a.overall.find)}${cell(a.overall.restraint)}
  ${cell(a.by_mode.full_access?.discovery_score)}${cell(a.by_mode.sequential?.discovery_score)}
  ${cell(a.overall.leak_rate)}
</tr>`;
    })
    .join("\n");
  return `<table class="results-table">
<caption>Discovery Score with its 95% bootstrap interval, its two factors, the score in each mode, and the leak rate. A gaming strategy can recover a cause on some worlds and still score zero, because it never says nothing where nothing is there.</caption>
<thead><tr><th scope="col">Agent</th><th scope="col">Kind</th><th scope="col">Discovery Score</th><th scope="col">95% interval</th><th scope="col">Find</th><th scope="col">Restraint</th><th scope="col">Full access</th><th scope="col">Sequential</th><th scope="col">Leak rate</th></tr></thead>
<tbody>
${rows}
</tbody>
</table>`;
}

export function statusLine(): string {
  const m = results.worlds.by_mode;
  const set = results.worlds.source === "benchmark public train" ? "public-train worlds" : "worlds";
  return `<p class="status">${results.worlds.count} ${set}: ${m.full_access} full access, ${m.sequential} sequential · ${esc(results.scorer)} · interface ${esc(results.interface_version)}</p>`;
}

function smokeLines(): string {
  return ["oracle", "univariate_bh", "giant_list"]
    .map((id) => agents.find((a) => a.id === id))
    .filter((a): a is Agent => Boolean(a))
    .map((a) => {
      const o = a.overall;
      const tail =
        a.kind === "cheater"
          ? `Leak ${fmt(o.leak_rate)}`
          : `Restraint ${(o.restraint ?? 0) >= 0 ? "+" : ""}${fmt(o.restraint)}`;
      return `<span class="terminal__out">${a.id.padEnd(16)} DS  ${fmt(o.discovery_score, 3)}   Find  ${fmt(o.find)}  ${tail}</span>`;
    })
    .join("\n");
}

export function replacements(): Record<string, string> {
  return {
    "<!--@results-table-->": resultsTable(),
    "<!--@band-static-->": staticBand(),
    "<!--@status-->": statusLine(),
    "@@smoke-lines@@": smokeLines(),
    "@@world-count@@": ((results.worlds as { published?: number }).published ?? results.worlds.count).toLocaleString("en-GB"),
    ...journeyReplacements(),
  };
}
