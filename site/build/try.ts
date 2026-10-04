/**
 * "Try one": the visitor plays a real training world. Markup is baked from src/data/play.json so it
 * reads without JavaScript (showing the evidence after 90 patients); src/lib/try.ts makes it playable.
 */
import play from "../src/data/play.json" with { type: "json" };
import journey from "../src/data/journey.json" with { type: "json" };
import { esc } from "./visuals";

const TYPES = ["expression", "copy_number", "protein", "clinical", "derived", "lab"];
const NAME: Record<string, string> = { expression: "Gene expression", copy_number: "Copy number", protein: "Protein", clinical: "Clinical", derived: "Derived", lab: "Lab", post_outcome: "After outcome" };
const money = (n: number) => Math.round(n).toLocaleString("en-GB");
type F = (typeof play.features)[number];
const group = (f: F) => (f.timing === "post_outcome" ? "post_outcome" : f.type);
export const ordered: F[] = [...TYPES.flatMap((t) => play.features.filter((f) => f.type === t && f.timing === "baseline")), ...play.features.filter((f) => f.timing === "post_outcome")];
/** Bars share one square-root scale across every purchase, so growth between purchases is real. */
export const MAX = Math.max(...play.stages.flatMap((s) => Object.values(s.evidence)));
export const height = (v: number) => Math.sqrt(Math.max(0, v) / MAX);

export function tryMarkup(): Record<string, string> {
  const fallback = play.stages.find((s) => s.patients === 90) ?? play.stages[0];
  const groups = [...new Set(ordered.map(group))]
    .map((g) => {
      const bars = ordered
        .filter((f) => group(f) === g)
        .map((f) => {
          const v = (fallback.evidence as Record<string, number>)[f.id];
          return `<button type="button" class="bar${f.timing === "post_outcome" ? " bar--post" : ""}" data-id="${esc(f.id)}" aria-pressed="false" aria-label="${esc(f.id)}, ${esc(NAME[group(f)])}${f.timing === "post_outcome" ? ", measured after the outcome" : ""}" style="--h:${height(v).toFixed(4)}"><i></i><b class="bar__rank" aria-hidden="true"></b><span class="bar__id">${esc(f.id)}</span></button>`;
        })
        .join("");
      // each group's width follows its number of measurements, so every bar is the same width
      const n = ordered.filter((f) => group(f) === g).length;
      return `<div class="bars__group${g === "post_outcome" ? " bars__group--post" : ""}" style="flex:${n} 1 0"><div class="bars__row">${bars}</div><span class="bars__name">${esc(NAME[g])}</span></div>`;
    })
    .join("");
  const stages = play.stages
    .map((s, i) => `<button type="button" class="buy" data-stage="${i}" aria-pressed="false"><b>${s.patients}</b><span>$${money(s.spent)}</span></button>`)
    .join("");
  return {
    "<!--@try-bars-->": groups,
    "<!--@try-stages-->": stages,
    "@@try-line@@": height(play.agent_threshold).toFixed(4),
    "@@try-budget@@": money(play.budget),
    "@@try-ref@@": money(play.reference_cost),
    "@@try-n@@": String(play.n_pool),
    "@@try-features@@": String(play.features.length),
    "@@try-per-patient@@": money(play.stages[0].spent / play.stages[0].patients),
    "@@try-truth@@": esc(journey.key.truth),
    "@@try-twin@@": esc(journey.key.equivalent[0] ?? ""),
  };
}
