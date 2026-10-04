/** Binds the journey's narrative markers in index.html to the recorded runtime data. */
import journey from "../src/data/journey.json" with { type: "json" };
import { analyses, certify, cohort, esc, evidence, n80, scoring, worldCard } from "./visuals";

export { esc };
const usd = (n: number) => Math.round(n).toLocaleString("en-GB");
const card = journey.card;
const key = journey.key;
const steps = journey.steps;
const submission = steps.find((s) => s.kind === "submit")?.ranking ?? [];

export function journeyReplacements(): Record<string, string> {
  const last = steps[steps.length - 1];
  const a0 = analyses[0];
  return {
    "<!--@pane-build-->": cohort(),
    "<!--@pane-certify-->": certify(),
    "<!--@pane-receive-->": worldCard(),
    "<!--@pane-act-->": evidence(),
    "<!--@pane-score-->": scoring(),
    "@@j-world@@": esc(journey.world),
    "@@j-n@@": String(card.n_pool),
    "@@j-features@@": String(card.features.length),
    "@@j-post@@": String(card.features.filter((f) => f.timing === "post_outcome").length),
    "@@j-budget@@": usd(card.budget),
    "@@j-ref@@": usd(key.reference_cost),
    "@@j-spent@@": usd(last.spent),
    "@@j-patients@@": String(last.patients),
    "@@j-truth@@": esc(key.truth),
    "@@j-twin@@": esc(key.equivalent[0] ?? ""),
    "@@j-first-n@@": String(a0.patients),
    "@@j-first-spent@@": usd(a0.spent),
    "@@j-submission@@": submission.map(esc).join(", "),
    "@@j-oracle-n80@@": String(n80 ?? "—"),
    "@@j-transport-data@@": esc(
      JSON.stringify({
        budget: card.budget,
        reference: key.reference_cost,
        cross: analyses.find((a) => (a.ranking ?? []).length)?.spent ?? null,
        steps: steps.map((s) => ({ kind: s.kind, patients: s.patients, spent: s.spent })),
      }),
    ),
  };
}
