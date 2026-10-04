/**
 * The 3D scenes' data, sliced at build time from the recorded runtime outputs so the browser
 * receives only what it draws. Columns follow the same order as the journey's SVG figures.
 */
import journey from "../src/data/journey.json" with { type: "json" };
import catalogue from "../src/data/catalogue.json" with { type: "json" };
import hero from "../src/data/hero.json" with { type: "json" };

const TYPES = ["expression", "copy_number", "protein", "clinical", "derived", "lab"];
type Col = { id: string; type: string; timing: string };
export type SceneCol = { id: string; group: string; post: boolean };
export type SceneWorld = {
  label: string;
  cols: SceneCol[];
  z: number[][];
  outcome: number[];
  evidence: number[];
  threshold: number;
  truth: string[];
};

const orderCols = (cols: Col[]) => [
  ...TYPES.flatMap((t) => cols.filter((c) => c.type === t && c.timing === "baseline")),
  ...cols.filter((c) => c.timing === "post_outcome"),
];
const sceneCols = (cols: Col[]): SceneCol[] =>
  orderCols(cols).map((c) => ({ id: c.id, group: c.timing === "post_outcome" ? "post_outcome" : c.type, post: c.timing === "post_outcome" }));
/** The journey agent's line: Bonferroni at 5% over the baseline measurements, as -log10 p. */
const bonferroni = (cols: Col[]) => -Math.log10(0.05 / cols.filter((c) => c.timing === "baseline").length);
const round = (v: number) => Math.round(v * 100) / 100;

function catalogueWorld(mechanic: string, label: string): SceneWorld {
  const w = catalogue.find((e) => e.mechanic === mechanic)!;
  const cols = w.features as Col[];
  const order = orderCols(cols);
  const index = order.map((c) => cols.findIndex((f) => f.id === c.id));
  // every patient of the world, from hero.json (the catalogue keeps only a preview)
  const full = (hero as Record<string, { features: string[]; z: number[][]; outcome: number[] }>)[w.world];
  if (full.features.join() !== cols.map((c) => c.id).join()) throw new Error(`hero.json columns differ for ${w.world}`);
  return {
    label,
    cols: sceneCols(cols),
    z: full.z.map((row) => index.map((k) => row[k])),
    outcome: full.outcome,
    evidence: order.map((c) => round((w.evidence as unknown as Record<string, number>)[c.id])),
    threshold: round(bonferroni(cols)),
    truth: (w.truth as string[][]).flat(),
  };
}

export function sceneData(): { journey: JourneyScene; hero: { signal: SceneWorld; none: SceneWorld } } {
  const cols = journey.card.features as Col[];
  const order = orderCols(cols);
  const index = order.map((c) => cols.findIndex((f) => f.id === c.id));
  const rows = Array.from({ length: 60 }, (_, r) => r * 4); // every fourth patient, as in the cohort figure
  const queue = journey.pool.queue as number[];
  const analyses = journey.steps.filter((s) => s.kind === "analyse");
  return {
    journey: {
      cols: sceneCols(cols),
      z: rows.map((r) => index.map((k) => journey.pool.z[r][k])),
      outcome: rows.map((r) => journey.pool.outcome[r]),
      // when each shown patient is recruited: their position in the world's recruitment queue
      queueRank: rows.map((r) => queue.indexOf(r)),
      stages: analyses.map((a) => ({
        patients: a.patients,
        evidence: order.map((c) => round((a.evidence as Record<string, number>)[c.id])),
        listed: (a.ranking ?? []) as string[],
      })),
      threshold: journey.agent_threshold,
      truth: journey.key.truth,
      twin: journey.key.equivalent[0],
    },
    hero: {
      signal: catalogueWorld("generating", "A world with a planted cause"),
      none: catalogueWorld("no_signal", "A world with none") as SceneWorld,
    },
  };
}
export type JourneyScene = {
  cols: SceneCol[];
  z: number[][];
  outcome: number[];
  queueRank: number[];
  stages: { patients: number; evidence: number[]; listed: string[] }[];
  threshold: number;
  truth: string;
  twin: string;
};
export type SceneData = ReturnType<typeof sceneData>;
