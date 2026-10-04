/**
 * The films' direction: which scenes exist, how each clip is staged, and what the page needs to
 * know about it (duration, labels, captions). Runs only at build time, inside build/render-media.
 *
 * Every clip is deterministic: a fixed time step, seeded jitter and recorded runtime data, so the
 * same build renders the same frames.
 */
import raw from "virtual:scene";
import type { SceneData, SceneWorld } from "../build/scene";
import { CohortScene, TINT, type CameraPose, type Label } from "./cohort3d";

const data = raw as SceneData;
export const FPS = 60; // smooth camera moves on 60 Hz displays

/** What each band of columns measures, in the reader's words. */
const KIND: Record<string, string> = {
  expression: "Gene expression",
  copy_number: "Copy number",
  protein: "Protein",
  clinical: "Clinical",
  derived: "Derived",
  lab: "Lab",
  post_outcome: "Post-outcome",
};

/** One step of a clip's script: at `at` seconds into the clip, do something to the scene. */
export type Cue = { at: number; run: () => void };
/** A clip to render: its scene, how long it runs, and its cues. `settle` ends it once still. */
export type Job = {
  name: string;
  scene: CohortScene;
  /** Prepare the starting state (runs before frame 0; the scene is snapped to it). */
  start?: () => void;
  cues: Cue[];
  duration?: number;
  settle?: { min: number; max: number };
  /** Record labels every this many frames (films whose labels move), or only at the end. */
  labelEvery?: number;
  tags?: { at: number; text: string }[];
  poster?: number; // frame index to save as a still
};

/* ------------------------------------------------------------------ the opening */

/** The same world laid out right to left (the opening shows no column labels in order). */
const mirrored = (w: SceneWorld): SceneWorld => ({
  ...w,
  cols: [...w.cols].reverse(),
  z: w.z.map((row) => [...row].reverse()),
  evidence: [...w.evidence].reverse(),
});

export const HERO_SIZES = { wide: [1920, 1080], square: [1080, 1080] } as const;
export const LOOP_SECONDS = 20;

/** The opening as two films per shape: an intro that plays once, then a seamless loop. */
export function heroJobs(shape: keyof typeof HERO_SIZES, canvas: HTMLCanvasElement): Job[] {
  const signal = mirrored(data.hero.signal);
  const none = mirrored(data.hero.none);
  const [width, height] = HERO_SIZES[shape];
  const scene = new CohortScene(canvas, signal.cols, signal.z.length, { width, height, fog: [14, 74], shadowDepth: 60 });
  scene.towerScale = 0.2;
  const planted = signal.cols.findIndex((c, i) => !c.post && signal.evidence[i] > signal.threshold);
  const tx = scene.colX[planted] + 0.5;
  const wide = shape === "wide";
  // a low horizon shot: the field fills the lower frame into fog; the planted tower stands right
  const rest: CameraPose = wide ? { pos: [tx - 4.5, 3, -19], look: [tx + 1, 4.45, 64], fov: 34 } : { pos: [tx - 3.2, 3.4, -21], look: [tx - 1.2, 4.6, 64], fov: 40 };
  const intro: CameraPose = wide ? { pos: [tx - 3, 2.2, -11], look: [tx + 1, 3.6, 64], fov: 30 } : { pos: [tx - 2, 2.4, -13], look: [tx - 1.2, 3.8, 64], fov: 36 };
  const caption = (w: SceneWorld) => {
    const over = w.cols.filter((c, i) => !c.post && w.evidence[i] > w.threshold).length;
    return w === signal ? `${w.label}: ${over === 1 ? "one measurement rises through the line" : `${over} measurements rise through the line`}` : `${w.label}: nothing reaches the line`;
  };
  const towers = (w: SceneWorld) => {
    scene.setTowers(w.evidence, w.threshold, "benchmark");
    const baseline = w.cols.map((c, i) => ({ c, i })).filter(({ c }) => !c.post);
    const over = baseline.filter(({ i }) => w.evidence[i] > w.threshold);
    const labels: Label[] = over.map(({ c, i }) => ({ key: `t-${c.id}`, id: c.id, text: "", cls: "is-red", at: () => scene.towerTop(i) }));
    if (!over.length) {
      // with no signal, the strongest measurement is named where it stops, short of the line
      const top = baseline.reduce((a, b) => (w.evidence[b.i] > w.evidence[a.i] ? b : a));
      labels.push({ key: `t-${top.c.id}`, id: top.c.id, text: "strongest, still below the line", cls: "is-blue is-left", at: () => scene.towerTop(top.i) });
    }
    if (wide)
      for (const g of scene.groups().filter((g) => g.group !== "post_outcome" && signal.cols.filter((c) => c.group === g.group).length >= 3)) labels.push({ key: `g-${g.group}`, text: KIND[g.group] ?? g.group, cls: `is-quiet is-kind${g.group === "post_outcome" ? " is-blue" : ""}`, at: () => [g.x, 1.3, 2] });
    labels.push({ key: "line", text: "multiple-testing line", cls: "is-red is-quiet is-flat", at: () => [tx - (wide ? 1.6 : 2.6), scene.lineY(), -5.6] });
    scene.setLabels(labels);
    scene.focusOn(over.length ? [scene.colX[over[0].i] + 0.5, 2, -5] : [tx - 4, 0.6, 18]);
  };
  const clear = () => scene.setLabels(wide ? scene.groups().filter((g) => g.group !== "post_outcome" && signal.cols.filter((c) => c.group === g.group).length >= 3).map((g) => ({ key: `g-${g.group}`, text: KIND[g.group] ?? g.group, cls: `is-quiet is-kind${g.group === "post_outcome" ? " is-blue" : ""}`, at: () => [g.x, 1.3, 2] as [number, number, number] })) : []);
  const L = LOOP_SECONDS;
  return [
    {
      // out of the dark the camera eases back, the cohort lands row by row, front first, and the
      // towers rise; the planted one pierces the line
      name: `hero-${shape}-intro`,
      scene,
      start: () => {
        scene.setValues(signal.z);
        scene.placeCamera(intro);
        scene.focusOn([tx, 1, 10]);
        clear();
      },
      cues: [
        { at: 0, run: () => { scene.setCamera(rest); scene.reveal(6); scene.assemble(1.6); } },
        { at: 2.3, run: () => towers(signal) },
      ],
      duration: 5.5,
      labelEvery: 6,
      tags: [{ at: 2.6, text: caption(signal) }],
      poster: Math.round(5.5 * FPS) - 1,
    },
    {
      // the loop starts and ends on the settled signal world, with the camera sway at zero
      name: `hero-${shape}-loop`,
      scene,
      cues: [
        { at: 4.0, run: () => { scene.lowerTowers(); clear(); } },
        { at: 5.2, run: () => { scene.setTowers(null, null); scene.swapValues(none.z, 1.7); } },
        { at: 7.8, run: () => towers(none) },
        { at: 12.8, run: () => { scene.lowerTowers(); clear(); } },
        { at: 14.0, run: () => { scene.setTowers(null, null); scene.swapValues(signal.z, 1.7); } },
        { at: 16.6, run: () => towers(signal) },
      ],
      duration: L,
      labelEvery: 6,
      tags: [
        { at: 0, text: caption(signal) },
        { at: 6.55, text: caption(none) },
        { at: 15.35, text: caption(signal) },
      ],
    },
  ];
}

/* ------------------------------------------------------------------ the journey */

export const JOURNEY_SIZE = [1728, 1152] as const; // within H.264 level 4.0 frame-area limits
export type StateName = "build-0" | "build-1" | "build-2" | "build-3" | "act-0" | "act-1" | "act-2" | "top-b3" | "top-a0" | "top-a2";

export function journey(canvas: HTMLCanvasElement) {
  const j = data.journey;
  const rows = j.z.length;
  const [width, height] = JOURNEY_SIZE;
  const scene = new CohortScene(canvas, j.cols, rows, { width, height, fog: [60, 130] });
  scene.towerScale = 0.62;
  const truth = j.cols.findIndex((c) => c.id === j.truth);
  const twin = j.cols.findIndex((c) => c.id === j.twin);
  const firstPost = j.cols.findIndex((c) => c.post);
  const midX = (scene.colX[0] + scene.colX[scene.cols.length - 1]) / 2 + 0.5;
  const byOutcome = j.outcome.map((_, r) => r).sort((a, b) => j.outcome[b] - j.outcome[a] || a - b);
  const sorted = j.outcome.map((_, r) => byOutcome.indexOf(r));
  const events = j.outcome.filter(Boolean).length;
  const byQueue = j.queueRank.map((_, r) => r).sort((a, b) => j.queueRank[a] - j.queueRank[b]);
  const recruited = j.queueRank.map((_, r) => byQueue.indexOf(r));
  const CAM: Record<string, CameraPose> = {
    intro: { pos: [2, 18, -12], look: [14, -1, 30], fov: 30 },
    "build-0": { pos: [-4, 30, -20], look: [14, -3, 28], fov: 33 },
    "build-1": { pos: [-7, 15, -11], look: [4, -1, 24], fov: 34 },
    "build-2": { pos: [-10, 7.5, -9], look: [3, 0, 38], fov: 36 },
    "build-3": { pos: [12, 22, -17], look: [21, -2, 26], fov: 33 },
    act: { pos: [0.5, 16, -31], look: [12.5, -0.6, 14], fov: 33 },
    top: { pos: [13.5, 64, 22], look: [13.5, 0, 30], fov: 30 },
  };
  const captions: Partial<Record<StateName, string>> = {};

  const build = (state: number) => {
    scene.setFieldEmphasis(1, 1);
    scene.tintColumns(state >= 1 ? { [truth]: TINT.red, [twin]: TINT.ink } : null);
    scene.setOrder(state >= 2 ? sorted : null, 0.6);
    scene.setOutcome(state >= 2 ? j.outcome : null);
    scene.setPost(state >= 3);
    scene.setGhost(() => 0);
    scene.setTowers(null, null);
    scene.setCamera(CAM[`build-${state}`]);
    const postX = scene.colX[firstPost + 1] + 2.1;
    scene.focusOn(state === 0 ? [midX, 0.5, rows * 0.35] : state === 1 ? [scene.colX[truth] + 0.5, 0.5, rows * 0.3] : state === 2 ? [-1.3, 0.6, rows * 0.15] : [postX, 0.5, rows * 0.4]);
    const labels: Label[] = [];
    if (state === 0)
      scene.groups().filter((g) => scene.cols.filter((c) => c.group === g.group).length >= 3).forEach((g, k) =>
        labels.push({ key: `g-${g.group}`, text: KIND[g.group] ?? g.group, cls: `is-quiet is-kind${g.group === "post_outcome" ? " is-blue" : ""}`, at: () => [g.x, k % 2 ? 2.6 : 1.2, 3] }),
      );
    if (state === 1 || state === 2) {
      labels.push({ key: "truth", text: "Planted driver", id: j.truth, cls: "is-red is-left", at: () => scene.columnHead(truth) });
      labels.push({ key: "twin", text: "Near-duplicate, also accepted", id: j.twin, cls: "is-ink", at: () => scene.columnHead(twin) });
    }
    if (state === 2) labels.push({ key: "outcome", text: `Outcome · ${events} of ${rows} had the event, sorted first`, cls: "is-quiet is-flat", at: () => [-3.2, 1.2, rows * 0.22] });
    if (state >= 3) labels.push({ key: "post", text: "Measured after the outcome", cls: "is-blue", at: () => [scene.columnHead(firstPost + 1)[0], 1.3, rows * 0.55] });
    scene.setLabels(labels);
    captions[`build-${state}` as StateName] = `<b>The cohort</b> · every fourth of ${rows * 4} patients × ${j.cols.length} measurements`;
  };

  const act = (state: number) => {
    const s = j.stages[state];
    const prev = state > 0 ? j.stages[state - 1] : null;
    const repeat = prev && JSON.stringify(prev.listed) === JSON.stringify(s.listed);
    const shownBought = j.queueRank.filter((q) => q < s.patients).length;
    scene.tintColumns(null);
    scene.setFieldEmphasis(0.55, 0.6);
    scene.setOrder(recruited, 0.5);
    scene.setOutcome(null);
    scene.setPost(true);
    // only the patients bought so far stand; the rest of the pool lies flat and dark
    scene.setGhost((r) => (j.queueRank[r] < s.patients ? 0 : 1), 0.8);
    scene.setTowers(s.evidence, j.threshold, "agent");
    scene.setCamera(CAM.act);
    scene.focusOn([midX, 2, -5]);
    const labels: Label[] = s.listed.map((id) => {
      const i = j.cols.findIndex((c) => c.id === id);
      return { key: `t-${id}`, text: "", id, cls: "is-red", at: () => scene.towerTop(i) };
    });
    labels.push({ key: "line", text: "Agent's line · Bonferroni 5%", cls: "is-quiet is-below", at: () => [0.2, scene.lineY(), -5.6] });
    labels.push({ key: "pool", text: "Patients not yet bought", cls: "is-quiet", at: () => [midX, 0.3, (shownBought + rows) / 2] });
    scene.setLabels(labels);
    captions[`act-${state}` as StateName] = `<b>Evidence per measurement</b> · −log<sub>10</sub> p · ${s.patients} patients bought (${shownBought} of the ${rows} shown) · list <code>[${s.listed.join(", ")}]</code>${repeat ? " · repeats, so it stops" : ""}`;
  };

  const apply = (name: StateName) => {
    if (name.startsWith("build-")) build(Number(name.slice(-1)));
    else if (name.startsWith("act-")) act(Number(name.slice(-1)));
    else {
      // hidden behind a 2D figure: the arrangement it left, seen from straight above
      apply(name === "top-b3" ? "build-3" : name === "top-a0" ? "act-0" : "act-2");
      scene.setCamera(CAM.top);
      scene.setLabels([]);
    }
  };

  /** A clip from one settled state to another. */
  const clip = (from: StateName | "enter", to: StateName): Job => ({
    name: `journey-${from}--${to}`,
    scene,
    start: () => {
      scene.setValues(j.z);
      if (from === "enter") {
        apply("build-0");
        scene.setLabels([]);
      } else apply(from);
      scene.snap();
      if (from === "enter") {
        scene.placeCamera(CAM.intro);
        scene.assemble(1.5);
        scene.reveal(10);
      }
    },
    cues: [{ at: 0, run: () => apply(to) }],
    settle: { min: 1.2, max: 3.2 },
  });

  const pairs: [StateName | "enter", StateName][] = [
    ["enter", "build-0"],
    ["build-0", "build-1"], ["build-1", "build-2"], ["build-2", "build-3"],
    ["build-1", "build-0"], ["build-2", "build-1"], ["build-3", "build-2"],
    ["build-3", "top-b3"], ["top-b3", "build-3"],
    ["top-b3", "act-0"], ["act-0", "top-b3"],
    ["act-0", "act-1"], ["act-1", "act-2"], ["act-1", "act-0"], ["act-2", "act-1"],
    ["act-2", "top-a2"], ["top-a2", "act-2"],
  ];
  return { scene, jobs: pairs.map(([a, b]) => clip(a, b)), captions };
}
