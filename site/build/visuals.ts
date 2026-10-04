/**
 * Build-time figures for the journey. Every mark is computed from the recorded runtime data in
 * src/data/; the browser only moves between states that already exist in this markup. The same
 * SVGs are exported to dist/figures/ as standalone, editable files.
 */
import * as Plot from "@observablehq/plot";
import { JSDOM } from "jsdom";
import { scaleBand, scaleLinear } from "d3-scale";
import { interpolateRgbBasis } from "d3-interpolate";
import { linkHorizontal } from "d3-shape";
import j from "../src/data/journey.json" with { type: "json" };
import o from "../src/data/oracle.json" with { type: "json" };

const doc = new JSDOM("").window.document;

export const C = {
  paper: "#0f1a2b",
  raised: "#13233a",
  horizon: "#1a2d48",
  well: "#0a1220",
  ink: "#e9eef2",
  muted: "#a9bacb",
  faint: "#7d93aa",
  blue: "#7fa7c9",
  deep: "#3e5f82",
  signal: "#d7263d",
  red: "#ff6b7b",
  rule: "#2b3d51",
};
const SANS = "'Archivo Variable', Archivo, system-ui, sans-serif";
const MONO = "'Azeret Mono Variable', 'Azeret Mono', ui-monospace, monospace";

export const esc = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
const money = (n: number) => Math.round(n).toLocaleString("en-GB");
const text = (x: number, y: number, s: string, attrs = "") => `<text x="${x}" y="${y}" ${attrs}>${esc(s)}</text>`;

/** One self-contained figure: its own type and colour rules travel with it into the downloads. */
const STYLE =
  `.fig text{fill:${C.ink};font-family:${SANS};font-size:15px}` +
  `.fig .quiet{fill:${C.muted};font-size:14px}` +
  `.fig .small{fill:${C.muted};font-size:13px}` +
  `.fig .mono{font-family:${MONO};font-variant-numeric:tabular-nums}` +
  `.fig .head{font-family:${MONO};font-size:11px;fill:${C.faint}}` +
  `.fig .large{font-family:${MONO};font-size:38px;letter-spacing:-1px}` +
  `.fig .red{fill:${C.red}}.fig .blue{fill:${C.blue}}` +
  `.fig .rule{stroke:${C.rule};fill:none}` +
  `.fig .grid{stroke:${C.faint};stroke-opacity:.16;fill:none}`;
const svg = (body: string, label: string, h = 470, w = 720, cls = "") =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" class="fig ${cls}" role="img" aria-label="${esc(label)}"><style>${STYLE}</style><rect class="fig-ground" width="${w}" height="${h}" fill="${C.paper}"/>${body}</svg>`;
const pane = (name: string, inner: string, caption: string) =>
  `<figure class="pane pane--${name}" data-pane="${name}">${inner}<figcaption class="caption">${caption}</figcaption></figure>`;

/* ------------------------------------------------------------------ shared data */

const TYPES = ["expression", "copy_number", "protein", "clinical", "derived", "lab"] as const;
const GROUP_NAME: Record<string, string> = {
  expression: "Expression",
  copy_number: "Copy no.",
  protein: "Protein",
  clinical: "Clinical",
  derived: "Derived",
  lab: "Lab",
  post_outcome: "After outcome",
};
type Feature = (typeof j.card.features)[number];
const groupOf = (f: Feature) => (f.timing === "post_outcome" ? "post_outcome" : f.type);
/** Measurements by kind, post-outcome last: the order every journey figure shares. */
export const ordered: Feature[] = [
  ...TYPES.flatMap((t) => j.card.features.filter((f) => f.type === t && f.timing === "baseline")),
  ...j.card.features.filter((f) => f.timing === "post_outcome"),
];
const groups = [...new Set(ordered.map(groupOf))];
export const analyses = j.steps.filter((s) => s.kind === "analyse");
const truth = j.key.truth;
const twin = j.key.equivalent[0];

function groupRules(x: (id: string) => number, bw: number, y: number, pad = 0) {
  return groups
    .map((g) => {
      const fs = ordered.filter((f) => groupOf(f) === g);
      const a = x(fs[0].id) - pad;
      const b = x(fs.at(-1)!.id) + bw + pad;
      const post = g === "post_outcome";
      return (
        `<path d="M${a} ${y} H${b}" class="rule"${post ? ` stroke-dasharray="3 3"` : ""}/>` +
        text((a + b) / 2, y + 17, GROUP_NAME[g], `text-anchor="middle" class="small${post ? " blue" : ""}"`)
      );
    })
    .join("");
}

/* ------------------------------------------------------------------ 1 · the cohort */

const ROWS = 60; // every fourth patient
export function cohort(): string {
  const x = scaleBand().domain(ordered.map((f) => f.id)).range([66, 704]).padding(0.14);
  const top = 112;
  const y = scaleBand<number>().domain(Array.from({ length: ROWS }, (_, i) => i)).range([top, 376]).padding(0.1);
  const step = y.step();
  const shade = interpolateRgbBasis([C.raised, C.deep, "#c9dae6"]); // the same ramp as the films: Crest Pale on top
  const rows = Array.from({ length: ROWS }, (_, r) => r * 4);
  // State 2 sorts patients by outcome, events first, so the driver's column visibly grades with it.
  const sorted = [...rows].sort((a, b) => j.pool.outcome[b] - j.pool.outcome[a] || a - b);
  const events = rows.filter((r) => j.pool.outcome[r]).length;
  let s = text(24, 30, "The cohort", 'font-size="18"') +
    text(704, 30, `${j.card.n_pool} patients × ${ordered.length} measurements`, 'text-anchor="end" class="quiet mono" font-size="13"');

  // Column heads: every measurement's masked identifier.
  for (const f of ordered) {
    const cx = x(f.id)! + x.bandwidth() / 2;
    const cls = f.id === truth ? "is-truth" : f.id === twin ? "is-twin" : f.timing === "post_outcome" ? "is-post" : "";
    s += `<text class="head colhead ${cls}" transform="translate(${cx + 3} ${top - 8}) rotate(-55)">${esc(f.id)}</text>`;
  }
  // Call-outs for the planted pair (from state 1).
  for (const [id, label, yy, cls] of [
    [truth, `Planted driver`, 50, "red"],
    [twin, `Near-duplicate, also accepted`, 68, "blue"],
  ] as const) {
    const cx = x(id)! + x.bandwidth() / 2;
    s += `<g class="from-1"><path d="M${cx} ${top - 40} V${yy - 4} H${cx + 8}" stroke="${id === truth ? C.red : C.blue}" fill="none"/>` +
      text(cx + 12, yy, label, `class="${cls}" font-size="14"`) + `</g>`;
  }
  // Outcome strip and patient rows; each row carries its sorted offset.
  for (const [k, row] of rows.entries()) {
    const dy = (sorted.indexOf(row) - k) * step;
    s += `<g class="r" style="--dy:${dy.toFixed(2)}px">`;
    s += `<rect x="28" y="${y(k)}" width="16" height="${y.bandwidth()}" class="oc ${j.pool.outcome[row] ? "y1" : "y0"}" fill="${j.pool.outcome[row] ? C.ink : C.deep}"/>`;
    for (const f of ordered) {
      const col = j.card.features.findIndex((a) => a.id === f.id);
      const z = j.pool.z[row][col];
      s += `<rect x="${x(f.id)}" y="${y(k)}" width="${x.bandwidth()}" height="${y.bandwidth()}" fill="${shade(Math.max(0, Math.min(1, (z + 2.5) / 5)))}"/>`;
    }
    s += "</g>";
  }
  s += `<g class="from-2">` + text(36, top - 8, "Outcome", `class="small" transform="rotate(-90 36 ${top - 8})" text-anchor="start"`) +
    `<path d="M24 ${top} V${top + events * step - 2}" stroke="${C.ink}"/>` +
    text(20, top + (events * step) / 2, `events`, `class="small" text-anchor="middle" transform="rotate(-90 20 ${top + (events * step) / 2})"`) + `</g>`;
  // Planted and post-outcome columns.
  for (const f of ordered) {
    const xx = x(f.id)!;
    if (f.id === truth || f.id === twin)
      s += `<rect x="${xx - 2}" y="${top - 3}" width="${x.bandwidth() + 4}" height="${376 - top + 6}" class="colmark from-1" fill="none" stroke="${f.id === truth ? C.red : C.blue}" stroke-width="1.5"/>`;
    if (f.timing === "post_outcome")
      s += `<rect x="${xx}" y="${top}" width="${x.bandwidth()}" height="${376 - top}" class="from-3" fill="url(#post-hatch)"/>`;
  }
  s = `<defs><pattern id="post-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="${C.paper}" fill-opacity=".55"/><line x1="0" y1="0" x2="0" y2="6" stroke="${C.blue}" stroke-opacity=".7" stroke-width="1.4"/></pattern></defs>` + s;
  s += groupRules((id) => x(id)!, x.bandwidth(), 392);
  // Legend.
  for (let k = 0; k < 100; k++) s += `<rect x="${86 + k * 1.4}" y="438" width="1.5" height="12" fill="${shade(k / 99)}"/>`;
  s += text(80, 448, "−2.5", 'text-anchor="end" class="small mono"') + text(232, 448, "+2.5 z", 'class="small mono"');
  s += `<g class="from-2">${text(704, 448, "Rows sorted by outcome", 'text-anchor="end" class="small"')}</g>`;
  return pane(
    "build",
    svg(s, `Cohort of ${j.card.n_pool} patients and ${ordered.length} measurements; the planted driver ${truth} and its near-duplicate ${twin}; patients sorted by outcome; three post-outcome measurements`),
    `Every fourth patient of ${j.card.n_pool}. Colour encodes the standardised value of each measurement.`,
  );
}

/* ------------------------------------------------------------------ 2 · what can be found */

type Curve = { n: number; rate: number; low: number; high: number };
const SERIES = [
  { key: "cause", label: `Driver ${o.features.cause}`, color: C.red },
  { key: "stand_in", label: `Near-duplicate ${o.features.stand_in}`, color: C.blue },
  { key: "unrelated", label: `Unrelated ${o.features.unrelated}`, color: C.faint },
] as const;
export const n80 = (o.curves.cause as Curve[]).find((c) => c.low >= o.bands.recoverable_lower)?.n ?? null;
export const RECOVERY = { width: 720, height: 430, left: 64, right: 150, top: 36, bottom: 56, domain: [30, 240] as [number, number] };

export function recovery(compact = false): string {
  const g = compact ? { ...RECOVERY, width: 420, left: 52, right: 24 } : RECOVERY;
  const rows = SERIES.flatMap((sr) => (o.curves[sr.key] as Curve[]).map((v) => ({ ...v, series: sr.key })));
  const ends = SERIES.map((sr) => ({ ...(o.curves[sr.key] as Curve[]).at(-1)!, series: sr.key, label: sr.label }));
  const p = Plot.plot({
    document: doc,
    className: compact ? "recovery-compact" : "recovery-plot",
    width: g.width,
    height: g.height,
    marginLeft: g.left,
    marginRight: g.right,
    marginTop: g.top,
    marginBottom: g.bottom,
    style: { background: C.paper, color: C.muted, fontFamily: SANS, fontSize: "14px" },
    x: { type: "log", domain: g.domain, ticks: compact ? [30, 60, 120, 240] : [30, 45, 60, 100, 160, 240], tickFormat: (n: number) => String(n), label: "Patients per cohort →", labelAnchor: "right" },
    y: { domain: [0, 1.04], ticks: [0, 0.2, 0.5, 0.8, 1], tickFormat: (n: number) => `${Math.round(n * 100)}%`, label: "↑ Detected", grid: true },
    color: { domain: SERIES.map((s) => s.key), range: SERIES.map((s) => s.color) },
    marks: [
      Plot.ruleY([o.bands.recoverable_lower], { stroke: C.ink, strokeOpacity: 0.55, strokeDasharray: "2,4" }),
      Plot.text([o.bands.recoverable_lower], { x: 100, y: (d: number) => d, text: () => "80%", dy: -9, textAnchor: "middle", fill: C.muted, fontSize: 13 }),
      Plot.areaY(rows, { x: "n", y1: "low", y2: "high", fill: "series", fillOpacity: 0.16 }),
      Plot.lineY(rows, { x: "n", y: "rate", stroke: "series", strokeWidth: 2.5 }),
      Plot.dot(rows, { x: "n", y: "rate", fill: "series", symbol: "square", r: 3.6 }),
      // direct labels at the line ends; the two saturated curves are nudged apart
      ...(compact ? [] : ends.map((e) => Plot.text([e], { x: "n", y: "rate", text: "label", fill: "series", dx: 12, dy: e.series === "cause" ? -10 : e.series === "stand_in" ? 10 : 0, textAnchor: "start", fontSize: 14 }))),
      // the red mark: the first cohort size whose lower 95% bound for the driver clears 80%
      ...(n80 ? [Plot.ruleX([n80], { y1: 0, y2: o.bands.recoverable_lower, stroke: C.red, strokeWidth: 1.25 }), ...(compact ? [] : [Plot.text([n80], { x: (d: number) => d, y: 0.3, text: () => "lower 95% bound ≥ 80%", dx: 6, textAnchor: "start", fill: C.red, fontSize: 12.5 })])] : []),
    ],
  });
  p.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  p.setAttribute("role", "img");
  p.setAttribute("aria-label", `Detection rate against cohort size: the driver is detected in at least 80% of cohorts from ${n80} patients; the unrelated measurement almost never`);
  p.querySelectorAll('[aria-label="line"] path').forEach((e) => e.classList.add("ci-line"));
  p.querySelectorAll('[aria-label="area"] path').forEach((e) => e.classList.add("ci-band"));
  return p.outerHTML;
}

/** The bar itself: the largest |z| of any measurement in 400 no-signal cohorts, beside the driver's |z|. */
export function nulls(): string {
  const x = scaleLinear().domain([0, 10.5]).range([40, 700]);
  const bin = 0.25;
  const perRow = 4;
  const sq = (x(bin) - x(0)) / perRow - 1;
  const base = 372;
  const stack = (values: number[], color: string, cls: string) => {
    const bins = new Map<number, number>();
    let out = "";
    for (const v of [...values].sort((a, b) => a - b)) {
      const b = Math.floor(v / bin);
      const k = bins.get(b) ?? 0;
      bins.set(b, k + 1);
      const cx = x(b * bin) + (k % perRow) * (sq + 1);
      const cy = base - (Math.floor(k / perRow) + 1) * (sq + 1);
      out += `<rect x="${cx.toFixed(2)}" y="${cy.toFixed(2)}" width="${sq.toFixed(2)}" height="${sq.toFixed(2)}" fill="${color}"/>`;
    }
    return `<g class="${cls}">${out}</g>`;
  };
  const thr = x(o.threshold);
  let s = text(24, 30, "Setting the bar", 'font-size="18"');
  for (const v of [0, 2, 4, 6, 8, 10]) s += `<line x1="${x(v)}" x2="${x(v)}" y1="${base}" y2="${base + 6}" class="rule"/>` + text(x(v), base + 22, String(v), 'text-anchor="middle" class="small mono"');
  s += `<line x1="40" x2="700" y1="${base}" y2="${base}" class="rule"/>` + text(700, base + 46, "|z|, strength of association →", 'text-anchor="end" class="small"');
  s += stack(o.null_max, C.blue, "null-stack");
  s += stack(o.replicate_z, C.signal, "driver-stack");
  s += `<line x1="${thr}" x2="${thr}" y1="64" y2="${base}" stroke="${C.signal}" stroke-width="1.25"/>` +
    text(thr + 8, 78, `Threshold ${o.threshold.toFixed(2)}`, 'class="red mono" font-size="14"') +
    text(thr + 8, 98, "95% of no-signal cohorts stay below", 'class="small"');
  s += text(40, 232, `${o.null_draws} cohorts with no signal:`, 'class="quiet"') + text(40, 252, "the strongest |z| in each", 'class="quiet"');
  s += text(x(6.4), 232, `Driver ${o.features.cause}:`, 'class="red"') + text(x(6.4), 252, `${o.replicates} cohorts of ${j.card.n_pool}`, 'class="quiet"');
  return svg(s, `Histogram of the largest association in ${o.null_draws} no-signal cohorts, its 95th percentile ${o.threshold.toFixed(2)} as the threshold, and the driver's association in ${o.replicates} cohorts, all above it`, 430, 720, "nulls-plot");
}

export function certify(): string {
  return pane(
    "certify",
    `<div class="pane__state pane__state--0"><div class="asset-wide">${recovery()}</div><div class="asset-compact">${recovery(true)}</div></div><div class="pane__state pane__state--1">${nulls()}</div>`,
    `Marginal tests on ${o.replicates} redraws of the outcome per cohort size, with 95% Wilson intervals; the red mark is where the driver's lower bound first clears 80%. The threshold comes from ${o.null_draws} no-signal outcomes.`,
  );
}

/* ------------------------------------------------------------------ 3 · what the agent receives */

export function worldCard(): string {
  const byGroup = groups.map((g) => {
    const fs = ordered.filter((f) => groupOf(f) === g);
    const chips = fs
      .map((f) => `<span class="fid${f.timing === "post_outcome" ? " fid--post" : ""}" title="${esc(f.id)} · ${esc(GROUP_NAME[groupOf(f)])} · $${f.price} per patient">${esc(f.id)}<i>$${f.price}</i></span>`)
      .join("");
    return `<div class="card-group"><span class="card-group__name">${esc(GROUP_NAME[g])}</span><span>${chips}</span></div>`;
  });
  return pane(
    "receive",
    `<div class="worldcard" role="group" aria-label="World card handed to the agent">
      <div class="worldcard__head"><b>Training world</b><span>sequential access</span></div>
      <div class="worldcard__row"><span>Patients available</span><b>${j.card.n_pool} · $${money(j.card.recruit_price)} each to recruit</b></div>
      <div class="worldcard__row"><span>Budget</span><b>$${money(j.card.budget)}</b></div>
      <div class="worldcard__groups"><span class="worldcard__note">Measurements · assay price per patient</span>${byGroup.join("")}</div>
      <div class="worldcard__row"><span>Submit</span><b>an ordered list, or nothing</b></div>
      <div class="worldcard__locked"><span>Answer key</span><b>held by the scorer</b></div>
    </div>`,
    "Identifiers are masked. Kind, timing and price are visible; hollow chips were measured after the outcome.",
  );
}

/* ------------------------------------------------------------------ 4 · acquiring evidence */

const X = scaleBand().domain(ordered.map((f) => f.id)).range([54, 691]).padding(0.5);
export const evidenceY = scaleLinear().domain([0, 10]).range([352, 57]);
export function evidence(): string {
  let s = text(28, 26, "Evidence per measurement", 'font-size="18"') + text(28, 48, "−log₁₀ p", 'class="small mono"');
  for (const v of [0, 2, 4, 6, 8, 10])
    s += `<line x1="50" x2="692" y1="${evidenceY(v)}" y2="${evidenceY(v)}" class="grid"/>` + text(40, evidenceY(v) + 4, String(v), 'text-anchor="end" class="small mono"');
  const line = evidenceY(j.agent_threshold);
  s += `<line x1="50" x2="692" y1="${line}" y2="${line}" stroke="${C.muted}" stroke-width="1.25" stroke-dasharray="5,4"/>` +
    text(690, line - 9, "Agent's line · Bonferroni 5%", 'text-anchor="end" class="quiet"');
  s += groupRules((id) => X(id)!, X.bandwidth(), 366, 3);
  const prev: Record<string, number> = {};
  analyses.forEach((a, k) => {
    const ev = a.evidence as Record<string, number>;
    const list = (a.ranking ?? []) as string[];
    const repeat = k > 0 && JSON.stringify(list) === JSON.stringify(analyses[k - 1].ranking);
    s += `<g class="act-frame" data-frame="${k}"${k === analyses.length - 1 ? " data-final" : ""}>` +
      text(692, 26, `${a.patients} patients · $${money(a.spent)}`, 'text-anchor="end" class="frame-tag mono" font-size="14"') +
      text(692, 46, `List  [${list.join(", ")}]${repeat ? "  repeats → stop" : ""}`, `text-anchor="end" class="frame-tag mono${repeat ? "" : " quiet"}" font-size="13"`);
    ordered.forEach((f) => {
      const value = ev[f.id];
      const listed = list.includes(f.id);
      const cx = X(f.id)! + X.bandwidth() / 2;
      const y = evidenceY(value);
      if (k > 0 && Math.abs(prev[f.id] - y) > 3)
        s += `<rect x="${cx - 4}" y="${prev[f.id] - 4}" width="8" height="8" class="pt-ghost" fill="none" stroke="${C.faint}" stroke-opacity=".7"/>` +
          `<line x1="${cx}" x2="${cx}" y1="${prev[f.id] + (y < prev[f.id] ? -5 : 5)}" y2="${y + (y < prev[f.id] ? 5 : -5)}" class="pt-ghost" stroke="${C.faint}" stroke-opacity=".45"/>`;
      s += `<rect x="${cx - 4}" y="${y - 4}" width="8" height="8" class="pt ${listed ? "pt--listed" : f.timing === "post_outcome" ? "pt--post" : "pt--a"}" data-f="${f.id}" ${listed ? `fill="${C.signal}"` : f.timing === "post_outcome" ? `fill="none" stroke="${C.blue}" stroke-width="1.5"` : `fill="${C.blue}"`}><title>${f.id}: −log10 p = ${value}</title></rect>`;
      if (listed) s += text(cx + 10, y + 5, f.id, 'class="pt-label mono" font-size="13"');
      prev[f.id] = y;
    });
    s += "</g>";
  });
  s += text(28, 440, "■ before the outcome", 'class="small"') + text(200, 440, "□ after the outcome: cannot count", 'class="small"') +
    text(692, 440, "red: over the agent's line", 'text-anchor="end" class="small"');
  return pane(
    "act",
    svg(s, `Association evidence for all ${ordered.length} measurements after the agent bought ${analyses.map((a) => a.patients).join(", ")} patients`),
    "Each square is one measurement, in fixed position. Outlines trace where it stood at the previous purchase.",
  );
}

/* ------------------------------------------------------------------ 5 · scoring */

export function scoring(): string {
  const a = j.scoring.agent;
  const link = linkHorizontal<{ source: [number, number]; target: [number, number] }, [number, number]>().x((d) => d[0]).y((d) => d[1]);
  const chip = (x: number, y: number, id: string, stroke = C.rule, w = 104) =>
    `<rect x="${x}" y="${y}" width="${w}" height="40" rx="2" fill="${C.well}" stroke="${stroke}"/>` + text(x + w / 2, y + 26, id, 'text-anchor="middle" class="mono"');
  const gate = (x: number, label: string, sub: string) =>
    `<line x1="${x}" x2="${x}" y1="64" y2="186" stroke="${C.rule}" stroke-dasharray="2 4"/>` + text(x, 44, label, 'text-anchor="middle" class="quiet"') + text(x, 206, sub, 'text-anchor="middle" class="small"');
  let s = "";
  // k0: the submission.
  s += `<g data-k="0">${text(28, 44, "Submitted", 'class="quiet"')}${chip(28, 78, a.listed[0])}${chip(28, 132, a.listed[1] ?? "")}</g>`;
  // k1: equivalents count once.
  s += `<g data-k="1">${gate(196, "Count equivalents once", "same cluster")}<path d="${link({ source: [132, 98], target: [244, 125] })}" stroke="${C.muted}" fill="none"/><path d="${link({ source: [132, 152], target: [244, 125] })}" stroke="${C.muted}" fill="none"/>${chip(244, 105, a.representatives[0])}</g>`;
  // k2: only the top R count.
  s += `<g data-k="2">${gate(400, `Keep the top R = ${j.key.depth}`, "recoverable causes")}<path d="M348 125 H450" stroke="${C.muted}"/>${chip(450, 105, a.top_r[0])}</g>`;
  // k3: compare with the key.
  s += `<g data-k="3">${gate(600, "Answer key", "held by the scorer")}<path d="M554 125 H588" stroke="${C.muted}"/><rect x="588" y="96" width="104" height="58" rx="2" fill="url(#key-hatch)" stroke="${C.rule}"/>${text(640, 122, `${truth} · ${twin}`, 'text-anchor="middle" class="mono small"')}${text(640, 143, a.raw >= 1 ? "match" : "no match", 'text-anchor="middle"')}</g>`;
  // k4: chance correction on a number line.
  const nl = scaleLinear().domain([0, 1]).range([330, 692]);
  s += `<g data-k="4"><line x1="28" x2="692" y1="238" y2="238" class="rule"/>${text(28, 270, "Correct for chance", 'class="quiet"')}` +
    `<line x1="${nl(0)}" x2="${nl(1)}" y1="296" y2="296" stroke="${C.faint}"/>` +
    [0, 1].map((v) => `<line x1="${nl(v)}" x2="${nl(v)}" y1="290" y2="302" stroke="${C.faint}"/>` + text(nl(v), 320, String(v), 'text-anchor="middle" class="small mono"')).join("") +
    `<rect x="${nl(a.chance) - 4}" y="292" width="8" height="8" fill="none" stroke="${C.muted}"/>` + text(nl(a.chance), 282, `random list ${a.chance.toFixed(2)}`, 'text-anchor="middle" class="small"') +
    `<rect x="${nl(a.raw) - 5}" y="291" width="10" height="10" fill="${C.ink}"/>` + text(nl(a.raw), 282, `this list ${a.raw.toFixed(2)}`, 'text-anchor="end" class="small"') +
    text(28, 300, `(${a.raw.toFixed(2)} − ${a.chance.toFixed(2)}) ÷ (1 − ${a.chance.toFixed(2)})`, 'class="small mono"') + text(28, 326, `Find = ${a.find.toFixed(2)}`, 'class="mono" font-size="17"') + `</g>`;
  // k5: post-outcome information.
  s += `<g data-k="5"><line x1="28" x2="692" y1="348" y2="348" class="rule"/>${text(28, 377, "Post-outcome measurements listed", 'class="quiet"')}${text(692, 377, a.leaked ? "yes: Find becomes 0" : "none", 'text-anchor="end" class="mono"')}</g>`;
  // k6: cost against the reference study.
  const spent = analyses.at(-1)!.spent;
  const cost = scaleLinear().domain([0, j.card.budget]).range([196, 692]);
  s += `<g data-k="6">${text(28, 420, "Cost", 'class="quiet"')}<rect x="196" y="410" width="496" height="8" fill="${C.well}"/><rect x="196" y="410" width="${cost(spent) - 196}" height="8" fill="${C.blue}"/>` +
    `<line x1="${cost(j.key.reference_cost)}" x2="${cost(j.key.reference_cost)}" y1="402" y2="426" stroke="${C.ink}" stroke-width="1.5"/>` +
    text(cost(spent) - 6, 446, `$${money(spent)} spent`, 'text-anchor="end" class="small mono"') + text(cost(j.key.reference_cost) + 6, 446, `$${money(j.key.reference_cost)} reference study`, 'class="small mono"') +
    text(692, 398, `${Math.round((100 * spent) / j.key.reference_cost)}% of the reference study`, 'text-anchor="end" class="small"') + `</g>`;
  s = `<defs><pattern id="key-hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)"><rect width="8" height="8" fill="${C.raised}"/><line x1="0" y1="0" x2="0" y2="8" stroke="${C.ink}" stroke-opacity=".08" stroke-width="3"/></pattern></defs>` + s;

  // k7: the same scorer, run on other answers to this world.
  const cases = [
    { key: "agent", label: "The agent's answer" },
    { key: "substitute_only", label: "The near-duplicate alone" },
    { key: "with_leak", label: "Add a post-outcome measurement" },
    { key: "empty", label: "Return nothing" },
    { key: "everything", label: "List everything, in random order" },
  ] as const;
  const cols = [{ x: 350, label: "Matches key" }, { x: 445, label: "No leak" }, { x: 540, label: "Lists something" }];
  let alt = text(28, 34, "Same world, other answers", 'font-size="18"') + cols.map((c) => text(c.x, 70, c.label, 'text-anchor="middle" class="small"')).join("") + text(692, 70, "Find", 'text-anchor="end" class="small"');
  const mark = (x: number, y: number, pass: boolean) =>
    pass ? `<rect x="${x - 6}" y="${y - 6}" width="12" height="12" fill="${C.ink}"/>` : `<g stroke="${C.faint}" stroke-width="1.5"><line x1="${x - 5}" y1="${y - 5}" x2="${x + 5}" y2="${y + 5}"/><line x1="${x + 5}" y1="${y - 5}" x2="${x - 5}" y2="${y + 5}"/></g>`;
  cases.forEach((c, i) => {
    const r = j.scoring[c.key] as (typeof j.scoring)["agent"] & { lucky_share?: number; expected_find_signed?: number };
    const y = 108 + i * 50;
    const listed = c.key === "everything" ? `all ${r.listed.length} baseline measurements` : r.listed.length ? r.listed.join(", ") : "[ ]";
    alt += `<g class="result-row"><line x1="28" x2="692" y1="${y - 26}" y2="${y - 26}" class="rule"/>${text(28, y - 2, c.label)}${text(28, y + 16, listed, 'class="small mono"')}`;
    if (c.key === "everything") alt += text(350, y + 4, `${Math.round((r.lucky_share ?? 0) * 100)}% of orders`, 'text-anchor="middle" class="small"');
    else alt += mark(350, y, r.raw >= 1);
    alt += mark(445, y, !r.leaked) + mark(540, y, !r.abstained);
    const mean = (r.expected_find_signed ?? 0).toFixed(2).replace("-", "−");
    alt += text(692, y + 5, c.key === "everything" ? `${mean} mean` : r.find.toFixed(2), 'text-anchor="end" class="mono"') + `</g>`;
  });
  alt += `<line x1="28" x2="692" y1="${108 + cases.length * 50 - 26}" y2="${108 + cases.length * 50 - 26}" class="rule"/>`;
  alt += text(28, 438, "On a world with no signal, only “return nothing” earns Restraint.", 'class="quiet"');
  return pane(
    "score",
    `<div class="score-flow">${svg(s, `Scoring the submission ${a.listed.join(", ")}: equivalents merge to ${a.representatives[0]}, the top R match the key, chance correction gives Find ${a.find.toFixed(2)}, no post-outcome measurement, cost below the reference study`)}</div>` +
      `<div class="score-alternatives" data-k="7">${svg(alt, "The same scorer applied to five other answers to this world, showing which check each one fails")}</div>`,
    "Find is credit for the planted cause, corrected for what a random list of the same kinds of measurement would earn.",
  );
}

/* ------------------------------------------------------------------ exports */

/** Standalone SVG downloads: each figure in its final state. */
export function exportFigures(): Record<string, string> {
  const figures: Record<string, [string, number]> = {
    cohort: [cohort(), 0],
    recovery: [recovery(), 0],
    threshold: [nulls(), 0],
    evidence: [evidence(), 0],
    scoring: [scoring(), 0],
    answers: [scoring(), 1],
  };
  return Object.fromEntries(
    Object.entries(figures).map(([name, [markup, index]]) => {
      const root = new JSDOM(markup).window.document.querySelectorAll("svg")[index];
      root.querySelectorAll(".act-frame:not([data-final]), .pt-ghost").forEach((e) => e.remove());
      root.querySelectorAll<SVGGElement>(".r").forEach((r) => {
        const dy = r.getAttribute("style")?.match(/--dy:([-\d.]+)px/)?.[1];
        if (dy) r.setAttribute("transform", `translate(0 ${dy})`);
        r.removeAttribute("style");
      });
      return [name, root.outerHTML];
    }),
  );
}
