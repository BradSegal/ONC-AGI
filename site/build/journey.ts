/**
 * Build-time panes of the journey replay. Each pane is a complete static figure (SVG or
 * HTML) baked from journey.json and oracle.json, so the replay reads without JavaScript
 * and under reduced motion; the client only moves between states that already exist.
 */
import journey from "../src/data/journey.json" with { type: "json" };
import oracle from "../src/data/oracle.json" with { type: "json" };

export const esc = (s: string) =>
  s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] as string);
const usd = (v: number) => `${Math.round(v).toLocaleString("en-GB")}`;
const f2 = (v: number) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);

type Feature = { id: string; type: string; timing: string; price: number };
const card = journey.card as { n_pool: number; budget: number; recruit_price: number; features: Feature[] };
const key = journey.key as {
  truth: string;
  equivalent: string[];
  reject_set: string[];
  depth: number;
  reference_cost: number;
};
const steps = journey.steps as {
  kind: string;
  count?: number;
  patients: number;
  spent: number;
  evidence?: Record<string, number>;
  ranking?: string[];
}[];
const analyses = steps.filter((s) => s.kind === "analyse");
const submission = (steps.find((s) => s.kind === "submit")?.ranking ?? []) as string[];
const TYPE_ORDER = ["expression", "copy_number", "protein", "clinical", "derived", "lab"];
const TYPE_LABEL: Record<string, string> = {
  expression: "expression",
  copy_number: "copy no.",
  protein: "protein",
  clinical: "clinical",
  derived: "derived",
  lab: "lab",
  post_outcome: "after outcome",
};

/** Features in plot order: grouped by data type like chromosome bands, post-outcome last. */
export const plotOrder: Feature[] = [
  ...TYPE_ORDER.flatMap((t) => card.features.filter((f) => f.type === t && f.timing === "baseline")),
  ...card.features.filter((f) => f.timing === "post_outcome"),
];
const groupOf = (f: Feature) => (f.timing === "post_outcome" ? "post_outcome" : f.type);

/* ----------------------------------------------------------------- build: the world's rows */

export function paneBuild(): string {
  const pool = journey.pool as { z: number[][]; outcome: number[] };
  const W = 640;
  const H = 400;
  const rows = 60; // every 4th patient for the static figure; the canvas draws all 240
  const cols = plotOrder.map((f) => card.features.findIndex((g) => g.id === f.id));
  const left = 26;
  const top = 34;
  const cw = (W - left - 8) / cols.length;
  const rh = (H - top - 30) / rows;
  const shade = (z: number) => {
    const t = Math.max(0, Math.min(1, (z + 2.5) / 5));
    return `rgb(${Math.round(30 + t * 97)},${Math.round(48 + t * 119)},${Math.round(72 + t * 129)})`;
  };
  let cells = "";
  for (let r = 0; r < rows; r++) {
    const i = r * 4;
    cells += `<g class="r">`;
    cells += `<rect x="4" y="${(top + r * rh).toFixed(1)}" width="14" height="${(rh - 0.6).toFixed(1)}" class="oc ${
      pool.outcome[i] ? "y1" : "y0"
    }"/>`;
    cols.forEach((j, c) => {
      cells += `<rect x="${(left + c * cw).toFixed(1)}" y="${(top + r * rh).toFixed(1)}" width="${(cw - 1.2).toFixed(
        1,
      )}" height="${(rh - 0.6).toFixed(1)}" fill="${shade(pool.z[i][j])}"/>`;
    });
    cells += `</g>`;
  }
  const heads = plotOrder
    .map((f, c) => {
      const cls = f.id === key.truth ? "is-truth" : key.equivalent.includes(f.id) ? "is-twin" : groupOf(f) === "post_outcome" ? "is-post" : "";
      return `<text x="${(left + c * cw + cw / 2).toFixed(1)}" y="26" class="colhead ${cls}" data-col="${esc(f.id)}">${esc(
        f.id,
      )}</text>`;
    })
    .join("");
  const marks = plotOrder
    .map((f, c) =>
      f.id === key.truth || key.equivalent.includes(f.id) || groupOf(f) === "post_outcome"
        ? `<rect x="${(left + c * cw - 0.6).toFixed(1)}" y="${top - 2}" width="${(cw + 0.2).toFixed(1)}" height="${(
            rows * rh +
            3
          ).toFixed(1)}" class="colmark ${f.id === key.truth ? "is-truth" : groupOf(f) === "post_outcome" ? "is-post" : "is-twin"}"/>`
        : "",
    )
    .join("");
  return `<figure class="pane pane--build" data-pane="build">
  <svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="pane-build-cap">
    <text x="${left}" y="12" class="axis-label">${cols.length} measurements, ${card.n_pool} patients (every 4th shown)</text>
    ${heads}${cells}${marks}
    <text x="4" y="${H - 8}" class="axis-label">each row is one real patient<tspan class="oc-label"> · left strip: outcome (white = event)</tspan></text>
  </svg>
  <figcaption id="pane-build-cap" class="caption">The journey world, <b>${esc(journey.world)}</b>: ${card.n_pool} patients × ${
    card.features.length
  } measurements.<span class="from-1"> The planted measurement and its near-duplicate are outlined.</span><span class="from-3"> The three post-outcome columns are marked.</span></figcaption>
</figure>`;
}

/* ----------------------------------------------------------------- certify: the oracle */

export function paneCertify(): string {
  const W = 640;
  const H = 400;
  const pad = { l: 46, r: 16, t: 24, b: 40 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const X = (n: number) => pad.l + ((n - 30) / 210) * pw;
  const Y = (r: number) => pad.t + ph - r * ph;
  const series = [
    ["unrelated", "s-unrelated", oracle.features.unrelated],
    ["stand_in", "s-twin", oracle.features.stand_in],
    ["cause", "s-cause", oracle.features.cause],
  ] as const;
  const curves = series
    .map(([k, cls, id]) => {
      const pts = (oracle.curves as Record<string, { n: number; rate: number; low: number; high: number }[]>)[k];
      const band = `${pts.map((q) => `${X(q.n).toFixed(1)},${Y(q.high).toFixed(1)}`).join(" ")} ${[...pts]
        .reverse()
        .map((q) => `${X(q.n).toFixed(1)},${Y(q.low).toFixed(1)}`)
        .join(" ")}`;
      const line = pts.map((q) => `${X(q.n).toFixed(1)},${Y(q.rate).toFixed(1)}`).join(" ");
      const last = pts[pts.length - 1];
      return `<g class="curve ${cls}"><polygon points="${band}" class="ci-band"/><polyline points="${line}" class="ci-line"/><text x="${X(
        last.n,
      ).toFixed(1)}" y="${(Y(last.rate) + (k === "stand_in" ? 16 : k === "cause" ? -8 : -8)).toFixed(
        1,
      )}" text-anchor="end" class="curve-label">${esc(id ?? k)}${k === "cause" ? " (planted)" : k === "stand_in" ? " (near-duplicate)" : " (unrelated)"}</text></g>`;
    })
    .join("");
  const ticks = [0, 0.2, 0.5, 0.8, 1]
    .map((r) => `<text x="${pad.l - 8}" y="${(Y(r) + 4).toFixed(1)}" text-anchor="end" class="axis-label">${r.toFixed(1)}</text>`)
    .join("");
  const nticks = [30, 60, 100, 160, 240]
    .map((n) => `<text x="${X(n).toFixed(1)}" y="${H - 22}" text-anchor="middle" class="axis-label">${n}</text>`)
    .join("");
  return `<figure class="pane pane--certify" data-pane="certify">
  <svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="pane-certify-cap">
    <rect x="${pad.l}" y="${Y(1)}" width="${pw}" height="${(Y(0.8) - Y(1)).toFixed(1)}" class="zone zone--ok"/>
    <rect x="${pad.l}" y="${Y(0.8)}" width="${pw}" height="${(Y(0.2) - Y(0.8)).toFixed(1)}" class="zone zone--amb"/>
    <rect x="${pad.l}" y="${Y(0.2)}" width="${pw}" height="${(Y(0) - Y(0.2)).toFixed(1)}" class="zone zone--neutral"/>
    <text x="${W - pad.r - 8}" y="${Y(0.86) + 4}" text-anchor="end" class="zone-label">recoverable</text>
    <text x="${W - pad.r - 8}" y="${Y(0.5) + 4}" text-anchor="end" class="zone-label">ambiguous: the world is rebuilt</text>
    <text x="${pad.l + 8}" y="${Y(0.1) + 4}" class="zone-label">neutral</text>
    ${ticks}${nticks}
    <text x="${pad.l}" y="${H - 6}" class="axis-label">patients in each replayed cohort</text>
    <text x="${pad.l}" y="14" class="axis-label">share of ${oracle.replicates} replays that clear the line</text>
    ${curves}
  </svg>
  <figcaption id="pane-certify-cap" class="caption">Computed on this world: the line is the 95th percentile of the strongest chance association over ${
    oracle.null_draws
  } no-signal outcomes (|z| = ${oracle.threshold.toFixed(2)}). Bands are 95% intervals.</figcaption>
</figure>`;
}

/* ----------------------------------------------------------------- receive: the world card */

export function paneReceive(): string {
  const groups = [...TYPE_ORDER, "post_outcome"]
    .map((g) => {
      const fs = plotOrder.filter((f) => groupOf(f) === g);
      if (!fs.length) return "";
      return `<div class="card-group"><span class="card-group__name">${TYPE_LABEL[g]}</span><span class="card-group__ids">${fs
        .map((f) => `<span class="fid${g === "post_outcome" ? " fid--post" : ""}">${esc(f.id)}</span>`)
        .join(" ")}</span></div>`;
    })
    .join("");
  return `<figure class="pane pane--receive" data-pane="receive">
  <div class="worldcard">
    <div class="worldcard__row"><span>world</span><b>${esc(journey.world)}</b></div>
    <div class="worldcard__row"><span>patients available</span><b>${card.n_pool}</b></div>
    <div class="worldcard__row"><span>outcome</span><b>binary</b></div>
    <div class="worldcard__row"><span>recruit</span><b>${card.recruit_price} USD / patient</b></div>
    <div class="worldcard__row"><span>assay</span><b>0.5 to 8 USD / measurement / patient</b></div>
    <div class="worldcard__row"><span>budget</span><b>${usd(card.budget)} USD (the whole cohort)</b></div>
    <div class="worldcard__groups">${groups}</div>
    <div class="worldcard__locked"><span>answer key</span><b>held by the scorer</b></div>
  </div>
  <figcaption class="caption">What the agent receives: names are fake, so prior knowledge of genes cannot help. Measurements taken after the outcome are marked as such.</figcaption>
</figure>`;
}

/* ----------------------------------------------------------------- act: the agent's evidence */

const ACT = { W: 640, H: 400, pad: { l: 40, r: 12, t: 28, b: 48 }, yMax: 10 };
export const actGeometry = ACT;

export function actX(i: number): number {
  const { W, pad } = ACT;
  const n = plotOrder.length;
  // a gap between data-type groups, like chromosome bands
  let gaps = 0;
  for (let k = 1; k <= i; k++) if (groupOf(plotOrder[k]) !== groupOf(plotOrder[k - 1])) gaps++;
  const totalGaps = plotOrder.reduce((g, f, k) => g + (k && groupOf(f) !== groupOf(plotOrder[k - 1]) ? 1 : 0), 0);
  const unit = (W - pad.l - pad.r) / (n + totalGaps * 0.8);
  return pad.l + unit * (i + gaps * 0.8 + 0.5);
}
export function actY(v: number): number {
  const { H, pad, yMax } = ACT;
  return pad.t + (H - pad.t - pad.b) * (1 - Math.min(v, yMax) / yMax);
}

function actPlot(k: number): string {
  const a = analyses[k];
  const ev = a.evidence as Record<string, number>;
  const listed = new Set(a.ranking ?? []);
  return plotOrder
    .map((f, i) => {
      const v = ev[f.id] ?? 0;
      const crossed = listed.has(f.id);
      const label = crossed ? `<text x="${(actX(i) + 9).toFixed(1)}" y="${(actY(v) + 4).toFixed(1)}" class="pt-label">${esc(f.id)}</text>` : "";
      return `<rect x="${(actX(i) - 4).toFixed(1)}" y="${(actY(v) - 4).toFixed(1)}" width="8" height="8" class="pt ${
        crossed ? "pt--listed" : groupOf(f) === "post_outcome" ? "pt--post" : i % 2 ? "pt--b" : "pt--a"
      }" data-f="${esc(f.id)}"/>${label}`;
    })
    .join("");
}

export function paneAct(): string {
  const { W, H, pad } = ACT;
  const line = actY(journey.agent_threshold);
  const groupLabels: string[] = [];
  let start = 0;
  plotOrder.forEach((f, i) => {
    const next = plotOrder[i + 1];
    if (!next || groupOf(next) !== groupOf(f)) {
      const mid = (actX(start) + actX(i)) / 2;
      // a bracket rule spans each group's points; labels share one baseline beneath
      const y0 = H - pad.b + 8;
      const g = groupOf(f);
      const label =
        g === "post_outcome"
          ? `<tspan x="${mid.toFixed(1)}">after</tspan><tspan x="${mid.toFixed(1)}" dy="12">outcome</tspan>`
          : TYPE_LABEL[g];
      groupLabels.push(
        `<line x1="${(actX(start) - 5).toFixed(1)}" x2="${(actX(i) + 5).toFixed(1)}" y1="${y0}" y2="${y0}" class="group-rule"/>` +
          `<text x="${mid.toFixed(1)}" y="${y0 + 15}" text-anchor="middle" class="axis-label">${label}</text>`,
      );
      start = i + 1;
    }
  });
  const yt = [0, 2, 4, 6, 8, 10]
    .map((v) => `<text x="${pad.l - 8}" y="${(actY(v) + 4).toFixed(1)}" text-anchor="end" class="axis-label">${v}</text>`)
    .join("");
  const frames = analyses
    .map(
      (a, k) =>
        `<g class="act-frame" data-frame="${k}"><text x="${W - pad.r}" y="16" text-anchor="end" class="frame-tag">after ${a.patients} patients · ${usd(
          a.spent,
        )} USD</text>${actPlot(k)}</g>`,
    )
    .join("");
  return `<figure class="pane pane--act" data-pane="act">
  <svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="pane-act-cap">
    ${yt}<text x="8" y="16" class="axis-label">evidence (−log10 p)</text>
    <line x1="${pad.l}" x2="${W - pad.r}" y1="${line.toFixed(1)}" y2="${line.toFixed(1)}" class="agent-line"/>
    <text x="${W - pad.r}" y="${(line - 6).toFixed(1)}" text-anchor="end" class="agent-line-label">the agent's own line: Bonferroni 5%</text>
    ${groupLabels.join("")}
    ${frames}
  </svg>
  <figcaption id="pane-act-cap" class="caption">The agent's evidence for every measurement (Welch t-test) on the patients it has bought so far. Recorded from a real run of the runtime.</figcaption>
</figure>`;
}

/* ----------------------------------------------------------------- submit and score */

type Breakdown = {
  listed: string[];
  representatives: string[];
  top_r: string[];
  raw: number;
  chance: number;
  find: number;
  leaked: boolean;
  abstained: boolean;
  efficiency: number;
  expected_find_signed?: number;
  lucky_share?: number;
};

export function paneScore(): string {
  const s = journey.scoring as Record<string, Breakdown>;
  const a = s.agent;
  const spent = steps[steps.length - 1].spent;
  const chips = (ids: string[], cls = "") => ids.map((i) => `<span class="fid ${cls}">${esc(i)}</span>`).join(" ");
  const merged = a.listed.filter((f) => !a.representatives.includes(f));
  return `<figure class="pane pane--score" data-pane="score">
  <ol class="ledger">
    <li data-k="0"><span class="ledger__q">Submitted</span><span class="ledger__a">${chips(submission)}</span></li>
    <li data-k="1"><span class="ledger__q">One answer per correlated cluster</span><span class="ledger__a">${chips(a.representatives)}${
      merged.length ? ` <span class="note">${merged.map(esc).join(", ")} is a near-duplicate of ${esc(a.representatives[0])}: counted once</span>` : ""
    }</span></li>
    <li data-k="2"><span class="ledger__q">Answers that count (R = ${key.depth})</span><span class="ledger__a">${chips(a.top_r)}</span></li>
    <li data-k="3"><span class="ledger__q">Matches the answer key?</span><span class="ledger__a">yes: ${esc(a.top_r[0] ?? "—")} is in {${[key.truth, ...key.equivalent]
      .map(esc)
      .join(", ")}}, the measurements the oracle cannot tell apart</span></li>
    <li data-k="4"><span class="ledger__q">Against chance</span><span class="ledger__a">a random list of the same kind hits ${f2(a.chance)} of the time · Find = (${f2(
      a.raw,
    )} − ${f2(a.chance)}) ÷ (1 − ${f2(a.chance)}) = <b>${f2(a.find)}</b></span></li>
    <li data-k="5"><span class="ledger__q">Post-outcome measurement listed?</span><span class="ledger__a">${a.leaked ? "yes: world scores 0" : "no"}</span></li>
    <li data-k="6"><span class="ledger__q">Data cost</span><span class="ledger__a">${usd(spent)} USD against a ${usd(
      key.reference_cost,
    )} USD reference study · efficiency <b>${f2(a.efficiency)}</b></span></li>
  </ol>
  <table class="whatif" data-k="7">
    <caption>The same world, other submissions (scored by the same code)</caption>
    <thead><tr><th scope="col">Submission</th><th scope="col">Result</th></tr></thead>
    <tbody>
      <tr><th scope="row">only the near-duplicate ${chips(s.substitute_only.listed)}</th><td>Find ${f2(s.substitute_only.find)}: equivalent answers earn credit</td></tr>
      <tr><th scope="row">the answer plus a post-outcome measurement ${chips(s.with_leak.listed)}</th><td>Find ${f2(s.with_leak.find)}: a leak anywhere zeroes the world</td></tr>
      <tr><th scope="row">nothing <span class="fid">[ ]</span></th><td>No credit, and abstaining on a world with a cause lowers Restraint</td></tr>
      <tr><th scope="row">every measurement, in an arbitrary order</th><td>Right first in ${Math.round((s.everything.lucky_share ?? 0) * 100)}% of orders; expected Find ${f2(
        s.everything.expected_find_signed ?? 0,
      )}; and it never says "nothing"</td></tr>
    </tbody>
  </table>
</figure>`;
}

/* ----------------------------------------------------------------- across worlds */

type WorldRow = { world: string; mechanism: string; mode: string; null: boolean; find: number; restrained: boolean; abstained: boolean; efficiency: number; spent: number };

const MECH_LABEL: Record<string, string> = {
  generating: "Generating",
  stand_in: "Stand-in",
  wrong_data_type: "Other data type",
  observed_confounder: "Confounder",
  leak: "Leak",
  interaction: "Interaction",
  module: "Module",
  neutral_group: "Neutral group",
  no_signal: "No signal",
};

export function paneAcross(): string {
  const across = journey.across as { summary: Record<string, number | null>; worlds: WorldRow[] };
  const sm = across.summary;
  const cell = (w: WorldRow | undefined) => {
    if (!w) return `<td></td>`;
    const ok = w.null ? w.restrained : w.find >= 0.99;
    const partial = !w.null && w.find > 0 && w.find < 0.99;
    const text = w.null ? (w.restrained ? "said nothing" : "claimed") : w.abstained ? "said nothing" : `Find ${f2(w.find)}`;
    const eff = w.efficiency < 0.99 ? ` · efficiency ${f2(w.efficiency)}` : "";
    return `<td class="${ok ? "ok" : partial ? "partial" : "miss"}${w.world === journey.world ? " is-journey" : ""}"><span>${text}${eff}</span></td>`;
  };
  const mechs = [...new Set(across.worlds.map((w) => w.mechanism))];
  const rows = mechs
    .map((m) => {
      const pick = (mode: string) => across.worlds.filter((w) => w.mechanism === m && w.mode === mode);
      const fa = pick("full_access");
      const sq = pick("sequential");
      return fa
        .map((w, i) => `<tr><th scope="row">${MECH_LABEL[m] ?? m}${fa.length > 1 ? ` ${String.fromCharCode(97 + i)}` : ""}</th>${cell(w)}${cell(sq[i])}</tr>`)
        .join("");
    })
    .join("");
  return `<figure class="pane pane--across" data-pane="across">
  <table class="worlds">
    <thead><tr><th scope="col">Mechanism</th><th scope="col">Full access</th><th scope="col">Sequential</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <p class="verdict">Find <b>${f2(sm.find as number)}</b> × Restraint <b>${f2(sm.restraint as number)}</b> = Discovery Score <b>${f2(
    sm.discovery_score as number,
  )}</b> <span class="caption">95% interval ${f2(sm.low as number)} to ${f2(sm.high as number)} · ${sm.n_worlds} worlds</span></p>
</figure>`;
}

export function journeyReplacements(): Record<string, string> {
  const across = journey.across as { summary: Record<string, number | null>; worlds: WorldRow[] };
  const last = steps[steps.length - 1];
  const a0 = analyses[0];
  const sm = across.summary;
  return {
    "<!--@pane-build-->": paneBuild(),
    "<!--@pane-certify-->": paneCertify(),
    "<!--@pane-receive-->": paneReceive(),
    "<!--@pane-act-->": paneAct(),
    "<!--@pane-score-->": paneScore(),
    "<!--@pane-across-->": paneAcross(),
    "@@j-world@@": esc(journey.world),
    "@@j-n@@": String(card.n_pool),
    "@@j-features@@": String(card.features.length),
    "@@j-budget@@": usd(card.budget),
    "@@j-ref@@": usd(key.reference_cost),
    "@@j-spent@@": usd(last.spent),
    "@@j-patients@@": String(last.patients),
    "@@j-truth@@": esc(key.truth),
    "@@j-twin@@": esc(key.equivalent[0] ?? ""),
    "@@j-first-n@@": String(a0.patients),
    "@@j-first-spent@@": usd(a0.spent),
    "@@j-agent-line@@": f2(journey.agent_threshold),
    "@@j-submission@@": submission.map(esc).join(", "),
    "@@j-oracle-thr@@": oracle.threshold.toFixed(2),
    "@@j-oracle-n80@@": String((oracle.curves.cause.find((c) => c.low >= 0.8) ?? { n: "—" }).n),
    "@@j-find@@": f2(sm.find as number),
    "@@j-restraint@@": f2(sm.restraint as number),
    "@@j-ds@@": f2(sm.discovery_score as number),
    "@@j-ds-low@@": f2(sm.low as number),
    "@@j-ds-high@@": f2(sm.high as number),
    "@@j-transport-data@@": esc(
      JSON.stringify({ budget: card.budget, reference: key.reference_cost, cross: analyses.find((a) => (a.ranking ?? []).length)?.spent ?? null, steps: steps.map((s) => ({ kind: s.kind, patients: s.patients, spent: s.spent })) }),
    ),
  };
}
