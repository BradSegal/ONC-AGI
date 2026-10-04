/**
 * "What makes it hard": one figure per mechanism, each showing what a naive shortcut does and
 * what the analysis the mechanism calls for finds. Every mark comes from src/data/failures.json:
 * the leak and no-signal panels are real toy worlds scored by the real scorer; the others are
 * seeded synthetic constructions (the provenance is recorded in failures.json, not shown on the page).
 */
import { scaleBand, scaleLinear, scaleSqrt } from "d3-scale";
import failures from "../src/data/failures.json" with { type: "json" };
import { C, esc } from "./visuals";

type Col = { id: string; type: string; post: boolean; role: string };
type Case = (typeof failures)[number] & {
  cols?: Col[];
  evidence?: Record<string, number>;
  adjusted?: Record<string, number>;
  line?: number;
  shortcut?: { ranking: string[]; find?: number; leaked?: boolean; claims?: boolean };
  proper?: { ranking: string[]; find?: number; leaked?: boolean; restrained?: boolean };
  joint?: { pair: string[]; evidence: number };
  scatter?: { x: number[]; y?: number[]; batch?: number[]; outcome: number[] };
  pooled?: { evidence: number; sign: number };
  within?: { evidence: number; sign: number };
};
const cases = failures as Case[];
const get = (key: string) => cases.find((c) => c.key === key)!;

const W = 560;
const H = 330;
const SANS = "'Archivo Variable', Archivo, system-ui, sans-serif";
const MONO = "'Azeret Mono Variable', 'Azeret Mono', ui-monospace, monospace";
const STYLE =
  `.fx text{fill:${C.ink};font-family:${SANS};font-size:14px}.fx .q{fill:${C.muted};font-size:13px}.fx .s{fill:${C.muted};font-size:12px}` +
  `.fx .m{font-family:${MONO}}.fx .r{fill:${C.red}}.fx .b{fill:${C.blue}}.fx .grid{stroke:${C.faint};stroke-opacity:.16}`;
const t = (x: number, y: number, s: string, a = "") => `<text x="${x}" y="${y}" ${a}>${esc(s)}</text>`;
const svg = (body: string, label: string) =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" class="fx" role="img" aria-label="${esc(label)}"><style>${STYLE}</style><rect width="${W}" height="${H}" fill="${C.paper}"/>${body}</svg>`;

/** Evidence bars for a set of measurements, with the multiple-testing line and named marks. */
function bars(cs: Case, opts: { values?: Record<string, number>; ghost?: Record<string, number>; mark: Record<string, { text: string; cls: string }>; max?: number }) {
  const cols = cs.cols!;
  const values = opts.values ?? cs.evidence!;
  const max = opts.max ?? Math.max(...Object.values(values), ...Object.values(opts.ghost ?? {}), cs.line! * 1.4);
  const x = scaleBand().domain(cols.map((c) => c.id)).range([56, W - 18]).padding(0.28);
  // square-root scale: strong and weak evidence share one readable axis
  const y = scaleSqrt().domain([0, max]).range([H - 58, 66]);
  let s = "";
  for (const v of [1, 2, 5, 10, 20, 40].filter((v) => v <= max)) s += `<line x1="56" x2="${W - 18}" y1="${y(v)}" y2="${y(v)}" class="grid"/>` + t(48, y(v) + 4, String(v), 'text-anchor="end" class="s m"');
  s += `<text x="16" y="18" class="s">Evidence, −log<tspan baseline-shift="sub" font-size="9">10</tspan> p</text>`;
  for (const c of cols) {
    const v = values[c.id];
    const top = y(v);
    const xx = x(c.id)!;
    const crossing = v > cs.line! && !c.post;
    const fill = c.post ? "none" : crossing ? C.signal : C.blue;
    if (opts.ghost) {
      const g = opts.ghost[c.id];
      s += `<rect x="${xx}" y="${y(g)}" width="${x.bandwidth()}" height="${y(0) - y(g)}" fill="none" stroke="${C.faint}" stroke-dasharray="3 2"/>`;
    }
    s += `<rect x="${xx}" y="${top}" width="${x.bandwidth()}" height="${Math.max(1, y(0) - top)}" fill="${fill}" ${c.post ? `stroke="${C.blue}" stroke-width="1.5"` : ""}/>`;
    s += `<text transform="translate(${xx + x.bandwidth() / 2 + 3} ${H - 50}) rotate(-60)" text-anchor="end" class="s m" font-size="10.5">${esc(c.id)}</text>`;
    const mk = opts.mark[c.id];
    if (mk) {
      // a 1px stem from the bar (or its dashed outline), then the id in mono and any note in the reading face
      const at = opts.ghost ? Math.min(top, y(opts.ghost[c.id])) : top;
      const cx = xx + x.bandwidth() / 2;
      const anchor = cx > W * 0.68 ? "end" : cx < W * 0.3 ? "start" : "middle";
      const [id, ...note] = mk.text.split(" · ");
      const label = `<tspan class="m">${esc(id)}</tspan>${note.length ? `<tspan> · ${esc(note.join(" · "))}</tspan>` : ""}`;
      s += `<path d="M${cx} ${at - 4} V${at - 16}" stroke="${mk.cls === "r" ? C.red : C.ink}"/><text x="${anchor === "end" ? cx + 6 : anchor === "start" ? cx - 6 : cx}" y="${at - 21}" text-anchor="${anchor}" class="${mk.cls}" font-size="12.5">${label}</text>`;
    }
  }
  const ly = y(cs.line!);
  // the line's label goes on whichever side has no bar rising through it
  const tall = cols.filter((c) => values[c.id] > cs.line!).map((c) => x(c.id)! + x.bandwidth() / 2);
  const clear = (a: number, b: number) => !tall.some((xx) => xx > a - 8 && xx < b + 8);
  const spots: [number, string][] = [[60, "start"], [W - 18, "end"], [W / 2, "middle"]];
  const ranges: Record<string, [number, number]> = { start: [60, 200], end: [W - 160, W - 18], middle: [W / 2 - 70, W / 2 + 70] };
  const [lx, anchor] = spots.find(([, a]) => clear(...ranges[a])) ?? spots[2];
  s += `<line x1="56" x2="${W - 18}" y1="${ly}" y2="${ly}" stroke="${C.signal}" stroke-width="1.25"/>` + t(lx, ly - 6, "multiple-testing line", `text-anchor="${anchor}" class="s r"`);
  return s;
}

function scatterPanel(xs: number[], ys: number[], outcome: number[], area: [number, number, number, number], domain: [number, number, number, number]) {
  const [x0, y0, x1, y1] = area;
  const x = scaleLinear().domain([domain[0], domain[1]]).range([x0, x1]).clamp(true);
  const y = scaleLinear().domain([domain[2], domain[3]]).range([y1, y0]).clamp(true);
  let s = `<rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" fill="none" stroke="${C.rule}"/>`;
  xs.forEach((v, i) => {
    const filled = outcome[i] === 1;
    // events in Chart Ink, no-events in the lower band blue; hollow is reserved for post-outcome
    s += `<rect x="${(x(v) - 2.5).toFixed(1)}" y="${(y(ys[i]) - 2.5).toFixed(1)}" width="5" height="5" fill="${filled ? C.ink : C.deep}"/>`;
  });
  return { s, x, y };
}

export function failureFigures(): { key: string; svg: string }[] {
  const out: { key: string; svg: string }[] = [];

  const leak = get("leak");
  const leakId = leak.shortcut!.ranking[0];
  const truth = leak.proper!.ranking[0];
  out.push({
    key: "leak",
    svg: svg(
      bars(leak, { mark: { [leakId]: { text: `${leakId} · after the outcome`, cls: "b" }, [truth]: { text: truth, cls: "r" } } }),
      `Evidence for each measurement; the post-outcome measurement ${leakId} towers over the planted cause ${truth}`,
    ),
  });

  const nul = get("no_signal");
  const named = nul.shortcut!.ranking[0];
  out.push({
    key: "no_signal",
    svg: svg(
      bars(nul, { mark: { [named]: { text: `${named} · strongest, still below`, cls: "b" } }, max: 8 }),
      `Evidence for each measurement in a world with no signal; all stay below the line, the strongest is ${named}`,
    ),
  });

  const it = get("interaction");
  {
    const sc = it.scatter!;
    const p = scatterPanel(sc.x, sc.y!, sc.outcome, [40, 36, 290, 286], [-2.6, 2.6, -2.6, 2.6]);
    let s = p.s + `<line x1="${p.x(0)}" x2="${p.x(0)}" y1="36" y2="286" class="grid"/><line x1="40" x2="290" y1="${p.y(0)}" y2="${p.y(0)}" class="grid"/>`;
    s += t(165, 310, "a1 →", 'text-anchor="middle" class="s m"') + t(22, 161, "a2", 'text-anchor="middle" class="s m" transform="rotate(-90 22 161)"');
    s += `<rect x="40" y="15" width="8" height="8" fill="${C.ink}"/>` + t(52, 23, "event", 'class="s"') + `<rect x="96" y="15" width="8" height="8" fill="${C.deep}"/>` + t(108, 23, "no event", 'class="s"');
    // marginal versus joint evidence
    const bx = scaleLinear().domain([0, Math.max(it.joint!.evidence, 10)]).range([400, W - 46]);
    const row = (y: number, label: string, v: number, cls: string) =>
      t(392, y + 10, label, 'text-anchor="end" class="q m"') + `<rect x="400" y="${y}" width="${Math.max(1.5, bx(v) - 400)}" height="12" fill="${cls === "r" ? C.signal : C.blue}"/>` + t(Math.max(bx(v), v < it.line! ? bx(it.line!) : 402) + 6, y + 10, v.toFixed(1), 'class="s m"');
    s += t(330, 56, "Evidence (−log₁₀ p)", 'class="q"');
    s += row(92, "a1 alone", it.evidence!.a1 ?? 0, "b") + row(140, "a2 alone", it.evidence!.a2 ?? 0, "b") + row(188, "a1 × a2", it.joint!.evidence, "r");
    s += `<line x1="${bx(it.line!)}" x2="${bx(it.line!)}" y1="76" y2="212" stroke="${C.signal}" stroke-width="1.25"/>` + t(bx(it.line!) + 4, 228, "multiple-testing line", 'class="s r"');
    out.push({ key: "interaction", svg: svg(s, "Events fall in opposite quadrants of a1 and a2: neither predicts alone, their product does") });
  }

  const cf = get("confounder");
  out.push({
    key: "confounder",
    svg: svg(
      bars(cf, {
        values: cf.adjusted,
        ghost: cf.evidence,
        mark: { x7: { text: "x7 · acts alone", cls: "r" }, [cf.shortcut!.ranking[0]]: { text: `${cf.shortcut!.ranking[0]} · lifted by the subtype`, cls: "q" } },
      }),
      "Dashed outlines: evidence with subtype ignored, where five subtype-lifted genes lead; solid bars: within each subtype, where only x7 remains",
    ),
  });

  const wt = get("wrong_data_type");
  {
    const s = bars(wt, { mark: { c1: { text: "c1 · copy number", cls: "r" }, e1: { text: "e1 · switched on by c1", cls: "q" } } });
    const x = scaleBand().domain(wt.cols!.map((c) => c.id)).range([56, W - 18]).padding(0.28);
    const cnEnd = x("c4")! + x.bandwidth();
    const group = `<path d="M60 ${H - 14} H${cnEnd}" stroke="${C.rule}"/>` + t((60 + cnEnd) / 2, H - 2, "Copy number", 'text-anchor="middle" class="s"') + `<path d="M${cnEnd + 8} ${H - 14} H${W - 20}" stroke="${C.rule}"/>` + t((cnEnd + W) / 2, H - 2, "Gene expression", 'text-anchor="middle" class="s"');
    out.push({ key: "wrong_data_type", svg: svg(s + group, "Copy-number change c1 drives the outcome; the expression gene it switches on, e1, is the strongest expression signal") });
  }

  const sp = get("simpson");
  {
    const sc = sp.scatter!;
    const jitter = sc.batch!.map((b, i) => b * 1 + (((i * 7919) % 100) / 100 - 0.5) * 0.7);
    const p = scatterPanel(sc.x, jitter, sc.outcome, [40, 40, W - 20, 250], [-3, 6, -0.6, 1.6]);
    let s = p.s;
    s += t(46, p.y(0) - 26, "Batch A", 'class="q"') + t(46, p.y(1) - 26, "Batch B", 'class="q"');
    s += t((40 + W - 20) / 2, 270, "marker level →", 'text-anchor="middle" class="s m"');
    const dir = (v: number) => (v > 0 ? "higher marker, more events" : "higher marker, fewer events");
    s += t(40, 300, `Pooled: ${dir(sp.pooled!.sign)} (−log₁₀ p ${sp.pooled!.evidence.toFixed(1)})`, 'class="q"');
    s += t(40, 320, `Within each batch: ${dir(sp.within!.sign)} (${sp.within!.evidence.toFixed(1)})`, 'class="r"');
    s += `<rect x="40" y="19" width="8" height="8" fill="${C.ink}"/>` + t(52, 27, "event", 'class="s"') + `<rect x="96" y="19" width="8" height="8" fill="${C.deep}"/>` + t(108, 27, "no event", 'class="s"');
    out.push({ key: "simpson", svg: svg(s, "Within each batch the marker raises the event rate, but batch B has a higher marker and fewer events, so the pooled association reverses") });
  }
  return out;
}

export const FAILURE_COPY: Record<string, { skill: string; mechanism: string; shortcut: string; proper: string }> = (() => {
  const leak = get("leak");
  const nul = get("no_signal");
  const cf = get("confounder");
  const wt = get("wrong_data_type");
  return {
    leak: {
      skill: "Respect time",
      mechanism: "A measurement taken after the outcome tracks it closely.",
      shortcut: `Rank everything by association: ${leak.shortcut!.ranking[0]} comes first, a leak, so Find falls to ${leak.shortcut!.find!.toFixed(0)}.`,
      proper: `Rank only what was measured before the outcome: ${leak.proper!.ranking[0]}, Find ${leak.proper!.find!.toFixed(0)}.`,
    },
    no_signal: {
      skill: "Know when there is nothing",
      mechanism: "A twin of a real world with every effect removed.",
      shortcut: `Always name the strongest measurement: ${nul.shortcut!.ranking[0]}, a false claim.`,
      proper: "Nothing crosses the line, so return nothing: the only answer that earns Restraint.",
    },
    interaction: {
      skill: "Look for joint effects",
      mechanism: "The outcome follows two measurements together; neither matters alone.",
      shortcut: "Screen one measurement at a time: nothing crosses the line.",
      proper: "Test the pair jointly: a1 × a2 stands far above it.",
    },
    confounder: {
      skill: "Separate cause from confounding",
      mechanism: "A clinical subtype raises risk and lifts five genes.",
      shortcut: `Screen genes on their own: ${cf.shortcut!.ranking.join(", ")} lead, all lifted by the subtype.`,
      proper: `Compare within each subtype: only ${cf.proper!.ranking.join(", ")} remains.`,
    },
    wrong_data_type: {
      skill: "Look across data types",
      mechanism: "A copy-number change drives the outcome and switches on a gene.",
      shortcut: `Read only gene expression: ${wt.shortcut!.ranking[0]}, the gene it switches on.`,
      proper: `Read every data type: ${wt.proper!.ranking[0]}, the copy-number cause.`,
    },
    simpson: {
      skill: "Respect how the data were pooled",
      mechanism: "Two batches differ in both marker level and event rate.",
      shortcut: "Pool the batches: the marker looks protective.",
      proper: "Compare within each batch: it raises risk in both.",
    },
  };
})();

/** Where each panel's point lies, as a fraction of its width: phones open the figure there. */
const FOCUS: Record<string, number> = { leak: 0.85, interaction: 0.8, confounder: 0.45, wrong_data_type: 0.4, simpson: 0.5, no_signal: 0.85 };

/** How to read each panel, printed beside it. */
const KEYS: Record<string, string> = {
  leak: "Hollow bars were measured after the outcome: they cannot count, even above the line.",
  no_signal: "Hollow bars were measured after the outcome.",
  confounder: "Dashed outlines: each gene with the subtype ignored. Solid bars: compared within each subtype.",
};

export function failuresMarkup(): string {
  const order = ["leak", "confounder", "interaction", "wrong_data_type", "simpson", "no_signal"];
  const figs = new Map(failureFigures().map((f) => [f.key, f.svg]));
  return order
    .map((key) => {
      const c = get(key);
      const copy = FAILURE_COPY[key];
      const source = `${c.n} patients.`;
      return `<figure class="hard" data-hard="${key}">
        <h3>${esc(copy.skill)}</h3>
        <p class="hard__mechanism">${esc(copy.mechanism)}</p>
        ${KEYS[key] ? `<p class="hard__key">${KEYS[key]}</p>` : ""}
        <div class="hard__figure" data-focus="${FOCUS[key] ?? 0.5}">${figs.get(key)}</div>
        <dl class="hard__verdict">
          <div><dt>Shortcut</dt><dd>${esc(copy.shortcut)}</dd></div>
          <div><dt>Analysis</dt><dd>${esc(copy.proper)}</dd></div>
        </dl>
        <figcaption class="caption">${source}</figcaption>
      </figure>`;
    })
    .join("");
}
