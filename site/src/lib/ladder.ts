/**
 * The results ladder: every agent on one 0–1 scale with its 95% interval, grouped by
 * kind, the oracle at 1 and the floor at 0 drawn as references. Values come from
 * results.json; the static table in the page carries the same numbers.
 */
import results from "../data/results.json";

type Summary = { discovery_score?: number | null; low?: number | null; high?: number | null };
type Agent = {
  id: string;
  label: string;
  kind: "frontier" | "reference" | "baseline" | "cheater";
  provenance: string;
  overall: Summary;
  by_mode: Record<string, Summary>;
};

const NS = "http://www.w3.org/2000/svg";
const GROUPS: { kind: Agent["kind"]; title: string }[] = [
  { kind: "frontier", title: "Frontier agents" },
  { kind: "reference", title: "References" },
  { kind: "baseline", title: "Classical baselines" },
  { kind: "cheater", title: "Gaming strategies" },
];
const ROW = 30;
const HEAD = 38;
const DOMAIN = [-0.1, 1] as const;

function el<K extends keyof SVGElementTagNameMap>(name: K, attrs: Record<string, string | number>, parent?: Element) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  parent?.append(node);
  return node;
}

export class Ladder {
  private rows = new Map<string, { dot: SVGRectElement; ci: SVGLineElement; value: SVGTextElement }>();
  private x: (v: number) => number = (v) => v;
  private mode = "overall";

  constructor(private svg: SVGSVGElement) {
    this.build();
    new ResizeObserver(() => this.build()).observe(svg.parentElement as Element);
  }

  private values(a: Agent): Summary {
    return this.mode === "overall" ? a.overall : a.by_mode[this.mode] ?? {};
  }

  private build(): void {
    const width = (this.svg.parentElement as HTMLElement).clientWidth || 1000;
    // Narrow screens stack each label above its row so marks never overprint names.
    const compact = width < 600;
    const ROW_H = compact ? 42 : ROW;
    const markY = compact ? 30 : ROW / 2;
    const labelW = compact ? 6 : Math.min(230, width * 0.36);
    const right = compact ? 40 : 64;
    // Only agents with scorecards are drawn; frontier rows appear when their runs land.
    const agents = (results.agents as Agent[]).filter((a) => a.provenance !== "placeholder");
    let height = 28;
    for (const g of GROUPS) height += HEAD + agents.filter((a) => a.kind === g.kind).length * ROW_H;
    this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    this.svg.replaceChildren(this.svg.querySelector("desc") as Node);

    const x0 = labelW;
    const x1 = width - right;
    this.x = (v: number) => x0 + ((Math.max(DOMAIN[0], Math.min(DOMAIN[1], v)) - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * (x1 - x0);

    const axis = el("g", { class: "axis" }, this.svg);
    for (const t of compact ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]) {
      el("line", { x1: this.x(t), x2: this.x(t), y1: 18, y2: height, class: t === 0 || t === 1 ? "ref" : "" }, axis);
      const anchor = compact && t === 0 ? "start" : compact && t === 1 ? "end" : "middle";
      const label = el("text", { x: this.x(t), y: 12, "text-anchor": anchor }, axis);
      label.textContent = t === 0 ? "0 floor" : t === 1 ? "1 oracle" : t.toFixed(2);
    }

    this.rows.clear();
    let y = 28;
    for (const g of GROUPS) {
      const members = agents
        .filter((a) => a.kind === g.kind)
        .sort((a, b) => (b.overall.discovery_score ?? -1) - (a.overall.discovery_score ?? -1));
      if (!members.length) continue;
      const head = el("text", { x: 0, y: y + 24, class: "kind-head" }, this.svg);
      head.textContent = g.title;
      y += HEAD;
      for (const a of members) {
        const row = el("g", { class: `row kind-${a.kind}`, transform: `translate(0 ${y})` }, this.svg);
        el("line", { x1: 0, x2: width, y1: ROW_H - 0.5, y2: ROW_H - 0.5, class: "row-rule" }, row);
        const name = el("text", { x: 0, y: compact ? 13 : ROW / 2 + 5, class: `row-label kind-${a.kind}` }, row);
        name.textContent = a.label;
        const ci = el("line", { x1: this.x(0), x2: this.x(0), y1: markY, y2: markY, class: "ci" }, row);
        const dot = el("rect", { x: this.x(0) - 5, y: markY - 5, width: 10, height: 10, class: "dot" }, row);
        const value = el("text", { x: width, y: markY + 4, "text-anchor": "end", class: "value" }, row);
        this.rows.set(a.id, { dot, ci, value });
        y += ROW_H;
      }
    }
    this.render(false);
  }

  setMode(mode: string): void {
    this.mode = mode;
    this.render(true);
  }

  render(animate: boolean): void {
    for (const a of results.agents as Agent[]) {
      const r = this.rows.get(a.id);
      if (!r) continue;
      const v = this.values(a);
      const ds = v.discovery_score ?? null;
      const cx = this.x(ds ?? 0);
      for (const node of [r.dot, r.ci]) node.style.transition = animate ? "all 0.9s cubic-bezier(0.16,1,0.3,1)" : "none";
      r.dot.setAttribute("x", String(cx - 5));
      r.dot.style.opacity = ds === null ? "0.2" : "1";
      r.ci.setAttribute("x1", String(this.x(v.low ?? ds ?? 0)));
      r.ci.setAttribute("x2", String(this.x(v.high ?? ds ?? 0)));
      r.value.textContent = ds === null ? "—" : (Math.abs(ds) < 0.005 ? 0 : ds).toFixed(2);
    }
  }

  /** Entrance: every dot starts at the floor and slides to its score. */
  enter(): void {
    const saved = this.mode;
    for (const r of this.rows.values()) {
      r.dot.style.transition = "none";
      r.dot.setAttribute("x", String(this.x(0) - 5));
      r.ci.setAttribute("x1", String(this.x(0)));
      r.ci.setAttribute("x2", String(this.x(0)));
    }
    requestAnimationFrame(() => requestAnimationFrame(() => {
      this.mode = saved;
      this.render(true);
    }));
  }
}
