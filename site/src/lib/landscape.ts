/**
 * The significance landscape on canvas: the page's one image. Every chapter that
 * uses it sets a scene (planted cause, stand-in, confounder, leak, no signal, a
 * cheater's claims) and the field redraws. A point that crosses the red threshold
 * changes state: it turns red and grows.
 */
import { bandX, CAUSE, LANDSCAPE, landscapePoints, rng, type FieldPoint } from "./field";
import { COLORS, MONO, clamp01, lerp, surface, whileVisible } from "./canvas";

export type Scene = {
  cause: number;
  standin: number;
  confounder: number;
  leak: number;
  nothing: number;
  breath: number;
  extras: number;
  /** A cheater's claimed measurements, surging over the line (0..1). */
  surge: number;
};

export const EMPTY: Scene = { cause: 0, standin: 0, confounder: 0, leak: 0, nothing: 0, breath: 1, extras: 0, surge: 0 };

export type FieldOptions = {
  density?: number;
  animate?: boolean;
  /** Top of the vertical axis, in -log10 p; smaller values fill the frame. */
  yMax?: number;
  /** Height the planted cause climbs to. */
  causePeak?: number;
  /** Vertical lift of the null field (1 = as generated). */
  lift?: number;
  /** Draw the -log10 p axis on the left. */
  axis?: boolean;
  /**
   * Fill the frame under the threshold: a saturating map that spreads the null field
   * up towards the line without ever crossing it (0 = as generated).
   */
  fill?: number;
};

const CONF_X = bandX(5, 0.85);

/** Which measurements each gaming strategy lists, as point index -> lift. */
function claims(kind: string, points: FieldPoint[]): Map<number, number> {
  const r = rng([...kind].reduce((s, c) => (s * 31 + c.charCodeAt(0)) >>> 0, 7));
  const out = new Map<number, number>();
  const n = points.length;
  const lift = () => LANDSCAPE.threshold - 0.4 + r() * 3.2;
  const pick = (count: number, filter: (p: FieldPoint) => boolean = () => true) => {
    let guard = 0;
    while (out.size < count && guard++ < count * 400) {
      const i = Math.floor(r() * n);
      if (filter(points[i])) out.set(i, lift());
    }
  };
  switch (kind) {
    case "giant_list":
      for (let i = 0; i < n; i += 9) out.set(i, LANDSCAPE.threshold - points[i].y + 0.3 + r() * 1.2);
      break;
    case "always_empty":
      break;
    case "random_abstain":
      pick(70);
      break;
    case "leak_exploiter":
      pick(6);
      break;
    case "metadata_only":
      pick(90, (p) => p.band === 3);
      break;
    case "hub_ranker":
      pick(80, (p) => p.band === 10 && p.x < bandX(10, 0.4));
      break;
    case "cluster_size_ranker":
      pick(80, (p) => p.band === 14);
      break;
    default:
      pick(60);
  }
  return out;
}

export class Landscape {
  scene: Scene = { ...EMPTY };
  labels = true;
  /** Scene caption drawn at the threshold's left end. */
  tag = "";
  /** Label of the planted cause; "" hides it. */
  causeLabel = "planted cause";
  surgeKind = "";
  onCause?: (x: number, y: number, visible: boolean, labelX: number) => void;
  private points: FieldPoint[];
  private size: ReturnType<typeof surface>;
  private loop: { kick: () => void };
  private causeX = bandX(CAUSE.band, CAUSE.offset);
  private surgeCache = new Map<string, Map<number, number>>();
  private opts: Required<FieldOptions>;

  constructor(canvas: HTMLCanvasElement, options: FieldOptions = {}) {
    this.opts = {
      density: 1,
      animate: true,
      yMax: LANDSCAPE.yMax,
      causePeak: 8.6,
      lift: 1,
      axis: false,
      fill: 0,
      ...options,
    };
    this.points = landscapePoints(Math.round(LANDSCAPE.count * this.opts.density));
    this.size = surface(canvas, () => this.draw(performance.now()));
    this.loop = this.opts.animate ? whileVisible(canvas, (t) => this.draw(t)) : { kick: () => this.draw(0) };
  }

  set(partial: Partial<Scene>): void {
    Object.assign(this.scene, partial);
    this.loop.kick();
  }

  private surgeSet(): Map<number, number> | null {
    if (!this.surgeKind || this.scene.surge <= 0) return null;
    let m = this.surgeCache.get(this.surgeKind);
    if (!m) {
      m = claims(this.surgeKind, this.points);
      this.surgeCache.set(this.surgeKind, m);
    }
    return m;
  }

  draw(t: number): void {
    const { ctx, w, h } = this.size();
    if (!w || !h) return;
    const s = this.scene;
    const o = this.opts;
    const left = s.extras > 0 ? Math.max(34, w * 0.045) : 0;
    const right = s.extras > 0 ? Math.max(34, w * 0.045) : 0;
    const axisW = o.axis ? 34 : 0;
    const pad = { l: 8 + left + axisW, r: 8 + right, t: Math.max(this.tag ? 34 : 18, h * 0.06), b: s.extras > 0 ? 24 : 6 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;
    const Y = (v: number) => pad.t + ph - (Math.min(v, o.yMax) / o.yMax) * ph;
    const X = (x: number) => pad.l + x * pw;
    const thr = LANDSCAPE.threshold;
    const size = w < 700 ? 1.6 : 2;
    const phase = t * 0.0012;
    const surge = this.surgeSet();
    const cap = thr - 0.3;

    ctx.clearRect(0, 0, w, h);

    if (o.axis) {
      ctx.font = `500 10px ${MONO}`;
      ctx.fillStyle = COLORS.ink3;
      ctx.textAlign = "right";
      for (let v = 0; v <= o.yMax - 0.5; v += 2) {
        ctx.fillText(String(v), pad.l - 10, Y(v) + 3);
        ctx.fillRect(pad.l - 6, Y(v), 4, 1);
      }
      ctx.save();
      ctx.translate(left + 8, pad.t + ph / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText("−log10 p", 0, 0);
      ctx.restore();
    }

    // The field, band by band so the fill colour changes 23 times, not 20,000.
    const hits: [number, number][] = [];
    let band = -1;
    for (let i = 0; i < this.points.length; i++) {
      const p = this.points[i];
      if (p.band !== band) {
        band = p.band;
        ctx.fillStyle = band % 2 ? COLORS.bandB : COLORS.bandA;
      }
      let v = p.y * o.lift * (1 + s.breath * p.amp * Math.sin(phase + p.phase));
      if (o.fill > 0) v = cap * (1 - Math.exp((-v * o.fill) / cap));
      if (s.confounder > 0) {
        // A whole region rises: the subtype shifts every measurement near it.
        const bump = Math.exp(-Math.pow((p.x - CONF_X) / 0.022, 2));
        if (bump > 0.01) v += s.confounder * bump * (2.6 + 3.2 * Math.abs(Math.sin(p.phase * 3)));
      }
      if (surge) {
        // A claimed measurement is lifted past the line at the surge's peak.
        const lift = surge.get(i);
        if (lift !== undefined) v += s.surge * Math.max(0, thr - v + 0.3 + (lift - thr + 0.4) * 0.35);
      }
      v *= 1 - 0.35 * s.nothing;
      if (v > thr) {
        hits.push([X(p.x), Y(v)]);
        continue;
      }
      ctx.fillRect(X(p.x), Y(v), size, size);
    }
    ctx.fillStyle = COLORS.signal;
    for (const [x, y] of hits) ctx.fillRect(x - 1, y - 1, size + 2, size + 2);

    // The threshold: the one horizon every chapter shares.
    const ty = Y(thr);
    ctx.fillStyle = COLORS.signal;
    ctx.fillRect(pad.l - left * 0.6, ty, pw + left * 0.6 + right * 0.6, 1.25);
    if (this.labels) {
      ctx.font = `500 11px ${MONO}`;
      ctx.textAlign = "right";
      // The line's name stays legible over anything that crosses it.
      const name = "family-wise 5%";
      const tw = ctx.measureText(name).width;
      // Narrow plots set it under the line, clear of anything rising through it.
      const ly = w < 560 ? ty + 15 : ty - 8;
      ctx.fillStyle = COLORS.ground;
      ctx.fillRect(w - pad.r - tw - 6, ly - 13, tw + 10, 17);
      ctx.fillStyle = COLORS.signalText;
      ctx.fillText(name, w - pad.r, ly);
      if (this.tag) {
        // The scene caption sits at the plot's top-left, clear of anything that crosses the line.
        ctx.textAlign = "left";
        ctx.fillStyle = COLORS.ink2;
        ctx.fillText(this.tag, pad.l, 13);
      }
    }

    // A planted feature: a stem down to its column, and a hairline leader into the margin.
    const mark = (x: number, v: number, label: string, alpha: number, leader = 64) => {
      if (alpha <= 0) return;
      const y = Y(v);
      const crossed = v > thr;
      ctx.globalAlpha = alpha * 0.4;
      ctx.fillStyle = crossed ? COLORS.signal : COLORS.ink2;
      ctx.fillRect(x - 0.5, y, 1, Math.max(0, Y(0.5) - y)); // stem down into its column
      ctx.globalAlpha = alpha;
      ctx.fillStyle = crossed ? COLORS.signal : COLORS.ink;
      const r = crossed ? 4 : 3;
      ctx.fillRect(x - r, y - r, r * 2, r * 2);
      if (this.labels && label) {
        const dir = x > w * 0.66 ? -1 : 1; // near the right edge the leader runs left
        ctx.fillStyle = COLORS.ink2;
        ctx.fillRect(dir > 0 ? x + 8 : x - 8 - leader, y, leader, 1);
        ctx.font = `500 12px ${MONO}`;
        ctx.textAlign = dir > 0 ? "left" : "right";
        ctx.fillStyle = COLORS.ink;
        ctx.fillText(label, x + (14 + leader) * dir, y + 4);
      }
      ctx.globalAlpha = 1;
    };

    const peak = o.causePeak;
    const causeV = lerp(2.1, peak, clamp01(s.cause)) * (1 - 0.82 * s.nothing);
    const cx = X(this.causeX);
    if (s.cause > 0) {
      if (this.onCause) {
        // The hero sets its label in the DOM (so it can type); the canvas draws stem and leader.
        mark(cx, causeV, "", 1);
        const dir = cx > w * 0.66 ? -1 : 1;
        const labelX = cx + dir * Math.min(240, w * 0.17);
        ctx.fillStyle = COLORS.ink2;
        ctx.globalAlpha = clamp01((s.cause - 0.85) * 6);
        ctx.fillRect(dir > 0 ? cx + 8 : labelX, Y(causeV), Math.abs(labelX - cx) - 8, 1);
        ctx.globalAlpha = 1;
        this.onCause(cx, Y(causeV), s.cause > 0.95 && s.nothing < 0.5, labelX);
      } else mark(cx, causeV, s.nothing > 0.5 ? "" : this.causeLabel, 1);
    }
    if (s.standin > 0)
      mark(
        cx + pw * 0.006,
        lerp(1.9, peak - 0.45, s.standin) * (1 - 0.82 * s.nothing),
        "stand-in, r = 0.94",
        clamp01(s.standin * 1.4),
        40,
      );

    // Extras: a clinical strip on the left, a post-outcome strip on the right.
    if (s.extras > 0) {
      ctx.globalAlpha = s.extras;
      ctx.font = `500 10px ${MONO}`;
      ctx.fillStyle = COLORS.ink3;
      ctx.textAlign = "left";
      ctx.fillText("clinical", 2, h - 6);
      ctx.textAlign = "right";
      ctx.fillText("post-outcome", w - 2, h - 6);
      for (let i = 0; i < 6; i++) ctx.fillRect(left * 0.15 + i * (left * 0.13), Y(0.6 + ((i * 37) % 10) / 9), 3, 3);
      for (let i = 0; i < 3; i++) ctx.fillRect(w - right * 0.8 + i * (right * 0.22), Y(0.7 + i * 0.3), 3, 3);
      ctx.globalAlpha = 1;
      const sub = lerp(0.9, peak - 1.2, s.confounder) * (1 - 0.82 * s.nothing);
      if (s.confounder > 0)
        mark(left * 0.15 + 2 * (left * 0.13) + 1, sub, "subtype: the cause", clamp01(s.confounder * 1.3), 40);
      const leakV = Math.max(s.leak, this.surgeKind === "leak_exploiter" ? s.surge : 0);
      if (leakV > 0)
        mark(
          w - right * 0.8 + 1,
          lerp(0.8, Math.min(o.yMax - 0.4, peak + 1.2), leakV),
          "post-outcome leak",
          clamp01(leakV * 1.3),
          40,
        );
    }
  }
}
