/**
 * The opening landscape as static SVG, baked from the same seeded field the canvas draws,
 * so the first view is complete without JavaScript. Hidden once the canvas takes over.
 */
import { bandX, CAUSE, LANDSCAPE, landscapePoints } from "../src/lib/field";

const W = 1440;
const H = 240;
const Y_MAX = 8.4;
const PEAK = 7.6;
const FILL = 1.8;

export function staticBand(): string {
  const pad = { l: 8, r: 8, t: 18, b: 6 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const X = (x: number) => pad.l + x * pw;
  const Y = (v: number) => pad.t + ph - (Math.min(v, Y_MAX) / Y_MAX) * ph;
  const cap = LANDSCAPE.threshold - 0.3;
  const paths = new Map<number, string[]>();
  for (const p of landscapePoints(Math.round(LANDSCAPE.count * 0.15))) {
    const v = cap * (1 - Math.exp((-p.y * FILL) / cap));
    const d = paths.get(p.band) ?? [];
    d.push(`M${X(p.x).toFixed(0)} ${Y(v).toFixed(0)}h2v2h-2z`);
    paths.set(p.band, d);
  }
  const field = [...paths]
    .map(([band, d]) => `<path fill="${band % 2 ? "#3e5f82" : "#7fa7c9"}" d="${d.join("")}"/>`)
    .join("");
  const ty = Y(LANDSCAPE.threshold).toFixed(1);
  const cx = X(bandX(CAUSE.band, CAUSE.offset)).toFixed(1);
  const cy = Y(PEAK).toFixed(1);
  return `<svg class="field-static" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${field}
<rect x="0" y="${ty}" width="${W}" height="1.25" fill="#d7263d"/>
<rect x="${(Number(cx) - 0.5).toFixed(1)}" y="${cy}" width="1" height="${(Y(0.5) - Number(cy)).toFixed(1)}" fill="#d7263d"/>
<rect x="${(Number(cx) - 4).toFixed(1)}" y="${(Number(cy) - 4).toFixed(1)}" width="8" height="8" fill="#d7263d"/>
</svg>`;
}
