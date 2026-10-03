/**
 * The synthetic significance landscape: one deterministic field of measurements laid
 * out like a Manhattan plot (bands = chromosomes), with -log10 p on the vertical axis.
 * It is an illustration, labelled as such on the page; benchmark worlds are not shown.
 */

export const LANDSCAPE = {
  count: 20530,
  /** -log10(0.05 / 20530): a family-wise 5% line for this many measurements. */
  threshold: 5.61,
  yMax: 11,
  /** Relative band widths (gene counts per chromosome, rounded). */
  bands: [2050, 1250, 1080, 760, 880, 1040, 920, 690, 780, 740, 1310, 1040, 330, 610, 600, 860, 1180, 270, 1470, 540, 240, 440, 840],
};

export type FieldPoint = {
  /** Position across the plot, 0..1, with gaps between bands. */
  x: number;
  /** Resting -log10 p. */
  y: number;
  band: number;
  /** Breathing phase and amplitude. */
  phase: number;
  amp: number;
};

/** Mulberry32: tiny seeded PRNG so the server SVG and the client canvas agree. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const total = LANDSCAPE.bands.reduce((s, b) => s + b, 0);
const GAP = 0.004;

/** Index of the planted cause, and its stand-in neighbour. */
export const CAUSE = { band: 16, offset: 0.43 };

export function landscapePoints(n: number = LANDSCAPE.count, seed = 20530): FieldPoint[] {
  const r = rng(seed);
  const pts: FieldPoint[] = [];
  const usable = 1 - GAP * (LANDSCAPE.bands.length - 1);
  let start = 0;
  LANDSCAPE.bands.forEach((size, band) => {
    const k = Math.max(1, Math.round((size / total) * n));
    const width = (size / total) * usable;
    let ld = 0; // linkage-like local correlation: neighbours share a slowly varying lift
    for (let i = 0; i < k; i++) {
      ld = ld * 0.92 + (r() < 0.004 ? 1.2 + r() * 1.6 : 0);
      const base = -Math.log10(Math.max(r(), 1e-9)); // null p-values: exponential in -log10
      const y = Math.min(LANDSCAPE.threshold - 0.6, base * 0.8 + ld * 0.9 * r());
      pts.push({ x: start + (i / k) * width, y, band, phase: r() * Math.PI * 2, amp: 0.08 + r() * 0.22 });
    }
    start += width + GAP;
  });
  return pts;
}

/** x position (0..1) of a band's offset, for placing the planted features. */
export function bandX(band: number, offset: number): number {
  const usable = 1 - GAP * (LANDSCAPE.bands.length - 1);
  let start = 0;
  for (let b = 0; b < band; b++) start += (LANDSCAPE.bands[b] / total) * usable + GAP;
  return start + offset * (LANDSCAPE.bands[band] / total) * usable;
}
