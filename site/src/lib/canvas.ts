/** A canvas sized to its box at device-pixel resolution, redrawn on resize. */
export type Surface = { ctx: CanvasRenderingContext2D; w: number; h: number; dpr: number };

export function surface(canvas: HTMLCanvasElement, onResize: () => void): () => Surface {
  const ctx = canvas.getContext("2d", { alpha: true }) as CanvasRenderingContext2D;
  let state: Surface = { ctx, w: 0, h: 0, dpr: 1 };
  const fit = (notify: boolean) => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    state = { ctx, w: width, h: height, dpr };
    if (notify) onResize();
  };
  // Size now without calling back (the owner is still being constructed); the
  // observer's first callback and every resize after it trigger the redraw.
  fit(false);
  new ResizeObserver(() => fit(true)).observe(canvas);
  return () => state;
}

/** Runs ``frame`` on animation frames only while the element is on screen. */
export function whileVisible(el: Element, frame: (t: number) => void): { kick: () => void } {
  let visible = false;
  let raf = 0;
  const loop = (t: number) => {
    frame(t);
    if (visible) raf = requestAnimationFrame(loop);
  };
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    cancelAnimationFrame(raf);
    if (visible) raf = requestAnimationFrame(loop);
  }).observe(el);
  return { kick: () => requestAnimationFrame(frame) };
}

export const COLORS = {
  ground: "#0f1a2b",
  ink: "#e9eef2",
  ink2: "#a9bacb",
  ink3: "#7d93aa",
  bandA: "#7fa7c9",
  bandB: "#3e5f82",
  signal: "#d7263d",
  signalText: "#ff6b7b",
  rule: "rgba(233,238,242,0.12)",
};

export const MONO = '"Azeret Mono Variable", "Azeret Mono", ui-monospace, monospace';

export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
/** Exponential ease-out on a 0..1 progress. */
export const easeOut = (t: number) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * clamp01(t)));
