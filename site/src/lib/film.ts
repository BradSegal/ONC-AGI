/**
 * The page's films: the opening and the journey's 3D beats, rendered at build time at full quality
 * (render/, build/render-media.mjs) and played here as video, so every visitor sees the same frames
 * whatever their hardware. Labels and captions stay live HTML, placed from coordinates recorded
 * when each film was rendered.
 */
import mediaJson from "../data/media.json";

type Placed = { key: string; text: string; id?: string; cls?: string; x: number; y: number };
type Point = [key: string, x: number, y: number];
type Clip = {
  name: string;
  duration: number;
  defs: Record<string, { text: string; id?: string; cls?: string }>;
  labels: Point[];
  timeline?: { t: number; p: Point[] }[];
  tags?: { at: number; text: string }[];
};
const media = mediaJson as unknown as Record<string, Clip> & { captions?: Record<string, string> };

const BASE = `${import.meta.env.BASE_URL}media/`;
/**
 * AV1 only where the device decodes it smoothly and power-efficiently, which in practice means in
 * hardware; H.264, decoded in hardware on nearly every laptop and phone, everywhere else.
 */
let ext: "av1" | "avc" = "avc";
const codecReady = (async () => {
  try {
    if (document.createElement("video").canPlayType('video/mp4; codecs="av01.0.08M.08"') !== "probably" || !navigator.mediaCapabilities) return;
    const info = await navigator.mediaCapabilities.decodingInfo({
      type: "file",
      video: { contentType: 'video/mp4; codecs="av01.0.08M.08"', width: 1920, height: 1080, bitrate: 2_100_000, framerate: 60 },
    });
    if (info.supported && info.smooth && info.powerEfficient) ext = "av1";
  } catch {
    /* H.264 */
  }
})();
const src = (name: string) => cache.get(name) ?? `${BASE}${name}.${ext}.mp4`;
const still = (name: string) => `${BASE}${name}-end.webp`;
const cache = new Map<string, string>(); // clip name -> blob URL once preloaded
/** Visitors who ask to save data get the films' settled stills instead of video. */
const saveData = !!(navigator as Navigator & { connection?: { saveData?: boolean } }).connection?.saveData;

function video(cls: string): HTMLVideoElement {
  const v = document.createElement("video");
  v.className = cls;
  v.muted = true;
  v.playsInline = true;
  v.preload = "auto";
  v.setAttribute("aria-hidden", "true");
  v.disablePictureInPicture = true;
  return v;
}

/** Labels drawn over a film, mapped through the video's object-fit: cover crop. */
class Overlay {
  private els = new Map<string, HTMLElement>();
  private wanted = new Set<string>();
  constructor(
    readonly host: HTMLElement,
    readonly frame: () => HTMLVideoElement,
    readonly position: [number, number],
  ) {}

  draw(labels: Placed[]) {
    const v = this.frame();
    const vw = v.videoWidth || 16;
    const vh = v.videoHeight || 9;
    // the film may sit inset in its host: map through the video's own rectangle
    const hr = this.host.getBoundingClientRect();
    const vr = v.getBoundingClientRect();
    const cw = vr.width;
    const ch = vr.height;
    const scale = Math.max(cw / vw, ch / vh);
    const ox = vr.left - hr.left + (cw - vw * scale) * this.position[0];
    const oy = vr.top - hr.top + (ch - vh * scale) * this.position[1];
    const minX = vr.left - hr.left + 24;
    const maxX = vr.left - hr.left + cw - 24;
    const minY = vr.top - hr.top + 12;
    const maxY = vr.top - hr.top + ch - 12;
    const keep = new Set(labels.map((l) => l.key));
    this.wanted = keep;
    for (const [key, el] of this.els) if (!keep.has(key)) el.classList.remove("is-on");
    for (const l of labels) {
      let el = this.els.get(l.key);
      if (!el) {
        el = document.createElement("span");
        el.innerHTML = "<i></i><b></b>";
        this.host.append(el);
        this.els.set(l.key, el);
      }
      el.className = `scene-label ${l.cls ?? ""}${el.classList.contains("is-on") ? " is-on" : ""}`;
      const b = el.querySelector("b")!;
      if (b.dataset.text !== `${l.text}|${l.id ?? ""}`) {
        b.dataset.text = `${l.text}|${l.id ?? ""}`;
        b.textContent = l.text;
        if (l.id) {
          const code = document.createElement("code");
          code.textContent = l.id;
          if (l.text) b.append(" ");
          b.append(code);
        }
      }
      // keep every label inside the visible frame, whatever the crop
      const px = Math.min(maxX, Math.max(minX, ox + l.x * vw * scale));
      const py = Math.min(maxY, Math.max(minY, oy + l.y * vh * scale));
      el.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
      const node = el;
      // reveal on the next frame, unless a later state has already dropped this label
      if (!node.classList.contains("is-on")) requestAnimationFrame(() => this.wanted.has(l.key) && node.classList.add("is-on"));
    }
  }
}

/** Points with their label text, from a clip's definitions. */
const placed = (clip: Clip, points: Point[]): Placed[] => points.map(([key, x, y]) => ({ key, x, y, ...clip.defs[key] }));
/** The labels at time t (the timeline holds a sample wherever any label moved). */
function sampleAt(clip: Clip, t: number): Placed[] {
  const tl = clip.timeline;
  if (!tl?.length) return placed(clip, clip.labels);
  let i = 0;
  while (i + 1 < tl.length && tl[i + 1].t <= t) i++;
  return placed(clip, tl[i].p);
}

/* ------------------------------------------------------------------ the opening */

export function opening(host: HTMLElement, motionless: boolean): void {
  void codecReady.then(() => openingReady(host, motionless));
}

function openingReady(host: HTMLElement, motionless: boolean): void {
  const reduced = motionless || saveData;
  const shape = matchMedia("(max-width: 900px)").matches ? "square" : "wide";
  const intro = media[`hero-${shape}-intro`];
  const loop = media[`hero-${shape}-loop`];
  if (!intro || !loop) return;
  const tag = document.querySelector<HTMLElement>("[data-world-tag]");
  const poster = host.querySelector<HTMLImageElement>(".film-poster");
  const a = video("film");
  const b = video("film");
  b.loop = true;
  a.poster = poster?.currentSrc || still(intro.name);
  host.append(a, b);
  const overlay = new Overlay(host, () => (b.classList.contains("is-on") ? b : a), [0.72, 0.5]);
  let lastTag = "";
  const setTag = (text: string) => {
    if (!tag || text === lastTag) return;
    lastTag = text;
    tag.classList.remove("is-on");
    window.setTimeout(() => {
      tag.textContent = text;
      tag.classList.add("is-on");
    }, 250);
  };
  const tagAt = (clip: Clip, t: number) => [...(clip.tags ?? [])].reverse().find((x) => x.at <= t)?.text;

  if (reduced) {
    // a still of the settled opening, with its labels and caption
    a.src = "";
    a.classList.add("is-on");
    overlay.draw(placed(intro, intro.labels));
    setTag(tagAt(intro, intro.duration) ?? "");
    return;
  }
  a.src = src(intro.name);
  b.src = src(loop.name);
  a.classList.add("is-on");
  // the intro plays once; the loop takes over on its first frame, which matches the intro's last
  a.addEventListener("ended", () => {
    b.currentTime = 0;
    void b.play().then(() => {
      b.classList.add("is-on");
      a.classList.remove("is-on");
    });
  });
  const tick = () => {
    const playing = b.classList.contains("is-on") ? b : a;
    const clip = playing === b ? loop : intro;
    overlay.draw(sampleAt(clip, playing.currentTime));
    const text = tagAt(clip, playing.currentTime);
    if (text) setTag(text);
    if (!playing.paused) requestAnimationFrame(tick);
  };
  const start = () => {
    const playing = b.classList.contains("is-on") ? b : a;
    void playing.play().then(() => {
      poster?.classList.add("is-hidden");
      requestAnimationFrame(tick);
    }).catch(() => undefined);
  };
  a.addEventListener("playing", () => requestAnimationFrame(tick));
  b.addEventListener("playing", () => requestAnimationFrame(tick));
  // films play only while on screen and the tab is visible
  let onScreen = true;
  new IntersectionObserver(([e]) => {
    onScreen = e.isIntersecting;
    if (onScreen && !document.hidden) start();
    else [a, b].forEach((v) => v.pause());
  }).observe(host);
  document.addEventListener("visibilitychange", () => (document.hidden ? [a, b].forEach((v) => v.pause()) : onScreen && start()));
  window.addEventListener("resize", () => requestAnimationFrame(tick));
}

/* ------------------------------------------------------------------ the journey */

export type Stage = { show(pane: string, state: number): void; preload(): void };

export function stage(host: HTMLElement, motionless: boolean): Stage {
  const reduced = motionless || saveData;
  const clips = Object.values(media).filter((c): c is Clip => !!c && typeof c === "object" && "name" in c && c.name.startsWith("journey-"));
  const edge = (from: string, to: string) => clips.find((c) => c.name === `journey-${from}--${to}`);
  const into = (to: string) => clips.find((c) => c.name.endsWith(`--${to}`));
  const a = video("film film-stage");
  const b = video("film film-stage");
  const overlay = document.createElement("div");
  overlay.className = "scene-overlay";
  overlay.setAttribute("aria-hidden", "true");
  const caption = document.createElement("div");
  caption.className = "scene-caption";
  overlay.append(caption);
  host.prepend(a, b, overlay);
  let front = a;
  const labels = new Overlay(overlay, () => front, [0.5, 0.5]);
  let cur: string | null = null; // the state the film shows (or is heading to)
  let want: string | null = null;
  let busy = false;
  let playing: { el: HTMLVideoElement; to: string } | null = null;

  const settle = (to: string, clip: Clip) => {
    cur = to;
    // fetch only what the reader may watch next: the clips leaving this state
    fetchClips(clips.filter((c) => c.name.startsWith(`journey-${to}--`)).map((c) => c.name));
    labels.draw(placed(clip, clip.labels));
    caption.innerHTML = media.captions?.[to] ?? "";
    busy = false;
    if (want && want !== cur) go(want);
  };
  /** Play a clip on the hidden element and swap it in on its first frame. */
  const play = (clip: Clip, to: string) => {
    busy = true;
    labels.draw([]);
    const back = front === a ? b : a;
    back.src = src(clip.name);
    back.currentTime = 0;
    // a reader already further on: this transition runs fast so the film catches up smoothly
    back.playbackRate = want && want !== to ? 2 : 1;
    playing = { el: back, to };
    const swap = () => {
      back.classList.add("is-on");
      front.classList.remove("is-on");
      front = back;
    };
    back.onended = () => {
      playing = null;
      settle(to, clip);
    };
    void back.play().then(swap).catch(() => jump(to));
  };
  /** Show a state's settled frame without playing (deep links, large jumps, reduced motion). */
  const jump = (to: string) => {
    const clip = into(to);
    if (!clip) return;
    busy = true;
    const back = front === a ? b : a;
    back.removeAttribute("src");
    back.poster = still(clip.name);
    back.load();
    back.classList.add("is-on");
    front.classList.remove("is-on");
    front = back;
    settle(to, clip);
  };
  const go = (target: string) => {
    want = target;
    if (busy) {
      if (playing && playing.to !== target) playing.el.playbackRate = 2;
      return;
    }
    if (target === cur) return;
    const clip = cur === null ? (target === "build-0" ? edge("enter", "build-0") : undefined) : edge(cur, target);
    if (clip && !reduced) play(clip, target);
    else jump(target);
  };

  return {
    show(pane, state) {
      const on = pane === "build" || pane === "act";
      host.classList.toggle("is-3d-on", on);
      if (on) return go(`${pane}-${state}`);
      // leaving for a 2D figure: the camera rises to look straight down, then the film fades
      if (cur && !cur.startsWith("top-")) {
        const exit = clips.find((c) => c.name.startsWith(`journey-${cur}--top-`));
        if (exit) go(exit.name.split("--")[1]);
      }
    },
    preload() {
      // fetch the opening clip and those leading out of the first beats, as blob URLs for instant swaps
      fetchClips(["journey-enter--build-0", "journey-build-0--build-1", "journey-build-1--build-0"]);
    },
  };
}

/** Fetch clips into blob URLs, one at a time, skipping any already held. */
function fetchClips(names: string[]) {
  (async () => {
        for (const name of names) {
          if (cache.has(name)) continue;
          try {
            const res = await fetch(`${BASE}${name}.${ext}.mp4`);
            if (res.ok) cache.set(name, URL.createObjectURL(await res.blob()));
          } catch {
            /* the network URL still works */
          }
        }
  })();
}
