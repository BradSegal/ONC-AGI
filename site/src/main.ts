import "@fontsource-variable/archivo/standard.css";
import "@fontsource-variable/azeret-mono";
import "./styles/main.css";

import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";

import { Landscape } from "./lib/landscape";
import { rollTo } from "./lib/roll";

gsap.registerPlugin(ScrollTrigger);

const root = document.documentElement;
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
if (!reduced) root.classList.add("motion");
// Phones read the journey inline: each figure follows its own text, and only the transport bar pins.
const mobile = window.matchMedia("(max-width: 900px)").matches;

const $ = <T extends Element = HTMLElement>(sel: string, scope: ParentNode = document) => scope.querySelector(sel) as T;
const $$ = <T extends Element = HTMLElement>(sel: string, scope: ParentNode = document) =>
  [...scope.querySelectorAll(sel)] as T[];

/* ------------------------------------------------------------------ scrolling */

let lenis: Lenis | null = null;
if (!reduced) {
  lenis = new Lenis({ duration: 1.05, easing: (t) => 1 - Math.pow(2, -10 * t) });
  lenis.on("scroll", ScrollTrigger.update);
  gsap.ticker.add((time) => lenis?.raf(time * 1000));
  gsap.ticker.lagSmoothing(0);
}
const scrollToY = (y: number, duration = 1.3) => (lenis ? lenis.scrollTo(y, { duration }) : window.scrollTo({ top: y }));

/* ------------------------------------------------------------------ opening band */

const band = new Landscape($<HTMLCanvasElement>('[data-field="band"]'), {
  density: mobile ? 0.35 : 0.7,
  animate: !reduced,
  yMax: 8.4,
  causePeak: 7.6,
  fill: 1.8,
});
band.labels = !mobile;
band.causeLabel = "a planted cause clears the line";
if (reduced) band.set({ cause: 1, breath: 0 });
else {
  const rise = { cause: 0 };
  gsap.to(rise, { cause: 1, duration: 2.4, delay: 0.5, ease: "expo.out", onUpdate: () => band.set({ cause: rise.cause }) });
  gsap.from(".opening__text > *", { y: 22, opacity: 0, duration: 1.2, stagger: 0.08, ease: "expo.out" });
}

/* ------------------------------------------------------------------ the journey replay */

type Step = { kind: string; patients: number; spent: number };
const transport = $(".transport");
const tdata = JSON.parse(transport.dataset.transport as string) as {
  budget: number;
  reference: number;
  cross: number | null;
  steps: Step[];
};
const analyses = tdata.steps.filter((s) => s.kind === "analyse");
const final = tdata.steps[tdata.steps.length - 1];
const meters = { patients: $('[data-meter="patients"]'), spent: $('[data-meter="spent"]') };
const spendBar = $("[data-meter-bar]");
$("[data-ref-mark]").style.left = `${(100 * tdata.reference) / tdata.budget}%`;
// red marks the spend at which the first measurement crossed the agent's line
const crossMark = $("[data-cross-mark]");
if (tdata.cross !== null) crossMark.style.left = `${(100 * tdata.cross) / tdata.budget}%`;

const panes = new Map($$<HTMLElement>("[data-pane]", $(".panes")).map((p) => [p.dataset.pane as string, p]));
const phases = $$<HTMLElement>(".transport__phases li");
const done = new Set<string>(); // pane entrance animations already played

/** Point a pane at one of its states; entrance motion plays once, state changes always. */
function show(pane: string, state: number, phase: number): void {
  phases.forEach((li, i) => {
    li.classList.toggle("is-current", i === phase);
    li.classList.toggle("is-past", i < phase);
  });
  const el = panes.get(pane);
  if (!mobile && el) {
    panes.forEach((p, name) => p.classList.toggle("is-active", name === pane));
    el.dataset.state = String(state);
    const first = !done.has(pane);
    done.add(pane);
    enter[pane]?.(el, state, first);
  }
  // spend meters follow the episode
  const at = pane === "act" ? analyses[Math.min(state, analyses.length - 1)] : phase > 3 ? final : null;
  const patients = at?.patients ?? 0;
  const spent = at?.spent ?? 0;
  rollTo(meters.patients, String(patients));
  rollTo(meters.spent, Math.round(spent).toLocaleString("en-GB"));
  spendBar.style.transform = `scaleX(${spent / tdata.budget})`;
  crossMark.classList.toggle("is-on", tdata.cross !== null && spent >= tdata.cross);
}

const tween = (targets: gsap.TweenTarget, vars: gsap.TweenVars) =>
  reduced ? gsap.set(targets, { ...vars, duration: 0 }) : gsap.to(targets, vars);

const SCRAMBLE = "bcdfghjklmnpqrstvwxz0123456789";
function scramble(el: SVGTextElement): void {
  const target = el.dataset.col ?? el.textContent ?? "";
  if (reduced) return;
  let n = 0;
  const id = window.setInterval(() => {
    n++;
    el.textContent = [...target].map((c, i) => (i < n / 2 ? c : SCRAMBLE[Math.floor(Math.random() * SCRAMBLE.length)])).join("");
    if (n > target.length * 2) {
      window.clearInterval(id);
      el.textContent = target;
    }
  }, 45);
}

let actLabels: SVGTextElement[][] | undefined;
const enter: Record<string, (el: HTMLElement, state: number, first: boolean) => void> = {
  build(el, state, first) {
    if (first && !reduced) gsap.from($$(".r", el), { opacity: 0, x: -6, duration: 0.5, stagger: 0.012, ease: "expo.out" });
    if (state === 3) $$<SVGTextElement>(".colhead", el).forEach(scramble);
  },
  certify(el, _state, first) {
    if (!first || reduced) return;
    $$<SVGPolylineElement>(".ci-line", el).forEach((line, i) => {
      const len = line.getTotalLength();
      gsap.fromTo(line, { strokeDasharray: len, strokeDashoffset: len }, { strokeDashoffset: 0, duration: 1.6, delay: i * 0.35, ease: "power2.out" });
    });
    gsap.from($$(".ci-band", el), { opacity: 0, duration: 1.2, delay: 0.6, stagger: 0.3 });
  },
  receive(el, _state, first) {
    if (first && !reduced) gsap.from($$(".worldcard > *", el), { opacity: 0, y: 8, duration: 0.6, stagger: 0.06, ease: "expo.out" });
  },
  act(el, state) {
    // One live frame; its points move to the recorded evidence at each analysis.
    const frames = $$<SVGGElement>(".act-frame", el);
    const live = frames[0];
    const k = Math.min(state, frames.length - 1);
    const target = frames[k];
    frames.forEach((f, i) => f.classList.toggle("is-shown", i === 0));
    // each frame's labels, captured once before the live frame is first rewritten
    actLabels ??= frames.map((f) => $$<SVGTextElement>(".pt-label, .frame-tag", f).map((t) => t.cloneNode(true) as SVGTextElement));
    $$<SVGRectElement>(".pt", live).forEach((pt) => {
      const twin = $<SVGRectElement>(`.pt[data-f="${pt.dataset.f}"]`, target);
      tween(pt, { attr: { y: Number(twin.getAttribute("y")) }, duration: 1.1, ease: "expo.out" });
      pt.setAttribute("class", twin.getAttribute("class") ?? "pt");
    });
    $$<SVGTextElement>(".pt-label, .frame-tag", live).forEach((t) => t.remove());
    actLabels[k].forEach((t) => {
      const copy = t.cloneNode(true) as SVGTextElement;
      live.append(copy);
      if (!reduced) gsap.from(copy, { opacity: 0, duration: 0.6, delay: 0.5 });
    });
  },
  score(el, state) {
    const visible = [3, 5, 7, 8][state] ?? 8;
    $$<HTMLElement>("[data-k]", el).forEach((row) => {
      const k = Number(row.dataset.k);
      const on = k < visible;
      if (on && !row.classList.contains("is-on") && !reduced) gsap.from(row, { opacity: 0, y: 10, duration: 0.6, ease: "expo.out", delay: (k % 3) * 0.12 });
      row.classList.toggle("is-on", on);
    });
  },
  across(el, _state, first) {
    if (first && !reduced) gsap.from($$("td", el), { opacity: 0, scale: 0.92, duration: 0.45, stagger: 0.035, ease: "expo.out" });
  },
};

const beats = $$<HTMLElement>(".beat");
const activate = (beat: HTMLElement) => {
  beats.forEach((b) => b.classList.toggle("is-current", b === beat));
  show(beat.dataset.pane as string, Number(beat.dataset.state), Number(beat.dataset.phase));
};
beats.forEach((beat) =>
  ScrollTrigger.create({
    trigger: beat,
    start: mobile ? "top 72%" : "top 58%",
    end: mobile ? "bottom 72%" : "bottom 58%",
    onToggle: (self) => self.isActive && activate(beat),
  }),
);
activate(beats[0]);

// Inline (phone) reading: every figure sits after its text in its final state and plays its
// entrance once on arrival; the act figure replays the episode 30 -> 60 -> 90 patients.
const FINAL: Record<string, number> = { build: 3, certify: 1, receive: 0, act: 0, score: 3, across: 1 };
if (mobile)
  panes.forEach((el, name) => {
    el.dataset.state = String(FINAL[name] ?? 0);
    const play = () => {
      enter[name]?.(el, FINAL[name] ?? 0, true);
      if (name === "act")
        analyses.slice(1).forEach((_, k) => window.setTimeout(() => enter.act(el, k + 1, false), (reduced ? 0 : 1500) * (k + 1)));
    };
    if (reduced) play();
    else ScrollTrigger.create({ trigger: el, start: "top 80%", once: true, onEnter: play });
  });

/* ------------------------------------------------------------------ copy */

$$<HTMLButtonElement>("[data-copy]").forEach((button) =>
  button.addEventListener("click", async () => {
    const text = $(`#${button.dataset.copy}`).textContent ?? "";
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select and copy";
    }
    window.setTimeout(() => (button.textContent = "Copy"), 1800);
  }),
);

/* ------------------------------------------------------------------ masthead */

const links = $$<HTMLAnchorElement>(".masthead__nav a");
const navObserver = new IntersectionObserver(
  (entries) =>
    entries.forEach((e) => {
      if (e.isIntersecting)
        links.forEach((a) => a.setAttribute("aria-current", String(a.getAttribute("href") === `#${e.target.id}`)));
    }),
  { rootMargin: "-40% 0px -55% 0px" },
);
links.forEach((a) => {
  const target = $(a.getAttribute("href") as string);
  if (target) navObserver.observe(target);
});
document.addEventListener("click", (e) => {
  const a = (e.target as Element).closest<HTMLAnchorElement>('a[href^="#"]');
  if (!a || !lenis) return;
  const target = $(a.getAttribute("href") as string);
  if (!target) return;
  e.preventDefault();
  lenis.scrollTo(target, { offset: -64, duration: 1.3 });
  history.pushState(null, "", a.getAttribute("href"));
});

/* ------------------------------------------------------------------ presenter mode */

// Stops: the opening, every beat of the journey, then the sections after it.
const stops: { label: string; el: HTMLElement; offset: number }[] = [
  { label: "The question", el: $(".opening"), offset: 0 },
  { label: "Why this task", el: $("#task"), offset: 0.12 },
  ...beats.map((b, i) => ({ label: `Journey ${i + 1}/${beats.length}`, el: b, offset: mobile ? 0.6 : 0.42 })),
  { label: "What it measures", el: $("#measures"), offset: 0.08 },
  { label: "Run your agent", el: $("#run"), offset: 0.08 },
];
const presenter = $("[data-presenter]");
const presenterLabel = $("[data-presenter-chapter]");
let stopIndex = 0;
const stopY = (s: (typeof stops)[number]) => s.el.getBoundingClientRect().top + window.scrollY - window.innerHeight * s.offset;
const goTo = (i: number) => {
  stopIndex = Math.max(0, Math.min(stops.length - 1, i));
  presenterLabel.textContent = stops[stopIndex].label;
  scrollToY(stopY(stops[stopIndex]), 1.4);
};
const togglePresenter = (on = !root.classList.contains("presenting")) => {
  root.classList.toggle("presenting", on);
  presenter.hidden = !on;
  if (on) {
    const y = window.scrollY;
    stopIndex = stops.reduce((best, s, i) => (Math.abs(stopY(s) - y) < Math.abs(stopY(stops[best]) - y) ? i : best), 0);
    goTo(stopIndex);
  }
};
if (new URLSearchParams(location.search).has("present")) togglePresenter(true);
document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "p" || e.key === "P") return togglePresenter();
  if (!root.classList.contains("presenting")) return;
  if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(e.key)) {
    e.preventDefault();
    goTo(stopIndex + 1);
  } else if (["ArrowLeft", "ArrowUp", "PageUp"].includes(e.key)) {
    e.preventDefault();
    goTo(stopIndex - 1);
  } else if (e.key === "Escape") togglePresenter(false);
});

document.fonts?.ready.then(() => ScrollTrigger.refresh());
