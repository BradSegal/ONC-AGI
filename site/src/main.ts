import "@fontsource-variable/archivo/standard.css";
import "@fontsource-variable/azeret-mono";
import "./styles/main.css";

import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";

import { rollTo } from "./lib/roll";
import { opening as filmOpening, stage as filmStage, type Stage } from "./lib/film";
import { tryOne } from "./lib/try";

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

/* ------------------------------------------------------------------ films
   The 3D scenes are rendered at build time and played as video, so every visitor gets the same
   full-quality frames, with or without a capable GPU. */

let stage3d: Stage | null = null;
let shown: [string, number] = ["build", 0];
const hero = document.querySelector<HTMLElement>("[data-scene-hero]");
if (hero) {
  filmOpening(hero, reduced);
  root.classList.add("has-hero-3d");
}
// the journey's films load only as the journey approaches, so the opening loads alone
if (!mobile) {
  const near = new IntersectionObserver(
    (entries) => {
      if (!entries[0].isIntersecting) return;
      near.disconnect();
      stage3d = filmStage($(".panes"), reduced);
      stage3d.preload();
      root.classList.add("has-3d");
      stage3d.show(...shown);
    },
    { rootMargin: "100% 0px" },
  );
  near.observe($(".journey"));
}

/* ------------------------------------------------------------------ opening */

if (!reduced) gsap.from(".opening__text > :not(.scene-tag)", { y: 18, opacity: 0, duration: 1, stagger: 0.08, ease: "power2.out" });

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
  shown = [pane, state];
  stage3d?.show(pane, state);
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

let actLabels: SVGTextElement[][] | undefined;
let actPoints: Map<string, {y: number; cls: string; title: string}>[] | undefined;
const enter: Record<string, (el: HTMLElement, state: number, first: boolean) => void> = {
  build(el, _state, first) {
    if (first && !reduced) gsap.from($$(".r", el), { opacity: 0, x: -6, duration: 0.5, stagger: 0.012, ease: "expo.out" });
  },
  certify(el, state, first) {
    if (reduced) return;
    if (first)
      $$<SVGGeometryElement>(".recovery-plot .ci-line", el).forEach((line, i) => {
        const len = line.getTotalLength();
        gsap.fromTo(line, { strokeDasharray: len, strokeDashoffset: len }, { strokeDashoffset: 0, duration: 1.2, delay: (i % 3) * 0.18, ease: "power2.out" });
      });
    // the no-signal cohorts stack up first, then the driver's simulations land beyond the bar
    if (state === 1 && !el.dataset.stacked) {
      el.dataset.stacked = "1";
      gsap.from($$(".null-stack rect", el), { opacity: 0, y: -24, duration: 0.5, stagger: { amount: 1.1, from: "start" }, ease: "power2.out" });
      gsap.from($$(".driver-stack rect", el), { opacity: 0, y: -24, duration: 0.5, delay: 1.2, stagger: { amount: 0.7 }, ease: "power2.out" });
    }
  },
  receive(el, _state, first) {
    if (first && !reduced) gsap.from($$(".worldcard > *", el), { opacity: 0, y: 8, duration: 0.6, stagger: 0.06, ease: "expo.out" });
  },
  act(el, state) {
    // One live frame; its points move to the recorded evidence at each analysis.
    const frames = $$<SVGGElement>(".act-frame", el);
    const live = frames[0];
    const k = Math.min(state, frames.length - 1);
    // Preserve original coordinates before the first frame becomes the live animation surface.
    actPoints ??= frames.map(frame => new Map($$<SVGRectElement>(".pt", frame).map(point => [point.dataset.f!, {y: Number(point.getAttribute("y")), cls: point.getAttribute("class")!, title: point.querySelector("title")?.textContent ?? ""}])));
    frames.forEach((f, i) => f.classList.toggle("is-shown", i === 0));
    // each frame's labels, captured once before the live frame is first rewritten
    actLabels ??= frames.map((f) => $$<SVGTextElement>(".pt-label, .frame-tag, .pt-ghost", f).map((t) => t.cloneNode(true) as SVGTextElement));
    $$<SVGRectElement>(".pt", live).forEach((pt) => {
      const target = actPoints![k].get(pt.dataset.f!)!;
      gsap.killTweensOf(pt);
      tween(pt, { attr: { y: target.y }, duration: 0.8, ease: "power2.inOut" });
      pt.setAttribute("class", target.cls);
      const title = pt.querySelector("title");
      if (title) title.textContent = target.title;
    });
    $$<SVGTextElement>(".pt-label, .frame-tag, .pt-ghost", live).forEach((t) => t.remove());
    actLabels[k].forEach((t) => {
      const copy = t.cloneNode(true) as SVGTextElement;
      live.append(copy);
      if (!reduced) gsap.from(copy, { opacity: 0, duration: 0.6, delay: 0.5 });
    });
  },
  score(el, state) {
    const visible = [3, 5, 7, 8][state] ?? 8;
    // gates still to come stay faintly visible; CSS brightens each one as its beat arrives
    $$<HTMLElement>("[data-k]", el).forEach((row) => row.classList.toggle("is-on", Number(row.dataset.k) < visible));
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
// Above the journey the stage always rests on its first state, however the reader arrived.
ScrollTrigger.create({ trigger: ".journey__column", start: mobile ? "top 72%" : "top 58%", onLeaveBack: () => activate(beats[0]) });
const activateAtScroll = () => activate([...beats].reverse().find(beat => beat.getBoundingClientRect().top <= innerHeight * (mobile ? 0.72 : 0.58)) ?? beats[0]);
activateAtScroll();
// a jump (anchor, presenter, scrollbar drag) can skip the toggles; settle on the true beat once scrolling ends
ScrollTrigger.addEventListener("scrollEnd", activateAtScroll);

// Inline (phone) reading: every figure sits after its text in its final state and plays its
// entrance once on arrival; the act figure replays the episode 30 -> 60 -> 90 patients.
const FINAL: Record<string, number> = { build: 3, certify: 1, receive: 0, act: 0, score: 3 };
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

/* ------------------------------------------------------------------ try one */

const tryRoot = document.querySelector<HTMLElement>("#try");
if (tryRoot) tryOne(tryRoot);

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
  { label: "Try one", el: $("#try"), offset: 0.08 },
  ...beats.map((b, i) => ({ label: `Journey ${i + 1}/${beats.length}`, el: b, offset: mobile ? 0.6 : 0.42 })),
  { label: "What makes it hard", el: $("#measures"), offset: 0.08 },
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

document.fonts?.ready.then(() => {
  ScrollTrigger.refresh();
  // Resolve deep links after fonts and the scroll layout establish their final sizes.
  const target = location.hash.startsWith("#j-") ? document.getElementById(location.hash.slice(1)) : null;
  if (target) {
    lenis?.resize();
    window.scrollTo({top: target.getBoundingClientRect().top + scrollY - 64, behavior: "instant"});
  }
  activateAtScroll();
});
