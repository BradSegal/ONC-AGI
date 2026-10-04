// Verifies the built page against its source data: structure, exact mark positions and downloads.
import { JSDOM } from "jsdom";
import fs from "node:fs";
import assert from "node:assert/strict";

const json = (name) => JSON.parse(fs.readFileSync(`src/data/${name}.json`));
const j = json("journey");
const o = json("oracle");
const html = fs.readFileSync("dist/index.html", "utf8");
const doc = new JSDOM(html).window.document;
const close = (a, b, msg) => assert(Math.abs(a - b) < 1e-6, `${msg}: ${a} vs ${b}`);

// Structure: five panes, fourteen beats, every beat pointing at an existing pane.
const panes = [...doc.querySelectorAll(".panes > .pane")].map((p) => p.dataset.pane);
assert.deepEqual(panes, ["build", "certify", "receive", "act", "score"]);
const beats = [...doc.querySelectorAll("article.beat")];
assert.equal(beats.length, 14);
for (const b of beats) assert(panes.includes(b.dataset.pane), `beat points at missing pane ${b.dataset.pane}`);

// Results are published separately: no scorecard values on this page.
for (const v of Object.values(j.across.summary).filter((x) => typeof x === "number" && x > 0 && x < 1))
  assert(!html.includes(`Discovery Score ${v.toFixed(2)}`), "a Discovery Score value leaked onto the page");
assert(!/font-family:\s*Arial/i.test(html), "a figure still declares Arial");

// Act: every point sits at its recorded evidence in every frame.
const analyses = j.steps.filter((s) => s.kind === "analyse");
const frames = [...doc.querySelectorAll(".act-frame")];
assert.equal(frames.length, analyses.length);
let actPoints = 0;
frames.forEach((frame, i) => {
  const points = frame.querySelectorAll("[data-f]");
  assert.equal(points.length, j.card.features.length);
  for (const p of points) {
    close(Number(p.getAttribute("y")), 352 - analyses[i].evidence[p.dataset.f] * 29.5 - 4, `act ${p.dataset.f}`);
    actPoints++;
  }
});

// Detection curves: Plot's square symbols at log-x, linear-y positions of every observation.
const R = { width: 720, height: 430, left: 64, right: 150, top: 36, bottom: 56 };
const dots = [...doc.querySelectorAll('.recovery-plot [aria-label="dot"] path')];
const observations = Object.values(o.curves).flat();
assert.equal(dots.length, observations.length);
dots.forEach((dot, i) => {
  const [x, y] = dot.getAttribute("transform").match(/[\d.]+/g).map(Number);
  const p = observations[i];
  close(x, R.left + ((Math.log(p.n) - Math.log(30)) / (Math.log(240) - Math.log(30))) * (R.width - R.left - R.right), "recovery x");
  close(y, R.height - R.bottom - (p.rate / 1.04) * (R.height - R.bottom - R.top), "recovery y");
});

// Threshold: one square per no-signal cohort and per driver simulation, and the bar at the 95th percentile.
assert.equal(doc.querySelectorAll(".nulls-plot .null-stack rect").length, o.null_max.length);
assert.equal(doc.querySelectorAll(".nulls-plot .driver-stack rect").length, o.replicate_z.length);
const bar = doc.querySelector(".nulls-plot line[stroke='#d7263d']");
close(Number(bar.getAttribute("x1")), 40 + (o.threshold / 10.5) * 660, "threshold position");

// Downloads: valid, labelled, finite, final-state SVGs.
const names = ["cohort", "recovery", "threshold", "evidence", "scoring", "answers"];
for (const name of names) {
  const raw = fs.readFileSync(`dist/figures/${name}.svg`, "utf8");
  const d = new JSDOM(raw, { contentType: "image/svg+xml" }).window.document;
  assert.equal(d.documentElement.namespaceURI, "http://www.w3.org/2000/svg");
  assert(d.documentElement.getAttribute("aria-label"), `${name}.svg has no label`);
  assert(!raw.includes("NaN"), `${name}.svg contains NaN`);
  if (name === "evidence") assert.equal(d.querySelectorAll(".act-frame").length, 1);
}

console.log(
  `PASS: ${panes.length} panes; ${beats.length} beats; ${actPoints} exact evidence positions; ${dots.length} exact detection positions; ` +
    `${o.null_max.length} + ${o.replicate_z.length} threshold squares; ${names.length} SVG downloads; no scorecard values.`,
);
