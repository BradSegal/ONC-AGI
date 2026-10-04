/**
 * Build-time rendering: every number on the page is baked into static HTML from the data
 * files, so the story, the smoke output and every figure read without JavaScript.
 * Client scripts only animate what is already there.
 */
import smoke from "../src/data/smoke.json" with { type: "json" };

import { esc, journeyReplacements } from "./journey";
import { tryMarkup } from "./try";
import { failuresMarkup } from "./failures";

const DOWNLOADS: [string, string][] = [
  ["cohort", "Cohort"],
  ["recovery", "Detection"],
  ["threshold", "Threshold"],
  ["evidence", "Evidence"],
  ["scoring", "Scoring"],
  ["answers", "Other answers"],
];

export function replacements(): Record<string, string> {
  const output = smoke.output.trimEnd();
  return {
    "<!--@figure-downloads-->": DOWNLOADS.map(([file, label]) => `<a href="./figures/${file}.svg" download>${label}</a>`).join(" "),
    // the real `onc-agi smoke` output, verbatim
    "@@smoke-output@@": esc(output),
    "@@smoke-last@@": esc(output.split("\n").at(-1) ?? ""),
    ...journeyReplacements(),
    ...tryMarkup(),
    "<!--@failures-->": failuresMarkup(),
  };
}
