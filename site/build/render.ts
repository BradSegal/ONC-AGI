/**
 * Build-time rendering: every number on the page is baked into static HTML from the data
 * files, so the story, the smoke output and every figure read without JavaScript.
 * Client scripts only animate what is already there.
 */
import smoke from "../src/data/smoke.json" with { type: "json" };
import { staticBand } from "./band";
import { esc, journeyReplacements } from "./journey";

export function replacements(): Record<string, string> {
  return {
    "<!--@band-static-->": staticBand(),
    // the real `onc-agi smoke` output, verbatim
    "@@smoke-output@@": esc(smoke.output.trimEnd()),
    ...journeyReplacements(),
  };
}
