# ONC-AGI website

The public benchmark site tells one story: the premise, why the task is hard, how the oracle certifies answers, how the score works, why gaming fails, the two modes, then where agents stand. It is a static build:
- Vite and TypeScript, with no UI framework;
- GSAP and Lenis for scroll-driven motion;
- canvas for the significance landscape, the oracle figure and the acquisition matrix.

Every number on the page is baked into the HTML at build time from `src/data/`. The story, the results table and the key figures read without JavaScript. Under `prefers-reduced-motion`, every figure shows its final state.

## Run it

```bash
cd site
npm install
npm run data      # rebuild src/data/*.json from the runtime (needs the repo's uv environment)
npm run dev       # local development
npm run build     # type-check and build into dist/
npm run preview   # serve dist/
```

## Present it

Press **P**, or open the page with `?present`, to enter presenter mode. Right, Down, Space and PageDown step forward, and a presentation clicker sends PageDown. Left, Up and PageUp step back. Esc exits. Each step is one chapter state, so the 90-second pitch is a fixed number of clicks.

## Data contract

`scripts/prepare_data.py` writes two files:

- **`src/data/results.json`** (`onc-agi-site-results/1`) has one row per agent: `overall`, `by_mode` (`full_access`, `sequential`), `by_mechanic` and `by_tier`, each with Discovery Score, interval, Find, Restraint, Strict, leak rate and data cost.
  - `status` names the release stage (`first-pass` now). Only rows backed by scorecards exist; nothing is invented.
  - Each row has a `provenance`: `toy-fixture`, `benchmark` or `final` (frontier rows take it from `meta.json`, default `benchmark`).
- **`src/data/oracle.json`** holds the oracle demonstration: the null-maximum threshold and the detection-rate curves, computed on a toy world.

## Updating with final runs

1. **Frontier agents.** For each agent, add a folder `results/frontier/<agent-id>/` containing:
   - `meta.json` (`{"label": "...", "model": "...", "harness": "..."}`);
   - the runtime's `Scorecard` JSON for each mode, `full_access.json` and `sequential.json`;
   - optionally `overall.json`. Without it, the overall row is the mean of the two modes, marked as such.

2. **Baselines, references and cheaters.** Point the script at a public-train benchmark store:

   ```bash
   ONC_AGI_SITE_STORE=/path/to/store ONC_AGI_SITE_PUBLISHED_WORLDS=<worlds in the release downloads> npm run data
   ```

   Without it, these rows come from the toy fixtures and are labelled "toy worlds".
3. **Rebuild.** Run `npm run build`. The design does not change.

Per-mechanic results need per-world scores. Only public-train scorecards carry them; eval-tier scorecards leave the mechanic matrix row empty, which the page shows as "—".

## What is synthetic

- **The opening landscape** (20,530 points) is a seeded synthetic illustration, labelled on the page.
- **The oracle figure** is computed by the real runtime on the `toy-stand-in` world, using marginal statistics as an illustration of the method.
- **The task card** is the real `toy-driver-seq` world card.
- **Frontier rows** appear only for models with scorecards in `results/frontier/`.
