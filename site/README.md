# ONC-AGI website

The public benchmark site tells one story: the premise, why the task is hard, one real task replayed from world construction to score, what the benchmark measures, then how to run an agent. It is a static build:
- Vite and TypeScript, with no UI framework;
- GSAP and Lenis for scroll-driven motion;
- canvas for the significance landscape, the oracle figure and the acquisition matrix.

Every number on the page is baked into the HTML at build time from `src/data/`. The story, the smoke output and the key figures read without JavaScript. Under `prefers-reduced-motion`, every figure shows its final state.

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

`scripts/prepare_data.py` writes three files, each computed now by the public runtime:

- **`src/data/smoke.json`** holds the stdout of `onc-agi smoke`, run in process on the packaged fixture worlds. The "Expected output" block prints it verbatim.
- **`src/data/oracle.json`** holds the oracle demonstration: the null-maximum threshold and the detection-rate curves, computed on a toy world.
- **`src/data/journey.json`** holds one recorded sequential episode on a toy world, its scoring, and the same policy played on every toy world.

Run `npm run build` after `npm run data` to bake them into the page.

## What is synthetic

- **The opening landscape** (20,530 points) is a seeded synthetic illustration, labelled on the page.
- **The oracle figure** is computed by the real runtime on the `toy-stand-in` world, using marginal statistics as an illustration of the method.
- **The task card** is the real `toy-driver-seq` world card.
