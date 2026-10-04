# ONC-AGI website

The public benchmark site centres the task: the question and premise, why it matters, one recorded synthetic task replayed from world construction to score, what makes the task hard, then how to run an agent. Results are published separately. It is a static build:
- Vite and TypeScript, with no UI framework;
- GSAP and Lenis for scroll-driven motion;
- D3 and Observable Plot at build time for the journey's 2D figures, baked into the HTML as SVG;
- films of the 3D cohort scenes, rendered at build time with Three.js and played as video.

Every number on the page is baked in at build time from `src/data/`. The story, the smoke output and every 2D figure read without JavaScript. Under `prefers-reduced-motion`, or when the browser asks to save data, the films show their settled stills.

## Run it

```bash
cd site
npm install
npm run data      # rebuild src/data/*.json from the runtime (see Data contract)
npm run render    # re-render the films into public/media/ (needs a GPU; see Films)
npm run dev       # local development
npm run build     # type-check and build into dist/
npm run check     # verify the built page against its source data
npm run preview   # serve dist/
```

## Films

The opening and the journey's 3D beats are films, so every visitor sees the same full-quality frames whatever their hardware, including laptops with no usable GPU. The page itself runs no WebGL: it plays video and draws the labels as live HTML.

- `render/cohort3d.ts` is the scene: one cube per patient per measurement, evidence towers, the threshold plane, soft shadows, ambient occlusion and a racked depth of field. It runs only at build time.
- `render/director.ts` stages every clip: the opening's intro and its seamless 20-second loop (wide and square), and one clip per journey transition, in both scroll directions and into and out of the 2D figures.
- `render/main.ts` renders each clip frame by frame at a fixed 1/30 s step and encodes it with WebCodecs as AV1 (smaller) and H.264 (plays everywhere). It records each label's position in the frame, and the captions' timing.
- `build/render-media.mjs` drives the render in Chromium on a GPU and writes `public/media/*.mp4`, a settled `-end.webp` still per clip, posters, and `src/data/media.json`. `npm run render -- <filter>` re-renders the clips whose names contain the filter. It refuses to run on a software rasteriser.

Renders are deterministic: fixed time step, seeded jitter and recorded runtime data. Re-run `npm run render` after changing the scene, the director or `src/data/`.

`src/lib/film.ts` plays them:
- **Codec:** AV1 where the browser decodes it, otherwise H.264.
- **Opening:** the intro plays once, then the loop takes over on a matching frame.
- **Journey:** each step change plays its transition clip; jumps show the target's settled still.
- **Labels:** placed through the video's crop from the recorded coordinates.
- **Loading:** a clip is fetched only when a reader may play it next.
- **Power:** films pause off-screen and in hidden tabs.

## Present it

Press **P**, or open the page with `?present`, to enter presenter mode. Right, Down, Space and PageDown step forward, and a presentation clicker sends PageDown. Left, Up and PageUp step back. Esc exits.

## Data contract

`scripts/prepare_data.py` computes these with the public runtime. Run it from the ONC-AGI checkout, where the runtime is installed: `PYTHONPATH=tools uv run --no-sync python <path to>/site/scripts/prepare_data.py`.

- **`smoke.json`** holds the stdout of `onc-agi smoke`, printed verbatim under "Expected output".
- **`oracle.json`** holds the oracle demonstration: detection rates by cohort size and the threshold from 400 no-signal outcomes, on a toy world.
- **`journey.json`** holds one recorded sequential episode on `toy-stand-in-seq` and its scoring.
- **`play.json`** holds the "try one" task: evidence at each purchase size and the scoring rule. It is self-checked against the real scorer on every list of up to three measurements.
- **`catalogue.json`** holds one toy world per mechanism, with its key roles and what a naive screen does on it.
- **`hero.json`** holds the opening's two toy worlds in full.

## What is synthetic

Every world on the page is a synthetic toy fixture: the journey's `toy-stand-in-seq`, and the opening's `toy-driver` (a planted cause) and `toy-null-a` (no signal). The opening lays its columns out right to left for composition. It shows no column order and labels only data-type bands. The detection figure uses marginal tests to illustrate the oracle's method; the oracle itself refits the planted model.

## Editable figures

`npm run build` exports the journey's 2D figures as standalone SVGs to `dist/figures/`, linked beneath the journey. `build/visuals.ts` owns their composition; `build/journey.ts` binds the narrative's numbers to the source data.
