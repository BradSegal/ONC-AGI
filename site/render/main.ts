/**
 * Renders the site's films in the browser, frame by frame at a fixed step, and encodes them with
 * WebCodecs: AV1 (smaller, for browsers that decode it) and H.264 (plays everywhere). Each file
 * goes back to build/render-media through window.saveFile.
 */
import { ArrayBufferTarget, Muxer } from "mp4-muxer";
import { FPS, HERO_SIZES, JOURNEY_SIZE, heroJobs, journey, type Job } from "./director";

declare global {
  interface Window {
    saveFile(name: string, base64: string): Promise<void>;
    renderFilms(only?: string): Promise<unknown>;
  }
}

type Codec = { key: "av1" | "avc"; codec: string; bitrate: number };
const codecsFor = (pixels: number): Codec[] => {
  const k = pixels / (1920 * 1080);
  return [
    // a locked-off camera over a dark field compresses well; these targets keep the edges crisp
    { key: "av1", codec: "av01.0.08M.08", bitrate: Math.round(2.1e6 * k) },
    { key: "avc", codec: "avc1.640032", bitrate: Math.round(4.2e6 * k) }, // level 5.0 for 60 fps
  ];
};

const toBase64 = (buf: ArrayBuffer) => {
  const bytes = new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
};

/** A WebP still from a top-down RGBA frame. */
async function still(rgba: Uint8Array, width: number, height: number): Promise<string> {
  const c = new OffscreenCanvas(width, height);
  c.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0);
  const blob = await c.convertToBlob({ type: "image/webp", quality: 0.86 });
  return toBase64(await blob.arrayBuffer());
}

/** Render one job to video files; returns what the page needs to know about the clip. */
async function render(job: Job, canvas: HTMLCanvasElement) {
  const { width, height } = canvas;
  const outputs = codecsFor(width * height).map((c) => {
    const target = new ArrayBufferTarget();
    const muxer = new Muxer({ target, video: { codec: c.key, width, height, frameRate: FPS }, fastStart: "in-memory", firstTimestampBehavior: "offset" });
    const encoder = new VideoEncoder({
      output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
      error: (e) => {
        throw e;
      },
    });
    encoder.configure({ codec: c.codec, width, height, bitrate: c.bitrate, framerate: FPS, latencyMode: "quality", bitrateMode: "variable", ...(c.key === "avc" ? { avc: { format: "avc" } } : {}) });
    return { c, target, muxer, encoder };
  });
  const scene = job.scene;
  const gl = canvas.getContext("webgl2")!;
  const row = width * 4;
  const pixels = new Uint8Array(row * height);
  const flipped = new Uint8Array(row * height);
  if (job.start) {
    job.start();
  }
  const cues = [...job.cues].sort((a, b) => a.at - b.at);
  // labels are stored compactly: each label's text once, then positions only when something moved
  const defs: Record<string, { text: string; id?: string; cls?: string }> = {};
  const timeline: { t: number; p: [string, number, number][] }[] = [];
  let lastSample = "";
  const sample = (t: number) => {
    const placed = scene.placedLabels();
    for (const l of placed) defs[l.key] ??= { text: l.text, id: l.id, cls: l.cls };
    const p = placed.map((l) => [l.key, +l.x.toFixed(3), +l.y.toFixed(3)] as [string, number, number]);
    const key = JSON.stringify(p);
    if (key !== lastSample) timeline.push({ t: +t.toFixed(2), p });
    lastSample = key;
    return p;
  };
  let quiet = 0; // consecutive frames with nothing moving
  let f = 0;
  const dt = 1 / FPS;
  for (;;) {
    const t = f / FPS;
    while (cues.length && cues[0].at <= t + 1e-9) cues.shift()!.run();
    const moving = scene.advance(dt);
    if (job.labelEvery && f % job.labelEvery === 0) sample(t);
    // read the finished frame straight from the drawing buffer: a frame built from the canvas
    // element would be the last one the browser presented, not the one just drawn
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    for (let y = 0; y < height; y++) flipped.set(pixels.subarray((height - 1 - y) * row, (height - y) * row), y * row);
    const frame = new VideoFrame(flipped, { format: "RGBA", codedWidth: width, codedHeight: height, timestamp: Math.round((f * 1e6) / FPS), duration: Math.round(1e6 / FPS) });
    for (const o of outputs) {
      while (o.encoder.encodeQueueSize > 8) await new Promise((r) => setTimeout(r, 1));
      o.encoder.encode(frame, { keyFrame: f % FPS === 0 });
    }
    frame.close();
    if (job.poster === f) await window.saveFile(`${job.name}.webp`, await still(flipped, width, height));
    // STILLS='{"<clip>": [frames]}' saves chosen frames, for checking a render without decoding it
    if ((window as { __stills?: Record<string, number[]> }).__stills?.[job.name]?.includes(f))
      await window.saveFile(`debug-${job.name}-f${f}.webp`, await still(flipped, width, height));
    f++;
    quiet = moving ? 0 : quiet + 1;
    if (job.duration !== undefined ? f >= Math.round(job.duration * FPS) : (f / FPS >= job.settle!.min && quiet >= 6 && !cues.length) || f / FPS >= job.settle!.max) break;
  }
  const end = sample(f / FPS);
  // a still of every clip's last frame: the settled state, shown when a reader jumps straight to it
  await window.saveFile(`${job.name}-end.webp`, await still(flipped, width, height));
  const sizes: Record<string, number> = {};
  for (const o of outputs) {
    await o.encoder.flush();
    o.encoder.close();
    o.muxer.finalize();
    const buf = o.target.buffer;
    sizes[o.c.key] = buf.byteLength;
    await window.saveFile(`${job.name}.${o.c.key}.mp4`, toBase64(buf));
  }
  return { name: job.name, duration: +(f / FPS).toFixed(3), sizes, defs, labels: end, timeline: job.labelEvery ? timeline : undefined, tags: job.tags };
}

window.renderFilms = async (only?: string) => {
  const results: Record<string, unknown> = {};
  const canvasOf = (w: number, h: number) => {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    document.body.append(c);
    return c;
  };
  const want = (name: string) => !only || name.includes(only);
  for (const shape of Object.keys(HERO_SIZES) as (keyof typeof HERO_SIZES)[]) {
    if (!want(`hero-${shape}`)) continue;
    const [w, h] = HERO_SIZES[shape];
    const canvas = canvasOf(w, h);
    for (const job of heroJobs(shape, canvas)) results[job.name] = await render(job, canvas);
    canvas.remove();
  }
  if (want("journey")) {
    const canvas = canvasOf(...JOURNEY_SIZE);
    const { jobs, captions } = journey(canvas);
    for (const job of jobs) if (want(job.name)) results[job.name] = await render(job, canvas);
    results.captions = captions;
    canvas.remove();
  }
  return results;
};
