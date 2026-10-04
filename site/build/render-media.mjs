// Renders the site's films (build time, needs a GPU): serves render.html with Vite, drives it in
// Chromium, and writes the videos and stills to public/media/ and their timing and labels to
// src/data/media.json. Usage: npm run render [-- <name filter>]
import { createServer } from "vite";
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";

const only = process.argv[2];
const out = path.resolve("public/media");
fs.mkdirSync(out, { recursive: true });
const server = await createServer({ server: { port: 4499, strictPort: true, host: "127.0.0.1" }, logLevel: "warn" });
await server.listen();
const browser = await chromium.launch({ args: ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"] });
try {
  const page = await browser.newPage();
  page.on("console", (m) => m.type() === "error" && console.error("[page]", m.text()));
  page.on("pageerror", (e) => console.error("[page]", e.message));
  await page.exposeFunction("saveFile", (name, b64) => fs.writeFileSync(path.join(out, name), Buffer.from(b64, "base64")));
  if (process.env.STILLS) await page.addInitScript((s) => (window.__stills = JSON.parse(s)), process.env.STILLS);
  await page.goto("http://127.0.0.1:4499/render.html", { waitUntil: "load" });
  const renderer = await page.evaluate(() => {
    const gl = document.createElement("canvas").getContext("webgl2");
    const i = gl.getExtension("WEBGL_debug_renderer_info");
    return gl.getParameter(i.UNMASKED_RENDERER_WEBGL);
  });
  if (/swiftshader|llvmpipe/i.test(renderer)) throw new Error(`no GPU available for rendering (${renderer})`);
  console.log("rendering on", renderer);
  const t0 = Date.now();
  const results = await page.evaluate((only) => window.renderFilms(only), only);
  const file = path.resolve("src/data/media.json");
  const previous = only && fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : {};
  fs.writeFileSync(file, JSON.stringify({ ...previous, ...results }, null, 1) + "\n");
  for (const [name, r] of Object.entries(results)) if (r.sizes) console.log(`${name}: ${r.duration}s, av1 ${(r.sizes.av1 / 1024).toFixed(0)} KB, h264 ${(r.sizes.avc / 1024).toFixed(0)} KB`);
  console.log(`rendered in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
} finally {
  await browser.close();
  await server.close();
}
