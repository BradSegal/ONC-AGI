import { execSync } from "node:child_process";
import { defineConfig, type Plugin } from "vite";
import { replacements } from "./build/render";
import { exportFigures } from "./build/visuals";
import { sceneData } from "./build/scene";

/**
 * The commit this page was built from, stamped into <meta name="onc-agi-build"> so anyone can
 * confirm that GitHub Pages and onc-agi.com serve the same build. Both hosts build the same repository.
 */
function buildId(): string {
  const sha = process.env.GITHUB_SHA ?? process.env.RAILWAY_GIT_COMMIT_SHA;
  if (sha) return sha.slice(0, 12);
  try {
    return execSync("git rev-parse --short=12 HEAD", { stdio: ["ignore", "pipe", "ignore"] })
      .toString()
      .trim();
  } catch {
    return "unknown";
  }
}

/** Bake data-derived HTML (results table, matrix, static figures, facts) into index.html. */
function bakeData(): Plugin {
  return {
    name: "onc-agi-bake-data",
    resolveId(id) {
      if (id === "virtual:scene") return "\0virtual:scene";
    },
    load(id) {
      if (id === "\0virtual:scene") return `export default ${JSON.stringify(sceneData())};`;
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const match = req.url?.match(
          /^\/figures\/(cohort|recovery|threshold|evidence|scoring|answers)\.svg$/,
        );
        if (!match) return next();
        res.setHeader("Content-Type", "image/svg+xml");
        res.end(exportFigures()[match[1]]);
      });
    },
    generateBundle() {
      for (const [name, source] of Object.entries(exportFigures())) {
        this.emitFile({
          type: "asset",
          fileName: `figures/${name}.svg`,
          source,
        });
      }
    },
    transformIndexHtml(html) {
      let out = html.split("@@build@@").join(buildId());
      for (const [marker, value] of Object.entries(replacements()))
        out = out.split(marker).join(value);
      const left = out.match(/@@[a-z-]+@@|<!--@[a-z-]+-->/);
      if (left) throw new Error(`unreplaced data marker ${left[0]}`);
      return out;
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [bakeData()],
  build: { target: "es2022", assetsInlineLimit: 0 },
  // reachable from other machines on the tailnet (MagicDNS names end in .ts.net)
  server: { host: true, allowedHosts: [".ts.net"] },
  preview: { host: true, port: 4317, allowedHosts: [".ts.net"] },
});
