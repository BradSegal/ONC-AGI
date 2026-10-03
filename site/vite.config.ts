import { defineConfig, type Plugin } from "vite";
import { replacements } from "./build/render";

/** Bake data-derived HTML (results table, matrix, static figures, facts) into index.html. */
function bakeData(): Plugin {
  return {
    name: "onc-agi-bake-data",
    transformIndexHtml(html) {
      let out = html;
      for (const [marker, value] of Object.entries(replacements())) out = out.split(marker).join(value);
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
