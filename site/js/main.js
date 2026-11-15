// Entry point: load the exported data, then mount each section's visual.
import { mountVideos } from "./videos.js";
import { mountEra1, mountTd3Chart } from "./charts.js";
import { mountCampaign } from "./campaign.js";
import { mountPaired } from "./paired.js";
import { mountDuel } from "./duel.js";

const load = (name) => fetch(`data/${name}.json`).then((r) => {
  if (!r.ok) throw new Error(`data/${name}.json: HTTP ${r.status}`);
  return r.json();
});

function safely(label, fn) {
  try { fn(); } catch (err) { console.error(`[${label}]`, err); }
}

async function start() {
  mountVideos();
  const [replays, era1, campaign, td3, paired] = await Promise.all(
    ["replays", "era1", "campaign", "td3", "paired"].map(load));

  safely("era1", () => mountEra1(era1));
  safely("campaign", () => mountCampaign(campaign));
  safely("td3", () => mountTd3Chart(td3));
  safely("paired", () => mountPaired(paired));
  safely("duel", () => mountDuel(replays, paired));

  // The 3D hero loads three.js from a CDN; keep the page working if that fails.
  import("./hero3d.js")
    .then((m) => m.mountHero(replays.hero))
    .catch((err) => console.error("[hero3d]", err));
}

start().catch((err) => console.error(err));
