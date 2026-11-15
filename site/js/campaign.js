// The 63 Nations campaign maps as tiny track outlines; finished maps lit. Click one to replay its run.
import { el } from "./charts.js";
import { createReplay, fetchJson, SPEED_LEGEND, SPEED_LEGEND_MAX } from "./replay.js";

const TIER_ORDER = ["White", "Green", "Blue", "Red", "Black"];
const DEFAULT_MAP = "A05-Race";   // the story map: first finished on the official-replay line
const PLAYBACK = 4;

function outlinePath(cl) {
  const xs = cl.map((p) => p[0]), zs = cl.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
  const span = Math.max(maxX - minX, maxZ - minZ) || 1;
  const ox = (span - (maxX - minX)) / 2, oz = (span - (maxZ - minZ)) / 2;
  const sx = (v) => (((v - minX + ox) / span) * 92 + 4).toFixed(1);
  const sz = (v) => (((v - minZ + oz) / span) * 92 + 4).toFixed(1);
  return cl.map((p, i) => `${i ? "L" : "M"}${sx(p[0])},${sz(p[1])}`).join("");
}

function tooltip() {
  const tip = document.createElement("div");
  tip.className = "tip";
  document.body.appendChild(tip);
  return tip;
}

const resultText = (m) => (m.finished ? `finished in ${m.time.toFixed(2)} s` : `stopped at ${m.completion.toFixed(0)}%`);

export function mountCampaign(data) {
  const root = document.getElementById("campaign");
  const tip = tooltip();
  const explorer = mountExplorer(data);
  for (const tier of TIER_ORDER) {
    const row = document.createElement("div");
    row.className = "crow";
    const label = document.createElement("span");
    label.className = "clabel";
    label.textContent = tier;
    row.appendChild(label);
    const maps = data.maps.filter((m) => m.tier === tier).sort((a, b) => a.map.slice(0, 3).localeCompare(b.map.slice(0, 3)));
    for (const m of maps) row.appendChild(makeTile(m, tip, explorer.select));
    root.appendChild(row);
  }
  fillTiers(document.getElementById("tier-list"), data);
  explorer.markSelected();
}

function makeTile(m, tip, select) {
  const tile = document.createElement("button");
  tile.className = `tile ${m.finished ? "ok" : "no"}`;
  tile.dataset.map = m.map;
  tile.setAttribute("aria-label", `${m.map}, ${m.tier}: ${resultText(m)}. Replay it`);
  const svg = el("svg", { viewBox: "0 0 100 100", "aria-hidden": "true" }, tile);
  el("path", { d: outlinePath(m.cl) }, svg);
  const tag = document.createElement("span");
  tag.textContent = m.map.slice(1, 3);
  tile.appendChild(tag);
  const show = () => {
    tip.textContent = `${m.map} · ${m.tier} · ${resultText(m)}`;
    const r = tile.getBoundingClientRect();
    tip.style.left = `${Math.max(8, Math.min(r.left, innerWidth - 268))}px`;
    tip.style.top = `${r.top - 36}px`;
    tip.style.opacity = 1;
  };
  const hide = () => { tip.style.opacity = 0; };
  tile.addEventListener("mouseenter", show);
  tile.addEventListener("focus", show);
  tile.addEventListener("mouseleave", hide);
  tile.addEventListener("blur", hide);
  tile.addEventListener("click", () => { hide(); select(m.map, true); });
  return tile;
}

function mountExplorer(data) {
  const replay = createReplay({
    canvas: document.getElementById("ex-canvas"),
    scrub: document.getElementById("ex-scrub"),
    playBtn: document.getElementById("ex-play"),
    timeEl: document.getElementById("ex-time"),
    playback: PLAYBACK,
  });
  const ramp = document.getElementById("ex-ramp");
  ramp.style.background = `linear-gradient(90deg, ${SPEED_LEGEND.join(", ")})`;
  document.getElementById("ex-ramp-max").textContent = `${Math.round(SPEED_LEGEND_MAX * 3.6)} km/h`;
  const state = { map: DEFAULT_MAP, request: 0 };

  const markSelected = () => document.querySelectorAll("#campaign .tile").forEach((t) =>
    t.setAttribute("aria-pressed", String(t.dataset.map === state.map)));

  async function select(map, scroll = false) {
    state.map = map;
    markSelected();
    const m = data.maps.find((x) => x.map === map);
    document.getElementById("ex-title").textContent = map;
    const sub = document.getElementById("ex-sub");
    sub.textContent = `${m.tier} · ${resultText(m)}`;
    sub.className = m.finished ? "ok" : "fail";
    if (scroll) document.getElementById("explorer").scrollIntoView({ block: "nearest" });
    const request = ++state.request;
    try {
      const d = await fetchJson(`data/campaign/${map}.json`);
      if (request !== state.request) return;
      replay.show({ cl: d.cl, tracks: [{ run: d.run, bySpeed: true }] });
      const run = d.run;
      document.getElementById("ex-note").textContent =
        `One run of the base stack (code v3b, before any learned model) on the official-replay line, replayed at ` +
        `${PLAYBACK}×. ${run.impacts ?? 0} suspected impacts, ${run.recovery ?? 0} recovery steps.`;
    } catch (err) {
      console.error("[explorer]", err);
      document.getElementById("ex-note").textContent = `Could not load ${map}: ${err.message}`;
    }
  }

  const io = new IntersectionObserver(([e]) => {
    if (!e.isIntersecting) return;
    io.disconnect();
    if (!state.request) select(state.map);   // skip if a tile was already clicked
  }, { rootMargin: "600px" });
  io.observe(document.getElementById("explorer"));
  return { select, markSelected };
}

function fillTiers(list, data) {
  for (const tier of TIER_ORDER) {
    const ms = data.maps.filter((m) => m.tier === tier);
    const fin = ms.filter((m) => m.finished).length;
    const li = document.createElement("li");
    li.innerHTML = `<span>${tier}</span><span class="bar"><i style="width:${(100 * fin) / ms.length}%"></i></span><b>${fin}/${ms.length}</b>`;
    list.appendChild(li);
  }
  const li = document.createElement("li");
  li.className = "total";
  li.innerHTML = `<span>All</span><span class="bar"><i style="width:${(100 * data.finished) / data.total}%"></i></span><b>${data.finished}/${data.total}</b>`;
  list.appendChild(li);
}
