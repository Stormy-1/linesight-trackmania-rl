// The 63 Nations campaign maps as tiny track outlines; finished maps lit.
import { el } from "./charts.js";

const TIER_ORDER = ["White", "Green", "Blue", "Red", "Black"];

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

export function mountCampaign(data) {
  const root = document.getElementById("campaign");
  const tip = tooltip();
  for (const tier of TIER_ORDER) {
    const row = document.createElement("div");
    row.className = "crow";
    const label = document.createElement("span");
    label.className = "clabel";
    label.textContent = tier;
    row.appendChild(label);
    const maps = data.maps.filter((m) => m.tier === tier).sort((a, b) => a.map.slice(0, 3).localeCompare(b.map.slice(0, 3)));
    for (const m of maps) row.appendChild(makeTile(m, tip));
    root.appendChild(row);
  }
  fillTiers(document.getElementById("tier-list"), data);
}

function makeTile(m, tip) {
  const tile = document.createElement("div");
  tile.className = `tile ${m.finished ? "ok" : "no"}`;
  tile.tabIndex = 0;
  const result = m.finished ? `finished in ${m.time.toFixed(2)} s` : `stopped at ${m.completion.toFixed(0)}%`;
  tile.setAttribute("aria-label", `${m.map}, ${m.tier}: ${result}`);
  const svg = el("svg", { viewBox: "0 0 100 100", "aria-hidden": "true" }, tile);
  el("path", { d: outlinePath(m.cl) }, svg);
  const tag = document.createElement("span");
  tag.textContent = m.map.slice(1, 3);
  tile.appendChild(tag);
  const show = () => {
    tip.textContent = `${m.map} · ${m.tier} · ${result}`;
    const r = tile.getBoundingClientRect();
    tip.style.left = `${Math.min(r.left, innerWidth - 260)}px`;
    tip.style.top = `${r.top - 36}px`;
    tip.style.opacity = 1;
  };
  tile.addEventListener("mouseenter", show);
  tile.addEventListener("focus", show);
  tile.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
  tile.addEventListener("blur", () => { tip.style.opacity = 0; });
  return tile;
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
