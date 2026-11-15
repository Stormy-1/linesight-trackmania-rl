// SVG charts: era-1 learning curve, era-1 benchmark table, residual-RL training episodes.
const NS = "http://www.w3.org/2000/svg";

export function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function text(parent, x, y, str, attrs = {}) {
  const t = el("text", { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}

const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

function frame(svg, W, H) {
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
}

function rollingMedian(values, window) {
  const out = [];
  for (let i = 0; i < values.length; i++) {
    const s = values.slice(Math.max(0, i - window), i + 1).sort((a, b) => a - b);
    out.push(s[Math.floor(s.length / 2)]);
  }
  return out;
}

export function mountEra1(era1) {
  drawEra1Curve(document.getElementById("era1-curve"), era1.curve);
  fillBenchmark(document.querySelector("#era1-bench tbody"), era1.benchmark);
}

function drawEra1Curve(svg, curve) {
  const W = 1000, H = 330, m = { l: 48, r: 16, t: 18, b: 34 };
  frame(svg, W, H);
  const yMax = 70, yMin = 52;
  const x = scale(0, 11e6, m.l, W - m.r);
  const y = scale(yMin, yMax, H - m.b, m.t);
  const grid = el("g", { class: "grid" }, svg);
  for (let s = 54; s <= yMax; s += 4) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(s), y2: y(s) }, grid);
    text(svg, m.l - 8, y(s) + 4, `${s} s`, { "text-anchor": "end" });
  }
  for (let f = 0; f <= 10e6; f += 2e6) text(svg, x(f), H - 10, `${f / 1e6}M`, { "text-anchor": "middle" });

  const dots = el("g", {}, svg);
  for (const [f, t] of curve) {
    if (t > yMax) continue;
    el("circle", { cx: x(f).toFixed(1), cy: y(t).toFixed(1), r: 1.6, fill: "#7fb3cf", opacity: 0.35 }, dots);
  }
  const med = rollingMedian(curve.map((p) => p[1]), 40);
  const d = curve.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(Math.min(med[i], yMax)).toFixed(1)}`).join("");
  el("path", { d, fill: "none", stroke: "#ffb000", "stroke-width": 2.5 }, svg);

  for (const [f, lbl] of [[3.18e6, "3.18M checkpoint: 0/50 on other maps"], [10.64e6, "10.64M: 8/50, all on A03"]]) {
    el("line", { x1: x(f), x2: x(f), y1: m.t, y2: H - m.b, stroke: "#8a8a84", "stroke-dasharray": "3 4" }, svg);
    text(svg, x(f) + (f > 9e6 ? -6 : 6), m.t + 12, lbl, { class: "lbl", "text-anchor": f > 9e6 ? "end" : "start" });
  }
  const best = Math.min(...curve.map((p) => p[1]));
  text(svg, W - m.r, y(best) + 16, `≈${Math.round(best)} s`, { class: "lbl", "text-anchor": "end" });
}

function fillBenchmark(tbody, rows) {
  const pick = (ckpt) => rows.filter((r) => r.checkpoint === ckpt);
  for (const [ckpt, label] of [["foundational_3.18M", "3.18M frames"], ["saturated_10.64M", "10.64M frames"]]) {
    const rs = pick(ckpt);
    const hock = rs.find((r) => r.track === "hock");
    const others = rs.filter((r) => r.track !== "hock");
    const fin = others.reduce((a, r) => a + r.finished, 0);
    const runs = others.reduce((a, r) => a + r.runs, 0);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${label}</td><td><b>${hock.finished}/${hock.runs}</b></td><td><b>${fin}/${runs}</b></td>`;
    tbody.appendChild(tr);
  }
}

export function mountTd3Chart(td3) {
  const svg = document.getElementById("td3-chart");
  const W = 1000, H = 320, m = { l: 48, r: 12, t: 46, b: 30 };
  frame(svg, W, H);
  const gens = ["td3_v1", "td3_v2", "td3_v3"];
  const total = gens.reduce((a, g) => a + td3.gens[g].length, 0);
  const x = scale(0, total, m.l, W - m.r);
  const y = scale(0, 100, H - m.b, m.t);
  const grid = el("g", { class: "grid" }, svg);
  for (const c of [0, 50, 100]) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(c), y2: y(c) }, grid);
    text(svg, m.l - 8, y(c) + 4, `${c}%`, { "text-anchor": "end" });
  }
  let offset = 0;
  for (const g of gens) {
    const eps = td3.gens[g];
    if (offset) el("line", { x1: x(offset), x2: x(offset), y1: m.t - 36, y2: H - m.b, stroke: "#3a3e45" }, svg);
    const fin = eps.filter((e) => e.fin).length;
    text(svg, x(offset) + 6, m.t - 22, g, { class: "lbl", "font-weight": 600 });
    text(svg, x(offset) + 6, m.t - 8, `${fin}/${eps.length} finished`);
    eps.forEach((e, i) => {
      const c = el("circle", { cx: x(offset + i + 0.5).toFixed(1), cy: y(e.c).toFixed(1), r: 3.2,
        fill: e.fin ? "#ffb000" : "#5b6068", opacity: e.fin ? 0.95 : 0.7 }, svg);
      el("title", {}, c).textContent = `${g} episode ${e.ep}: ${e.map}, ${e.c}%${e.fin ? " (finished)" : ""}`;
    });
    offset += eps.length;
  }
  text(svg, W - m.r, H - 8, "episode →", { "text-anchor": "end" });
}
