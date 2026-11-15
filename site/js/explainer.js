// How the stack drives, on one logged run: pure pursuit drawn heading-up around the car, and the speed plan
// over the whole lap. The geometry is recomputed here from the logged pose with the controller's own constants.
import { el } from "./charts.js";
import { indexAt } from "./replay.js";

const VIEW_M = 62;            // metres from the car to the canvas edge
const ROAD_M = 14;            // drawn road width (illustration only)
const TRAIL_S = 5;            // seconds of past path drawn behind the car
const KMH = 3.6;
const C = { road: "#1b1e23", lock: "#24323b", line: "#3a3e45", trail: "#7fb3cf", ink: "#eceae4", muted: "#8a8a84", rl: "#ffb000" };
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

function makeLine(line) {
  const at = (s) => {   // position on the racing line at progress s (linear between 2 m samples)
    const S = line.s, n = S.length - 1;
    if (s <= S[0]) return [line.x[0], line.z[0]];
    if (s >= S[n]) return [line.x[n], line.z[n]];
    const i = Math.max(0, indexAt(S, s)), f = (s - S[i]) / (S[i + 1] - S[i]);
    return [line.x[i] + f * (line.x[i + 1] - line.x[i]), line.z[i] + f * (line.z[i + 1] - line.z[i])];
  };
  const valueAt = (key, s) => {
    const i = Math.min(Math.max(0, indexAt(line.s, s)), line.s.length - 2);
    const f = Math.min(Math.max((s - line.s[i]) / (line.s[i + 1] - line.s[i]), 0), 1);
    return line[key][i] + f * (line[key][i + 1] - line[key][i]);
  };
  return { at, valueAt };
}

// The controller's steering rule, from PurePursuitControllerConfig (exported with the data).
function purePursuit(pp, line, run, i) {
  const v = run.v[i], L = pp.lookahead_m + pp.lookahead_time_s * v;
  const [tx, tz] = line.at(run.s[i] + L);
  const bearing = Math.atan2(tz - run.z[i], tx - run.x[i]);
  const alpha = ((bearing - run.yaw[i] + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
  const span = pp.steer_limit_end_mps - pp.steer_limit_start_mps;
  const frac = Math.min(Math.max((v - pp.steer_limit_start_mps) / span, 0), 1);
  const limit = 1 - frac * (1 - pp.high_speed_steer_limit);
  const steer = Math.max(-limit, Math.min(limit, pp.steering_gain * alpha));
  return { v, L, target: [tx, tz], alpha, limit, steer };
}

export function mountExplainer(d) {
  const canvas = document.getElementById("drive-canvas");
  const ctx = canvas.getContext("2d");
  const scrub = document.getElementById("drive-scrub");
  const playBtn = document.getElementById("drive-play");
  const timeEl = document.getElementById("drive-time");
  const line = makeLine(d.line);
  const run = d.run;
  const duration = run.t.at(-1);
  const state = { t: 0, playing: !reduced, visible: false, last: 0, size: [0, 0] };
  const plan = drawPlan(document.getElementById("drive-plan"), d, (s) => seek(run.t[firstAtOrAfter(run.s, s)]));
  const out = Object.fromEntries(["v", "L", "a", "steer", "cap", "target", "pedal"].map((k) => [k, document.getElementById(`dr-${k}`)]));
  const steerBar = document.getElementById("dr-steer-bar");

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    state.size = [r.width, r.height];
    draw();
  }

  function draw() {
    const i = indexAt(run.t, state.t);
    const p = purePursuit(d.pp, line, run, i);
    drawView(ctx, state.size, d, line, run, i, p);
    plan.cursor(run.s[i], run.v[i]);
    out.v.textContent = `${Math.round(p.v * KMH)} km/h`;
    out.L.textContent = `${p.L.toFixed(1)} m`;
    out.a.textContent = `${((p.alpha * 180) / Math.PI).toFixed(1)}°`;
    out.steer.textContent = `${p.steer >= 0 ? "+" : ""}${p.steer.toFixed(2)} ${p.steer > 0.02 ? "right" : p.steer < -0.02 ? "left" : ""}`;
    out.cap.textContent = `±${p.limit.toFixed(2)}`;
    steerBar.style.left = `${50 + Math.min(p.steer, 0) * 50}%`;
    steerBar.style.width = `${Math.abs(p.steer) * 50}%`;
    out.target.textContent = `${Math.round(run.target[i] * KMH)} km/h`;
    out.pedal.textContent = run.brk[i] > 0 ? "brake" : `throttle ${Math.round(run.thr[i] * 100)}%`;
    timeEl.textContent = `${state.t.toFixed(1)} s`;
    scrub.value = Math.round((state.t / duration) * 1000);
  }

  function seek(t) { state.t = Math.min(Math.max(t, 0), duration); draw(); }

  function tick(now) {
    if (state.playing && state.visible) {
      state.t += state.last ? (now - state.last) / 1000 : 0;
      if (state.t > duration + 1.5) state.t = 0;
      draw();
    }
    state.last = now;
    requestAnimationFrame(tick);
  }

  function setPlaying(on) {
    state.playing = on;
    playBtn.textContent = on ? "❚❚" : "▶";
    playBtn.setAttribute("aria-label", on ? "Pause" : "Play");
  }

  playBtn.addEventListener("click", () => setPlaying(!state.playing));
  scrub.addEventListener("input", () => seek((scrub.value / 1000) * duration));
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver(([e]) => { state.visible = e.isIntersecting; }, { threshold: 0.2 }).observe(canvas);
  setPlaying(state.playing);
  if (reduced) state.t = run.t[firstAtOrAfter(run.s, d.jumps[0]?.start_m ?? 0)];
  requestAnimationFrame(tick);
}

function firstAtOrAfter(values, s) {
  const i = values.findIndex((v) => v >= s);
  return i < 0 ? values.length - 1 : i;
}

// Heading-up view centred low on the canvas, so more of the road ahead shows.
function drawView(ctx, [w, h], d, line, run, i, p) {
  const k = Math.min(w, h) / (2 * VIEW_M);
  const cx = w / 2, cy = h * 0.7;
  const yaw = run.yaw[i], cos = Math.cos(yaw), sin = Math.sin(yaw);
  const toScreen = (x, z) => {
    const dx = x - run.x[i], dz = z - run.z[i];
    return [cx + (-dx * sin + dz * cos) * k, cy - (dx * cos + dz * sin) * k];
  };
  ctx.clearRect(0, 0, w, h);
  ctx.lineCap = ctx.lineJoin = "round";

  const S = d.line.s, s0 = run.s[i] - 1.6 * VIEW_M, s1 = run.s[i] + 2.2 * VIEW_M;
  const stroke = (from, to, color, width, keep = () => true) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    let open = false;
    for (let j = from; j <= to; j++) {
      if (!keep(j)) { open = false; continue; }
      const [X, Y] = toScreen(d.line.x[j], d.line.z[j]);
      open ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      open = true;
    }
    ctx.stroke();
  };
  const a = Math.max(0, indexAt(S, s0)), b = Math.min(S.length - 1, indexAt(S, s1) + 1);
  stroke(a, b, C.road, ROAD_M * k);
  stroke(a, b, C.lock, ROAD_M * k, (j) => d.line.lock[j] || d.line.lock[j - 1]);
  ctx.setLineDash([6, 8]);
  stroke(a, b, C.line, 1.5);
  ctx.setLineDash([]);

  // Past path of the car.
  const from = indexAt(run.t, run.t[i] - TRAIL_S);
  ctx.strokeStyle = C.trail; ctx.lineWidth = 2.5; ctx.globalAlpha = 0.85; ctx.beginPath();
  for (let j = from; j <= i; j++) { const [X, Y] = toScreen(run.x[j], run.z[j]); j === from ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y); }
  ctx.stroke(); ctx.globalAlpha = 1;

  // Lookahead circle, target point, aim line.
  ctx.strokeStyle = C.muted; ctx.lineWidth = 1; ctx.setLineDash([3, 5]);
  ctx.beginPath(); ctx.arc(cx, cy, p.L * k, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
  const [tx, ty] = toScreen(p.target[0], p.target[1]);
  ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(tx, ty); ctx.stroke();
  ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(tx, ty, 5, 0, Math.PI * 2); ctx.fill();

  // Heading ray and the angle alpha between heading and aim.
  ctx.strokeStyle = C.trail; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy - p.L * k); ctx.stroke();
  const r = Math.min(46, p.L * k * 0.55);
  ctx.strokeStyle = C.rl; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + p.alpha, p.alpha < 0); ctx.stroke();
  ctx.fillStyle = C.rl; ctx.font = "500 13px 'IBM Plex Mono', monospace";
  const mid = -Math.PI / 2 + p.alpha / 2, side = p.alpha >= 0 ? 1 : -1;
  ctx.fillText("α", cx + Math.cos(mid) * (r + 12) - 4 + side * 4, cy + Math.sin(mid) * (r + 12) + 4);
  ctx.fillStyle = C.ink;
  ctx.fillText(`L = ${p.L.toFixed(1)} m`, tx + 10, ty - 8);

  // The car, nose up, front wheels turned by the steering command.
  ctx.save(); ctx.translate(cx, cy);
  const cw = Math.max(2.2 * k, 9), cl = Math.max(4.4 * k, 18);
  ctx.fillStyle = C.trail; ctx.shadowColor = C.trail; ctx.shadowBlur = 12;
  ctx.fillRect(-cw / 2, -cl / 2, cw, cl); ctx.shadowBlur = 0;
  ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
  for (const sx of [-1, 1]) {
    ctx.save(); ctx.translate((sx * cw) / 2, -cl * 0.32); ctx.rotate(p.steer * 0.5);
    ctx.beginPath(); ctx.moveTo(0, -4); ctx.lineTo(0, 4); ctx.stroke(); ctx.restore();
  }
  ctx.restore();
}

const BAND_GAP_M = 25;        // merge locked stretches closer than this
const BAND_MIN_M = 10;        // and drop slivers shorter than this

function lockBands(L) {
  const bands = [];
  L.lock.forEach((on, j) => {
    if (!on) return;
    const last = bands.at(-1);
    if (last && L.s[j] - last[1] <= BAND_GAP_M) last[1] = L.s[j];
    else bands.push([L.s[j], L.s[j]]);
  });
  return bands.filter(([a, b]) => b - a >= BAND_MIN_M);
}

// Speed plan over the lap (km/h): hand-rule plan, learned corner limit, final target, and the logged speed.
function drawPlan(svg, d, onPick) {
  // Drawn at the element's own width so axis text stays readable on phones.
  const W = Math.round(Math.min(Math.max(svg.getBoundingClientRect().width, 320), 1100));
  const H = W < 700 ? 240 : 300, m = { l: 40, r: 12, t: 18, b: 30 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const L = d.line, sMax = L.s.at(-1), vMax = 260;
  const X = (s) => m.l + (s / sMax) * (W - m.l - m.r);
  const Y = (kmh) => H - m.b - (kmh / vMax) * (H - m.t - m.b);
  const grid = el("g", { class: "grid" }, svg);
  for (let v = 0; v <= vMax; v += 50) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid);
    el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, svg).textContent = v;
  }
  const tick = W < 700 ? 200 : 100;
  for (let s = 0; s <= sMax; s += tick) el("text", { x: X(s), y: H - 8, "text-anchor": "middle" }, svg).textContent = `${s} m`;

  // Jump windows: the target is a floor and a ceiling there. Curve points inside a run-up stay unlocked,
  // so short gaps are merged to draw one band per jump.
  for (const [a, b] of lockBands(L)) {
    el("rect", { x: X(a), y: m.t, width: X(b) - X(a), height: H - m.t - m.b, class: "lock" }, svg);
    if (X(b) - X(a) > 40) el("text", { x: X(a) + 6, y: m.t + 14, class: "lbl" }, svg).textContent = "jump";
  }
  const path = (xs, ys, cls) => el("path", { class: cls, d: xs.map((s, j) => `${j ? "L" : "M"}${X(s).toFixed(1)},${Y(ys[j] * KMH).toFixed(1)}`).join("") }, svg);
  path(L.s, L.hand, "p-hand");
  path(L.s, L.limit, "p-limit");
  path(L.s, L.target, "p-target");
  path(d.run.s, d.run.v, "p-speed");
  const cursor = el("line", { class: "p-cursor", y1: m.t, y2: H - m.b }, svg);
  const dot = el("circle", { r: 5, class: "p-dot" }, svg);

  svg.addEventListener("pointerdown", (e) => {
    const r = svg.getBoundingClientRect();
    const s = (((e.clientX - r.left) / r.width) * W - m.l) / (W - m.l - m.r) * sMax;
    onPick(Math.min(Math.max(s, 0), sMax));
  });
  return {
    cursor(s, v) {
      cursor.setAttribute("x1", X(s)); cursor.setAttribute("x2", X(s));
      dot.setAttribute("cx", X(s)); dot.setAttribute("cy", Y(v * KMH));
    },
  };
}
