// Top-down "ghost" replay of one noisy run pair: base stack vs + residual TD3, synced by race time.
const COLORS = { base: "#7fb3cf", td3_v3: "#ffb000", line: "#2a2e35", bad: "#ff6252", ok: "#57c98a" };
const PLAYBACK = 3;          // replay at 3x game time
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

function indexAt(ts, t) {   // last index with ts[i] <= t
  let lo = 0, hi = ts.length - 1;
  if (t >= ts[hi]) return hi;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ts[mid] <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

function fitTransform(cl, runs, w, h) {
  const xs = [...cl.map((p) => p[0]), ...runs.flatMap((r) => r.x)];
  const zs = [...cl.map((p) => p[1]), ...runs.flatMap((r) => r.z)];
  const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
  const pad = 28;
  const s = Math.min((w - 2 * pad) / (maxX - minX), (h - 2 * pad) / (maxZ - minZ));
  const ox = (w - s * (maxX - minX)) / 2, oz = (h - s * (maxZ - minZ)) / 2;
  return (x, z) => [ox + (x - minX) * s, oz + (z - minZ) * s];
}

function outcome(run) {
  return run.finished ? `finished · ${run.time.toFixed(2)} s` : `stopped at ${run.completion.toFixed(1)}%`;
}

export function mountDuel(replays, paired) {
  const canvas = document.getElementById("duel-canvas");
  const ctx = canvas.getContext("2d");
  const scrub = document.getElementById("duel-scrub");
  const playBtn = document.getElementById("duel-play");
  const timeEl = document.getElementById("duel-time");
  const state = { key: "b07", t: 0, playing: !reduced, visible: false, last: 0, project: null };

  const current = () => replays[state.key];
  const duration = () => Math.max(...["base", "td3_v3"].map((w) => current()[w].t.at(-1)));

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const d = current();
    state.project = fitTransform(d.cl, [d.base, d.td3_v3], r.width, r.height);
    draw();
  }

  function drawTrail(run, upto, color) {
    const P = state.project;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.4;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    for (let i = 0; i <= upto; i++) {
      const [px, pz] = P(run.x[i], run.z[i]);
      i ? ctx.lineTo(px, pz) : ctx.moveTo(px, pz);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = COLORS.bad;
    for (let i = 0; i <= upto; i++) {
      if (!run.imp[i]) continue;
      const [px, pz] = P(run.x[i], run.z[i]);
      ctx.fillRect(px - 2, pz - 2, 4, 4);
    }
  }

  function drawCar(run, i, color, ended) {
    const [px, pz] = state.project(run.x[i], run.z[i]);
    if (ended && !run.finished) {
      ctx.strokeStyle = COLORS.bad;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(px - 7, pz - 7); ctx.lineTo(px + 7, pz + 7);
      ctx.moveTo(px + 7, pz - 7); ctx.lineTo(px - 7, pz + 7);
      ctx.stroke();
      return;
    }
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 14;
    ctx.beginPath(); ctx.arc(px, pz, 6, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  }

  function draw() {
    const d = current();
    const r = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, r.width, r.height);
    ctx.strokeStyle = COLORS.line;
    ctx.lineWidth = 9;
    ctx.lineJoin = ctx.lineCap = "round";
    ctx.beginPath();
    d.cl.forEach((p, i) => { const [x, z] = state.project(p[0], p[1]); i ? ctx.lineTo(x, z) : ctx.moveTo(x, z); });
    ctx.stroke();
    for (const who of ["base", "td3_v3"]) {
      const run = d[who];
      const i = indexAt(run.t, state.t);
      drawTrail(run, i, COLORS[who]);
      drawCar(run, i, COLORS[who], state.t >= run.t.at(-1));
    }
    timeEl.textContent = `${state.t.toFixed(1)} s`;
    scrub.value = Math.round((state.t / duration()) * 1000);
  }

  function tick(now) {
    if (state.playing && state.visible) {
      const dt = state.last ? (now - state.last) / 1000 : 0;
      state.t += dt * PLAYBACK;
      if (state.t > duration() + 2) state.t = 0;     // hold the end for a moment, then loop
      draw();
    }
    state.last = now;
    requestAnimationFrame(tick);
  }

  function setPlaying(on) {
    state.playing = on;
    playBtn.textContent = on ? "❚❚" : "▶";
    playBtn.setAttribute("aria-label", on ? "Pause replay" : "Play replay");
  }

  function select(key) {
    state.key = key;
    state.t = 0;
    document.querySelectorAll("[data-duel]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.duel === key)));
    fillLegend(current(), paired);
    resize();
  }

  document.querySelectorAll("[data-duel]").forEach((b) => b.addEventListener("click", () => select(b.dataset.duel)));
  playBtn.addEventListener("click", () => setPlaying(!state.playing));
  scrub.addEventListener("input", () => { state.t = (scrub.value / 1000) * duration(); draw(); });
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver(([e]) => { state.visible = e.isIntersecting; }, { threshold: 0.2 }).observe(canvas);
  setPlaying(state.playing);
  select("b07");
  if (reduced) { state.t = duration(); draw(); }
  requestAnimationFrame(tick);
}

function fillLegend(d, paired) {
  const set = (id, run) => {
    const b = document.getElementById(id);
    b.textContent = outcome(run);
    b.className = run.finished ? "ok" : "fail";
  };
  set("duel-base", d.base);
  set("duel-rl", d.td3_v3);
  const m = paired.maps.find((x) => x.map === d.map);
  const fb = m.base.filter((r) => r.f).length, fr = m.td3_v3.filter((r) => r.f).length;
  const extra = d.map.startsWith("C02")
    ? " The cause on C02 was left uninvestigated on purpose: finding it would mean studying test-map logs."
    : "";
  document.getElementById("duel-note").textContent =
    `${d.map}, noisy run ${d.pair} of 10, replayed at ${PLAYBACK}× from the 50 ms logs. Red squares: suspected impacts. ` +
    `Across all 10 pairs on this map: base ${fb}/10, with RL ${fr}/10.${extra}`;
}
