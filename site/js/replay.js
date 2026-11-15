// Top-down replay of logged runs on a canvas, synced by race time. Used by the run-pair duel and the
// campaign explorer. A track is { run, color } or { run, bySpeed: true } (trail tinted slow -> fast).
export const COLORS = { base: "#7fb3cf", rl: "#ffb000", line: "#2a2e35", bad: "#ff6252" };
const SPEED_RAMP = ["#2b4352", "#3f6377", "#57839a", "#7fb3cf", "#a9cfe2", "#e6f2f8"];
const SPEED_MAX = 75;        // m/s at the top of the ramp
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

export function indexAt(ts, t) {   // last index with ts[i] <= t
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
  const s = Math.min((w - 2 * pad) / Math.max(maxX - minX, 1), (h - 2 * pad) / Math.max(maxZ - minZ, 1));
  const ox = (w - s * (maxX - minX)) / 2, oz = (h - s * (maxZ - minZ)) / 2;
  return (x, z) => [ox + (x - minX) * s, oz + (z - minZ) * s];
}

const speedBin = (v) => Math.min(SPEED_RAMP.length - 1, Math.floor((v / SPEED_MAX) * SPEED_RAMP.length));

export function createReplay({ canvas, scrub, playBtn, timeEl, playback = 3 }) {
  const ctx = canvas.getContext("2d");
  const state = { scene: null, t: 0, playing: !reduced, visible: false, last: 0, project: null };

  const duration = () => Math.max(...state.scene.tracks.map((k) => k.run.t.at(-1)));

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!state.scene) return;
    state.project = fitTransform(state.scene.cl, state.scene.tracks.map((k) => k.run), r.width, r.height);
    draw();
  }

  function strokePath(run, from, upto, color) {
    const P = state.project;
    ctx.strokeStyle = color;
    ctx.beginPath();
    for (let i = from; i <= upto; i++) {
      const [px, pz] = P(run.x[i], run.z[i]);
      i === from ? ctx.moveTo(px, pz) : ctx.lineTo(px, pz);
    }
    ctx.stroke();
  }

  function drawTrail(track, upto) {
    const { run } = track;
    ctx.lineWidth = 2.4;
    if (!track.bySpeed) {
      strokePath(run, 0, upto, track.color);
    } else {   // one path per speed bin keeps it to a handful of strokes per frame
      const P = state.project;
      const paths = SPEED_RAMP.map(() => new Path2D());
      for (let i = 1; i <= upto; i++) {
        const [ax, az] = P(run.x[i - 1], run.z[i - 1]), [bx, bz] = P(run.x[i], run.z[i]);
        const path = paths[speedBin(run.v[i])];
        path.moveTo(ax, az); path.lineTo(bx, bz);
      }
      paths.forEach((p, b) => { ctx.strokeStyle = SPEED_RAMP[b]; ctx.stroke(p); });
    }
    ctx.fillStyle = COLORS.bad;
    for (let i = 0; i <= upto; i++) {
      if (!run.imp[i]) continue;
      const [px, pz] = state.project(run.x[i], run.z[i]);
      ctx.fillRect(px - 2.5, pz - 2.5, 5, 5);
    }
  }

  function drawCar(track, i, ended) {
    const { run } = track;
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
    const color = track.color || "#ffffff";
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 14;
    ctx.beginPath(); ctx.arc(px, pz, 6, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  }

  function draw() {
    const r = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, r.width, r.height);
    const { cl, tracks } = state.scene;
    ctx.strokeStyle = COLORS.line;
    ctx.lineWidth = 9;
    ctx.lineJoin = ctx.lineCap = "round";
    ctx.beginPath();
    cl.forEach((p, i) => { const [x, z] = state.project(p[0], p[1]); i ? ctx.lineTo(x, z) : ctx.moveTo(x, z); });
    ctx.stroke();
    for (const track of tracks) {
      const i = indexAt(track.run.t, state.t);
      drawTrail(track, i);
      drawCar(track, i, state.t >= track.run.t.at(-1));
    }
    timeEl.textContent = `${state.t.toFixed(1)} s`;
    scrub.value = Math.round((state.t / duration()) * 1000);
  }

  function tick(now) {
    if (state.scene && state.playing && state.visible) {
      const dt = state.last ? (now - state.last) / 1000 : 0;
      state.t += dt * playback;
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

  playBtn.addEventListener("click", () => setPlaying(!state.playing));
  scrub.addEventListener("input", () => {
    if (!state.scene) return;
    state.t = (scrub.value / 1000) * duration();
    draw();
  });
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver(([e]) => { state.visible = e.isIntersecting; }, { threshold: 0.2 }).observe(canvas);
  setPlaying(state.playing);
  requestAnimationFrame(tick);

  return {
    show(scene) {
      state.scene = scene;
      state.t = 0;
      resize();
      if (reduced) { state.t = duration(); draw(); }
    },
  };
}

export const SPEED_LEGEND = SPEED_RAMP;
export const SPEED_LEGEND_MAX = SPEED_MAX;

const cache = new Map();
export function fetchJson(url) {   // each map file is fetched once, on first use
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
      return r.json();
    }).catch((err) => { cache.delete(url); throw err; }));
  }
  return cache.get(url);
}
