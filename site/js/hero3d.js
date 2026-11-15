// Hero: a logged run in 3D (x, y, z from the 50 ms telemetry), ribbon coloured by speed, with drop
// lines to the ground so the jumps read. The car replays the run; the camera orbits slowly.
import * as THREE from "three";

const PLAYBACK = 1.5;
const RIBBON_HALF_WIDTH = 9;     // metres (wider than the real track so it reads from far away)
const DROP_EVERY = 4;            // one ground line every 4 samples (0.2 s)
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

function speedColor(v, vMax) {
  const slow = new THREE.Color("#3f7fa3"), mid = new THREE.Color("#a9d3e6"), fast = new THREE.Color("#ffb000");
  const u = Math.min(v / vMax, 1);
  return u < 0.55 ? slow.clone().lerp(mid, u / 0.55) : mid.clone().lerp(fast, (u - 0.55) / 0.45);
}

function centred(run) {
  const n = run.x.length;
  const cx = run.x.reduce((a, b) => a + b, 0) / n, cz = run.z.reduce((a, b) => a + b, 0) / n;
  const yMin = Math.min(...run.y);
  const pts = run.x.map((x, i) => new THREE.Vector3(x - cx, run.y[i] - yMin, run.z[i] - cz));
  const radius = Math.max(...pts.map((p) => Math.hypot(p.x, p.z)));
  return { pts, radius };
}

function ribbon(pts, speeds) {
  const vMax = Math.max(...speeds);
  const pos = [], col = [], idx = [];
  for (let i = 0; i < pts.length; i++) {
    const a = pts[Math.max(i - 1, 0)], b = pts[Math.min(i + 1, pts.length - 1)];
    const dir = new THREE.Vector3(b.x - a.x, 0, b.z - a.z).normalize();
    const side = new THREE.Vector3(-dir.z, 0, dir.x).multiplyScalar(RIBBON_HALF_WIDTH);
    const p = pts[i], c = speedColor(speeds[i], vMax);
    pos.push(p.x + side.x, p.y, p.z + side.z, p.x - side.x, p.y, p.z - side.z);
    col.push(c.r, c.g, c.b, c.r, c.g, c.b);
    if (i) { const k = 2 * i; idx.push(k - 2, k - 1, k, k - 1, k + 1, k); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  return g;
}

function groundLines(pts) {
  const drop = [], shadow = [];
  pts.forEach((p, i) => {
    if (i % DROP_EVERY === 0 && p.y > 2) drop.push(p.x, p.y, p.z, p.x, 0, p.z);
    shadow.push(p.x, 0, p.z);
  });
  const dg = new THREE.BufferGeometry();
  dg.setAttribute("position", new THREE.Float32BufferAttribute(drop, 3));
  const sg = new THREE.BufferGeometry();
  sg.setAttribute("position", new THREE.Float32BufferAttribute(shadow, 3));
  return [
    new THREE.LineSegments(dg, new THREE.LineBasicMaterial({ color: 0x7fb3cf, transparent: true, opacity: 0.12 })),
    new THREE.Line(sg, new THREE.LineBasicMaterial({ color: 0x3a3e45, transparent: true, opacity: 0.8 })),
  ];
}

function buildScene(run) {
  const scene = new THREE.Scene();
  const { pts, radius } = centred(run);
  scene.fog = new THREE.Fog(0x0c0d0f, radius * 1.2, radius * 4.5);
  const geo = ribbon(pts, run.v);
  const ghost = new THREE.Mesh(geo.clone(), new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.28 }));
  const trail = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  scene.add(ghost, trail, ...groundLines(pts));
  const grid = new THREE.GridHelper(radius * 2.6, 26, 0x1d2025, 0x16181c);
  scene.add(grid);
  const car = new THREE.Mesh(new THREE.SphereGeometry(5, 20, 14), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  const halo = new THREE.Mesh(new THREE.SphereGeometry(12, 20, 14), new THREE.MeshBasicMaterial({ color: 0xffb000, transparent: true, opacity: 0.25 }));
  car.add(halo);
  scene.add(car);
  return { scene, pts, radius, trail, car };
}

export function mountHero(run) {
  const canvas = document.getElementById("hero3d");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "low-power" });
  renderer.setClearColor(0x0c0d0f);
  const { scene, pts, radius, trail, car } = buildScene(run);
  const camera = new THREE.PerspectiveCamera(38, 1, 1, 6000);
  const hud = { t: document.getElementById("hud-t"), v: document.getElementById("hud-v"), rs: document.getElementById("hud-rs") };
  const state = { t: reduced ? run.t.at(-1) : 0, last: 0, visible: true, angle: 0.6 };

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // On wide screens push the track right, away from the headline.
    if (w > 900) camera.setViewOffset(w, h, -w * 0.24, -h * 0.02, w, h); else camera.clearViewOffset();
    camera.updateProjectionMatrix();
  }

  function frameAt(t) {
    let i = 0;
    while (i < run.t.length - 1 && run.t[i + 1] <= t) i++;
    return i;
  }

  function render() {
    const i = frameAt(state.t);
    trail.geometry.setDrawRange(0, Math.max(0, i * 6));
    car.position.copy(pts[i]);
    const r = radius * 1.9;
    camera.position.set(Math.cos(state.angle) * r, radius * 1.0, Math.sin(state.angle) * r);
    camera.lookAt(0, 20, 0);
    renderer.render(scene, camera);
    hud.t.textContent = run.t[i].toFixed(2);
    hud.v.textContent = Math.round(run.v[i] * 3.6);
    const rs = run.rs[i];
    hud.rs.textContent = `${rs >= 0 ? "+" : "−"}${Math.abs(rs).toFixed(2)}`;
  }

  function loop(now) {
    const dt = state.last ? Math.min((now - state.last) / 1000, 0.1) : 0;
    state.last = now;
    if (state.visible) {
      state.t += dt * PLAYBACK;
      if (state.t > run.t.at(-1) + 2.5) state.t = 0;
      state.angle += dt * 0.035;
      render();
    }
    requestAnimationFrame(loop);
  }

  new ResizeObserver(() => { resize(); render(); }).observe(canvas);
  new IntersectionObserver(([e]) => { state.visible = e.isIntersecting; }).observe(canvas);
  resize();
  render();
  if (!reduced) requestAnimationFrame(loop);
}
