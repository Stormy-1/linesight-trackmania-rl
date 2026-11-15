// Run-pair explorer: any of the 10 held-out maps x 10 noisy runs, base stack vs + residual TD3,
// replayed top-down and synced by race time. Each map's 20 runs load on first use.
import { COLORS, createReplay, fetchJson } from "./replay.js";

const PLAYBACK = 3;
const PAIR_STATE = { both: "both finished", onlyb: "only the base finished", onlyr: "only with RL finished", none: "neither finished" };

export const pairState = (b, r) => (b.f && r.f ? "both" : b.f ? "onlyb" : r.f ? "onlyr" : "none");

function outcome(run) {
  return run.finished ? `finished · ${run.time.toFixed(2)} s` : `stopped at ${run.completion.toFixed(1)}%`;
}

export function mountDuel(stories, paired) {
  const replay = createReplay({
    canvas: document.getElementById("duel-canvas"),
    scrub: document.getElementById("duel-scrub"),
    playBtn: document.getElementById("duel-play"),
    timeEl: document.getElementById("duel-time"),
    playback: PLAYBACK,
  });
  const mapSelect = document.getElementById("duel-map");
  const runsEl = document.getElementById("duel-runs");
  const state = { map: stories.b07.map, pair: stories.b07.pair, request: 0 };

  for (const m of paired.maps) {
    const opt = document.createElement("option");
    opt.value = m.map;
    opt.textContent = `${m.map} · ${m.tier}`;
    mapSelect.appendChild(opt);
  }

  function renderRunButtons() {
    const m = paired.maps.find((x) => x.map === state.map);
    runsEl.replaceChildren(...m.base.map((b, i) => {
      const k = i + 1, s = pairState(b, m.td3_v3[i]);
      const btn = document.createElement("button");
      btn.className = `run-chip ${s}`;
      btn.textContent = k;
      btn.setAttribute("aria-pressed", String(k === state.pair));
      btn.setAttribute("aria-label", `Run ${k}: ${PAIR_STATE[s]}`);
      btn.addEventListener("click", () => select(state.map, k));
      return btn;
    }));
  }

  function syncControls() {
    mapSelect.value = state.map;
    document.querySelectorAll("[data-story]").forEach((b) => {
      const s = stories[b.dataset.story];
      b.setAttribute("aria-pressed", String(s.map === state.map && s.pair === state.pair));
    });
    renderRunButtons();
  }

  async function select(map, pair) {
    Object.assign(state, { map, pair });
    syncControls();
    const request = ++state.request;
    try {
      const data = await fetchJson(`data/pairs/${map}.json`);
      if (request !== state.request) return;   // a newer choice won
      const base = data.base[pair - 1], rl = data.td3_v3[pair - 1];
      replay.show({ cl: data.cl, tracks: [{ run: base, color: COLORS.base }, { run: rl, color: COLORS.rl }] });
      fillLegend(map, pair, base, rl, paired);
    } catch (err) {
      console.error("[duel]", err);
      document.getElementById("duel-note").textContent = `Could not load ${map}: ${err.message}`;
    }
  }

  document.querySelectorAll("[data-story]").forEach((b) => b.addEventListener("click", () => {
    const s = stories[b.dataset.story];
    select(s.map, s.pair);
  }));
  mapSelect.addEventListener("change", () => select(mapSelect.value, 1));
  document.addEventListener("duel:select", (e) => {
    select(e.detail.map, e.detail.pair);
    document.getElementById("duel").scrollIntoView({ block: "start" });
  });

  // Fetch the first pair only when the section comes near the screen.
  const io = new IntersectionObserver(([e]) => {
    if (!e.isIntersecting) return;
    io.disconnect();
    if (!state.request) select(state.map, state.pair);   // skip if a pair was already picked
  }, { rootMargin: "600px" });
  io.observe(document.getElementById("duel"));
  syncControls();
}

function fillLegend(map, pair, base, rl, paired) {
  const set = (id, run) => {
    const b = document.getElementById(id);
    b.textContent = outcome(run);
    b.className = run.finished ? "ok" : "fail";
  };
  set("duel-base", base);
  set("duel-rl", rl);
  const m = paired.maps.find((x) => x.map === map);
  const fb = m.base.filter((r) => r.f).length, fr = m.td3_v3.filter((r) => r.f).length;
  const extra = map.startsWith("C02")
    ? " The cause on C02 was left uninvestigated on purpose: finding it would mean studying test-map logs."
    : "";
  document.getElementById("duel-note").textContent =
    `${map}, noisy run ${pair} of 10, replayed at ${PLAYBACK}× from the logs. Red squares: suspected impacts. ` +
    `Across all 10 pairs on this map: base ${fb}/10, with RL ${fr}/10.${extra}`;
}
