// 10 test maps x 10 noisy runs: each cell is one run pair, base stack vs + td3_v3.
const STATE_LABEL = { both: "both finished", onlyb: "only the base finished", onlyr: "only with RL finished", none: "neither finished" };

export function mountPaired(data) {
  const root = document.getElementById("paired");
  const head = document.createElement("div");
  head.className = "prow head";
  head.innerHTML = `<span class="name"></span>${Array.from({ length: 10 }, (_, i) => `<span class="cell">${i + 1}</span>`).join("")}<span class="tot">base → RL</span>`;
  root.appendChild(head);

  for (const m of data.maps) {
    const row = document.createElement("div");
    row.className = "prow";
    const name = document.createElement("span");
    name.className = "name";
    name.innerHTML = `${m.map.split("-")[0]}<small>${m.map.split("-")[1]} · ${m.tier}</small>`;
    row.appendChild(name);
    m.base.forEach((b, i) => {
      const r = m.td3_v3[i];
      const state = b.f && r.f ? "both" : b.f ? "onlyb" : r.f ? "onlyr" : "none";
      const cell = document.createElement("span");
      cell.className = `cell ${state}`;
      cell.title = `${m.map}, run ${i + 1}: ${STATE_LABEL[state]} (base ${b.c}%, RL ${r.c}%)`;
      row.appendChild(cell);
    });
    const fb = m.base.filter((r) => r.f).length, fr = m.td3_v3.filter((r) => r.f).length;
    const tot = document.createElement("span");
    tot.className = "tot";
    tot.innerHTML = `${fb} → <b>${fr}</b>`;
    row.appendChild(tot);
    row.setAttribute("aria-label", `${m.map}: base finished ${fb} of 10, with RL ${fr} of 10`);
    root.appendChild(row);
  }
}
