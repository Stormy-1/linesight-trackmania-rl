# Progress

## Status (2026-11-17)

| Item | State |
|---|---|
| `main.tex` (IEEEtran conference, sections as specified) | drafted, self-reviewed, revised |
| `refs.bib` | 7 entries, only the references named in PROMPT.md |
| `make_tables.py` → `generated/` | 5 tables, 1 figure, 176 number macros; reruns `scripts/reanalyze_clustered.py` and checks it matches `review/clustered_reanalysis.json` (it does) |
| `self_review.md` | five reasons to reject; fixes applied to the text |
| `overleaf.zip` | built: main.tex, refs.bib, generated/ (IEEEtran.cls left out, Overleaf ships it) |
| Compiled PDF | **not compiled**: no TeX engine (tectonic, latexmk, pdflatex) on this machine, and none was installed |
| Page count | unknown until compiled. Estimate: about 5 pages (about 3,300 words of body text, 5 tables, 1 figure, references) |

## Regenerating

```bash
python paper/make_tables.py
```

Use `--reuse-reanalysis` to skip the 20 s rerun of the map-level analysis. The script also rebuilds `overleaf.zip`.
The ruff warnings left in it are style only (implicit string concatenation inside the row lists is deliberate).

## How numbers are sourced

- Every number in `main.tex` is a macro from `generated/numbers.tex` or a cell of a generated table.
- `make_tables.py` reads `eval_results/` (robustness, single-run, dev JSONs, model validations, behaviour, td3_v3
  config and log, per-run `summary.json`/`runXX.csv`), `maps/map_split.json`, `maps/official_manifest.json`, and
  the reanalysis JSON.
- A few design constants (observation size, ray count and range, network widths, impact threshold) are imported
  from the code that uses them.
- The remaining literal digits are track names (A09, C02, ...), exponents, α = 0.05, the 95% interval level, and
  "0" in "tracks at 0 or 10 finishes".
- **One number has no result-file source:** "developed on five of those tracks" (the campaign result). It comes
  from the project record (`RESULTS.md` §4, claim C1 in `review/claims.md`), and the text says so.

Rounding differs slightly from README/RESULTS in two places, because the macros round the raw values:
- td3_v2's heading correlation is −0.135, printed −0.14 (README says −0.13);
- the grip model's demand-only AUC is 0.7545, printed 0.755 (README says 0.754).

## TODO citations

The paper cites these keys, and none of them has an entry in `refs.bib`. Each needs a real, checked reference
before submission. Until then, BibTeX will print warnings and "[?]" for each.

| Key | Where | What it should support |
|---|---|---|
| `TODO-gt-sophy` | Intro, Related work | deep RL reaching expert level in a racing simulator at scale |
| `TODO-rl-generalisation` | Related work | deep RL overfitting to its training environments |
| `TODO-racing-survey` | Related work | survey of autonomous racing (planning and control) |
| `TODO-mpc-racing` | Related work | optimisation-based / MPC control for racing |
| `TODO-rl-statistics` | Related work | reported deep-RL gains within run-to-run variance; statistical practice in RL evaluation |
| `TODO-design-effect` | Related work | design effect / effective sample size for clustered samples |
| `TODO-tminterface` | Method | TMInterface, the TrackMania scripting interface (software citation) |
| `TODO-sim2real` | Limitations | sim-to-real gap for learned driving / robot policies |

## Open items

1. Compile once (Overleaf, or install a TeX engine with the user's OK) and check:
   - the page count;
   - overfull boxes in the three `\resizebox` tables;
   - `xspace` spacing after macros before `\,` and `\%` (exceptions are added in the preamble).
2. Fill the TODO citations above, then delete this list's entries.
3. Author block: it names the GitHub account and repository. Remove both for a double-blind venue, and replace
   the handle with the author's name for a single-blind one.
4. The README says "The other eight maps don't change" for td3_v3. A09 also moves (5 → 7 finishes), so seven don't
   change. The paper is correct; README.md and the site still need the fix (outside this branch's scope).
5. Optional: a second figure with per-tier campaign finishes, if a venue allows 6 pages.

## Overleaf

`overleaf.zip` is rebuilt by `make_tables.py` on every run, from `main.tex`, `refs.bib` and `generated/*.tex` plus
`generated/fig_maps_power.pdf`. Upload it as a new project. The compiler is pdfLaTeX, and the main document is
`main.tex`.
