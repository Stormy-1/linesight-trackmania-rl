# Progress

## Status (2026-11-17)

| Item | State |
|---|---|
| `main.tex` (IEEEtran conference, sections as specified) | drafted, self-reviewed, revised |
| `refs.bib` | 16 entries: the 7 named in PROMPT.md and 9 that replaced the TODO keys, all checked |
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

## Citations

`refs.bib` has 16 entries, and every `\cite` key in `main.tex` has one (checked by script: no missing or unused
keys). The seven named in PROMPT.md were there from the start. The eight `TODO-...` keys were filled on
2026-11-17, each checked against the publisher, proceedings or project page:

| Was | Now | Checked against |
|---|---|---|
| `TODO-gt-sophy` | Wurman et al., Nature 602:223--228, 2022 (GT Sophy) | nature.com DOI 10.1038/s41586-021-04357-7; Peter Stone's publication page for the author order |
| `TODO-rl-generalisation` | Cobbe et al., ICML 2019, PMLR 97:1282--1289 | proceedings.mlr.press/v97/cobbe19a |
| `TODO-racing-survey` | Betz et al., IEEE OJ-ITS 3:458--488, 2022 | DOI 10.1109/OJITS.2022.3181510; arXiv 2202.07008 |
| `TODO-mpc-racing` | Liniger, Domahidi, Morari, Optim. Control Appl. Methods 36(5):628--647, 2015 | arXiv 1711.07300 and journal record |
| `TODO-rl-statistics` | Henderson et al., AAAI 2018; Agarwal et al., NeurIPS 2021 | ojs.aaai.org article 11694; NeurIPS 2021 proceedings |
| `TODO-design-effect` | Kish, *Survey Sampling*, Wiley, 1965 | the standard reference that introduced the term "design effect" |
| `TODO-tminterface` | donadigo, TMInterface (software) | the project page the README already links |
| `TODO-sim2real` | Tobin et al., IROS 2017 (domain randomization) | arXiv 1703.06907; IEEE Xplore 8202133 |

The text was adjusted where needed, so each citation supports exactly the sentence it's attached to:
- The intro now says that an agent trained at scale beat champion players of a racing simulator. It no longer
  calls racing games "a common testbed" on GT Sophy's authority.
- The simulator-only limitation now cites Tobin et al. for the narrower point that sim-to-real transfer needs
  methods of its own.

## Open items

1. Compile once (Overleaf, or install a TeX engine with the user's OK) and check:
   - the page count;
   - overfull boxes in the three `\resizebox` tables;
   - `xspace` spacing after macros before `\,` and `\%` (exceptions are added in the preamble).
2. Author block: it names the GitHub account and repository. Remove both for a double-blind venue, and replace
   the handle with the author's name for a single-blind one.
3. The README says "The other eight maps don't change" for td3_v3. A09 also moves (5 → 7 finishes), so seven don't
   change. The paper is correct; README.md and the site still need the fix (outside this branch's scope).
4. Optional: a second figure with per-tier campaign finishes, if a venue allows 6 pages.

## Overleaf

`overleaf.zip` is rebuilt by `make_tables.py` on every run, from `main.tex`, `refs.bib` and `generated/*.tex` plus
`generated/fig_maps_power.pdf`. Upload it as a new project. The compiler is pdfLaTeX, and the main document is
`main.tex`.
