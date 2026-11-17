# Paper prompt

The instructions this paper was written to, kept so later edits can be checked against them.

## Setup

- Work on branch `paper/workshop`, cut from `main` at 5c71382.
- Write `paper/PROMPT.md` (this file), `paper/OUTLINE.md` (section plan, with the numbers each section needs and
  their source files) and `paper/PROGRESS.md` (status, open items, TODO citations).
- Read everything in `review/`, plus `RESULTS.md`, `ARCHITECTURE.md` and `README.md`.

## Task

Write a 4 to 6 page workshop-style paper from this project in `paper/main.tex`, using IEEEtran, with these
sections: abstract, introduction, related work, method, experimental protocol, results, limitations, conclusion.

## Rules

- Every number in the text must come from a generated table or figure, or from a cited file in `eval_results/`.
  Prefer `\input` of generated tables: a small script, `paper/make_tables.py`, builds `.tex` tables from the
  JSON/CSV files and from `scripts/reanalyze_clustered.py`'s output.
- References go in `paper/refs.bib`. They must be real papers the author is certain of:
  - IQN (Dabney et al. 2018)
  - TD3 (Fujimoto et al. 2018)
  - TD3+BC (Fujimoto and Gu 2021)
  - residual RL (Johannink et al. 2019; Silver et al. 2018)
  - pure pursuit (Coulter 1992)
  - the Linesight repository

  For anything else, write `\cite{TODO-...}` and list it in `PROGRESS.md`. Never invent a citation, author,
  venue or year.
- Limitations get a full section covering:
  - the seven test-set looks, and the leakage from the gravity fix;
  - inferred collisions;
  - the single-run campaign;
  - simulator-only results;
  - the small number of test maps (only 3 of 10 change between versions; see `review/stats.md`);
  - the shared initialisation of the RL policies (td3_v3 started from td3_v2).
- Style: plain, direct academic prose. No em dashes, no hype words, no "In this paper we" padding, and no summary
  paragraph that repeats the abstract. State a position: classical robotics generalised where end-to-end RL
  didn't, and the residual mostly reshuffles which maps finish.
- Compile with tectonic or latexmk if they are installed. Don't install a TeX engine without asking. Otherwise
  produce `paper/overleaf.zip` with everything Overleaf needs: main.tex, refs.bib, generated tables, figures,
  and IEEEtran.cls only if needed and licence-compatible (Overleaf ships it, so it can usually be left out).
- Finally, act as a hostile reviewer. Write `paper/self_review.md` with the five strongest reasons to reject the
  paper, then fix whatever can be fixed in the text.

## Done when

- `paper/` holds main.tex, refs.bib, the generated tables and the script that makes them, PROMPT.md, OUTLINE.md,
  PROGRESS.md, self_review.md and overleaf.zip.
- It is committed on `paper/workshop` and pushed to origin, with Stormy-1 as the only author and no AI trailers.
  Not merged into main.
- The final report gives the page count (or says it wasn't compiled) and lists the TODO citations.
