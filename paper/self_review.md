# Hostile self-review

Read as a reviewer who wants to reject the paper. Below are the five strongest reasons, most damaging first. Each
says what was changed in `main.tex` in response, and what can't be fixed in the text.

## 1. The headline comparison of the two eras is confounded

The title claims the classical stack "generalises where end-to-end RL did not". The two sides aren't comparable.
The stack gets the official replay of every track as its reference line, which is an expert demonstration per
track at test time. The IQN agent got pixels and one training track. The IQN agent was never trained on more than
one track in any committed run (the multi-track fine-tuning collapse in the README has no committed log, claim E7
in `review/claims.md`). It is evaluated on five tracks, the stack on 63, one run each, and the stack was developed
on five of them. "Generalises" is supported only in a weak sense: a pipeline with no track-specific parameters
transfers when it is handed a reference lap.

**Fixed in the text.** The abstract now says the stack is given a reference lap per track. The results section
already said the comparison is not symmetric; a new limitation, "The era comparison is not controlled", lists the
privileged reference line, the single-track IQN training and the unequal evaluation. The title keeps the position
(it's the claim the author wants to defend), but the paper no longer implies a controlled comparison.

**Not fixable in the text.** A fair comparison needs an end-to-end agent trained on many tracks, or a stack
without the reference lap. Neither exists in the repo.

## 2. A null from an underpowered design is uninformative, yet the paper draws a conclusion from it

The paper's own power analysis says ten tracks detect only a gain of about 15 to 20 points of finish rate. So
"the residual mostly reshuffles which tracks finish" can't be told apart from "the residual has a real gain of a
few points that this design can't see". Calling it a "negative result" overstates what the data shows.

**Fixed in the text.** The abstract now calls it a null result and says what it can't rule out. The words "fixes"
and "breaks" for B07 and C02 became "improves" and "damages", since neither survives correction (p ≈ 0.16). The
reshuffling claim is now tied to what was observed: two tracks move in opposite directions and seven don't move.
It no longer reads as a claim about the policy's true effect.

**Not fixable in the text.** Only more tracks would help.

## 3. The test set is contaminated, and the selection rule is weak

The test set was looked at seven times, and test-track failure logs drove the gravity fix. td3_v3 was admitted to
its test look by a dev rule. That rule's margin (60 against 59 of 70 runs) is itself noise (p = 1), so it barely
selects anything. And the rule exists only in the docs, with no committed timestamp proving it predates the dev
runs (`review/repro.md`).

**Fixed in the text.** The protocol section already reported the dev difference as noise. The limitations now
also say that the pre-registration has no committed timestamp and that the rule's margin is within noise.

**Not fixable in the text.** A fresh set of tracks is the remedy. The conclusion names the 26 tracks that remain
unused for dev, test and RL, and says they aren't fresh for the learned models.

## 4. The residual-RL design choices aren't ablated

The method presents zero initialisation, critic burn-in, a weakened anchor and held exploration noise as the
choices "that made training start from the base". Only the zero initialisation has a direct check (760 of 760
identical steps). The others rest on training-time diagnostics (the critic's fit to returns, an actor that didn't
move in td3_v1). None is tested on held-out tracks. There's also no baseline for "what else could the same effort
buy": for example, tuning the base stack's own parameters on dev, or a non-residual policy with the same
observation.

**Fixed in the text.** The method now says these choices are supported by training diagnostics, not by held-out
ablations. It also introduces td3_v1 before referring to it, and corrects "three choices" to four.

**Not fixable in the text.** Ablations need new live runs in the game.

## 5. The measurements and the statistics rest on unvalidated components

- Contacts are inferred from a speed-drop threshold that has never been checked against ground truth. Yet wall
  hits appear in the results, and the impact penalty is part of the reward.
- The power simulation assumes a common odds ratio across tracks and independent runs, and smooths the base
  rates so that tracks at 0 or 10 finishes can move. The six tracks that never finish stall early on geometry, so
  that smoothing makes the simulation optimistic.
- The intraclass correlation is estimated from one version's ten clusters.

**Fixed in the text.** Section V-C now states the power simulation's assumptions and why they're optimistic. The
limitation on inferred collisions already says that wall counts and the reward inherit the classifier's errors.
The results now say that the wall and recovery reductions were never tested at the track level.

**Not fixable in the text.** Ground-truth contact data isn't available from the game.

## Smaller points, also fixed

- The abstract said "the other eight do not change". A09 also moves (5 to 7 finishes), so the abstract now says
  seven tracks don't change. The README has the same slip ("The other eight maps don't change"). It is listed in
  `PROGRESS.md` as a follow-up, not fixed here.
- The figure caption and the tables give their sources and settings, so each number can be traced to
  `make_tables.py`.
- The author block names the GitHub account and repository. A double-blind venue needs it removed (see
  `PROGRESS.md`).
