# PREREG_AXSUP_v1 — Amendment 1: the interval of the analysis

**Registered 2026-11-11 21:43:17 EET** (time from `date`; hash in the ledger). No outcome of the blind runs has been read:
- `results_axsup_blind/` does not exist;
- the remote stage has finished 16 of 72 runs, and none of their numbers has been looked at.

PREREG_AXSUP_v1 and p277 are left unchanged.

## What was found
At ~21:40, building the supplementary package, the blind-set intervals of the paper (p254, p252) were found to come from replicates in which an architecture drawn twice counted once (a sum over `a in archs`). Each replicate was therefore a random subset of distinct architectures rather than a cluster bootstrap sample, and the intervals were too narrow (p281; corrected in paper v49_61). p277 was written from p252 and has the same flaw.

## Amendment
- **Primary rule unchanged.** The rule "the layer helps the neural controller" is unchanged and does not use the interval: pooled gain > 0, a majority of the changed composite architectures better, and no architecture more than 1 % worse.
- **Interval.** The reported 95 % interval is the standard cluster bootstrap: the same seed and draws, every drawn architecture entering with all its episodes as often as drawn. It is computed by **p277b** (= p277 with that one change).
- **Transparency.** p277b also reports the interval of p277's rule, next to it.
- **Run order.** Both p277 and p277b are run; p277b's interval is the one reported.
- **p278 is unaffected.** It uses permutation tests and rank correlations, not this bootstrap.
