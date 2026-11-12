# Frozen score variant (decided 2026-10-20T23:56:09Z, before any test evaluation)
Selection data: 36 development injections in the validation partition + calibration data only (no test labels).
Models used for the decision: search__W30_inv1.0_phys0.1 s17 (TailGuard seed 17) and search__W30_inv0.0_phys0.0 s17.
Grid: ECDF reference {pooled, per-regime} x top-k {1,3,5} x smoothing {10,30,60} s x fusion {mean beta 0/.25/.5/.75, max} x threshold {pooled, per-regime}.
Paper scoring (top-3, 30 s, mean beta=0.5): 15/36 and 1/36 development faults detected.
Frozen variant "max tail surprise": top-1 channel, fusion = max(forecast, constraint), 30 s causal mean, pooled ECDF, pooled threshold
  -> 21/36 and 18/36; robust across both models (per-regime ECDF / thresholds gave no consistent gain).
Rule: applied unchanged to every method that uses the tail-surprise score (TailGuard, matched GRU, ablations, deep baselines),
evaluated once on the test set; reported as a post-hoc scoring extension next to the prespecified results.

## Comparator scoring rule (fixed 2026-10-21, before any test evaluation)
Each comparator is scored with whichever of the two scorings (paper top-3 mean / max tail surprise) detects more of
its own development faults (ties -> paper scoring):
ridge 11 vs 9 -> paper; PCA 10 vs 10 -> paper; persistence 0 vs 1 -> max; Isolation Forest -> own score;
USAD, TranAD, GDN (re-run) -> max (higher or equal for every seed); matched GRU and ablations -> max (higher for every seed).
Note (2026-10-21, GDN re-run completed, still before any test evaluation): the rule is applied per method on the mean
over seeds. GDN: max 6.6/36 vs paper 4.8/36 -> max (seed 101 alone is 3 vs 5).

## Re-selection under the frozen scoring (decided 2026-10-21 12:10, after the variant test evaluation)
The paper's own selection rule (D10: maximise development-fault recall at the calibrated budget, beta=0.5 slot = max fusion;
ties -> lower worst-regime validation loss) is re-applied to the 27 search models of seed 17 under the frozen scoring.
Validation data and development faults only. Disclosed in the paper as a post-hoc re-selection.
