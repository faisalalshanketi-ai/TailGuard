# HAI 22.04 confirmation protocol (frozen 2026-10-22, before any HAI 22.04 data were prepared or labels read)

Purpose: an evaluation set never used for any design decision of TailGuard.

Frozen from HAI 23.05 (no re-tuning on HAI 22.04):
- Model: causal GRU, 2 layers x 64, W = 30, lambda_inv = 1, lambda_phys = 0.1, regime environments, balanced batches,
  Huber delta 1, lambda_2 = 1e-5, Adam 1e-3, batch 256, <= 100 epochs, patience 10 on worst-regime validation loss.
- Scores: prespecified (top-3 mean, beta = 0.5) and max tail surprise (top-1, max fusion), both 30 s; thresholds by
  the monotone-closure rule at 1 alarm episode per 24 h on calibration data.
- Comparators: matched GRU (pooled ERM, beta = 0), A1 (lambda_phys = 0), USAD, TranAD, GDN (authors' settings),
  ridge AR (alpha = 1), PCA (k = 20), persistence, Isolation Forest; comparator score choice as in max_score_protocol.md
  (ridge and PCA prespecified score; others max score in the max-score analysis).
- Seeds 17, 29, 43, 71, 101.

Applied unchanged to HAI 22.04 data (rules, not values):
- Step 2 rules: chronological 60/20/20 split of train1-6, W_MAX = 120 eligibility, feature roles and robust scalers from
  fitting rows, regimes from P4_ST_PS with edges 10 and 20.
- Step 4 rules: the same 14-relation registry; windows, coefficients, scales and screening refitted on HAI 22.04 fitting data.
- Step 3 rules: 36 development faults (seed 17) on HAI 22.04 validation data, used for nothing but reporting.

Metrics and tests: as in the paper (event recall, FA24, F1/AP/AUROC, delays, paired block bootstrap with 6 h blocks,
sign-flip tests). Test labels are read once, by step6 with TAILGUARD_PREFIX=hai2204, after all runs have finished.
No published HAI 22.04 numbers are used for any choice.
