The CSV files in `reports/` are the result tables written by `scripts/step6_evaluate.py --unseal` for the paper
(Kaggle, 15 October 2026; the test labels were joined once, see `manifests/protocol_deviations.md`).
Method keys: `GRU (pooled ERM)` is the matched GRU of the paper; `A1_no_physics` is A1 (lambda_phys = 0, scored at beta = 0);
`A4_beta0` is the full model scored at beta = 0; `Residual-only (beta=1)` is A5. Protocol keys `[14]` and `[15]` are
references [22] and [23] of the paper.

The CSV files in `reports_max/` are the tables of the max tail-surprise analysis (Section 4.5 and Table 10 of the paper),
written by `scripts/step6_evaluate_max.py --unseal` on 21 October 2026 after the score was frozen on development faults
(`manifests/max_score_protocol.md`; the second reading of the test labels is logged in `manifests/unseal_log_max.txt`).
USAD, TranAD and GDN in these tables are the CPU re-runs that kept per-channel errors (`scripts/step7_deep_rescore.py`).
