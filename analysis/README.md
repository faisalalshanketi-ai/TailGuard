# Analyses of the frozen score streams

All analyses here read frozen score streams or saved checkpoints; none retrains a model or changes a threshold rule. Every reading of test labels is appended to `manifests/unseal_log.txt` (HAI 23.05) or `hai2204/manifests/unseal_log.txt` (HAI 22.04).

| Manuscript item | Script | Result files (`analysis/results/`) |
|---|---|---|
| Refitted regime hold-out, H2 (A2, A10), A7 randomized constraints (Kaggle runs) | `scripts/holdout_a7_runs.py` (training), `analysis/scripts/holdout_a7_eval.py` | `hai2305/clean_holdout_*.csv`, `A7_*.csv` |
| Withheld-regime false alarms with the main episode definition (Table 12, Figure 6) | `fa_consistent.py`, `fig6.py` | `hai2305/regime_fa_consistent*.csv` |
| Regime recalibration (Section 4.4) | `recal_consistent.py` | `hai2305/recalibration_consistent*.csv` |
| Alarm-budget sweep, matched burden, block length, duration-aware metrics, missed attacks, development faults (Tables 3, 9; Figure 4a) | `frozen_score_analyses.py` | `hai2305/budget_sensitivity.csv`, `matched_burden.csv`, `block_length.csv`, `duration_aware.csv`, `attacks_detection.csv`, `dev_injections_spec.csv` |
| Tail-fit sensitivity (Section 4.3) | `tail_sensitivity.py` | `hai2305/tail_sensitivity*.csv` |
| Residual scaling (Table 11) | `score_ablation.py` | `hai2305/score_ablation_2305.csv`, `hai2204/score_ablation_2204.csv` |
| Ridge AR fused with the constraint stream (Section 4.3) | `ridge_constraint_fusion.py` | `hai2305/ridge_constraint_fusion_2305.csv`, `hai2204/ridge_constraint_fusion_2204.csv` |
| HAI 22.04 confirmation (Section 4.6, Table 14) | `scripts/step2_prepare_2204.py`, `scripts/confirm_2204.py`, steps 3/4/7 with `TAILGUARD_PREFIX=hai2204`, `build_runs_max_2204.py`, `step6_evaluate*.py --unseal` | `hai2204/reports*`; protocol `hai2204/manifests/HAI2204_PROTOCOL.md` |
| Reproduction of the common-protocol results from the saved streams | `extended_analyses.py ... reproduce` | `hai2305/reproduce_hai2305.csv`, `hai2204/reproduce_hai2204.csv` |
| HAI 22.04 alarm-budget sweep (Figure 4b, Section 4.6) | `extended_analyses.py --prefix hai2204 ... budget` | `hai2204/budget_sensitivity_hai2204.csv`, `matched_burden_hai2204.csv` |
| Calibration-window length (Section 4.2, Table A8) | `extended_analyses.py ... calwin` | `hai2305/calibration_window_hai2305.csv` |
| Invariance coefficient over the 27 search candidates (Section 4.3, Table A9) | `extended_analyses.py ... lambda_inv` | `hai2305/lambda_inv_hai2305.csv`, `lambda_inv_summary_hai2305.csv` |
| Degraded constraint registry (Section 4.3, Table A10) | `extended_analyses.py ... constraints` | `hai2305/constraint_degradation*_hai2305.csv` |
| Selection with one fault type left out; detection by fault kind and duration (Section 4.5, Table A11) | `fault_type_selection.py` | `hai2305/fault_type_selection.csv`, `fault_detection_by_candidate.csv`, `fault_detection_by_duration.csv` |
| Streaming latency (Section 4.6) | `latency.py` | `hai2305/latency.csv` |

`regime_counts.csv` and `recalibration.csv` count withheld-regime false episodes after removing attack rows (an alternative definition that removes attack rows before counting episodes); they are kept for the record and superseded by the `*_consistent*` files, which use the main FA24 definition. `a7_point.py` computes the F1 and AP of A7 for Table 10.

`np_gru.py` evaluates the saved forecaster checkpoints with NumPy (PyTorch gate convention) and `constraints_np.py` computes the constraint residuals of observed telemetry with the arithmetic of `scripts/tailguard_train.py`; `fault_type_selection.py` and `extended_analyses.py` check both against the stored development-fault counts and the saved constraint stream before reporting anything.

## Appendix analyses

| Manuscript item | Script | Result file (`analysis/results/appendix/`) |
|---|---|---|
| Tail-surprise calibration by regime (Section 4.3, Table A3) | `appendix_analyses.py <root> tail <out>` | `tail_diagnostics.csv` |
| Development-fault sensitivity: other seeds and magnitudes (Section 4.5, Table A4) | `appendix_analyses.py <root> dev <out>` | `dev_fault_sensitivity.csv` |
| Per-regime forecast and constraint risk of the refitted hold-out models (Section 4.4, Table A6) | `appendix_analyses.py <root> risk <out> runs/holdout_a7` | `holdout_risks.csv` |
| Absolute recall and false-alarm intervals (Section 4.2, Table A5) | `abs_ci.py <root> <prefix> <out.csv>` | `abs_ci_2305.csv`, `abs_ci_2204.csv` |

These read no test labels except `abs_ci.py`. Tables A1 (constraint registry), A2 (protocol conditions) and A7 (notation) are built from `manifests/constraints.json`, the cited papers and Section 3.
