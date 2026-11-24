# TailGuard: reproducible code

Code, frozen protocol manifests and result tables for

> F. Alshanketi, *Tail-Probability Scores with Dependence-Robust False-Alarm Guarantees for Intrusion Detection in Industrial Control Systems*, *Mathematics* (MDPI), 2026 (submitted).

Every number in the paper can be regenerated from the public HAI 23.05, HAI 22.04 and BATADAL releases with the scripts below. The result tables produced for the paper are included: `results/reports/` (prespecified analysis, Sections 4.1–4.3), `results/reports_max/` (max tail-surprise analysis, Section 4.5), `analysis/results/hai2305/` (refitted regime hold-out, A7, alarm-budget sweep, residual scaling, calibration window, invariance coefficient, degraded registry, fault-type selection, latency; Sections 4.2–4.6), `analysis/results/hai2204/` (frozen HAI 22.04 confirmation and its budget sweep, Section 4.6), `analysis/results/appendix/` (Tables A3–A6), `analysis/results/propositions/` (Propositions 4 and 5, simulations, tail-fit diagnostics and the selection-free calibration subset; Section 4.7, Appendix B) and `batadal/results/` (BATADAL confirmation, Section 4.8). `paper/check_claims.py`, `paper/check_claims_max.py`, `paper/check_claims_extended.py` and `paper/check_claims_revision.py` recompute the claims from them.

**Repository layout.** `scripts/` pipeline steps · `manifests/` HAI 23.05 frozen protocol, data hashes, constraint registry and unsealing log · `hai2204/manifests/` the HAI 22.04 confirmation protocol (frozen before its data were prepared), split, constraints and unsealing log · `results/` result tables of the original analysis · `analysis/` scripts and result tables of the further analyses of the frozen scores · `batadal/` the BATADAL confirmation: frozen protocol, attack intervals, pipeline, split and constraint manifests, result tables · `paper/` figures and claim checks · `tests/` metric contracts · `kaggle/` the notebook cells used for GPU training (`train_cell.py`, `holdout_a7_cell.py`). Python 3.11 or later.

## 1. Quick check (no data or GPU needed, < 1 minute)

```bash
pip install -r requirements.txt
python tests/test_metrics.py                 # alarm-episode, event and calibration contracts
python paper/check_claims.py results/reports # recomputes 59 headline claims of the paper from the result tables
python paper/check_claims_max.py             # recomputes the claims of Section 4.5 from results/reports_max
python paper/check_claims_extended.py        # recomputes the claims of Sections 4.2-4.6 and Appendix A (Tables 3, 9, 11, 12, 14, A3-A11)
python paper/check_claims_revision.py        # recomputes the claims of Table B2, the tail-fit and per-attack analyses, the split-calibration check (Section 4.7) and BATADAL (Section 4.8, Table 17)
python paper/make_figures.py                 # redraws Figures 1-6 (600 dpi) from the result tables
python paper/verify_numbers.py <manuscript.txt> results/reports manifests/step2_report.txt  # every number in a text export of the paper
```

## 2. Data

HAI 23.05 is distributed by its authors under CC BY-SA 4.0 through Git LFS:

```bash
git clone https://github.com/icsdataset/hai.git data/raw/hai
cd data/raw/hai && git checkout 2a814ce && git lfs pull --include "hai-23.05/*" && cd ../../..
```

The SHA-256 hashes, sizes and row counts of all files used in the paper are recorded in `manifests/hai2305_data_manifest.json`; Step 1 recomputes them and stops if any file differs. The data files are not redistributed here.

BATADAL (Section 4.8) is distributed by its organisers at https://www.batadal.net/data.html. Place `BATADAL_dataset03.csv`, `BATADAL_dataset04.csv` and `BATADAL_test_dataset.csv` (unzipped) in `batadal/raw/`; their SHA-256 hashes and row counts are recorded in `batadal/split_manifest.json`.

## 3. Full pipeline

| Step | Script | What it does | Paper | Hardware / time used |
|---|---|---|---|---|
| 1 | `scripts/step1_audit_hai.py` | Verifies hashes, rows, sampling, labels (read-only) | §3.3 | CPU |
| 2 | `scripts/step2_prepare.py` | Chronological 60/20/20 split, eligibility mask (W=120), feature roles, regimes, label join; writes `data/derived/` and `manifests/split_manifest.json` | §3.2.1, §3.4.1–3.4.2 | CPU (48 s on our desktop) |
| 3 | `scripts/step3_baselines.py` | Persistence, ridge AR, PCA, Isolation Forest; 36 development injections (seed 17) | §3.4.3, Table 3 | CPU |
| 4 | `scripts/step4_constraints.py` | Fits and screens the 14 candidate process relations; writes `manifests/constraints.json` (11 retained) | §3.2.1, §3.2.3, Fig. 2 | CPU |
| 5 | `scripts/tailguard_train.py --stage all` | 27-candidate search, selection on development injections, 5 seeds of TailGuard, matched GRU, ablations A1–A3, A10, A11, leave-one-regime-out runs (92 runs) | §3.2.2, §3.4.3, §3.6 | 2 × Tesla T4 (Kaggle), two sessions of at most 11 h; resumable |
| 6 | `scripts/step7_deep_baselines.py --model all` | USAD, TranAD, GDN (authors' model definitions at pinned commits, common training adapter), 5 seeds each | Table 3 | CPU or GPU |
| 7 | `scripts/step6_evaluate.py --unseal` | Joins test labels once and writes all result tables, paired block bootstrap (10,000 draws), sign-flip tests and Holm correction | §3.5, §4 | CPU |

After Steps 2–3, `python tests/check_baselines_against_paper.py` re-evaluates the classical baselines and compares them with Table 6 (reproduced to all printed decimals in our re-run).

Run the steps in order from the repository root (`run_all.sh` or `run_all.bat`). Every training script checkpoints each epoch and skips finished runs, so an interrupted run continues where it stopped. `kaggle/train_cell.py` is the notebook cell used for Step 5. Test labels are read only by Step 7; `--fake-labels` runs the evaluation on synthetic labels for code testing. The one-time unsealing is logged in `manifests/unseal_log.txt`.

Ablations A4, A5 and A9 need no retraining: they are computed from the saved score streams. The stability screen of Step 4 rejected no relation that passed the identifiability rule, so A6 equals the full model. A7 (randomised constraint tags) and the refitted regime hold-out are trained in Section 6.

## 4. Mapping to the paper

| Paper item | Produced by | File |
|---|---|---|
| Table 6, Fig. 3 (published protocols) | Step 7 | `results/reports/table_published_protocols.csv` |
| Table 7, Fig. 4 (common protocol) | Step 7 | `results/reports/table_common_protocol.csv` |
| Table 8 (paired effects) | Step 7; A7 row from Section 6 | `results/reports/table_paired_vs_tailguard.csv`, `analysis/results/hai2305/A7_vs_tailguard.csv` |
| Table 10, Fig. 5 (ablations) | Step 7; A7 row from Section 6 | `table_common_protocol.csv`, `table_published_protocols.csv`, `analysis/results/hai2305/A7_runs_*.csv` |
| Table 3 (development faults) | Step 3 | `analysis/results/hai2305/dev_injections_spec.csv` |
| Table 9 (matched burden, duration) | Section 6 | `analysis/results/hai2305/matched_burden.csv`, `budget_sensitivity.csv`, `duration_aware.csv` |
| Table 11 (residual scaling) | Section 6 | `analysis/results/hai2305/score_ablation_2305.csv`, `analysis/results/hai2204/score_ablation_2204.csv` |
| Table 12, Fig. 6 (refitted hold-out) | Section 6 | `analysis/results/hai2305/clean_holdout_*.csv`, `regime_fa_consistent*.csv`, `recalibration_consistent*.csv` |
| Table 13 (max score) | Section 5 | `results/reports_max/` |
| Table 14 (HAI 22.04 confirmation) | Section 7 | `analysis/results/hai2204/reports/`, `reports_max/` |
| Table A1 (constraint registry) | Step 4 | `manifests/constraints.json` |
| Tables A3–A6 (tail calibration, development-fault sensitivity, absolute intervals, hold-out risks) | `analysis/scripts/appendix_analyses.py`, `abs_ci.py` | `analysis/results/appendix/` |
| Fig. 2, §3.2.3 (constraint registry) | Step 4 | `manifests/constraints.json`, `manifests/step4_report.txt` |
| §3.3, §3.4 data facts (rows, hours, regimes, cut points) | Steps 1–2 | `manifests/hai2305_audit_report.txt`, `manifests/step2_report.txt` |
| Figure 4 (alarm-budget sweep, both releases) | Section 6 g | `analysis/results/hai2305/budget_sensitivity.csv`, `analysis/results/hai2204/budget_sensitivity_hai2204.csv` |
| Tables A7–A11, ridge fusion, latency | Section 6 e, g–i | `analysis/results/hai2305/calibration_window_hai2305.csv`, `lambda_inv*_hai2305.csv`, `constraint_degradation*_hai2305.csv`, `fault_type_selection.csv`, `fault_detection_by_*.csv`, `ridge_constraint_fusion_*.csv`, `latency.csv` |
| Table 16, Table A12 (block-maximum and closure thresholds, block and episode level) | `analysis/scripts/prop5_eval.py` | `analysis/results/propositions/p5_*.csv`, `p5h_*.csv` |
| Section 4.7 (thresholds and block level from disjoint halves of the calibration record) | Section 8 a | `analysis/results/propositions/split_selection.csv`, `split_prop5.csv` |
| Section 4.3 (generalized Pareto shape of the excesses) | Section 8 b | `analysis/results/propositions/tail_fit_gpd.csv` |
| Sections 4.3 and 4.6 (attacks detected by TailGuard and by ridge autoregression) | Section 6 d; Section 8 c | `analysis/results/hai2305/attacks_detection.csv`, `analysis/results/hai2204/attacks_detection_2204.csv` |
| Table B1, Figure B1 (Gaussian AR(1) simulation) | `analysis/scripts/sim.py`, `figB1.py` | `analysis/results/propositions/sim.csv`, `sim_blocks.csv` |
| Table B2 (heavy-tailed and volatility-clustered simulation) | Section 8 d | `analysis/results/propositions/sim_t3.csv`, `sim_garch.csv`, `sim_garcht.csv`, `simb_t3.csv`, `simb_garch.csv` |
| Table 17 (BATADAL) | Section 9 | `batadal/results/results_summary.csv`, `results_runs.csv` |
| Deviations from the prespecified protocol | — | `manifests/protocol_deviations.md` |

Result tables use the internal reference keys of the protocol: `[14]` and `[15]` in the CSV files are references [22] and [23] of the paper (Radovanovic et al., MAKE 2026, articles 276 and 306).

## 5. Max tail-surprise analysis (Section 4.5, Proposition 3)

A second, disclosed analysis re-scores the frozen models with the channel-maximum tail surprise. The score was chosen on the 36 development faults only and frozen before its single test evaluation (`manifests/max_score_protocol.md`, `manifests/unseal_log_max.txt`).

| Step | Script | What it does |
|---|---|---|
| a | `scripts/extract_errors.py` then `scripts/dev_score_variants.py` | Per-channel errors of a trained model, then the development-fault grid of 180 score variants (no labels) |
| b | `scripts/rescore_max.py --models runs/train` | Re-scores every trained GRU run (CPU, ~2–6 min per model); writes `runs_max/train/` |
| c | `scripts/rescore_max_classical.py` | Persistence, ridge, PCA under the max score (Isolation Forest keeps its own score) |
| d | `scripts/step7_deep_rescore.py --model all` then `scripts/rescore_max_deep.py` | Retrains USAD, TranAD, GDN keeping per-channel errors, then re-scores them |
| e | `scripts/step6_evaluate_max.py --unseal` | One test evaluation; writes `reports_max/` (released as `results/reports_max/`) |

Each comparator uses whichever score detects more of its own development faults (ridge and PCA keep the prespecified score), as recorded in the protocol file.

## 6. Refitted regime hold-out, A7 and analyses of the frozen scores (HAI 23.05)

| Step | Script | What it does | Hardware |
|---|---|---|---|
| a | `scripts/holdout_a7_runs.py` (cell `kaggle/holdout_a7_cell.py`) | 60 refitted hold-out runs (TailGuard, matched GRU, A2, A10 × 3 regimes × 5 seeds; scalers, discrete levels, constraints and screening refitted on retained regimes, window-crossing targets moved to the withheld side) and 5 A7 runs; saves both scores | 2 × T4, 8.5 h |
| b | `analysis/scripts/holdout_a7_eval.py --root . --runs runs/holdout_a7 --out <dir>` | Withheld-regime false alarms and recall, H2 tests, A7 paired bootstrap | CPU |
| c | `analysis/scripts/fa_consistent.py`, `recal_consistent.py` | Withheld-regime false episodes counted as in FA24 (episodes over the whole sequence, false if they start on a normal row); recalibration from 1–32 h of withheld-regime normal data | CPU |
| d | `analysis/scripts/frozen_score_analyses.py --root . --out <dir>` | Alarm-budget sensitivity (A9), matched burden, block length, duration-aware metrics, missed attacks, development-fault list | CPU |
| e | `analysis/scripts/tail_sensitivity.py`, `score_ablation.py`, `ridge_constraint_fusion.py` | Tail-fit sensitivity; the same residuals under tail surprise, max z-score and mean squared error; ridge fused with the constraint stream | CPU |
| f | `analysis/scripts/fig6.py` | Figure 6 | CPU |
| g | `analysis/scripts/extended_analyses.py --root <release root> --prefix hai2305\|hai2204 --out <dir> reproduce budget calwin lambda_inv constraints` | Reproduces the common-protocol table from the saved streams, alarm-budget sweep (Figure 4), calibration-window length (Table A8), invariance coefficient over the 27 search candidates (Table A9), degraded constraint registry (Table A10) | CPU, no PyTorch |
| h | `analysis/scripts/fault_type_selection.py --root . --out <dir>` | Re-scores the 27 search checkpoints on the development faults (NumPy forecaster `np_gru.py`, checked against the stored counts) and repeats selection with one fault type left out (Table A11) | CPU, about 1 h |
| i | `analysis/scripts/latency.py --root . --out <csv>` | Streaming cost per second of telemetry on one CPU thread (Section 4.6) | CPU |

Every reading of test labels is appended to `manifests/unseal_log.txt`. See `analysis/README.md` for the file-to-table map.

## 7. HAI 22.04 confirmation (Section 4.6)

The protocol in `hai2204/manifests/HAI2204_PROTOCOL.md` was frozen before any HAI 22.04 data were prepared: the same rules, hyperparameters, scores and comparators as HAI 23.05, no selection on HAI 22.04, and one reading of the test labels after all runs.

```bash
git -C data/raw/hai lfs pull --include "hai-22.04/*"
mkdir -p hai2204/scripts && cp scripts/*.py hai2204/scripts/      # HAI 22.04 is evaluated in its own working root
cd hai2204
export TAILGUARD_PREFIX=hai2204 HAI2204_DIR=../data/raw/hai/hai-22.04
python scripts/step2_prepare_2204.py                               # split, eligibility, roles, regimes, labels (Attack column)
python scripts/step4_constraints.py                                # same 14-relation registry, refitted (11 retained)
python scripts/step3_baselines.py                                  # classical baselines + 36 development faults (seed 17)
python scripts/confirm_2204.py --part 0 --parts 2 --max-hours 11   # TailGuard, matched GRU, A1 x 5 seeds (GPU; part 1 on the second GPU)
python scripts/step7_deep_rescore.py --model all                   # USAD, TranAD, GDN x 5 seeds (GPU or CPU)
mkdir -p runs/deep && cp -r runs/deep_rescore/* runs/deep/
python ../analysis/scripts/build_runs_max_2204.py                    # runs_max/ from the saved max-score arrays
python scripts/rescore_max_classical.py --only persistence && python scripts/rescore_max_deep.py   # max score for comparators
python scripts/step6_evaluate.py --unseal && python scripts/step6_evaluate_max.py --unseal          # the single evaluation
```

`build_runs_max_2204.py` builds `runs_max/` from the `max_*` arrays that `confirm_2204.py` saves next to the prespecified ones; ridge, PCA and Isolation Forest keep their prespecified scores, as in the protocol. The result tables are released in `analysis/results/hai2204/`. Its labels are rebuilt from the `Attack` column of the four test files by `step2_prepare_2204.py`.

## 8. Calibration checks: disjoint calibration halves, tail fit, per-attack overlap, further simulations

| Step | Script | What it does | Hardware |
|---|---|---|---|
| a | `analysis/scripts/split_cal.py --root . --out <dir>` | Splits the calibration partition in time order into two halves. Repeats the selection over the 27 search candidates with every closure threshold computed on the first half, and computes the block-maximum level of Proposition 5 on the second half, which no selection step has seen (Section 4.7). Checks the stored development-fault counts before reporting | CPU, about 1 h |
| b | `analysis/scripts/tail_fit_gpd.py --root . --out <csv>` | Generalized Pareto shape and mean-excess ratio of the excesses over the 99th percentile, per forecast channel and constraint, on validation and calibration data (Section 4.3) | CPU |
| c | `analysis/scripts/attack_overlap_2204.py --root hai2204 --raw data/raw/hai/hai-22.04 --out <csv>` | For each of the 58 HAI 22.04 attacks, the number of TailGuard seeds and whether ridge autoregression raise an alarm inside it (Section 4.6). The HAI 23.05 counterpart is `attacks_detection.csv` of Section 6 d | CPU |
| d | `analysis/scripts/sim_ext.py scripts <out.csv> <t3\|garch\|garch_t> <first rep> <last rep + 1> [rho ...]` | The simulation of Appendix B with Student-t (3 d.f.), GARCH(1,1) and GARCH(1,1)-t (5 d.f.) innovations in place of Gaussian ones; record lengths, scores and threshold rules as in `sim.py` (Table B2). Released files: repetitions 0-2 at five correlations (`sim_*.csv`) and repetitions 3-22 at 0.99 for the block share (`simb_*.csv`) | CPU |

Steps a and c read test labels and append a line to the unsealing log; b and d read none.

## 9. BATADAL confirmation (Section 4.8)

The protocol in `batadal/BATADAL_PROTOCOL.md` was frozen before any model was fitted on BATADAL: the HAI 23.05 model, hyperparameters, scores and threshold rules, with the adaptations that hourly data require listed in the file, no hyperparameter search and no selection on BATADAL. The 14 attack intervals are those published with the competition (`batadal/batadal_attacks.json`); the `ATT_FLAG` column is dropped when the files are read.

```bash
python batadal/tailguard_batadal.py --stage prepare      # audit, 60/20/20 split, scaling; writes batadal/split_manifest.json
python batadal/tailguard_batadal.py --stage constraints  # documented C-Town relations, screened; writes batadal/constraints.json
python batadal/tailguard_batadal.py --stage train        # TailGuard, matched GRU, A1: 5 seeds each (CPU, about 2 min per run)
python batadal/tailguard_batadal.py --stage baselines    # ridge autoregression, PCA, persistence
python batadal/tailguard_batadal.py --stage evaluate     # thresholds on calibration data, then the 14 attacks; writes results_runs.csv, results_summary.csv
```

The tables written for the paper are released in `batadal/results/` (`results_summary.csv`: means over seeds for each score, smoothing length and threshold rule; `results_runs.csv`: every run; `train.log`: epochs and validation loss ratio of each run). Table 17 is the block `smooth = 30`, `score = prespecified`. The competition metrics (S score with its time-to-detection and classification components, F1) are computed on the official test file at the same calibrated threshold.

## 10. Determinism

Seeds are fixed (`17, 29, 43, 71, 101` for models; `17` for development injections; `20261011` for the bootstrap). Data preparation, classical baselines, constraint fitting and evaluation are deterministic. GPU training is seeded but cuDNN kernels are not bit-deterministic, so retrained models can differ in the last digits; the reported means and standard deviations over five seeds are the quantities to compare.

## 11. Licence and third-party code

Code in this repository is released under the MIT licence (`LICENSE`). `scripts/deep_models.py` contains model definitions vendored from the USAD, TranAD and GDN repositories at the commits listed in its header, under their own licences (`THIRD_PARTY_NOTICES.md`). The HAI and BATADAL datasets are not included.
