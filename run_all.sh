#!/usr/bin/env bash
# Full TailGuard pipeline. Every step is resumable; re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/step1_audit_hai.py                 # verify data hashes (stops on mismatch)
python scripts/step2_prepare.py                   # splits, eligibility mask, regimes, label join
python scripts/step3_baselines.py                 # classical baselines + development injections
python scripts/step4_constraints.py               # process-constraint registry
python scripts/tailguard_train.py --stage all      # search, selection, seeds, controls, ablations, hold-outs (GPU recommended)
python scripts/step7_deep_baselines.py --model all
python scripts/step6_evaluate.py --unseal         # one-time test-label join; writes results to reports/
# --- max tail-surprise analysis (Section 4.5); CPU, several hours
python scripts/rescore_max.py --models runs/train
python scripts/rescore_max_classical.py
python scripts/step7_deep_rescore.py --model all
python scripts/rescore_max_deep.py
python scripts/step6_evaluate_max.py --unseal     # writes reports_max/
python paper/make_figures.py reports paper/figures --max-results reports_max
python paper/check_claims.py reports
# --- refitted hold-out and A7 (Sections 4.2-4.4): train runs/holdout_a7 with scripts/holdout_a7_runs.py on GPU (kaggle/holdout_a7_cell.py), then on CPU:
python analysis/scripts/holdout_a7_eval.py --root . --runs runs/holdout_a7 --out analysis/results/hai2305
python analysis/scripts/fa_consistent.py . runs/holdout_a7 analysis/results/hai2305
python analysis/scripts/recal_consistent.py . runs/holdout_a7 analysis/results/hai2305
python analysis/scripts/frozen_score_analyses.py --root . --out analysis/results/hai2305
TAILGUARD_PREFIX=hai2305 python analysis/scripts/score_ablation.py . analysis/results/hai2305/score_ablation_2305.csv
python analysis/scripts/fig6.py analysis/results/hai2305/regime_fa_consistent_runs.csv paper/figures/fig6_regimes.png
python analysis/scripts/extended_analyses.py --root . --prefix hai2305 --out analysis/results/hai2305 reproduce calwin lambda_inv constraints
python analysis/scripts/fault_type_selection.py --root . --out analysis/results/hai2305
python analysis/scripts/latency.py --root . --out analysis/results/hai2305/latency.csv
python paper/check_claims_extended.py
# --- HAI 22.04 confirmation (Section 4.6): see README Section 7 (separate working root hai2204/)
