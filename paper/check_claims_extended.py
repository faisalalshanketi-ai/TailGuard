"""Recompute the claims of Sections 4.2-4.6 and Appendix A (Tables 3, 9, 11, 12, 14, A3-A11) from the released result tables.
Usage: python paper/check_claims_extended.py   (no data or GPU needed)"""
import os, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "analysis", "results", "hai2305"); B = os.path.join(ROOT, "analysis", "results", "hai2204")
R = lambda d, f, **k: pd.read_csv(os.path.join(d, f), **k)
ok = fail = 0
def check(name, value, target, tol):
    global ok, fail
    good = abs(value - target) <= tol; ok += good; fail += not good
    print(f"{'PASS' if good else 'FAIL'}  {name}: {value:.4f} (paper {target})")
def ms(x): return float(str(x).split("±")[0])

# ---------------- refitted regime hold-out (Table 12, Section 4.4) and H2
P = R(A, "clean_holdout_pooled.csv"); g = lambda s, m, c: float(P[(P.score == s) & (P.method == m)][c].iloc[0])
check("withheld FA24, TailGuard, prespecified", g("prespecified", "TailGuard", "test_FA24"), 0.95, 0.005)
check("withheld FA24, matched GRU, prespecified", g("prespecified", "Matched GRU", "test_FA24"), 1.20, 0.005)
check("withheld FA24, matched GRU, max", g("max", "Matched GRU", "test_FA24"), 1.27, 0.005)
check("withheld recall, TailGuard, prespecified", g("prespecified", "TailGuard", "recall"), 0.235, 0.0005)
check("withheld recall, TailGuard, max", g("max", "TailGuard", "recall"), 0.269, 0.0005)
check("withheld detected, TailGuard, max", g("max", "TailGuard", "detected"), 14.0, 0.05)
check("withheld detected, GRU, max", g("max", "Matched GRU", "detected"), 11.2, 0.05)
check("withheld normal-training FA24, TailGuard, max", g("max", "TailGuard", "train_FA24"), 2.50, 0.005)
H = R(A, "clean_holdout_H2.csv")
check("H2: TailGuard − GRU detected, max, sign-flip p", float(H[(H.score == "max") & H.comparison.str.contains("Matched GRU") & H.comparison.str.startswith("TailGuard")]["detected_p_signflip"].iloc[0]), 0.012, 0.0015)
check("H2: TailGuard − A2 detected difference, prespecified", float(H[(H.score == "prespecified") & H.comparison.str.startswith("TailGuard − A2")]["detected_diff_total"].iloc[0]), 0, 0)
C = R(A, "regime_fa_consistent.csv"); c = lambda r: float(C[(C.score == "pre") & (C.model == "tailguard") & (C.variant == "clean hold-out") & (C.regime == r)]["FA24"].iloc[0])
check("withheld FA24 low", c("low"), 1.29, 0.005); check("withheld FA24 mid", c("mid"), 2.32, 0.005); check("withheld FA24 high", c("high"), 0.0, 0)
# ---------------- recalibration (Section 4.4)
Q = R(A, "recalibration_consistent.csv"); q = Q[(Q.score == "pre") & (Q.method == "TailGuard") & (Q.regime == "low") & Q.recal_hours.isin([1, 2, 4])]
check("recalibrated low-regime FA24 max (1-4 h)", q.FA24.max(), 1.55, 0.005); check("recalibrated low-regime detections min", q.detected.min(), 7.8, 0.05)
qg = Q[(Q.score == "pre") & (Q.method == "GRU") & (Q.regime == "low") & Q.recal_hours.isin([1, 2, 4])]
check("recalibrated low-regime GRU FA24 max", qg.FA24.max(), 78.6, 0.05)
# ---------------- A7 (Sections 4.3, 4.5)
S7 = R(A, "A7_vs_tailguard.csv").set_index("score")
check("A7 recall, prespecified", S7.loc["prespecified", "A7_recall"], 0.281, 0.0005)
check("TailGuard − A7 recall difference", S7.loc["prespecified", "recall_diff"], 0.073, 0.0015)
check("  95% CI low", S7.loc["prespecified", "CI_low"], 0.025, 0.0015); check("  95% CI high", S7.loc["prespecified", "CI_high"], 0.132, 0.0015)
check("  sign-flip p", S7.loc["prespecified", "p_signflip"], 0.031, 0.0015)
check("TailGuard − A7, max score", S7.loc["max", "recall_diff"], 0.055, 0.0015)
# ---------------- matched burden, duration, missed attacks, block length, tail (Tables 8-10)
M = R(A, "matched_burden.csv", index_col=0)
check("matched 1.5/day TailGuard", M.loc["TailGuard", "1.5"], 0.374, 0.0005); check("matched 1.5/day TailGuard max", M.loc["TailGuard, max score", "1.5"], 0.382, 0.0005)
check("matched 1.5/day GRU", M.loc["Matched GRU", "1.5"], 0.325, 0.0005); check("matched 1.5/day ridge", M.loc["Ridge AR", "1.5"], 0.223, 0.0005)
check("matched 0.5/day USAD", M.loc["USAD", "0.5"], 0.251, 0.0005); check("matched 2/day TailGuard", M.loc["TailGuard", "2.0"], 0.483, 0.0005)
D = R(A, "duration_aware.csv").set_index("method")
check("attack time alarmed, TailGuard", D.loc["TailGuard", "frac_attack_time_alarmed"], 0.154, 0.0005)
check("detected within 300 s, TailGuard", D.loc["TailGuard", "detected_within_300s"], 0.335, 0.0005)
check("capped median delay, TailGuard (s)", D.loc["TailGuard", "median_capped_delay_s"], 96, 0.5)
AT = R(A, "attacks_detection.csv"); cols = [c_ for c_ in AT.columns if "seeds" in c_]
check("attacks missed by every detector", float((AT[cols].sum(axis=1) == 0).sum()), 23, 0)
BL = R(A, "block_length.csv"); deep = BL[BL.comparison.str.contains("USAD|TranAD|GDN")]
check("deep-baseline CIs excluding zero at 3/6/12 h blocks (count of 18)", float((deep.CI_low >= 0).sum()), 18, 0)
TS = R(A, "tail_sensitivity.csv", header=[0, 1], index_col=[0, 1])
check("prespecified recall without exponential tail", float(TS.loc[("prespecified (top-3, beta 0.5)", "no exponential tail (ECDF only)"), ("event_recall", "mean")]), 0.100, 0.0005)
# ---------------- residual scaling (Table 11)
for d_, f_, rows in ((A, "score_ablation_2305.csv", [("ridge", "tail surprise (TailGuard policy)", "F1_5pctFPR", 0.683), ("ridge", "max z-score", "F1_5pctFPR", 0.687), ("ridge", "mean squared error", "F1_5pctFPR", 0.077)]),
                     (B, "score_ablation_2204.csv", [("ridge", "tail surprise (TailGuard policy)", "event_recall", 0.621), ("ridge", "max z-score", "event_recall", 0.621)])):
    X = R(d_, f_).groupby(["detector", "score"]).mean(numeric_only=True)
    for det, sc, col, t in rows: check(f"{f_[15:19]} {det} {sc} {col}", X.loc[(det, sc), col], t, 0.0005)
# ---------------- HAI 22.04 confirmation (Section 4.6, Table 14)
T0 = R(os.path.join(B, "reports"), "table_common_protocol.csv", index_col=0); TX = R(os.path.join(B, "reports_max"), "table_common_protocol.csv", index_col=0)
check("HAI 22.04 TailGuard recall, prespecified", ms(T0.loc["TailGuard", "event_recall"]), 0.276, 0.0005)
check("HAI 22.04 TailGuard FA24", ms(T0.loc["TailGuard", "FA24"]), 0.50, 0.005)
check("HAI 22.04 TailGuard recall, max", ms(TX.loc["TailGuard", "event_recall"]), 0.307, 0.0005)
check("HAI 22.04 matched GRU recall", ms(T0.loc["GRU (pooled ERM)", "event_recall"]), 0.272, 0.0005)
check("HAI 22.04 constraint residual only", ms(T0.loc["Residual-only (beta=1)", "event_recall"]), 0.448, 0.0005)
check("HAI 22.04 ridge recall", ms(T0.loc["ridge", "event_recall"]), 0.621, 0.0005)
check("HAI 22.04 GDN FA24 > 30", float(ms(T0.loc["GDN", "FA24"]) > 30), 1, 0)
check("HAI 22.04 A1 recall", ms(T0.loc["A1_no_physics", "event_recall"]), 0.252, 0.0005)
check("HAI 22.04 A4 recall", ms(T0.loc["A4_beta0 (temporal score only)", "event_recall"]), 0.224, 0.0005)
PB = R(os.path.join(B, "reports"), "table_published_protocols.csv"); f1 = lambda m: ms(PB[(PB.method == m) & PB.protocol.str.contains("5% FPR")]["F1"].iloc[0])
check("HAI 22.04 F1 at 5% FPR, TailGuard", f1("TailGuard"), 0.336, 0.0005); check("HAI 22.04 F1 at 5% FPR, GRU", f1("GRU (pooled ERM)"), 0.231, 0.0005)
PX = R(os.path.join(B, "reports_max"), "table_paired_vs_tailguard.csv", index_col=0)
check("HAI 22.04 TailGuard − TranAD, max", PX.loc["TranAD", "recall_diff"], 0.097, 0.0005)

# ---------------- Appendix A
C2 = os.path.join(ROOT, "analysis", "results", "appendix"); import numpy as np
TD = R(C2, "tail_diagnostics.csv"); tl = TD[TD.level_c.astype(str) != "tail"].copy(); tl["c"] = tl.level_c.astype(float)
fc2 = tl[(tl.stream == "forecast channels") & (tl.c == 2)].median_rate
check("forecast median rate at c=2, min over regimes", fc2.min(), 0.0102, 0.00005); check("  max", fc2.max(), 0.0136, 0.00005)
cr2 = tl[(tl.stream == "constraint residuals") & (tl.c == 2)].median_rate
check("constraint median rate at c=2, min", cr2.min(), 0.0086, 0.00005); check("  max", cr2.max(), 0.0102, 0.00005)
mx = tl[(tl.stream == "forecast channels") & (tl.c == 2)]; check("forecast maximum ratio at c=2, high regime", float(mx[mx.regime == "high"].max_channel_rate_at_c_plus_log10m.iloc[0] / 0.01), 7.09, 0.01)
DV = R(C2, "dev_fault_sensitivity.csv"); pv = DV.pivot_table(index="fault_set", columns=["model", "score"], values="detected")
gain = pv[("TailGuard", "max")] - pv[("TailGuard", "prespecified")]
check("max - prespecified, TailGuard, min over 7 sets", gain.min(), 4, 0); check("  max", gain.max(), 6, 0)
check("TailGuard max leads GRU max and A1 max on every set", float(((pv[("TailGuard", "max")] > pv[("Matched GRU", "max")]) & (pv[("TailGuard", "max")] > pv[("A1 (no constraint term)", "max")])).all()), 1, 0)
RK = R(C2, "holdout_risks.csv"); RK["grp"] = np.where(RK.group.str.startswith("ret"), "ret", "wh")
w = RK.groupby(["withheld", "model", "grp"]).apply(lambda x: np.average(x.constraint_risk, weights=x.n_targets)).unstack("grp")
check("withheld constraint risk, TailGuard, max over regimes", w.loc[(slice(None), "tailguard"), "wh"].max(), 0.224, 0.0005)
check("withheld constraint risk, GRU, min over regimes", w.loc[(slice(None), "gru"), "wh"].min(), 9.35, 0.005)
AB = R(C2, "abs_ci_2305.csv").set_index("method")
check("TailGuard FA24 interval low", AB.loc["TailGuard (prespecified)", "FA24_CI_low"], 0.36, 0.005); check("  high", AB.loc["TailGuard (prespecified)", "FA24_CI_high"], 2.47, 0.005)

# ---------------- alarm-budget sweeps (Figure 4), calibration window (Table A8), invariance coefficient (Table A9),
# ---------------- degraded registry (Table A10), ridge + constraint stream, HAI 22.04 paired rows (Sections 4.2-4.6)
BS = R(A, "budget_sensitivity.csv"); BS4 = R(B, "budget_sensitivity_hai2204.csv")
def curve(T, m):
    g = T[T.method == m].sort_values("FA24"); return g.FA24.to_numpy(), g.event_recall.to_numpy()
xs = np.linspace(1.3, 3.2, 1901)
def crossing(m):
    d = np.interp(xs, *curve(BS, m)) - np.interp(xs, *curve(BS, "Ridge AR")); return float(xs[np.where(np.diff(np.sign(d)) != 0)[0][0]])
check("TailGuard above ridge up to (FA/day)", crossing("TailGuard"), 2.4, 0.05); check("  max score", crossing("TailGuard, max score"), 2.2, 0.05)
r4 = BS4[BS4.method == "Ridge AR"].set_index("budget_per_day"); s4 = BS4[BS4.method == "TailGuard"].set_index("budget_per_day")
check("HAI 22.04 ridge leads TailGuard at every budget", float((r4.event_recall > s4.event_recall).all()), 1, 0)
check("HAI 22.04 ridge recall at budget 0.25", r4.loc[0.25, "event_recall"], 0.517, 0.0005); check("  its FA24", r4.loc[0.25, "FA24"], 0.50, 0.005)
CW = R(A, "calibration_window_hai2305.csv"); cw = lambda m, h: CW[(CW.method == m) & (CW.calibration_hours == h)].iloc[0]
for h, v in ((6, 1.52), (12, 1.39), (24, 1.39), (49.8, 1.33)): check(f"TailGuard FA24 with {h} h calibration", cw("TailGuard", h).FA24, v, 0.005)
check("ridge FA24 with 12 h calibration", cw("Ridge AR", 12).FA24, 2.53, 0.005); check("ridge FA24 with 6 h calibration", cw("Ridge AR", 6).FA24, 2.53, 0.005)
LI = R(A, "lambda_inv_summary_hai2305.csv").set_index("lambda_inv")
for k, v in ((0.0, 0.331), (0.1, 0.342), (1.0, 0.350)): check(f"mean test recall, lambda_inv={k}", LI.loc[k, "test_event_recall"], v, 0.0005)
check("mean development faults, lambda_inv=0", LI.loc[0.0, "dev_detected"], 10.4, 0.05); check("  lambda_inv=1", LI.loc[1.0, "dev_detected"], 10.8, 0.05)
check("mean validation loss, lambda_inv=0", LI.loc[0.0, "val_worst_regime_loss"], 0.61, 0.005); check("  lambda_inv=1", LI.loc[1.0, "val_worst_regime_loss"], 0.72, 0.005)
check("largest change in any onset regime", float((LI[["recall_low", "recall_mid", "recall_high"]].max() - LI[["recall_low", "recall_mid", "recall_high"]].min()).max()), 0.034, 0.0005)
CD = R(A, "constraint_degradation_summary_hai2305.csv").set_index("variant"); dr = CD[CD.index.str.startswith("drop ")]
check("drop-one recall min", dr.event_recall.min(), 0.335, 0.0005); check("  max", dr.event_recall.max(), 0.362, 0.0005)
check("drop-one FA24 min", dr.FA24.min(), 1.27, 0.005); check("  max", dr.FA24.max(), 1.65, 0.005)
for k, v in ((8, 0.345), (5, 0.336), (3, 0.334)): check(f"random subsets of {k}", CD.loc[f"random subset of {k}", "event_recall"], v, 0.0005)
check("coefficients ±10%", CD.loc["coefficients perturbed ±10%", "event_recall"], 0.325, 0.0005); check("coefficients ±25%", CD.loc["coefficients perturbed ±25%", "event_recall"], 0.325, 0.0005)
check("forecast score alone", CD.loc["forecast only (0)", "event_recall"], 0.354, 0.0005)
for rel, tag in ((A, "2305"), (B, "2204")):
    F = R(rel, f"ridge_constraint_fusion_{tag}.csv"); fz = lambda s: F[F.score == s].iloc[0]
    d_pre = fz("ridge + constraints, prespecified (beta 0.5)").event_recall - fz("ridge, prespecified (top-3)").event_recall
    d_max = fz("ridge + constraints, max fusion").event_recall - fz("ridge, max (top-1)").event_recall
    check(f"ridge + constraints, prespecified, HAI {tag} (points)", 100 * d_pre, {"2305": 1.9, "2204": -10.3}[tag], 0.05)
    check(f"ridge + constraints, max, HAI {tag} (points)", 100 * d_max, {"2305": 0.0, "2204": 3.4}[tag], 0.05)
F = R(A, "ridge_constraint_fusion_2305.csv"); check("ridge max rule alone, HAI 23.05", F[F.score == "ridge, max (top-1)"].event_recall.iloc[0], 0.327, 0.0005)
check("ridge + constraints prespecified FA24, HAI 23.05", F[F.score == "ridge + constraints, prespecified (beta 0.5)"].FA24.iloc[0], 2.22, 0.005)
P4 = R(os.path.join(B, "reports"), "table_paired_vs_tailguard.csv", index_col=0)
check("HAI 22.04 TailGuard − GRU recall (points)", 100 * P4.loc["GRU (pooled ERM)", "recall_diff"], 0.4, 0.05)
check("HAI 22.04 TailGuard − ridge recall (points)", 100 * P4.loc["ridge", "recall_diff"], -34.5, 0.05)
lo, hi = [float(x) for x in P4.loc["ridge", "recall_CI95"].strip("[]").split(",")]; check("  CI low", 100 * lo, -46.7, 0.05); check("  CI high", 100 * hi, -22.6, 0.05)
for name, (v, t) in {"reproduced recall, HAI 23.05 TailGuard": (R(A, "reproduce_hai2305.csv").set_index("method").loc["TailGuard", "event_recall"], 0.354),
                     "reproduced recall, HAI 22.04 TailGuard": (R(B, "reproduce_hai2204.csv").set_index("method").loc["TailGuard", "event_recall"], 0.276)}.items():
    check(name, v, t, 0.0005)
FT = R(A, "fault_type_selection.csv"); FC = R(A, "fault_detection_by_candidate.csv"); FD = R(A, "fault_detection_by_duration.csv")
check("search candidates reproducing their stored development-fault counts", float((FC["all"] == FC["stored_detected"]).sum()), 27, 0)
check("selection with bias or freeze left out keeps the selected configuration", float((FT.selected.iloc[1:3] == FT.selected.iloc[0]).all()), 1, 0)
check("every leave-one-type-out choice uses the constraint term", float(FT["uses constraint term"].all()), 1, 0)
check("selected model: freeze faults detected", FD[FD.kind == "freeze"]["sum"].sum(), 1, 0)
check("selected model: bias faults detected", FD[FD.kind == "bias"]["sum"].sum(), 7, 0); check("selected model: ramp faults detected", FD[FD.kind == "ramp"]["sum"].sum(), 7, 0)
LT = R(A, "latency.csv"); check("TailGuard ms per second of telemetry", LT.ms_per_sample.iloc[0], 4.4, 0.05); check("ridge ms per second", LT.ms_per_sample.iloc[1], 1.8, 0.05)
check("TailGuard parameters", LT.parameters.iloc[0], 56183, 0)
print(f"\n{ok} passed, {fail} failed"); sys.exit(1 if fail else 0)
