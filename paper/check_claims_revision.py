"""TailGuard | recomputes the claims added in the revised manuscript from the released result tables (no data or GPU needed).
Covers Table B2 and Section 3.2.7 (simulation beyond Gaussian AR(1)), Section 4.3 (tail-fit diagnostics, per-attack overlap),
Section 4.6 (HAI 22.04 overlap), Section 4.7 (selection-free calibration subset) and Section 4.8, Table 17 (BATADAL).
  python paper/check_claims_revision.py"""
import os, sys
import pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = lambda *a: os.path.join(R, *a)
PR = lambda f: pd.read_csv(P("analysis", "results", "propositions", f)); ok = bad = 0
def chk(name, got, want, tol=0.0):
    global ok, bad
    good = abs(got - want) <= tol + 1e-12; ok += good; bad += not good
    print(("ok   " if good else "FAIL ") + f"{name}: {got:.4g} (paper {want})")
# ---- simulation beyond Gaussian AR(1) (Table B2)
fam = {"t3": PR("sim_t3.csv"), "garch": PR("sim_garch.csv"), "garch_t": PR("sim_garcht.csv")}
allr = pd.concat(fam.values()); upto = allr[allr.rho <= 0.99].groupby(["family", "rho"]).mean(numeric_only=True)
chk("single-channel exceedance at c=2, smallest mean", round(upto.single_c2.min(), 4), 0.0100)
chk("single-channel exceedance at c=2, largest mean", round(upto.single_c2.max(), 4), 0.0103)
chk("single-channel exceedance at c=3, smallest mean", round(upto.single_c3.min(), 4), 0.0010)
chk("single-channel exceedance at c=3, largest mean", round(upto.single_c3.max(), 4), 0.0013)
g = allr.groupby(["family", "rho"]).mean(numeric_only=True)
chk("top-three closure threshold, smallest mean", round(g.top3_tau_closure.min(), 1), 2.5)
chk("top-three closure threshold, largest mean", round(g.top3_tau_closure.max(), 1), 5.1)
st = g.loc[["t3", "garch_t"]]   # the two Student-t laws; the text rounds the Table B2 entries 4.15 and 11.76 to 4.2-11.8
chk("max-score closure threshold under Student-t laws, smallest mean (Table B2)", round(st.mx_tau_closure.min(), 2), 4.15)
chk("max-score closure threshold under Student-t laws, largest mean (Table B2)", round(st.mx_tau_closure.max(), 2), 11.76)
chk("top-three closure threshold, largest single run", round(allr.top3_tau_closure.max(), 1), 7.2)
chk("max-score closure threshold, largest single run", round(allr.mx_tau_closure.max(), 1), 17.7)
for f, a, b in (("simb_t3.csv", 0.032, 0.029), ("simb_garch.csv", 0.049, 0.035)):
    d = PR(f); chk(f"{f}: nominal block share", d.top3_block_nominal.mean(), 0.040)
    chk(f"{f}: realised block share, top-three", round(d.top3_block_realised.mean(), 3), a)
    chk(f"{f}: realised block share, max", round(d.mx_block_realised.mean(), 3), b)
# ---- tail-fit diagnostics (Section 4.3)
t = PR("tail_fit_gpd.csv").groupby("stream").median(numeric_only=True)
chk("GPD shape, forecast channels, validation (median)", round(t.xi_val["forecast"], 2), 0.10)
chk("GPD shape, forecast channels, calibration (median)", round(t.xi_cal["forecast"], 2), 0.06)
chk("GPD shape, constraints, validation (median)", round(t.xi_val["constraint"], 2), 0.12)
chk("GPD shape, constraints, calibration (median)", round(t.xi_cal["constraint"], 2), 0.48)
chk("mean-excess ratio, forecast channels, calibration (median)", round(t.mean_excess_ratio_cal["forecast"], 2), 1.15)
# ---- selection-free calibration subset (Section 4.7)
s = PR("split_selection.csv")
chk("median closure threshold, full calibration record", round(s.tau_full.median(), 2), 7.89)
chk("median closure threshold, first half", round(s.tau_C1.median(), 2), 5.26)
chk("stored development-fault counts reproduced (candidates differing)", int((s.stored_detected != s.detected_full).sum()), 0)
top = s.sort_values("detected_C1", ascending=False).iloc[0]
chk("selected candidate with first-half thresholds: W", top.W, 60); chk("  its lambda_inv + lambda_phys", top.lambda_inv + top.lambda_phys, 0.0)
chk("  development faults detected (of 36)", top.detected_C1, 21)
p = s[(s.W == 30) & (s.lambda_inv == 1.0) & (s.lambda_phys == 0.1)].iloc[0]
chk("released configuration with first-half thresholds: faults detected", p.detected_C1, 18)
chk("  rank among the 27 candidates", int((s.detected_C1 > p.detected_C1).sum()) + 1, 9)
b = PR("split_prop5.csv"); b = b[b.block_hours == 1.0].groupby("method").mean(numeric_only=True)
for m_, rec in (("TailGuard", 11.5), ("Matched GRU", 11.5), ("Ridge AR", 9.6)):
    chk(f"second-half block level, {m_}: alarmed normal test hours (of 25)", b.alarmed_normal_blocks[m_], 1)
    chk(f"  guaranteed share", b.guaranteed[m_], 0.040); chk(f"  event recall (%)", round(100 * b.recall[m_], 1), rec)
# ---- per-attack overlap (Sections 4.3 and 4.6)
a = pd.read_csv(P("analysis", "results", "hai2305", "attacks_detection.csv")); tg, rd = a["TailGuard (seeds detecting)"] > 0, a["Ridge AR (seeds detecting)"] > 0
for nm, v, w in (("both", tg & rd, 18), ("neither", ~tg & ~rd, 26), ("ridge only", ~tg & rd, 5), ("TailGuard only", tg & ~rd, 3)): chk(f"HAI 23.05 attacks, {nm}", int(v.sum()), w)
a = pd.read_csv(P("analysis", "results", "hai2204", "attacks_detection_2204.csv")); tg, rd = a.TG > 0, a.ridge > 0
for nm, v, w in (("both", tg & rd, 17), ("ridge only", ~tg & rd, 19), ("TailGuard only", tg & ~rd, 0), ("neither", ~tg & ~rd, 22)): chk(f"HAI 22.04 attacks, {nm}", int(v.sum()), w)
ro = a[~tg & rd].regime.value_counts()
for k, w in ((2, 10), (1, 6), (0, 3)): chk(f"HAI 22.04 ridge-only attacks in regime {('low', 'mid', 'high')[k]}", int(ro.get(k, 0)), w)
chk("HAI 22.04 ridge event recall (%)", round(100 * rd.mean(), 1), 62.1)
# ---- BATADAL (Section 4.8, Table 17)
d = pd.read_csv(P("batadal", "results", "results_runs.csv")); d = d[(d.smooth == 30) & (d.score == "prespecified") & (d.rule == "closure")].groupby("method").mean(numeric_only=True)
T17 = {"ridge": (14, 1.09, 0.090, 0.879, 0.729), "tailguard": (4, 0.27, 0.025, 0.250, None), "gru": (4.6, 0.49, 0.035, None, None), "A1_nophys": (8.0, 0.30, 0.414, None, None)}
for m_, (det, fa, sh, S, F1) in T17.items():
    chk(f"BATADAL {m_}: attacks detected (of 14)", round(d.detected[m_], 1), det); chk(f"  false alarm episodes per 30 days", round(d.FA_per_30d[m_], 2), fa)
    chk(f"  share of normal hours in alarm", round(d.normal_time_in_alarm[m_], 3), sh)
    if S is not None: chk(f"  S score on the official test file", round(d.test_S[m_], 3), S)
    if F1 is not None: chk(f"  F1 on the official test file", round(d.test_F1[m_], 3), F1)
chk("BATADAL ridge: S_TTD", round(d.test_S_TTD["ridge"], 3), 0.882); chk("BATADAL ridge: S_CLF", round(d.test_S_CM["ridge"], 3), 0.876)
bl = pd.read_csv(P("batadal", "results", "results_runs.csv")); bl = bl[(bl.smooth == 30) & (bl.score == "prespecified") & (bl.rule == "block")].groupby("method").mean(numeric_only=True)
for m_, w in (("ridge", 0.135), ("tailguard", 0.034), ("gru", 0.048)): chk(f"BATADAL {m_}: share of alarmed normal days, block rule", round(bl.alarmed_normal_day_share[m_], 3), w)
print(f"\n{ok} claims reproduced, {bad} failed"); sys.exit(1 if bad else 0)
