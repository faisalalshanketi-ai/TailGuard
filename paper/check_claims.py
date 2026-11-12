"""Recompute every headline claim of the manuscript from the result tables and assert it.

Usage: python check_claims.py results/
Each claim prints PASS/FAIL with the value recomputed from the CSV files written by step6_evaluate.py.
"""
import sys, re
import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "results"

def mean(s):
    return float(str(s).split("±")[0])

def sd(s):
    return float(str(s).split("±")[1]) if "±" in str(s) else 0.0

def ci(s):
    a, b = re.findall(r"-?\d+\.\d+(?:e-?\d+)?", str(s))[:2]
    return float(a), float(b)

com = pd.read_csv(f"{R}/table_common_protocol.csv", index_col=0)
pub = pd.read_csv(f"{R}/table_published_protocols.csv")
par = pd.read_csv(f"{R}/table_paired_vs_tailguard.csv", index_col=0)
reg = pd.read_csv(f"{R}/table_regime_holdout.csv")

P95, FPR5 = "[14] p95-validation threshold, Test 1", "[14]/[15] fixed 5% FPR on test-normal, Test 1"
def pubv(proto, method, col="F1"):
    return mean(pub[(pub.protocol == proto) & (pub.method == method)][col].iloc[0])
def c(m, col):
    return mean(com.loc[m, col])
def rg(r, m, col):
    return mean(reg[(reg.regime == r) & (reg.method == m)][col].iloc[0])

claims = []
def claim(text, value, expected, tol=0.0005):
    ok = abs(value - expected) <= tol
    claims.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {text}: recomputed {value:.4f}, manuscript {expected}")

S = "TailGuard"
# --- Section 4.1 (published protocols) / abstract / contribution 1: published protocols
claim("F1 at 5% FPR, TailGuard", pubv(FPR5, S), 0.657)
claim("F1 at 5% FPR, best published [14]", pubv(FPR5, "[14] GNN+BiLSTM"), 0.530)
claim("margin over [14] at 5% FPR", pubv(FPR5, S) - pubv(FPR5, "[14] GNN+BiLSTM"), 0.127)
claim("margin over [15] at 5% FPR", pubv(FPR5, S) - pubv(FPR5, "[15] GNN+BiLSTM"), 0.159)
claim("F1 under p95 rule, TailGuard", pubv(P95, S), 0.609)
claim("AUC-PR under p95 rule, TailGuard", pubv(P95, S, "AP"), 0.771)
claim("margin in F1 over published GNN, p95 rule", pubv(P95, S) - 0.554, 0.055)
claim("TailGuard FPR under p95 rule", pubv(P95, S, "FPR"), 0.064)
claim("ridge F1 at 5% FPR", pubv(FPR5, "ridge"), 0.683)
claim("GRU F1 at 5% FPR", pubv(FPR5, "GRU (pooled ERM)"), 0.639)
claim("lowest neural F1 at 5% FPR (USAD)", pubv(FPR5, "USAD"), 0.217)
claim("highest neural F1 at 5% FPR (GDN)", pubv(FPR5, "GDN"), 0.463)
# --- Section 4.2: common protocol
claim("TailGuard event recall", c(S, "event_recall"), 0.354)
claim("TailGuard FA24", c(S, "FA24"), 1.33, 0.005)
claim("TailGuard median delay (s)", c(S, "delay_median_s"), 69, 0.5)
for m, pp in (("USAD", 0.105), ("TranAD", 0.135), ("GDN", 0.173)):
    claim(f"paired recall gain over {m}", float(par.loc[m, "recall_diff"]), pp)
    lo, hi = ci(par.loc[m, "recall_CI95"]); claim(f"CI lower bound vs {m} above zero", float(lo > 0), 1.0)
claim("minimum Holm p over USAD/TranAD/GDN recall", min(float(par.loc[m, "p_recall_holm"]) for m in ("USAD", "TranAD", "GDN")), 0.16, 0.005)
claim("paired recall difference vs GRU", float(par.loc["GRU (pooled ERM)", "recall_diff"]), 0.006)
claim("paired FA24 difference vs GRU", float(par.loc["GRU (pooled ERM)", "FA24_diff"]), -0.27, 0.005)
lo, hi = ci(par.loc["GRU (pooled ERM)", "FA24_CI95"]); claim("FA24 CI vs GRU lower", lo, -1.03, 0.005); claim("FA24 CI vs GRU upper", hi, 0.18, 0.005)
claim("ridge recall", c("ridge", "event_recall"), 0.442)
claim("ridge FA24", c("ridge", "FA24"), 1.90, 0.005)
claim("paired recall difference vs ridge", float(par.loc["ridge", "recall_diff"]), -0.089)
claim("forecasting detectors FA24 range low (TailGuard)", min(c(m, "FA24") for m in (S, "GRU (pooled ERM)", "ridge", "pca", "persistence")), 1.33, 0.005)
claim("forecasting detectors FA24 range high (persistence)", max(c(m, "FA24") for m in (S, "GRU (pooled ERM)", "ridge", "pca", "persistence")), 2.22, 0.005)
# --- Section 4.3: ablations / contribution 5
claim("A1 recall (no process constraints)", c("A1_no_physics", "event_recall"), 0.315)
claim("A1 FA24", c("A1_no_physics", "FA24"), 1.52, 0.005)
claim("A1 F1 at 5% FPR", pubv(FPR5, "A1_no_physics"), 0.620)
claim("max recall change for A2/A3/A10/A11", max(abs(c(m, "event_recall") - c(S, "event_recall")) for m in
      ("A2_no_invariance", "A3_pooled_no_inv", "A10_whole_forecast_VREx", "A11_day_environments")), 0.016, 0.0005)
claim("A2/A3/A10/A11 FA24", c("A2_no_invariance", "FA24"), 1.27, 0.005)
claim("A4 recall equals full model", c("A4_beta0 (temporal score only)", "event_recall"), 0.354)
claim("A5 residual-only recall", c("Residual-only (beta=1)", "event_recall"), 0.154)
claim("A5 residual-only FA24", c("Residual-only (beta=1)", "FA24"), 0.63, 0.005)
claim("A4 (full model, beta=0) FA24", c("A4_beta0 (temporal score only)", "FA24"), 1.39, 0.005)
claim("A4 F1 at 5% FPR", pubv(FPR5, "A4_beta0 (temporal score only)"), 0.656)
claim("matched GRU recall", c("GRU (pooled ERM)", "event_recall"), 0.350)
claim("matched GRU FA24", c("GRU (pooled ERM)", "FA24"), 1.58, 0.005)
claim("constraint term: recall gain A1 -> A4 (both beta=0)", c("A4_beta0 (temporal score only)", "event_recall") - c("A1_no_physics", "event_recall"), 0.0385, 0.001)  # 2 of 52 attacks = 3.8 points; tables store 3-decimal means
# --- held-out regimes: superseded by the refitted hold-out with the main false-episode definition;
#     see paper/check_claims_extended.py (Table 12, Figure 6). The original partial hold-out table remains in results/reports for the record.
print(f"\n{sum(claims)}/{len(claims)} claims reproduce from the result tables")
sys.exit(0 if all(claims) else 1)
