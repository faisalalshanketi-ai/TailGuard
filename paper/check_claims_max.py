"""Recompute the claims of Section 4.5 / Table 10 (max tail-surprise score) from results/reports_max."""
import os, sys, json
import pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "reports_max")
ms = lambda c: float(str(c).split("±")[0])
C = pd.read_csv(os.path.join(R, "table_common_protocol.csv"), index_col=0)
P = pd.read_csv(os.path.join(R, "table_published_protocols.csv"))
D = pd.read_csv(os.path.join(R, "table_paired_vs_tailguard.csv"), index_col=0)
H = pd.read_csv(os.path.join(R, "table_regime_holdout.csv"))
pub = lambda proto, m, col="F1": ms(P[(P.protocol.str.startswith(proto)) & (P.method == m)][col].iloc[0])
ci = lambda m: json.loads(D.loc[m, "recall_CI95"])
claims = [
    ("F1 at 5% FPR, TailGuard 0.670", round(pub("[14]/[15] fixed", "TailGuard"), 3) == 0.670),
    ("F1 at p95 threshold 0.691", round(pub("[14] p95", "TailGuard"), 3) == 0.691),
    ("AUC-PR at p95 threshold 0.795", round(pub("[14] p95", "TailGuard", "AP"), 3) == 0.795),
    ("test FPR at p95 threshold 0.045", round(pub("[14] p95", "TailGuard", "FPR"), 3) == 0.045),
    ("margin over published F1 at 5% FPR 0.140", round(pub("[14]/[15] fixed", "TailGuard") - 0.530, 3) == 0.140),
    ("margin over published F1 at p95 0.137", round(pub("[14] p95", "TailGuard") - 0.554, 3) == 0.137),
    ("event recall 40.8%", round(ms(C.loc["TailGuard", "event_recall"]), 3) == 0.408),
    ("false alarms 1.52/day", round(ms(C.loc["TailGuard", "FA24"]), 2) == 1.52),
    ("median delay 65 s", round(ms(C.loc["TailGuard", "delay_median_s"])) == 65),
    ("A1 recall 34.6% at 1.52/day", round(ms(C.loc["A1_no_physics", "event_recall"]), 3) == 0.346 and round(ms(C.loc["A1_no_physics", "FA24"]), 2) == 1.52),
    ("A4 recall 38.5%", round(ms(C.loc["A4_beta0 (temporal score only)", "event_recall"]), 3) == 0.385),
    ("A2/A3 recall 43.1%/43.8%", round(ms(C.loc["A2_no_invariance", "event_recall"]), 3) == 0.431 and round(ms(C.loc["A3_pooled_no_inv", "event_recall"]), 3) == 0.438),
    ("F1 at 5% FPR A4 0.668, A1 0.631", round(pub("[14]/[15] fixed", "A4_beta0 (temporal score only)"), 3) == 0.668 and round(pub("[14]/[15] fixed", "A1_no_physics"), 3) == 0.631),
    ("gain vs USAD/TranAD/GDN 20.8/15.4/20.4 points", [round(100 * D.loc[m, "recall_diff"], 1) for m in ("USAD", "TranAD", "GDN")] == [20.8, 15.4, 20.4]),
    ("deep-baseline CIs exclude zero (5.2 to 32.9)", min(ci(m)[0] for m in ("USAD", "TranAD", "GDN")) > 0 and round(100 * min(ci(m)[0] for m in ("USAD", "TranAD", "GDN")), 1) == 5.2 and round(100 * max(ci(m)[1] for m in ("USAD", "TranAD", "GDN")), 1) == 32.9),
    ("sign-flip p 0.008-0.031, Holm 0.08-0.22", round(min(D.loc[m, "p_recall"] for m in ("USAD", "TranAD", "GDN")), 3) == 0.008 and round(max(D.loc[m, "p_recall"] for m in ("USAD", "TranAD", "GDN")), 3) == 0.031
        and round(min(D.loc[m, "p_recall_holm"] for m in ("USAD", "TranAD", "GDN")), 2) == 0.08 and round(max(D.loc[m, "p_recall_holm"] for m in ("USAD", "TranAD", "GDN")), 2) == 0.22),
    ("vs matched GRU +5.5 points (0.7, 11.9), 0.08 fewer alarms", round(100 * D.loc["GRU (pooled ERM)", "recall_diff"], 1) == 5.5 and [round(100 * x, 1) for x in ci("GRU (pooled ERM)")] == [0.7, 11.9]
        and round(-D.loc["GRU (pooled ERM)", "FA24_diff"], 2) == 0.08),
    ("ridge 44.2% at 1.90/day, diff -3.5 (-13.7, 4.9)", round(ms(C.loc["ridge", "event_recall"]), 3) == 0.442 and round(100 * D.loc["ridge", "recall_diff"], 1) == -3.5 and [round(100 * x, 1) for x in ci("ridge")] == [-13.7, 4.9]),
]
bad = [n for n, ok in claims if not ok]
for n, ok in claims: print(("PASS " if ok else "FAIL ") + n)
print(f"{len(claims) - len(bad)}/{len(claims)} Section 4.5 claims reproduce from {R}"); sys.exit(1 if bad else 0)
