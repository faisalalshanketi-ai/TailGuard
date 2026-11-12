"""After Steps 2-3: re-evaluate the classical baselines with the paper's evaluation code and compare with Table 6 (needs test labels)."""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import step6_evaluate as E

te = pd.read_parquet(os.path.join(ROOT, "data/derived/hai2305_test.parquet"), columns=["timestamp", "file_id", "segment_id", "pos_in_segment", "eligible", "regime"])
tr = pd.read_parquet(os.path.join(ROOT, "data/derived/hai2305_train.parquet"), columns=["partition", "eligible", "regime"])
lab = pd.read_parquet(os.path.join(ROOT, "data/derived/hai2305_test_labels.parquet"))
M, _ = E.collect(ROOT, te, tr)
mask = te["eligible"].to_numpy()
pub = pd.read_csv(os.path.join(ROOT, "results/reports/table_common_protocol.csv"), index_col=0)
keys = ["event_recall", "FA24", "F1", "AP", "AUROC", "delay_median_s"]
worst = 0.0
for m in ("persistence", "ridge", "pca", "iforest"):
    r = E.common_metrics(M[m][0]["s_test"], M[m][0]["tau"], lab, te, mask)
    row = []
    for k in keys:
        ref = float(str(pub.loc[m, k]).split("±")[0]); val = r[k]
        worst = max(worst, abs(val - ref) / max(abs(ref), 1e-9)); row.append(f"{k} {val:.3f}/{ref:.3f}")
    print(f"{m:12s}", "  ".join(row))
print(f"largest relative difference to the published table: {worst:.2e} (the table is rounded to 3 decimals)")
