"""Proposition 5 evaluation: block-exchangeable (hourly block-maximum) threshold vs monotone-closure threshold.
python prop5_eval.py <root> <prefix> <runs_dir: runs|runs_max> <out.csv> [block_hours]
Reads frozen scores and the test labels already unsealed for the published analyses (no model or score is changed)."""
import sys, os, glob, json
import numpy as np, pandas as pd
root, prefix, rdir, out = sys.argv[1:5]; H = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
sys.path.insert(0, os.path.join(root, "scripts")); os.environ["TAILGUARD_PREFIX"] = prefix
from tailguard_lib import calibrate_threshold, episodes, event_stats, alarm_stats

der = os.path.join(root, "data", "derived")
tr = pd.read_parquet(os.path.join(der, f"{prefix}_train.parquet"), columns=["partition", "eligible", "segment_id"])
te = pd.read_parquet(os.path.join(der, f"{prefix}_test.parquet"), columns=["timestamp", "file_id", "segment_id", "eligible"])
lab = pd.read_parquet(os.path.join(der, f"{prefix}_test_labels.parquet"))
calm = (tr.eligible & (tr.partition == "cal")).to_numpy(); cseg = tr.segment_id.to_numpy()[calm]
cal_hours = calm.sum() / 3600.0
mask = te.eligible.to_numpy(); y = lab.label.to_numpy().astype(bool); ev = np.where(mask, lab.event_id.to_numpy(), "")
tseg = te.segment_id.to_numpy(); L = int(round(H * 3600))

def blocks(seg, n):
    """Consecutive blocks of L rows within each segment (partial tail dropped); returns list of index arrays."""
    out = []
    for s in pd.unique(seg):
        idx = np.where(seg == s)[0]
        for j in range(len(idx) // L): out.append(idx[j * L:(j + 1) * L])
    return out

CB = blocks(cseg, calm.sum()); B = len(CB)
k = max(1, int(np.floor((B + 1) * H / 24.0)))          # budget A = 1 alarmed block per 24 h
el = np.where(mask)[0]; TB = [el[b] for b in blocks(tseg[mask], mask.sum())]
TBn = [b for b in TB if not y[b].any()]                 # normal-only test blocks

def evaluate(s_test, s_cal, tau_saved):
    tau_c, _ = calibrate_threshold(s_cal, cseg, cal_hours)
    bmax = np.sort([s_cal[b].max() for b in CB])[::-1]; tau_b = float(bmax[k - 1])
    rows = {}
    for name, tau in (("closure", tau_c), ("block", tau_b)):
        a = (s_test > tau) & mask
        r = event_stats(a[mask], ev[mask]); r.update(alarm_stats(a[mask], tseg[mask], ~y[mask]))
        r["tau"] = tau; r["alarmed_normal_block_frac"] = float(np.mean([a[b].any() for b in TBn]))
        r["cal_alarmed_block_frac"] = float(np.mean([(s_cal[b] > tau).any() for b in CB]))
        rows[name] = r
    rows["closure"]["tau_saved"] = tau_saved
    return rows

def fuse(p, v, beta, mx):
    return np.maximum(p, v) if (mx and beta == 0.5) else (1 - beta) * p + beta * v

res = []; mx = rdir == "runs_max"; T = os.path.join(root, rdir, "train")
sel = json.load(open(os.path.join(T, "selected.json")))
core = lambda tag, s, li, lp, bal="bal": os.path.join(T, f"{tag}__W{sel['W']}_inv{li}_phys{lp}_phys_regime_{bal}_s{s}")
for s in (17, 29, 43, 71, 101):
    for meth, d, beta in (("TailGuard", core("search" if s == 17 else "tailguard", s, sel["l_inv"], sel["l_phys"]), 0.5),
                          ("Matched GRU", core("gru_erm", s, 0.0, 0.0, "pool"), 0.0)):
        if not os.path.exists(os.path.join(d, "scores.npz")): continue
        z = np.load(os.path.join(d, "scores.npz")); sm = json.load(open(os.path.join(d, "summary.json")))
        st = fuse(z["p_test"], z["v_test"], beta, mx); sc = fuse(z["p_train"], z["v_train"], beta, mx)[calm]
        st = np.where(np.isfinite(st), st, -np.inf)
        for rule, r in evaluate(st, sc, sm["tau"][f"b{beta}"]).items(): res.append({"method": meth, "seed": s, "rule": rule, **r})
bd = os.path.join(root, rdir, "baselines", "ridge_a1.0")
if os.path.exists(bd):
    sm = json.load(open(os.path.join(bd, "summary.json"))); sct = pd.read_parquet(os.path.join(bd, "test_scores_SEALED.parquet"))
    st = np.full(len(te), -np.inf); key = te.loc[te.eligible, ["file_id", "timestamp"]].reset_index()
    mm = key.merge(sct[["file_id", "timestamp", "score"]], on=["file_id", "timestamp"], how="left")
    st[mm["index"].to_numpy()] = mm["score"].to_numpy()
    sc = np.load(os.path.join(bd, "train_scores.npz"))["s_train"][calm]
    for rule, r in evaluate(st, sc, sm["tau"]).items(): res.append({"method": "Ridge AR", "seed": 17, "rule": rule, **r})
df = pd.DataFrame(res).drop(columns=["delay_iqr_s"], errors="ignore")
df["B"] = B; df["k"] = k; df["nominal_block_rate"] = k / (B + 1); df["n_normal_test_blocks"] = len(TBn); df["block_hours"] = H
df.to_csv(out, index=False)
print(f"B={B} k={k} nominal={k/(B+1):.4f} normal test blocks={len(TBn)} cal_hours={cal_hours:.1f}")
print(df.groupby(["method", "rule"])[["event_recall", "FA24", "alarmed_normal_block_frac", "cal_alarmed_block_frac", "tau"]].mean().round(3))
cl = df[df.rule == "closure"]; print("max |tau_recomputed - tau_saved|:", float(np.nanmax(np.abs(cl.tau - cl.tau_saved))))
