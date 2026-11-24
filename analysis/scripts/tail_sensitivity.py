"""Sensitivity of results to the tail fit of the surprise score (tail quantile, mean-excess floor,
no exponential tail). Frozen TailGuard models (5 seeds); descriptive test evaluation, logged as a label reading."""
import sys, os, glob, numpy as np, pandas as pd
ROOT, ERR, OUT = sys.argv[1:4]; sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tailguard_lib import ecdf_ref, causal_smooth, calibrate_threshold, alarm_stats, event_stats, point_stats
te = pd.read_parquet(os.path.join(ROOT, "data/derived/hai2305_test.parquet"), columns=["file_id", "segment_id", "eligible"])
lab = pd.read_parquet(os.path.join(ROOT, "data/derived/hai2305_test_labels.parquet"))
with open(os.path.join(ROOT, "manifests", "unseal_log.txt"), "a") as f: f.write("2026-10-22 test labels read by tail_sensitivity.py (descriptive tail-fit sensitivity)\n")
mask = te.eligible.to_numpy(); y = lab.label.to_numpy().astype(bool); ev = np.where(mask, lab.event_id.to_numpy(), ""); segt = te.segment_id.to_numpy()
t1 = mask & (te.file_id.to_numpy() == "hai-test1")
def surprise(E, ref, q, floor, tail=True):
    n = ref.shape[0]; out = np.empty(E.shape, np.float32); k = int(np.floor(q * n))
    for j in range(E.shape[1]):
        col = ref[:, j]; Q = col[k]; me = max(float(np.mean(col[k:] - Q)), floor)
        r = np.searchsorted(col, E[:, j], side="right"); body = -np.log10(np.maximum(1 - r / (n + 1), 1 / (n + 1)))
        if tail:
            tl = -np.log10(1 - q) + (E[:, j] - Q) / (me * np.log(10)); out[:, j] = np.where(E[:, j] > Q, np.maximum(tl, -np.log10(1 - q)), body)
        else: out[:, j] = body
    return out
def agg(T, pos, k):
    s = np.sort(T, 1)[:, -k:].mean(1); s[pos < 120] = 0; return causal_smooth(s, 30)
rows = []
for f in sorted(glob.glob(os.path.join(ERR, "*.npz"))):
    Z = np.load(f, allow_pickle=True); part, elig, seg, pos, post = Z["part"], Z["elig"], Z["seg"], Z["pos"], Z["pos_te"]
    valm, calm = elig & (part == "val"), elig & (part == "cal"); rE, rV = ecdf_ref(Z["E"][valm]), ecdf_ref(Z["V"][valm])
    for name, q, fl, tail in (("paper: q=0.99, floor 1e-3", .99, 1e-3, True), ("q=0.98", .98, 1e-3, True), ("q=0.995", .995, 1e-3, True),
                              ("floor 1e-2", .99, 1e-2, True), ("no exponential tail (ECDF only)", .99, 1e-3, False)):
        qE, qEt = surprise(Z["E"], rE, q, fl, tail), surprise(Z["Ete"], rE, q, fl, tail)
        qV, qVt = surprise(Z["V"], rV, q, fl, tail), surprise(Z["Vte"], rV, q, fl, tail)
        for score, k, fuse in (("prespecified (top-3, beta 0.5)", 3, lambda a, b: 0.5 * a + 0.5 * b), ("max tail surprise", 1, np.maximum)):
            s = fuse(agg(qE, pos, k), agg(qV, pos, k)); st = fuse(agg(qEt, post, k), agg(qVt, post, k))
            tau, _ = calibrate_threshold(s[calm], seg[calm], calm.sum() / 3600); a = (st > tau) & mask
            sc = np.where(np.isfinite(st), st, -1e9); tt = np.percentile(sc[t1 & ~y], 95)
            rows.append({"model": os.path.basename(f)[:-4], "tail_fit": name, "score": score,
                         "event_recall": event_stats(a[mask], ev[mask])["event_recall"], "FA24": alarm_stats(a[mask], segt[mask], ~y[mask])["FA24"],
                         "F1_5pctFPR_test1": point_stats(sc[t1] > tt, y[t1], sc[t1])["F1"]})
    print(f, flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "tail_sensitivity_runs.csv"), index=False)
g = df.groupby(["score", "tail_fit"])[["event_recall", "FA24", "F1_5pctFPR_test1"]].agg(["mean", "std"]).round(3)
g.to_csv(os.path.join(OUT, "tail_sensitivity.csv")); print(g.to_string())
