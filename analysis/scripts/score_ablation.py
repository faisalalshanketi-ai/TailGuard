"""Score ablation (descriptive): the same forecasters' errors scored by the tail-surprise policy versus two
conventional scores, all with the same 30 s smoothing and the same closure threshold at one episode per day.
  TAILGUARD_PREFIX=hai2305|hai2204 python score_ablation.py <root> <out.csv>
conventional scores: 'mean squared error' (mean over channels of squared robust-scaled errors) and
'max z-score' (largest per-channel error divided by its validation standard deviation)."""
import os, sys, json, glob, time
import numpy as np, pandas as pd
ROOT, OUT = sys.argv[1:3]; sys.path.insert(0, os.path.join(ROOT, "scripts"))
PREFIX = os.environ["TAILGUARD_PREFIX"]
import tailguard_lib as L
from tailguard_lib import load, ecdf_ref, tail_surprise, causal_smooth, calibrate_threshold, alarm_stats, event_stats, point_stats, lagged
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
W_MAX = 120; LAGS = [1, 2, 5, 10, 30, 60]
tr, te = load("train"), load("test"); ids, idte = tr["df_ids"], te["df_ids"]
part, elig, seg, pos = ids["partition"].to_numpy(), ids["eligible"].to_numpy(), ids["segment_id"].to_numpy(), ids["pos_in_segment"].to_numpy()
mask, seg_te, pos_te = idte["eligible"].to_numpy(), idte["segment_id"].to_numpy(), idte["pos_in_segment"].to_numpy()
valm, calm, fitm = elig & (part == "val"), elig & (part == "cal"), elig & (part == "fit")
lab = pd.read_parquet(os.path.join(ROOT, "data", "derived", f"{PREFIX}_test_labels.parquet"))
with open(os.path.join(ROOT, "manifests", "unseal_log.txt"), "a") as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} test labels read by score_ablation.py (descriptive score ablation)\n")
y = lab["label"].to_numpy().astype(bool); ev = np.where(mask, lab["event_id"].to_numpy(), "")
t1 = mask & (idte["file_id"].to_numpy() == "hai-test1") if PREFIX == "hai2305" else mask.copy()
nT = tr["Z"].shape[1]

def scores(E, Ete, ref):
    sd = np.maximum(E[valm].std(0), 1e-6); out = {}
    def fin(s, p): s = s.astype(np.float32); s[p < W_MAX] = 0; return causal_smooth(s, 30)
    ts = lambda X: np.concatenate([np.sort(tail_surprise(X[k:k + 100000], ref), 1)[:, -3:].mean(1) for k in range(0, len(X), 100000)])
    out["tail surprise (TailGuard policy)"] = (fin(ts(E), pos), fin(ts(Ete), pos_te))
    out["mean squared error"] = (fin((E ** 2).mean(1), pos), fin((Ete ** 2).mean(1), pos_te))
    out["max z-score"] = (fin((E / sd).max(1), pos), fin((Ete / sd).max(1), pos_te))
    return out

def evaluate(name, seed, S):
    R = []
    for sc, (st, ss) in S.items():
        tau = calibrate_threshold(st[calm], seg[calm], calm.sum() / 3600)[0]; a = (ss > tau) & mask
        es = event_stats(a[mask], ev[mask]); fa = alarm_stats(a[mask], seg_te[mask], ~y[mask])
        s5 = np.where(np.isfinite(ss), ss, -1e9); tt = np.percentile(s5[t1 & ~y], 95); ps = point_stats(s5[t1] > tt, y[t1], s5[t1])
        R.append({"dataset": PREFIX, "detector": name, "seed": seed, "score": sc, "event_recall": es["event_recall"], "FA24": fa["FA24"],
                  "F1_5pctFPR": ps["F1"], "AP": ps.get("AP", np.nan)})
        print(f"  {name:8s} s{seed:<4} {sc:34s} recall {es['event_recall']:.3f} FA24 {fa['FA24']:.2f} F1@5%FPR {ps['F1']:.3f}", flush=True)
    return R

rows = []
# classical forecasters, refitted exactly as in step 3 (ridge alpha = 1, PCA k = 20)
X = np.hstack([tr["Z"], tr["C"], tr["D"]]); Xte = np.hstack([te["Z"], te["C"], te["D"]]); Z, Zte = tr["Z"], te["Z"]
def ridge_pred(m, XX, chunk=200000):
    return np.vstack([m.predict(np.nan_to_num(lagged(XX[max(0, k - 60):k + chunk], LAGS)[(k - max(0, k - 60)):], copy=False)) for k in range(0, len(XX), chunk)]).astype(np.float32)
idx = np.where(fitm)[0][::4]
F = np.nan_to_num(lagged(X, LAGS)[idx], copy=False); m = Ridge(alpha=1.0).fit(F, Z[idx]); del F
E = np.abs(Z - ridge_pred(m, X)); Ete = np.abs(Zte - ridge_pred(m, Xte))
rows += evaluate("ridge", 17, scores(E, Ete, ecdf_ref(E[valm]))); del E, Ete
p = PCA(20, random_state=17).fit(Z[np.where(fitm)[0][::2]])
E = np.abs(Z - p.inverse_transform(p.transform(Z))).astype(np.float32); Ete = np.abs(Zte - p.inverse_transform(p.transform(Zte))).astype(np.float32)
rows += evaluate("PCA", 17, scores(E, Ete, ecdf_ref(E[valm]))); del E, Ete, X, Xte
# neural reference detectors from their saved per-channel errors
for f in sorted(glob.glob(os.path.join(ROOT, "runs", "deep_rescore", "*", "s*", "chan.npz"))):
    name = os.path.basename(os.path.dirname(os.path.dirname(f))); sd_ = int(os.path.basename(os.path.dirname(f))[1:])
    c = np.load(f); E = np.zeros((len(Z), nT), np.float32); E[c["rows"]] = c["E"]
    rows += evaluate(name, sd_, scores(E, c["Ete"], ecdf_ref(E[valm]))); del E
pd.DataFrame(rows).to_csv(OUT, index=False); print("written", OUT)
