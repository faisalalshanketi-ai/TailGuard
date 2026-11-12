"""Score-level variants evaluated on development injections only (validation partition, synthetic faults).
Threshold of every variant is calibrated on calibration data at 1 false-alarm episode per 24 h."""
import sys, os, json, numpy as np
ROOT = sys.argv[1]; F = np.load(sys.argv[2], allow_pickle=True)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tailguard_lib import ecdf_ref, tail_surprise, causal_smooth, calibrate_threshold, alarm_stats, event_stats
W_MAX = 120
part, reg, elig, seg, pos, evid = F["part"], F["reg"], F["elig"], F["seg"], F["pos"], F["evid"]
valm, calm, devm = elig & (part == "val"), elig & (part == "cal"), elig & (part == "val")
print("hours per regime  val:", {g: round((valm & (reg == g)).sum() / 3600, 1) for g in np.unique(reg)},
      " cal:", {g: round((calm & (reg == g)).sum() / 3600, 1) for g in np.unique(reg)})

def tsurp(X, Xi, by_regime):
    if not by_regime:
        ref = ecdf_ref(X[valm]); return tail_surprise(X, ref), tail_surprise(Xi, ref)
    q, qi = np.zeros_like(X), np.zeros_like(Xi); pooled = ecdf_ref(X[valm])
    for g in np.unique(reg):
        mk = reg == g; vm = valm & mk
        ref = ecdf_ref(X[vm]) if vm.sum() >= 3600 * 3 else pooled
        q[mk] = tail_surprise(X[mk], ref); qi[mk] = tail_surprise(Xi[mk], ref)
    return q, qi

def agg(q, k, w):
    s = np.sort(q, 1)[:, -min(k, q.shape[1]):].mean(1); s[pos < W_MAX] = 0; return causal_smooth(s, w)

def evaluate(s, si, per_regime_tau=False):
    if not per_regime_tau:
        tau, _ = calibrate_threshold(s[calm], seg[calm], calm.sum() / 3600); thr = np.full(len(s), tau)
    else:
        tau0, _ = calibrate_threshold(s[calm], seg[calm], calm.sum() / 3600); thr = np.full(len(s), tau0)
        for g in np.unique(reg):
            cm = calm & (reg == g)
            if cm.sum() >= 3600 * 3:
                t, _ = calibrate_threshold(s[cm], seg[cm], cm.sum() / 3600); thr[reg == g] = t
    a = si[devm] > thr[devm]; ev = evid[devm]
    st = event_stats(a, ev); fa = alarm_stats(a, seg[devm], ev == "")["FA24"]
    vclean = alarm_stats(s[valm] > thr[valm], seg[valm], np.ones(valm.sum(), bool))["FA24"]
    return st["detected"], st["event_recall"], fa, vclean

rows = []
cache = {}
for byreg in (False, True):
    qE, qEi = tsurp(F["E"], F["Ei"], byreg); qV, qVi = tsurp(F["V"], F["Vi"], byreg)
    for k in (1, 3, 5):
        for w in (10, 30, 60):
            p, pi = agg(qE, k, w), agg(qEi, k, w); v, vi = agg(qV, k, w), agg(qVi, k, w)
            for fuse in ("mean0.0", "mean0.25", "mean0.5", "mean0.75", "max"):
                if fuse == "max": s, si = np.maximum(p, v), np.maximum(pi, vi)
                else:
                    b = float(fuse[4:]); s, si = (1 - b) * p + b * v, (1 - b) * pi + b * vi
                for prt in (False, True):
                    d, r, fa, vc = evaluate(s, si, prt)
                    rows.append(dict(ecdf="regime" if byreg else "pooled", k=k, w=w, fuse=fuse, tau="regime" if prt else "pooled",
                                     dev_detected=d, dev_recall=round(r, 3), dev_FA24=round(fa, 2), val_FA24=round(vc, 2)))
import pandas as pd
df = pd.DataFrame(rows); df.to_csv(sys.argv[3], index=False)
base = df[(df.ecdf == "pooled") & (df.k == 3) & (df.w == 30) & (df.fuse == "mean0.5") & (df.tau == "pooled")]
print("baseline (paper's scoring):\n", base.to_string(index=False))
print(df.sort_values(["dev_detected", "val_FA24"], ascending=[False, True]).head(25).to_string(index=False))
