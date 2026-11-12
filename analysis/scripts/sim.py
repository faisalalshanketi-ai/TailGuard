"""Simulation: serial dependence vs the exceedance statements of Propositions 1, 3, 4 and 5.
Residuals |x| with x an AR(1) Gaussian process per channel (lag-1 correlation rho); same record lengths as HAI 23.05
(validation reference 49.8 h subsampled every 5th sample, calibration 49.8 h, fresh normal test 96 h); scoring with the
released functions (tail_surprise, causal 30 s mean, top-3 and max)."""
import sys, os, numpy as np, pandas as pd
from scipy.signal import lfilter
sys.path.insert(0, sys.argv[1]); from tailguard_lib import ecdf_ref, tail_surprise, causal_smooth, calibrate_threshold, episodes
OUT = sys.argv[2]
m, Lr, Lc, Lt, w = 66, 179280, 179280, 345600, 30
tauA = np.log10(np.e * 86400 * m)   # Proposition 4: log10(e*S*m/A), A = 1
def ar1(n, rho, rng):
    e = rng.standard_normal((n + 2000, m)).astype(np.float32) * np.sqrt(1 - rho ** 2)
    x = lfilter([1.0], [1.0, -rho], e, axis=0)[2000:]; return np.abs(x).astype(np.float32)
rows = []
for rho in (0.0, 0.5, 0.9, 0.99, 0.999):
    for rep in range(3):
        rng = np.random.default_rng(1000 * rep + int(rho * 1000))
        ref = ecdf_ref(ar1(Lr, rho, rng), step=5)
        res = {}
        for part, L in (("cal", Lc), ("test", Lt)):
            q = tail_surprise(ar1(L, rho, rng), ref)
            res[part] = dict(q=q, top3=causal_smooth(np.sort(q, 1)[:, -3:].mean(1), w), mx=causal_smooth(q.max(1), w), qmax=q.max(1))
        T = res["test"]; r = {"rho": rho, "rep": rep}
        r["single_c2"] = float((T["q"] >= 2).mean()); r["single_c3"] = float((T["q"] >= 3).mean())
        r["max_c2"] = float((T["qmax"] >= 2 + np.log10(m)).mean()); r["max_c3"] = float((T["qmax"] >= 3 + np.log10(m)).mean())
        seg_c = np.zeros(Lc, int); seg_t = np.zeros(Lt, int)
        for sc in ("top3", "mx"):
            tc, _ = calibrate_threshold(res["cal"][sc], seg_c, Lc / 3600)
            r[f"{sc}_tau_closure"] = tc
            for nm, tau in (("closure", tc), ("tauA", tauA)):
                r[f"{sc}_FA24_{nm}"] = float(episodes(T[sc] > tau, seg_t).sum() * 24 / (Lt / 3600))
            cb = res["cal"][sc][: (Lc // 3600) * 3600].reshape(-1, 3600).max(1); B = len(cb); k = max(1, int((B + 1) / 24))
            tb = np.sort(cb)[::-1][k - 1]
            tbm = T[sc][: (Lt // 3600) * 3600].reshape(-1, 3600).max(1)
            r[f"{sc}_block_nominal"] = k / (B + 1); r[f"{sc}_block_realised"] = float((tbm > tb).mean())
            r[f"{sc}_FA24_block"] = float(episodes(T[sc] > tb, seg_t).sum() * 24 / (Lt / 3600))
        rows.append(r); print(rho, rep, {k: round(v, 4) for k, v in r.items() if k not in ("rho", "rep")}, flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
print(df.groupby("rho").mean().drop(columns="rep").round(4).T.to_string()); print("tauA", round(tauA, 3))
