"""TailGuard | extension of analysis/scripts/sim.py (Appendix B, Table B2) beyond stationary Gaussian AR(1).
Same record lengths, channels, scoring functions, closure rule and block rule as sim.py; only the innovation law changes.
  python sim_ext.py <scripts_dir> <out.csv> <family> <rep_start> <rep_end> [rho ...]
families: gauss (as sim.py), t3 (Student-t, 3 d.f., heavy-tailed innovations), garch (GARCH(1,1) innovations,
alpha 0.10, beta 0.85: time-varying volatility), garch_t (GARCH(1,1) with Student-t 5 d.f. innovations).
All innovations are scaled to unit variance; the AR(1) recursion and the absolute value are as in sim.py."""
import sys, numpy as np, pandas as pd
from scipy.signal import lfilter
sys.path.insert(0, sys.argv[1]); from tailguard_lib import ecdf_ref, tail_surprise, causal_smooth, calibrate_threshold, episodes
OUT, FAM, R0, R1 = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
RHOS = [float(x) for x in sys.argv[6:]] or [0.0, 0.5, 0.9, 0.99, 0.999]
m, Lr, Lc, Lt, w = 66, 179280, 179280, 345600, 30
tauA = np.log10(np.e * 86400 * m)
GA, GB = 0.10, 0.85
def innov(n, rng):
    if FAM == "gauss": return rng.standard_normal((n, m))
    if FAM == "t3": return rng.standard_t(3, (n, m)) / np.sqrt(3.0)
    z = rng.standard_normal((n, m)) if FAM == "garch" else rng.standard_t(5, (n, m)) / np.sqrt(5.0 / 3.0)
    e = np.empty_like(z); s2 = np.ones(m); om = 1.0 - GA - GB
    for t in range(n):
        e[t] = np.sqrt(s2) * z[t]; s2 = om + GA * e[t] ** 2 + GB * s2
    return e
def ar1(n, rho, rng):
    e = innov(n + 2000, rng) * np.sqrt(1 - rho ** 2)
    x = lfilter([1.0], [1.0, -rho], e, axis=0)[2000:]; return np.abs(x).astype(np.float32)
FOFF = {"gauss": 0, "t3": 100000, "garch": 200000, "garch_t": 300000}[FAM]
rows = []
for rho in RHOS:
    for rep in range(R0, R1):
        rng = np.random.default_rng(FOFF + 1000 * rep + int(rho * 1000))
        raw = ar1(Lr, rho, rng); ref = ecdf_ref(raw, step=5)
        # tail-fit diagnostic of the reference: KS distance of the excesses over Q to the fitted exponential (median over channels)
        n = ref.shape[0]; k = int(np.floor(0.99 * n)); ks = []
        for j in range(m):
            ex = ref[k + 1:, j] - ref[k, j]; me = max(float(np.mean(ref[k:, j] - ref[k, j])), 1e-3)
            F = 1 - np.exp(-np.sort(ex) / me); i = np.arange(1, len(ex) + 1)
            ks.append(max(np.max(i / len(ex) - F), np.max(F - (i - 1) / len(ex))))
        del raw; res = {}
        for part, L in (("cal", Lc), ("test", Lt)):
            q = tail_surprise(ar1(L, rho, rng), ref)
            res[part] = dict(q=q, top3=causal_smooth(np.sort(q, 1)[:, -3:].mean(1), w), mx=causal_smooth(q.max(1), w), qmax=q.max(1))
        T = res["test"]; r = {"family": FAM, "rho": rho, "rep": rep, "ks_median": float(np.median(ks))}
        r["single_c2"] = float((T["q"] >= 2).mean()); r["single_c3"] = float((T["q"] >= 3).mean()); r["single_c4"] = float((T["q"] >= 4).mean())
        r["max_c2"] = float((T["qmax"] >= 2 + np.log10(m)).mean()); r["max_c3"] = float((T["qmax"] >= 3 + np.log10(m)).mean())
        seg_c = np.zeros(Lc, int); seg_t = np.zeros(Lt, int)
        for sc in ("top3", "mx"):
            tc, _ = calibrate_threshold(res["cal"][sc], seg_c, Lc / 3600)
            r[f"{sc}_tau_closure"] = tc
            for nm, tau in (("closure", tc), ("tauA", tauA)):
                r[f"{sc}_FA24_{nm}"] = float(episodes(T[sc] > tau, seg_t).sum() * 24 / (Lt / 3600))
            cb = res["cal"][sc][: (Lc // 3600) * 3600].reshape(-1, 3600).max(1); B = len(cb); kk = max(1, int((B + 1) / 24))
            tb = np.sort(cb)[::-1][kk - 1]
            tbm = T[sc][: (Lt // 3600) * 3600].reshape(-1, 3600).max(1)
            r[f"{sc}_block_nominal"] = kk / (B + 1); r[f"{sc}_block_realised"] = float((tbm > tb).mean())
            r[f"{sc}_FA24_block"] = float(episodes(T[sc] > tb, seg_t).sum() * 24 / (Lt / 3600))
        rows.append(r); print(FAM, rho, rep, {a: round(v, 4) for a, v in r.items() if a not in ("family", "rho", "rep")}, flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
print("tauA", round(tauA, 3))
