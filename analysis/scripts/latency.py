"""
TailGuard | streaming cost per second of telemetry on CPU (single thread, no PyTorch).
For each new sample: forecast from the last W samples (NumPy GRU, analysis/scripts/np_gru.py), 55 forecast and
11 constraint residuals, tail surprises against the validation references, top-3 means, beta-fusion, 30 s causal
mean and threshold. Ridge AR (lags 1-60 s) is timed the same way. Reports per-sample latency and model/reference sizes.
  python analysis/scripts/latency.py --root <repo root> --out <csv> [--steps 3000]
"""
import argparse, glob, json, os, sys, time, platform
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=3000)
A = ap.parse_args(); ROOT = os.path.abspath(A.root)
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tailguard_lib import load, manifest, ecdf_ref, tail_surprise
from constraints_np import constraint_matrix
from np_gru import load_state, forecast
from sklearn.linear_model import Ridge

tr = load("train"); m = manifest(); tg = tr["targets"]; sc = m["scaler_fit"]
scale = np.array([sc[c]["scale"] for c in tg]); med = np.array([sc[c]["median"] for c in tg])
ids = tr["df_ids"]; Z = tr["Z"]; X = np.hstack([tr["Z"], tr["C"], tr["D"]]).astype(np.float32)
part = ids["partition"].to_numpy(); elig = ids["eligible"].to_numpy(); valm = elig & (part == "val"); calm = elig & (part == "cal")
sel = json.load(open(os.path.join(ROOT, "runs", "train", "selected.json"))); W = sel["W"]
run = os.path.join(ROOT, "runs", "train", f"search__W{W}_inv{sel['l_inv']}_phys{sel['l_phys']}_phys_regime_bal_s17")
S = load_state(os.path.join(run, "model_best.pt")); n_par = int(sum(v.size for v in S.values()))
rels = [r for r in json.load(open(os.path.join(ROOT, "manifests", "constraints.json")))["relations"] if r["status"] == "retained"]
raw = Z.astype(np.float64) * scale + med; V = constraint_matrix(rels, tg, raw)
vi = np.where(valm)[0]
E = np.abs(forecast(S, X, vi[::5], W) - Z[vi[::5]]); refE = np.sort(E, 0); refV = ecdf_ref(V[valm])
ci = np.where(calm)[0]; start = int(ci[len(ci) // 2]); steps = range(start, start + A.steps)
HIST = 61                                                         # longest constraint window + 1 (diff30 / mean30)

def tailguard_step(t, buf):
    zhat = forecast(S, X, np.array([t]), W, batch=1)[0]
    q = tail_surprise(np.abs(zhat - Z[t])[None, :], refE)[0]
    r = constraint_matrix(rels, tg, raw[t - HIST:t + 1])[-1:]
    qv = tail_surprise(r, refV)[0]
    s = 0.5 * np.sort(q)[-3:].mean() + 0.5 * np.sort(qv)[-3:].mean()
    buf.append(s); buf[:] = buf[-30:]; return np.mean(buf) > 7.1

def timeit(f):
    buf = []; [f(t, buf) for t in list(steps)[:50]]
    t0 = time.perf_counter(); [f(t, buf) for t in steps]; return 1e3 * (time.perf_counter() - t0) / A.steps

LAGS = [1, 2, 5, 10, 30, 60]; fit = np.where(elig & (part == "fit"))[0][::4]
feats = lambda rows: np.hstack([X[rows - L] for L in LAGS])        # = lagged(X, LAGS)[rows] for eligible rows (history >= 120)
rg = Ridge(alpha=1.0).fit(feats(fit), Z[fit]); Wr, br = rg.coef_.astype(np.float32), rg.intercept_.astype(np.float32)
vr = np.where(valm)[0]; refR = np.sort(np.abs(Z[vr] - (feats(vr) @ Wr.T + br))[::5], 0)
def ridge_step(t, buf):
    f = np.concatenate([X[t - L] for L in LAGS]); zhat = Wr @ f + br
    q = tail_surprise(np.abs(zhat - Z[t])[None, :], refR)[0]; buf.append(np.sort(q)[-3:].mean()); buf[:] = buf[-30:]; return np.mean(buf) > 7.1

rows = [{"detector": "TailGuard (GRU forecast + 11 constraints + tail surprise + fusion)", "ms_per_sample": timeit(tailguard_step),
         "parameters": n_par, "model_kB": round(4 * n_par / 1024, 1), "reference_MB": round((refE.nbytes + refV.nbytes) / 2 ** 20, 1)},
        {"detector": "Ridge AR (lags 1-60 s + tail surprise)", "ms_per_sample": timeit(ridge_step),
         "parameters": int(Wr.size + br.size), "model_kB": round(4 * (Wr.size + br.size) / 1024, 1), "reference_MB": round(refR.nbytes / 2 ** 20, 1)}]
df = pd.DataFrame(rows); df["cpu"] = platform.processor() or platform.machine(); df["threads"] = 1
df.to_csv(A.out, index=False); print(df.to_string())
