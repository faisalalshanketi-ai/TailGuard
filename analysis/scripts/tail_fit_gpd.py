"""TailGuard | tail-fit diagnostics (Section 4.3): generalized Pareto shape of the excesses over Q, per channel,
for the seed-17 TailGuard forecast errors and the constraint residuals; reference = validation (every fifth target),
check = calibration. A shape of 0 is the exponential excess assumed by Propositions 1(d), 3(b) and 4.
  python tail_fit_gpd.py --root <repo root> --out <csv>"""
import argparse, json, os, sys, numpy as np, pandas as pd
from scipy.stats import genpareto
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); A = ap.parse_args()
ROOT = os.path.abspath(A.root); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "analysis", "scripts"))
from tailguard_lib import load, manifest, ecdf_ref
from constraints_np import constraint_matrix
from np_gru import load_state, forecast
tr = load("train"); m = manifest(); tg = tr["targets"]; sc = m["scaler_fit"]
scale = np.array([sc[c]["scale"] for c in tg]); med = np.array([sc[c]["median"] for c in tg])
ids = tr["df_ids"]; Z = tr["Z"]; X = np.hstack([tr["Z"], tr["C"], tr["D"]]).astype(np.float32)
part = ids["partition"].to_numpy(); elig = ids["eligible"].to_numpy()
rels = [r for r in json.load(open(os.path.join(ROOT, "manifests", "constraints.json")))["relations"] if r["status"] == "retained"]
V = constraint_matrix(rels, tg, Z.astype(np.float64) * scale + med)
S = load_state(os.path.join(ROOT, "runs", "train", "search__W30_inv1.0_phys0.1_phys_regime_bal_s17", "model_best.pt"))
val = np.where(elig & (part == "val"))[0]; cal = np.where(elig & (part == "cal"))[0]
E = {k: np.abs(forecast(S, X, r, 30) - Z[r]) for k, r in (("val", val), ("cal", cal))}
rows = []
for stream, D in (("forecast", E), ("constraint", {"val": V[val], "cal": V[cal]})):
    ref = ecdf_ref(D["val"]); n = ref.shape[0]; k = int(np.floor(0.99 * n))
    for j in range(ref.shape[1]):
        Q = ref[k, j]; M = max(float(np.mean(ref[k:, j] - Q)), 1e-3); r = {"stream": stream, "channel": j, "Q": Q, "M": M}
        for nm, x in (("val", ref[:, j]), ("cal", D["cal"][::5, j])):
            ex = x[x > Q] - Q; r[f"n_exc_{nm}"] = len(ex)
            if len(ex) >= 50 and ex.std() > 0:
                xi, _, sg = genpareto.fit(ex, floc=0); r[f"xi_{nm}"] = xi; r[f"mean_excess_ratio_{nm}"] = float(ex.mean() / M)
                r[f"q99_excess_ratio_{nm}"] = float(np.quantile(ex, 0.99) / (M * np.log(100)))
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(A.out, index=False)
for s_, g in df.groupby("stream"):
    for nm in ("val", "cal"):
        x = g[f"xi_{nm}"].dropna(); q = g[f"q99_excess_ratio_{nm}"].dropna()
        print(f"{s_:10s} {nm}: channels {len(x)}/{len(g)}  GPD shape median {x.median():.2f} IQR {x.quantile(.25):.2f}..{x.quantile(.75):.2f}  share xi>0.1 {(x>0.1).mean():.2f}  share xi<-0.1 {(x<-0.1).mean():.2f}  99% excess quantile / exponential: median {q.median():.2f}, share>1.5 {(q>1.5).mean():.2f}; mean-excess ratio median {g[f'mean_excess_ratio_{nm}'].median():.2f}")
