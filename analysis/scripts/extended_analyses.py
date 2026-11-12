"""
TailGuard | extended analyses of the frozen score streams (no retraining).

  python analysis/scripts/extended_analyses.py --root <release root> --prefix hai2305|hai2204 --out <dir> <analysis> [...]

<release root> holds scripts/, manifests/, data/derived/ and runs/ of one release (the repository root for HAI 23.05,
hai2204/ for HAI 22.04). Analyses:
  reproduce    recomputes the common-protocol results of the paper from the saved streams (consistency check)
  budget       alarm-budget sweep 0.25-8 episodes/day: event recall against realised false alarms per day (Pareto view)
  calwin       threshold calibrated on the most recent 6, 12, 24 h or all calibration data; realised test rate
  lambda_inv   invariance coefficient: the 27 search candidates grouped by lambda_inv (development faults, validation
               loss, test recall/FA24 by attack-onset regime)
  constraints  degraded constraint registry in the score: drop one relation, keep random subsets, perturb coefficients
Test labels are read by every analysis; each reading is appended to
manifests/unseal_log.txt. Torch is not needed.
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--prefix", required=True); ap.add_argument("--out", required=True)
ap.add_argument("analyses", nargs="+")
A = ap.parse_args()
ROOT, PFX = os.path.abspath(A.root), A.prefix
os.environ["TAILGUARD_PREFIX"] = PFX
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tailguard_lib import (calibrate_threshold, alarm_stats, event_stats, point_stats, ecdf_ref, tail_surprise,
                          causal_smooth, lagged)
os.makedirs(A.out, exist_ok=True)
SEEDS = [17, 29, 43, 71, 101]; REG = ["low", "mid", "high"]; W_MAX = 120; BUDGETS = (0.25, 0.5, 1, 2, 4, 8)

# ------------------------------------------------------------------ data
DER = os.path.join(ROOT, "data", "derived")
te = pd.read_parquet(os.path.join(DER, f"{PFX}_test.parquet"), columns=["timestamp", "file_id", "segment_id", "pos_in_segment", "eligible", "regime"])
tr = pd.read_parquet(os.path.join(DER, f"{PFX}_train.parquet"), columns=["partition", "eligible", "regime", "segment_id", "pos_in_segment"])
mask = te["eligible"].to_numpy(); seg_te = te["segment_id"].to_numpy(); reg_te = te["regime"].to_numpy()
part = tr["partition"].to_numpy(); elig = tr["eligible"].to_numpy(); seg = tr["segment_id"].to_numpy(); pos = tr["pos_in_segment"].to_numpy()
calm, valm = elig & (part == "cal"), elig & (part == "val")
HCAL = calm.sum() / 3600
_lab = None


def labels(why):
    global _lab
    if _lab is None:
        _lab = pd.read_parquet(os.path.join(DER, f"{PFX}_test_labels.parquet"))
        assert len(_lab) == len(te) and (_lab["timestamp"].values == te["timestamp"].values).all()
    with open(os.path.join(ROOT, "manifests", "unseal_log.txt"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} test labels read by extended_analyses.py ({why}; descriptive, frozen scores)\n")
    return _lab["label"].to_numpy().astype(bool), np.where(mask, _lab["event_id"].to_numpy(), "")


sel = json.load(open(os.path.join(ROOT, "runs", "train", "selected.json"))); W, LI, LP = sel["W"], sel["l_inv"], sel["l_phys"]
def rid(tag, s, li=LI, lp=LP, inv="phys", env="regime", bal="bal", w=None):
    return f"{tag}__W{w or W}_inv{li}_phys{lp}_{inv}_{env}_{bal}_s{s}"
def main_run(s): return rid("search", 17) if s == 17 else rid("tailguard", s)
def gru(run, fuse, kind=""):
    z = np.load(os.path.join(ROOT, "runs", "train", run, "scores.npz")); p, v = kind + "p", kind + "v"
    f = {"mean0.5": lambda a, b: 0.5 * a + 0.5 * b, "max": np.maximum, "p": lambda a, b: a, "v": lambda a, b: b}[fuse]
    return f(z[p + "_train"], z[v + "_train"]), f(z[p + "_test"], z[v + "_test"])
def baseline(name):
    d = os.path.join(ROOT, "runs", "baselines", name); sc = pd.read_parquet(os.path.join(d, "test_scores_SEALED.parquet"))
    s = np.full(len(te), -np.inf, np.float32); key = te.loc[te.eligible, ["file_id", "timestamp"]].reset_index()
    mm = key.merge(sc[["file_id", "timestamp", "score"]], on=["file_id", "timestamp"], how="left"); s[mm["index"].to_numpy()] = mm["score"].to_numpy()
    return np.load(os.path.join(d, "train_scores.npz"))["s_train"], s
def deep(m, s):
    z = np.load(os.path.join(ROOT, "runs", "deep", m, f"s{s}", "scores.npz")); return z["s_train"], z["s_test"]

def methods():
    M = {}
    has_max = "max_p_test" in np.load(os.path.join(ROOT, "runs", "train", main_run(17), "scores.npz")).files
    for s in SEEDS:
        M.setdefault("TailGuard", []).append(gru(main_run(s), "mean0.5"))
        M.setdefault("Matched GRU", []).append(gru(rid("gru_erm", s, 0.0, 0.0, bal="pool"), "p"))
        if has_max:
            M.setdefault("TailGuard, max score", []).append(gru(main_run(s), "max", "max_"))
            M.setdefault("Matched GRU, max score", []).append(gru(rid("gru_erm", s, 0.0, 0.0, bal="pool"), "p", "max_"))
        for m in ("USAD", "TranAD", "GDN"):
            if os.path.exists(os.path.join(ROOT, "runs", "deep", m, f"s{s}", "scores.npz")):
                M.setdefault(m, []).append(deep(m, s))
    for name, d in (("Ridge AR", "ridge_a1.0"), ("PCA", "pca_k20"), ("Persistence", "persistence")):
        M[name] = [baseline(d)]
    M["Constraint residual only"] = [gru(main_run(17), "v")]
    return M

def evaluate(st, ss, y, ev, B=1.0, cal=None):
    cal = calm if cal is None else cal
    tau, _ = calibrate_threshold(st[cal], seg[cal], cal.sum() / 3600, budget_per_24h=B)
    a = (ss > tau) & mask; r = event_stats(a[mask], ev[mask]); fa = alarm_stats(a[mask], seg_te[mask], ~y[mask])
    return {"event_recall": r["event_recall"], "detected": r["detected"], "FA24": fa["FA24"], "false_episodes": fa["false_episodes"], "tau": tau}

def mean_over(rr, keys=("event_recall", "detected", "FA24", "false_episodes")):
    d = {k: float(np.mean([r[k] for r in rr])) for k in keys}
    d.update({k + "_sd": float(np.std([r[k] for r in rr], ddof=1)) if len(rr) > 1 else 0.0 for k in ("event_recall", "FA24")})
    return d

# ------------------------------------------------------------------ analyses
def reproduce():
    y, ev = labels("reproduction of the common-protocol table"); rows = []
    for name, runs in methods().items():
        rows.append({"method": name, "n_seeds": len(runs), **mean_over([evaluate(st, ss, y, ev) for st, ss in runs])})
        print(f"  {name:28s} recall {rows[-1]['event_recall']:.3f}  FA24 {rows[-1]['FA24']:.2f}")
    pd.DataFrame(rows).to_csv(os.path.join(A.out, f"reproduce_{PFX}.csv"), index=False)

def budget():
    y, ev = labels("alarm-budget sweep"); rows = []
    for name, runs in methods().items():
        for B in BUDGETS:
            rows.append({"method": name, "budget_per_day": B, "n_seeds": len(runs), **mean_over([evaluate(st, ss, y, ev, B) for st, ss in runs])})
    bs = pd.DataFrame(rows); bs.to_csv(os.path.join(A.out, f"budget_sensitivity_{PFX}.csv"), index=False)
    mb = []
    for name, g in bs.groupby("method"):
        g = g.sort_values("FA24"); x, r = g["FA24"].to_numpy(), g["event_recall"].to_numpy()
        for t in (0.5, 1.0, 1.5, 2.0):
            mb.append({"method": name, "realised_FA24": t, "event_recall_interp": float(np.interp(t, x, r)) if x.min() <= t <= x.max() else np.nan})
    pd.DataFrame(mb).pivot(index="method", columns="realised_FA24", values="event_recall_interp").to_csv(os.path.join(A.out, f"matched_burden_{PFX}.csv"))
    print(bs.pivot_table(index="method", columns="budget_per_day", values=["event_recall", "FA24"]).round(3).to_string())

def calwin():
    y, ev = labels("calibration-window length"); rows = []
    ci = np.where(calm)[0]; M = methods()
    for name in ("TailGuard", "Matched GRU", "Ridge AR"):
        for hours in (6, 12, 24, None):
            cal = calm.copy()
            if hours is not None:
                cal[:] = False; cal[ci[-int(hours * 3600):]] = True       # most recent calibration hours before deployment
            rr = [evaluate(st, ss, y, ev, 1.0, cal) for st, ss in M[name]]
            rows.append({"method": name, "calibration_hours": round(cal.sum() / 3600, 1), **mean_over(rr)})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(A.out, f"calibration_window_{PFX}.csv"), index=False); print(df.round(3).to_string())

def lambda_inv():
    y, ev = labels("invariance-coefficient sensitivity of the search candidates"); rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "train", "search__*_s17", "summary.json"))):
        sm = json.load(open(f)); c = sm["config"]; st, ss = gru(c["id"], "mean0.5")
        tau = sm["tau"]["b0.5"]; a = (ss > tau) & mask
        r = {"W": c["W"], "lambda_inv": c["l_inv"], "lambda_phys": c["l_phys"], "dev_detected": sm["dev"]["b0.5"]["detected"],
             "val_worst_regime_loss": sm["selection_loss"], "test_event_recall": event_stats(a[mask], ev[mask])["event_recall"],
             "test_FA24": alarm_stats(a[mask], seg_te[mask], ~y[mask])["FA24"]}
        for h_i, h in enumerate(REG):
            on = {e for e in pd.unique(ev[mask & y]) if e and reg_te[np.where(ev == e)[0][0]] == h_i}
            evh = np.where(np.isin(ev, list(on)), ev, ""); nm = mask & (reg_te == h_i) & ~y
            r[f"recall_{h}"] = event_stats(a[mask], evh[mask])["event_recall"] if on else np.nan
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["W", "lambda_phys", "lambda_inv"]); df.to_csv(os.path.join(A.out, f"lambda_inv_{PFX}.csv"), index=False)
    g = df.groupby("lambda_inv")[["dev_detected", "val_worst_regime_loss", "test_event_recall", "test_FA24", "recall_low", "recall_mid", "recall_high"]].mean()
    g.to_csv(os.path.join(A.out, f"lambda_inv_summary_{PFX}.csv")); print(df.round(3).to_string()); print(g.round(3).to_string())

from constraints_np import constraint_matrix

def v_stream(V, Vte, pos_te):
    ref = ecdf_ref(V[valm]); k = min(3, V.shape[1])
    def s(X, p):
        q = np.concatenate([np.sort(tail_surprise(X[i:i + 200000], ref), 1)[:, -k:].mean(1) for i in range(0, len(X), 200000)])
        q[p < W_MAX] = 0; return causal_smooth(q, 30)
    return s(V, pos), s(Vte, pos_te)

def raw_arrays():
    man = json.load(open(os.path.join(ROOT, "manifests", "split_manifest.json"))); tg = man["feature_roles"]["continuous_targets"]; sc = man["scaler_fit"]
    trd = pd.read_parquet(os.path.join(DER, f"{PFX}_train.parquet"), columns=tg); ted = pd.read_parquet(os.path.join(DER, f"{PFX}_test.parquet"), columns=tg)
    # same float32 scaling round trip as the training code (load() then Data.raw())
    f = lambda d: np.column_stack([((d[c].to_numpy(np.float64) - sc[c]["median"]) / sc[c]["scale"]).astype(np.float32).astype(np.float64) * sc[c]["scale"] + sc[c]["median"] for c in tg])
    return tg, f(trd), f(ted)

def constraints():
    y, ev = labels("degraded constraint registry"); rels = [r for r in json.load(open(os.path.join(ROOT, "manifests", "constraints.json")))["relations"] if r["status"] == "retained"]
    tg, raw, rawte = raw_arrays(); pos_te = te["pos_in_segment"].to_numpy()
    P = [gru(main_run(s), "p") for s in SEEDS]
    v_full, vte_full = v_stream(constraint_matrix(rels, tg, raw), constraint_matrix(rels, tg, rawte), pos_te)
    z = np.load(os.path.join(ROOT, "runs", "train", main_run(17), "scores.npz"))
    print("max |v - saved v| train/test:", float(np.nanmax(np.abs(v_full - z["v_train"]))), float(np.nanmax(np.abs(vte_full - z["v_test"]))))
    def run(label, rr):
        if not rr: return None
        v, vte = v_stream(constraint_matrix(rr, tg, raw), constraint_matrix(rr, tg, rawte), pos_te)
        res = mean_over([evaluate(0.5 * pt + 0.5 * v, 0.5 * pe + 0.5 * vte, y, ev) for pt, pe in P])
        print(f"  {label:40s} recall {res['event_recall']:.3f} FA24 {res['FA24']:.2f}", flush=True); return res
    rows = [{"variant": "full registry (11)", "n_relations": len(rels), "draw": 0, **run("full", rels)}]
    rows.append({"variant": "forecast only (0)", "n_relations": 0, "draw": 0, **mean_over([evaluate(pt, pe, y, ev) for pt, pe in P])})
    for i, r in enumerate(rels):
        rows.append({"variant": f"drop {r['id']}", "n_relations": len(rels) - 1, "draw": 0, **run("drop " + r["id"], rels[:i] + rels[i + 1:])})
    rng = np.random.default_rng(20261102)
    for k in (8, 5, 3):
        for d in range(5):
            keep = sorted(rng.choice(len(rels), k, replace=False)); rows.append({"variant": f"random subset of {k}", "n_relations": k, "draw": d, **run(f"subset {k} #{d}", [rels[i] for i in keep])})
    for eps in (0.05, 0.10, 0.25):
        for d in range(5):
            rr = []
            for r in rels:
                q = dict(r); q["coef"] = [c * (1 + rng.uniform(-eps, eps)) for c in r["coef"]]; rr.append(q)
            rows.append({"variant": f"coefficients perturbed ±{int(eps * 100)}%", "n_relations": len(rels), "draw": d, **run(f"perturb {eps} #{d}", rr)})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(A.out, f"constraint_degradation_{PFX}.csv"), index=False)
    df.groupby("variant", sort=False)[["n_relations", "event_recall", "FA24"]].mean().to_csv(os.path.join(A.out, f"constraint_degradation_summary_{PFX}.csv"))

for a in A.analyses:
    print(f"== {a} ({PFX})", flush=True); globals()[a]()
