"""
TailGuard | Step 3: classical baselines under the common protocol.
Selection and score constants use normal validation data; threshold uses calibration data only.
Test scores are written SEALED (no labels joined). Development injections on validation data give a
detection signal for design decisions without touching the test labels.
Run: python scripts\\step3_baselines.py
"""
import json, os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tailguard_lib import (ecdf_ref, tail_surprise, policy_score, SCORE_POLICY, load, lagged, calibrate_threshold, alarm_stats, event_stats, make_injections,
                          apply_injections, RUNS, manifest)
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

LAGS = [1, 2, 5, 10, 30, 60]
B_FLOOR = 0.01
rng = np.random.default_rng(17)


def huber(e, d=1.0):
    a = np.abs(e); return np.where(a <= d, 0.5 * a * a, d * (a - 0.5 * d))


class Persistence:
    name = "persistence"
    def fit(self, X, Z, fitmask): return self
    def predict(self, X, Z): P = np.empty_like(Z); P[1:] = Z[:-1]; P[0] = Z[0]; return P


class RidgeAR:
    def __init__(self, alpha): self.alpha = alpha; self.name = f"ridge_a{alpha}"
    def feats(self, X): return np.nan_to_num(lagged(X, LAGS), copy=False)
    def fit(self, X, Z, fitmask):
        idx = np.where(fitmask)[0][::4]
        self.m = Ridge(alpha=self.alpha).fit(self.feats(X)[idx], Z[idx]); return self
    def predict(self, X, Z): return self.m.predict(self.feats(X)).astype(np.float32)


class PCARecon:
    def __init__(self, k): self.k = k; self.name = f"pca_k{k}"
    def fit(self, X, Z, fitmask):
        idx = np.where(fitmask)[0][::2]; self.m = PCA(self.k, random_state=17).fit(Z[idx]); return self
    def predict(self, X, Z): return self.m.inverse_transform(self.m.transform(Z)).astype(np.float32)


class IForest:
    name = "iforest"
    def feats(self, Z):
        d1 = np.empty_like(Z); d1[1:] = Z[1:] - Z[:-1]; d1[0] = 0
        cs = np.cumsum(np.r_[np.zeros((1, Z.shape[1]), np.float32), Z], axis=0)
        m10 = (cs[10:] - cs[:-10]) / 10; m10 = np.r_[np.repeat(m10[:1], 9, 0), m10]
        return np.hstack([Z, d1, Z - m10])
    def fit(self, X, Z, fitmask):
        idx = np.where(fitmask)[0]; idx = rng.choice(idx, 100_000, replace=False)
        self.m = IsolationForest(n_estimators=200, max_samples=256, random_state=17).fit(self.feats(Z)[idx]); return self
    def score(self, Z): return -self.m.score_samples(self.feats(Z))


def evaluate(method, X, Z, ids, Xte, Zte, idte, dev_events, sel_only=False):
    elig = ids["eligible"].to_numpy(); part = ids["partition"].to_numpy(); seg = ids["segment_id"].to_numpy()
    fitm, valm, calm = elig & (part == "fit"), elig & (part == "val"), elig & (part == "cal")
    method.fit(X, Z, fitm)
    if isinstance(method, IForest):
        s_all = policy_score(method.score(Z)); b = None
        sel_loss = np.nan
    else:
        P = method.predict(X, Z)
        E = np.abs(Z - P)
        b = ecdf_ref(E[valm])
        s_all = policy_score(tail_surprise(E, b))
        # worst-regime validation Huber loss, normalised by persistence loss (selection rule)
        Ep = np.zeros_like(Z); Ep[1:] = Z[1:] - Z[:-1]
        regs = ids["regime"].to_numpy()
        ratios = []
        for r in np.unique(regs[valm]):
            mk = valm & (regs == r)
            ratios.append(huber(Z[mk] - P[mk]).mean() / max(huber(Ep[mk]).mean(), 1e-6))
        sel_loss = float(max(ratios))
    if sel_only:
        return {"selection_loss": sel_loss}
    hours_cal = calm.sum() / 3600
    tau, allowed = calibrate_threshold(s_all[calm], seg[calm], hours_cal)
    res = {"score_policy": SCORE_POLICY, "method": method.name, "selection_loss_worst_regime": sel_loss, "tau": tau,
           "cal_allowed_episodes": allowed}
    res["val_normal"] = alarm_stats(s_all[valm] > tau, seg[valm], np.ones(valm.sum(), bool))
    res["cal_normal"] = alarm_stats(s_all[calm] > tau, seg[calm], np.ones(calm.sum(), bool))
    # per-regime false alarms on validation (shift sensitivity)
    regs = ids["regime"].to_numpy()
    res["val_FA24_by_regime"] = {}
    for r in np.unique(regs[valm]):
        mk = valm & (regs == r)
        res["val_FA24_by_regime"][["low", "mid", "high"][r]] = alarm_stats(s_all[mk] > tau, seg[mk], np.ones(mk.sum(), bool))["FA24"]
    # development injections on validation data
    Zi, evid = apply_injections(Z, dev_events)
    Xi = np.hstack([Zi, X[:, Z.shape[1]:]])
    if isinstance(method, IForest):
        si = policy_score(method.score(Zi))
    else:
        si = policy_score(tail_surprise(np.abs(Zi - method.predict(Xi, Zi)), b))
    a = si[valm] > tau
    ev = evid[valm]
    res["dev_injection"] = {**event_stats(a, ev), **{k: v for k, v in alarm_stats(a, seg[valm], ev == "").items()
                                                    if k in ("FA24", "normal_FPR")}}
    by_kind = {}
    for kind in ("bias", "freeze", "ramp"):
        ids_k = {f"dev{k}" for k, e in enumerate(dev_events) if e["kind"] == kind}
        evk = np.where(np.isin(ev, list(ids_k)), ev, "")
        by_kind[kind] = event_stats(a, evk)["event_recall"]
    res["dev_recall_by_kind"] = by_kind
    # sealed test scores
    elt = idte["eligible"].to_numpy()
    st = policy_score(method.score(Zte)) if isinstance(method, IForest) else policy_score(tail_surprise(np.abs(Zte - method.predict(Xte, Zte)), b))
    out = idte.loc[elt, ["timestamp", "file_id", "segment_id"]].copy()
    out["score"] = st[elt].astype(np.float32); out["alarm"] = out["score"] > tau
    d = os.path.join(RUNS, "baselines", method.name); os.makedirs(d, exist_ok=True)
    out.to_parquet(os.path.join(d, "test_scores_SEALED.parquet"), index=False)
    np.savez_compressed(os.path.join(d, "train_scores.npz"), s_train=s_all.astype(np.float32))  # val/cal scores for matched protocols
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    return res


def main():
    t0 = time.time()
    tr = load("train"); te = load("test")
    Z, Zte = tr["Z"], te["Z"]
    X = np.hstack([tr["Z"], tr["C"], tr["D"]]); Xte = np.hstack([te["Z"], te["C"], te["D"]])
    ids, idte = tr["df_ids"], te["df_ids"]
    m = manifest()
    elig = ids["eligible"].to_numpy(); part = ids["partition"].to_numpy()
    dev = make_injections(len(Z), ids["segment_id"].to_numpy(), elig & (part == "val"), Z.shape[1], None)
    os.makedirs(os.path.join(RUNS, "baselines"), exist_ok=True)
    with open(os.path.join(RUNS, "baselines", "dev_injections.json"), "w") as f:
        json.dump(dev, f, indent=1)
    print(f"{len(dev)} development injections on validation data")
    results = []
    # selection among grids (normal validation only)
    best = {}
    for fam, cands in {"ridge": [RidgeAR(a) for a in (0.1, 1.0, 10.0)], "pca": [PCARecon(k) for k in (5, 10, 20)]}.items():
        losses = [(evaluate(c, X, Z, ids, Xte, Zte, idte, dev, sel_only=True)["selection_loss"], c) for c in cands]
        for l, c in losses: print(f"  selection {c.name}: worst-regime normalised loss {l:.4f}")
        best[fam] = min(losses, key=lambda t: t[0])[1]
    for meth in [Persistence(), best["ridge"], best["pca"], IForest()]:
        r = evaluate(meth, X, Z, ids, Xte, Zte, idte, dev)
        results.append(r)
        print(f"{r['method']:14s} tau={r['tau']:.3f}  val FA24={r['val_normal']['FA24']:.2f} "
              f"(by regime {', '.join(f'{k}:{v:.2f}' for k, v in r['val_FA24_by_regime'].items())})  "
              f"dev recall={r['dev_injection']['event_recall']:.2f} {r['dev_recall_by_kind']}  [{time.time()-t0:.0f}s]")
    pd.DataFrame([{"method": r["method"], "tau": r["tau"], "val_FA24": r["val_normal"]["FA24"],
                   **{f"val_FA24_{k}": v for k, v in r["val_FA24_by_regime"].items()},
                   "val_normal_FPR": r["val_normal"]["normal_FPR"], "dev_event_recall": r["dev_injection"]["event_recall"],
                   "dev_delay_median_s": r["dev_injection"]["delay_median_s"],
                   **{f"dev_recall_{k}": v for k, v in r["dev_recall_by_kind"].items()}} for r in results]
                 ).to_csv(os.path.join(RUNS, "baselines", "baseline_dev_summary.csv"), index=False)


if __name__ == "__main__":
    main()
