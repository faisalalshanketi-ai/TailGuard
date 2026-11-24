"""
TailGuard | refitted regime hold-out and A7 runs (GPU recommended; resumable, finished runs are skipped).

  refitted regime hold-out: for the withheld regime h, robust scalers, discrete levels and the
      constraint registry (coefficients, scales, screening) are refitted on retained-regime fitting rows only, and every
      target whose input window reaches into the withheld regime is excluded from fitting, validation and calibration.
      Models: TailGuard, matched GRU, A2 (no invariance penalty) and A10 (invariance on forecast risk)  -> H2 controls.
  A7 randomised constraints: every retained relation keeps its target, lags and transforms, but its
      input tags are replaced by tags drawn at random from the other continuous channels; coefficients and scales are
      refitted on fitting data; no screening. Five seeds (one random registry per seed).
  Every run saves the prespecified score (top-3 mean, beta) and the max tail-surprise score (top-1, maximum fusion).

  python scripts/holdout_a7_runs.py --part 0 --parts 2 --max-hours 11     (one process per GPU)
Outputs: runs/holdout_a7/<run_id>/{summary.json, scores.npz, model_best.pt}
Test labels are never read here.
"""
import argparse, copy, json, os, sys, time
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import tailguard_train as T
import step4_constraints as S4
from tailguard_lib import (ecdf_ref, tail_surprise, causal_smooth, calibrate_threshold, alarm_stats, event_stats,
                          apply_injections, manifest)
from sklearn.linear_model import HuberRegressor

T.OUT = os.path.join(ROOT, "runs", "holdout_a7"); os.makedirs(T.OUT, exist_ok=True)
REG = T.REGIMES


# ------------------------------------------------------------------ data variants
def raw_frame(split):
    return pd.read_parquet(os.path.join(ROOT, "data", "derived", f"{os.environ.get('TAILGUARD_PREFIX', 'hai2305')}_{split}.parquet"))


def choose_window(df, fitmask, rel):
    """Step 4 rule: the averaging window K in {1,3,10,30} with the highest fitting-data R^2 (on rows in fitmask)."""
    best = None
    for K in S4.WINDOWS:
        cand = dict(rel, terms=[[t[0], f"mean{K}" if t[1] == "meanK" else t[1], t[2]] for t in rel["terms"]])
        Xc, yc = S4.design(df, cand); ii = np.where(fitmask)[0][::10]; ii = ii[~np.isnan(Xc[ii]).any(1) & ~np.isnan(yc[ii])]
        Xs = (Xc[ii] - Xc[ii].mean(0)) / (Xc[ii].std(0) + 1e-12)
        pr = HuberRegressor(max_iter=300).fit(Xs, yc[ii]).predict(Xs); r2 = 1 - np.mean((yc[ii] - pr) ** 2) / np.var(yc[ii])
        if best is None or r2 > best[0]: best = (r2, K)
    rel = copy.deepcopy(rel); rel["terms"] = [[t[0], f"mean{best[1]}" if t[1] == "meanK" else t[1], t[2]] for t in rel["terms"]]
    rel["window_s"] = best[1]; return rel


def fit_relations(df, fitmask, rels, screen=True):
    """Refit coefficients/scales (and averaging windows, for registry entries with meanK terms) on rows in fitmask."""
    reg = df["regime"].to_numpy(); out = []
    for rel in rels:
        rel = copy.deepcopy(rel)
        if any(t[1] == "meanK" for t in rel["terms"]): rel = choose_window(df, fitmask, rel)
        X, y = S4.design(df, rel)
        idx = np.where(fitmask)[0][::10]; idx = idx[~np.isnan(X[idx]).any(1) & ~np.isnan(y[idx])]
        mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-12
        hr = HuberRegressor(epsilon=1.35, alpha=1e-6, max_iter=500).fit((X[idx] - mu) / sd, y[idx])
        coef = hr.coef_ / sd; icpt = hr.intercept_ - (hr.coef_ * mu / sd).sum()
        r = y - (X @ coef + icpt); ok = fitmask & ~np.isnan(r); rf = r[ok]
        mad = 1.4826 * np.median(np.abs(rf - np.median(rf))); floor = 1e-3 * (np.nanstd(y[fitmask]) + 1e-12)
        r2 = float(1 - np.mean(rf ** 2) / np.nanvar(y[fitmask])); q = np.percentile(np.abs(rf), 95)
        per = {REG[g]: float(np.percentile(np.abs(r[ok & (reg == g)]), 95) / max(q, 1e-12))
               for g in np.unique(reg[fitmask]) if (ok & (reg == g)).sum() >= S4.MIN_N}
        sign_ok = all(np.sign(coef[[t[0] for t in rel["terms"]].index(tag)]) == s for tag, s in rel.get("sign_check", {}).items())
        keep = (max(per.values()) <= S4.RATIO and len(per) >= 2 and sign_ok and r2 >= S4.MIN_R2) if screen else True
        rel.update(coef=coef.tolist(), intercept=float(icpt), scale_a=float(max(mad, floor)), r2_fit=r2,
                   q95_ratio_by_regime=per, status="retained" if keep else "rejected")
        out.append(rel)
    return out


class CleanData(T.Data):
    """Data with everything fitted on retained regimes only and window-crossing targets moved to the withheld side."""
    def __init__(self, hold):
        super().__init__()
        h = REG.index(hold); m = manifest(); roles = m["feature_roles"]
        tr, te = raw_frame("train"), raw_frame("test")
        fitm = (tr["partition"].to_numpy() == "fit") & (tr["regime"].to_numpy() != h)
        sc = {}
        for c in roles["continuous_targets"] + roles["context_setpoints"]:
            v = tr.loc[fitm, c].to_numpy(float); q1, q3 = np.percentile(v, [25, 75])
            sc[c] = (float(np.median(v)), float(max(q3 - q1, max(0.01 * (v.max() - v.min()), 1e-6))))
        Z = lambda df: np.column_stack([(df[c].to_numpy(float) - sc[c][0]) / sc[c][1] for c in roles["continuous_targets"]]).astype(np.float32)
        C = lambda df: np.column_stack([(df[c].to_numpy(float) - sc[c][0]) / sc[c][1] for c in roles["context_setpoints"]]).astype(np.float32)
        def Dm(df):
            out = []
            for c in roles["discrete_actuators"]:
                lv = sorted(np.unique(tr.loc[fitm, c]).astype(int).tolist()); v = df[c].to_numpy()
                out += [(v == l).astype(np.float32) for l in lv] + [(~np.isin(v, lv)).astype(np.float32)]
            return np.column_stack(out).astype(np.float32)
        self.Z, self.Zte = Z(tr), Z(te)
        self.X = np.hstack([self.Z, C(tr), Dm(tr)]); self.Xte = np.hstack([self.Zte, C(te), Dm(te)])
        self.nF = self.X.shape[1]
        self.scale = np.array([sc[c][1] for c in self.targets]); self.med = np.array([sc[c][0] for c in self.targets])
        usable = set(roles["continuous_targets"]) | set(roles["context_setpoints"])
        base = [r for r in copy.deepcopy(S4.REGISTRY) if ({r["target"][0]} | {t[0] for t in r["terms"]}) <= usable]
        refit = fit_relations(tr, (tr["eligible"].to_numpy() & fitm), base, screen=True)
        self.rels = [r for r in refit if r["status"] == "retained"]
        self.clean_registry = [{k: r[k] for k in ("id", "status", "r2_fit", "q95_ratio_by_regime")} for r in refit]
        # targets whose input window (W_MAX samples) reaches into the withheld regime are treated as withheld
        bad = (self.reg == h).astype(np.int32); cs = np.r_[0, np.cumsum(bad)]
        n = len(bad); lo = np.maximum(np.arange(n) - T.W_MAX, 0)
        crosses = (cs[np.arange(n)] - cs[lo]) > 0
        self.crossing_rows = int((crosses & (self.reg != h)).sum())
        self.reg = np.where(crosses, h, self.reg)
        print(f"clean hold-out {hold}: {len(self.rels)} relations retained on retained regimes, {self.crossing_rows} crossing targets moved")


class RandomData(T.Data):
    """A7: retained relations with input tags drawn at random from other continuous channels, refitted, unscreened."""
    def __init__(self, seed):
        super().__init__()
        rng = np.random.default_rng(1000 + seed); tr = raw_frame("train")
        fitm = tr["eligible"].to_numpy() & (tr["partition"].to_numpy() == "fit")
        pool = list(self.targets); rels = []
        for r in self.rels:
            r = copy.deepcopy(r); tgt = r["target"][0]
            cand = [t for t in pool if t != tgt]
            r["terms"] = [[str(rng.choice(cand)), lag, trf] for tag, lag, trf in r["terms"]]
            r.pop("sign_check", None); rels.append(r)
        self.rels = fit_relations(tr, fitm, rels, screen=False)
        self.random_registry = [{"id": r["id"], "target": r["target"][0], "inputs": [t[0] for t in r["terms"]], "r2_fit": r["r2_fit"]} for r in self.rels]
        print("A7 random registry:", [(r["id"], [t[0] for t in r["terms"]]) for r in self.rels])


# ------------------------------------------------------------------ scoring: prespecified and max tail-surprise
def streams_k(E, V, pos, refE, refV, k):
    p = np.sort(tail_surprise(E, refE), 1)[:, -k:].mean(1); v = np.sort(tail_surprise(V, refV), 1)[:, -min(k, V.shape[1]):].mean(1)
    bad = pos < T.W_MAX; p[bad] = 0; v[bad] = 0
    return causal_smooth(p, T.SMOOTH), causal_smooth(v, T.SMOOTH)


def score_both(cfg, model, D, G, keep, state):
    model.eval(); W = cfg["W"]; n = len(D.Z)
    E = T.predict_errors(model, G["X"], G["Z"], W, n); Ete = T.predict_errors(model, G["Xte"], G["Zte"], W, len(D.Zte))
    V, Vte = G["V"], G["Vte"]
    valm = D.elig & (D.part == "val") & keep; calm = D.elig & (D.part == "cal") & keep
    refE, refV = ecdf_ref(E[valm]), ecdf_ref(V[valm])
    Zi, evid = apply_injections(D.Z, D.dev_events); Xi = D.X.copy(); Xi[:, :D.nT] = Zi
    Ei = T.predict_errors(model, torch.as_tensor(Xi, device=T.DEV), torch.as_tensor(Zi, device=T.DEV), W, n)
    rawi = D.raw(Zi); Vi = D.resid_obs(rawi, D.B_terms(rawi))
    out = {"config": cfg, "selection_loss": state["best"], "epochs": state["epoch"], "history": state["history"],
           "tau": {}, "val_clean": {}, "dev": {}, "holdout": {}, "_arrays": {}}
    for k, tagk in ((3, ""), (1, "max_")):
        p, v = streams_k(E, V, D.pos, refE, refV, k); pte, vte = streams_k(Ete, Vte, D.pos_te, refE, refV, k)
        pi_, vi_ = streams_k(Ei, Vi, D.pos, refE, refV, k)
        fus = (lambda a, b, beta: (1 - beta) * a + beta * b) if k == 3 else (lambda a, b, beta: a if beta == 0 else b if beta == 1 else np.maximum(a, b))
        for beta in (0.0, 0.5, 1.0):
            s, si = fus(p, v, beta), fus(pi_, vi_, beta); key = f"{tagk}b{beta}"
            tau, _ = calibrate_threshold(s[calm], D.seg[calm], calm.sum() / 3600); out["tau"][key] = tau
            out["val_clean"][key] = alarm_stats(s[valm] > tau, D.seg[valm], np.ones(valm.sum(), bool))
            devm = D.elig & (D.part == "val") & keep; a = si[devm] > tau; ev = evid[devm]
            out["dev"][key] = {**event_stats(a, ev), "FA24": alarm_stats(a, D.seg[devm], ev == "")["FA24"]}
            if cfg.get("holdout"):
                hm = D.elig & ~keep; out["holdout"][key] = alarm_stats(s[hm] > tau, D.seg[hm], np.ones(hm.sum(), bool))
        out["_arrays"].update({f"{tagk}p_train": p.astype(np.float32), f"{tagk}v_train": v.astype(np.float32),
                               f"{tagk}p_test": pte.astype(np.float32), f"{tagk}v_test": vte.astype(np.float32)})
    if hasattr(D, "clean_registry"): out["clean_registry"] = D.clean_registry; out["crossing_rows"] = D.crossing_rows
    if hasattr(D, "random_registry"): out["random_registry"] = D.random_registry
    return out


T.score_and_summarise = score_both


def tensors(D):
    G = {"X": torch.as_tensor(D.X, device=T.DEV), "Z": torch.as_tensor(D.Z, device=T.DEV),
         "Xte": torch.as_tensor(D.Xte, device=T.DEV), "Zte": torch.as_tensor(D.Zte, device=T.DEV),
         "scale": torch.as_tensor(D.scale, dtype=torch.float32, device=T.DEV), "med": torch.as_tensor(D.med, dtype=torch.float32, device=T.DEV)}
    raw, rawte = D.raw(D.Z), D.raw(D.Zte); Bt, Btte = D.B_terms(raw), D.B_terms(rawte)
    G["Bt"] = [(ti, lag0, torch.as_tensor(B, dtype=torch.float32, device=T.DEV), a_) for ti, lag0, B, a_ in Bt]
    G["V"], G["Vte"] = D.resid_obs(raw, Bt), D.resid_obs(rawte, Btte)
    return G


def plan():
    sel = json.load(open(os.path.join(ROOT, "runs", "train", "selected.json"))); W, li, lp = sel["W"], sel["l_inv"], sel["l_phys"]
    P = []
    for h in REG:
        for s in T.SEEDS:
            P += [("clean", h, T.mk("clean_tailguard", W, li, lp, s, holdout=h)),
                  ("clean", h, T.mk("clean_gru", W, 0.0, 0.0, s, balanced=False, holdout=h)),
                  ("clean", h, T.mk("clean_A2", W, 0.0, lp, s, holdout=h)),
                  ("clean", h, T.mk("clean_A10", W, max(li, 0.1), lp, s, inv_on="forecast", holdout=h))]
    for s in T.SEEDS:
        P.append(("random", s, T.mk("A7_random", W, li, lp, s)))
    return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", type=int, default=0); ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=0.0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    if a.smoke: T.MAX_EPOCHS = 1
    P = plan()[a.part::a.parts]
    if a.smoke: P = [P[0]] + [x for x in P if x[0] == "random"][:1]
    print(f"part {a.part}/{a.parts}: {len(P)} runs, device {T.DEV}")
    cache_key, D, G = None, None, None
    for i, (kind, arg, cfg) in enumerate(P):
        if os.path.exists(os.path.join(T.OUT, cfg["id"], "summary.json")): continue
        if a.max_hours and (time.time() - t0) / 3600 > a.max_hours:
            print("TIME BUDGET REACHED - run the same command again to continue"); return
        if (kind, arg) != cache_key:
            D = None; G = None; torch.cuda.empty_cache() if T.DEV == "cuda" else None
            D = CleanData(arg) if kind == "clean" else RandomData(arg); G = tensors(D); cache_key = (kind, arg)
        print(f"[{i+1}/{len(P)}] {cfg['id']}", flush=True); T.run(cfg, D, G)
    print("HOLD-OUT AND A7 RUNS COMPLETE (this part)")


if __name__ == "__main__":
    main()
