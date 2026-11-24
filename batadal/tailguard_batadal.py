"""
TailGuard on BATADAL (C-Town water distribution network): confirmation run under batadal/BATADAL_PROTOCOL.md.
The model, losses, scoring functions and threshold rules are imported from the released TailGuard code
(scripts/tailguard_lib.py, scripts/tailguard_train.py); this file only adapts data handling to the hourly BATADAL files.
Stages (resumable; finished runs are skipped):
  python batadal/tailguard_batadal.py --stage prepare      # audit, split, scaling            (no labels read)
  python batadal/tailguard_batadal.py --stage constraints  # documented relations, screened   (fitting data only)
  python batadal/tailguard_batadal.py --stage train        # TailGuard, matched GRU, A1: 5 seeds each
  python batadal/tailguard_batadal.py --stage baselines    # ridge AR, PCA, persistence
  python batadal/tailguard_batadal.py --stage evaluate     # thresholds on calibration data, then the 14 attacks
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tailguard_lib import ecdf_ref, tail_surprise, causal_smooth, calibrate_threshold, episodes, lagged
RAW, OUT = os.path.join(HERE, "raw"), os.path.join(HERE, "runs"); os.makedirs(OUT, exist_ok=True)
W, SEEDS, TOPK, SMOOTHS, BETA = 30, [17, 29, 43, 71, 101], 3, (30, 6, 1), 0.5
BUDGET_H, BLOCK, BATCH, MAX_EPOCHS, PATIENCE, LR, L2 = 720.0, 24, 256, 100, 10, 1e-3, 1e-5
FILES = {"train": "BATADAL_dataset03.csv", "eval04": "BATADAL_dataset04.csv", "evaltest": "BATADAL_test_dataset.csv"}
P = lambda *a: os.path.join(HERE, *a)

def read(key):
    d = pd.read_csv(os.path.join(RAW, FILES[key])); d.columns = [c.strip() for c in d.columns]
    d["timestamp"] = pd.to_datetime(d["DATETIME"].str.strip(), format="%d/%m/%y %H")
    return d.drop(columns=[c for c in ("DATETIME", "ATT_FLAG") if c in d.columns])       # ATT_FLAG is never kept

# ============================================================================ prepare
def prepare():
    tr = read("train"); tags = [c for c in tr.columns if c != "timestamp"]
    const = [c for c in tags if tr[c].nunique() == 1]; tags = [c for c in tags if c not in const]
    disc = [c for c in tags if c.startswith("S_")]; tgt = [c for c in tags if not c.startswith("S_")]
    n = len(tr); n_fit, n_val = int(0.6 * n), int(0.2 * n)
    part = np.array(["fit"] * n_fit + ["val"] * n_val + ["cal"] * (n - n_fit - n_val))
    fit = tr.iloc[:n_fit]; sc = {}
    for c in tgt:
        v = fit[c].to_numpy(float); q1, q3 = np.percentile(v, [25, 75]); floor = max(0.01 * (v.max() - v.min()), 1e-6)
        sc[c] = {"median": float(np.median(v)), "scale": float(max(q3 - q1, floor))}
    env = np.full(n, -1); env[:n_fit] = np.minimum(np.arange(n_fit) * 3 // n_fit, 2)
    man = {"files": {k: {"name": f, "sha256": hashlib.sha256(open(os.path.join(RAW, f), "rb").read()).hexdigest(), "rows": int(len(read(k)))} for k, f in FILES.items()},
           "dropped_constant": const, "targets": tgt, "discrete": disc, "discrete_levels_fit": {c: sorted(int(x) for x in fit[c].unique()) for c in disc},
           "scaler_fit": sc, "partition_rows": {"fit": n_fit, "val": n_val, "cal": n - n_fit - n_val},
           "partition_start": {k: str(tr.timestamp[np.where(part == k)[0][0]]) for k in ("fit", "val", "cal")},
           "environments": "three equal consecutive time bands of the fitting data", "sampling": "hourly", "window": W}
    json.dump(man, open(P("split_manifest.json"), "w"), indent=1)
    np.savez(P("partition.npz"), part=part, env=env)
    for k in FILES:
        d = read(k); ok = d.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all()
        print(f"{k}: {len(d)} rows {d.timestamp.iloc[0]} -> {d.timestamp.iloc[-1]}  hourly without gaps: {ok}")
    print(f"targets {len(tgt)}, discrete {len(disc)}, dropped constant {const}"); print("partition", man["partition_rows"], man["partition_start"])

class Data:
    def __init__(self):
        m = json.load(open(P("split_manifest.json"))); self.m = m; self.targets = m["targets"]; sc = m["scaler_fit"]
        self.scale = np.array([sc[c]["scale"] for c in self.targets]); self.med = np.array([sc[c]["median"] for c in self.targets])
        pz = np.load(P("partition.npz")); self.part, self.env = pz["part"], pz["env"]
        self.sets = {}
        for k in FILES:
            d = read(k); Z = ((d[self.targets].to_numpy(float) - self.med) / self.scale).astype(np.float32); D = []
            for c in m["discrete"]:
                lv = m["discrete_levels_fit"][c]; v = d[c].to_numpy()
                D += [(v == l).astype(np.float32) for l in lv]; D.append((~np.isin(v, lv)).astype(np.float32))
            self.sets[k] = {"Z": Z, "X": np.hstack([Z, np.column_stack(D)]).astype(np.float32), "ts": d.timestamp.to_numpy(), "raw": d[self.targets].to_numpy(float)}
        self.n = len(self.sets["train"]["Z"]); self.pos = np.arange(self.n); self.elig = self.pos >= W
        self.nF, self.nT = self.sets["train"]["X"].shape[1], len(self.targets)
        cp = P("constraints.json"); self.rels = [r for r in json.load(open(cp))["relations"] if r["status"] == "retained"] if os.path.exists(cp) else []
    def raw(self, Z): return Z.astype(np.float64) * self.scale + self.med

# ============================================================================ constraints
def registry():
    feeds = {"T1": ["PU1", "PU2"], "T2": ["V2"], "T3": ["PU4"], "T4": ["PU6", "PU7"], "T5": ["PU8"], "T7": ["PU10", "PU11"]}
    R = [{"id": f"mass_{t}", "class": "mass balance", "evidence": f"ctown_map.inp [CONTROLS]: {', '.join(a)} open and close on the level of {t}, which they fill",
          "target": [f"L_{t}", "diff1"], "terms": [[f"F_{x}", "mean1", "id"] for x in a], "sign_check": {f"F_{x}": 1 for x in a}} for t, a in feeds.items()]
    for link, up, dn in (("PU2", "J280", "J269"), ("PU6", "J289", "J415"), ("PU10", "J307", "J317"), ("V2", "J14", "J422")):
        R.append({"id": f"head_{link}", "class": "head across a pump or valve", "evidence": f"ctown_map.inp: {link} connects {up} to {dn}; head change is a function of its flow",
                  "target": [f"P_{dn}", "level"], "terms": [[f"P_{up}", 0, "id"], [f"F_{link}", 0, "id"]]})
    return R
def constraints():
    sys.path.insert(0, os.path.join(ROOT, "scripts")); from step4_constraints import design, RATIO, MIN_N, MIN_R2
    from sklearn.linear_model import HuberRegressor
    D = Data(); df = pd.DataFrame(D.sets["train"]["raw"], columns=D.targets); fit = D.elig & (D.part == "fit"); out, lines = [], []
    for rel in registry():
        tags = {rel["target"][0]} | {t[0] for t in rel["terms"]}
        if not tags <= set(D.targets):
            rel.update(status="unavailable", reason=f"tags not usable: {sorted(tags - set(D.targets))}"); out.append(rel); lines.append(f"{rel['id']:12s} UNAVAILABLE {rel['reason']}"); continue
        X, y = design(df, rel); idx = np.where(fit)[0]; idx = idx[~np.isnan(X[idx]).any(1) & ~np.isnan(y[idx])]
        mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-12
        hr = HuberRegressor(epsilon=1.35, alpha=1e-6, max_iter=500).fit((X[idx] - mu) / sd, y[idx])
        coef = hr.coef_ / sd; icpt = hr.intercept_ - (hr.coef_ * mu / sd).sum(); r = y - (X @ coef + icpt); rf = r[fit & ~np.isnan(r)]
        a = float(max(1.4826 * np.median(np.abs(rf - np.median(rf))), 1e-3 * (np.nanstd(y[fit]) + 1e-12)))
        r2 = float(1 - np.mean(rf ** 2) / np.nanvar(y[fit])); qp = np.percentile(np.abs(rf), 95); per = {}
        for g in range(3):
            mk = fit & (D.env == g) & ~np.isnan(r)
            if mk.sum() >= MIN_N: per[f"band{g + 1}"] = float(np.percentile(np.abs(r[mk]), 95) / max(qp, 1e-12))
        worst = max(per.values()) if per else np.inf
        sign_ok = all(np.sign(coef[[t[0] for t in rel["terms"]].index(tag)]) == s for tag, s in rel.get("sign_check", {}).items())
        keep = worst <= RATIO and len(per) >= 2 and sign_ok and r2 >= MIN_R2
        rel.update(coef=coef.tolist(), intercept=float(icpt), scale_a=a, r2_fit=r2, q95_ratio_by_environment=per, worst_ratio=float(worst), status="retained" if keep else "rejected",
                   reason="" if keep else ("sign contradicts documented physics" if not sign_ok else f"not identifiable: fitting R2 {r2:.2f} < {MIN_R2}" if r2 < MIN_R2 else f"instability: max q95 ratio {worst:.2f} > {RATIO}"))
        out.append(rel); lines.append(f"{rel['id']:12s} {rel['status'].upper():9s} R2={r2:.3f} q95 ratios " + ", ".join(f"{k}:{v:.2f}" for k, v in per.items()) + f"  {rel['reason']}")
    json.dump({"screen_ratio": RATIO, "min_targets_per_environment": MIN_N, "relations": out}, open(P("constraints.json"), "w"), indent=1)
    lines.append(f"Retained {sum(r['status'] == 'retained' for r in out)} of {len(out)} relations."); print("\n".join(lines)); open(P("constraints_report.txt"), "w").write("\n".join(lines))

# ============================================================================ scoring shared by every detector
def stream(T, w):
    s = np.sort(T, 1)[:, -min(TOPK, T.shape[1]):].mean(1); s[:W] = 0; return causal_smooth(s, w)
def mstream(T, w):
    s = T.max(1).copy(); s[:W] = 0; return causal_smooth(s, w)
def score_sets(E, V, D):
    """E, V: dict set -> |error| arrays (V may be None). Returns per set: forecast and constraint streams, top-3 and max, per smoothing."""
    valm = D.elig & (D.part == "val"); refE = ecdf_ref(E["train"][valm], step=1); refV = ecdf_ref(V["train"][valm], step=1) if V else None; out = {}
    for k in E:
        TE = tail_surprise(E[k], refE); TV = tail_surprise(V[k], refV) if V else None
        for w in SMOOTHS:
            out[f"{k}_p{w}"] = stream(TE, w); out[f"{k}_pm{w}"] = mstream(TE, w)
            if V: out[f"{k}_v{w}"] = stream(TV, w); out[f"{k}_vm{w}"] = mstream(TV, w)
    return out

# ============================================================================ train
def train():
    import torch
    from tailguard_train import GRUForecaster, windows, phys_resid_torch, huber_rows, Data as HD
    torch.set_num_threads(2); D = Data(); T = D.sets["train"]; dev = "cpu"
    G = {k: torch.as_tensor(D.sets[k]["X"]) for k in FILES}; Zg = {k: torch.as_tensor(D.sets[k]["Z"]) for k in FILES}
    scale_g, med_g = torch.as_tensor(D.scale, dtype=torch.float32), torch.as_tensor(D.med, dtype=torch.float32)
    Bt = {k: HD.B_terms(D, D.raw(D.sets[k]["Z"])) for k in FILES}; V = {k: HD.resid_obs(D, D.raw(D.sets[k]["Z"]), Bt[k]) for k in FILES}
    Bt_g = [(ti, lag0, torch.as_tensor(B, dtype=torch.float32), a) for ti, lag0, B, a in Bt["train"]]
    fit_idx = np.where(D.elig & (D.part == "fit"))[0]; val_idx = np.where(D.elig & (D.part == "val"))[0]
    envs = [fit_idx[D.env[fit_idx] == e] for e in range(3)]
    plan = [("tailguard", 1.0, 0.1, True)] + [("gru", 0.0, 0.0, False), ("A1_nophys", 1.0, 0.0, True)]
    for tag, l_inv, l_phys, balanced in plan:
        for seed in SEEDS:
            rdir = os.path.join(OUT, f"{tag}_s{seed}"); os.makedirs(rdir, exist_ok=True)
            if os.path.exists(os.path.join(rdir, "scores.npz")): continue
            t0 = time.time(); torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = GRUForecaster(D.nF, D.nT); opt = torch.optim.Adam(model.parameters(), LR)
            best, bad, best_state, hist = np.inf, 0, None, []; steps = len(fit_idx) // BATCH; per_env = max(1, BATCH // len(envs))
            while len(hist) < MAX_EPOCHS and bad < PATIENCE:
                model.train()
                for _ in range(steps):
                    if balanced: parts = [rng.choice(e, per_env) for e in envs]
                    else:
                        b = rng.choice(fit_idx, BATCH); parts = [b[D.env[b] == e] for e in np.unique(D.env[b])]
                    rows = torch.as_tensor(np.concatenate(parts)); zhat = model(windows(G["train"], rows, W)); z = Zg["train"][rows]
                    fr = huber_rows(zhat - z); pr = huber_rows(phys_resid_torch(zhat, rows, Bt_g, scale_g, med_g)); R, Pq, o = [], [], 0
                    for s_ in [len(p) for p in parts]:
                        if s_ == 0: continue
                        R.append(fr[o:o + s_].mean()); Pq.append(pr[o:o + s_].mean()); o += s_
                    R, Pq = torch.stack(R), torch.stack(Pq); var = Pq.var(unbiased=False) if len(Pq) >= 2 else torch.zeros(())
                    loss = R.mean() + l_inv * var + l_phys * Pq.mean() + L2 * sum((p ** 2).sum() for p in model.parameters())
                    opt.zero_grad(); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    rows = torch.as_tensor(val_idx); zhat = model(windows(G["train"], rows, W)); z = Zg["train"][rows]
                    vl = huber_rows(zhat - z).sum().item() / max(huber_rows(z - Zg["train"][rows - 1]).sum().item(), 1e-9)
                hist.append(vl)
                if vl < best - 1e-6: best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
                else: bad += 1
            model.load_state_dict(best_state); model.eval(); E = {}
            with torch.no_grad():
                for k in FILES:
                    n = len(D.sets[k]["Z"]); e = np.zeros((n, D.nT), np.float32); rows = torch.arange(W, n)
                    e[W:] = (model(windows(G[k], rows, W)) - Zg[k][rows]).abs().numpy(); E[k] = e
            np.savez_compressed(os.path.join(rdir, "scores.npz"), **score_sets(E, V, D))
            torch.save(best_state, os.path.join(rdir, "model_best.pt"))
            json.dump({"tag": tag, "seed": seed, "l_inv": l_inv, "l_phys": l_phys, "balanced": balanced, "epochs": len(hist), "val_loss": best, "history": hist, "seconds": time.time() - t0}, open(os.path.join(rdir, "summary.json"), "w"), indent=1)
            print(f"{tag} seed {seed}: {len(hist)} epochs, validation loss ratio {best:.4f}, {time.time() - t0:.0f}s", flush=True)

# ============================================================================ classical baselines
def baselines():
    from sklearn.linear_model import Ridge; from sklearn.decomposition import PCA
    D = Data(); fit = D.elig & (D.part == "fit"); LAGS = [1, 2, 5, 10, 30]; Z = {k: D.sets[k]["Z"] for k in FILES}; X = {k: D.sets[k]["X"] for k in FILES}
    feats = lambda x: np.nan_to_num(lagged(x, LAGS), copy=False)
    rm = Ridge(alpha=1.0).fit(feats(X["train"])[fit], Z["train"][fit]); pm = PCA(20, random_state=17).fit(Z["train"][fit])
    preds = {"ridge": {k: rm.predict(feats(X[k])).astype(np.float32) for k in FILES}, "pca": {k: pm.inverse_transform(pm.transform(Z[k])).astype(np.float32) for k in FILES},
             "persistence": {k: np.r_[Z[k][:1], Z[k][:-1]] for k in FILES}}
    for name, pr in preds.items():
        rdir = os.path.join(OUT, f"{name}_s17"); os.makedirs(rdir, exist_ok=True)
        np.savez_compressed(os.path.join(rdir, "scores.npz"), **score_sets({k: np.abs(pr[k] - Z[k]) for k in FILES}, None, D))
        json.dump({"tag": name, "seed": 17}, open(os.path.join(rdir, "summary.json"), "w")); print("saved", name)

# ============================================================================ evaluate
def labels(D):
    att = json.load(open(P("batadal_attacks.json"))); out = {}
    for k, key in (("eval04", "dataset04"), ("evaltest", "test")):
        ts = pd.to_datetime(D.sets[k]["ts"]); ev = np.full(len(ts), "", dtype=object)
        for i, (a, b) in enumerate(att[key]):
            ev[(ts >= pd.to_datetime(a, format="%d/%m/%Y %H:%M")) & (ts <= pd.to_datetime(b, format="%d/%m/%Y %H:%M"))] = f"{key}_{i + 1}"
        out[k] = ev
    return out
def metrics(a, ev, elig):
    """a: alarm mask, ev: event ids ('' normal), elig: rows with full history. One evaluation file = one segment."""
    a = a & elig; y = ev != ""; nm = elig & ~y; st = episodes(a, np.zeros(len(a), int)); ids = [e for e in pd.unique(ev) if e]
    det, ttd = 0, []
    for e in ids:
        ix = np.where(ev == e)[0]; h = a[ix]
        if h.any(): det += 1; ttd.append(int(np.argmax(h)) / len(ix))
        else: ttd.append(1.0)
    tp, fp, fn, tn = (a & y).sum(), (a & nm).sum(), (~a & y & elig).sum(), (~a & nm).sum()
    return {"events": len(ids), "detected": det, "false_episodes": int((st & nm).sum()), "normal_hours": int(nm.sum()), "normal_hours_in_alarm": int(fp),
            "S_TTD": 1 - float(np.mean(ttd)), "TPR": tp / max(tp + fn, 1), "TNR": tn / max(tn + fp, 1), "F1": 2 * tp / max(2 * tp + fp + fn, 1), "ttd": ttd}
def evaluate():
    D = Data(); L = labels(D); calm = D.elig & (D.part == "cal"); cal_h = int(calm.sum()); rows = []
    ci = np.where(calm)[0]; CB = [ci[j * BLOCK:(j + 1) * BLOCK] for j in range(len(ci) // BLOCK)]; B = len(CB); kb = max(1, int(np.floor((B + 1) * BLOCK / BUDGET_H)))
    for rd in sorted(os.listdir(OUT)):
        z = np.load(os.path.join(OUT, rd, "scores.npz")); sm = json.load(open(os.path.join(OUT, rd, "summary.json"))); has_v = "train_v30" in z.files
        variants = {"prespecified": ("p", "v"), "max": ("pm", "vm")}
        for w in SMOOTHS:
            for vn, (pk, vk) in variants.items():
                f = lambda k: ((1 - BETA) * z[f"{k}_{pk}{w}"] + BETA * z[f"{k}_{vk}{w}"]) if (has_v and vn == "prespecified") else \
                              (np.maximum(z[f"{k}_{pk}{w}"], z[f"{k}_{vk}{w}"]) if has_v else z[f"{k}_{pk}{w}"])
                st = f("train"); tau, allowed = calibrate_threshold(st[calm], np.zeros(cal_h, int), cal_h * 24.0 / BUDGET_H)
                tau_k = float(np.sort([st[b].max() for b in CB])[::-1][kb - 1])
                for rule, th in (("closure", tau), ("block", tau_k)):
                    r = {"method": sm["tag"], "seed": sm["seed"], "score": vn, "smooth": w, "rule": rule, "tau": th, "cal_hours": cal_h, "allowed_cal_episodes": allowed, "B": B, "k": kb, "block_guarantee": kb / (B + 1)}
                    tot = {"events": 0, "detected": 0, "false_episodes": 0, "normal_hours": 0, "normal_hours_in_alarm": 0, "alarmed_normal_days": 0, "normal_days": 0}
                    for k in ("eval04", "evaltest"):
                        s = f(k); el = np.arange(len(s)) >= W; mt = metrics(s > th, L[k], el)
                        for q in ("events", "detected", "false_episodes", "normal_hours", "normal_hours_in_alarm"): tot[q] += mt[q]
                        ei = np.where(el)[0]; TB = [ei[j * BLOCK:(j + 1) * BLOCK] for j in range(len(ei) // BLOCK)]; TB = [b for b in TB if not (L[k][b] != "").any()]
                        tot["normal_days"] += len(TB); tot["alarmed_normal_days"] += int(sum((s[b] > th).any() for b in TB))
                        if k == "evaltest": r.update({"test_S_TTD": mt["S_TTD"], "test_S_CM": (mt["TPR"] + mt["TNR"]) / 2, "test_S": 0.5 * mt["S_TTD"] + 0.5 * (mt["TPR"] + mt["TNR"]) / 2,
                                                      "test_F1": mt["F1"], "test_TPR": mt["TPR"], "test_TNR": mt["TNR"], "test_detected": mt["detected"]})
                        r[f"{k}_detected"] = mt["detected"]; r[f"{k}_false_episodes"] = mt["false_episodes"]
                    r.update(tot); r["event_recall"] = tot["detected"] / tot["events"]; r["FA_per_30d"] = tot["false_episodes"] * BUDGET_H / tot["normal_hours"]
                    r["normal_time_in_alarm"] = tot["normal_hours_in_alarm"] / tot["normal_hours"]; r["alarmed_normal_day_share"] = tot["alarmed_normal_days"] / max(tot["normal_days"], 1); rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(P("results_runs.csv"), index=False)
    g = df.groupby(["smooth", "score", "rule", "method"]); cols = ["tau", "event_recall", "detected", "FA_per_30d", "normal_time_in_alarm", "alarmed_normal_day_share", "test_S", "test_S_TTD", "test_S_CM", "test_F1"]
    s = g[cols].mean().round(3); s["seeds"] = g.size(); s["recall_sd"] = g["event_recall"].std().round(3); s.to_csv(P("results_summary.csv"))
    pd.set_option("display.width", 250); print(s.loc[30].to_string()); print(f"calibration {cal_h} h, allowed episodes {cal_h / BUDGET_H:.2f}; blocks B={B}, k={kb}, guarantee {kb / (B + 1):.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["prepare", "constraints", "train", "baselines", "evaluate"]); a = ap.parse_args()
    {"prepare": prepare, "constraints": constraints, "train": train, "baselines": baselines, "evaluate": evaluate}[a.stage]()
