"""
TailGuard | Step 5: training, selection, seeds, matched GRU baseline, ablations, regime hold-outs (GPU recommended).
Resumable: every run checkpoints each epoch; finished runs are skipped. Test labels are NEVER read here.

Usage:  python scripts/tailguard_train.py --stage all
Stages: search -> select -> final (seeds, GRU baseline, ablations, regime hold-outs)
Outputs: runs/train/<run_id>/{ckpt.pt, summary.json, scores.npz}, runs/train/selected.json, runs/train/progress.csv
"""
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from tailguard_lib import (load, manifest, ecdf_ref, tail_surprise, causal_smooth, calibrate_threshold,
                          alarm_stats, event_stats, apply_injections)

OUT = os.path.join(ROOT, "runs", "train"); os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [17, 29, 43, 71, 101]
GRID_W, GRID_INV, GRID_PHYS = [30, 60, 120], [0.0, 0.1, 1.0], [0.0, 0.1, 1.0]
BATCH, MAX_EPOCHS, PATIENCE, LR, L2, W_MAX, SMOOTH, TOPK = 256, 100, 10, 1e-3, 1e-5, 120, 30, 3
REGIMES = ["low", "mid", "high"]


# ============================================================================ data
class Data:
    def __init__(self):
        m = manifest(); self.m = m
        tr, te = load("train"), load("test")
        self.targets = tr["targets"]; sc = m["scaler_fit"]
        self.scale = np.array([sc[c]["scale"] for c in self.targets]); self.med = np.array([sc[c]["median"] for c in self.targets])
        self.ids, self.idte = tr["df_ids"], te["df_ids"]
        self.Z, self.Zte = tr["Z"], te["Z"]
        self.X = np.hstack([tr["Z"], tr["C"], tr["D"]]).astype(np.float32)
        self.Xte = np.hstack([te["Z"], te["C"], te["D"]]).astype(np.float32)
        self.part = self.ids["partition"].to_numpy(); self.reg = self.ids["regime"].to_numpy()
        self.elig = self.ids["eligible"].to_numpy(); self.seg = self.ids["segment_id"].to_numpy()
        self.pos = self.ids["pos_in_segment"].to_numpy(); self.day = self.ids["env_id"].to_numpy()
        self.pos_te = self.idte["pos_in_segment"].to_numpy(); self.seg_te = self.idte["segment_id"].to_numpy()
        self.rels = [r for r in json.load(open(os.path.join(ROOT, "manifests", "constraints.json")))["relations"]
                     if r["status"] == "retained"]
        self.dev_events = json.load(open(os.path.join(ROOT, "runs", "baselines", "dev_injections.json")))
        self.nF, self.nT = self.X.shape[1], self.Z.shape[1]
        print(f"data: train {self.X.shape}, test {self.Xte.shape}, {len(self.rels)} constraints, device {DEV}")

    def raw(self, Z):
        return Z.astype(np.float64) * self.scale + self.med

    # ---- constraint residual r_k = x_target - sum_lag0 c f(x) - B, with B from observed past values only
    def B_terms(self, raw):
        n = raw.shape[0]; out = []
        for r in self.rels:
            ti = self.targets.index(r["target"][0]); mode = r["target"][1]
            B = np.full(n, r["intercept"])
            if mode.startswith("diff"):
                k = int(mode[4:] or 1); B += np.r_[np.full(k, np.nan), raw[:-k, ti]]
            lag0 = []
            for (tag, lag, trf), c in zip(r["terms"], r["coef"]):
                j = self.targets.index(tag)
                if lag == 0:
                    lag0.append((j, c, trf)); continue
                v = raw[:, j]
                if isinstance(lag, str):
                    k = int(lag[4:]); cs = np.r_[0.0, np.cumsum(v)]
                    mv = np.full(n, np.nan); mv[k:] = (cs[k:n] - cs[:n - k]) / k; v = mv
                else:
                    v = np.r_[np.full(lag, np.nan), v[:-lag]]
                B += c * (np.sqrt(np.maximum(v, 0)) if trf == "sqrt" else v)
            out.append((ti, lag0, np.nan_to_num(B), r["scale_a"]))
        return out

    def resid_obs(self, raw, Bt):
        cols = []
        for ti, lag0, B, a in Bt:
            r = raw[:, ti] - B
            for j, c, trf in lag0:
                r = r - c * (np.sqrt(np.maximum(raw[:, j], 0)) if trf == "sqrt" else raw[:, j])
            cols.append(np.abs(r) / a)
        return np.column_stack(cols).astype(np.float32)


# ============================================================================ model
class GRUForecaster(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.gru = nn.GRU(n_in, 64, num_layers=2, batch_first=True)
        self.head = nn.Linear(64, n_out)
    def forward(self, x):
        h, _ = self.gru(x); return self.head(h[:, -1])


def windows(Xg, idx, W):
    off = torch.arange(-W, 0, device=Xg.device)
    return Xg[idx[:, None] + off[None, :]]


def phys_resid_torch(zhat, rows, Bt_g, scale_g, med_g):
    raw = zhat * scale_g + med_g; cols = []
    for ti, lag0, B, a in Bt_g:
        r = raw[:, ti] - B[rows]
        for j, c, trf in lag0:
            r = r - c * (torch.sqrt(torch.clamp(raw[:, j], min=1e-6)) if trf == "sqrt" else raw[:, j])
        cols.append(r / a)
    return torch.stack(cols, 1)


def huber_rows(e):
    return F.huber_loss(e, torch.zeros_like(e), reduction="none", delta=1.0).mean(1)


# ============================================================================ one run
def run(cfg, D, G):
    rid = cfg["id"]; rdir = os.path.join(OUT, rid); os.makedirs(rdir, exist_ok=True)
    if os.path.exists(os.path.join(rdir, "summary.json")):
        return json.load(open(os.path.join(rdir, "summary.json")))
    t0 = time.time(); W, seed, hold = cfg["W"], cfg["seed"], cfg.get("holdout")
    keep = (D.reg != REGIMES.index(hold)) if hold else np.ones(len(D.reg), bool)
    fit_idx = np.where(D.elig & (D.part == "fit") & keep & (D.pos >= W))[0]
    env_key = D.day if cfg["env"] == "day" else D.reg
    envs = [fit_idx[env_key[fit_idx] == e] for e in np.unique(env_key[fit_idx])]
    envs = [e for e in envs if len(e) >= 1000]
    val_idx = np.where(D.elig & (D.part == "val") & keep & (D.pos >= W))[0][::5]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = GRUForecaster(D.nF, D.nT).to(DEV); opt = torch.optim.Adam(model.parameters(), LR)
    state = {"epoch": 0, "best": np.inf, "bad": 0, "best_state": None, "history": []}
    ck = os.path.join(rdir, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)        # RNG states must stay on the CPU
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); state = s["state"]
        rng.bit_generator.state = s["np_rng"]; torch.set_rng_state(s["torch_rng"].cpu())
        if DEV == "cuda" and s.get("cuda_rng") is not None: torch.cuda.set_rng_state(s["cuda_rng"].cpu())
        print(f"  [{rid}] resumed at epoch {state['epoch']}")
    steps = len(fit_idx) // BATCH
    per_env = max(1, BATCH // len(envs))
    while state["epoch"] < MAX_EPOCHS and state["bad"] < PATIENCE:
        model.train()
        for _ in range(steps):
            if cfg["balanced"]:
                parts = [rng.choice(e, per_env) for e in envs]
            else:
                b = rng.choice(fit_idx, BATCH); parts = [b[env_key[b] == e] for e in np.unique(env_key[b])]
            rows = torch.as_tensor(np.concatenate(parts), device=DEV)
            zhat = model(windows(G["X"], rows, W)); z = G["Z"][rows]
            fr = huber_rows(zhat - z)
            pr = huber_rows(phys_resid_torch(zhat, rows, G["Bt"], G["scale"], G["med"]))
            sizes = [len(p) for p in parts]; R, P = [], []
            o = 0
            for s_ in sizes:
                if s_ == 0: continue
                R.append(fr[o:o + s_].mean()); P.append(pr[o:o + s_].mean()); o += s_
            R, P = torch.stack(R), torch.stack(P)
            pen_src = P if cfg["inv_on"] == "phys" else R
            var = pen_src.var(unbiased=False) if len(pen_src) >= 2 else torch.zeros((), device=DEV)
            loss = R.mean() + cfg["l_inv"] * var + cfg["l_phys"] * P.mean() + L2 * sum((p ** 2).sum() for p in model.parameters())
            opt.zero_grad(); loss.backward(); opt.step()
        # validation: worst-regime Huber normalised by persistence (selection and early stopping)
        model.eval(); ratios = []
        with torch.no_grad():
            for g in np.unique(D.reg[val_idx]):
                vi = val_idx[D.reg[val_idx] == g]; num = den = 0.0
                for k in range(0, len(vi), 8192):
                    rows = torch.as_tensor(vi[k:k + 8192], device=DEV)
                    zhat = model(windows(G["X"], rows, W)); z = G["Z"][rows]
                    num += huber_rows(zhat - z).sum().item(); den += huber_rows(z - G["Z"][rows - 1]).sum().item()
                ratios.append(num / max(den, 1e-9))
        vl = float(max(ratios)); state["epoch"] += 1; state["history"].append(vl)
        if vl < state["best"] - 1e-6:
            state["best"] = vl; state["bad"] = 0
            state["best_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            state["bad"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                    "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if DEV == "cuda" else None}, ck)
        print(f"  [{rid}] epoch {state['epoch']:3d} val worst-regime loss {vl:.4f} (best {state['best']:.4f}) {time.time()-t0:.0f}s", flush=True)
    model.load_state_dict({k: v.to(DEV) for k, v in state["best_state"].items()})
    summ = score_and_summarise(cfg, model, D, G, keep, state)
    summ["train_seconds"] = time.time() - t0
    np.savez_compressed(os.path.join(rdir, "scores.npz"), **summ.pop("_arrays"))
    json.dump(summ, open(os.path.join(rdir, "summary.json"), "w"), indent=1, default=float)
    os.remove(ck) if os.path.exists(ck) else None
    torch.save(state["best_state"], os.path.join(rdir, "model_best.pt"))
    with open(os.path.join(OUT, "progress.csv"), "a") as f:
        f.write(f"{rid},{summ['selection_loss']:.4f},{summ['epochs']},{summ['dev']['b0.5']['event_recall']:.3f},{summ['train_seconds']:.0f}\n")
    return summ


@torch.no_grad()
def predict_errors(model, Xg, Zg, W, n):
    E = np.zeros((n, Zg.shape[1]), np.float32)
    for k in range(W_MAX, n, 8192):
        rows = torch.arange(k, min(k + 8192, n), device=DEV)
        E[k:k + len(rows)] = (model(windows(Xg, rows, W)) - Zg[rows]).abs().cpu().numpy()
    return E


def streams(E, V, pos, refE, refV):
    p = np.sort(tail_surprise(E, refE), 1)[:, -TOPK:].mean(1)
    v = np.sort(tail_surprise(V, refV), 1)[:, -min(TOPK, V.shape[1]):].mean(1)
    bad = pos < W_MAX; p[bad] = 0; v[bad] = 0
    return causal_smooth(p, SMOOTH), causal_smooth(v, SMOOTH)


def score_and_summarise(cfg, model, D, G, keep, state):
    model.eval(); W = cfg["W"]; n = len(D.Z)
    E = predict_errors(model, G["X"], G["Z"], W, n)
    Ete = predict_errors(model, G["Xte"], G["Zte"], W, len(D.Zte))
    V, Vte = G["V"], G["Vte"]
    valm = D.elig & (D.part == "val") & keep; calm = D.elig & (D.part == "cal") & keep
    refE, refV = ecdf_ref(E[valm]), ecdf_ref(V[valm])
    p, v = streams(E, V, D.pos, refE, refV); pte, vte = streams(Ete, Vte, D.pos_te, refE, refV)
    out = {"config": cfg, "selection_loss": state["best"], "epochs": state["epoch"], "history": state["history"],
           "tau": {}, "val_clean": {}, "dev": {}, "holdout": {}}
    # development injections (validation copy)
    Zi, evid = apply_injections(D.Z, D.dev_events)
    Xi = D.X.copy(); Xi[:, :D.nT] = Zi
    Xi_g = torch.as_tensor(Xi, device=DEV); Zi_g = torch.as_tensor(Zi, device=DEV)
    Ei = predict_errors(model, Xi_g, Zi_g, W, n); del Xi_g, Zi_g
    rawi = D.raw(Zi); Vi = D.resid_obs(rawi, D.B_terms(rawi))
    pi_, vi_ = streams(Ei, Vi, D.pos, refE, refV)
    fitm = D.elig & (D.part == "fit") & keep
    for beta in (0.0, 0.5, 1.0):
        s = (1 - beta) * p + beta * v; si = (1 - beta) * pi_ + beta * vi_
        tau, _ = calibrate_threshold(s[calm], D.seg[calm], calm.sum() / 3600)
        tau_fit, _ = calibrate_threshold(s[fitm], D.seg[fitm], fitm.sum() / 3600)
        key = f"b{beta}"; out["tau"][key] = tau; out["tau"]["provisional_fit_" + key] = tau_fit
        out["val_clean"][key] = alarm_stats(s[valm] > tau, D.seg[valm], np.ones(valm.sum(), bool))
        out["val_clean"]["provisional_" + key] = alarm_stats(s[valm] > tau_fit, D.seg[valm], np.ones(valm.sum(), bool))
        devm = D.elig & (D.part == "val")
        a = si[devm] > tau; ev = evid[devm]
        out["dev"][key] = {**event_stats(a, ev), "FA24": alarm_stats(a, D.seg[devm], ev == "")["FA24"]}
        if cfg.get("holdout"):
            hm = D.elig & ~keep
            out["holdout"][key] = alarm_stats(s[hm] > tau, D.seg[hm], np.ones(hm.sum(), bool))
    out["_arrays"] = {"p_train": p.astype(np.float32), "v_train": v.astype(np.float32),
                      "p_test": pte.astype(np.float32), "v_test": vte.astype(np.float32)}
    return out


# ============================================================================ plan
def cfg_id(c):
    s = f"W{c['W']}_inv{c['l_inv']}_phys{c['l_phys']}_{c['inv_on']}_{c['env']}_{'bal' if c['balanced'] else 'pool'}_s{c['seed']}"
    return (c.get("tag", "") + "__" if c.get("tag") else "") + s + (f"_hold-{c['holdout']}" if c.get("holdout") else "")


def mk(tag, W, li, lp, seed, inv_on="phys", env="regime", balanced=True, holdout=None):
    c = {"tag": tag, "W": W, "l_inv": li, "l_phys": lp, "inv_on": inv_on, "env": env, "balanced": balanced,
         "seed": seed, "holdout": holdout}
    c["id"] = cfg_id(c); return c


def select(results):
    """D10 (approved 2026-10-14): maximise event recall on the 36 development injections in validation data at the
    calibrated budget of one false alarm per 24 h (beta = 0.5); ties -> lower worst-regime validation forecast loss.
    Uses validation data and synthetic faults only; test labels are never read."""
    best = max(results, key=lambda r: (round(r["dev"]["b0.5"]["event_recall"], 6), -r["selection_loss"]))
    sel = {k: best["config"][k] for k in ("W", "l_inv", "l_phys")}
    sel["rule"] = ("max development-injection event recall at the calibrated 1/24 h alarm budget (beta=0.5); "
                   "ties by lower worst-regime normalised validation loss")
    sel["dev_recall"] = best["dev"]["b0.5"]["event_recall"]; sel["selection_loss"] = best["selection_loss"]
    sel["candidates"] = {r["config"]["id"]: [r["dev"]["b0.5"]["event_recall"], r["selection_loss"]] for r in results}
    json.dump(sel, open(os.path.join(OUT, "selected.json"), "w"), indent=1)
    print("selected:", {k: sel[k] for k in ("W", "l_inv", "l_phys", "dev_recall", "selection_loss")}); return sel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop starting new runs after this many hours (0 = no limit)")
    a = ap.parse_args(); t_start = time.time()
    def out_of_time():
        if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
            print("TIME BUDGET REACHED - all finished runs are saved; run the same command again to continue"); return True
        return False
    global MAX_EPOCHS
    if a.smoke: MAX_EPOCHS = 1
    D = Data()
    G = {"X": torch.as_tensor(D.X, device=DEV), "Z": torch.as_tensor(D.Z, device=DEV),
         "Xte": torch.as_tensor(D.Xte, device=DEV), "Zte": torch.as_tensor(D.Zte, device=DEV),
         "scale": torch.as_tensor(D.scale, dtype=torch.float32, device=DEV), "med": torch.as_tensor(D.med, dtype=torch.float32, device=DEV)}
    raw, rawte = D.raw(D.Z), D.raw(D.Zte)
    Bt, Btte = D.B_terms(raw), D.B_terms(rawte)
    G["Bt"] = [(ti, lag0, torch.as_tensor(B, dtype=torch.float32, device=DEV), a_) for ti, lag0, B, a_ in Bt]
    G["V"], G["Vte"] = D.resid_obs(raw, Bt), D.resid_obs(rawte, Btte)
    if a.smoke:
        print("smoke test: one epoch of one configuration"); print(run(mk("smoke", 60, 0.1, 0.1, 17), D, G)["dev"]); return
    search = [mk("search", W, li, lp, 17) for W in GRID_W for li in GRID_INV for lp in GRID_PHYS]
    if a.stage in ("search", "all"):
        for i, c in enumerate(search):
            if out_of_time(): return
            print(f"[search {i+1}/{len(search)}] {c['id']}"); run(c, D, G)
    if a.stage in ("select", "final", "all"):
        sp = os.path.join(OUT, "selected.json")
        sel = json.load(open(sp)) if os.path.exists(sp) and a.stage == "final" else \
              select([json.load(open(os.path.join(OUT, c["id"], "summary.json"))) for c in search])
    if a.stage in ("final", "all"):
        W, li, lp = sel["W"], sel["l_inv"], sel["l_phys"]
        plan = []
        plan += [mk("tailguard", W, li, lp, s) for s in SEEDS[1:]]                       # seed 17 = search run
        plan += [mk("gru_erm", W, 0.0, 0.0, s, balanced=False) for s in SEEDS]           # matched GRU baseline
        plan += [mk("A1_nophys", W, li, 0.0, s) for s in SEEDS]
        plan += [mk("A2_noinv", W, 0.0, lp, s) for s in SEEDS]
        plan += [mk("A3_pooled", W, 0.0, lp, s, balanced=False) for s in SEEDS]
        plan += [mk("A10_vrex", W, max(li, 0.1), lp, s, inv_on="forecast") for s in SEEDS]
        plan += [mk("A11_dayenv", W, li, lp, s, env="day") for s in SEEDS]
        for h in REGIMES:
            plan += [mk("hold_tailguard", W, li, lp, s, holdout=h) for s in SEEDS]
            plan += [mk("hold_gru", W, 0.0, 0.0, s, balanced=False, holdout=h) for s in SEEDS]
        for i, c in enumerate(plan):
            if out_of_time(): return
            print(f"[final {i+1}/{len(plan)}] {c['id']}"); run(c, D, G)
        print("ALL RUNS COMPLETE")


if __name__ == "__main__":
    main()
