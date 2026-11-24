"""Appendix analyses (no test labels are read):
  tail   : tail-fit and exceedance diagnostics of the tail surprise on calibration data, by operating regime (MC3)
  dev    : development-fault sensitivity: other fault seeds and magnitudes, both scores, TailGuard vs GRU vs A1 (MC6)
  risk   : per-regime forecast risk and constraint risk of the refitted hold-out models (TailGuard, GRU, A2, A10; MC5)
Usage: python appendix_analyses.py <root> <mode> <out_dir> [holdout_runs_dir]"""
import os, sys, json, glob, numpy as np, pandas as pd, torch
ROOT, MODE, OUT = sys.argv[1:4]; REV = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.makedirs(OUT, exist_ok=True)
import tailguard_train as T
from tailguard_lib import ecdf_ref, tail_surprise, causal_smooth, event_stats, make_injections, apply_injections
torch.set_num_threads(2); REG = ["low", "mid", "high"]; SEEDS = [17, 29, 43, 71, 101]

def rid(tag, s, li=1.0, lp=0.1, inv="phys", bal="bal", h=""): return f"{tag}__W30_inv{li}_phys{lp}_{inv}_regime_{bal}_s{s}{h}"
def load_model(D, path):
    m = T.GRUForecaster(D.nF, D.nT); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval(); return m
@torch.no_grad()
def predict(m, X, rows, W=30, bs=8192):
    Xg = torch.as_tensor(X); out = np.zeros((len(rows), m.head.out_features), np.float32)
    for k in range(0, len(rows), bs):
        r = torch.as_tensor(rows[k:k + bs]); out[k:k + len(r)] = m(T.windows(Xg, r, W)).numpy()
    return out
def streams_k(E, V, pos, refE, refV, k):
    p = np.sort(tail_surprise(E, refE), 1)[:, -k:].mean(1); v = np.sort(tail_surprise(V, refV), 1)[:, -min(k, V.shape[1]):].mean(1)
    p[pos < T.W_MAX] = 0; v[pos < T.W_MAX] = 0; return causal_smooth(p, T.SMOOTH), causal_smooth(v, T.SMOOTH)

if MODE in ("tail", "dev"):
    D = T.Data(); n = len(D.Z); elig, part, reg, pos, seg = D.elig, D.part, D.reg, D.pos, D.seg
    valm, calm = elig & (part == "val"), elig & (part == "cal")
    rows_vc = np.where(elig & np.isin(part, ["val", "cal"]))[0]
    raw = D.raw(D.Z); V = D.resid_obs(raw, D.B_terms(raw)); refV = ecdf_ref(V[valm])
    MODELS = {"TailGuard": rid("search", 17), "Matched GRU": rid("gru_erm", 17, 0.0, 0.0, bal="pool"), "A1 (no constraint term)": rid("A1_nophys", 17, 1.0, 0.0)}
    def errors(m, Z, X):
        E = np.zeros((n, D.nT), np.float32); E[rows_vc] = np.abs(predict(m, X, rows_vc) - Z[rows_vc]); return E

if MODE == "tail":
    R = []
    m = load_model(D, os.path.join(ROOT, "runs", "train", MODELS["TailGuard"], "model_best.pt")); E = errors(m, D.Z, D.X)
    refE = ecdf_ref(E[valm])
    for name, A, ref in (("forecast channels", E, refE), ("constraint residuals", V, refV)):
        for r in range(3):
            mk = calm & (reg == r)
            if mk.sum() == 0: continue
            q = tail_surprise(A[mk], ref); mstar = q.max(1); mch = q.shape[1]
            for c in (1.0, 2.0, 3.0):
                rate = (q >= c).mean(0); nominal = 10 ** -c
                R.append({"stream": name, "regime": REG[r], "cal_hours": mk.sum() / 3600, "level_c": c, "nominal": nominal,
                          "median_rate": float(np.median(rate)), "p90_rate": float(np.quantile(rate, 0.9)),
                          "share_channels_within_x2": float(((rate >= nominal / 2) & (rate <= nominal * 2)).mean()),
                          "share_channels_above_x2": float((rate > nominal * 2).mean()),
                          "max_channel_rate_at_c_plus_log10m": float((mstar >= c + np.log10(mch)).mean()), "max_nominal": nominal})
        # exponential-tail fit: excess over Q on validation (reference) vs calibration, per channel
        k = int(np.floor(0.99 * len(ref)))
        for r in range(3):
            mk = calm & (reg == r)
            if mk.sum() == 0: continue
            ex_share, ks = [], []
            for j in range(A.shape[1]):
                col = ref[:, j]; Q = col[k]; M = max(float(np.mean(col[k:] - Q)), 1e-3); x = A[mk, j]
                ex = x[x > Q] - Q; ex_share.append((x > Q).mean())
                if len(ex) >= 20:
                    s = np.sort(ex); emp = np.arange(1, len(s) + 1) / len(s); ks.append(float(np.max(np.abs(emp - (1 - np.exp(-s / M))))))
            R.append({"stream": name, "regime": REG[r], "level_c": "tail", "nominal": 0.01, "median_rate": float(np.median(ex_share)),
                      "p90_rate": float(np.quantile(ex_share, 0.9)), "median_KS_exponential": float(np.median(ks)) if ks else np.nan,
                      "channels_with_20plus_exceedances": len(ks)})
    pd.DataFrame(R).to_csv(os.path.join(OUT, "tail_diagnostics.csv"), index=False); print(pd.DataFrame(R).round(4).to_string())

if MODE == "dev":
    SETS = [("seed 17 (selection set)", 17, 1.0), ("seed 29", 29, 1.0), ("seed 43", 43, 1.0), ("seed 71", 71, 1.0), ("seed 101", 101, 1.0),
            ("seed 29, half magnitude", 29, 0.5), ("seed 29, double magnitude", 29, 2.0)]
    R = []
    for mname, run in MODELS.items():
        m = load_model(D, os.path.join(ROOT, "runs", "train", run, "model_best.pt")); E = errors(m, D.Z, D.X); refE = ecdf_ref(E[valm])
        taus = {"prespecified": json.load(open(os.path.join(ROOT, "runs", "train", run, "summary.json")))["tau"],
                "max": json.load(open(os.path.join(ROOT, "runs_max", "train", run, "summary.json")))["tau"]}
        for sname, sd, mag in SETS:
            ev = make_injections(n, seg, valm, D.nT, None, seed=sd)
            for e in ev: e["mag"] *= mag
            Zi, evid = apply_injections(D.Z, ev); Xi = D.X.copy(); Xi[:, :D.nT] = Zi
            Ei = errors(m, Zi, Xi); rawi = D.raw(Zi); Vi = D.resid_obs(rawi, D.B_terms(rawi))
            for score, k in (("prespecified", 3), ("max", 1)):
                p, v = streams_k(Ei, Vi, pos, refE, refV, k)
                if mname == "TailGuard": s = 0.5 * p + 0.5 * v if score == "prespecified" else np.maximum(p, v); tau = taus[score]["b0.5"]
                else: s = p; tau = taus[score]["b0.0"]
                a = s[valm] > tau; es = event_stats(a, evid[valm])
                kinds = {kd: event_stats(a, np.where(np.isin(evid[valm], [f"dev{i}" for i, e in enumerate(ev) if e["kind"] == kd]), evid[valm], ""))["detected"] for kd in ("bias", "freeze", "ramp")}
                R.append({"model": mname, "fault_set": sname, "score": score, "detected": es["detected"], "events": es["events"], **kinds})
                print(R[-1], flush=True)
            del Zi, Xi, Ei, Vi
        del E
    pd.DataFrame(R).to_csv(os.path.join(OUT, "dev_fault_sensitivity.csv"), index=False)

if MODE == "risk":
    import holdout_a7_runs as RR
    R = []
    for h in REG:
        D = RR.CleanData(h); hi = REG.index(h); n = len(D.Z)
        raw = D.raw(D.Z); Bt = D.B_terms(raw)
        Bt_g = [(ti, lag0, torch.as_tensor(B, dtype=torch.float32), a) for ti, lag0, B, a in Bt]
        sc_g, md_g = torch.as_tensor(D.scale, dtype=torch.float32), torch.as_tensor(D.med, dtype=torch.float32)
        groups = {"retained regimes, validation": D.elig & (D.part == "val") & (D.reg != hi),
                  "withheld regime, all normal data": D.elig & (D.reg == hi)}
        for tag, li, lp, inv, bal in (("clean_tailguard", 1.0, 0.1, "phys", "bal"), ("clean_gru", 0.0, 0.0, "phys", "pool"),
                                      ("clean_A2", 0.0, 0.1, "phys", "bal"), ("clean_A10", 1.0, 0.1, "forecast", "bal")):
            path = os.path.join(REV, rid(tag, 17, li, lp, inv, bal, f"_hold-{h}"), "model_best.pt")
            m = load_model(D, path)
            for gname, mk in groups.items():
                rows = np.where(mk & (D.pos >= 30))[0][::5]
                zhat = torch.as_tensor(predict(m, D.X, rows)); z = torch.as_tensor(D.Z[rows])
                fr = T.huber_rows(zhat - z).numpy()
                cr = T.huber_rows(T.phys_resid_torch(zhat, torch.as_tensor(rows), Bt_g, sc_g, md_g)).numpy()
                for rr in np.unique(D.reg[rows]):
                    sel = D.reg[rows] == rr
                    R.append({"withheld": h, "model": tag.replace("clean_", ""), "group": gname, "regime": REG[rr], "n_targets": int(sel.sum()),
                              "forecast_risk": float(np.nanmean(fr[sel])), "constraint_risk": float(np.nanmean(cr[sel]))})
                    print(R[-1], flush=True)
        del D
    pd.DataFrame(R).to_csv(os.path.join(OUT, "holdout_risks.csv"), index=False)
