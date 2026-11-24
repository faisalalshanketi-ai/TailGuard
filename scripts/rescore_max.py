"""
TailGuard | post-hoc scoring extension "max tail surprise" (frozen in FROZEN_VARIANT.md before any test evaluation).
Re-scores trained models without retraining: per-channel tail surprises against validation data, the single largest
channel (k = 1), fusion by maximum of forecast and constraint streams, 30 s causal mean, threshold on calibration data.
Writes runs_max/train/<run_id>/{summary.json, scores.npz} in the format read by step6_evaluate_max.py. No labels are read.
  python scripts/rescore_max.py --models <dir with run_id/model_best.pt + summary.json>
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import tailguard_train as T
from tailguard_lib import ecdf_ref, tail_surprise, causal_smooth, calibrate_threshold, alarm_stats, event_stats, apply_injections
K, SMOOTH, W_MAX = 1, 30, 120
OUT = os.environ.get("RESCORE_OUT", os.path.join(ROOT, "runs_max", "train"))


def stream(X, ref, pos):
    s = tail_surprise(X, ref).max(1); s[pos < W_MAX] = 0   # K = 1
    return causal_smooth(s, SMOOTH)


def rescore(rdir, D, G, V, Vte, Vi, evid, Xi_g, Zi_g):
    sm = json.load(open(os.path.join(rdir, "summary.json"))); cfg = sm["config"]; rid = cfg["id"]
    od = os.path.join(OUT, rid); os.makedirs(od, exist_ok=True)
    if os.path.exists(os.path.join(od, "summary.json")): return
    W = cfg["W"]; hold = cfg.get("holdout")
    keep = (D.reg != T.REGIMES.index(hold)) if hold else np.ones(len(D.reg), bool)
    m = T.GRUForecaster(D.nF, D.nT); m.load_state_dict(torch.load(os.path.join(rdir, "model_best.pt"), map_location="cpu")); m.eval()
    n = len(D.Z)
    E = T.predict_errors(m, G["X"], G["Z"], W, n); Ete = T.predict_errors(m, G["Xte"], G["Zte"], W, len(D.Zte))
    Ei = T.predict_errors(m, Xi_g, Zi_g, W, n)
    valm = D.elig & (D.part == "val") & keep; calm = D.elig & (D.part == "cal") & keep
    refE, refV = ecdf_ref(E[valm]), ecdf_ref(V[valm])
    p, pte, pi_ = stream(E, refE, D.pos), stream(Ete, refE, D.pos_te), stream(Ei, refE, D.pos)
    del E, Ete, Ei
    v, vte, vi_ = stream(V, refV, D.pos), stream(Vte, refV, D.pos_te), stream(Vi, refV, D.pos)
    out = {"config": cfg, "variant": "max tail surprise (k=1, max fusion, 30 s)", "tau": {}, "val_clean": {}, "dev": {}, "holdout": {}}
    devm = D.elig & (D.part == "val"); ev = evid[devm]
    for key, s, si in (("b0.0", p, pi_), ("b1.0", v, vi_), ("b0.5", np.maximum(p, v), np.maximum(pi_, vi_))):
        tau, _ = calibrate_threshold(s[calm], D.seg[calm], calm.sum() / 3600); out["tau"][key] = tau
        out["val_clean"][key] = alarm_stats(s[valm] > tau, D.seg[valm], np.ones(valm.sum(), bool))
        a = si[devm] > tau
        out["dev"][key] = {**event_stats(a, ev), "FA24": alarm_stats(a, D.seg[devm], ev == "")["FA24"]}
        if hold:
            hm = D.elig & ~keep; out["holdout"][key] = alarm_stats(s[hm] > tau, D.seg[hm], np.ones(hm.sum(), bool))
    np.savez_compressed(os.path.join(od, "scores.npz"), p_train=p.astype(np.float32), v_train=v.astype(np.float32),
                        p_test=pte.astype(np.float32), v_test=vte.astype(np.float32))
    json.dump(out, open(os.path.join(od, "summary.json"), "w"), indent=1, default=float)
    print(f"{rid}: dev max {out['dev']['b0.5']['detected']}/36 (paper scoring {sm['dev']['b0.5']['detected']}/36), "
          f"forecast-only {out['dev']['b0.0']['detected']}/36 (paper {sm['dev']['b0.0']['detected']}/36)", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    D = T.Data()
    G = {k: torch.as_tensor(v) for k, v in {"X": D.X, "Z": D.Z, "Xte": D.Xte, "Zte": D.Zte}.items()}
    raw, rawte = D.raw(D.Z), D.raw(D.Zte)
    V, Vte = D.resid_obs(raw, D.B_terms(raw)), D.resid_obs(rawte, D.B_terms(rawte))
    Zi, evid = apply_injections(D.Z, D.dev_events); Xi = D.X.copy(); Xi[:, :D.nT] = Zi
    rawi = D.raw(Zi); Vi = D.resid_obs(rawi, D.B_terms(rawi)); del raw, rawte, rawi
    Xi_g, Zi_g = torch.as_tensor(Xi), torch.as_tensor(Zi)
    runs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(a.models, "*", "model_best.pt")))
    print(len(runs), "trained runs"); t0 = time.time()
    for i, r in enumerate(runs):
        rescore(r, D, G, V, Vte, Vi, evid, Xi_g, Zi_g); print(f"  [{i+1}/{len(runs)}] {time.time()-t0:.0f}s", flush=True)
    sel = os.path.join(a.models, "selected.json")
    if os.path.exists(sel): os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "selected.json"), "w").write(open(sel).read())


if __name__ == "__main__":
    main()
