"""Classical baselines under the frozen "max tail surprise" scoring (k = 1, 30 s). Isolation Forest keeps its own score.
Writes runs_max/baselines/<method>/ in the format of step3. No labels are read."""
import argparse, json, os, shutil, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import step3_baselines as S3
from tailguard_lib import ecdf_ref, tail_surprise, causal_smooth, calibrate_threshold, alarm_stats, event_stats, apply_injections, load
OUT = os.path.join(ROOT, "runs_max", "baselines")

def pol(E, b):
    q = tail_surprise(E, b); return causal_smooth(np.sort(q, 1)[:, -1:].mean(1), 30)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None, help="subset, e.g. persistence"); opt = ap.parse_args()
    tr, te = load("train"), load("test"); Z, Zte = tr["Z"], te["Z"]
    X = np.hstack([tr["Z"], tr["C"], tr["D"]]); Xte = np.hstack([te["Z"], te["C"], te["D"]])
    ids, idte = tr["df_ids"], te["df_ids"]
    elig = ids["eligible"].to_numpy(); part = ids["partition"].to_numpy(); seg = ids["segment_id"].to_numpy()
    fitm, valm, calm = elig & (part == "fit"), elig & (part == "val"), elig & (part == "cal")
    dev = json.load(open(os.path.join(ROOT, "runs", "baselines", "dev_injections.json")))
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "runs", "baselines", "dev_injections.json"), OUT)
    shutil.copytree(os.path.join(ROOT, "runs", "baselines", "iforest"), os.path.join(OUT, "iforest"), dirs_exist_ok=True)
    for meth in (S3.Persistence(), S3.RidgeAR(1.0), S3.PCARecon(20)):
        if opt.only and meth.name not in opt.only: continue
        meth.fit(X, Z, fitm); E = np.abs(Z - meth.predict(X, Z)); b = ecdf_ref(E[valm]); s = pol(E, b)
        tau, _ = calibrate_threshold(s[calm], seg[calm], calm.sum() / 3600)
        Zi, evid = apply_injections(Z, dev); Xi = np.hstack([Zi, X[:, Z.shape[1]:]])
        si = pol(np.abs(Zi - meth.predict(Xi, Zi)), b); a = si[valm] > tau; ev = evid[valm]
        st = pol(np.abs(Zte - meth.predict(Xte, Zte)), b)
        d = os.path.join(OUT, meth.name); os.makedirs(d, exist_ok=True)
        elt = idte["eligible"].to_numpy(); o = idte.loc[elt, ["timestamp", "file_id", "segment_id"]].copy()
        o["score"] = st[elt].astype(np.float32); o["alarm"] = o["score"] > tau; o.to_parquet(os.path.join(d, "test_scores_SEALED.parquet"), index=False)
        np.savez_compressed(os.path.join(d, "train_scores.npz"), s_train=s.astype(np.float32))
        res = {"method": meth.name, "variant": "max tail surprise (k=1, 30 s)", "tau": tau,
               "val_normal": alarm_stats(s[valm] > tau, seg[valm], np.ones(valm.sum(), bool)),
               "dev_injection": {**event_stats(a, ev), "FA24": alarm_stats(a, seg[valm], ev == "")["FA24"]}}
        json.dump(res, open(os.path.join(d, "summary.json"), "w"), indent=1, default=float)
        print(f"{meth.name}: dev {res['dev_injection']['detected']}/36  val FA24 {res['val_normal']['FA24']:.2f}", flush=True)

if __name__ == "__main__":
    main()
