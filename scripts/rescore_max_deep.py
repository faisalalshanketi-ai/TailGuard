"""Deep baselines under the frozen "max tail surprise" scoring (k = 1, 30 s), from the per-channel errors saved by
step7_deep_rescore.py. Writes runs_max/deep/<MODEL>/s<seed>/{summary.json, scores.npz}. No labels are read."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from tailguard_lib import load, ecdf_ref, tail_surprise, causal_smooth, calibrate_threshold, alarm_stats, event_stats, apply_injections
W_MAX = 120

def stream(E, pos):
    return E

def main():
    tr, te = load("train"), load("test"); ids = tr["df_ids"]
    part, elig, seg, pos = ids["partition"].to_numpy(), ids["eligible"].to_numpy(), ids["segment_id"].to_numpy(), ids["pos_in_segment"].to_numpy()
    pos_te = te["df_ids"]["pos_in_segment"].to_numpy(); n = len(part); nT = tr["Z"].shape[1]
    valm, calm = elig & (part == "val"), elig & (part == "cal")
    dev = json.load(open(os.path.join(ROOT, "runs", "baselines", "dev_injections.json"))); _, evid = apply_injections(tr["Z"], dev)
    def pol(E, ref, p_):
        s = np.sort(tail_surprise(E, ref), 1)[:, -1:].mean(1); s[p_ < W_MAX] = 0; return causal_smooth(s, 30)
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "deep_rescore", "*", "s*", "chan.npz"))):
        rd = os.path.dirname(f); name = os.path.basename(os.path.dirname(rd)); seed = int(os.path.basename(rd)[1:])
        c = np.load(f); E = np.zeros((n, nT), np.float32); E[c["rows"]] = c["E"]
        Ei = np.zeros((n, nT), np.float32); Ei[valm] = c["Ei"]
        ref = ecdf_ref(E[valm]); s = pol(E, ref, pos); ste = pol(c["Ete"], ref, pos_te); si = pol(Ei, ref, pos)
        tau, _ = calibrate_threshold(s[calm], seg[calm], calm.sum() / 3600)
        a = si[valm] > tau; ev = evid[valm]
        out = {"seed": seed, "variant": "max tail surprise (k=1, 30 s)", "tau": float(tau),
               "val_clean": alarm_stats(s[valm] > tau, seg[valm], np.ones(valm.sum(), bool)),
               "dev": {**event_stats(a, ev), "FA24": alarm_stats(a, seg[valm], ev == "")["FA24"]}}
        od = os.path.join(ROOT, "runs_max", "deep", name, f"s{seed}"); os.makedirs(od, exist_ok=True)
        np.savez_compressed(os.path.join(od, "scores.npz"), s_train=s.astype(np.float32), s_test=ste.astype(np.float32))
        json.dump(out, open(os.path.join(od, "summary.json"), "w"), indent=1, default=float)
        orig = json.load(open(os.path.join(rd, "summary.json")))
        print(f"{name} s{seed}: dev max {out['dev']['detected']}/36 (paper scoring {orig['dev']['detected']}/36)")

if __name__ == "__main__":
    main()
