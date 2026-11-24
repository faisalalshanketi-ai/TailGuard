"""Block-bootstrap 95% intervals for absolute event recall and FA24 (6 h blocks; seeds averaged within each draw)."""
import os, sys, json, numpy as np, pandas as pd
ROOT, PREFIX, OUT = sys.argv[1:4]; os.environ["TAILGUARD_PREFIX"] = PREFIX; sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tailguard_lib import calibrate_threshold
import step6_evaluate as S6
te = pd.read_parquet(f"{ROOT}/data/derived/{PREFIX}_test.parquet", columns=["timestamp", "file_id", "segment_id", "eligible"])
tr = pd.read_parquet(f"{ROOT}/data/derived/{PREFIX}_train.parquet", columns=["partition", "eligible", "segment_id"])
lab = pd.read_parquet(f"{ROOT}/data/derived/{PREFIX}_test_labels.parquet"); mask = te.eligible.to_numpy()
calm = (tr.eligible & (tr.partition == "cal")).to_numpy(); seg = tr.segment_id.to_numpy()
t = pd.to_datetime(te.timestamp); start = t.groupby(te.file_id).transform("min")
blk = (te.file_id + "#" + ((t - start).dt.total_seconds() // (6 * 3600)).astype(int).astype(str)).to_numpy(); ub = list(pd.unique(blk[mask]))
def rid(tag, s, li=1.0, lp=0.1, bal="bal"): return f"{tag}__W30_inv{li}_phys{lp}_phys_regime_{bal}_s{s}"
M = {}
for score, tdir in (("prespecified", "runs/train"), ("max", "runs_max/train")):
    for name, f, fuse in (("TailGuard", lambda s: rid("search" if s == 17 else "tailguard", s), "both"), ("Matched GRU", lambda s: rid("gru_erm", s, 0.0, 0.0, "pool"), "p")):
        for s in (17, 29, 43, 71, 101):
            z = np.load(f"{ROOT}/{tdir}/{f(s)}/scores.npz")
            if fuse == "p": st, ss = z["p_train"], z["p_test"]
            elif score == "max": st, ss = np.maximum(z["p_train"], z["v_train"]), np.maximum(z["p_test"], z["v_test"])
            else: st, ss = .5 * z["p_train"] + .5 * z["v_train"], .5 * z["p_test"] + .5 * z["v_test"]
            tau = calibrate_threshold(st[calm], seg[calm], calm.sum() / 3600)[0]
            M.setdefault(f"{name} ({score})", []).append(S6.block_counts(ss, tau, lab, te, mask, blk, ub))
for m in ("USAD", "TranAD", "GDN"):
    for score, dd in (("prespecified", "runs/deep"), ("max", "runs_max/deep")):
        for d in sorted(os.listdir(f"{ROOT}/{dd}/{m}")):
            sm = json.load(open(f"{ROOT}/{dd}/{m}/{d}/summary.json")); z = np.load(f"{ROOT}/{dd}/{m}/{d}/scores.npz")
            M.setdefault(f"{m} ({score})", []).append(S6.block_counts(z["s_test"], sm["tau"], lab, te, mask, blk, ub))
for b in ("ridge_a1.0",):
    sm = json.load(open(f"{ROOT}/runs/baselines/{b}/summary.json")); sc = pd.read_parquet(f"{ROOT}/runs/baselines/{b}/test_scores_SEALED.parquet")
    s_ = np.full(len(te), -np.inf, np.float32); key = te.loc[te.eligible, ["file_id", "timestamp"]].reset_index()
    mm = key.merge(sc[["file_id", "timestamp", "score"]], on=["file_id", "timestamp"], how="left"); s_[mm["index"].to_numpy()] = mm["score"].to_numpy()
    M["Ridge AR"] = [S6.block_counts(s_, sm["tau"], lab, te, mask, blk, ub)]
rng = np.random.default_rng(20261025); R = []
for name, L in M.items():
    nb = len(L[0][0]); idx = rng.integers(0, nb, (10000, nb))
    rec = np.mean([l[0][idx].sum(1) / np.maximum(l[1][idx].sum(1), 1) for l in L], 0)
    fa = np.mean([24 * l[2][idx].sum(1) / np.maximum(l[3][idx].sum(1) / 3600, 1e-9) for l in L], 0)
    r0 = np.mean([l[0].sum() / l[1].sum() for l in L]); f0 = np.mean([24 * l[2].sum() / (l[3].sum() / 3600) for l in L])
    R.append({"dataset": PREFIX, "method": name, "n_blocks": nb, "event_recall": r0, "recall_CI_low": np.percentile(rec, 2.5), "recall_CI_high": np.percentile(rec, 97.5),
              "FA24": f0, "FA24_CI_low": np.percentile(fa, 2.5), "FA24_CI_high": np.percentile(fa, 97.5)})
pd.DataFrame(R).to_csv(OUT, index=False); print(pd.DataFrame(R).round(3).to_string())
