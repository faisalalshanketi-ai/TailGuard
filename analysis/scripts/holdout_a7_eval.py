"""
TailGuard | evaluation of the refitted regime hold-out (H2 controls) and A7 random-constraint runs.
  python holdout_a7_eval.py --root <repo root> --runs <runs/holdout_a7 dir> --out <dir>
Reads frozen score streams and the official test labels; thresholds are those calibrated in each run
(retained-regime calibration data for the hold-out runs).
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--runs", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--draws", type=int, default=10000); A = ap.parse_args()
ROOT = A.root; sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tailguard_lib import calibrate_threshold, alarm_stats, event_stats, episodes
import step6_evaluate as S6
os.makedirs(A.out, exist_ok=True)
SEEDS = [17, 29, 43, 71, 101]; REG = ["low", "mid", "high"]
te = pd.read_parquet(os.path.join(ROOT, "data/derived/hai2305_test.parquet"), columns=["timestamp", "file_id", "segment_id", "eligible", "regime"])
tr = pd.read_parquet(os.path.join(ROOT, "data/derived/hai2305_train.parquet"), columns=["partition", "eligible", "regime", "segment_id"])
lab = pd.read_parquet(os.path.join(ROOT, "data/derived/hai2305_test_labels.parquet"))
mask = te["eligible"].to_numpy(); y = lab["label"].to_numpy().astype(bool); ev = lab["event_id"].to_numpy()
seg_te = te["segment_id"].to_numpy(); reg_te = te["regime"].to_numpy()
part = tr["partition"].to_numpy(); elig = tr["eligible"].to_numpy(); reg = tr["regime"].to_numpy(); seg = tr["segment_id"].to_numpy()
calm = elig & (part == "cal")
# fusion used by each arm: GRU has no constraint stream in its score (forecast only); the others fuse both streams
FUSE = {"clean_tailguard": "b0.5", "clean_gru": "b0.0", "clean_A2": "b0.5", "clean_A10": "b0.5", "A7_random": "b0.5"}
NAME = {"clean_tailguard": "TailGuard", "clean_gru": "Matched GRU", "clean_A2": "A2 (no invariance)", "clean_A10": "A10 (invariance on forecast risk)",
        "A7_random": "A7 (random constraint tags)"}
runs = sorted(d for d in os.listdir(A.runs) if os.path.isdir(os.path.join(A.runs, d)))
def load(run, score):
    sm = json.load(open(os.path.join(A.runs, run, "summary.json"))); z = np.load(os.path.join(A.runs, run, "scores.npz"))
    tag = run.split("__")[0]; key = ("max_" if score == "max" else "") + FUSE[tag]
    pre = "max_" if score == "max" else ""
    if key.endswith("b0.0"): st, ss = z[pre + "p_train"], z[pre + "p_test"]
    elif score == "max": st, ss = np.maximum(z["max_p_train"], z["max_v_train"]), np.maximum(z["max_p_test"], z["max_v_test"])
    else: st, ss = 0.5 * z["p_train"] + 0.5 * z["v_train"], 0.5 * z["p_test"] + 0.5 * z["v_test"]
    return sm, key, st, ss
# ---------------- clean regime hold-out
rows = []; streams = {}
for run in runs:
    if not run.startswith("clean_"): continue
    tag = run.split("__")[0]; s = int(run.split("_s")[-1].split("_")[0]); h = run.split("hold-")[1]; h_i = REG.index(h)
    for score in ("prespecified", "max"):
        sm, key, st, ss = load(run, score); tau = sm["tau"][key]
        keep = reg != h_i; cm = calm & keep                       # sanity: recompute tau from the retained calibration rows
        tau_chk = calibrate_threshold(st[cm], seg[cm], cm.sum() / 3600)[0]
        pass  # tau taken from the run: its calibration set also excludes window-crossing rows moved to the withheld side
        nm = mask & (reg_te == h_i) & ~y
        onset = sorted({e for e in pd.unique(ev[mask & y]) if e and reg_te[np.where(ev == e)[0][0]] == h_i})
        evh = np.where(np.isin(ev, onset) & mask, ev, "")
        a = (ss > tau) & mask
        E_ = np.zeros(len(a), bool); E_[mask] = episodes(a[mask], seg_te[mask])   # main definition: episodes over the whole sequence
        fe = int((E_ & nm).sum()); fa = {"false_episodes": fe, "FA24": 24 * fe / (nm.sum() / 3600)}; es = event_stats(a[mask], evh[mask])
        hv = sm["holdout"][key]
        rows.append({"withheld": h, "method": NAME[tag], "score": score, "seed": s, "tau": tau,
                     "train_withheld_normal_hours": round(hv["normal_hours"], 2), "train_withheld_false_episodes": hv["false_episodes"],
                     "train_withheld_FA24": hv["FA24"], "test_withheld_normal_hours": round(nm.sum() / 3600, 2),
                     "test_withheld_false_episodes": fa["false_episodes"], "test_withheld_FA24": fa["FA24"],
                     "attacks_in_regime": len(onset), "detected": es["detected"],
                     "retained_val_FA24": sm["val_clean"][key]["FA24"], "dev_recall": sm["dev"][key]["event_recall"]})
        streams[(tag, score, h, s)] = (a, evh, nm)
df = pd.DataFrame(rows); df.to_csv(os.path.join(A.out, "clean_holdout_runs.csv"), index=False)
agg = df.groupby(["score", "withheld", "method"]).agg(
    train_false_episodes=("train_withheld_false_episodes", "mean"), train_FA24=("train_withheld_FA24", "mean"),
    test_false_episodes=("test_withheld_false_episodes", "mean"), test_FA24=("test_withheld_FA24", "mean"),
    attacks=("attacks_in_regime", "first"), detected=("detected", "mean"), detected_sd=("detected", "std"),
    train_hours=("train_withheld_normal_hours", "first"), test_hours=("test_withheld_normal_hours", "first")).reset_index()
agg.to_csv(os.path.join(A.out, "clean_holdout_summary.csv"), index=False)
pooled = df.groupby(["score", "method"]).apply(lambda g: pd.Series({
    "train_false_episodes": g.groupby("seed")["train_withheld_false_episodes"].sum().mean(),
    "train_hours": g.groupby("seed")["train_withheld_normal_hours"].sum().mean(),
    "test_false_episodes": g.groupby("seed")["test_withheld_false_episodes"].sum().mean(),
    "test_hours": g.groupby("seed")["test_withheld_normal_hours"].sum().mean(),
    "detected": g.groupby("seed")["detected"].sum().mean(), "attacks": g.groupby("seed")["attacks_in_regime"].sum().mean()})).reset_index()
pooled["train_FA24"] = 24 * pooled.train_false_episodes / pooled.train_hours; pooled["test_FA24"] = 24 * pooled.test_false_episodes / pooled.test_hours
pooled["recall"] = pooled.detected / pooled.attacks
pooled.to_csv(os.path.join(A.out, "clean_holdout_pooled.csv"), index=False)
# paired H2 tests on pooled withheld-regime data: unit = (regime, seed); seed-paired differences, sign-flip over seeds x regimes
rng = np.random.default_rng(20261022); h2 = []
for score in ("prespecified", "max"):
    for a_, b_ in (("clean_tailguard", "clean_A2"), ("clean_A10", "clean_A2"), ("clean_tailguard", "clean_gru"), ("clean_tailguard", "clean_A10")):
        dd = []
        for h in REG:
            for s in SEEDS:
                ra = df[(df.method == NAME[a_]) & (df.score == score) & (df.withheld == h) & (df.seed == s)].iloc[0]
                rb = df[(df.method == NAME[b_]) & (df.score == score) & (df.withheld == h) & (df.seed == s)].iloc[0]
                dd.append((ra.detected - rb.detected, ra.train_withheld_false_episodes - rb.train_withheld_false_episodes,
                           ra.test_withheld_false_episodes - rb.test_withheld_false_episodes))
        dd = np.array(dd, float); out = {"score": score, "comparison": f"{NAME[a_]} − {NAME[b_]}", "units": len(dd)}
        for j, nmk in enumerate(("detected", "train_false_episodes", "test_false_episodes")):
            d = dd[:, j]; obs = d.sum()
            flips = rng.choice([-1, 1], (20000, len(d))); null = (flips * d).sum(1)
            out[nmk + "_diff_total"] = obs; out[nmk + "_p_signflip"] = float((np.abs(null) >= abs(obs) - 1e-9).mean())
        h2.append(out)
pd.DataFrame(h2).to_csv(os.path.join(A.out, "clean_holdout_H2.csv"), index=False)
# ---------------- A7 vs TailGuard on the full test set (paired block bootstrap, 6 h blocks)
def blocks(hours):
    t = pd.to_datetime(te["timestamp"]); start = t.groupby(te["file_id"]).transform("min")
    return (te["file_id"] + "#" + ((t - start).dt.total_seconds() // (hours * 3600)).astype(int).astype(str)).to_numpy()
sel = json.load(open(os.path.join(ROOT, "runs", "train", "selected.json"))); W, li, lp = sel["W"], sel["l_inv"], sel["l_phys"]
def rid(tag, s): return f"{tag}__W{W}_inv{li}_phys{lp}_phys_regime_bal_s{s}"
blk = blocks(6); ub = list(pd.unique(blk[mask])); hours_cal = calm.sum() / 3600; a7 = []
for score, tdir in (("prespecified", os.path.join(ROOT, "runs", "train")), ("max", os.path.join(ROOT, "runs_max", "train"))):
    BS, BA, R = [], [], []
    for s in SEEDS:
        z = np.load(os.path.join(tdir, rid("search" if s == 17 else "tailguard", s), "scores.npz"))
        st, ss = ((0.5 * z["p_train"] + 0.5 * z["v_train"], 0.5 * z["p_test"] + 0.5 * z["v_test"]) if score == "prespecified"
                  else (np.maximum(z["p_train"], z["v_train"]), np.maximum(z["p_test"], z["v_test"])))
        tau = calibrate_threshold(st[calm], seg[calm], hours_cal)[0]; BS.append(S6.block_counts(ss, tau, lab, te, mask, blk, ub))
        sm, key, st7, ss7 = load(rid("A7_random", s), score); tau7 = sm["tau"][key]; BA.append(S6.block_counts(ss7, tau7, lab, te, mask, blk, ub))
        for nm_, (ss_, t_) in (("TailGuard", (ss, tau)), ("A7", (ss7, tau7))):
            a = (ss_ > t_) & mask; e = np.where(mask, ev, "")
            r = event_stats(a[mask], e[mask]); fa = alarm_stats(a[mask], seg_te[mask], ~y[mask])
            R.append({"score": score, "method": nm_, "seed": s, "event_recall": r["event_recall"], "FA24": fa["FA24"],
                      "dev_recall": sm["dev"][key]["event_recall"] if nm_ == "A7" else np.nan})
    r = S6.paired(BS, BA, A.draws, rng); Rd = pd.DataFrame(R)
    m = Rd.groupby("method")[["event_recall", "FA24"]].mean()
    a7.append({"score": score, "TailGuard_recall": m.loc["TailGuard", "event_recall"], "A7_recall": m.loc["A7", "event_recall"],
               "TailGuard_FA24": m.loc["TailGuard", "FA24"], "A7_FA24": m.loc["A7", "FA24"],
               "recall_diff": r["recall_diff"], "CI_low": r["recall_CI95"][0], "CI_high": r["recall_CI95"][1], "p_signflip": r["p_recall"],
               "FA24_diff": r["FA24_diff"], "FA_CI_low": r["FA24_CI95"][0], "FA_CI_high": r["FA24_CI95"][1],
               "A7_dev_recall": Rd[Rd.method == "A7"].dev_recall.mean()})
    Rd.to_csv(os.path.join(A.out, f"A7_runs_{score}.csv"), index=False)
pd.DataFrame(a7).to_csv(os.path.join(A.out, "A7_vs_tailguard.csv"), index=False)
regs = {}
for s in SEEDS:
    regs[s] = json.load(open(os.path.join(A.runs, rid("A7_random", s), "summary.json")))["random_registry"]
json.dump(regs, open(os.path.join(A.out, "A7_registries.json"), "w"), indent=1)
print("done")
