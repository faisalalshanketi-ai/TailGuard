"""
TailGuard | analyses of the frozen score streams (no retraining; reads frozen score streams and the official test labels).
  python frozen_score_analyses.py --root <repo root with runs/ and runs_max/> --out <dir>
Writes: budget_sensitivity.csv, matched_burden.csv, block_length.csv, regime_counts.csv, recalibration.csv,
        attacks_detection.csv, duration_aware.csv, dev_injections_spec.csv
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--draws", type=int, default=10000); A = ap.parse_args()
ROOT = A.root; sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tailguard_lib import calibrate_threshold, alarm_stats, event_stats, episodes
import step6_evaluate as S6
os.makedirs(A.out, exist_ok=True)
SEEDS = [17, 29, 43, 71, 101]; REG = ["low", "mid", "high"]
te = pd.read_parquet(os.path.join(ROOT, "data/derived/hai2305_test.parquet"), columns=["timestamp", "file_id", "segment_id", "pos_in_segment", "eligible", "regime"])
tr = pd.read_parquet(os.path.join(ROOT, "data/derived/hai2305_train.parquet"), columns=["partition", "eligible", "regime", "segment_id"])
lab = pd.read_parquet(os.path.join(ROOT, "data/derived/hai2305_test_labels.parquet"))
with open(os.path.join(ROOT, "manifests", "unseal_log.txt"), "a") as f:
    f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " test labels read by frozen_score_analyses.py (descriptive analyses of frozen scores)\n")
mask = te["eligible"].to_numpy(); y = lab["label"].to_numpy().astype(bool); ev = lab["event_id"].to_numpy()
seg_te = te["segment_id"].to_numpy(); reg_te = te["regime"].to_numpy()
part = tr["partition"].to_numpy(); elig = tr["eligible"].to_numpy(); reg = tr["regime"].to_numpy(); seg = tr["segment_id"].to_numpy()
calm = elig & (part == "cal"); valm = elig & (part == "val")
sel = json.load(open(os.path.join(ROOT, "runs", "train", "selected.json"))); W, li, lp = sel["W"], sel["l_inv"], sel["l_phys"]
def rid(tag, s, li=li, lp=lp, inv="phys", env="regime", bal="bal", h=""):
    return f"{tag}__W{W}_inv{li}_phys{lp}_{inv}_{env}_{bal}_s{s}{h}"
def gru_streams(tdir, run, fuse):
    z = np.load(os.path.join(tdir, run, "scores.npz"))
    f = {"mean0.5": lambda p, v: 0.5 * p + 0.5 * v, "max": np.maximum, "p": lambda p, v: p, "v": lambda p, v: v}[fuse]
    return f(z["p_train"], z["v_train"]), f(z["p_test"], z["v_test"])
def baseline_streams(bdir, name):
    sc = pd.read_parquet(os.path.join(bdir, name, "test_scores_SEALED.parquet"))
    s = np.full(len(te), -np.inf, np.float32); key = te.loc[te.eligible, ["file_id", "timestamp"]].reset_index()
    mm = key.merge(sc[["file_id", "timestamp", "score"]], on=["file_id", "timestamp"], how="left"); s[mm["index"].to_numpy()] = mm["score"].to_numpy()
    return np.load(os.path.join(bdir, name, "train_scores.npz"))["s_train"], s
def deep_streams(ddir, m, s):
    z = np.load(os.path.join(ddir, m, f"s{s}", "scores.npz")); return z["s_train"], z["s_test"]
# ---------------- method registry: name -> list of (s_train, s_test) over seeds
T0, TM = os.path.join(ROOT, "runs", "train"), os.path.join(ROOT, "runs_max", "train")
M = {}
for s in SEEDS:
    main = rid("search", 17) if s == 17 else rid("tailguard", s)
    M.setdefault("TailGuard", []).append(gru_streams(T0, main, "mean0.5"))
    M.setdefault("TailGuard (A4, forecast score)", []).append(gru_streams(T0, main, "p"))
    M.setdefault("Matched GRU", []).append(gru_streams(T0, rid("gru_erm", s, 0.0, 0.0, bal="pool"), "p"))
    M.setdefault("A1 (no constraint term)", []).append(gru_streams(T0, rid("A1_nophys", s, li, 0.0), "p"))
    M.setdefault("TailGuard, max score", []).append(gru_streams(TM, main, "max"))
    M.setdefault("Matched GRU, max score", []).append(gru_streams(TM, rid("gru_erm", s, 0.0, 0.0, bal="pool"), "p"))
    M.setdefault("A1, max score", []).append(gru_streams(TM, rid("A1_nophys", s, li, 0.0), "p"))
    for m in ("USAD", "TranAD", "GDN"):
        M.setdefault(m, []).append(deep_streams(os.path.join(ROOT, "runs", "deep"), m, s))
        M.setdefault(m + ", max score", []).append(deep_streams(os.path.join(ROOT, "runs_max", "deep"), m, s))
for name, d in (("Ridge AR", "ridge_a1.0"), ("PCA", "pca_k20"), ("Persistence", "persistence"), ("Isolation Forest", "iforest")):
    M[name] = [baseline_streams(os.path.join(ROOT, "runs", "baselines"), d)]
hours_cal = calm.sum() / 3600
def at_budget(st, ss, B):
    tau, _ = calibrate_threshold(st[calm], seg[calm], hours_cal, budget_per_24h=B)
    a = (ss > tau) & mask; e = np.where(mask, ev, "")
    r = event_stats(a[mask], e[mask]); fa = alarm_stats(a[mask], seg_te[mask], ~y[mask])
    return {"event_recall": r["event_recall"], "detected": r["detected"], "FA24": fa["FA24"], "false_episodes": fa["false_episodes"],
            "false_alarm_sec_per_hour": fa["false_alarm_sec_per_hour"], "tau": tau}
# ---------------- 1. alarm-budget sensitivity (A9) and matched realised burden
rows = []
for name, runs in M.items():
    for B in (0.25, 0.5, 1, 2, 4, 8):
        rr = [at_budget(st, ss, B) for st, ss in runs]
        d = {k: float(np.mean([r[k] for r in rr])) for k in ("event_recall", "detected", "FA24", "false_episodes", "false_alarm_sec_per_hour")}
        d.update({k + "_sd": float(np.std([r[k] for r in rr], ddof=1)) if len(rr) > 1 else 0.0 for k in ("event_recall", "FA24")})
        rows.append({"method": name, "budget_per_day": B, "n_seeds": len(rr), **d})
bs = pd.DataFrame(rows); bs.to_csv(os.path.join(A.out, "budget_sensitivity.csv"), index=False)
mb = []
for name, g in bs.groupby("method"):
    g = g.sort_values("FA24"); x, r = g["FA24"].to_numpy(), g["event_recall"].to_numpy()
    for target in (0.5, 1.0, 1.5, 2.0):
        mb.append({"method": name, "realised_FA24": target, "event_recall_interp": float(np.interp(target, x, r)) if x.min() <= target <= x.max() else np.nan})
pd.DataFrame(mb).pivot(index="method", columns="realised_FA24", values="event_recall_interp").to_csv(os.path.join(A.out, "matched_burden.csv"))
# ---------------- 2. block-length sensitivity of paired tests (calibrated budget 1/day)
def blocks(hours):
    t = pd.to_datetime(te["timestamp"]); start = t.groupby(te["file_id"]).transform("min")
    return (te["file_id"] + "#" + ((t - start).dt.total_seconds() // (hours * 3600)).astype(int).astype(str)).to_numpy()
def tau_of(st): return calibrate_threshold(st[calm], seg[calm], hours_cal)[0]
rng = np.random.default_rng(20261022); bl = []
pairs = [("TailGuard", c) for c in ("Matched GRU", "USAD", "TranAD", "GDN", "Ridge AR")] + \
        [("TailGuard, max score", c) for c in ("Matched GRU, max score", "USAD, max score", "TranAD, max score", "GDN, max score", "Ridge AR")] + \
        [("TailGuard (A4, forecast score)", "A1 (no constraint term)"), ("TailGuard, max score", "A1, max score")]
for H in (3, 6, 12):
    blk = blocks(H); ub = list(pd.unique(blk[mask]))
    BC = {m: [S6.block_counts(ss, tau_of(st), lab, te, mask, blk, ub) for st, ss in M[m]] for m in set(sum(([a, b] for a, b in pairs), []))}
    for a, b in pairs:
        r = S6.paired(BC[a], BC[b], A.draws, rng)
        bl.append({"block_hours": H, "n_blocks": r["n_blocks"], "comparison": f"{a} − {b}", "recall_diff": r["recall_diff"],
                   "CI_low": r["recall_CI95"][0], "CI_high": r["recall_CI95"][1], "p_signflip": r["p_recall"],
                   "FA24_diff": r["FA24_diff"], "FA_CI_low": r["FA24_CI95"][0], "FA_CI_high": r["FA24_CI95"][1]})
pd.DataFrame(bl).to_csv(os.path.join(A.out, "block_length.csv"), index=False)
# ---------------- 3. regime hold-out raw counts and 4. recalibration from withheld-regime normal data
rc, rec = [], []
for h_i, h in enumerate(REG):
    nm = mask & (reg_te == h_i) & ~y; nh = nm.sum() / 3600
    onset = sorted({e for e in pd.unique(ev[mask & y]) if e and reg_te[np.where(ev == e)[0][0]] == h_i})
    evh = np.where(np.isin(ev, onset) & mask, ev, "")
    pool = np.where(elig & (reg == h_i))[0]                       # withheld-regime normal training rows, chronological
    for kind, tag, fuse in (("TailGuard", "hold_tailguard", "mean0.5"), ("GRU", "hold_gru", "p")):
        for s in SEEDS:
            run = rid(tag, s, h=f"_hold-{h}") if kind == "TailGuard" else rid(tag, s, 0.0, 0.0, bal="pool", h=f"_hold-{h}")
            st, ss = gru_streams(T0, run, fuse)
            keep = reg != h_i; cm = calm & keep
            tau = calibrate_threshold(st[cm], seg[cm], cm.sum() / 3600)[0]
            a = (ss > tau) & mask
            fa = alarm_stats(a[nm], seg_te[nm], np.ones(nm.sum(), bool)); es = event_stats(a[mask], evh[mask])
            rc.append({"regime": h, "method": kind, "seed": s, "normal_test_hours": round(nh, 2), "false_episodes": fa["false_episodes"],
                       "FA24": fa["FA24"], "attacks_in_regime": len(onset), "detected": es["detected"]})
            for N in (1, 2, 4, 8, 16, 32):
                rows_ = pool[: N * 3600]
                if len(rows_) < N * 3600: continue
                th = calibrate_threshold(st[rows_], seg[rows_], len(rows_) / 3600)[0]
                thr = np.where(reg_te == h_i, th, tau); a2 = (ss > thr) & mask
                fa2 = alarm_stats(a2[nm], seg_te[nm], np.ones(nm.sum(), bool)); es2 = event_stats(a2[mask], evh[mask])
                rec.append({"regime": h, "method": kind, "seed": s, "recal_hours": N, "FA24": fa2["FA24"], "false_episodes": fa2["false_episodes"],
                            "detected": es2["detected"], "attacks_in_regime": len(onset)})
pd.DataFrame(rc).to_csv(os.path.join(A.out, "regime_counts.csv"), index=False)
pd.DataFrame(rec).to_csv(os.path.join(A.out, "recalibration.csv"), index=False)
# ---------------- 5. per-attack detection, missed attacks and duration-aware metrics (budget 1/day)
atk = [e for e in pd.unique(ev[mask & y]) if e]
info = []
for e in atk:
    idx = np.where((ev == e) & mask)[0]
    info.append({"event": e, "file": te["file_id"].iloc[idx[0]], "regime": REG[reg_te[idx[0]]], "duration_s": len(idx)})
AT = pd.DataFrame(info).set_index("event"); da = []
for name in ("TailGuard", "TailGuard, max score", "Matched GRU", "USAD", "TranAD", "GDN", "Ridge AR"):
    det = np.zeros(len(atk)); first = {e: [] for e in atk}; cover = []
    for st, ss in M[name]:
        a = (ss > tau_of(st)) & mask
        for k, e in enumerate(atk):
            idx = np.where((ev == e) & mask)[0]; hit = np.where(a[idx])[0]
            if len(hit): det[k] += 1; first[e].append(hit[0])
            cover.append(a[idx].mean())
    AT[name + " (seeds detecting)"] = det
    n = len(M[name]); delays = np.concatenate([np.array(v, float) for v in first.values()]) if any(first.values()) else np.array([])
    durs = AT["duration_s"].to_numpy()
    capped = np.concatenate([[min(np.mean(first[e]) if first[e] else d, d) for e, d in zip(atk, durs)]])
    da.append({"method": name, "event_recall": det.sum() / (n * len(atk)), "frac_attack_time_alarmed": float(np.mean(cover)),
               "detected_within_60s": sum(sum(1 for x in first[e] if x <= 60) for e in atk) / (n * len(atk)),
               "detected_within_300s": sum(sum(1 for x in first[e] if x <= 300) for e in atk) / (n * len(atk)),
               "median_delay_detected_s": float(np.median(delays)) if len(delays) else np.nan,
               "median_capped_delay_s": float(np.median(capped))})
AT.to_csv(os.path.join(A.out, "attacks_detection.csv")); pd.DataFrame(da).to_csv(os.path.join(A.out, "duration_aware.csv"), index=False)
# ---------------- 6. development-fault specification
targets = json.load(open(os.path.join(ROOT, "manifests", "split_manifest.json")))["feature_roles"]["continuous_targets"] if os.path.exists(os.path.join(ROOT, "manifests", "split_manifest.json")) else None
dev = json.load(open(os.path.join(ROOT, "runs", "baselines", "dev_injections.json")))
pd.DataFrame([{"fault": f"dev{k}", "kind": e["kind"], "target_index": e["tag"], "target_tag": (targets[e["tag"]] if targets else ""),
               "start_row": e["start"], "duration_s": e["end"] - e["start"], "magnitude_robust_scales": e["mag"] if e["kind"] != "freeze" else 0.0,
               "regime": REG[reg[e["start"]]]} for k, e in enumerate(dev)]).to_csv(os.path.join(A.out, "dev_injections_spec.csv"), index=False)
print("written", A.out)
