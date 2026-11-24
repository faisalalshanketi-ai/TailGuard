"""
TailGuard | Step 6 (post-hoc extension): ONE-TIME test evaluation of the frozen "max tail surprise" scoring.
  python scripts/step6_evaluate.py --unseal            (real labels; logged in manifests/unseal_log.txt)
  python scripts/step6_evaluate.py --fake-labels       (synthetic events; code test only, writes reports_max_FAKE/)
Reads frozen scores only (runs/baselines, runs/train, optional runs/deep). Nothing is re-tuned here.
Outputs: reports/ (results.json, table_*.csv, summary.md)
"""
import argparse, glob, json, os, sys, time
from itertools import product
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tailguard_lib import event_stats, alarm_stats, point_stats, episodes, causal_smooth

REG = ["low", "mid", "high"]; SEEDS = [17, 29, 43, 71, 101]


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unseal", action="store_true"); ap.add_argument("--fake-labels", action="store_true")
    ap.add_argument("--root", default=os.path.dirname(HERE)); ap.add_argument("--draws", type=int, default=10000)
    return ap.parse_args()


# ------------------------------------------------------------------ collection of frozen scores
def collect(root, te, tr_ids):
    """Return {method: [ {seed, s_test(full), tau, s_val(val eligible rows)} ]}"""
    M = {}
    valm = (tr_ids["eligible"] & (tr_ids["partition"] == "val")).to_numpy()
    for d in sorted(glob.glob(os.path.join(root, "runs_max", "baselines", "*", "summary.json"))):
        bd = os.path.dirname(d); name = os.path.basename(bd); sm = json.load(open(d))
        sc = pd.read_parquet(os.path.join(bd, "test_scores_SEALED.parquet"))
        s = np.full(len(te), -np.inf, np.float32)
        key = te.loc[te.eligible, ["file_id", "timestamp"]].reset_index()
        mm = key.merge(sc[["file_id", "timestamp", "score"]], on=["file_id", "timestamp"], how="left")
        s[mm["index"].to_numpy()] = mm["score"].to_numpy()
        sv = np.load(os.path.join(bd, "train_scores.npz"))["s_train"][valm] if os.path.exists(os.path.join(bd, "train_scores.npz")) else None
        M[name.split("_")[0]] = [{"seed": 17, "s_test": s, "tau": sm["tau"], "s_val": sv}]
    tdir = os.path.join(root, "runs_max", "train")
    sel = json.load(open(os.path.join(tdir, "selected.json"))) if os.path.exists(os.path.join(tdir, "selected.json")) else None

    def add(method, run_dir, beta, seed, holdout=None):
        f = os.path.join(run_dir, "summary.json")
        if not os.path.exists(f): return
        sm = json.load(open(f)); z = np.load(os.path.join(run_dir, "scores.npz"))
        fuse = (lambda p, v: np.maximum(p, v)) if beta == 0.5 else (lambda p, v: (1 - beta) * p + beta * v)
        s = fuse(z["p_test"], z["v_test"]); sv = fuse(z["p_train"], z["v_train"])[valm]
        M.setdefault(method, []).append({"seed": seed, "s_test": s.astype(np.float32), "tau": sm["tau"][f"b{beta}"],
                                         "s_val": sv, "holdout": holdout, "summary": sm})
    if sel:
        W, li, lp = sel["W"], sel["l_inv"], sel["l_phys"]
        core = lambda tag, s, li=li, lp=lp, inv="phys", env="regime", bal="bal", h="": \
            os.path.join(tdir, f"{tag}__W{W}_inv{li}_phys{lp}_{inv}_{env}_{bal}_s{s}{h}")
        for s in SEEDS:
            main = core("search", 17) if s == 17 else core("tailguard", s)
            add("TailGuard", main, 0.5, s)
            add("A4_beta0 (temporal score only)", main, 0.0, s)
            add("Residual-only (beta=1)", main, 1.0, s)
            add("GRU (pooled ERM)", core("gru_erm", s, 0.0, 0.0, bal="pool"), 0.0, s)
            add("A1_no_physics", core("A1_nophys", s, li, 0.0), 0.0, s)
            add("A2_no_invariance", core("A2_noinv", s, 0.0, lp), 0.5, s)
            add("A3_pooled_no_inv", core("A3_pooled", s, 0.0, lp, bal="pool"), 0.5, s)
            add("A10_whole_forecast_VREx", core("A10_vrex", s, max(li, 0.1), lp, inv="forecast"), 0.5, s)
            add("A11_day_environments", core("A11_dayenv", s, li, lp, env="day"), 0.5, s)
            for h in REG:
                add(f"HOLD-{h} TailGuard", core("hold_tailguard", s, h=f"_hold-{h}"), 0.5, s, h)
                add(f"HOLD-{h} GRU", core("hold_gru", s, 0.0, 0.0, bal="pool", h=f"_hold-{h}"), 0.0, s, h)
    for d in sorted(glob.glob(os.path.join(root, "runs_max", "deep", "*", "s*", "summary.json"))):   # USAD/TranAD/GDN (later)
        rd = os.path.dirname(d); name = os.path.basename(os.path.dirname(rd)); sm = json.load(open(d))
        z = np.load(os.path.join(rd, "scores.npz"))
        M.setdefault(name, []).append({"seed": sm.get("seed", 0), "s_test": z["s_test"], "tau": sm["tau"],
                                       "s_val": z["s_train"][valm] if "s_train" in z else None})
    return M, sel


# ------------------------------------------------------------------ metrics
def common_metrics(s, tau, lab, te, mask):
    a = (s > tau) & mask; el = mask
    ev = np.where(el, lab["event_id"].to_numpy(), ""); y = lab["label"].to_numpy().astype(bool)
    seg = te["segment_id"].to_numpy()
    out = event_stats(a[el], ev[el])
    out.update(alarm_stats(a[el], seg[el], ~y[el]))
    sc = np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]))
    out.update(point_stats(a[el], y[el], sc[el]))
    return out


def agg(rows, keys):
    df = pd.DataFrame(rows)
    return {k: (float(df[k].mean()), float(df[k].std(ddof=1)) if len(df) > 1 else 0.0) for k in keys if k in df}


def blocks_of(te):
    t = pd.to_datetime(te["timestamp"])
    start = t.groupby(te["file_id"]).transform("min")
    return (te["file_id"] + "#" + ((t - start).dt.total_seconds() // (6 * 3600)).astype(int).astype(str)).to_numpy()


def block_counts(s, tau, lab, te, mask, blk, ublk):
    a = (s > tau) & mask; y = lab["label"].to_numpy().astype(bool); ev = lab["event_id"].to_numpy()
    seg = te["segment_id"].to_numpy(); st = np.zeros(len(a), bool); st[mask] = episodes(a[mask], seg[mask])
    det = np.zeros(len(ublk)); tot = np.zeros(len(ublk)); fe = np.zeros(len(ublk)); ns = np.zeros(len(ublk))
    bi = {b: i for i, b in enumerate(ublk)}
    for e in pd.unique(ev[mask & y]):
        if not e: continue
        idx = np.where((ev == e) & mask)[0]; k = bi[blk[idx[0]]]
        tot[k] += 1; det[k] += a[idx].any()
    for k, b in enumerate(ublk):
        m = (blk == b) & mask & ~y
        fe[k] = (st & m).sum(); ns[k] = m.sum()
    return det, tot, fe, ns


def paired(A, B, draws, rng):
    """A, B: lists over seeds of (det, tot, fe, ns) per block. Returns CI of recall and FA24 differences + sign-flip p."""
    nb = len(A[0][0]); idx = rng.integers(0, nb, (draws, nb))
    def stat(L, ii):
        rec = np.mean([L_[0][ii].sum(1) / np.maximum(L_[1][ii].sum(1), 1) for L_ in L], 0)
        fa = np.mean([24 * L_[2][ii].sum(1) / np.maximum(L_[3][ii].sum(1) / 3600, 1e-9) for L_ in L], 0)
        return rec, fa
    ra, fa_a = stat(A, idx); rb, fa_b = stat(B, idx)
    drec, dfa = ra - rb, fa_a - fa_b
    # sign-flip randomisation on per-block differences (seed-averaged)
    def per_block(L):
        det = np.mean([l[0] for l in L], 0); tot = np.mean([l[1] for l in L], 0)
        rate = np.mean([24 * l[2] / np.maximum(l[3] / 3600, 1e-9) for l in L], 0); return det, tot, rate
    da, ta, rta = per_block(A); db, tb, rtb = per_block(B)
    d_rec = da - db; d_fa = rta - rtb
    signs = np.array(list(product([1, -1], repeat=nb))) if nb <= 16 else rng.choice([1, -1], (draws, nb))
    def p_of(d):
        obs = abs(d.sum()); null = np.abs(signs @ d); return float((null >= obs - 1e-12).mean())
    return {"recall_diff": float(np.mean(drec)), "recall_CI95": np.percentile(drec, [2.5, 97.5]).tolist(),
            "FA24_diff": float(np.mean(dfa)), "FA24_CI95": np.percentile(dfa, [2.5, 97.5]).tolist(),
            "p_recall": p_of(d_rec), "p_FA24": p_of(d_fa), "n_blocks": nb}


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()


# ------------------------------------------------------------------ main
def main():
    a = args(); root = a.root
    if not (a.unseal or a.fake_labels):
        sys.exit("refusing to read test labels: pass --unseal (one-time) or --fake-labels (code test)")
    te = pd.read_parquet(os.path.join(root, "data", "derived", os.environ.get("TAILGUARD_PREFIX", "hai2305") + "_test.parquet"),
                         columns=["timestamp", "file_id", "segment_id", "pos_in_segment", "eligible", "regime"])
    tr_ids = pd.read_parquet(os.path.join(root, "data", "derived", os.environ.get("TAILGUARD_PREFIX", "hai2305") + "_train.parquet"),
                             columns=["partition", "eligible", "regime"])
    rng = np.random.default_rng(20261011)
    if a.fake_labels:
        out_dir = os.path.join(root, "reports_max_FAKE"); y = np.zeros(len(te), np.int8); ev = np.full(len(te), "", object)
        starts = np.sort(rng.choice(np.where(te.eligible)[0][:-700], 40, replace=False))
        for k, s0 in enumerate(starts):
            L = int(rng.integers(60, 600)); y[s0:s0 + L] = 1; ev[s0:s0 + L] = f"fake{k}"
        lab = pd.DataFrame({"label": y, "event_id": ev})
    else:
        out_dir = os.path.join(root, "reports_max")
        lab = pd.read_parquet(os.path.join(root, "data", "derived", os.environ.get("TAILGUARD_PREFIX", "hai2305") + "_test_labels.parquet"))
        assert len(lab) == len(te) and (lab["timestamp"].values == te["timestamp"].values).all()
        with open(os.path.join(root, "manifests", "unseal_log.txt"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} test labels read by step6_evaluate_max.py (post-hoc max-tail-surprise extension, frozen variant)\n")
    os.makedirs(out_dir, exist_ok=True)
    M, sel = collect(root, te, tr_ids)
    print("methods found:", {k: len(v) for k, v in M.items()})
    mask = te["eligible"].to_numpy()
    keys = ["event_recall", "FA24", "false_alarm_sec_per_hour", "normal_FPR", "precision", "recall", "F1", "AP",
            "AUROC", "delay_median_s", "capped_delay_median_s"]

    # ---- Table 6: common protocol (both official test files)
    common, per_seed = {}, {}
    for m, runs in M.items():
        if m.startswith("HOLD-"): continue
        rows = [common_metrics(r["s_test"], r["tau"], lab, te, mask) for r in runs]
        per_seed[m] = rows; common[m] = {"n_seeds": len(rows), **agg(rows, keys)}
    pd.DataFrame({m: {k: f"{v[0]:.3f} ± {v[1]:.3f}" if isinstance(v, tuple) else v for k, v in d.items()}
                  for m, d in common.items()}).T.to_csv(os.path.join(out_dir, "table_common_protocol.csv"))

    # ---- Table 5: matched published protocols on HAI 23.05 Test 1
    t1 = mask & (te["file_id"].to_numpy() == "hai-test1")
    if os.environ.get("TAILGUARD_PREFIX", "hai2305") != "hai2305": t1 = mask.copy()   # no published Test-1 protocol: point metrics on all test files
    y = lab["label"].to_numpy().astype(bool)
    pub = {}
    for m, runs in M.items():
        if m.startswith("HOLD-"): continue
        r14a, r14b = [], []
        for r in runs:
            s = r["s_test"]; sc = np.where(np.isfinite(s), s, -1e9)
            if r.get("s_val") is not None:
                tv = np.percentile(r["s_val"], 95); al = sc > tv
                ps = point_stats(al[t1], y[t1], sc[t1]); ps["FPR"] = float((al[t1] & ~y[t1]).sum() / (~y[t1]).sum())
                r14a.append(ps)
            tt = np.percentile(sc[t1 & ~y], 95); r14b.append(point_stats(sc[t1] > tt, y[t1], sc[t1]))
        pub[m] = {"[14] p95-validation threshold, Test 1": agg(r14a, ["F1", "AP", "FPR", "precision", "recall"]) if r14a else None,
                  "[14]/[15] fixed 5% FPR on test-normal, Test 1": agg(r14b, ["F1", "precision", "recall"])}
    published = {"[14] p95-validation threshold, Test 1": {"best published (GNN)": {"F1": 0.554, "AP": 0.572, "FPR": 0.030}},
                 "[14]/[15] fixed 5% FPR on test-normal, Test 1": {"[14] GNN+BiLSTM": {"F1": 0.530}, "[15] GNN+BiLSTM": {"F1": 0.498}, "[15] USAD": {"F1": 0.419}}}
    rows = []
    for proto, refs in published.items():
        for ref, vals in refs.items():
            rows.append({"protocol": proto, "method": ref, **{k: f"{v:.3f}" for k, v in vals.items()}})
        for m in pub:
            if pub[m][proto]:
                rows.append({"protocol": proto, "method": m, **{k: f"{v[0]:.3f} ± {v[1]:.3f}" for k, v in pub[m][proto].items()}})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "table_published_protocols.csv"), index=False)

    # ---- regime hold-outs (held-out regime never seen in fitting, validation or calibration)
    reg = te["regime"].to_numpy(); ev = lab["event_id"].to_numpy(); hold = []
    for h_i, h in enumerate(REG):
        for kind in ("TailGuard", "GRU"):
            runs = M.get(f"HOLD-{h} {kind}", [])
            if not runs: continue
            rr = []
            for r in runs:
                a_ = (r["s_test"] > r["tau"]) & mask
                nm = mask & (reg == h_i) & ~y
                onset = {e for e in pd.unique(ev[mask & y]) if e and reg[np.where(ev == e)[0][0]] == h_i}
                evh = np.where(np.isin(ev, list(onset)) & mask, ev, "")
                st_ = np.zeros(len(a_), bool); st_[mask] = episodes(a_[mask], te["segment_id"].to_numpy()[mask])   # episodes over the whole sequence, as in FA24
                rr.append({"test_FA24_heldout_regime": 24 * (st_ & nm).sum() / (nm.sum() / 3600),
                           "event_recall_heldout_regime": event_stats(a_[mask], evh[mask])["event_recall"] if onset else np.nan,
                           "train_normal_FA24_heldout": r["summary"]["holdout"]["b0.5" if kind == "TailGuard" else "b0.0"]["FA24"]})
            hold.append({"regime": h, "method": kind, "n_seeds": len(rr),
                         **{k: f"{v[0]:.3f} ± {v[1]:.3f}" for k, v in agg(rr, list(rr[0].keys())).items()}})
    pd.DataFrame(hold).to_csv(os.path.join(out_dir, "table_regime_holdout.csv"), index=False)

    # ---- paired block bootstrap + sign-flip, Holm within dataset across comparators x 2 endpoints
    blk = blocks_of(te); ublk = list(pd.unique(blk[mask]))
    if "TailGuard" in M:
        BC = {m: [block_counts(r["s_test"], r["tau"], lab, te, mask, blk, ublk) for r in runs] for m, runs in M.items()
              if not m.startswith("HOLD-") and not m.startswith("A")}
        comps = [m for m in ("GRU (pooled ERM)", "Residual-only (beta=1)", "GDN", "USAD", "TranAD", "ridge", "pca", "iforest", "persistence") if m in BC]
        res = {c: paired(BC["TailGuard"], BC[c], a.draws, rng) for c in comps}
        fam = [c for c in comps if c in ("GRU (pooled ERM)", "Residual-only (beta=1)", "GDN", "USAD", "TranAD")]
        ps = [res[c]["p_recall"] for c in fam] + [res[c]["p_FA24"] for c in fam]
        adj = holm(ps) if ps else []
        for i, c in enumerate(fam):
            res[c]["p_recall_holm"], res[c]["p_FA24_holm"] = adj[i], adj[i + len(fam)]
        pd.DataFrame(res).T.to_csv(os.path.join(out_dir, "table_paired_vs_tailguard.csv"))
    else:
        res = {}
    json.dump({"selected": sel, "common": common, "published_protocols": pub, "regime_holdout": hold, "paired": res,
               "labels": "FAKE" if a.fake_labels else "official"}, open(os.path.join(out_dir, "results.json"), "w"),
              indent=1, default=float)
    print("written:", out_dir)
    for m, d in common.items():
        print(f"  {m:32s} recall {d['event_recall'][0]:.3f}  FA24 {d['FA24'][0]:.2f}  F1 {d['F1'][0]:.3f}  ({d['n_seeds']} seed(s))")


if __name__ == "__main__":
    main()
