"""TailGuard | selection-free calibration subset for Proposition 5 (Section 4.7; no retraining, no test-label use beyond
the block evaluation already reported in Table 16).
The calibration partition is split in time order into C1 (first half) and C2 (second half).
 (1) Selection is repeated with every candidate's closure threshold computed on C1 only (development faults on validation).
 (2) The block-maximum level tau_k of Proposition 5 is computed on C2 only, which no selection step has seen.
  python split_cal.py --root <repo root> --out <dir>"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); A = ap.parse_args()
ROOT = os.path.abspath(A.root); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "analysis", "scripts"))
from tailguard_lib import load, manifest, ecdf_ref, tail_surprise, causal_smooth, apply_injections, calibrate_threshold, episodes
from constraints_np import constraint_matrix
from np_gru import load_state, forecast
os.makedirs(A.out, exist_ok=True); W_MAX, TOPK, SMOOTH = 120, 3, 30
tr = load("train"); m = manifest(); tg = tr["targets"]; sc = m["scaler_fit"]
scale = np.array([sc[c]["scale"] for c in tg]); med = np.array([sc[c]["median"] for c in tg])
ids = tr["df_ids"]; Z = tr["Z"]; X = np.hstack([tr["Z"], tr["C"], tr["D"]]).astype(np.float32)
part = ids["partition"].to_numpy(); elig = ids["eligible"].to_numpy(); posa = ids["pos_in_segment"].to_numpy(); seg = ids["segment_id"].to_numpy()
calm = elig & (part == "cal"); ci = np.where(calm)[0]; half = len(ci) // 2
c1 = np.zeros(len(part), bool); c1[ci[:half]] = True; c2 = np.zeros(len(part), bool); c2[ci[half:]] = True
print(f"calibration {calm.sum()/3600:.1f} h; C1 {c1.sum()/3600:.1f} h; C2 {c2.sum()/3600:.1f} h", flush=True)
# ---------------- (1) selection with thresholds from C1 only
rels = [r for r in json.load(open(os.path.join(ROOT, "manifests", "constraints.json")))["relations"] if r["status"] == "retained"]
dev = json.load(open(os.path.join(ROOT, "runs", "baselines", "dev_injections.json")))
Zi, evid = apply_injections(Z, dev); Xi = X.copy(); Xi[:, :Z.shape[1]] = Zi
raw = lambda z: z.astype(np.float64) * scale + med
V, Vi = constraint_matrix(rels, tg, raw(Z)), constraint_matrix(rels, tg, raw(Zi))
vrows = np.where(part == "val")[0]; lo, hi = int(vrows.min()), int(vrows.max()) + 1
valm = (elig & (part == "val"))[lo:hi]; pos = posa[lo:hi]; ev = evid[lo:hi][valm]
names = [f"dev{k}" for k in range(len(dev))]; rows_all = np.arange(lo, hi); pred_rows = rows_all[pos >= W_MAX]
def stream(Q, ref):
    s = np.sort(tail_surprise(Q, ref), 1)[:, -min(TOPK, Q.shape[1]):].mean(1); s[pos < W_MAX] = 0; return causal_smooth(s, SMOOTH)
def errors(S, XX, ZZ, W):
    E = np.zeros((hi - lo, ZZ.shape[1]), np.float32); E[pos >= W_MAX] = np.abs(forecast(S, XX, pred_rows, W) - ZZ[pred_rows]); return E
refV = ecdf_ref(V[lo:hi][valm]); v_inj = stream(Vi[lo:hi], refV); rows, t0 = [], time.time()
for i, rd in enumerate(sorted(os.path.dirname(f) for f in glob.glob(os.path.join(ROOT, "runs", "train", "search__*_s17", "model_best.pt")))):
    sm = json.load(open(os.path.join(rd, "summary.json"))); c = sm["config"]; S = load_state(os.path.join(rd, "model_best.pt"))
    E = errors(S, X, Z, c["W"]); refE = ecdf_ref(E[valm]); sd = (0.5 * stream(errors(S, Xi, Zi, c["W"]), refE) + 0.5 * v_inj)[valm]
    z = np.load(os.path.join(rd, "scores.npz")); s = 0.5 * z["p_train"] + 0.5 * z["v_train"]
    tau_full = calibrate_threshold(s[calm], seg[calm], calm.sum() / 3600)[0]; tau_c1 = calibrate_threshold(s[c1], seg[c1], c1.sum() / 3600)[0]
    d = lambda tau: int(sum(bool((sd[ev == e] > tau).any()) for e in names))
    rows.append({"id": c["id"], "W": c["W"], "lambda_inv": c["l_inv"], "lambda_phys": c["l_phys"], "selection_loss": sm["selection_loss"],
                 "tau_saved": sm["tau"]["b0.5"], "tau_full": tau_full, "tau_C1": tau_c1, "stored_detected": sm["dev"]["b0.5"]["detected"],
                 "detected_full": d(tau_full), "detected_C1": d(tau_c1)})
    print(f"[{i+1}] {c['id']}: stored {rows[-1]['stored_detected']} full {rows[-1]['detected_full']} C1 {rows[-1]['detected_C1']} tau {tau_full:.2f}/{tau_c1:.2f} {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(A.out, "split_selection.csv"), index=False)
pick = lambda col: df.sort_values([col, "selection_loss"], ascending=[False, True]).iloc[0]["id"]
print("selected (full calibration):", pick("detected_full")); print("selected (C1 only):       ", pick("detected_C1"))
# ---------------- (2) Proposition 5 on C2 only
te = pd.read_parquet(os.path.join(ROOT, "data", "derived", "hai2305_test.parquet"), columns=["timestamp", "file_id", "segment_id", "eligible"])
lab = pd.read_parquet(os.path.join(ROOT, "data", "derived", "hai2305_test_labels.parquet"))
with open(os.path.join(ROOT, "manifests", "unseal_log.txt"), "a") as f_: f_.write(time.strftime("%Y-%m-%d %H:%M:%S") + " test labels read by split_cal.py (block evaluation with the level from the second half of calibration)\n")
mask = te.eligible.to_numpy(); y = lab.label.to_numpy().astype(bool); tseg = te.segment_id.to_numpy()
def blocks(sg, L):
    out = []
    for s_ in pd.unique(sg):
        idx = np.where(sg == s_)[0]
        for j in range(len(idx) // L): out.append(idx[j * L:(j + 1) * L])
    return out
T = os.path.join(ROOT, "runs", "train"); res = []
runs = [("TailGuard", s_, os.path.join(T, f"{'search' if s_ == 17 else 'shiftics'}__W30_inv1.0_phys0.1_phys_regime_bal_s{s_}"), 0.5) for s_ in (17, 29, 43, 71, 101)]
runs += [("Matched GRU", s_, os.path.join(T, f"gru_erm__W30_inv0.0_phys0.0_phys_regime_pool_s{s_}"), 0.0) for s_ in (17, 29, 43, 71, 101)]
S_ = {}
for meth, s_, d_, beta in runs:
    z = np.load(os.path.join(d_, "scores.npz")); st = (1 - beta) * z["p_test"] + beta * z["v_test"]
    S_[(meth, s_)] = (np.where(np.isfinite(st), st, -np.inf), (1 - beta) * z["p_train"] + beta * z["v_train"])
bd = os.path.join(ROOT, "runs", "baselines", "ridge_a1.0"); sct = pd.read_parquet(os.path.join(bd, "test_scores_SEALED.parquet"))
st = np.full(len(te), -np.inf); key = te.loc[te.eligible, ["file_id", "timestamp"]].reset_index()
mm = key.merge(sct[["file_id", "timestamp", "score"]], on=["file_id", "timestamp"], how="left"); st[mm["index"].to_numpy()] = mm["score"].to_numpy()
S_[("Ridge AR", 17)] = (st, np.load(os.path.join(bd, "train_scores.npz"))["s_train"])
el = np.where(mask)[0]
for H in (1.0, 0.5):
    L = int(H * 3600); CB = blocks(seg[c2], L); B = len(CB); k = max(1, int(np.floor((B + 1) * H / 24.0)))
    TBn = [b for b in [el[b] for b in blocks(tseg[mask], L)] if not y[b].any()]
    for (meth, s_), (stt, strn) in S_.items():
        sc2 = strn[c2]; tau_k = float(np.sort([sc2[b].max() for b in CB])[::-1][k - 1]); a = (stt > tau_k) & mask
        ne = int((episodes(a[mask], tseg[mask]) & ~y[mask]).sum())
        res.append({"method": meth, "seed": s_, "block_hours": H, "B": B, "k": k, "guaranteed": k / (B + 1), "tau_k_C2": tau_k,
                    "alarmed_normal_blocks": int(sum(a[b].any() for b in TBn)), "normal_test_blocks": len(TBn),
                    "share": float(np.mean([a[b].any() for b in TBn])), "false_episodes": ne, "FA24": 24 * ne / ((~y[mask]).sum() / 3600)})
r = pd.DataFrame(res); r.to_csv(os.path.join(A.out, "split_prop5.csv"), index=False)
print(r.groupby(["block_hours", "method"])[["B", "k", "guaranteed", "share", "alarmed_normal_blocks", "normal_test_blocks", "FA24", "tau_k_C2"]].mean().round(3).to_string())
