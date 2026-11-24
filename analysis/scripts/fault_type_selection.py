"""
TailGuard | sensitivity of model selection to the development-fault types (no retraining, no test labels, no PyTorch).

Re-scores the 27 saved search checkpoints (runs/train/search__*/model_best.pt) on the validation copy with the 36
development faults, under the prespecified score (beta = 0.5) and each candidate's calibrated threshold, and records
which faults each candidate detects; the totals must equal those stored by the training runs. It then repeats the
selection rule of Section 3.4.3 with one fault type (bias, freeze or ramp) left out and reports the configuration it
picks and how that configuration ranks on the left-out type. Detection by fault kind and duration is reported for
the selected configuration.
  python analysis/scripts/fault_type_selection.py --root <repo root> --out <dir>
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); A = ap.parse_args()
ROOT = os.path.abspath(A.root); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tailguard_lib import load, manifest, ecdf_ref, tail_surprise, causal_smooth, apply_injections
from constraints_np import constraint_matrix
from np_gru import load_state, forecast
os.makedirs(A.out, exist_ok=True)
W_MAX, TOPK, SMOOTH = 120, 3, 30

tr = load("train"); m = manifest(); tg = tr["targets"]; sc = m["scaler_fit"]
scale = np.array([sc[c]["scale"] for c in tg]); med = np.array([sc[c]["median"] for c in tg])
ids = tr["df_ids"]; Z = tr["Z"]; X = np.hstack([tr["Z"], tr["C"], tr["D"]]).astype(np.float32)
part = ids["partition"].to_numpy(); elig = ids["eligible"].to_numpy(); posa = ids["pos_in_segment"].to_numpy()
rels = [r for r in json.load(open(os.path.join(ROOT, "manifests", "constraints.json")))["relations"] if r["status"] == "retained"]
dev = json.load(open(os.path.join(ROOT, "runs", "baselines", "dev_injections.json")))
Zi, evid = apply_injections(Z, dev); Xi = X.copy(); Xi[:, :Z.shape[1]] = Zi
raw = lambda z: z.astype(np.float64) * scale + med
V, Vi = constraint_matrix(rels, tg, raw(Z)), constraint_matrix(rels, tg, raw(Zi))
vrows = np.where(part == "val")[0]; lo, hi = int(vrows.min()), int(vrows.max()) + 1      # contiguous validation block
valm = (elig & (part == "val"))[lo:hi]; pos = posa[lo:hi]; ev = evid[lo:hi][valm]
kind = {f"dev{k}": e["kind"] for k, e in enumerate(dev)}; dur = {f"dev{k}": e["end"] - e["start"] for k, e in enumerate(dev)}
rows_all = np.arange(lo, hi); pred_rows = rows_all[pos >= W_MAX]


def stream(Q, ref):
    s = np.sort(tail_surprise(Q, ref), 1)[:, -min(TOPK, Q.shape[1]):].mean(1); s[pos < W_MAX] = 0
    return causal_smooth(s, SMOOTH)


def errors(S, XX, ZZ, W):
    E = np.zeros((hi - lo, ZZ.shape[1]), np.float32)
    E[pos >= W_MAX] = np.abs(forecast(S, XX, pred_rows, W) - ZZ[pred_rows]); return E


refV = ecdf_ref(V[lo:hi][valm]); v_inj = stream(Vi[lo:hi], refV)
rows, t0 = [], time.time()
runs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(ROOT, "runs", "train", "search__*_s17", "model_best.pt")))
for i, rd in enumerate(runs):
    sm = json.load(open(os.path.join(rd, "summary.json"))); c = sm["config"]; S = load_state(os.path.join(rd, "model_best.pt"))
    E = errors(S, X, Z, c["W"]); refE = ecdf_ref(E[valm]); Ei = errors(S, Xi, Zi, c["W"])
    a = (0.5 * stream(Ei, refE) + 0.5 * v_inj)[valm] > sm["tau"]["b0.5"]
    det = {e: bool(a[ev == e].any()) for e in kind}
    rows.append({"id": c["id"], "W": c["W"], "lambda_inv": c["l_inv"], "lambda_phys": c["l_phys"], "selection_loss": sm["selection_loss"],
                 "stored_detected": sm["dev"]["b0.5"]["detected"], **det})
    print(f"[{i + 1}/{len(runs)}] {c['id']}: {sum(det.values())}/36 (stored {sm['dev']['b0.5']['detected']})  {time.time() - t0:.0f}s", flush=True)

df = pd.DataFrame(rows); faults = list(kind)
for k in ("bias", "freeze", "ramp"):
    df[k] = df[[f for f in faults if kind[f] == k]].sum(1)
df["all"] = df[["bias", "freeze", "ramp"]].sum(1)
df.to_csv(os.path.join(A.out, "fault_detection_by_candidate.csv"), index=False)
print("reproduced stored totals for", int((df["all"] == df["stored_detected"]).sum()), "of", len(df), "candidates")

pick = lambda d, col: d.sort_values([col, "selection_loss"], ascending=[False, True]).iloc[0]
full = pick(df, "all")
out = [{"selection on": "bias + freeze + ramp", "selected": full["id"], "held-out type": "", "held-out detected": "",
        "rank on held-out type": "", "best on held-out type": "", "uses constraint term": full["lambda_phys"] > 0}]
for k in ("bias", "freeze", "ramp"):
    rest = [x for x in ("bias", "freeze", "ramp") if x != k]; d = df.assign(sel=df[rest].sum(1)); s = pick(d, "sel")
    out.append({"selection on": " + ".join(rest), "selected": s["id"], "held-out type": k, "held-out detected": f"{int(s[k])}/12",
                "rank on held-out type": f"{int((df[k] > s[k]).sum()) + 1} of {len(df)}", "best on held-out type": f"{int(df[k].max())}/12",
                "uses constraint term": s["lambda_phys"] > 0})
pd.DataFrame(out).to_csv(os.path.join(A.out, "fault_type_selection.csv"), index=False)
sel = df[df["id"] == full["id"]].iloc[0]
bd = pd.DataFrame([{"kind": kind[f], "duration_s": dur[f], "detected": bool(sel[f])} for f in faults]).groupby(["kind", "duration_s"])["detected"].agg(["sum", "count"])
bd.to_csv(os.path.join(A.out, "fault_detection_by_duration.csv"))
print(pd.DataFrame(out).to_string()); print(bd)
