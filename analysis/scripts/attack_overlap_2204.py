"""TailGuard | per-attack detection overlap on HAI 22.04 (Section 4.6, Table A13).
For each of the 58 attacks: number of TailGuard seeds (0-5) that raise an alarm inside it and whether ridge autoregression
does, both at their stored closure thresholds (prespecified score). Reads frozen score streams; trains nothing.
  python attack_overlap_2204.py --root <HAI 22.04 working root> --raw <dir with test1-4.csv> --out <csv>
Labels are the Attack column of the official test files; an attack is a maximal run of Attack > 0 within a file."""
import argparse, glob, json, os, time
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
A = ap.parse_args(); R = A.root
te = pd.read_parquet(os.path.join(R, "data/derived/hai2204_test.parquet"), columns=["timestamp", "file_id", "eligible", "regime"])
lab = []
for fid in pd.unique(te.file_id):
    d = pd.read_csv(os.path.join(A.raw, f"{fid}.csv"), usecols=lambda c: c.strip() in ("timestamp", "Attack"))
    d.columns = [c.strip() for c in d.columns]; d["file_id"] = fid; d["timestamp"] = pd.to_datetime(d["timestamp"]); lab.append(d)
lab = pd.concat(lab, ignore_index=True); te["timestamp"] = pd.to_datetime(te["timestamp"])
y = te.merge(lab, on=["file_id", "timestamp"], how="left")["Attack"].fillna(0).to_numpy() > 0
log = os.path.join(R, "manifests", "unseal_log.txt")
if os.path.isdir(os.path.dirname(log)):
    with open(log, "a") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " test labels read by attack_overlap_2204.py (per-attack overlap of frozen scores)\n")
mask = te.eligible.to_numpy(); fid = te.file_id.to_numpy()
start = np.where(y & ~(np.r_[False, y[:-1]] & (np.r_[fid[:1], fid[:-1]] == fid)))[0]; ev = np.full(len(y), -1); ev[y] = np.cumsum(np.isin(np.where(y)[0], start)) - 1
tg = []
for rd in sorted(glob.glob(os.path.join(R, "runs/train/*__W30_inv1.0_phys0.1_phys_regime_bal_s*"))):
    z = np.load(os.path.join(rd, "scores.npz")); tau = json.load(open(os.path.join(rd, "summary.json")))["tau"]["b0.5"]
    tg.append(((0.5 * z["p_test"] + 0.5 * z["v_test"]) > tau) & mask)
b = os.path.join(R, "runs/baselines/ridge_a1.0"); sc = pd.read_parquet(os.path.join(b, "test_scores_SEALED.parquet")); sc["timestamp"] = pd.to_datetime(sc["timestamp"])
s = te[["file_id", "timestamp"]].merge(sc[["file_id", "timestamp", "score"]], on=["file_id", "timestamp"], how="left")["score"].fillna(-np.inf).to_numpy()
ridge = (s > json.load(open(os.path.join(b, "summary.json")))["tau"]) & mask
rows = []
for e in range(ev.max() + 1):
    idx = np.where((ev == e) & mask)[0]
    if not len(idx): continue
    rows.append({"event": e, "file": fid[idx[0]], "regime": int(te.regime.iloc[idx[0]]), "dur": len(idx),
                 "TG": int(sum(a[idx].any() for a in tg)), "ridge": int(ridge[idx].any())})
df = pd.DataFrame(rows); df.to_csv(A.out, index=False)
t, r = df.TG > 0, df.ridge > 0
print(f"{len(tg)} TailGuard seeds; attacks {len(df)}: both {int((t & r).sum())}, ridge only {int((~t & r).sum())}, TailGuard only {int((t & ~r).sum())}, neither {int((~t & ~r).sum())}")
