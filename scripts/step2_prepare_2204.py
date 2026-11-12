"""
TailGuard  |  Step 2 for the HAI 22.04 confirmation set (same rules as HAI 23.05; labels are the Attack column of the test files)

Folder layout (repository root):
    hai-23.05\\            official CSVs (verified in Step 1)
    scripts\\step2_prepare.py
Run:
    conda activate tailguard
    python scripts\\step2_prepare.py
Writes:
    data\\derived\\hai2305_train.parquet        model-visible training rows (raw values + ids)
    data\\derived\\hai2305_test.parquet         model-visible test rows (NO labels)
    data\\derived\\hai2305_test_labels.parquet  evaluation-only labels + event ids
    manifests\\split_manifest.json              frozen protocol decisions
    manifests\\step2_report.txt
Deterministic and safe to re-run.
"""
import json, os, sys, hashlib, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "scripts" else HERE
DATA = os.environ.get("HAI2204_DIR", os.path.join(ROOT, "hai-22.04"))
if not os.path.isdir(DATA):
    DATA = os.path.join(ROOT, "data", "raw", "hai", "hai-22.04")
DER = os.path.join(ROOT, "data", "derived"); MAN = os.path.join(ROOT, "manifests")
os.makedirs(DER, exist_ok=True); os.makedirs(MAN, exist_ok=True)

TRAIN_FILES = [f"train{i}.csv" for i in range(1, 7)]  # chronological
TEST_FILES = [f"test{i}.csv" for i in range(1, 5)]
SPLIT = (0.60, 0.80)                 # fitting | validation | calibration
W_MAX = 120                          # largest window in the grid -> common eligibility mask
MIN_ENV_TARGETS = 1000
CONTEXT_TAGS = ["P2_AutoSD", "P2_ManualSD", "P4_ST_PS", "P4_HT_PS", "P1_PP04SP", "P1_STSP"]  # set points
REGIME_TAG, REGIME_EDGES = "P4_ST_PS", [10.0, 20.0]   # approved 2026-10-13: low <10, mid 10-20, high >=20
REGIME_NAMES = ["low", "mid", "high"]

lines = []
def log(s=""):
    print(s); lines.append(str(s))


def read(f):
    df = pd.read_csv(os.path.join(DATA, f), encoding="utf-8-sig", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.insert(1, "file_id", f.replace(".csv", ""))
    return df


def regime_of(v):
    return np.digitize(v, REGIME_EDGES).astype(np.int8)


def main():
    t0 = time.time()
    log("TailGuard Step 2 report | HAI 22.04 (confirmation set)")
    tr = pd.concat([read(f) for f in TRAIN_FILES], ignore_index=True)
    assert tr["timestamp"].is_monotonic_increasing, "training files are not in chronological order"
    assert (tr["Attack"] == 0).all(), "attack rows in HAI 22.04 training files"
    tr = tr.drop(columns=["Attack"])
    tags = [c for c in tr.columns if c not in ("timestamp", "file_id")]
    n = len(tr)
    c60, c80 = int(np.floor(SPLIT[0] * n)), int(np.floor(SPLIT[1] * n))
    part = np.empty(n, dtype=object); part[:c60] = "fit"; part[c60:c80] = "val"; part[c80:] = "cal"
    tr.insert(2, "partition", part)
    tr.insert(3, "segment_id", tr["file_id"] + "|" + tr["partition"])
    tr.insert(4, "pos_in_segment", tr.groupby("segment_id").cumcount().astype(np.int32))
    tr.insert(5, "eligible", tr["pos_in_segment"] >= W_MAX)

    fit = tr[tr.partition == "fit"]
    # ---- feature roles: decided on fitting data only
    constant, discrete, continuous, context = [], [], [], []
    for c in tags:
        v = fit[c].to_numpy(dtype=float)
        if np.nanmax(v) == np.nanmin(v):
            constant.append(c)
        elif c in CONTEXT_TAGS:
            context.append(c)
        elif np.all(np.mod(v, 1) == 0) and len(np.unique(v)) <= 20:
            discrete.append(c)
        else:
            continuous.append(c)
    # ---- robust scaler on fitting data (continuous targets + context)
    scaler = {}
    for c in continuous + context:
        v = fit[c].to_numpy(dtype=float)
        med = float(np.median(v)); q1, q3 = np.percentile(v, [25, 75])
        rng = float(np.max(v) - np.min(v))
        floor = max(0.01 * rng, 1e-6)          # instrument-scale floor: 1% of fitting range
        scaler[c] = {"median": med, "iqr": float(q3 - q1), "scale": float(max(q3 - q1, floor)), "floor": floor}
    disc_levels = {c: sorted(np.unique(fit[c]).astype(int).tolist()) for c in discrete}

    # ---- environments: calendar day within segment, short days merged into the previous one
    tr["day"] = tr["timestamp"].dt.strftime("%Y-%m-%d")
    tr["env_id"] = tr["segment_id"] + "|" + tr["day"]
    env_map = {}
    for seg, g in tr.groupby("segment_id", sort=False):
        envs = list(dict.fromkeys(g["env_id"]))
        counts = g.loc[g.eligible].groupby("env_id").size().reindex(envs).fillna(0).astype(int)
        merged, cur = {}, None
        for e in envs:
            if cur is None or counts[e] >= MIN_ENV_TARGETS:
                cur = e
            merged[e] = cur
        # if the first env is short, merge it forward
        first = envs[0]
        if counts[first] < MIN_ENV_TARGETS and len(envs) > 1:
            nxt = next((merged[e] for e in envs[1:] if merged[e] != first), first)
            for e in envs:
                if merged[e] == first:
                    merged[e] = nxt
        env_map.update(merged)
    tr["env_id"] = tr["env_id"].map(env_map)
    tr["regime"] = regime_of(tr[REGIME_TAG].to_numpy(dtype=float))
    tr = tr.drop(columns=["day"])

    # ---- test rows and evaluation-only labels (positional join, verified in Step 1)
    te_parts, lab_parts = [], []
    for f in TEST_FILES:
        d = read(f)
        y = (pd.to_numeric(d["Attack"]) > 0).astype(np.int8).to_numpy(); d = d.drop(columns=["Attack"])
        d.insert(2, "segment_id", d["file_id"] + "|test")
        d.insert(3, "pos_in_segment", np.arange(len(d), dtype=np.int32))
        d.insert(4, "eligible", d["pos_in_segment"] >= W_MAX)
        d["regime"] = regime_of(d[REGIME_TAG].to_numpy(dtype=float))
        dd = np.diff(np.r_[0, y, 0]); starts = np.where(dd == 1)[0]; ends = np.where(dd == -1)[0] - 1
        ev = np.full(len(y), -1, dtype=np.int32)
        for k, (s, e) in enumerate(zip(starts, ends)):
            ev[s:e + 1] = k
        fid = f.replace(".csv", "")
        lab_parts.append(pd.DataFrame({"timestamp": d["timestamp"], "file_id": fid, "label": y,
                                       "event_id": np.where(ev >= 0, np.char.add(fid + "_E", ev.astype(str)), "")}))
        te_parts.append(d)
    te = pd.concat(te_parts, ignore_index=True); labs = pd.concat(lab_parts, ignore_index=True)

    # ---- save
    keep = ["timestamp", "file_id", "segment_id", "pos_in_segment", "eligible", "regime"]
    tr_out = tr[keep[:2] + ["partition"] + keep[2:] + ["env_id"] + tags].copy()
    te_out = te[keep + tags].copy()
    for df in (tr_out, te_out):
        for c in tags:
            df[c] = df[c].astype(np.float32)
    tr_out.to_parquet(os.path.join(DER, "hai2204_train.parquet"), index=False)
    te_out.to_parquet(os.path.join(DER, "hai2204_test.parquet"), index=False)
    labs.to_parquet(os.path.join(DER, "hai2204_test_labels.parquet"), index=False)

    # ---- manifest + report
    hrs = lambda m: round(float(m) / 3600, 2)
    part_tab = tr.groupby("partition", sort=False).agg(rows=("timestamp", "size"), start=("timestamp", "min"),
                                                      end=("timestamp", "max"), eligible=("eligible", "sum"))
    env_tab = tr[tr.eligible].groupby(["partition", "env_id"], sort=False).size()
    reg_tab = pd.crosstab(tr["partition"], tr["regime"]).reindex(index=["fit", "val", "cal"]) / 3600
    reg_te = pd.crosstab(te["file_id"], te["regime"]) / 3600
    events = labs[labs.label == 1].groupby("event_id").agg(file_id=("file_id", "first"), start=("timestamp", "min"),
                                                            end=("timestamp", "max"), seconds=("label", "size"))
    ev_reg = []
    for eid, r in events.iterrows():
        m = (te.file_id == r.file_id) & (te.timestamp == r.start)
        ev_reg.append(int(te.loc[m, "regime"].iloc[0]))
    events["regime_at_onset"] = ev_reg
    manifest = {
        "dataset": "HAI 22.04", "approved_decisions": {
            "test_description": "official HAI 22.04 test split (confirmation set; protocol frozen on HAI 23.05)",
            "regime_tag": REGIME_TAG, "regime_edges": REGIME_EDGES, "regime_names": REGIME_NAMES,
            "label_join": "Attack column of the official test files"},
        "split_fractions": SPLIT, "cut_rows": [c60, c80],
        "cut_timestamps": [str(tr.timestamp.iloc[c60]), str(tr.timestamp.iloc[c80])],
        "partitions": {p: {"rows": int(r.rows), "eligible": int(r.eligible), "start": str(r.start), "end": str(r.end)}
                       for p, r in part_tab.iterrows()},
        "w_max_eligibility": W_MAX,
        "feature_roles": {"continuous_targets": continuous, "context_setpoints": context,
                          "discrete_actuators": discrete, "excluded_constant_in_fit": constant},
        "discrete_levels_fit": disc_levels, "scaler_fit": scaler,
        "environments": {f"{p}::{e}": int(v) for (p, e), v in env_tab.items()},
        "regime_hours_train": {p: {REGIME_NAMES[int(k)]: round(float(v), 2) for k, v in row.items()} for p, row in reg_tab.iterrows()},
        "regime_hours_test": {f: {REGIME_NAMES[int(k)]: round(float(v), 2) for k, v in row.items()} for f, row in reg_te.iterrows()},
        "events": {eid: {"file": r.file_id, "start": str(r.start), "end": str(r.end), "seconds": int(r.seconds),
                         "regime_at_onset": REGIME_NAMES[r.regime_at_onset]} for eid, r in events.iterrows()},
    }
    for f in ["hai2204_train.parquet", "hai2204_test.parquet", "hai2204_test_labels.parquet"]:
        h = hashlib.sha256(open(os.path.join(DER, f), "rb").read()).hexdigest()
        manifest.setdefault("derived_sha256", {})[f] = h
    with open(os.path.join(MAN, "split_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    log(f"Training rows {n:,} ({hrs(n)} h); cut rows {c60:,} / {c80:,}")
    log(f"Cut timestamps: fit|val at {manifest['cut_timestamps'][0]}, val|cal at {manifest['cut_timestamps'][1]}")
    for p, r in part_tab.iterrows():
        log(f"  {p}: {hrs(r.rows)} h  {r.start} -> {r.end}  eligible targets {int(r.eligible):,}")
    log(f"Feature roles (from fitting data): {len(continuous)} continuous targets, {len(context)} set-point context, "
        f"{len(discrete)} discrete actuators, {len(constant)} excluded constant")
    log(f"  context: {context}")
    log(f"  discrete: {discrete}")
    log(f"  excluded: {constant}")
    log("Environments (eligible targets):")
    for (p, e), v in env_tab.items():
        log(f"  {p:4s} {e.split('|')[0]} {e.split('|')[-1]}  {v:,}")
    log("Regime hours by partition (low/mid/high):")
    log(reg_tab.round(1).to_string())
    log("Regime hours by test file:")
    log(reg_te.round(1).to_string())
    log(f"Attack events: {len(events)}  by onset regime: {events.regime_at_onset.map(lambda k: REGIME_NAMES[k]).value_counts().to_dict()}")
    lost = int((labs.label.values == 1)[~te.eligible.values].sum())
    log(f"Attack seconds lost to the first-{W_MAX} eligibility rule: {lost}")
    log(f"Done in {time.time() - t0:.0f} s")
    with open(os.path.join(MAN, "step2_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
