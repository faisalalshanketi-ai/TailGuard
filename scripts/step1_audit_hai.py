"""
TailGuard  |  Step 1: HAI 23.05 payload audit (read-only, no modelling)

Run from the repository root:
    python scripts/step1_audit_hai.py

Reads   data/raw/hai/hai-23.05/*.csv  (+ summary_label*.txt)
Writes  manifests/hai2305_data_manifest.json
        manifests/hai2305_audit_report.txt   <- send this file back

Safe to re-run at any time (e.g. after a power cut); it only reads the raw files.
"""
import hashlib, json, os, subprocess, sys, time
from collections import Counter
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "scripts" else HERE
REPO = os.path.join(ROOT, "data", "raw", "hai")
DATA = os.path.join(REPO, "hai-23.05")
OUT = os.path.join(ROOT, "manifests")
CHUNK = 200_000
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"

# Set-point tags and normal ranges documented for HAI 23.05 normal operation
# (HMM-generated set-point commands; HAI technical details v4.0, Fig. 15).
DOC_SETPOINTS = {
    "P1_B3004": (400, 500), "P1_B400B": (950, 1050), "P1_B2016": (0.1, 0.2),
    "P1_B4022": (25, 27), "P2_AutoSD": (100, 120), "P2_ManualSD": (11, 12),
    "P4_HT_PS": (0, 15), "P4_ST_PS": (0, 50),
}

os.makedirs(OUT, exist_ok=True)
report_lines = []


def log(s=""):
    print(s)
    report_lines.append(str(s))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lfs_index():
    """Map file path -> LFS oid from 'git lfs ls-files --long' (optional)."""
    try:
        out = subprocess.run(["git", "-C", REPO, "lfs", "ls-files", "--long"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        log(f"  (git lfs ls-files unavailable: {e})")
        return {}
    idx = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            idx[os.path.normpath(parts[-1])] = parts[0]
    return idx


def repo_commit():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "UNKNOWN"


def runs_of_ones(arr):
    """Return list of (start_idx, end_idx_inclusive) for contiguous 1-runs."""
    a = np.asarray(arr).astype(np.int8)
    if a.size == 0:
        return []
    d = np.diff(np.concatenate([[0], a, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def audit_csv(path):
    name = os.path.basename(path)
    info = {"file": name, "bytes": os.path.getsize(path)}
    with open(path, "rb") as f:
        head = f.read(200)
    if head.startswith(LFS_HEADER):
        info["status"] = "FAIL: Git LFS pointer, payload not downloaded"
        return info, None
    info["sha256"] = sha256(path)

    # header / dialect
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline().strip()
        sample = [f.readline().strip() for _ in range(3)]
    delim = "," if header.count(",") >= header.count(";") else ";"
    cols = [c.strip() for c in header.split(delim)]
    info.update(delimiter=delim, n_columns=len(cols), first_columns=cols[:5],
                sample_rows=[s[:160] for s in sample])
    tcol = cols[0]

    n = 0; ts_min = ts_max = None; prev_last = None
    diffs = Counter(); dup = 0; nonmono = 0
    na = Counter(); inf = Counter(); mins = {}; maxs = {}
    uniq = {c: set() for c in cols[1:]}; is_int = {c: True for c in cols[1:]}
    ts_parse_fail = 0
    sp_series = {c: [] for c in DOC_SETPOINTS if c in cols}
    for ch in pd.read_csv(path, sep=delim, chunksize=CHUNK, encoding="utf-8-sig",
                          skipinitialspace=True):
        ch.columns = [c.strip() for c in ch.columns]
        ts = pd.to_datetime(ch[tcol], errors="coerce")
        ts_parse_fail += int(ts.isna().sum())
        n += len(ch)
        tmin, tmax = ts.min(), ts.max()
        ts_min = tmin if ts_min is None or tmin < ts_min else ts_min
        ts_max = tmax if ts_max is None or tmax > ts_max else ts_max
        full = ts if prev_last is None else pd.concat([pd.Series([prev_last]), ts], ignore_index=True)
        dsec = full.diff().dt.total_seconds().dropna()
        diffs.update(dsec.round(3).value_counts().to_dict())
        dup += int((dsec == 0).sum()); nonmono += int((dsec < 0).sum())
        prev_last = ts.iloc[-1]
        num = ch.drop(columns=[tcol]).apply(pd.to_numeric, errors="coerce")
        for c in num.columns:
            v = num[c].to_numpy(dtype=float)
            na[c] += int(np.isnan(v).sum()); inf[c] += int(np.isinf(v).sum())
            fv = v[np.isfinite(v)]
            if fv.size:
                mins[c] = min(mins.get(c, np.inf), float(fv.min()))
                maxs[c] = max(maxs.get(c, -np.inf), float(fv.max()))
                if is_int[c] and not np.all(np.mod(fv, 1) == 0):
                    is_int[c] = False
                if len(uniq[c]) <= 20:
                    uniq[c].update(np.unique(fv)[:25].tolist())
        for c in sp_series:
            sp_series[c].append(pd.DataFrame({"t": ts.values, "v": num[c].values}))

    feats = cols[1:]
    label_like = [c for c in feats if c.lower() in ("attack", "label") or c.lower().startswith("attack")]
    constant = [c for c in feats if c in mins and mins[c] == maxs[c]]
    discrete = [c for c in feats if is_int[c] and len(uniq[c]) <= 20 and c not in constant]
    info.update(
        status="PASS", rows=n, timestamp_column=tcol,
        ts_min=str(ts_min), ts_max=str(ts_max), ts_parse_failures=ts_parse_fail,
        duration_hours=round((ts_max - ts_min).total_seconds() / 3600, 2) if n else 0,
        interval_histogram_top={str(k): int(v) for k, v in diffs.most_common(6)},
        duplicate_timestamps=dup, nonmonotonic_steps=nonmono,
        n_features_excl_time=len(feats), label_like_columns=label_like,
        constant_columns=constant, discrete_columns=discrete,
        n_continuous=len(feats) - len(constant) - len(discrete) - len(label_like),
        columns_with_missing={c: k for c, k in na.items() if k},
        columns_with_inf={c: k for c, k in inf.items() if k},
        ordered_columns=cols,
    )
    sp = {c: pd.concat(v, ignore_index=True) for c, v in sp_series.items()} if sp_series else {}
    return info, sp


def setpoint_profile(name, sp):
    """Occupancy of documented set points: how many distinct operating points exist."""
    prof = {}
    for c, df in sp.items():
        v = df["v"].to_numpy(dtype=float)
        lo, hi = DOC_SETPOINTS[c]
        changes = int(np.sum(np.abs(np.diff(v)) > 1e-9))
        vals, counts = np.unique(np.round(v, 3), return_counts=True)
        order = np.argsort(-counts)[:8]
        prof[c] = {
            "documented_range": [lo, hi],
            "observed_min": float(np.nanmin(v)), "observed_max": float(np.nanmax(v)),
            "n_changes": changes, "n_distinct_values": int(len(vals)),
            "top_values_hours": {str(float(vals[i])): round(float(counts[i]) / 3600, 2) for i in order},
        }
    return prof


def audit_labels(test_info, label_path, test_path):
    out = {"file": os.path.basename(label_path)}
    lab = pd.read_csv(label_path, encoding="utf-8-sig", skipinitialspace=True)
    lab.columns = [c.strip() for c in lab.columns]
    out.update(rows=len(lab), columns=list(lab.columns))
    if test_info is None or test_info.get("status") != "PASS":
        out["join"] = "SKIPPED: test payload missing"
        return out
    lcol = [c for c in lab.columns[1:]]
    ycol = lcol[0] if lcol else None
    out["label_column"] = ycol
    out["label_values"] = {str(k): int(v) for k, v in lab[ycol].value_counts().items()} if ycol else {}
    tcol = test_info["timestamp_column"]
    tts = pd.read_csv(test_path, usecols=[0], encoding="utf-8-sig", skipinitialspace=True).iloc[:, 0]
    lts_col = lab.columns[0]
    lts = pd.to_datetime(lab[lts_col], errors="coerce")
    tts = pd.to_datetime(tts, errors="coerce")
    out["same_length"] = bool(len(lts) == len(tts))
    if lts.notna().all():
        out["timestamps_identical"] = bool(len(lts) == len(tts) and (lts.values == tts.values).all())
        out["label_ts_unique"] = bool(lts.is_unique)
        missing = int((~tts.isin(lts)).sum())
        out["test_rows_without_label"] = missing
    else:
        out["timestamps_identical"] = "label file has no parseable timestamp in first column"
    if ycol:
        y = (pd.to_numeric(lab[ycol], errors="coerce").fillna(0) > 0).astype(int).to_numpy()
        runs = runs_of_ones(y)
        out["positive_rows"] = int(y.sum())
        out["positive_rate"] = round(float(y.mean()), 5)
        out["positive_runs"] = len(runs)
        out["run_lengths_s_first20"] = [e - s + 1 for s, e in runs[:20]]
    return out


def main():
    t0 = time.time()
    log("TailGuard Step 1 audit report  |  HAI 23.05")
    log(f"Run at {time.strftime('%Y-%m-%d %H:%M:%S')}  python {sys.version.split()[0]}  pandas {pd.__version__}")
    if not os.path.isdir(DATA):
        log(f"FAIL: {DATA} not found. Clone the repository first (see instructions).")
        return finish({})
    commit = repo_commit()
    log(f"Repository commit: {commit}")
    lfs = lfs_index()
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".csv"))
    log(f"CSV files found: {files}")
    manifest = {"dataset": "HAI 23.05", "repo_commit": commit, "files": {}, "labels": {},
                "setpoints": {}}
    for f in files:
        if f.startswith("label"):
            continue
        p = os.path.join(DATA, f)
        log(f"\n=== {f} ===")
        info, sp = audit_csv(p)
        key = os.path.normpath(os.path.join("hai-23.05", f))
        oid = lfs.get(key)
        if oid and "sha256" in info:
            info["lfs_oid_match"] = info["sha256"].startswith(oid) or oid.startswith(info["sha256"][:len(oid)])
        manifest["files"][f] = info
        for k in ("status", "bytes", "sha256", "lfs_oid_match", "rows", "ts_min", "ts_max", "duration_hours",
                  "interval_histogram_top", "duplicate_timestamps", "nonmonotonic_steps", "ts_parse_failures",
                  "n_features_excl_time", "n_continuous", "label_like_columns", "constant_columns",
                  "discrete_columns", "columns_with_missing", "columns_with_inf", "first_columns"):
            if k in info:
                log(f"  {k}: {info[k]}")
        if sp:
            prof = setpoint_profile(f, sp)
            manifest["setpoints"][f] = prof
            log("  documented set points (operating-point occupancy):")
            for c, pr in prof.items():
                log(f"    {c}: range_obs=[{pr['observed_min']}, {pr['observed_max']}] doc={pr['documented_range']} "
                    f"changes={pr['n_changes']} distinct={pr['n_distinct_values']} top_hours={pr['top_values_hours']}")
        elif info.get("status") == "PASS":
            log("  none of the documented set-point tags found by name")
    for f in files:
        if not f.startswith("label"):
            continue
        test = f.replace("label-", "hai-")
        log(f"\n=== {f}  (paired with {test}) ===")
        res = audit_labels(manifest["files"].get(test), os.path.join(DATA, f), os.path.join(DATA, test))
        manifest["labels"][f] = res
        for k, v in res.items():
            log(f"  {k}: {v}")
    for s in sorted(x for x in os.listdir(DATA) if x.startswith("summary")):
        with open(os.path.join(DATA, s), "rb") as fh:
            raw = fh.read()
        log(f"\n=== {s} ({len(raw)} bytes) ===")
        log("  LFS pointer, not downloaded" if raw.startswith(LFS_HEADER) else raw.decode("utf-8", "replace")[:3000])
    fails = [f for f, i in manifest["files"].items() if i.get("status") != "PASS"]
    manifest["audit_status"] = "PASS" if not fails else f"FAIL: {fails}"
    log(f"\nAUDIT STATUS: {manifest['audit_status']}   ({time.time() - t0:.0f} s)")
    finish(manifest)


def finish(manifest):
    with open(os.path.join(OUT, "hai2305_data_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(os.path.join(OUT, "hai2305_audit_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\nWrote", os.path.join(OUT, "hai2305_audit_report.txt"), "and hai2305_data_manifest.json")


if __name__ == "__main__":
    main()
