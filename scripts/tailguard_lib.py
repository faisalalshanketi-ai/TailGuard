"""TailGuard shared library: data loading, alarm calibration, metrics, development injections."""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DER = os.path.join(ROOT, "data", "derived")
MAN = os.path.join(ROOT, "manifests")
RUNS = os.path.join(ROOT, "runs")
PREFIX = os.environ.get("TAILGUARD_PREFIX", "hai2305")   # derived-data prefix (hai2305 = HAI 23.05, hai2204 = HAI 22.04)


def manifest():
    with open(os.path.join(MAN, "split_manifest.json")) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- data
def load(split="train"):
    """Return dict with scaled arrays. Labels are never loaded here."""
    m = manifest(); roles = m["feature_roles"]; sc = m["scaler_fit"]
    df = pd.read_parquet(os.path.join(DER, f"{PREFIX}_{split}.parquet"))
    tgt, ctx, disc = roles["continuous_targets"], roles["context_setpoints"], roles["discrete_actuators"]
    Z = np.column_stack([(df[c].to_numpy(np.float64) - sc[c]["median"]) / sc[c]["scale"] for c in tgt]).astype(np.float32)
    C = np.column_stack([(df[c].to_numpy(np.float64) - sc[c]["median"]) / sc[c]["scale"] for c in ctx]).astype(np.float32)
    D = []
    for c in disc:
        lv = m["discrete_levels_fit"][c]; v = df[c].to_numpy()
        D += [(v == l).astype(np.float32) for l in lv]
        D.append((~np.isin(v, lv)).astype(np.float32))          # unseen level indicator
    D = np.column_stack(D).astype(np.float32)
    out = {"df_ids": df[[c for c in ("timestamp", "file_id", "partition", "segment_id", "pos_in_segment",
                                     "eligible", "regime", "env_id") if c in df.columns]].copy(),
           "Z": Z, "C": C, "D": D, "targets": tgt}
    return out


def lagged(X, lags):
    """Stack X shifted by each lag (rows are contiguous per segment; eligibility guarantees lag < pos)."""
    cols = []
    for L in lags:
        S = np.empty_like(X); S[L:] = X[:-L]; S[:L] = np.nan
        cols.append(S)
    return np.concatenate(cols, axis=1)


# ----------------------------------------------------------------------------- alarms
def episodes(alarm, seg):
    """Boolean start-of-episode mask. alarm, seg: arrays over eligible rows in time order."""
    alarm = np.asarray(alarm, bool); seg = np.asarray(seg)
    prev = np.r_[False, alarm[:-1]]
    newseg = np.r_[True, seg[1:] != seg[:-1]]
    return alarm & (~prev | newseg)


def calibrate_threshold(score, seg, hours, budget_per_24h=1.0, n_cand=3000):
    """Smallest tau such that every tau' >= tau gives <= budget episodes on normal calibration data
    (monotone closure: scanning down from the maximum and stopping at the first violation, so a detector
    that alarms continuously cannot pass by producing one long episode). Comparison is strict (s > tau)."""
    score = np.asarray(score, np.float64)
    allowed = budget_per_24h * hours / 24.0
    top = np.nextafter(score.max(), np.inf)
    qs = np.unique(np.quantile(score, np.linspace(0.90, 1.0, n_cand)))[::-1]
    tau = top
    for q in qs:
        k = episodes(score > q, seg).sum()
        if k > allowed:
            break
        tau = q
    return float(tau), float(allowed)


def alarm_stats(alarm, seg, normal_mask, seconds_per_row=1.0):
    st = episodes(alarm, seg)
    nh = normal_mask.sum() * seconds_per_row / 3600
    false_starts = int((st & normal_mask).sum())
    return {"FA24": 24 * false_starts / nh if nh else np.nan, "false_episodes": false_starts,
            "normal_hours": nh, "false_alarm_sec_per_hour": float((alarm & normal_mask).sum() / nh) if nh else np.nan,
            "normal_FPR": float((alarm & normal_mask).sum() / normal_mask.sum()) if normal_mask.sum() else np.nan}


def event_stats(alarm, event_id, onset_idx=None):
    """event_id: '' for normal rows, an id string for attack rows (contiguous)."""
    ev = pd.Series(event_id)
    ids = [e for e in pd.unique(ev) if e]
    det, delays, durations = 0, [], []
    for e in ids:
        idx = np.where(ev.values == e)[0]
        a = np.asarray(alarm)[idx]
        durations.append(len(idx))
        if a.any():
            det += 1; delays.append(int(np.argmax(a)))
    capped = delays + [d for d, e in zip(durations, ids) if not np.asarray(alarm)[ev.values == e].any()]
    return {"events": len(ids), "detected": det, "event_recall": det / len(ids) if ids else np.nan,
            "delay_median_s": float(np.median(delays)) if delays else np.nan,
            "delay_iqr_s": [float(np.percentile(delays, 25)), float(np.percentile(delays, 75))] if delays else None,
            "capped_delay_median_s": float(np.median(capped)) if capped else np.nan}


def point_stats(alarm, label, score=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    a = np.asarray(alarm, bool); y = np.asarray(label, bool)
    tp = int((a & y).sum()); fp = int((a & ~y).sum()); fn = int((~a & y).sum())
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    out = {"precision": prec, "recall": rec, "F1": f1}
    if score is not None and y.any() and (~y).any():
        out["AP"] = float(average_precision_score(y, score)); out["AUROC"] = float(roc_auc_score(y, score))
    return out


# ----------------------------------------------------------------------------- development injections
def make_injections(n_rows, seg, eligible, n_targets, scale_by_target, seed=17, n_events=36):
    """Development-only sensor manipulations on validation data (never reported as results).
    Types: bias (+/-2 or 4 robust scales), freeze (hold onset value), ramp (to 4 scales), on one target tag."""
    rng = np.random.default_rng(seed)
    elig_idx = np.where(eligible)[0]
    used = np.zeros(n_rows, bool); events = []
    durs = [60, 180, 600]; kinds = ["bias", "freeze", "ramp"]
    tries = 0
    while len(events) < n_events and tries < 10000:
        tries += 1
        d = int(rng.choice(durs)); s = int(rng.choice(elig_idx))
        e = s + d
        if e + 1800 >= n_rows or used[max(0, s - 1800):e + 1800].any() or seg[s] != seg[e] or not eligible[s:e].all():
            continue
        used[s:e] = True
        events.append({"start": s, "end": e, "tag": int(rng.integers(n_targets)), "kind": kinds[len(events) % 3],
                       "mag": float(rng.choice([2.0, 4.0]) * rng.choice([-1, 1]))})
    return events


def apply_injections(Z, events):
    Z = Z.copy(); ev_id = np.full(len(Z), "", dtype=object)
    for k, e in enumerate(events):
        s, t, j = e["start"], e["end"], e["tag"]
        if e["kind"] == "bias":
            Z[s:t, j] += e["mag"]
        elif e["kind"] == "freeze":
            Z[s:t, j] = Z[s - 1, j]
        else:
            Z[s:t, j] += np.linspace(0, e["mag"], t - s, dtype=np.float32)
        ev_id[s:t] = f"dev{k}"
    return Z, ev_id


# ----------------------------------------------------------------------------- common score policy (frozen 2026-10-14)
SCORE_POLICY = {"per_channel": "tail surprise -log10(1-F_j(|e|)), F_j = ECDF of normal validation errors up to q99, exponential (mean-excess) tail above",
                "aggregate": "mean of the 3 largest channel surprises", "smooth_s": 30,
                "reason": "unsmoothed mean of median-scaled errors detected 0/36 development injections for every "
                          "method because legitimate set-point transients dominate calibration; chosen on development "
                          "injections and validation false alarms only, applied identically to all methods"}


def ecdf_ref(E_val, step=5):
    return np.sort(E_val[::step], axis=0)


def tail_surprise(E, ref, tail_q=0.99):
    """-log10 exceedance probability. ECDF of normal validation errors up to the 99th percentile; above it an
    exponential tail fitted to the top 1% (mean excess), so violations beyond anything seen in normal data keep
    growing instead of saturating at log10(n)."""
    n = ref.shape[0]; out = np.empty(E.shape, np.float32)
    k = int(np.floor(tail_q * n))
    for j in range(E.shape[1]):
        col = ref[:, j]; q = col[k]
        me = max(float(np.mean(col[k:] - q)), 1e-3)   # floor: instrument resolution in standardized units
        r = np.searchsorted(col, E[:, j], side="right")
        body = -np.log10(np.maximum(1 - r / (n + 1), 1 / (n + 1)))
        tail = -np.log10(1 - tail_q) + (E[:, j] - q) / (me * np.log(10))
        out[:, j] = np.where(E[:, j] > q, np.maximum(tail, -np.log10(1 - tail_q)), body)
    return out


def causal_smooth(s, w):
    if w <= 1: return s
    c = np.cumsum(np.r_[0.0, s]); m = (c[w:] - c[:-w]) / w
    return np.r_[np.full(w - 1, m[0]), m].astype(np.float32)


def policy_score(T, k=3, w=30):
    s = np.sort(T, axis=1)[:, -k:].mean(1) if T.ndim == 2 else T
    return causal_smooth(s, w)
