"""Constraint residuals of observed telemetry without PyTorch: the same arithmetic as scripts/tailguard_train.py
(Data.B_terms and Data.resid_obs). extended_analyses.py checks that it reproduces the saved constraint stream exactly."""
import numpy as np


def constraint_matrix(rels, targets, raw):
    n = raw.shape[0]; cols = []
    for r in rels:
        ti = targets.index(r["target"][0]); mode = r["target"][1]; B = np.full(n, r["intercept"])
        if mode.startswith("diff"):
            k = int(mode[4:] or 1); B += np.r_[np.full(k, np.nan), raw[:-k, ti]]
        lag0 = []
        for (tag, lag, trf), c in zip(r["terms"], r["coef"]):
            j = targets.index(tag)
            if lag == 0: lag0.append((j, c, trf)); continue
            v = raw[:, j]
            if isinstance(lag, str):
                k = int(lag[4:]); cs = np.r_[0.0, np.cumsum(v)]; mv = np.full(n, np.nan); mv[k:] = (cs[k:n] - cs[:n - k]) / k; v = mv
            else:
                v = np.r_[np.full(lag, np.nan), v[:-lag]]
            B += c * (np.sqrt(np.maximum(v, 0)) if trf == "sqrt" else v)
        rr = raw[:, ti] - np.nan_to_num(B)
        for j, c, trf in lag0:
            rr = rr - c * (np.sqrt(np.maximum(raw[:, j], 0)) if trf == "sqrt" else raw[:, j])
        cols.append(np.abs(rr) / r["scale_a"])
    return np.column_stack(cols).astype(np.float32)

