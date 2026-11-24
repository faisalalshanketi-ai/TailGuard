"""
TailGuard | Step 4: process-constraint registry (fitting data only).
Each relation comes from the HAI technical manual (control loops, tag descriptions, units). Coefficients are fitted
by robust (Huber) regression on fitting data; a relation is retained only if its 95th-percentile |residual| in every
operating regime is <= 2 x the pooled value, with >= 1,000 fitting targets per regime.
Run: python scripts\\step4_constraints.py   ->  manifests\\constraints.json, manifests\\step4_report.txt
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MAN = os.path.join(ROOT, "manifests"); DER = os.path.join(ROOT, "data", "derived")
RATIO, MIN_N, MIN_R2, WINDOWS = 2.0, 1000, 0.5, (1, 3, 10, 30)

# target: (tag, mode) with mode "level" (x_t) or "diff" (x_t - x_{t-1}); terms: (tag, lag, transform)
REGISTRY = [
    # level forms: a biased or frozen sensor breaks the relation for the whole fault, not only at its onset.
    # "meanK" = mean over the previous K seconds; K in {1,3,10,30} is chosen by fitting-data R^2 (WINDOWS).
    *[{"id": f"valve_{v}", "class": "actuator tracking", "evidence": "manual: P1 control loops; *_D position command, *_Z current position (%)",
       "target": [f"P1_{v}Z", "level"], "terms": [[f"P1_{v}D", "meanK", "id"]]}
      for v in ("FCV01", "FCV02", "FCV03", "LCV01", "PCV01", "PCV02")],
    *[{"id": f"flowconv_{f}", "class": "sensor conversion", "evidence": f"manual: P1_{f}Z is the flow rate (l/h) converted from differential pressure P1_{f} (mmH2O)",
       "target": [f"P1_{f}Z", "level"], "terms": [[f"P1_{f}", 0, "sqrt"], [f"P1_{f}", 0, "id"]]}
      for f in ("FT01", "FT02", "FT03")],
    {"id": "mass_P1_return_tank", "class": "mass balance", "evidence": "manual: LIT01 level of the return water tank; FT01Z inflow, FT03Z outflow (l/h)",
     "target": ["P1_LIT01", "diff30"], "terms": [["P1_FT01Z", "mean30", "id"], ["P1_FT03Z", "mean30", "id"]],
     "sign_check": {"P1_FT01Z": 1, "P1_FT03Z": -1}},
    {"id": "mass_P3_upper_tank", "class": "mass balance", "evidence": "manual: P3_LIT01 upper-tank level, P3_FIT01 inflow, outflow via LCV01 valve / LCP01 pump",
     "target": ["P3_LIT01", "diff30"], "terms": [["P3_FIT01", "mean30", "id"], ["P3_LCV01D", "mean30", "id"], ["P3_LCP01D", "mean30", "id"]],
     "sign_check": {"P3_FIT01": 1}},
    {"id": "power_P4_steam", "class": "generator tracking", "evidence": "manual: P4 steam-turbine generator output power follows its load demand",
     "target": ["P4_ST_PO", "level"], "terms": [["P4_ST_LD", "meanK", "id"]]},
    {"id": "power_P4_hydro", "class": "generator tracking", "evidence": "manual: P4 hydro-turbine generator output power follows its scheduled demand",
     "target": ["P4_HT_PO", "level"], "terms": [["P4_HT_PS", "meanK", "id"]]},
    {"id": "speed_P2_turbine", "class": "rotor tracking", "evidence": "manual: P2 speed controller output SCO drives turbine speed SIT01 (RPM)",
     "target": ["P2_SIT01", "level"], "terms": [["P2_SCO", "meanK", "id"]]},
]


def design(df, rel):
    cols = []
    for tag, lag, tr in rel["terms"]:
        v = df[tag].to_numpy(np.float64)
        if isinstance(lag, str):                       # "meanK": mean of v[t-K .. t-1]
            k = int(lag[4:]); c = np.r_[0.0, np.cumsum(v)]
            m = np.full(len(v), np.nan); m[k:] = (c[k:len(v)] - c[:len(v) - k]) / k; v = m
        elif lag: v = np.r_[np.full(lag, np.nan), v[:-lag]]
        cols.append(np.sqrt(np.maximum(v, 0)) if tr == "sqrt" else v)
    tag, mode = rel["target"]; y = df[tag].to_numpy(np.float64)
    if mode.startswith("diff"):
        k = int(mode[4:] or 1); y = y - np.r_[np.full(k, np.nan), y[:-k]]
    return np.column_stack(cols), y


def main():
    df = pd.read_parquet(os.path.join(DER, os.environ.get("TAILGUARD_PREFIX", "hai2305") + "_train.parquet"))
    m = json.load(open(os.path.join(MAN, "split_manifest.json")))
    usable = set(m["feature_roles"]["continuous_targets"]) | set(m["feature_roles"]["context_setpoints"])
    elig = df["eligible"].to_numpy(); part = df["partition"].to_numpy(); reg = df["regime"].to_numpy()
    fit = elig & (part == "fit")
    out, lines = [], ["TailGuard Step 4 constraint registry (fitting data only)"]
    for rel in REGISTRY:
        tags = {rel["target"][0]} | {t[0] for t in rel["terms"]}
        if not tags <= usable:
            rel.update(status="unavailable", reason=f"tags not usable: {sorted(tags - usable)}"); out.append(rel); continue
        if any(t[1] == "meanK" for t in rel["terms"]):
            best = None
            for K in WINDOWS:
                cand = dict(rel, terms=[[t[0], f"mean{K}" if t[1] == "meanK" else t[1], t[2]] for t in rel["terms"]])
                Xc, yc = design(df, cand); ii = np.where(fit)[0][::10]; ii = ii[~np.isnan(Xc[ii]).any(1) & ~np.isnan(yc[ii])]
                h = HuberRegressor(max_iter=300).fit((Xc[ii] - Xc[ii].mean(0)) / (Xc[ii].std(0) + 1e-12), yc[ii])
                pr = h.predict((Xc[ii] - Xc[ii].mean(0)) / (Xc[ii].std(0) + 1e-12)); r2 = 1 - np.mean((yc[ii] - pr) ** 2) / np.var(yc[ii])
                if best is None or r2 > best[0]: best = (r2, K)
            rel["terms"] = [[t[0], f"mean{best[1]}" if t[1] == "meanK" else t[1], t[2]] for t in rel["terms"]]
            rel["window_s"] = best[1]
        X, y = design(df, rel)
        idx = np.where(fit)[0][::10]
        idx = idx[~np.isnan(X[idx]).any(1) & ~np.isnan(y[idx])]
        mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-12
        hr = HuberRegressor(epsilon=1.35, alpha=1e-6, max_iter=500).fit((X[idx] - mu) / sd, y[idx])
        coef = hr.coef_ / sd; icpt = hr.intercept_ - (hr.coef_ * mu / sd).sum()
        r = y - (X @ coef + icpt)
        rf = r[fit & ~np.isnan(r)]
        mad = 1.4826 * np.median(np.abs(rf - np.median(rf)))
        floor = 1e-3 * (np.nanstd(y[fit]) + 1e-12)
        a = float(max(mad, floor))
        ss_res = np.mean(rf ** 2); ss_tot = np.nanvar(y[fit])
        q_pool = np.percentile(np.abs(rf), 95)
        per = {}
        for g in np.unique(reg[fit]):
            mk = fit & (reg == g) & ~np.isnan(r)
            if mk.sum() >= MIN_N:
                per[["low", "mid", "high"][g]] = float(np.percentile(np.abs(r[mk]), 95) / max(q_pool, 1e-12))
        worst = max(per.values()) if per else np.inf
        sign_ok = all(np.sign(coef[[t[0] for t in rel["terms"]].index(tag)]) == sgn
                      for tag, sgn in rel.get("sign_check", {}).items())
        r2 = float(1 - ss_res / ss_tot)
        keep = worst <= RATIO and len(per) >= 2 and sign_ok and r2 >= MIN_R2
        rel.update(coef=coef.tolist(), intercept=float(icpt), scale_a=a, r2_fit=float(1 - ss_res / ss_tot),
                   q95_ratio_by_regime=per, worst_ratio=float(worst),
                   status="retained" if keep else "rejected",
                   reason="" if keep else ("sign contradicts documented physics" if not sign_ok
                                           else f"not identifiable: fitting R2 {r2:.2f} < {MIN_R2}" if r2 < MIN_R2
                                           else f"regime instability: max q95 ratio {worst:.2f} > {RATIO}"))
        out.append(rel)
    with open(os.path.join(MAN, "constraints.json"), "w") as f:
        json.dump({"screen_ratio": RATIO, "min_targets_per_regime": MIN_N, "relations": out}, f, indent=2)
    for rel in out:
        if rel["status"] == "unavailable":
            lines.append(f"{rel['id']:22s} UNAVAILABLE  {rel['reason']}")
        else:
            lines.append(f"{rel['id']:22s} {rel['status'].upper():9s} R2={rel['r2_fit']:.3f}  regime q95 ratios "
                         + ", ".join(f"{k}:{v:.2f}" for k, v in rel["q95_ratio_by_regime"].items()) + f"  {rel['reason']}")
    n = sum(r["status"] == "retained" for r in out)
    lines.append(f"Retained {n} of {len(out)} relations.")
    print("\n".join(lines))
    open(os.path.join(MAN, "step4_report.txt"), "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    main()
