"""Descriptive (post-hoc): ridge forecaster fused with the documented-constraint stream (Section 4.3).
  TAILGUARD_PREFIX=hai2305|hai2204 python ridge_constraint_fusion.py <root> <out.csv> <scores.npz of TailGuard seed 17> <max-score scores.npz>"""
import os, sys, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'score_ablation.py')).read().split('rows = []')[0].replace('test labels read by score_ablation.py (descriptive score ablation)','test labels read by ridge_constraint_fusion.py (descriptive, post-hoc ridge + constraint stream)'))
VD = sys.argv[3]; VM = sys.argv[4]
zv = np.load(VD); zm = np.load(VM)
v3t, v3e = zv['v_train'], zv['v_test']; v1t, v1e = (zm['max_v_train'], zm['max_v_test']) if 'max_v_train' in zm.files else (zm['v_train'], zm['v_test'])
X = np.hstack([tr["Z"], tr["C"], tr["D"]]); Xte = np.hstack([te["Z"], te["C"], te["D"]]); Z, Zte = tr["Z"], te["Z"]
def ridge_pred(m, XX, chunk=200000):
    return np.vstack([m.predict(np.nan_to_num(lagged(XX[max(0, k - 60):k + chunk], LAGS)[(k - max(0, k - 60)):], copy=False)) for k in range(0, len(XX), chunk)]).astype(np.float32)
idx = np.where(fitm)[0][::4]; F = np.nan_to_num(lagged(X, LAGS)[idx], copy=False); m = Ridge(alpha=1.0).fit(F, Z[idx]); del F
E = np.abs(Z - ridge_pred(m, X)); Ete = np.abs(Zte - ridge_pred(m, Xte)); ref = ecdf_ref(E[valm])
def strm(EE, p_, k):
    s = np.concatenate([np.sort(tail_surprise(EE[i:i + 100000], ref), 1)[:, -k:].mean(1) for i in range(0, len(EE), 100000)]); s[p_ < W_MAX] = 0; return causal_smooth(s, 30)
p3t, p3e, p1t, p1e = strm(E, pos, 3), strm(Ete, pos_te, 3), strm(E, pos, 1), strm(Ete, pos_te, 1)
S = {"ridge, prespecified (top-3)": (p3t, p3e), "ridge + constraints, prespecified (beta 0.5)": (.5 * p3t + .5 * v3t, .5 * p3e + .5 * v3e),
     "ridge, max (top-1)": (p1t, p1e), "ridge + constraints, max fusion": (np.maximum(p1t, v1t), np.maximum(p1e, v1e)),
     "constraints only, prespecified": (v3t, v3e), "constraints only, max": (v1t, v1e)}
import pandas as pd; R = evaluate("ridge+v", 17, S); pd.DataFrame(R).to_csv(OUT, index=False)
