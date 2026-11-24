"""Draw Figures 1-6 of the TailGuard paper. Figures 3, 5 and 6 read every plotted value from results/reports/*.csv,\nFigure 4 from the alarm-budget sweeps in analysis/results/.
Usage: python paper/make_figures.py [results/reports] [paper/figures]
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="?", default=os.path.join(ROOT, "results", "reports"))
ap.add_argument("out", nargs="?", default=os.path.join(HERE, "figures"))
ap.add_argument("--ref-a", default="[22]", help="citation label of Radovanovic et al., Reality Gap Analysis (MAKE 8(9))")
ap.add_argument("--max-results", default=os.path.join(ROOT, "results", "reports_max"), help="tables of the max tail-surprise analysis (Section 4.5); skipped if absent")
ap.add_argument("--ref-b", default="[23]", help="citation label of Radovanovic et al., Reality Gap Taxonomy (MAKE 8(10))")
A = ap.parse_args(); RES, OUT, RA, RB = A.results, A.out, A.ref_a, A.ref_b
os.makedirs(OUT, exist_ok=True)

def ms(cell):
    p = str(cell).split("±"); return float(p[0]), (float(p[1]) if len(p) > 1 else 0.0)
COM = pd.read_csv(os.path.join(RES, "table_common_protocol.csv"), index_col=0)
PUB = pd.read_csv(os.path.join(RES, "table_published_protocols.csv"))
REG = pd.read_csv(os.path.join(RES, "table_regime_holdout.csv"))
P95, FPR5 = "[14] p95-validation threshold, Test 1", "[14]/[15] fixed 5% FPR on test-normal, Test 1"
def pub(proto, method, col="F1", T=None):
    T = PUB if T is None else T
    return ms(T[(T.protocol == proto) & (T.method == method)][col].iloc[0])
MAXR = A.max_results if os.path.isdir(A.max_results) else None
if MAXR:
    COMX = pd.read_csv(os.path.join(MAXR, "table_common_protocol.csv"), index_col=0)
    PUBX = pd.read_csv(os.path.join(MAXR, "table_published_protocols.csv"))
def com(m, col): return ms(COM.loc[m, col])
def regv(r, m, col): return ms(REG[(REG.regime == r) & (REG.method == m)][col].iloc[0])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.7,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
BLUE, ORANGE, AQUA, GRAY, LGRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#d8d7d2", "#0b0b0b", "#52514e"
NAVY = "#173f7a"

def box(ax, x, y, w, h, title, body="", fc="#ffffff", ec=INK2, tc=INK, lw=0.9, bold=True, fs=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=lw))
    if body:
        ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=fs + 0.4, fontweight="bold" if bold else "normal", color=tc)
        ax.text(x + w / 2, y + h - 0.085, body, ha="center", va="top", fontsize=fs - 0.6, color=INK2, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc)

def arrow(ax, x1, y1, x2, y2, color=INK2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, color=color, lw=0.9, linestyle=ls,
                                 shrinkA=0, shrinkB=0))

# ---------------- Figure 1: training (a) and detection (b) ----------------
# Faithful to the code: regimes enter the objective as environments; the constraint registry is fitted on fitting
# data and used in training (constraint risk on predictions) and detection (residuals of observed telemetry);
# validation data give the tail-surprise reference; calibration data give the threshold.
fig, ax = plt.subplots(figsize=(7.2, 4.3)); ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
FS = 8.2; W = 0.11; H = 0.11
WHITE, BLUEF, ORANGEF, GRAYF = "#ffffff", "#eaf2fb", "#fff4ee", "#f2f2f0"
def B(x, y, t, kind="data"):
    fc, ec = {"data": (GRAYF, INK2), "model": (BLUEF, BLUE), "phys": (ORANGEF, ORANGE), "step": (WHITE, INK2)}[kind]
    box(ax, x, y, W, H, t, fc=fc, ec=ec, fs=FS); return (x, y)
def R(b): return (b[0] + W, b[1] + H / 2)
def L(b): return (b[0], b[1] + H / 2)
def T(b): return (b[0] + W / 2, b[1] + H)
def Bo(b): return (b[0] + W / 2, b[1])
def line(p, q, arrowhead=True, ls="-"):
    arrow(ax, p[0], p[1], q[0], q[1], style="-|>" if arrowhead else "-", ls=ls)
def elbow(p, q, via="h", ls="-"):
    m = (q[0], p[1]) if via == "h" else (p[0], q[1])
    line(p, m, False, ls); line(m, q, True, ls)

ax.text(0.0, 0.985, "(a) Training", fontsize=9.2, fontweight="bold", color=INK, va="top")
ax.text(0.0, 0.53, "(b) Detection", fontsize=9.2, fontweight="bold", color=INK, va="top")
ax.plot([0, 1], [0.555, 0.555], color=LGRAY, lw=0.8)
# (a) training
fitd = B(0.00, 0.79, "Fitting\ndata")
regm = B(0.27, 0.79, "Operating\nregimes", "step")
reg_a = B(0.27, 0.60, "Constraint\nrisk", "phys")
obj = B(0.56, 0.695, "Training\nobjective", "model")
gru_a = B(0.83, 0.695, "Causal GRU\nforecaster", "model")
line(R(fitd), L(regm)); elbow((fitd[0] + W / 2, fitd[1]), L(reg_a), via="v")
elbow(R(regm), (obj[0] + W / 2, obj[1] + H), via="h"); elbow(R(reg_a), (obj[0] + W / 2, obj[1]), via="h")
arrow(ax, R(obj)[0], R(obj)[1], L(gru_a)[0], L(gru_a)[1], style="<|-|>")
# (b) detection
cols = [0.00, 0.165, 0.305, 0.47, 0.61, 0.75, 0.89]
yC, yD, yM = 0.235, 0.04, 0.1375
tele = B(cols[0], yM, "Test\ntelemetry")
gru_b = B(cols[1], yC, "Trained\nGRU", "model"); reg_b = B(cols[1], yD, "Constraint\nregistry", "phys")
fres = B(cols[2], yC, "Forecast\nresiduals", "step"); cres = B(cols[2], yD, "Constraint\nresiduals", "phys")
tail = B(cols[3], yM, "Tail\nsurprise", "step"); fuse = B(cols[4], yM, "Fused\nscore", "step")
thr = B(cols[5], yM, "Threshold τ", "model"); alm = B(cols[6], yM, "Alarm", "step")
valid = B(cols[3], 0.345, "Validation\ndata"); calib = B(cols[5], 0.345, "Calibration\ndata")
xm = (cols[0] + W + cols[1]) / 2
line(R(tele), (xm, R(tele)[1]), False); line((xm, yD + H / 2), (xm, yC + H / 2), False)
line((xm, yC + H / 2), L(gru_b)); line((xm, yD + H / 2), L(reg_b))
line(R(gru_b), L(fres)); line(R(reg_b), L(cres))
xj = (cols[2] + W + cols[3]) / 2
line(R(fres), (xj, R(fres)[1]), False); line(R(cres), (xj, R(cres)[1]), False)
line((xj, yD + H / 2), (xj, yC + H / 2), False); line((xj, yM + H / 2), L(tail))
line(R(tail), L(fuse)); line(R(fuse), L(thr)); line(R(thr), L(alm))
line(Bo(valid), T(tail)); line(Bo(calib), T(thr))
fig.savefig(f"{OUT}/fig1_pipeline.png"); plt.close(fig)

# ---------------- Figure 2: constraint map (names only) ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
def panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", fc="#ffffff", ec=INK2, lw=0.9))
    ax.text(x + 0.012, y + h - 0.03, title, va="top", fontweight="bold", fontsize=8.8)
def item(x, y, w, text, keep=True):
    ax.add_patch(FancyBboxPatch((x, y), w, 0.066, boxstyle="round,pad=0.003,rounding_size=0.006",
                 fc="#eaf2fb" if keep else "#f2f2f0", ec=BLUE if keep else GRAY, lw=0.7, ls="-" if keep else "--"))
    ax.text(x + 0.012, y + 0.033, text, va="center", fontsize=8.2, color=INK if keep else INK2)
panel(0.0, 0.02, 0.50, 0.96, "P1 Boiler")
rows = ["FCV01 valve tracking", "FCV02 valve tracking", "FCV03 valve tracking", "LCV01 valve tracking",
        "PCV01 valve tracking", "FT01–FT03 flow conversion", "Return-tank mass balance"]
y = 0.83
for r in rows: item(0.015, y, 0.47, r); y -= 0.085
item(0.015, y - 0.02, 0.47, "PCV02 valve tracking", keep=False)
panel(0.52, 0.70, 0.48, 0.28, "P2 Turbine")
item(0.535, 0.78, 0.45, "Speed tracking")
panel(0.52, 0.36, 0.48, 0.30, "P3 Water treatment")
item(0.535, 0.45, 0.45, "Tank balance", keep=False)
panel(0.52, 0.02, 0.48, 0.30, "P4 HIL power simulation")
item(0.535, 0.14, 0.45, "Steam power tracking")
item(0.535, 0.055, 0.45, "Hydro power tracking", keep=False)
fig.savefig(f"{OUT}/fig2_constraints.png"); plt.close(fig)

# ---------------- Figure 3: published comparison ----------------
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
names = [f"USAD {RB}", f"GNN+BiLSTM {RB}", f"GNN+BiLSTM {RA}", "TailGuard"]
src = [pub(FPR5, "[15] USAD"), pub(FPR5, "[15] GNN+BiLSTM"), pub(FPR5, "[14] GNN+BiLSTM"), pub(FPR5, "TailGuard")]
cs = [GRAY, GRAY, GRAY, BLUE]
if MAXR: names.append("TailGuard, max score"); src.append(pub(FPR5, "TailGuard", T=PUBX)); cs.append(NAVY)
vals = [v for v, _ in src]
ax.barh(names, vals, color=cs, height=0.55, xerr=[e for _, e in src], error_kw={"lw": 0.8})
for i, v in enumerate(vals): ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
ax.set_xlim(0, 0.85); ax.set_xlabel("F1 at fixed 5% false-positive rate"); ax.set_title("(a) Fixed-FPR protocol, Test 1", loc="left")
ax.grid(axis="x", color=LGRAY, lw=0.5); ax.set_axisbelow(True)
ax = axs[1]
x = np.arange(2); w = 0.36
pubv = [pub(P95, "best published (GNN)")[0], pub(P95, "best published (GNN)", "AP")[0]]
ours = [pub(P95, "TailGuard")[0], pub(P95, "TailGuard", "AP")[0]]; sd = [pub(P95, "TailGuard")[1], pub(P95, "TailGuard", "AP")[1]]
if MAXR: w = 0.26
ax.bar(x - w - 0.01 if MAXR else x - w / 2 - 0.01, pubv, w, color=GRAY, label=f"Best published GNN {RA}")
ax.bar(x if MAXR else x + w / 2 + 0.01, ours, w, color=BLUE, yerr=sd, error_kw={"lw": 0.8}, label="TailGuard")
for i in range(2):
    ax.text(x[i] - w - 0.01 if MAXR else x[i] - w / 2 - 0.01, pubv[i] + 0.015, f"{pubv[i]:.3f}", ha="center", fontsize=7.2)
    ax.text(x[i] if MAXR else x[i] + w / 2 + 0.01, ours[i] + 0.025, f"{ours[i]:.3f}", ha="center", fontsize=7.2)
if MAXR:
    ox = [pub(P95, "TailGuard", T=PUBX)[0], pub(P95, "TailGuard", "AP", T=PUBX)[0]]; osd = [pub(P95, "TailGuard", T=PUBX)[1], pub(P95, "TailGuard", "AP", T=PUBX)[1]]
    ax.bar(x + w + 0.01, ox, w, color=NAVY, yerr=osd, error_kw={"lw": 0.8}, label="TailGuard, max score")
    for i in range(2): ax.text(x[i] + w + 0.01, ox[i] + 0.025, f"{ox[i]:.3f}", ha="center", fontsize=7.2)
ax.set_xticks(x, ["F1", "AUC-PR"]); ax.set_ylim(0, 1.18 if MAXR else 0.9); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_title("(b) Validation-percentile protocol, Test 1", loc="left")
ax.grid(axis="y", color=LGRAY, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7.2, loc="upper left", ncol=2 if MAXR else 1, columnspacing=0.8, handlelength=1.2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_published.png"); plt.close(fig)

# ---------------- Figure 4: event recall against realised false alarms over the alarm-budget sweep ----------------
AR = os.path.join(ROOT, "analysis", "results")
SW = [("HAI 23.05 (52 attacks)", os.path.join(AR, "hai2305", "budget_sensitivity.csv")),
      ("HAI 22.04 (58 attacks)", os.path.join(AR, "hai2204", "budget_sensitivity_hai2204.csv"))]
CURVES = [("TailGuard", BLUE, "o", "-", 2.0), ("TailGuard, max score", BLUE, "o", "--", 1.4), ("Matched GRU", AQUA, "D", "-", 1.1),
          ("Ridge AR", INK, "^", "-", 1.3), ("USAD", ORANGE, "s", "-", 1.0), ("TranAD", "#b3471d", "v", "-", 1.0), ("GDN", "#e8a33d", "P", "-", 1.0)]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, (title, f) in zip(axs, SW):
    d = pd.read_csv(f)
    for name, c, m, ls, lw in CURVES:
        g = d[d.method == name].sort_values("budget_per_day")
        if g.empty or (name == "GDN" and "22.04" in title): continue
        ax.plot(g.FA24, g.event_recall, ls=ls, color=c, lw=lw, marker=m, ms=3.6, mec="white", mew=0.5, label=name, zorder=3)
        one = g[g.budget_per_day == 1]
        ax.plot(one.FA24, one.event_recall, m, color=c, ms=7, mec=INK, mew=0.9, zorder=4)
    ax.axvline(1.0, color=INK2, lw=0.7, ls=":"); ax.set_xlim(0, 3.5); ax.set_ylim(0, 0.75)
    ax.set_title(title); ax.set_xlabel("Realised false alarms per day (FA24)"); ax.grid(color=LGRAY, lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel("Event recall")
hh, ll = axs[0].get_legend_handles_labels()
fig.legend(hh, ll, frameon=False, fontsize=7.4, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{OUT}/fig4_tradeoff.png"); plt.close(fig)

# ---------------- Figure 5: ablations ----------------
ABL = [("TailGuard (full)", "TailGuard"), ("A1 λ_phys = 0 (β = 0)", "A1_no_physics"), ("A2 λ_inv = 0", "A2_no_invariance"),
       ("A3 pooled, λ_inv = 0", "A3_pooled_no_inv"), ("A4 β = 0", "A4_beta0 (temporal score only)"), ("A5 β = 1", "Residual-only (beta=1)"),
       ("A10 Var(R_e)", "A10_whole_forecast_VREx"), ("A11 day environments", "A11_day_environments"), ("Matched GRU (β = 0)", "GRU (pooled ERM)")]
ab = [(n, *com(k, "event_recall"), *com(k, "FA24")) for n, k in ABL]
labels = [a[0].replace("λ_inv", "λ$_{inv}$").replace("λ_phys", "λ$_{phys}$").replace("R_e", "R$_e$") for a in ab][::-1]
rec = [a[1] for a in ab][::-1]; rsd = [a[2] for a in ab][::-1]; fa = [a[3] for a in ab][::-1]; fsd = [a[4] for a in ab][::-1]
cols_ = [BLUE if l.startswith("TailGuard") else (ORANGE if l.startswith("A1 ") else GRAY) for l in labels]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
y = np.arange(len(labels))
axs[0].barh(y, rec, xerr=rsd, color=cols_, height=0.6, error_kw={"lw": 0.7})
axs[0].set_yticks(y, labels); axs[0].set_xlabel("Event recall"); axs[0].set_xlim(0, 0.45); axs[0].set_title("(a) Event recall", loc="left")
axs[1].barh(y, fa, xerr=fsd, color=cols_, height=0.6, error_kw={"lw": 0.7})
axs[1].set_xlabel("False alarms per 24 h"); axs[1].set_xlim(0, 2.0); axs[1].set_title("(b) False alarms", loc="left")
for i in range(len(labels)):
    axs[0].text(rec[i] + rsd[i] + 0.008, y[i], f"{rec[i]:.3f}", va="center", fontsize=7.4)
    axs[1].text(fa[i] + fsd[i] + 0.03, y[i], f"{fa[i]:.2f}", va="center", fontsize=7.4)
for a in axs: a.grid(axis="x", color=LGRAY, lw=0.5); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_ablation.png"); plt.close(fig)

# ---------------- Figure 6: refitted leave-one-regime-out, main false-episode definition ----------------
import subprocess
F6 = os.path.join(ROOT, "analysis", "results", "hai2305", "regime_fa_consistent_runs.csv")
subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "scripts", "fig6.py"), F6, f"{OUT}/fig6_regimes.png"], check=True)
print("figures written to", OUT)
