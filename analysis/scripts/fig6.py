"""Figure 6: refitted leave-one-regime-out, prespecified score, main false-episode definition."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
RUNS, OUT = sys.argv[1:3]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.facecolor": "white", "axes.edgecolor": "#52514e", "axes.linewidth": 0.7})
BLUE, ORANGE, INK2, LGRAY = "#2a78d6", "#eb6834", "#52514e", "#d8d7d2"
d = pd.read_csv(RUNS); d = d[d.score == "pre"]; d["FA24"] = 24 * d.ep / d.h; d["rec"] = d.det / d.n
RG = ["low", "mid", "high"]; x = np.arange(3); w = 0.2
def ms(model, var, col):
    g = d[(d.model == model) & (d.variant == var)].groupby("regime")[col]; return g.mean().reindex(RG).to_numpy(), g.std().reindex(RG).fillna(0).to_numpy()
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
for k, (m, c, lab) in enumerate((("tailguard", BLUE, "TailGuard"), ("gru", ORANGE, "Matched GRU"))):
    a, s = ms(m, "clean hold-out", "FA24"); xx = x + (k - 0.5) * 0.36
    axs[0].bar(xx, a, 0.34, yerr=s, color=c, label=lab, error_kw={"lw": 0.7})
    for i in range(3): axs[0].text(xx[i], a[i] + s[i] + 0.1, f"{a[i]:.2f}", ha="center", fontsize=7.4)
axs[0].axhline(1.0, color=INK2, lw=0.8, ls="--"); axs[0].text(2.55, 1.1, "budget", fontsize=7.2, color=INK2, ha="right")
axs[0].set_ylim(0, 5.6); axs[0].set_ylabel("False alarms per 24 h"); axs[0].set_title("(a) False alarms, withheld regime", loc="left")
axs[0].legend(frameon=False, fontsize=7.6, loc="upper right")
for k, (m, var, c, h, lab) in enumerate((("tailguard", "all regimes", BLUE, None, "TailGuard, all regimes trained"), ("tailguard", "clean hold-out", BLUE, "////", "TailGuard, regime withheld"),
                                        ("gru", "all regimes", ORANGE, None, "GRU, all regimes trained"), ("gru", "clean hold-out", ORANGE, "////", "GRU, regime withheld"))):
    a, s = ms(m, var, "rec"); xx = x + (k - 1.5) * w
    axs[1].bar(xx, a, w * 0.92, yerr=s, color=c if h is None else "white", edgecolor=c, hatch=h, lw=0.8, label=lab, error_kw={"lw": 0.7})
axs[1].set_ylim(0, 0.62); axs[1].set_ylabel("Event recall"); axs[1].set_title("(b) Recall by regime", loc="left")
axs[1].legend(frameon=False, fontsize=6.6, loc="upper left", ncol=2)
for ax in axs:
    ax.set_xticks(x, [r.capitalize() + " regime" for r in RG]); ax.grid(axis="y", color=LGRAY, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT); print("saved", OUT)
