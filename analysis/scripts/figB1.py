import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_csv("sim.csv"); g = d.groupby("rho"); R = sorted(d.rho.unique()); x = np.arange(len(R)); lab = [str(r) for r in R]
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for col, mk, nm, c in (("top3_tau_closure", "o", "Top-3 score", "#1f4e79"), ("mx_tau_closure", "s", "Max score", "#b2182b")):
    m, lo, hi = g[col].mean(), g[col].min(), g[col].max()
    ax[0].errorbar(x, m, yerr=[m - lo, hi - m], marker=mk, color=c, capsize=3, label=nm)
ax[0].axhline(7.19, ls="--", color="k", lw=1, label=r"$\tau_A$")
ax[0].axhspan(7.10, 9.81, color="#1f4e79", alpha=0.12, label="HAI top-3")
ax[0].axhspan(12.34, 17.56, color="#b2182b", alpha=0.12, label="HAI max")
ax[0].set_xticks(x, lab); ax[0].set_xlabel("Lag-1 correlation ρ"); ax[0].set_ylabel("Closure threshold"); ax[0].set_title("(a) Thresholds")
ax[0].legend(fontsize=7, ncol=2, loc="center right", bbox_to_anchor=(1.0, 0.6)); ax[0].set_ylim(0, 19)
for col, nom, mk, nm, c in (("single_c2", 0.01, "o", "Single channel, c = 2", "#1f4e79"), ("single_c3", 0.001, "^", "Single channel, c = 3", "#4393c3"),
                            ("max_c2", 0.01, "s", "Channel maximum, c = 2", "#b2182b"), ("max_c3", 0.001, "D", "Channel maximum, c = 3", "#f4a582")):
    ax[1].plot(x, g[col].mean() / nom, marker=mk, color=c, label=nm)
ax[1].axhline(1, ls="--", color="k", lw=1); ax[1].set_yscale("log"); from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter; ax[1].yaxis.set_major_locator(FixedLocator([0.3,0.5,1,2,5])); ax[1].yaxis.set_minor_locator(NullLocator()); ax[1].yaxis.set_major_formatter(FormatStrFormatter("%g")); ax[1].set_xticks(x, lab)
ax[1].set_xlabel("Lag-1 correlation ρ"); ax[1].set_ylabel("Realised / nominal"); ax[1].set_title("(b) Exceedance"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("fig/figB1_simulation.png", dpi=300)
