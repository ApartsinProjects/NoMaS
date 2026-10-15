import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# plotted order matches the original figure
mods   = ["Time-series", "Image", "Tabular", "Text"]
random = [0.020, 0.074, 0.093, 0.082]
fixed  = [0.002, 0.013, 0.022, 0.022]   # Table 5 fixed-ensemble (text corrected 0.065 -> 0.022)
auto   = [0.002, 0.005, 0.020, 0.006]
autoerr= [0.000, 0.000, 0.001, 0.002]
redlbl = ["-90%", "-93%", "-78%", "-93%"]   # auto vs random

c_rand, c_fixed, c_auto = "#c8cdd2", "#6d84a0", "#14385c"
x = np.arange(len(mods)); w = 0.26
fig, ax = plt.subplots(figsize=(9.75, 5.46), dpi=100)
ax.bar(x - w, random, w, color=c_rand,  label="Random pick")
ax.bar(x,     fixed,  w, color=c_fixed, label="ADRank (fixed ensemble)")
ax.bar(x + w, auto,   w, color=c_auto,  label="ADRank (auto-calibrated)",
       yerr=autoerr, ecolor="#111418", capsize=3)
for i in range(len(mods)):
    ax.text(x[i] + w, auto[i] + autoerr[i] + 0.0028, redlbl[i], ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#14385c")
ax.set_xticks(x); ax.set_xticklabels(mods, fontsize=12)
ax.set_ylabel("regret@1 (gap to best detector)", fontsize=12)
ax.set_title("Auto-calibration (discriminative regime weighting) matches the oracle",
             fontsize=14, color="#111418", pad=14)
ax.legend(loc="upper left", fontsize=10, frameon=True)
ax.set_ylim(0, 0.099)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", labelsize=10)
plt.tight_layout()
plt.savefig("docs/figures/fig4_regret.png", dpi=100, facecolor="white")
print("wrote docs/figures/fig4_regret.png")
