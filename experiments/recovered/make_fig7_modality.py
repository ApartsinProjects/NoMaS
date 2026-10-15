# -*- coding: utf-8 -*-
"""Figure 7: SPARC vs best fixed detector vs entropy, by modality, ALL on the classical pool (matched),
1000-point probe. Values from Table 6b. Group annotations do not overlap the legend."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# classical-pool regret (Table 6b): [Time series, OddBench, OvrBench]
GROUPS = ["Time series\n(n=38)", "OddBench\n(n=39)", "OvrBench\n(n=66)"]
SPARC = [0.037, 0.161, 0.135]
FIXED = [0.038, 0.198, 0.145]
ENTRO = [0.145, 0.164, 0.162]
NOTE  = ["not significant", "largest gap (n.s.)", "not significant"]
x = np.arange(3); w = 0.26
C_S, C_F, C_E = "#1b4965", "#b5384a", "#8d99ae"
fig, ax = plt.subplots(figsize=(9.2, 4.8))
b1 = ax.bar(x - w, SPARC, w, color=C_S, edgecolor="white", label="SPARC (ours)")
b2 = ax.bar(x,     FIXED, w, color=C_F, edgecolor="white", label="Best fixed detector")
b3 = ax.bar(x + w, ENTRO, w, color=C_E, edgecolor="white", label="Excess-mass")
for bars in (b1, b2, b3):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004, f"{r.get_height():.2f}",
                ha="center", va="bottom", fontsize=8)
gtop = [max(SPARC[i], FIXED[i], ENTRO[i]) for i in range(3)]
for i in range(3):
    ax.text(x[i], gtop[i] + 0.028, NOTE[i], ha="center", va="bottom", fontsize=9,
            color="#1b4965", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(GROUPS, fontsize=10)
ax.set_ylabel("selection regret (lower is better)", fontsize=10)
ax.set_ylim(0, 0.25); ax.set_axisbelow(True); ax.yaxis.grid(True, color="#e9e9e9", lw=0.7)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("SPARC's regret is at or below the best fixed detector on every source but significantly so on none;\n"
             "the internal metric (excess-mass) is far behind on time series", fontsize=11.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9, frameon=False)
fig.tight_layout()
import os
for out in [r"E:\Projects\Submitted\ADRank\docs\figures\fig_modality.png",
            r"E:\Projects\Submitted\ADRank\_tex\figures\fig_modality.png",
            r"E:\Projects\Submitted\ADRank\paper\latex\figures\fig_modality.png"]:
    if os.path.isdir(os.path.dirname(out)): fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)

