# -*- coding: utf-8 -*-
"""Figure 6: grouped three-pool selection-regret bar, rendering the audited Table 6 numbers exactly.
Methods ordered by classical-pool regret; three bars each (classical 25 / deep 7 / full 32); reference
lines at the classical best-fixed-detector (0.130) and classical random pick (0.219)."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# --- Table 6 (mean regret, 151 tasks, 200-point probe) ---
ROWS = [  # name, classical, deep, full
    ("Goswami injection", 0.122, 0.169, 0.151),
    ("Entropy (EM)",      0.163, 0.196, 0.176),
    ("Mass-volume (MV)",  0.165, 0.188, 0.174),
    ("Consensus",         0.218, 0.244, 0.232),
    ("HITS",              0.219, 0.259, 0.235),
    ("Model-centrality",  0.223, 0.243, 0.233),
    ("IREOS",             0.233, 0.218, 0.240),
    ("IFOREST-R",         0.241, 0.275, 0.241),
]
FIXED_CLS, RANDOM_CLS = 0.130, 0.219
ROWS.sort(key=lambda r: r[1])
names = [r[0] for r in ROWS]
cls = np.array([r[1] for r in ROWS]); deep = np.array([r[2] for r in ROWS]); full = np.array([r[3] for r in ROWS])
x = np.arange(len(ROWS)); w = 0.26
C_CLS, C_FULL, C_DEEP = "#1b4965", "#5b8aa6", "#ccd6dd"
fig, ax = plt.subplots(figsize=(10, 5.2))
b1 = ax.bar(x - w, cls,  w, color=C_CLS,  edgecolor="white", label="Classical pool (25)")
b2 = ax.bar(x,     full, w, color=C_FULL, edgecolor="white", label="Full pool (32)")
b3 = ax.bar(x + w, deep, w, color=C_DEEP, edgecolor="white", label="Deep-only pool (7)")
for bars in (b1, b2, b3):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004, f"{r.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.2)
l_fix = ax.axhline(FIXED_CLS, color="#c1121f", ls="--", lw=1.2, label="best fixed detector (LOF k=10), classical")
l_rnd = ax.axhline(RANDOM_CLS, color="#6b7280", ls=":", lw=1.2, label="random pick, classical")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5, rotation=18, ha="right")
ax.set_ylabel("selection regret (base-rate-normalized AP; lower is better)", fontsize=10)
ax.set_xlim(-0.6, len(ROWS) - 0.4)
ax.set_ylim(0, 0.285); ax.set_axisbelow(True); ax.yaxis.grid(True, color="#e6e6e6", lw=0.7)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("On the matched classical pool no selector significantly beats the fixed detector; Goswami comes closest,\n"
             "and adding the deep detectors raises every selector's regret", fontsize=11.5)
ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2, columnspacing=1.4, handlelength=1.6)
fig.tight_layout()
for out in [r"E:\Projects\Submitted\ADRank\docs\figures\fig_leaderboard.png",
            r"E:\Projects\Submitted\ADRank\_tex\figures\fig_leaderboard.png",
            r"E:\Projects\Submitted\ADRank\paper\latex\figures\fig_leaderboard.png"]:
    if os.path.isdir(os.path.dirname(out)):
        fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
