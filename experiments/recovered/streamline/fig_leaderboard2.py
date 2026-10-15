# -*- coding: utf-8 -*-
"""Figure 4 (revised): selection regret on all 151 tasks, WITH the best-single-detector baseline
and the Goswami injection baseline. The best default detector (LOF_k10, no per-task selection) is the
bar every method must beat; only SPARC and Goswami's injection selector match it, the internal metrics
fall short, and the agreement selectors fall below random."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
NAV = "#14385c"; PROBE = "#3f7cac"; GREY = "#8d99ae"; LIGHT = "#ced4da"; REF = "#c23b52"
rows = [
    ("Goswami injection [35]", 0.122, PROBE),
    ("SPARC (ours)", 0.124, NAV),
    ("Best single detector (LOF, no selection)", 0.130, REF),
    ("Mass-volume", 0.174, GREY),
    ("Entropy", 0.176, GREY),
    ("Random pick", 0.220, "#e0e0e0"),
    ("Consensus", 0.232, LIGHT),
    ("Model-centrality", 0.233, LIGHT),
    ("HITS", 0.235, LIGHT),
]
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(8.6, 4.7)); y = np.arange(len(rows))
bars = ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], edgecolor="#333", linewidth=0.5)
# hatch the reference bar
for i, r in enumerate(rows):
    if r[0].startswith("Best single"):
        bars[i].set_hatch("///"); bars[i].set_edgecolor("#7a1f2e")
for i, r in enumerate(rows):
    ax.text(r[1] + 0.004, i, f"{r[1]:.3f}", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5); ax.invert_yaxis()
ax.set_xlabel("selection regret (base-rate-normalized average precision; lower is better)")
ax.set_xlim(0, 0.265)
ax.axvline(0.130, color=REF, ls="--", lw=1.1, alpha=.8)
ax.axvline(0.220, color="#999", ls=":", lw=1.0, alpha=.7)
ax.set_title("No selection method significantly beats the best single detector\n"
             "SPARC and Goswami's injection selector match it; entropy and mass-volume fall short",
             fontsize=10.5)
plt.tight_layout()
D = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/streamline"
fig.savefig(D + "/FIG_leaderboard2.png", dpi=125, facecolor="white", bbox_inches="tight")
print("saved")
