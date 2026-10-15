# -*- coding: utf-8 -*-
"""Two paper figures on the corrected v3 benchmark (standardized pool, 151 in-benchmark datasets):
FIG_leaderboard: horizontal-bar selection-regret leaderboard, SPARC vs the field, with paired-Wilcoxon
significance vs SPARC. FIG_modality: SPARC vs EM/MV per modality (time series / ovrbench / oddbench)."""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\streamline"
df = pd.read_csv(os.path.join(D, "STREAM_V3_LEADERBOARD.csv"))
keep = (df.oracle_ap > 0.05) & (df.reg_random > 0.02) & (df.n_h >= 100); g = df[keep]
a = g.reg_combined_cls.values
NAV = "#14385c"; GREY = "#8d99ae"; LIGHT = "#ced4da"
# ---- FIG 1: leaderboard ----
board = [("SPARC (ours)", "reg_combined_cls", NAV), ("Entropy (EM)", "reg_em", GREY), ("Mass-volume (MV)", "reg_mv", GREY),
         ("HITS", "reg_hits", LIGHT), ("ModelCentrality", "reg_mc", LIGHT), ("Consensus", "reg_consensus", LIGHT), ("Random pick", "reg_random", "#e0e0e0")]
rows = []
for nm, c, col in board:
    v = g[c].mean(); p = wilcoxon(a, g[c].values).pvalue if c != "reg_combined_cls" else None
    rows.append((nm, v, col, p))
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(8.4, 4.6)); yp = np.arange(len(rows))
ax.barh(yp, [r[1] for r in rows], color=[r[2] for r in rows], edgecolor="#333", linewidth=0.5)
for i, r in enumerate(rows):
    lab = f"{r[1]:.3f}" + ("" if r[3] is None else (f"   p<0.001" if r[3] < 1e-3 else f"   p={r[3]:.3f}"))
    ax.text(r[1] + 0.004, i, lab, va="center", fontsize=9)
ax.set_yticks(yp); ax.set_yticklabels([r[0] for r in rows], fontsize=10); ax.invert_yaxis()
ax.set_xlabel("selection regret (ap_norm; lower is better)"); ax.set_xlim(0, max(r[1] for r in rows) * 1.22)
ax.axvline(g.reg_random.mean(), color="#c00", ls="--", lw=1, alpha=.5)
ax.set_title(f"Model-selection regret on the {len(g)}-dataset ADReal benchmark\nSPARC is the only selector that significantly beats both entropy and mass-volume (p<0.001)", fontsize=10.5)
plt.tight_layout(); fig.savefig(os.path.join(D, "FIG_leaderboard_v3.png"), dpi=125); plt.close()
# ---- FIG 2: per-modality SPARC vs EM/MV ----
mods = [("Time series", "tsbad_m"), ("Tabular (ovrbench)", "ovrbench"), ("Tabular (oddbench)", "oddbench")]
fig, ax = plt.subplots(figsize=(8.4, 4.4)); x = np.arange(len(mods)); w = 0.26
for j, (nm, c, col) in enumerate([("SPARC (ours)", "reg_combined_cls", NAV), ("Entropy (EM)", "reg_em", GREY), ("Mass-volume (MV)", "reg_mv", LIGHT)]):
    vals = [g[g.corpus == k][c].mean() for _, k in mods]; ax.bar(x + (j - 1) * w, vals, w, label=nm, color=col, edgecolor="#333", linewidth=0.5)
    for i, v in enumerate(vals): ax.text(x[i] + (j - 1) * w, v + 0.004, f"{v:.2f}", ha="center", fontsize=8)
ax.set_ylim(0, 0.235)
# significance stars vs EM (fixed height, clear of the title)
for i, (nm, k) in enumerate(mods):
    gg = g[g.corpus == k]; p = wilcoxon(gg.reg_combined_cls.values, gg.reg_em.values).pvalue
    star = "***" if p < 1e-3 else ("*" if p < 0.05 else "n.s.")
    ax.text(x[i], 0.215, f"vs EM: {star}", ha="center", fontsize=9, color=NAV, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"{nm}\n(n={sum(g.corpus==k)})" for nm, k in mods], fontsize=9.5)
ax.set_ylabel("selection regret (lower is better)"); ax.legend(fontsize=9, frameon=False, loc="center left")
ax.set_title("SPARC dominates on time series, ties entropy/mass-volume on tabular\n(the synthetic probes transfer to time-series structure, weakly to coarse tabular features)", fontsize=10.5)
plt.tight_layout(); fig.savefig(os.path.join(D, "FIG_modality_v3.png"), dpi=125); plt.close()
print("saved FIG_leaderboard_v3.png, FIG_modality_v3.png")
print("leaderboard:", [(r[0], round(r[1], 3)) for r in rows])
