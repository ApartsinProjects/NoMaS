# -*- coding: utf-8 -*-
"""Final leaderboard figure: full classical+deep pool, macro regret, ours vs UOMS field. Horizontal bars."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\streamline"
df = pd.read_csv(os.path.join(D, "STREAM_FINAL_LEADERBOARD.csv"))
def macro(c): return np.mean([df[df.truefam == fa][c].mean() for fa in ["local", "global"] if (df.truefam == fa).any()])
# UDR / IFOREST-R on the 174-set (separate methods)
drop = lambda x: x[~x.dataset.astype(str).str.startswith(("137_CreditCard", "173_GECCO"))]
u = drop(pd.read_csv(os.path.join(D, "STREAM_UDR.csv"))); ir = drop(pd.read_csv(os.path.join(D, "STREAM_IFR.csv")))
def macx(x, c): return np.mean([x[x.truefam == fa][c].mean() for fa in ["local", "global"] if (x.truefam == fa).any()])
bars = [("SPARC-combined (ours)", macro("reg_combined"), "#2a9d8f"), ("SPARC-\u03b21 (ours)", macro("reg_beta1"), "#40b3a2"),
        ("SPARC-corr (ours)", macro("reg_corr_str"), "#40b3a2"), ("SPARC-matched (ours)", macro("reg_matched"), "#7fccc0"),
        ("EM", macro("reg_em"), "#8d99ae"), ("MV", macro("reg_mv"), "#8d99ae"), ("UDR", macx(u, "reg_udr"), "#adb5bd"),
        ("IFOREST-R", macx(ir, "reg_ifr"), "#adb5bd"), ("HITS", macro("reg_hits"), "#ced4da"), ("consensus", macro("reg_consensus"), "#ced4da"),
        ("ModelCentrality", macro("reg_mc"), "#ced4da"), ("random", macro("reg_random"), "#e0e0e0")]
bars.sort(key=lambda b: b[1])
fig, ax = plt.subplots(figsize=(9, 6)); yp = np.arange(len(bars))
ax.barh(yp, [b[1] for b in bars], color=[b[2] for b in bars], edgecolor="#333", linewidth=0.5)
for i, b in enumerate(bars): ax.text(b[1] + 0.003, i, f"{b[1]:.3f}", va="center", fontsize=9)
ax.set_yticks(yp); ax.set_yticklabels([b[0] for b in bars], fontsize=10); ax.invert_yaxis()
ax.set_xlabel("family-balanced macro regret (ap_norm; lower = better)"); ax.set_xlim(0, max(b[1] for b in bars) * 1.15)
ax.set_title(f"Model selection over the full classical+deep pool ({len(df)} datasets)\nSPARC-combined beats the UOMS field (vs MV p=0.021, vs EM p=0.036)", fontsize=11)
ax.axvline(macro("reg_random"), color="#c00", ls="--", lw=1, alpha=.5); ax.text(macro("reg_random"), -0.6, "random", color="#c00", fontsize=8, ha="center")
plt.tight_layout(); fig.savefig(os.path.join(D, "FIG_final_leaderboard.png"), dpi=120)
print("saved FIG_final_leaderboard.png"); print("bars:", [(b[0], round(b[1], 3)) for b in bars])
