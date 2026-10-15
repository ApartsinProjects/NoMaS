# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
rows = [("oracle (floor)", 0.000, 0.000, "ref"),
        ("NoMaS matched (ours)", 0.138, 0.153, "ours"),
        ("best-fixed LOF_k20 (cheats)", 0.118, 0.165, "cheat"),
        ("NoMaS beta=1 (ours)", 0.143, 0.163, "ours"),
        ("beta=-4 extreme synth", 0.185, 0.190, "base"),
        ("EM (Excess-Mass)", 0.196, 0.193, "em"),
        ("MV (292-set)", 0.220, np.nan, "base"),
        ("consensus/MC/HITS (292-set)", 0.278, np.nan, "base"),
        ("random", 0.271, 0.267, "base")]
df = pd.DataFrame(rows, columns=["method", "micro", "macro", "kind"]).sort_values("micro")
col = {"ours": "#2a7fb8", "em": "#d1495b", "cheat": "#b0b0b0", "base": "#c9c9c9", "ref": "#5aa469"}
fig, ax = plt.subplots(figsize=(11, 5.6)); y = np.arange(len(df))
ax.barh(y, df.micro, color=[col[k] for k in df.kind], edgecolor="white")
for i, (m, mi, ma) in enumerate(zip(df.method, df.micro, df.macro)):
    ax.text(mi + 0.004, i, f"{mi:.3f}" + (f"  (macro {ma:.3f})" if ma == ma else ""), va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(df.method, fontsize=10); ax.invert_yaxis()
ax.set_xlabel("regret on ap_norm (lower = better)"); ax.set_xlim(0, 0.33)
ax.set_title("Label-free anomaly-detector selection leaderboard (HADB, 162 datasets)\nblue = ours; red = EM (prior SOTA); grey = baselines; green = oracle floor", fontsize=12)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=col["ours"], label="NoMaS (ours)"), Patch(color=col["em"], label="EM (prior SOTA)"),
                   Patch(color=col["cheat"], label="best-fixed (label-cheating)"), Patch(color=col["base"], label="baselines")], fontsize=9, loc="lower right")
fig.tight_layout(); out = os.path.join(S, "FIG_leaderboard.png"); fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
