# -*- coding: utf-8 -*-
"""Detector + ranking-method scatter panels vs dataset structure (162 datasets, saved results)."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
df = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv"))
df["family_adv"] = df.best_local - df.best_global
CL = {"local": "#2a7fb8", "global": "#d1495b"}
fig, ax = plt.subplots(2, 2, figsize=(14, 11))

# (A) detectors vs STATIC modality
a = ax[0, 0]
for f, g in df.groupby("truefam"):
    a.scatter(g.m_sil, g.family_adv, s=26, c=CL.get(f, "#888"), alpha=0.7, edgecolors="white", linewidths=0.3, label=f"{f}-wins")
z = np.polyfit(df.m_sil, df.family_adv, 1); xs = np.linspace(df.m_sil.min(), df.m_sil.max(), 50)
a.plot(xs, np.polyval(z, xs), color="#333", lw=1.6, ls="--")
a.axhline(0, color="#aaa", lw=0.8)
a.set_title(f"Detectors vs STATIC modality (silhouette)\nrho(family advantage) = {spearmanr(df.m_sil, df.family_adv).statistic:+.2f}  (weak)", fontsize=11)
a.set_xlabel("silhouette (multimodality ->)"); a.set_ylabel("best-local minus best-global ap_norm"); a.legend(fontsize=9)

# (B) detectors vs MEASURED locality
b = ax[0, 1]
for f, g in df.groupby("truefam"):
    b.scatter(g.local_ev, g.family_adv, s=26, c=CL.get(f, "#888"), alpha=0.7, edgecolors="white", linewidths=0.3, label=f"{f}-wins")
z = np.polyfit(df.local_ev, df.family_adv, 1); xs = np.linspace(df.local_ev.min(), df.local_ev.max(), 50)
b.plot(xs, np.polyval(z, xs), color="#333", lw=1.6, ls="--"); b.axhline(0, color="#aaa", lw=0.8)
b.set_title(f"Detectors vs MEASURED locality (local_ev)\nrho(family advantage) = {spearmanr(df.local_ev, df.family_adv).statistic:+.2f}  (3x stronger)", fontsize=11)
b.set_xlabel("local_ev  (local structure ->)"); b.set_ylabel("best-local minus best-global ap_norm"); b.legend(fontsize=9)

# (C) MULTIMODAL vs SINGLE-MODAL panel: ranking-method regret in each group (split by silhouette median)
c = ax[1, 0]
thr = df.m_sil.median(); df["grp"] = np.where(df.m_sil >= thr, "multimodal", "unimodal")
meths = ["reg_em", "reg_beta1", "reg_matched", "reg_random"]; labs = ["EM", "beta=1", "matched", "random"]
groups = ["multimodal", "unimodal"]; x = np.arange(len(groups)); w = 0.2
cols = ["#8c8c8c", "#2a7fb8", "#e0a12b", "#c0c0c0"]
for i, (m, lab) in enumerate(zip(meths, labs)):
    vals = [df[df.grp == gg][m].mean() for gg in groups]
    c.bar(x + (i - 1.5) * w, vals, w, label=lab, color=cols[i])
c.set_xticks(x); c.set_xticklabels([f"{gg}\n(n={int((df.grp==gg).sum())})" for gg in groups])
c.set_ylabel("mean regret (lower better)"); c.set_title("Ranking methods: multimodal vs single-modal\n(split by silhouette median)", fontsize=11); c.legend(fontsize=9)

# (D) ranking methods vs MEASURED locality (binned)
d = ax[1, 1]
df["bin"] = pd.qcut(df.local_ev, 5, labels=False, duplicates="drop")
ctr = df.groupby("bin").local_ev.mean()
for m, lab, col in zip(["reg_em", "reg_beta1", "reg_matched"], ["EM", "beta=1", "matched"], ["#8c8c8c", "#2a7fb8", "#e0a12b"]):
    mn = df.groupby("bin")[m].mean(); d.plot(ctr.values, mn.values, "o-", color=col, lw=2, label=lab)
d.set_xlabel("local_ev bin center  (unimodal <- -> multimodal/local)"); d.set_ylabel("mean regret")
d.set_title("Ranking-method regret vs measured locality", fontsize=11); d.legend(fontsize=9)
fig.suptitle("Detector-family dominance and selection-method performance vs dataset structure (162 datasets)", fontsize=13, y=0.997)
fig.tight_layout()
out = os.path.join(S, "FIG_scatter_panels.png"); fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
# quick numbers
print("multimodal group mean regret:", {l: round(df[df.grp=='multimodal'][m].mean(),3) for m,l in zip(meths,labs)})
print("unimodal   group mean regret:", {l: round(df[df.grp=='unimodal'][m].mean(),3) for m,l in zip(meths,labs)})
print("family_adv>0 (local wins) frac:", round((df.family_adv>0).mean(),2))
