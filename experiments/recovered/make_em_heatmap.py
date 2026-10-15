# -*- coding: utf-8 -*-
"""Heatmap: EM's systematic over/under-rating (em_err) by detector class x dataset class,
dataset class from an INDEPENDENT normals-only metric."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
df = pd.read_csv(os.path.join(S, "EM_SYSTEMATIC.csv"))
FAMS = ["local", "global", "other"]


def cell(sub):
    if len(sub) < 8: return np.nan, np.nan, len(sub)
    return sub.em_err.mean(), ttest_1samp(sub.em_err, 0).pvalue, len(sub)


def stars(p): return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 5e-2 else ""))

specs = [("m_sil", ["unimodal", "multimodal"], "multimodality (silhouette, normals-only)"),
         ("d", ["low-dim", "high-dim"], "dimensionality")]
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1, 2, 2]})
# marginal panel
axm = axes[0]; Mm = np.array([[cell(df[df.fam == f])[0]] for f in FAMS])
im = axm.imshow(Mm, cmap="RdBu_r", vmin=-0.18, vmax=0.18, aspect="auto")
for i, f in enumerate(FAMS):
    m, p, n = cell(df[df.fam == f]); axm.text(0, i, f"{m:+.3f}{stars(p)}\nn={n}", ha="center", va="center", fontsize=10)
axm.set_xticks([0]); axm.set_xticklabels(["all datasets"]); axm.set_yticks(range(3)); axm.set_yticklabels(FAMS)
axm.set_title("marginal", fontsize=11)
for ax, (metric, classes, title) in zip(axes[1:], specs):
    thr = df[metric].median(); df["_c"] = np.where(df[metric] >= thr, classes[1], classes[0])
    M = np.array([[cell(df[(df.fam == f) & (df._c == c)])[0] for c in classes] for f in FAMS])
    ax.imshow(M, cmap="RdBu_r", vmin=-0.18, vmax=0.18, aspect="auto")
    for i, f in enumerate(FAMS):
        for j, c in enumerate(classes):
            m, p, n = cell(df[(df.fam == f) & (df._c == c)])
            ax.text(j, i, f"{m:+.3f}{stars(p)}\nn={n}", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels([f"{c}\n(< {thr:.3g} | >=)" if k == 0 else c for k, c in enumerate(classes)])
    ax.set_yticks(range(3)); ax.set_yticklabels(FAMS); ax.set_title(f"dataset class by {title}", fontsize=11)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01); cb.set_label("EM over-rates (+) / under-rates (-)  [pct-rank error]")
fig.suptitle("EM systematically over-rates IForest/LODA and under-rates local detectors\n(em_err = EM percentile-rank minus true percentile-rank; *** p<1e-3, ** p<1e-2, * p<5e-2)", fontsize=12, y=1.06)
out = os.path.join(S, "FIG_em_systematic.png"); fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
