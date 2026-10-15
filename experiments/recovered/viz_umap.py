# -*- coding: utf-8 -*-
"""2D embedding of two contrasting datasets: multimodal/local-win vs unimodal/global-win.
Normals (grey) + real anomalies (red). UMAP -> t-SNE -> PCA fallback."""
import os, sys, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"

def embed(X):
    Z = StandardScaler().fit_transform(np.nan_to_num(X))
    try:
        import umap; return umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=0).fit_transform(Z), "UMAP"
    except Exception: pass
    try:
        from sklearn.manifold import TSNE; return TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z), "t-SNE"
    except Exception:
        from sklearn.decomposition import PCA; return PCA(2, random_state=0).fit_transform(Z), "PCA"

def load(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X); r = np.random.default_rng(0)
    ni = np.where(y == 0)[0]; ai = np.where(y == 1)[0]
    if len(ni) > 1200: ni = r.choice(ni, 1200, replace=False)
    if len(ai) > 300: ai = r.choice(ai, 300, replace=False)
    return X[ni], X[ai]

PANELS = [("oddbench", "DiamondClarity", "multimodal  ->  LOCAL wins (LOF)"),
          ("ovrbench", "PatientSurvival1", "unimodal  ->  GLOBAL wins (HBOS)")]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, (corpus, name, sub) in zip(axes, PANELS):
    Xn, Xa = load(corpus, name)
    emb, algo = embed(np.vstack([Xn, Xa])); en, ea = emb[:len(Xn)], emb[len(Xn):]
    ax.scatter(en[:, 0], en[:, 1], s=8, c="#9aa4b2", alpha=0.55, linewidths=0, label=f"normal (n={len(Xn)})")
    ax.scatter(ea[:, 0], ea[:, 1], s=22, c="#d1495b", alpha=0.85, edgecolors="white", linewidths=0.3, label=f"anomaly (n={len(Xa)})")
    ax.set_title(f"{name}\n{sub}", fontsize=12); ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    for s in ax.spines.values(): s.set_edgecolor("#d0d0d0")
fig.suptitle(f"Real anomalies vs normal structure ({algo} 2D embedding)", fontsize=13, y=1.0)
fig.tight_layout()
out = os.path.join(S, "FIG_umap_two_examples.png"); fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out, "algo", algo)
