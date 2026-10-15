# -*- coding: utf-8 -*-
"""Illustrate MULTIMODALITY of the normal data (real feature space, no UMAP topology tricks).
Row per dataset: (A) 2D kernel-density contour on PC1-PC2 with GMM mode count; (B) 1D density
along the most-bimodal axis with the fitted Gaussian-mixture components. Anomalies overlaid red."""
import os, sys, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"


def load(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X); r = np.random.default_rng(0)
    ni = np.where(y == 0)[0]; ai = np.where(y == 1)[0]
    if len(ni) > 2000: ni = r.choice(ni, 2000, replace=False)
    if len(ai) > 300: ai = r.choice(ai, 300, replace=False)
    return X[ni], X[ai]


def bic_k(Z, kmax=5):
    b = {}
    for k in range(1, kmax + 1):
        try: b[k] = GaussianMixture(k, reg_covar=1e-3, random_state=0).fit(Z).bic(Z)
        except Exception: pass
    return min(b, key=b.get) if b else 1


PANELS = [("oddbench", "DiamondClarity", "multimodal normals  ->  LOCAL detector wins"),
          ("ovrbench", "PatientSurvival1", "unimodal normals  ->  GLOBAL detector wins")]
fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
for row, (corpus, name, sub) in enumerate(PANELS):
    Xn, Xa = load(corpus, name); sc = StandardScaler().fit(Xn)
    Zn = sc.transform(Xn); Za = sc.transform(np.nan_to_num(Xa))
    # ---- (A) 2D PCA kernel-density contour ----
    pca = PCA(2, random_state=0).fit(Zn); Pn = pca.transform(Zn); Pa = pca.transform(Za)
    k2 = bic_k(Pn); axA = axes[row, 0]
    try:
        kde = gaussian_kde(Pn.T)
        xs = np.linspace(Pn[:, 0].min(), Pn[:, 0].max(), 120); ys = np.linspace(Pn[:, 1].min(), Pn[:, 1].max(), 120)
        gx, gy = np.meshgrid(xs, ys); dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        axA.contourf(gx, gy, dens, levels=12, cmap="Blues")
        axA.contour(gx, gy, dens, levels=8, colors="#3b5b8c", linewidths=0.4, alpha=0.6)
    except Exception:
        axA.scatter(Pn[:, 0], Pn[:, 1], s=6, c="#9aa4b2", alpha=0.5)
    axA.scatter(Pa[:, 0], Pa[:, 1], s=16, c="#d1495b", alpha=0.8, edgecolors="white", linewidths=0.3, label="anomaly")
    axA.set_title(f"{name}\nnormal-density on PC1-PC2  (GMM modes k*={k2})", fontsize=11)
    axA.set_xlabel("PC1"); axA.set_ylabel("PC2"); axA.legend(loc="upper right", fontsize=8)
    # ---- (B) 1D density along most-bimodal axis ----
    corr = 3 * (len(Zn) - 1) ** 2 / max((len(Zn) - 2) * (len(Zn) - 3), 1)
    bcs = [((skew(Zn[:, j]) ** 2 + 1) / (kurtosis(Zn[:, j], fisher=True) + corr)) if np.std(Zn[:, j]) > 1e-9 else 0 for j in range(Zn.shape[1])]
    j = int(np.argmax(bcs)); vn = Zn[:, j]; va = Za[:, j]; axB = axes[row, 1]
    axB.hist(vn, bins=45, density=True, color="#c3d0e6", alpha=0.7, label="normal")
    xs = np.linspace(vn.min(), vn.max(), 400)
    try: axB.plot(xs, gaussian_kde(vn)(xs), color="#26456e", lw=2, label="KDE")
    except Exception: pass
    g1 = GaussianMixture(bic_k(vn.reshape(-1, 1), 4), reg_covar=1e-3, random_state=0).fit(vn.reshape(-1, 1))
    for mmu, wt in zip(g1.means_.ravel(), g1.weights_):
        comp = wt * np.exp(-0.5 * ((xs - mmu) ** 2) / g1.covariances_.ravel()[list(g1.means_.ravel()).index(mmu)]) / np.sqrt(2 * np.pi * g1.covariances_.ravel()[list(g1.means_.ravel()).index(mmu)])
        axB.plot(xs, comp, "--", color="#7a3b5b", lw=1.1)
    axB.plot([], [], "--", color="#7a3b5b", lw=1.1, label=f"GMM comps (k*={g1.n_components})")
    axB.scatter(va, np.full_like(va, -0.01 * axB.get_ylim()[1] if axB.get_ylim()[1] else -0.005), s=14, c="#d1495b", marker="|", alpha=0.7, label="anomaly")
    bcv = bcs[j]; axB.set_title(f"most-bimodal feature (BC={bcv:.2f}; >0.55 = multimodal)\n{sub}", fontsize=11)
    axB.set_xlabel(f"feature #{j} (standardized)"); axB.set_ylabel("density"); axB.legend(loc="upper right", fontsize=8)
fig.suptitle("Visualizing multimodality of the normal data", fontsize=14, y=1.0)
fig.tight_layout()
out = os.path.join(S, "FIG_modality_illustration.png"); fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
