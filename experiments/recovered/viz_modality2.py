# -*- coding: utf-8 -*-
"""Robust multimodality illustration. For each dataset: project normals onto the axis that
MAXIMALLY separates the two dominant Gaussian-mixture modes (LDA-like), trim outliers, and show
the 1D density with the fitted component curves. Genuine two-peakedness -> two humps; unimodal ->
one hump. A 2D density on the top mode-separating plane accompanies it. Datasets via argv pairs."""
import os, sys, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
# default pair; override with: python viz_modality2.py corpusA:nameA:labelA corpusB:nameB:labelB
PAIR = [("oddbench", "DiamondClarity", "multimodal -> LOCAL wins"),
        ("ovrbench", "PatientSurvival1", "unimodal -> GLOBAL wins")]
if len(sys.argv) >= 3:
    PAIR = [tuple(a.split(":", 2)) for a in sys.argv[1:3]]


def load(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X); r = np.random.default_rng(0); ni = np.where(y == 0)[0]
    if len(ni) > 2000: ni = r.choice(ni, 2000, replace=False)
    return X[ni]


def sil_k(Z):
    best = (0.0, 1)
    for k in range(2, 5):
        try:
            lab = KMeans(k, n_init=5, random_state=0).fit_predict(Z)
            if len(set(lab)) > 1:
                s = silhouette_score(Z, lab); best = max(best, (s, k))
        except Exception: pass
    return best


def mode_axis(Z):
    """unit direction between the two most separated GMM(k=2) means (Mahalanobis-ish via shared std)."""
    g = GaussianMixture(2, reg_covar=1e-3, random_state=0).fit(Z)
    v = g.means_[1] - g.means_[0]; n = np.linalg.norm(v)
    return (v / n) if n > 1e-9 else np.eye(Z.shape[1])[0]


fig, axes = plt.subplots(len(PAIR), 2, figsize=(12.5, 5.2 * len(PAIR)))
if len(PAIR) == 1: axes = axes[None, :]
for row, (corpus, name, sub) in enumerate(PAIR):
    Xn = load(corpus, name); Z = StandardScaler().fit_transform(Xn)
    sil, ksil = sil_k(Z)
    a1 = mode_axis(Z); resid = Z - np.outer(Z @ a1, a1)
    # second axis = top PC of residual
    u, s2, vt = np.linalg.svd(resid - resid.mean(0), full_matrices=False); a2 = vt[0]
    p1 = Z @ a1; p2 = Z @ a2
    # ---- 2D density on mode-separating plane (trim 1-99 pct) ----
    axA = axes[row, 0]
    lo1, hi1 = np.percentile(p1, [1, 99]); lo2, hi2 = np.percentile(p2, [1, 99])
    mask = (p1 >= lo1) & (p1 <= hi1) & (p2 >= lo2) & (p2 <= hi2); q1, q2 = p1[mask], p2[mask]
    try:
        kde = gaussian_kde(np.vstack([q1, q2])); gx, gy = np.meshgrid(np.linspace(lo1, hi1, 120), np.linspace(lo2, hi2, 120))
        dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        axA.contourf(gx, gy, dens, levels=12, cmap="Blues"); axA.contour(gx, gy, dens, levels=8, colors="#3b5b8c", linewidths=0.4, alpha=0.6)
    except Exception:
        axA.scatter(q1, q2, s=6, c="#6b7280", alpha=0.4)
    axA.set_title(f"{name}\ndensity on mode-separating plane  (silhouette={sil:.2f}, k*={ksil})", fontsize=11)
    axA.set_xlabel("mode axis"); axA.set_ylabel("2nd axis")
    # ---- 1D density along mode axis + GMM components ----
    axB = axes[row, 1]; v = p1[(p1 >= lo1) & (p1 <= hi1)]
    axB.hist(v, bins=45, density=True, color="#c3d0e6", alpha=0.7, label="normal")
    xs = np.linspace(v.min(), v.max(), 400)
    try: axB.plot(xs, gaussian_kde(v)(xs), color="#26456e", lw=2, label="KDE")
    except Exception: pass
    g = GaussianMixture(max(2, ksil), reg_covar=1e-3, random_state=0).fit(v.reshape(-1, 1))
    var = g.covariances_.ravel()
    for i, (mmu, wt) in enumerate(zip(g.means_.ravel(), g.weights_)):
        axB.plot(xs, wt * np.exp(-0.5 * (xs - mmu) ** 2 / var[i]) / np.sqrt(2 * np.pi * var[i]), "--", color="#7a3b5b", lw=1.2)
    sep = abs(g.means_.ravel()[0] - g.means_.ravel()[1]) / np.sqrt(var.mean()) if g.n_components >= 2 else 0
    axB.plot([], [], "--", color="#7a3b5b", lw=1.2, label=f"GMM comps (k*={g.n_components})")
    axB.set_title(f"density along mode axis  (mode gap = {sep:.2f} sd)\n{sub}", fontsize=11)
    axB.set_xlabel("projection on mode axis"); axB.set_ylabel("density"); axB.legend(loc="upper right", fontsize=8)
fig.suptitle("Visualizing multimodality of the normal data (robust: mode-separating projection, outliers trimmed)", fontsize=13, y=1.0)
fig.tight_layout()
out = os.path.join(S, "FIG_modality_illustration.png"); fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
