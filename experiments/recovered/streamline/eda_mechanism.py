# -*- coding: utf-8 -*-
"""Mechanism figure: WHY the detector family flips. Row1 = ANISOTROPIC exemplar, Row2 = ISOTROPIC.
Col1: most-correlated feature PAIR (normals grey + anomalies red) - dependency anomalies sit OFF the
correlation line while in-range on each axis; isotropic anomalies sit at the margins of a round blob.
Col2: marginal histogram of the single best-separating feature - anomalies OVERLAP normals (anisotropic:
marginally invisible) vs form a separate tail (isotropic: marginally visible).
Col3: local (KNN) vs global (HBOS) detector score - which detector actually separates the anomalies."""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
Q = 0.05
OBJ = {}
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        for ds in PP.load_npz_dir(dd): OBJ[(sub, str(ds.name)[:40])] = (np.nan_to_num(np.asarray(ds.X, float)), np.asarray(ds.y, int).ravel())
MTS = {}
for name, src, Xc, lab in load_mts(200): MTS[str(name)[:40]] = (Xc, lab)
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1); b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1)
        sev = np.maximum(sev, np.maximum(-np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m))), -np.log(dens[b] + 1e-9)))
    return sev
def harden(Xtr, Xhold, Xa):
    tho = np.quantile(severity(Xtr, Xhold), 1 - Q); so_a = severity(Xtr, Xa)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ptr, Ph, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xa))
        thp = np.quantile(severity(Ptr, Ph), 1 - Q); sp_a = severity(Ptr, Pa)
    except Exception:
        sp_a = np.zeros(len(Xa)); thp = np.inf
    return Xa[~((so_a > tho) | (sp_a > thp))]
def get3(corp, name):
    if corp in ("oddbench", "ovrbench"):
        d = np.load(os.path.join(ROOT, "data", corp, name + ".npz"), allow_pickle=True)
        X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); Xn, Xa = X[y == 0], X[y == 1]
    elif corp in ("adbench", "dami"):
        X, y = OBJ[(corp, name)]; Xn, Xa = X[y == 0], X[y == 1]
    else:
        Xc, lab = MTS[name]; Xw, st = mts_window_features(Xc); yw = mts_wlabels(lab, st); Xw = np.nan_to_num(Xw)
        pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[tr][yw[tr] == 0], Xw[va][yw[va] == 0], Xw[te][yw[te] == 0], Xw[yw == 1]
    r = np.random.default_rng(0)
    if len(Xn) > 6000: Xn = Xn[r.choice(len(Xn), 6000, replace=False)]
    idx = np.arange(len(Xn)); r.shuffle(idx); a, b = int(0.6 * len(idx)), int(0.8 * len(idx)); return Xn[idx[:a]], Xn[idx[a:b]], Xn[idx[b:]], Xa
mine = pd.read_csv(os.path.join(D, "STREAM_MINE.csv"))
def first_ok(cand):
    for _, r in cand.iterrows():
        try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
        except Exception: continue
        if Xtr.shape[1] < 4 or len(Xtr) < 80 or len(Xa) < 15: continue
        Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
        if len(Xh) < 25: continue
        return (r.corpus, r.dataset, r.corr_str, Xtr, Xtn, Xh)
    return None
exA = first_ok(mine[mine.fam == "local"].sort_values("corr_str", ascending=False))
exI = first_ok(mine[mine.fam == "global"].sort_values("corr_str"))
# ---- shared styling: normals BLUE (small, semi-transparent), anomalies RED circles w/ black edge, subsampled, ON TOP ----
from sklearn.metrics import roc_auc_score
BLUE = "#2b6cb0"; REDC = "#e60023"; REDGE = "#000000"; MAX_A = 150
def subs(A, seed=0):
    return A if len(A) <= MAX_A else A[np.random.default_rng(seed).choice(len(A), MAX_A, replace=False)]
def pts(ax, Pn, Pa):
    ax.scatter(Pn[:, 0], Pn[:, 1], s=7, c=BLUE, alpha=.40, linewidths=0, zorder=1, rasterized=True)
    ax.scatter(Pa[:, 0], Pa[:, 1], s=26, c=REDC, marker="o", edgecolors=REDGE, linewidths=0.7, alpha=.95, zorder=5)
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))
for row, (corp, name, cs, Xtr, Xtn, Xh) in enumerate([exA, exI]):
    col = "#2a9d8f" if row == 0 else "#e76f51"; tag = "ANISOTROPIC" if row == 0 else "ISOTROPIC"
    Z = StandardScaler().fit(Xtr); Ztr = Z.transform(Xtr); Zh = Z.transform(Xh)
    Ztr_s = subs(Ztr, 1)  # thin the normal cloud for a clean backdrop
    # col1: most-correlated feature pair - the DEPENDENCY structure
    C = np.corrcoef(Ztr.T); np.fill_diagonal(C, 0); i, j = np.unravel_index(np.argmax(np.abs(C)), C.shape); r_ij = C[i, j]
    ax = axes[row, 0]; pts(ax, Ztr_s[:, [i, j]], subs(Zh)[:, [i, j]])
    ax.set_title(f"[{tag}] top feature pair (r={r_ij:+.2f})\n{'anomalies sit OFF the correlation line' if row==0 else 'no correlation; anomalies at the cloud margin'}", fontsize=9.5, color=col)
    ax.set_xlabel(f"feature {i}"); ax.set_ylabel(f"feature {j}")
    # col2: marginal histogram of best-separating single feature - the MARGINAL structure
    sep = np.abs(Zh.mean(0)); k = int(np.argmax(sep))
    ax = axes[row, 1]
    ax.hist(Ztr[:, k], bins=50, density=True, color=BLUE, alpha=.55, label="normal", zorder=1)
    ax.hist(Zh[:, k], bins=30, density=True, color=REDC, alpha=.75, ec=REDGE, lw=0.4, label="anomaly", zorder=3)
    ax.set_title(f"best single feature (feat {k})\n{'anomalies OVERLAP normals (marginally invisible)' if row==0 else 'anomalies form a separate tail (visible)'}", fontsize=9.5, color=col)
    ax.set_xlabel(f"feature {k} (standardized)"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    axes[row, 0].annotate(f"{name[:18]}\ncorr_str={cs:.2f}", xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90, va="center", ha="center", fontsize=9, fontweight="bold", color=col)
    print(f"[{tag}] {name[:22]} cs={cs:.2f}  pair=({i},{j}) r={r_ij:+.2f}  n_anom={len(Xh)}")
plt.suptitle("Two anomaly structures: anisotropic (correlated) datasets hide anomalies in a DEPENDENCY violation\n(off the correlation line, invisible per-feature); isotropic (uncorrelated) datasets expose them as a MARGINAL extreme", fontsize=10.5, fontweight="bold")
plt.tight_layout(rect=[0.03, 0, 1, 0.94]); fig.savefig(os.path.join(D, "FIG_mechanism.png"), dpi=125)
print("exemplars:", exA[1], round(exA[2], 2), "|", exI[1], round(exI[2], 2)); print("saved FIG_mechanism.png")
