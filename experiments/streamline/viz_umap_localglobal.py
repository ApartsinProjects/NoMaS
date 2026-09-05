# -*- coding: utf-8 -*-
"""UMAP grid: top-5 most-LOCAL and top-5 most-GLOBAL datasets (by max_local_ap - max_global_ap gap from
the co-computed pool), normals (grey) + hardened anomalies (red). 2 rows (local top, global bottom)."""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
try:
    import umap; HAVE_UMAP = True
except Exception: HAVE_UMAP = False
print("UMAP available:", HAVE_UMAP)
Q = 0.05
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
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
# rank by local-global gap from the co-computed pool ap (STREAM_DEEP_MODAL) merged to corpus/dataset
dm = pd.read_csv(os.path.join(D, "STREAM_DEEP_MODAL.csv")); idx = pd.read_csv(os.path.join(D, "bundle", "_index.csv"))[["key", "corpus", "dataset"]]
m = dm.merge(idx, on="key", how="inner")
loc_cols = [c for c in dm.columns if str(c).startswith(LOCAL)]; glo_cols = [c for c in dm.columns if str(c).startswith(GLOBAL)]
m["maxL"] = m[loc_cols].max(1); m["maxG"] = m[glo_cols].max(1); m["gap"] = m.maxL - m.maxG
m = m[(m.maxL > 0.15) | (m.maxG > 0.15)]                                   # skip all-bad (nothing to see)
top_local = m.sort_values("gap", ascending=False).head(5)
top_global = m.sort_values("gap").head(5)
def embed(Xn, Xa):
    Z = StandardScaler().fit_transform(np.vstack([Xn, Xa]))
    if HAVE_UMAP and Z.shape[0] > 15:
        try: return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Z)
        except Exception: pass
    return PCA(2, random_state=0).fit_transform(Z)
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for row, (label, sel) in enumerate([("LOCAL (neighbor detectors win)", top_local), ("GLOBAL (histogram detectors win)", top_global)]):
    for col, (_, r) in enumerate(sel.iterrows()):
        ax = axes[row, col]
        try:
            Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset); Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
            rng = np.random.default_rng(0)
            Xn = Xtn if len(Xtn) <= 1500 else Xtn[rng.choice(len(Xtn), 1500, replace=False)]
            Xhp = Xh if len(Xh) <= 400 else Xh[rng.choice(len(Xh), 400, replace=False)]
            E = embed(Xn, Xhp); nN = len(Xn)
            ax.scatter(E[:nN, 0], E[:nN, 1], s=5, c="#c8c8c8", alpha=.5, linewidths=0)
            ax.scatter(E[nN:, 0], E[nN:, 1], s=16, c="#d1495b", alpha=.85, linewidths=0)
            ax.set_title(f"{str(r.dataset)[:20]}\nL={r.maxL:.2f} G={r.maxG:.2f}", fontsize=9)
        except Exception as e:
            ax.set_title(f"{str(r.dataset)[:18]}\n(err)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    axes[row, 0].set_ylabel(label, fontsize=12, fontweight="bold")
plt.suptitle("UMAP: normals (grey) + hard anomalies (red) - top-5 LOCAL (row1) vs top-5 GLOBAL (row2) datasets", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(D, "STREAM_UMAP_LOCALGLOBAL.png"), dpi=115)
print("local:", list(top_local.dataset)); print("global:", list(top_global.dataset))
print("saved streamline/STREAM_UMAP_LOCALGLOBAL.png")
