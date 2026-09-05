# -*- coding: utf-8 -*-
"""Compare 2D views for showing hard anomalies: PCA / t-SNE / UMAP / mechanism (off-manifold vs marginal).
High-contrast styling: normals = dense slate small dots; anomalies = red diamonds w/ dark edge, drawn ON TOP."""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from hadb_ts_final import block_split3
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
    if corp in ("adbench", "dami"):
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
        if Xtr.shape[1] < 5 or len(Xtr) < 120 or len(Xa) < 20: continue
        Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
        if len(Xh) < 25: continue
        return (r.corpus, r.dataset, r.corr_str, Xtr, Xtn, Xh)
    return None
exA = first_ok(mine[mine.fam == "local"].sort_values("corr_str", ascending=False))
exI = first_ok(mine[mine.fam == "global"].sort_values("corr_str"))

# ---- styling: denser darker grey normals, red anomalies subsampled + dark edge, drawn ON TOP ----
GREY = "#2b6cb0"; RED = "#e60023"; REDGE = "#000000"; MAX_A = 120  # normals = blue
def auc1(sn, sa):  # univariate AUC of a score separating anomalies(sa) from normals(sn)
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sa))], np.r_[sn, sa])
def sub_anom(Ph, seed=0):
    if len(Ph) <= MAX_A: return Ph
    return Ph[np.random.default_rng(seed).choice(len(Ph), MAX_A, replace=False)]
def draw(ax, Pn, Ph, logxy=False):
    ax.scatter(Pn[:, 0], Pn[:, 1], s=7, c=GREY, alpha=.45, linewidths=0, zorder=1, rasterized=True)
    ax.scatter(Ph[:, 0], Ph[:, 1], s=26, c=RED, marker="o", edgecolors=REDGE, linewidths=0.7, alpha=.95, zorder=5)
    if logxy: ax.set_xscale("log"); ax.set_yscale("log")

def embeds(Xtr, Xtn, Xh, seed=0):
    """PCA + mechanism views. Normals = held-out test normals (Xtn) so counts are comparable to anoms."""
    sc = StandardScaler().fit(Xtr); Ztr = sc.transform(Xtr); Zn = sc.transform(Xtn); Zh = sc.transform(Xh)
    out = {}
    P2 = PCA(2, random_state=seed).fit(Ztr); out["PCA"] = (P2.transform(Zn), P2.transform(Zh))
    k = max(1, int(np.searchsorted(np.cumsum(PCA().fit(Ztr).explained_variance_ratio_), 0.90) + 1))
    pk = PCA(k, random_state=seed).fit(Ztr)
    def resid(Z): R = Z - pk.inverse_transform(pk.transform(Z)); return np.sqrt((R ** 2).sum(1)) + 1e-3
    def marg(Z): return np.abs(Z).max(1) + 1e-3
    rn, rh = resid(Zn), resid(Zh); mn, mh = marg(Zn), marg(Zh)
    out["mechanism"] = (np.c_[rn, mn], np.c_[rh, mh])
    out["_auc"] = (auc1(rn, rh), auc1(mn, mh), k)  # (residual-AUC, marginal-AUC, k)
    return out

views = ["PCA", "mechanism", "mechanism"]  # 3rd = log-scaled mechanism
fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
for row, (corp, name, cs, Xtr, Xtn, Xh) in enumerate([exA, exI]):
    tag = "ANISOTROPIC" if row == 0 else "ISOTROPIC"; col = "#2a9d8f" if row == 0 else "#e76f51"
    E = embeds(Xtr, Xtn, Xh); ar, am, k = E["_auc"]
    print(f"[{tag}] {name[:22]} cs={cs:.2f} k={k}  residual-AUC={ar:.2f}  marginal-AUC={am:.2f}  (n_anom={len(Xh)})")
    for c, v in enumerate(views):
        ax = axes[row, c]; Pn, Ph = E[v]; Phs = sub_anom(Ph)
        draw(ax, Pn, Phs, logxy=(c == 2))
        if v == "mechanism":
            ax.set_xlabel("off-manifold residual" + (" (log)" if c == 2 else ""), fontsize=8); ax.set_ylabel("marginal max|z|" + (" (log)" if c == 2 else ""), fontsize=8)
            ax.set_title((f"mechanism  (resid-AUC {ar:.2f} / marg-AUC {am:.2f})" if c == 1 else "mechanism (log-log)"), fontsize=9)
        else:
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title("PCA top-2", fontsize=9)
    axes[row, 0].set_ylabel(f"{tag}\n{name[:18]} (cs={cs:.2f})", fontsize=9, color=col, fontweight="bold")
plt.suptitle("Hard anomalies (red) vs normals (grey): PCA hides them; the residual/marginal mechanism view separates by dataset type", fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(D, "FIG_embed_explore.png"), dpi=115)
print("saved FIG_embed_explore.png")
