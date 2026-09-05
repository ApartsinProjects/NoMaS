# -*- coding: utf-8 -*-
"""EDA money-figure: ANISOTROPIC (correlated, steep spectrum, local-detector) vs ISOTROPIC (uncorrelated,
flat spectrum, global-detector) datasets. Per exemplar: (col1) PCA eigenvalue spectrum bar (steep vs flat),
(col2) PC1-PC2 scatter of normals (grey) + hard anomalies (red) - anomalies fill voids/off-manifold for
anisotropic, sit as marginal extremes for isotropic. 2 anisotropic rows + 2 isotropic rows."""
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
aniso = mine[mine.fam == "local"].sort_values("corr_str", ascending=False)
iso = mine[mine.fam == "global"].sort_values("corr_str")
def pick(cand, k=2):
    out = []
    for _, r in cand.iterrows():
        try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
        except Exception: continue
        if Xtr.shape[1] < 3 or len(Xtr) < 60 or len(Xa) < 15: continue
        Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
        if len(Xh) < 20: continue
        out.append((r.corpus, r.dataset, r.corr_str, r.eff_dim, Xtr, Xh))
        if len(out) >= k: break
    return out
rows = pick(aniso, 2) + pick(iso, 2)
labels = ["ANISOTROPIC (correlated) \u2192 LOCAL detector", "ANISOTROPIC (correlated) \u2192 LOCAL detector",
          "ISOTROPIC (uncorrelated) \u2192 GLOBAL detector", "ISOTROPIC (uncorrelated) \u2192 GLOBAL detector"]
fig, axes = plt.subplots(len(rows), 2, figsize=(11, 3.1 * len(rows)))
for i, (corp, name, cs, ed, Xtr, Xh) in enumerate(rows):
    sc = StandardScaler().fit(Xtr); Ztr = sc.transform(Xtr); pca = PCA().fit(Ztr); ev = pca.explained_variance_ratio_
    col = "#2a9d8f" if i < 2 else "#e76f51"; tag = "ANISOTROPIC" if i < 2 else "ISOTROPIC"
    axs = axes[i, 0]; kk = min(15, len(ev)); axs.bar(range(1, kk + 1), ev[:kk], color=col)
    axs.set_title(f"[{tag}]  {name[:22]} \u2014 {'STEEP' if i<2 else 'FLAT'} spectrum", fontsize=9, color=col, fontweight="bold"); axs.set_xlabel("principal component"); axs.set_ylabel("variance frac"); axs.set_ylim(0, max(ev[0] * 1.1, 0.1))
    axs.text(0.97, 0.9, f"corr_str={cs:.2f}\neff_dim={ed:.1f}", transform=axs.transAxes, ha="right", va="top", fontsize=8.5, bbox=dict(fc="white", ec=col, alpha=.85))
    axc = axes[i, 1]; P2 = PCA(2, random_state=0).fit(Ztr); Pn = P2.transform(Ztr); Ph = P2.transform(sc.transform(Xh))
    rng = np.random.default_rng(0); Pn = Pn if len(Pn) <= 1200 else Pn[rng.choice(len(Pn), 1200, replace=False)]
    axc.scatter(Pn[:, 0], Pn[:, 1], s=5, c="#c8c8c8", alpha=.5, linewidths=0); axc.scatter(Ph[:, 0], Ph[:, 1], s=14, c="#c0392b", alpha=.85, linewidths=0)
    axc.set_title(f"{'thin manifold, anomalies off it \u2192 LOCAL' if i<2 else 'round cloud, anomalies at margins \u2192 GLOBAL'}", fontsize=9, color=col); axc.set_xlabel("PC1"); axc.set_ylabel("PC2"); axc.set_xticks([]); axc.set_yticks([])
plt.suptitle("The anisotropic \u2194 isotropic axis: the normal-data spectrum shape determines which detector wins", fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.965]); fig.savefig(os.path.join(D, "FIG_spectrum_eda.png"), dpi=120)
print("exemplars:", [(r[1][:20], round(r[2], 2)) for r in rows]); print("saved FIG_spectrum_eda.png")
