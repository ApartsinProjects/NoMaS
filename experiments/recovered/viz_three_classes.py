"""Visualize the 3 classes (normal, SPARC synthetic, real anomaly) for a task: PCA-2D, UMAP-2D, and
per-feature histograms (normal vs synthetic). Shows synthetic marginals match normals (each value real)
while the joint/position differs, and where synthetic sits vs real in the embedding."""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
BUN = os.path.join(S, "streamline", "bundle_v2")
def _gen_beta(Xn, ns, beta=1.0, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d)); Hs = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt/max(cnt.sum(),1)
        lo, hi = np.percentile(col,2), np.percentile(col,98); ctr = (edges[:-1]+edges[1:])/2
        allowed = (cnt>0)&(ctr>=lo)&(ctr<=hi); w = np.where(allowed, np.maximum(dens,1e-6)**beta, 0.0)
        wn = w/w.sum() if w.sum()>0 else None; binidx = np.clip(np.digitize(col, edges[1:-1]),0,len(cnt)-1)
        Hs.append((wn,col,binidx))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1,int(frac*d)), replace=False):
            wn,col,binidx = Hs[j]
            base[j] = Xn[rng.integers(n),j] if wn is None else (lambda c: c[rng.integers(len(c))] if len(c) else Xn[rng.integers(n),j])(col[binidx==rng.choice(len(wn),p=wn)])
        out[r] = base
    return out
C = {"normal":"#9bb7d4", "synthetic":"#e8873a", "real":"#c0392b"}
def viz(key, nfeat=6):
    z = np.load(os.path.join(BUN, key+".npz"))
    Xtr, Xval, Xh = z["Xtr"].astype("float64"), z["Xval"].astype("float64"), z["Xh"].astype("float64")
    rng = np.random.default_rng(0)
    Xn = Xval[rng.choice(len(Xval), min(800,len(Xval)), replace=False)]
    Xr = Xh[rng.choice(len(Xh), min(400,len(Xh)), replace=False)]
    Xs = _gen_beta(Xval, 200)
    ss = StandardScaler().fit(Xtr)
    Zn, Zs, Zr = ss.transform(Xn), ss.transform(Xs), ss.transform(Xr)
    d = Xtr.shape[1]
    fig = plt.figure(figsize=(13, 8)); gs = fig.add_gridspec(2, 3, height_ratios=[1.25,1])
    # PCA
    allz = np.vstack([Zn, Zs, Zr]); pc = PCA(2, random_state=0).fit(ss.transform(Xtr))
    P = pc.transform(allz); a,b = len(Zn), len(Zn)+len(Zs)
    axp = fig.add_subplot(gs[0,0])
    axp.scatter(*P[:a].T, s=6, c=C["normal"], alpha=.5, label="normal", edgecolors="none")
    axp.scatter(*P[a:b].T, s=14, c=C["synthetic"], alpha=.8, label="SPARC synthetic", edgecolors="none", marker="^")
    axp.scatter(*P[b:].T, s=10, c=C["real"], alpha=.7, label="real anomaly", edgecolors="none")
    axp.set_title(f"PCA  ({d} features)"); axp.legend(fontsize=8, loc="best"); axp.set_xticks([]); axp.set_yticks([])
    # UMAP
    try:
        U = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(allz)
        axu = fig.add_subplot(gs[0,1])
        axu.scatter(*U[:a].T, s=6, c=C["normal"], alpha=.5, edgecolors="none")
        axu.scatter(*U[a:b].T, s=14, c=C["synthetic"], alpha=.8, edgecolors="none", marker="^")
        axu.scatter(*U[b:].T, s=10, c=C["real"], alpha=.7, edgecolors="none")
        axu.set_title("UMAP"); axu.set_xticks([]); axu.set_yticks([])
    except Exception as e:
        fig.add_subplot(gs[0,1]).text(.5,.5,f"UMAP failed\n{e}",ha="center")
    # normality-distance histogram (the difficulty view)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(5,len(Xtr)-1)).fit(ss.transform(Xtr))
    sc = lambda X: nn.kneighbors(ss.transform(X))[0].mean(1)
    axh = fig.add_subplot(gs[0,2]); bins = 40
    for lab,X in [("normal",Xn),("synthetic",Xs),("real",Xr)]:
        axh.hist(sc(X), bins=bins, density=True, histtype="step", lw=1.8, color=C[lab], label=lab)
    axh.set_title("distance-to-normals (difficulty)"); axh.set_xlabel("kNN distance"); axh.legend(fontsize=8)
    # per-feature marginal histograms: normal vs synthetic
    vsel = list(range(min(nfeat, d)))
    for i, j in enumerate(vsel):
        ax = fig.add_subplot(gs[1, :].subgridspec(1, len(vsel))[0, i]) if False else fig.add_subplot(2, len(vsel), len(vsel)+i+1)
        ax.hist(Xval[:, j], bins=25, density=True, color=C["normal"], alpha=.6, label="normal")
        ax.hist(Xs[:, j], bins=25, density=True, histtype="step", color=C["synthetic"], lw=1.6, label="synthetic")
        ax.set_title(f"feat {j}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        if i==0: ax.legend(fontsize=7)
    fig.suptitle(f"{key}: normal vs SPARC synthetic vs real anomaly", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    out = os.path.join(S, f"VIZ_3CLASS_{key.replace('__','_')}.png")
    fig.savefig(out, dpi=130); print("saved", out)
for key in sys.argv[1:]:
    viz(key)
