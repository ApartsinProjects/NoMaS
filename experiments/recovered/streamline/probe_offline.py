# -*- coding: utf-8 -*-
"""For the top anisotropic (local-family, high corr_str) datasets, find the feature pair whose NORMAL
correlation the anomalies most VIOLATE (max perpendicular residual off the normals' major axis), among
pairs with 0.4<=|r|<=0.985 (exclude degenerate near-perfect pairs). Rank exemplars by that off-line ratio."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
def offline_ratio(Ztr, Zh):
    """Best pair by anomaly off-line residual ratio. Returns (ratio, i, j, r_ij)."""
    C = np.corrcoef(Ztr.T); d = C.shape[0]; best = (0, -1, -1, 0)
    for i in range(d):
        for j in range(i + 1, d):
            r = C[i, j]
            if not (0.4 <= abs(r) <= 0.985): continue
            A = Ztr[:, [i, j]]; H = Zh[:, [i, j]]
            p = PCA(2).fit(A); axis = p.components_[0]; perp = p.components_[1]  # minor axis = off-line direction
            mu = A.mean(0); rn = np.abs((A - mu) @ perp); rh = np.abs((H - mu) @ perp)
            ratio = np.median(rh) / (np.quantile(rn, 0.95) + 1e-9)
            if ratio > best[0]: best = (ratio, i, j, r)
    return best
mine = pd.read_csv(os.path.join(D, "STREAM_MINE.csv"))
cand = mine[mine.fam == "local"].sort_values("corr_str", ascending=False).head(25)
rows = []
for _, r in cand.iterrows():
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if Xtr.shape[1] < 4 or len(Xtr) < 80 or len(Xa) < 15: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 25: continue
    Z = StandardScaler().fit(Xtr); Ztr = Z.transform(Xtr); Zh = Z.transform(Xh)
    ratio, i, j, rij = offline_ratio(Ztr, Zh)
    rows.append((r.dataset[:24], round(r.corr_str, 2), round(ratio, 2), i, j, round(rij, 2), len(Xh)))
rows.sort(key=lambda x: -x[2])
print(f"{'dataset':26}{'cs':>6}{'offline':>9}{'pair':>10}{'r':>7}{'nA':>6}")
for d, cs, ratio, i, j, rij, nA in rows:
    print(f"{d:26}{cs:>6}{ratio:>9}{f'({i},{j})':>10}{rij:>7}{nA:>6}")
