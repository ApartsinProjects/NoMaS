# -*- coding: utf-8 -*-
"""Supported-claim check: best SINGLE-FEATURE AUC vs best MULTIVARIATE-detector AUC, per dataset, by family.
Hypothesis: isotropic anomalies are marginally separable (single-feat AUC already high, joint adds little);
anisotropic anomalies are NOT marginally separable (single-feat AUC low, joint detector adds a lot)."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.pca import PCA as PPCA
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
def best_single(Xtr, Xtn, Xh):
    y = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; Z = StandardScaler().fit(Xtr)
    Vn, Vh = Z.transform(Xtn), Z.transform(Xh); best = 0.5
    for c in range(Vn.shape[1]):
        s = np.r_[Vn[:, c], Vh[:, c]]; a = roc_auc_score(y, s); best = max(best, a, 1 - a)
    return best
def best_multi(Xtr, Xtn, Xh):
    y = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; Xte = np.vstack([Xtn, Xh]); best = 0.5
    for M in (KNN(n_neighbors=10), HBOS(n_bins=20), IForest(random_state=0), PPCA(random_state=0)):
        try:
            M.fit(Xtr); s = M.decision_function(Xte); best = max(best, roc_auc_score(y, s))
        except Exception: pass
    return best
mine = pd.read_csv(os.path.join(D, "STREAM_MINE.csv"))
out = os.path.join(D, "STREAM_MARGJOINT.csv"); done = set()
if os.path.exists(out):
    done = set(pd.read_csv(out).key.astype(str))
else:
    pd.DataFrame(columns=["key", "dataset", "fam", "corr_str", "single_auc", "multi_auc", "gap", "nA"]).to_csv(out, index=False)
for _, r in mine.iterrows():
    key = f"{r.corpus}/{r.dataset}"
    if key in done: continue
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if Xtr.shape[1] < 3 or len(Xtr) < 60 or len(Xa) < 12 or len(Xtn) < 12: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 15: continue
    sa = best_single(Xtr, Xtn, Xh); ma = best_multi(Xtr, Xtn, Xh)
    pd.DataFrame([[key, r.dataset[:30], r.fam, round(float(r.corr_str), 3), round(sa, 3), round(ma, 3), round(ma - sa, 3), len(Xh)]],
                 columns=["key", "dataset", "fam", "corr_str", "single_auc", "multi_auc", "gap", "nA"]).to_csv(out, mode="a", header=False, index=False)
df = pd.read_csv(out)
for fam in ["local", "global"]:
    g = df[df.fam == fam]
    print(f"{fam:7} n={len(g):3}  single_auc={g.single_auc.mean():.3f}  multi_auc={g.multi_auc.mean():.3f}  joint_gain(multi-single)={g.gap.mean():+.3f}")
print("saved", out, "rows", len(df))
