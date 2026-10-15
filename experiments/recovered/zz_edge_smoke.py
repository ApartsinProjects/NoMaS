# -*- coding: utf-8 -*-
"""SMOKE TEST the edge/tail pseudo-anomaly selector before any full run.

Idea (user): cluster the validation normals (k may be 1), take the ~1% farthest-from-center
points as pseudo-anomalies, fit each detector on the rest, rank detectors by how well they
separate the edge points. Pick argmax.

Expectation stated in advance: for this to be worth a full run, the edge-pick's mean regret
must be BELOW random, and ideally approach EM (0.204 on the partial). If it ties random or
favours only distance detectors, drop it like the earlier refuted pseudo-anomaly variants.

Reconstructs validation features for included OddBench datasets (seed 0) exactly as the arm
does, computes the edge pick, and looks up the pick's TEST ap_norm from the frozen arm CSV.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(S, "scratchpad"))
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.pca import PCA as PPCA
from pyod.models.cblof import CBLOF
from pyod.models.loda import LODA

def build_pool():
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)),
         ("LOF", lambda: LOF(n_neighbors=20)), ("KNN", lambda: KNN(n_neighbors=5)),
         ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()), ("HBOS", lambda: HBOS(n_bins=10)),
         ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)), ("LODA", lambda: LODA(n_bins=10))]
    for k in [3,10,35,50,100,200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k)))
        p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5,20,50,100]: p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3,0.7,0.9,0.99]: p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    for ncl in [4,16,32]: p.append((f"CBLOF_c{ncl}", lambda ncl=ncl: CBLOF(n_clusters=ncl, random_state=0)))
    for ne in [50,300]: p.append((f"IF_n{ne}", lambda ne=ne: IForest(n_estimators=ne, random_state=0)))
    for nb in [20,50]: p.append((f"LODA_b{nb}", lambda nb=nb: LODA(n_bins=nb)))
    return p
POOL = build_pool()

def val_features(name):
    """Reconstruct validation-normal features for one oddbench dataset, seed 0, as the arm does."""
    d = np.load(os.path.join(ROOT, "data", "oddbench", name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # mixed-label dedup
    uq, first, inv = np.unique(X, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel(); lo = np.full(len(uq),2,int); hi = np.full(len(uq),-1,int)
    np.minimum.at(lo,inv,y); np.maximum.at(hi,inv,y); mixed = lo!=hi
    keep = np.zeros(len(X),bool); keep[first]=True; keep &= ~mixed[inv]
    X,y = X[keep],y[keep]
    if len(X)>5000:
        r=np.random.RandomState(0); k=r.choice(len(X),5000,replace=False); X,y=X[k],y[k]
    nm = np.where(y==0)[0]
    g=np.random.default_rng(0); idx=np.arange(len(nm)); g.shuffle(idx)
    c1=int(0.6*len(idx))
    va = nm[idx[c1:int(0.8*len(idx))]]
    return X[va]

def edge_pick(Xval, q=0.01, k_clusters=1, seed=0):
    """Fit on the core (1-q), score the edge (top q farthest from cluster centre), rank by AUC."""
    Xs = StandardScaler().fit_transform(Xval)
    Z = Xs if Xs.shape[1] <= 16 else PCA(n_components=16, random_state=0).fit_transform(Xs)
    if k_clusters <= 1:
        centers = Z.mean(0, keepdims=True); lab = np.zeros(len(Z), int)
    else:
        km = MiniBatchKMeans(n_clusters=min(k_clusters, max(2,len(Z)//20)), random_state=seed, n_init=5).fit(Z)
        lab = km.labels_; centers = km.cluster_centers_
    dist = np.linalg.norm(Z - centers[lab], axis=1)
    n_edge = max(10, int(q*len(Z)))
    edge = np.argsort(-dist)[:n_edge]
    core = np.setdiff1d(np.arange(len(Z)), edge)
    g = np.random.default_rng(seed); c = core.copy(); g.shuffle(c)
    cut = int(0.7*len(c)); tr, ho = c[:cut], c[cut:]
    if len(tr) < 30 or len(ho) < 10:
        return None
    Xtr = Xval[tr]; Xev = np.vstack([Xval[ho], Xval[edge]]); yev = np.r_[np.zeros(len(ho)), np.ones(len(edge))]
    best_v, best_auc = None, -1
    scores = {}
    import io, contextlib
    for vname, ctor in POOL:
        try:
            m = ctor()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            s = np.asarray(m.decision_function(Xev), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12:
                scores[vname] = roc_auc_score(yev, s)
        except Exception:
            continue
    return scores

# ---- run on included oddbench datasets ----
M = pd.read_csv(os.path.join(S,"HADB_MANIFEST.csv"))
inc = M[(M.include)&(M.corpus=="oddbench")]
D = pd.read_csv(os.path.join(S,"hadb_oddbench.csv"))
D["ap_norm"]=(D.ap-D.base_rate)/(1-D.base_rate)
apn = D.groupby(["dataset","variant"]).ap_norm.mean()
em_v = D.groupby(["dataset","variant"]).em.mean()

names = list(inc.dataset)[:18]
rows=[]
for nm in names:
    try: Xval = val_features(nm)
    except Exception as e: continue
    if len(Xval) < 60: continue
    sc = edge_pick(Xval, q=0.01, k_clusters=1)
    if not sc or len(sc)<3: continue
    ap = apn.loc[nm]; best = ap.max()
    edge_v = max(sc, key=sc.get)
    em_pick = em_v.loc[nm].idxmax()
    rows.append(dict(dataset=nm, best=best,
                     edge_regret=best-ap.get(edge_v,np.nan),
                     em_regret=best-ap.get(em_pick,np.nan),
                     random_regret=best-ap.mean()))
R=pd.DataFrame(rows)
print(f"edge smoke on {len(R)} oddbench datasets (k=1 cluster, top 1% edge):")
print(f"  edge   mean regret {R.edge_regret.mean():.4f}")
print(f"  EM     mean regret {R.em_regret.mean():.4f}")
print(f"  random mean regret {R.random_regret.mean():.4f}")
print(f"  edge beats random on {int((R.edge_regret<R.random_regret).sum())}/{len(R)} datasets")
print(f"  VERDICT: {'PROMISING - full run' if R.edge_regret.mean() < R.random_regret.mean()-0.01 else 'no better than random - drop'}")
