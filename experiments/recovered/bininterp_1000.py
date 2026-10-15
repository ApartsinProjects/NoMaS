# -*- coding: utf-8 -*-
"""Table 7 ablation second row at 1000 points: the bin-INTERPOLATION probe (draw a uniform value WITHIN
the chosen histogram bin, not an observed value) at 1000 pts, classical pool, regret vs STREAM_V3 oracle.
The published @200 value is 0.122; observed-value @1000 is 0.109 (recompute_1000). Uses clstd ap."""
import json, glob, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
ST = os.path.join(S, "streamline"); BUN = os.path.join(ST, "bundle_v2")
def classical_pool():
    from pyod.models.iforest import IForest; from pyod.models.lof import LOF; from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS; from pyod.models.ecod import ECOD; from pyod.models.copod import COPOD
    from pyod.models.pca import PCA as PPCA; from pyod.models.cblof import CBLOF; from pyod.models.loda import LODA
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)), ("LOF", lambda: LOF(n_neighbors=20)),
         ("KNN", lambda: KNN(n_neighbors=5)), ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
         ("HBOS", lambda: HBOS(n_bins=10)), ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)), ("LODA", lambda: LODA(n_bins=10))]
    for k in [3, 10, 35, 100, 200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k))); p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50]: p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9]: p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    return p

def gen_bininterp(Xn, ns, beta=1.0, frac=0.4, seed=0):
    """bin-interpolation: draw a UNIFORM value within the chosen histogram bin (continuous, not observed)."""
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d)); Hs = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 2), np.percentile(col, 98); ctr = (edges[:-1] + edges[1:]) / 2
        allowed = (cnt > 0) & (ctr >= lo) & (ctr <= hi); w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
        wn = w / w.sum() if w.sum() > 0 else None
        Hs.append((wn, edges))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            wn, edges = Hs[j]
            if wn is None: base[j] = Xn[rng.integers(n), j]
            else:
                b = rng.choice(len(wn), p=wn); base[j] = rng.uniform(edges[b], edges[b + 1])
        out[r] = base
    return out

clsap = {}
for l in open(os.path.join(S, "clsstd_valuni", "ap.jsonl")):
    r = json.loads(l); clsap[r["key"]] = r["ap"]
lb = pd.read_csv(os.path.join(ST, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
files = [f for f in sorted(glob.glob(os.path.join(BUN, "*.npz"))) if os.path.basename(f)[:-4] in keep.index and os.path.basename(f)[:-4] in clsap]
print(f"datasets: {len(files)}", flush=True)
R = []
for f in files:
    key = os.path.basename(f)[:-4]
    z = np.load(f); Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
    if len(Zval) < 20: continue
    P = gen_bininterp(Zval, 1000); y = np.r_[np.zeros(len(Zval)), np.ones(len(P))]
    g = {}
    for nm, ctor in classical_pool():
        try: m = ctor(); m.fit(Ztr)
        except Exception: continue
        sv = m.decision_function(Zval)
        if nm in clsap[key]: g[nm] = roc_auc_score(y, np.r_[sv, m.decision_function(P)])
    dets = [d for d in g if d in clsap[key]]
    if len(dets) < 5: continue
    sel = max(dets, key=lambda d: g[d]); R.append(oracle[key] - clsap[key][sel])
print(f"\nbin-interpolation probe @1000, classical regret = {np.mean(R):.4f}  (published @200 = 0.122; observed-value @1000 = 0.109)")

