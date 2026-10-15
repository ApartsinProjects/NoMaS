"""Prototype: density-filtered SPARC probe. Generate 200 synthetic, score them by normal-data density
(kNN distance to train normals), FILTER by that score, re-grade classical detectors on the FILTERED
probe, and compare SPARC regret to the unfiltered baseline. Filters (all label-free, normals only):
  baseline   : all 200
  drop_p99   : drop synthetic above the 99th pct of normal scores (too easy)
  drop_p95   : drop above the 95th pct
  band_40_95 : keep only the [40th,95th] pct band (drop too-normal AND too-easy)
  band_40_90 : keep [40th,90th]
regret vs STREAM_V3 oracle, ap from clstd. Classical pool (SPARC's actual candidate set)."""
import json, glob, os, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
BUN = os.path.join(S, "streamline", "bundle_v2")
def _classical_pool():
    from pyod.models.iforest import IForest; from pyod.models.lof import LOF; from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS; from pyod.models.ecod import ECOD; from pyod.models.copod import COPOD
    from pyod.models.pca import PCA as PPCA; from pyod.models.cblof import CBLOF; from pyod.models.loda import LODA
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)), ("LOF", lambda: LOF(n_neighbors=20)),
         ("KNN", lambda: KNN(n_neighbors=5)), ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
         ("HBOS", lambda: HBOS(n_bins=10)), ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)), ("LODA", lambda: LODA(n_bins=10, random_state=0))]
    for k in [3,10,35,100,200]: p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k))); p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5,20,50]: p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3,0.7,0.9]: p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    return p
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
def ld(d):
    J={}
    for fn in glob.glob(os.path.join(d,"shard_*.jsonl")):
        for ln in open(fn):
            try: r=json.loads(ln)
            except: continue
            if "key" in r: J[r["key"]]=r
    return J
clstd = ld(os.path.join(S,"streamline","deepres_v2_clstd"))
import pandas as pd
lb = pd.read_csv(os.path.join(S,"streamline","STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap>0.05)&(lb.reg_random>0.02)&(lb.n_h>=100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
FILTERS = {"baseline": None, "drop_p99": (0,99), "drop_p95": (0,95), "band_40_95": (40,95), "band_40_90": (40,90)}
pool = _classical_pool()
reg = {f: [] for f in FILTERS}
for f in sorted(glob.glob(os.path.join(BUN,"*.npz"))):
    key = os.path.basename(f)[:-4]
    if key not in oracle or key not in clstd: continue
    z = np.load(f); Xtr, Xval = z["Xtr"].astype("float64"), z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr, Zval = ss.transform(Xtr), ss.transform(Xval)
    Zsyn = ss.transform(_gen_beta(Xval, 200))
    kk = min(5, len(Ztr)-1); nn = NearestNeighbors(n_neighbors=kk).fit(Ztr)
    sc = lambda X: nn.kneighbors(X)[0].mean(1)
    sn, sy = sc(Zval), sc(Zsyn)
    scores = {nm: None for nm,_ in pool}
    for nm, ctor in pool:
        try:
            m = ctor(); m.fit(Ztr)
            scores[nm] = (np.asarray(m.decision_function(Zval),float), np.asarray(m.decision_function(Zsyn),float))
        except Exception: pass
    ap = clstd[key]["ap"]; o = oracle[key]
    for fn_name, band in FILTERS.items():
        if band is None: mask = np.ones(len(Zsyn), bool)
        else:
            lo = np.percentile(sn, band[0]); hi = np.percentile(sn, band[1]); mask = (sy >= lo) & (sy <= hi)
        if mask.sum() < 20: mask = np.ones(len(Zsyn), bool)   # too few -> fall back to all
        y = np.r_[np.zeros(len(Zval)), np.ones(int(mask.sum()))]
        a1 = {}
        for nm,(sv,ssy) in [(nm, v) for nm,v in scores.items() if v is not None]:
            s = np.r_[sv, ssy[mask]]
            if np.nanstd(s) > 1e-12 and np.all(np.isfinite(s)) and nm in ap: a1[nm] = roc_auc_score(y, s)
        if len(a1) < 5: continue
        reg[fn_name].append(o - ap[max(a1, key=a1.get)])
base = np.array(reg["baseline"])
print(f"tasks: {len(base)}   (SPARC classical pool; regret lower=better; baseline unfiltered = published 0.113)")
for fn_name in FILTERS:
    r = np.array(reg[fn_name]); n = min(len(r), len(base))
    p = wilcoxon(r[:n], base[:n]).pvalue if fn_name != "baseline" and (r[:n]!=base[:n]).any() else float("nan")
    print(f"  {fn_name:11s}: mean regret = {r.mean():.4f}   delta vs baseline = {r[:n].mean()-base[:n].mean():+.4f}  p={p:.3f}")
