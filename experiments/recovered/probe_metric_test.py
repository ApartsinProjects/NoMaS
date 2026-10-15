"""Does grading the probe by ap_norm (the eval metric) beat grading by ROC-AUC (current SPARC)?
Local standardized classical re-score: per task, fit each classical detector on Ztr, score Zval
(normals) + the fixed observed-value probe syn1 (200 pts), grade by (a) ROC-AUC and (b) ap_norm on
the probe. SPARC pick = argmax of each grade; regret vs STREAM_V3 oracle, ap from clstd."""
import json, glob, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
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

def _gen_beta(Xn, ns, beta, frac=0.4, seed=0):
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

def apn(y, s): bb=y.mean(); return float((average_precision_score(y,s)-bb)/(1-bb+1e-12))

def loaddir(d):
    J={}
    for fn in glob.glob(os.path.join(d,"shard_*.jsonl")):
        for ln in open(fn):
            try: r=json.loads(ln)
            except: continue
            if "key" in r: J[r["key"]]=r
    return J
clstd = loaddir(os.path.join(S,"streamline","deepres_v2_clstd"))
lb = pd.read_csv(os.path.join(S,"streamline","STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap>0.05)&(lb.reg_random>0.02)&(lb.n_h>=100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
pool = _classical_pool()
reg_auc, reg_ap, ties = [], [], 0
for f in sorted(glob.glob(os.path.join(BUN,"*.npz"))):
    key = os.path.basename(f)[:-4]
    if key not in oracle or key not in clstd: continue
    z = np.load(f); Xtr, Xval = z["Xtr"].astype("float64"), z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr, Zval = ss.transform(Xtr), ss.transform(Xval)
    syn = ss.transform(_gen_beta(Xval, 200, 1.0))  # match probefix deep probe: gen on raw Xval, then standardize
    y = np.r_[np.zeros(len(Zval)), np.ones(200)]
    a_auc, a_ap = {}, {}
    for nm, ctor in pool:
        try:
            m = ctor(); m.fit(Ztr)
            sv = m.decision_function(Zval); s1 = m.decision_function(syn); s = np.r_[sv, s1]
            if np.nanstd(s) < 1e-12 or not np.all(np.isfinite(s)): continue
            a_auc[nm] = roc_auc_score(y, s); a_ap[nm] = apn(y, s)
        except Exception: pass
    ap = clstd[key]["ap"]; o = oracle[key]
    ca = [d for d in a_auc if d in ap]; cp = [d for d in a_ap if d in ap]
    if len(ca) < 5 or len(cp) < 5: continue
    pa = max(ca, key=lambda d: a_auc[d]); pp = max(cp, key=lambda d: a_ap[d])
    reg_auc.append(o - ap[pa]); reg_ap.append(o - ap[pp])
    if pa == pp: ties += 1
reg_auc, reg_ap = np.array(reg_auc), np.array(reg_ap)
print(f"tasks: {len(reg_auc)}")
print(f"SPARC probe by ROC-AUC (current) : mean regret = {reg_auc.mean():.4f}   (published 0.113)")
print(f"SPARC probe by ap_norm (eval metric): mean regret = {reg_ap.mean():.4f}")
print(f"delta (ap_norm - AUC) = {reg_ap.mean()-reg_auc.mean():+.4f}  p={wilcoxon(reg_auc,reg_ap).pvalue:.4f}  same pick {ties}/{len(reg_auc)}")
