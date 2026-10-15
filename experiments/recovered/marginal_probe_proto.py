"""Prototype the MARGINAL-GAP probe to close the gap (marginal detectors under-rated by the dependency
probe). For a subset of features, resample the value to a RARE (low-density) observed value of that
feature's own marginal -> marginally atypical, in-range (realistic), grades HBOS/ECOD/PCA. Test:
SPARC by dependency (0.113), marginal alone, and Borda(dependency+marginal). Diagnostic: does the
marginal probe lift HBOS's grade off random? Classical pool; regret vs STREAM_V3 oracle, ap=clstd."""
import json, glob, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, wilcoxon
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
def _gen_dep(Xn, ns, frac=0.4, seed=0):   # dependency probe: resample from OBSERVED marginal (typical)
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d)); Hs = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt/max(cnt.sum(),1)
        lo, hi = np.percentile(col,2), np.percentile(col,98); ctr = (edges[:-1]+edges[1:])/2
        allowed = (cnt>0)&(ctr>=lo)&(ctr<=hi); w = np.where(allowed, np.maximum(dens,1e-6), 0.0)
        wn = w/w.sum() if w.sum()>0 else None; binidx = np.clip(np.digitize(col, edges[1:-1]),0,len(cnt)-1)
        Hs.append((wn,col,binidx))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1,int(frac*d)), replace=False):
            wn,col,binidx = Hs[j]
            base[j] = Xn[rng.integers(n),j] if wn is None else (lambda c: c[rng.integers(len(c))] if len(c) else Xn[rng.integers(n),j])(col[binidx==rng.choice(len(wn),p=wn)])
        out[r] = base
    return out
def _gen_marginal(Xn, ns, frac=0.4, seed=0):  # marginal-gap: resample to RARE (low-density) observed value
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d)); LV = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30)
        binidx = np.clip(np.digitize(col, edges[1:-1]), 0, len(cnt)-1)
        nz = cnt > 0
        thr = np.percentile(cnt[nz], 30) if nz.sum() > 3 else 0        # rare = low-count nonzero bins
        rare = np.where(nz & (cnt <= max(thr, 1)))[0]
        lowvals = col[np.isin(binidx, rare)]
        LV.append(lowvals if len(lowvals) >= 3 else col)               # fall back to full col if degenerate
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1,int(frac*d)), replace=False):
            lv = LV[j]; base[j] = lv[rng.integers(len(lv))]
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
lb = pd.read_csv(os.path.join(S,"streamline","STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap>0.05)&(lb.reg_random>0.02)&(lb.n_h>=100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
pool = _classical_pool()
reg = {"dep": [], "marg": [], "combo": []}
hbos_dep, hbos_marg = [], []
for f in sorted(glob.glob(os.path.join(BUN,"*.npz"))):
    key = os.path.basename(f)[:-4]
    if key not in oracle or key not in clstd: continue
    z = np.load(f); Xtr, Xval = z["Xtr"].astype("float64"), z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr, Zval = ss.transform(Xtr), ss.transform(Xval)
    dep = ss.transform(_gen_dep(Xval, 200)); marg = ss.transform(_gen_marginal(Xval, 200))
    y = np.r_[np.zeros(len(Zval)), np.ones(200)]
    a_dep, a_marg = {}, {}
    for nm, ctor in pool:
        try:
            m = ctor(); m.fit(Ztr); sv = np.asarray(m.decision_function(Zval), float)
            for probe, dd in [(dep, a_dep), (marg, a_marg)]:
                s = np.r_[sv, np.asarray(m.decision_function(probe), float)]
                if np.nanstd(s) > 1e-12 and np.all(np.isfinite(s)) and nm in clstd[key]["ap"]: dd[nm] = roc_auc_score(y, s)
        except Exception: pass
    ap = clstd[key]["ap"]; o = oracle[key]
    if len(a_dep) < 5 or len(a_marg) < 5: continue
    reg["dep"].append(o - ap[max(a_dep, key=a_dep.get)])
    reg["marg"].append(o - ap[max(a_marg, key=a_marg.get)])
    common = [d for d in a_dep if d in a_marg]
    rd = rankdata([a_dep[d] for d in common]); rm = rankdata([a_marg[d] for d in common])
    reg["combo"].append(o - ap[common[int(np.argmax(rd+rm))]])
    if "HBOS_b50" in a_dep and "HBOS_b50" in a_marg: hbos_dep.append(a_dep["HBOS_b50"]); hbos_marg.append(a_marg["HBOS_b50"])
base = np.array(reg["dep"])
print(f"tasks: {len(base)}   (SPARC classical pool; dependency probe = published 0.113)")
for name in ["dep","marg","combo"]:
    r = np.array(reg[name]); n = min(len(r), len(base))
    p = wilcoxon(r[:n], base[:n]).pvalue if name!="dep" and (r[:n]!=base[:n]).any() else float("nan")
    print(f"  {name:6s}: mean regret = {r.mean():.4f}   delta vs dep = {r[:n].mean()-base[:n].mean():+.4f}  p={p:.3f}")
print(f"\nHBOS_b50 mean grade: dependency probe={np.mean(hbos_dep):.3f} (was ~random 0.51)  marginal probe={np.mean(hbos_marg):.3f}")
