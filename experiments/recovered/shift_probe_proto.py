"""Prototype: cluster-SHIFT probe as a complement to SPARC's dependency probe. Cluster the normals,
coherently displace a cluster outward into lower density (scaled, clipped to observed ranges) to make
a plausible-valued displaced subpopulation. Grade classical detectors by AUC(shift-probe vs normals).
Compare SPARC by: dependency probe (current), shift probe alone, and Borda(dependency+shift). Also
check whether the shift probe recovers the marginal-detector oracle on SpectralImaging.
regret vs STREAM_V3 oracle, ap from clstd. Classical pool (SPARC's candidate set)."""
import json, glob, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
def _gen_shift(Ztr, Zval, ns=200, scale=1.0, seed=0):
    """coherently displace a cluster of normals outward into lower density (clip to observed range)."""
    rng = np.random.default_rng(seed); k = max(2, min(8, len(Ztr)//100))
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(Ztr)
    g = Ztr.mean(0); lo, hi = Ztr.min(0), Ztr.max(0); lab = km.predict(Zval); out = []
    tries = 0
    while len(out) < ns and tries < 200:
        tries += 1; c = rng.integers(k); pts = Zval[lab == c]
        if len(pts) < 3: continue
        d = km.cluster_centers_[c] - g; nd = np.linalg.norm(d)
        if nd < 1e-6:
            d = rng.standard_normal(Ztr.shape[1]); d = d/np.linalg.norm(d) * (np.linalg.norm(km.cluster_centers_[c]-g)+1.0)
        v = d * scale
        take = pts[rng.choice(len(pts), min(len(pts), ns-len(out)), replace=False)]
        out.extend(np.clip(take + v, lo, hi))
    while len(out) < ns: out.append(out[rng.integers(len(out))] if out else Zval[0])
    return np.array(out[:ns])
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
SCALES = [0.5, 1.0, 2.0]
pool = _classical_pool()
reg = {"dep": [], "shift@0.5": [], "shift@1.0": [], "shift@2.0": [], "combo@1.0": []}
spectral = {}
for f in sorted(glob.glob(os.path.join(BUN,"*.npz"))):
    key = os.path.basename(f)[:-4]
    if key not in oracle or key not in clstd: continue
    z = np.load(f); Xtr, Xval = z["Xtr"].astype("float64"), z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr, Zval = ss.transform(Xtr), ss.transform(Xval)
    dep = ss.transform(_gen_beta(Xval, 200))
    shifts = {s: _gen_shift(Ztr, Zval, 200, s) for s in SCALES}
    a1dep, a1sh = {}, {s: {} for s in SCALES}
    for nm, ctor in pool:
        try:
            m = ctor(); m.fit(Ztr); sv = np.asarray(m.decision_function(Zval), float)
            def auc(P):
                s = np.r_[sv, np.asarray(m.decision_function(P), float)]; y = np.r_[np.zeros(len(sv)), np.ones(len(P))]
                return roc_auc_score(y, s) if np.nanstd(s) > 1e-12 and np.all(np.isfinite(s)) else None
            ad = auc(dep)
            if ad is not None and nm in clstd[key]["ap"]: a1dep[nm] = ad
            for s in SCALES:
                ash = auc(shifts[s])
                if ash is not None and nm in clstd[key]["ap"]: a1sh[s][nm] = ash
        except Exception: pass
    ap = clstd[key]["ap"]; o = oracle[key]
    if len(a1dep) < 5: continue
    reg["dep"].append(o - ap[max(a1dep, key=a1dep.get)])
    for s in SCALES:
        if len(a1sh[s]) >= 5: reg[f"shift@{s}"].append(o - ap[max(a1sh[s], key=a1sh[s].get)])
    # combo Borda of dep + shift@1.0 over the shared detector set
    sh = a1sh[1.0]; common = [d for d in a1dep if d in sh]
    if len(common) >= 5:
        rd = rankdata([a1dep[d] for d in common]); rs = rankdata([sh[d] for d in common])
        tot = rd + rs; reg["combo@1.0"].append(o - ap[common[int(np.argmax(tot))]])
    if key == "ovrbench__SpectralImaging":
        spectral = {"dep_pick": max(a1dep, key=a1dep.get), "shift_pick": max(a1sh[1.0], key=a1sh[1.0].get),
                    "hbos_dep_a1": a1dep.get("HBOS_b50"), "hbos_shift_a1": a1sh[1.0].get("HBOS_b50"),
                    "hbos_ap": ap.get("HBOS_b50"), "oracle": max(ap.values())}
base = np.array(reg["dep"])
print(f"tasks: {len(base)}   (SPARC classical pool; dependency probe = published 0.113)")
for name in ["dep", "shift@0.5", "shift@1.0", "shift@2.0", "combo@1.0"]:
    r = np.array(reg[name]); n = min(len(r), len(base))
    p = wilcoxon(r[:n], base[:n]).pvalue if name != "dep" and (r[:n]!=base[:n]).any() else float("nan")
    print(f"  {name:11s}: mean regret = {r.mean():.4f}   delta vs dep = {r[:n].mean()-base[:n].mean():+.4f}  p={p:.3f}  (n={len(r)})")
print(f"\nSpectralImaging diagnostic (oracle is HBOS, ap={spectral.get('oracle'):.3f}):")
print(f"  dependency-probe HBOS_b50 a1={spectral.get('hbos_dep_a1')}  -> dep picks {spectral.get('dep_pick')}")
print(f"  SHIFT-probe HBOS_b50 a1={spectral.get('hbos_shift_a1')}  (ap={spectral.get('hbos_ap')})  -> shift picks {spectral.get('shift_pick')}")
