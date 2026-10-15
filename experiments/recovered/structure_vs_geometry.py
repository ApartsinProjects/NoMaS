"""Does the NORMAL data's structure predict the anomaly GEOMETRY (which detector family is the oracle)?
Hypothesis (user): dense / low-joint-structure normals -> marginal anomalies dominate -> marginal
detectors win; manifold / high-joint-structure normals -> joint anomalies -> neighbor detectors win.
If so, a label-free structure statistic can route the probe weighting.
S (normal structure, from Xtr only): effective dim (PCs for 90% var / d), mean|feature correlation|,
  kNN-distance CV, and dependency-probe separability. G (geometry): best-marginal ap - best-neighbor ap
  (>0 => marginal oracle). Correlate S vs G across 151 tasks."""
import json, glob, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
BUN = os.path.join(S, "streamline", "bundle_v2"); DEEP = lambda d: d.startswith("D_")
def ld(d):
    J={}
    for fn in glob.glob(os.path.join(d,"shard_*.jsonl")):
        for ln in open(fn):
            try: r=json.loads(ln)
            except: continue
            if "key" in r: J[r["key"]]=r
    return J
clstd = ld(os.path.join(S,"streamline","deepres_v2_clstd")); clfix = ld(os.path.join(S,"streamline","deepres_v2_clfix"))
lb = pd.read_csv(os.path.join(S,"streamline","STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap>0.05)&(lb.reg_random>0.02)&(lb.n_h>=100)].set_index("key")
MARG = lambda d: d.startswith(("HBOS","ECOD","COPOD"))
NEIGH = lambda d: d.startswith(("LOF","KNN"))
rows = []
for f in sorted(glob.glob(os.path.join(BUN,"*.npz"))):
    key = os.path.basename(f)[:-4]
    if key not in keep.index or key not in clstd: continue
    ap = {d:v for d,v in clstd[key]["ap"].items() if not DEEP(d)}
    marg = [ap[d] for d in ap if MARG(d)]; neigh = [ap[d] for d in ap if NEIGH(d)]
    if not marg or not neigh: continue
    G = max(marg) - max(neigh)                          # >0 => marginal detector is the oracle
    z = np.load(f); Xtr = z["Xtr"].astype("float64")
    Zt = StandardScaler().fit_transform(Xtr); n,d = Zt.shape
    # S1 effective dimensionality (fraction of dims for 90% variance) -> low = strong manifold/joint
    pca = PCA().fit(Zt); cum = np.cumsum(pca.explained_variance_ratio_); effdim = (np.searchsorted(cum,0.90)+1)/d
    # S2 mean abs off-diagonal correlation -> high = strong joint dependency
    C = np.corrcoef(Zt.T); mc = np.mean(np.abs(C[np.triu_indices(d,1)])) if d>1 else 0.0
    # S3 kNN-distance CV -> high = varying density / clustered
    nn = NearestNeighbors(n_neighbors=min(6,n-1)).fit(Zt); dd = nn.kneighbors(Zt)[0][:,1:].mean(1); cv = dd.std()/(dd.mean()+1e-9)
    # S4 dependency-probe separability = max classical a1 (how detectable is a joint violation)
    a1 = [v for dd_,v in clfix[key]["a1"].items() if not DEEP(dd_)]; sep = max(a1) if a1 else np.nan
    rows.append(dict(key=key, G=G, effdim=effdim, meancorr=mc, knn_cv=cv, dep_sep=sep, d=d))
df = pd.DataFrame(rows)
print(f"tasks: {len(df)}   marginal-oracle (G>0): {(df.G>0).sum()}  neighbor-oracle (G<0): {(df.G<0).sum()}")
print("\n=== correlation of each NORMAL-structure statistic with anomaly geometry G (>0=marginal oracle) ===")
print("(hypothesis: dense/independent normals -> marginal; so expect effdim +, meancorr -, knn_cv +/-)")
for s in ["effdim","meancorr","knn_cv","dep_sep","d"]:
    rho,p = spearmanr(df[s], df.G)
    print(f"  {s:9s} vs G : Spearman rho={rho:+.3f}  p={p:.4f}")
# can any statistic separate marginal- from neighbor-oracle tasks?
print("\n=== median statistic value, marginal-oracle vs neighbor-oracle tasks ===")
for s in ["effdim","meancorr","knn_cv","dep_sep","d"]:
    print(f"  {s:9s}: marginal={df[df.G>0][s].median():.3f}  neighbor={df[df.G<0][s].median():.3f}")
df.to_csv(os.path.join(S,"STRUCTURE_VS_GEOMETRY.csv"), index=False)
