"""Cheap LOCAL check of the user's hypothesis, no Modal:
 (1) are the normals genuinely multi-cluster (so an inter-cluster VOID exists)?
 (2) my current NULL probe (val -> NEAREST train normal interpolant) sits WITHIN a cluster;
     a VOID probe (between two DIFFERENT clusters) sits in the gap.
 (3) neighbor distance: void points are FAR from all normals (LOF/KNN flag them),
     within-cluster null points are CLOSE (undetectable) -> confirms the two probes test
     different regions. If real anomalies also sit in the void, a void detector predicts
     real quality; deep that merges clusters scores the void as normal.
"""
import glob, os, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, silhouette_score
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
BUN = os.path.join(S, "streamline", "bundle_v2")
rng = np.random.default_rng(0)
files = sorted(glob.glob(os.path.join(BUN, "*.npz")))[:40]
rows = []
for f in files:
    key = os.path.basename(f)[:-4]
    try:
        z = np.load(f); Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
        Xh = z["Xh"].astype("float64")
    except Exception: continue
    if len(Xtr) < 60 or Xtr.shape[1] < 2: continue
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval); Zh = ss.transform(Xh)
    # pick #clusters by silhouette (2..6)
    best_k, best_s = 1, -1
    for k in range(2, 7):
        if len(Ztr) < k * 10: break
        lab = KMeans(k, n_init=5, random_state=0).fit_predict(Ztr)
        try: sc = silhouette_score(Ztr, lab)
        except Exception: sc = -1
        if sc > best_s: best_s, best_k = sc, k
    km = KMeans(best_k, n_init=5, random_state=0).fit(Ztr); cen = km.cluster_centers_
    nn = NearestNeighbors(n_neighbors=1).fit(Ztr)
    d_norm = nn.kneighbors(Zval, return_distance=True)[0][:, 0].mean()      # typical normal->normal dist
    # NULL probe (current): val -> nearest train normal
    T = Ztr[nn.kneighbors(Zval, return_distance=False)[:, 0]]
    lam = rng.uniform(0.25, 0.75, (len(Zval), 1)); Znull = lam * Zval + (1 - lam) * T
    d_null = nn.kneighbors(Znull, return_distance=True)[0][:, 0].mean()
    # VOID probe (proposed): midpoints between DIFFERENT cluster centroids (only if >1 cluster)
    d_void = np.nan; auc_void_vs_null = np.nan
    if best_k >= 2:
        pairs = [(i, j) for i in range(best_k) for j in range(i + 1, best_k)]
        mids = []
        for _ in range(200):
            i, j = pairs[rng.integers(len(pairs))]
            t = rng.uniform(0.35, 0.65); mids.append(t * cen[i] + (1 - t) * cen[j])
        Zvoid = np.array(mids)
        d_void = nn.kneighbors(Zvoid, return_distance=True)[0][:, 0].mean()
    # real hard anomalies: how far from the normal manifold?
    d_h = nn.kneighbors(Zh, return_distance=True)[0][:, 0].mean()
    rows.append(dict(key=key[:34], k=best_k, sil=round(best_s, 2),
                     d_norm=round(d_norm, 2), d_null=round(d_null, 2),
                     d_void=round(d_void, 2) if np.isfinite(d_void) else None,
                     d_anom=round(d_h, 2)))
import pandas as pd
df = pd.DataFrame(rows)
multi = df[df.k >= 2]
print(f"datasets checked: {len(df)}   multi-cluster (k>=2 by silhouette): {len(multi)}")
print("\nmean nearest-normal distance by point type (standardized units):")
print(f"  normal (val)        : {df.d_norm.mean():.2f}   <- reference")
print(f"  current NULL probe  : {df.d_null.mean():.2f}   <- {'WITHIN cluster (close to normal)' if df.d_null.mean() < 1.5*df.d_norm.mean() else 'far'}")
if len(multi):
    print(f"  proposed VOID probe : {multi.d_void.mean():.2f}   <- {'in the GAP (far from normals)' if multi.d_void.mean() > 1.5*multi.d_norm.mean() else 'still close'}")
print(f"  real hard anomaly   : {df.d_anom.mean():.2f}   <- where the truth lives")
print("\nper-dataset (first 15):")
print(df.head(15).to_string(index=False))
