# -*- coding: utf-8 -*-
"""Internal-void probe via the SECOND DERIVATIVE of a Gaussian KDE. An internal void is a local MINIMUM
of density (surrounded by higher density), detected by Laplacian Dp(x) > 0 (point enclosed at balanced
distance) rather than an outer tail (Dp <= 0 / monotone descent). Candidates are marginally-safe by
construction (convex combos of distant normal pairs + dependency-shuffles), then FILTERED to Dp>0 and
binned by density level. Detectors graded by nAP separating void points from normals per level."""
import os, sys, glob, json, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score
LEVELS = [50, 70, 85, 95]     # density percentiles (low p = deeper void); graded difficulty


def _pool():
    from pyod.models.iforest import IForest; from pyod.models.lof import LOF; from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS; from pyod.models.ecod import ECOD; from pyod.models.copod import COPOD
    from pyod.models.pca import PCA as PPCA; from pyod.models.cblof import CBLOF; from pyod.models.loda import LODA
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)), ("LOF", lambda: LOF(n_neighbors=20)),
         ("KNN", lambda: KNN(n_neighbors=5)), ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
         ("HBOS", lambda: HBOS(n_bins=10)), ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)), ("LODA", lambda: LODA(n_bins=10, random_state=0))]
    for k in [3, 10, 35, 100, 200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k))); p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50]:
        p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9]:
        p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    return p


def apn(y, s):
    b = y.mean(); return float((average_precision_score(y, s) - b) / (1 - b + 1e-12))


def kde_p_lap(X, ref, h):
    """Gaussian-KDE density p and Laplacian Dp at rows of X, against ref points (bandwidth h)."""
    d = ref.shape[1]; P = np.empty(len(X)); L = np.empty(len(X))
    h2 = h * h
    for i, x in enumerate(X):
        dif = ref - x; r2 = np.einsum("ij,ij->i", dif, dif)
        K = np.exp(-r2 / (2 * h2))
        s = K.sum() + 1e-300
        P[i] = s / len(ref)
        L[i] = (K * (r2 / (h2 * h2) - d / h2)).sum() / len(ref)   # Laplacian of KDE
    return P, L


def gen_void(Xtr, Xval, seed=0):
    rng = np.random.default_rng(seed)
    ref = Xtr if len(Xtr) <= 1200 else Xtr[rng.choice(len(Xtr), 1200, replace=False)]
    n, d = ref.shape
    nn = NearestNeighbors(n_neighbors=min(11, n)).fit(ref)
    h = np.median(nn.kneighbors(Xval)[0][:, 1:]) + 1e-9
    # candidate pool, all marginally-safe (values stay convex combos / real draws)
    C = []
    # (a) convex combos of DISTANT normal pairs -> land in gaps
    idx = rng.integers(len(Xval), size=(1200, 2))
    for a, b in idx:
        t = rng.uniform(0.35, 0.65); C.append(t * Xval[a] + (1 - t) * Xval[b])
    # (b) dependency-shuffle: swap ~40% coords between two normals
    for a, b in rng.integers(len(Xval), size=(800, 2)):
        x = Xval[a].copy(); j = rng.choice(d, max(1, int(0.4 * d)), replace=False); x[j] = Xval[b][j]; C.append(x)
    C = np.array(C)
    P, L = kde_p_lap(C, ref, h)
    Pn, Ln = kde_p_lap(Xval[rng.choice(len(Xval), min(400, len(Xval)), replace=False)], ref, h)
    enclosed = L > 0                                   # Laplacian>0 = local-min curvature = internal void
    Cv = C[enclosed]; Pv = P[enclosed]
    if len(Cv) < 20:
        Cv, Pv = C, P                                  # fallback
    # bin by density level (lower p percentile = deeper void)
    out = {}
    for q in LEVELS:
        thr = np.percentile(Pv, 100 - q)               # keep the lowest-density q% ... graded
        sel = Cv[Pv <= np.percentile(Pv, q)]
        out[q] = sel if len(sel) >= 10 else Cv
    return out, float(np.median(np.max(np.abs(Cv), axis=1))), float((L > 0).mean())


def void_select(Xtr, Xval):
    syn, mz, frac_enc = gen_void(Xtr, Xval)
    scores = {}
    for nm, ctor in _pool():
        try:
            m = ctor(); m.fit(Xtr); sv = np.asarray(m.decision_function(Xval), float)
            curve = []
            for q in LEVELS:
                sa = np.asarray(m.decision_function(syn[q]), float)
                y = np.r_[np.zeros(len(sv)), np.ones(len(sa))]; s = np.r_[sv, sa]
                curve.append(apn(y, s) if np.isfinite(s).all() and np.std(s) > 1e-12 else 0.0)
            w = np.array([1.0 / (i + 1) for i in range(len(LEVELS))])   # weight near-normal (hard) levels
            scores[nm] = float(np.dot(w, curve) / w.sum())
        except Exception:
            pass
    return max(scores, key=scores.get), scores, mz, frac_enc


if __name__ == "__main__":
    D = os.path.dirname(os.path.abspath(__file__))
    Jc = {}
    for fn in glob.glob(os.path.join(D, "deepres_v2_clstd", "shard_*.jsonl")):
        for ln in open(fn):
            try:
                r = json.loads(ln)
                if "ap" in r: Jc[r["key"]] = r["ap"]
            except Exception: pass
    import pandas as pd
    lb = pd.read_csv(os.path.join(D, "STREAM_V3_LEADERBOARD.csv")).set_index("key")
    targets = ["Annthyroid", "138_CATSv2", "ShillBidding", "ProtestViolence", "AxieInfinityBug", "BugSeverity"]
    for f in sorted(glob.glob(os.path.join(D, "bundle_v2", "*.npz"))):
        key = os.path.basename(f)[:-4]
        if not any(t in key for t in targets) or key not in Jc:
            continue
        z = np.load(f); Xtr, Xval = z["Xtr"].astype(float), z["Xval"].astype(float)
        ss = StandardScaler().fit(Xtr); Xtr, Xval = ss.transform(Xtr), ss.transform(Xval)
        sel, sc, mz, fe = void_select(Xtr, Xval)
        ap = Jc[key]; orc = lb.loc[key, "oracle_ap"] if key in lb.index else max(ap.values())
        print(f"{key[:28]:28s} VOID->{sel:9s} reg {orc-ap.get(sel,0):.3f} | oracle {max(ap,key=ap.get):9s}({orc:.2f}) "
              f"LOF_k10 reg {orc-ap.get('LOF_k10',0):.3f} | max|z|={mz:.2f}(norm~1.85) enclosed={fe*100:.0f}% top3={sorted(sc,key=sc.get,reverse=True)[:3]}")
