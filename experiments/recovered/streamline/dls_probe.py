# -*- coding: utf-8 -*-
"""Density-Level-Set (DLS) evaluation for anomaly-detector selection.
From validation normals, move synthetic points toward lower-density regions along the negative gradient
of a Gaussian-KDE density, stopping at predefined sparsity percentiles (90/95/97.5/99) measured by a
kNN sparsity statistic. For each detector, nAP separating synthetic points at each level from normals
gives a response curve; the detector is scored by a (difficulty-weighted) mean nAP over levels and the
argmax is selected. Prototype: run on a few tasks and compare the pick to the oracle / LOF_k10 default."""
import os, sys, glob, json, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score

LEVELS = [90, 95, 97.5, 99]
NSYN = 150


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


def gen_dls(Xtr, Xval, seed=0):
    """Move Xval-seeded points to lower density; return {level: array of synthetic points}."""
    rng = np.random.default_rng(seed)
    ref = Xtr if len(Xtr) <= 1500 else Xtr[rng.choice(len(Xtr), 1500, replace=False)]
    d = ref.shape[1]
    nn = NearestNeighbors(n_neighbors=min(11, len(ref))).fit(ref)
    # kNN sparsity (mean dist to 10 NN) on Xval -> level thresholds
    dv, _ = nn.kneighbors(Xval); sparial = dv[:, 1:].mean(1)
    thr = {q: np.percentile(sparial, q) for q in LEVELS}
    h = np.median(dv[:, 1:]) + 1e-9                       # KDE bandwidth ~ local scale
    pts = Xval[rng.choice(len(Xval), NSYN, replace=True)].astype(float).copy()
    out = {q: [] for q in LEVELS}
    hit = [set() for _ in range(NSYN)]                   # levels already recorded per point
    lr = 0.5 * h; pending = list(range(NSYN))
    for step in range(80):
        if not pending:
            break
        for i in list(pending):
            x = pts[i]
            dist = np.linalg.norm(ref - x, axis=1)
            wsel = np.argsort(dist)[:50]                  # local neighbours
            w = np.exp(-(dist[wsel] ** 2) / (2 * h * h)); w = w / (w.sum() + 1e-12)
            grad = ((ref[wsel] - x) * w[:, None]).sum(0) / (h * h)   # ∇p (toward data)
            g = grad / (np.linalg.norm(grad) + 1e-9)
            pts[i] = x - lr * g                            # step toward lower density
            s = np.sort(np.linalg.norm(ref - pts[i], axis=1))[1:11].mean()
            for q in LEVELS:
                if q not in hit[i] and s >= thr[q]:
                    out[q].append(pts[i].copy()); hit[i].add(q)
            if LEVELS[-1] in hit[i]:
                pending.remove(i)
    return {q: (np.array(v) if len(v) >= 5 else Xval[rng.choice(len(Xval), 20)]) for q, v in out.items()}


def dls_select(Xtr, Xval, weight="difficulty"):
    syn = gen_dls(Xtr, Xval)
    scores = {}
    for nm, ctor in _pool():
        try:
            m = ctor(); m.fit(Xtr); sv = np.asarray(m.decision_function(Xval), float)
            curve = []
            for q in LEVELS:
                sa = np.asarray(m.decision_function(syn[q]), float)
                y = np.r_[np.zeros(len(sv)), np.ones(len(sa))]; s = np.r_[sv, sa]
                curve.append(apn(y, s) if np.isfinite(s).all() and np.std(s) > 1e-12 else 0.0)
            # difficulty weight: harder (near-normal, low q) levels weighted more
            w = np.array([1.0 / (i + 1) for i in range(len(LEVELS))]) if weight == "difficulty" else np.ones(len(LEVELS))
            scores[nm] = float(np.dot(w, curve) / w.sum())
        except Exception:
            pass
    sel = max(scores, key=scores.get)
    return sel, scores


if __name__ == "__main__":
    D = os.path.dirname(os.path.abspath(__file__))
    # cached ap for regret/oracle
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
        if not any(t in key for t in targets):
            continue
        z = np.load(f); Xtr, Xval = z["Xtr"].astype(float), z["Xval"].astype(float)
        ss = StandardScaler().fit(Xtr); Xtr, Xval = ss.transform(Xtr), ss.transform(Xval)
        sel, sc = dls_select(Xtr, Xval)
        if key not in Jc:
            print(f"{key[:30]}: no cached ap"); continue
        ap = Jc[key]; orc = lb.loc[key, "oracle_ap"] if key in lb.index else max(ap.values())
        top3 = sorted(sc, key=sc.get, reverse=True)[:3]
        print(f"{key[:30]:30s} DLS->{sel:9s}(ap {ap.get(sel,float('nan')):.2f}, reg {orc-ap.get(sel,0):.3f}) "
              f"oracle {max(ap,key=ap.get):9s}({orc:.2f}) LOF_k10 reg {orc-ap.get('LOF_k10',0):.3f}  top3={top3}")
