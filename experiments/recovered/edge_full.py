# -*- coding: utf-8 -*-
"""Edge-stratified pseudo-tasks, k-means vs HDBSCAN arms. DUMP ONLY (analysis separate).

Emits one row per (dataset, arm, group, detector):
  group = a held-out cluster, or for HDBSCAN also the NOISE set (-1).
Each row carries the group's LABEL-FREE edge tags, the detector's pseudo-AUC on that
group, and the detector's TRUE auc (labels used only for that evaluation column).
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import embed, cluster, fit_and_score  # noqa: E402

DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
SEED, K = 0, 30
SUB = sys.argv[1] if len(sys.argv) > 1 else "dami"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\edge_full_dami.csv"
DATA = os.path.join(ROOT, "data", SUB)


def tags(Z, mask):
    """Label-free edge measures for the point set Z[mask] against the rest."""
    C, R = Z[mask], Z[~mask]
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    cen = float(np.linalg.norm((C.mean(0) - mu) / sd) / np.sqrt(Z.shape[1]))
    dist = np.linalg.norm((Z - mu) / sd, axis=1)
    frac = float((np.linalg.norm((C - mu) / sd, axis=1) > np.percentile(dist, 90)).mean())
    pct = np.array([(Z[:, k] < C[:, k].mean()).mean() for k in range(Z.shape[1])])
    marg = float(np.mean(np.abs(pct - 0.5)) * 2)
    q = max(1, min(Z.shape[1] - 1, Z.shape[1] // 2))
    p = PCA(n_components=q, random_state=SEED).fit(R)
    ec = float(np.mean(np.sum((C - p.inverse_transform(p.transform(C))) ** 2, 1)))
    er = float(np.mean(np.sum((R - p.inverse_transform(p.transform(R))) ** 2, 1)))
    return dict(edge_centroid=cen, edge_frac_outer=frac, edge_marginal=marg,
                edge_recon=ec / (er + 1e-12))


def run_group(rows, ds, arm, gid, Xn, Z, mask, true, rng):
    nj = int(mask.sum())
    if nj < 15 or nj > 0.25 * len(Xn):
        return 0
    rest = np.where(~mask)[0]
    rng.shuffle(rest)
    nh = max(20, int(0.2 * len(rest)))
    hold, tr = rest[:nh], rest[nh:]
    Xsc = np.vstack([Xn[mask], Xn[hold]])
    yt = np.r_[np.ones(nj), np.zeros(len(hold))]
    t = tags(Z, mask)
    got = 0
    for d in DETS:
        s = fit_and_score(d, Xn[tr], Xsc)
        if s is None or d not in true:
            continue
        rows.append(dict(dataset=ds, arm=arm, group=gid, size=nj, detector=d,
                         pseudo_auc=float(roc_auc_score(yt, s)), true_auc=true[d], **t))
        got = 1
    return got


rows = []
files = sorted(f for f in os.listdir(DATA) if f.endswith(".npz"))
t0 = time.time()
for fi, fn in enumerate(files, 1):
    ds = fn[:-4]
    a = np.load(os.path.join(DATA, fn))
    X, y = np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()
    if X.shape[0] < 200 or X.shape[0] > 20000 or y.sum() < 5:
        print(f"[{fi}/{len(files)}] skip {ds} n={X.shape[0]}", flush=True); continue
    true = {}
    for d in DETS:
        s = fit_and_score(d, X, X)
        if s is not None:
            true[d] = float(roc_auc_score(y, s))
    if len(true) < 6:
        print(f"[{fi}/{len(files)}] skip {ds} detectors-failed", flush=True); continue

    Xn = X[y == 0]
    Z = embed(Xn, dim=16)
    rng = np.random.RandomState(SEED)

    # ---- arm A: k-means (the paper's clusterer) ----
    lab = cluster(Z, K, seed=SEED)
    na = sum(run_group(rows, ds, "kmeans", int(j), Xn, Z, lab == j, true, rng)
             for j in np.unique(lab))

    # ---- arm B: HDBSCAN clusters + its NOISE set as an extra pseudo-anomaly group ----
    mcs = max(15, len(Xn) // 100)
    hl = HDBSCAN(min_cluster_size=mcs).fit_predict(Z)
    nb = sum(run_group(rows, ds, "hdbscan", int(j), Xn, Z, hl == j, true, rng)
             for j in np.unique(hl) if j >= 0)
    nnoise = int((hl == -1).sum())
    nb += run_group(rows, ds, "hdbscan", -1, Xn, Z, hl == -1, true, rng)

    print(f"[{fi}/{len(files)}] {ds} n={len(X)} d={X.shape[1]} best={max(true,key=true.get)} | "
          f"kmeans groups={na} | hdbscan clusters={(hl>=0).sum() and len(set(hl[hl>=0]))} "
          f"used={nb} noise={nnoise} | {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)}  elapsed={time.time()-t0:.0f}s")
