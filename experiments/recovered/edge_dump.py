# -*- coding: utf-8 -*-
"""Dump the FULL per-(dataset, cluster, detector) pseudo-AUC matrix plus label-free
edge tags, so the stratified-scoring analysis can be re-run cheaply without refitting.

Labels are used ONLY to compute each detector's true AUC (evaluation); the pseudo-task
and the edge tags never see y.

Usage:  python edge_dump.py <dir> <max_datasets> <out.csv>
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import embed, cluster, fit_and_score  # noqa: E402

DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
SEED, K = 0, 30

sub = sys.argv[1] if len(sys.argv) > 1 else "dami"
MAXDS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
OUT = sys.argv[3] if len(sys.argv) > 3 else \
    r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\edge_dump_dami.csv"
DATA = os.path.join(ROOT, "data", sub)


def edge_tags(Z, labels, j):
    m = labels == j
    C, R = Z[m], Z[~m]
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


rows = []
files = sorted(f for f in os.listdir(DATA) if f.endswith(".npz"))[:MAXDS]
for fi, fn in enumerate(files, 1):
    ds = fn[:-4]
    a = np.load(os.path.join(DATA, fn))
    X, y = np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()
    if X.shape[0] < 200 or X.shape[0] > 20000 or y.sum() < 5:
        print(f"[{fi}/{len(files)}] skip {ds} (n={X.shape[0]}, anom={int(y.sum())})", flush=True)
        continue
    true = {}
    for d in DETS:
        s = fit_and_score(d, X, X)
        if s is not None:
            true[d] = float(roc_auc_score(y, s))
    if len(true) < 6:
        print(f"[{fi}/{len(files)}] skip {ds} (detectors failed)", flush=True)
        continue

    Xn = X[y == 0]
    Z = embed(Xn, dim=16)
    lab = cluster(Z, K, seed=SEED)
    rng = np.random.RandomState(SEED)
    nc = 0
    for j in np.unique(lab):
        m = lab == j
        nj = int(m.sum())
        if nj < 15 or nj > 0.25 * len(Xn):
            continue
        rest = np.where(~m)[0]; rng.shuffle(rest)
        nh = max(20, int(0.2 * len(rest)))
        hold, tr = rest[:nh], rest[nh:]
        Xsc = np.vstack([Xn[m], Xn[hold]])
        yt = np.r_[np.ones(nj), np.zeros(len(hold))]
        tags = edge_tags(Z, lab, j)
        for d in DETS:
            s = fit_and_score(d, Xn[tr], Xsc)
            if s is None or d not in true:
                continue
            rows.append(dict(dataset=ds, cluster=int(j), size=nj, detector=d,
                             pseudo_auc=float(roc_auc_score(yt, s)),
                             true_auc=true[d], **tags))
        nc += 1
    print(f"[{fi}/{len(files)}] {ds}: n={X.shape[0]} d={X.shape[1]} clusters={nc} "
          f"best={max(true, key=true.get)}", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT}  rows={len(rows)}")
