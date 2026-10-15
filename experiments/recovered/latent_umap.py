# -*- coding: utf-8 -*-
"""UMAP supplement to latent_sep.py: same datasets, same protocol, UMAP latents only.

UMAP is the sharpest test of the "clearer clusters -> better selection" premise: it
maximizes apparent separation and is documented to EXAGGERATE it. If the premise holds,
UMAP should win. If UMAP tops the silhouette ranking but not the regret ranking, that
isolates the failure: engineered separation is not the same as real structure.

Two settings span the faithfulness/separation trade-off:
  umap15  n_neighbors=15 (default, balanced)
  umap5   n_neighbors=5  (very local, maximally cluster-y / most exaggerating)
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (roc_auc_score, silhouette_score,
                             calinski_harabasz_score, davies_bouldin_score)
import umap

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import fit_and_score  # noqa: E402

DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
SEED, K, MAX_CLUSTERS, MAX_N = 0, 20, 12, 5000
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\latent_umap.csv"
SETS = [("dami", "Annthyroid"), ("dami", "Cardiotocography"), ("dami", "PageBlocks"),
        ("dami", "SpamBase"), ("dami", "Stamps"), ("dami", "WDBC"),
        ("dami", "Pima"), ("dami", "Waveform"), ("dami", "Wilt"),
        ("adbench", "14_glass"), ("adbench", "20_letter"), ("adbench", "22_magic.gamma")]

rows, t0 = [], time.time()
for sub, name in SETS:
    p = os.path.join(ROOT, "data", sub, name + ".npz")
    if not os.path.exists(p):
        continue
    a = np.load(p)
    X, y = np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()
    rng = np.random.RandomState(SEED)
    if len(X) > MAX_N:
        keep = rng.choice(len(X), MAX_N, replace=False)
        X, y = X[keep], y[keep]
    if y.sum() < 5:
        continue
    true = {}
    for d in DETS:
        s = fit_and_score(d, X, X)
        if s is not None:
            true[d] = float(roc_auc_score(y, s))
    if len(true) < 6:
        continue
    best = max(true.values())
    Xn = X[y == 0]
    Xs = StandardScaler().fit_transform(Xn)

    for lname, nn in [("umap15", 15), ("umap5", 5)]:
        try:
            Z = umap.UMAP(n_components=min(16, Xs.shape[1]), n_neighbors=nn,
                          random_state=SEED, verbose=False).fit_transform(Xs)
        except Exception as e:
            print(f"{name} {lname}: FAIL {e}", flush=True); continue
        lab = MiniBatchKMeans(n_clusters=min(K, max(2, len(Z) // 20)), random_state=SEED,
                              n_init=5, batch_size=256).fit_predict(Z)
        idx = rng.choice(len(Z), min(2000, len(Z)), replace=False)
        if len(np.unique(lab[idx])) < 2:
            continue
        sep = dict(silhouette=float(silhouette_score(Z[idx], lab[idx])),
                   calinski=float(calinski_harabasz_score(Z[idx], lab[idx])),
                   davies=float(-davies_bouldin_score(Z[idx], lab[idx])))
        acc = {d: [] for d in DETS}
        used = 0
        for j in np.unique(lab):
            if used >= MAX_CLUSTERS:
                break
            m = lab == j
            nj = int(m.sum())
            if nj < 15 or nj > 0.25 * len(Xn):
                continue
            rest = np.where(~m)[0]; rng.shuffle(rest)
            nh = max(20, int(0.2 * len(rest)))
            hold, tr = rest[:nh], rest[nh:]
            Xsc = np.vstack([Xn[m], Xn[hold]])
            yt = np.r_[np.ones(nj), np.zeros(len(hold))]
            for d in DETS:
                s = fit_and_score(d, Xn[tr], Xsc)
                if s is not None and d in true:
                    acc[d].append(roc_auc_score(yt, s))
            used += 1
        score = {d: np.mean(v) for d, v in acc.items() if v}
        if len(score) < 5 or used < 4:
            continue
        pick = max(score, key=score.get)
        rows.append(dict(dataset=name, latent=lname, dim=Z.shape[1], n_groups=used,
                         regret=best - true[pick], pick=pick, **sep))
        print(f"{name} {lname}: sil={sep['silhouette']:.3f} regret={best-true[pick]:.4f} "
              f"| {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)} elapsed={time.time()-t0:.0f}s")
