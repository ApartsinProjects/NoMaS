# -*- coding: utf-8 -*-
"""Synthetic anomalies that lie OUTSIDE the clusters but REMAIN VALID in the original space.

Target property: marginally plausible, jointly implausible. Every feature value should be
something you could actually observe; the COMBINATION should not occur in the data.

Generators (all keep the clustering untouched; they only manufacture scored points):
  real     the actual held-out cluster                          (current NoMaS)
  pca4     sample 4 sigma off-cluster in PCA space, inverse-map  (unconstrained; may be
           numerically impossible in the original space)
  clip4    pca4 then PROJECTED BACK TO LEGALITY: clip to observed per-feature range,
           round integer-valued columns, snap low-cardinality columns to observed levels
  shuffle  each column resampled INDEPENDENTLY from the real normals: every value is a
           genuinely observed value, only the dependence is destroyed
           (NB: this is ADBench's "dependency" anomaly type as a generator)
  qpush    real cluster points with a random 30% of features replaced by values drawn from
           that feature's observed extreme tails (real values, extreme combination)
  xmix     cross-cluster recombination: feature BLOCKS taken from points in different
           clusters; each block real and internally consistent, the assembly is not

VALIDITY IS MEASURED, NOT ASSUMED:
  in_range  fraction of generated points with EVERY feature inside the observed [min,max]
  int_ok    fraction of integer-valued columns whose generated values are still integers
  nn_dist   mean standardized distance to the nearest real normal point (plausibility)

Labels are used only for the true-AUC evaluation column.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import embed, cluster, fit_and_score  # noqa: E402

DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
fam = lambda d: "local" if d in LOCAL else ("global" if d in GLOBAL else "other")
SEED, K, MAX_CLUSTERS, MAX_N = 0, 20, 8, 4000
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\gen_valid.csv"
SETS = [("dami", "Annthyroid"), ("dami", "Cardiotocography"), ("dami", "PageBlocks"),
        ("dami", "SpamBase"), ("dami", "Stamps"), ("dami", "WDBC"),
        ("dami", "Waveform"), ("dami", "Wilt"),
        ("adbench", "20_letter"), ("adbench", "22_magic.gamma")]


def col_types(X):
    """Which columns are integer-valued / low-cardinality (categorical-like)?"""
    is_int = np.array([np.allclose(X[:, k], np.round(X[:, k])) for k in range(X.shape[1])])
    card = np.array([len(np.unique(X[:, k])) for k in range(X.shape[1])])
    return is_int, card


def project_valid(V, Xref, is_int, card):
    """Push generated points back into the ORIGINAL-SPACE feasible set."""
    lo, hi = Xref.min(0), Xref.max(0)
    V = np.clip(V, lo, hi)
    for k in range(V.shape[1]):
        if card[k] <= 10:                      # categorical-like: snap to observed levels
            levels = np.unique(Xref[:, k])
            V[:, k] = levels[np.abs(V[:, k][:, None] - levels[None, :]).argmin(1)]
        elif is_int[k]:
            V[:, k] = np.round(V[:, k])
    return V


def validity(V, Xref, is_int, sd):
    lo, hi = Xref.min(0), Xref.max(0)
    in_range = float(np.mean(np.all((V >= lo - 1e-9) & (V <= hi + 1e-9), axis=1)))
    if is_int.any():
        int_ok = float(np.mean(np.isclose(V[:, is_int], np.round(V[:, is_int]))))
    else:
        int_ok = float("nan")
    A = (V - Xref.mean(0)) / sd
    B = (Xref - Xref.mean(0)) / sd
    idx = np.random.RandomState(0).choice(len(B), min(1500, len(B)), replace=False)
    d = np.sqrt(((A[:, None, :] - B[None, idx, :]) ** 2).sum(-1)).min(1)
    return in_range, int_ok, float(d.mean())


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

    Xn = X[y == 0]
    is_int, card = col_types(Xn)
    sd = Xn.std(0) + 1e-9
    Zp = embed(Xn, dim=16)
    lab = cluster(Zp, K, seed=SEED)
    mu_raw = Xn.mean(0)
    Xs = (Xn - mu_raw) / sd
    pca = PCA(n_components=min(16, Xs.shape[1], len(Xs) - 1), random_state=SEED).fit(Xs)
    Zl = pca.transform(Xs)
    dcol = Xn.shape[1]

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
        Xtr = Xn[tr]
        Xother = Xn[rest]
        yt = np.r_[np.ones(nj), np.zeros(len(hold))]

        # pca4: sample 4 sigma off-cluster in PCA space, inverse-map (unconstrained)
        Zc = Zl[m]
        v = rng.normal(size=(nj, Zc.shape[1])); v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        Zs = Zc.mean(0) + v * (Zc.std(0) + 1e-9) * 4.0
        pca4 = pca.inverse_transform(Zs) * sd + mu_raw

        gens = {"real": Xn[m], "pca4": pca4,
                "clip4": project_valid(pca4.copy(), Xtr, is_int, card)}

        # shuffle: independent per-column resample from the training normals
        gens["shuffle"] = np.column_stack(
            [Xtr[rng.randint(0, len(Xtr), nj), k] for k in range(dcol)])

        # qpush: real cluster points, 30% of features pushed to observed extreme tails
        q = Xn[m].copy()
        nfe = max(1, int(0.3 * dcol))
        for i in range(nj):
            ks = rng.choice(dcol, nfe, replace=False)
            for k in ks:
                col = Xtr[:, k]
                tail = col[col >= np.percentile(col, 95)] if rng.rand() < 0.5 else \
                       col[col <= np.percentile(col, 5)]
                if len(tail):
                    q[i, k] = tail[rng.randint(len(tail))]
        gens["qpush"] = q

        # xmix: feature blocks assembled from points in DIFFERENT clusters
        nblk = min(4, dcol)
        bounds = np.array_split(np.arange(dcol), nblk)
        xm = np.empty((nj, dcol))
        for i in range(nj):
            for b in bounds:
                src = Xother[rng.randint(len(Xother))]
                xm[i, b] = src[b]
        gens["xmix"] = xm

        for gname, V in gens.items():
            ir, io_, nd = validity(V, Xtr, is_int, sd)
            Xsc = np.vstack([V, Xn[hold]])
            for d in DETS:
                s = fit_and_score(d, Xtr, Xsc)
                if s is None or d not in true:
                    continue
                rows.append(dict(dataset=name, cluster=int(j), gen=gname, detector=d,
                                 fam=fam(d), pseudo_auc=float(roc_auc_score(yt, s)),
                                 true_auc=true[d], in_range=ir, int_ok=io_, nn_dist=nd))
        used += 1
    print(f"{name}: clusters={used} best={max(true,key=true.get)} | {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)} elapsed={time.time()-t0:.0f}s")
