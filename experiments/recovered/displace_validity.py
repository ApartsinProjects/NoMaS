# -*- coding: utf-8 -*-
"""Is the displacement win REAL, or the out-of-range artifact?

The validity run showed unconstrained generated points were only 41.6% in-range / 2.1%
integrality-preserving, and that CLIPPING them to legality destroyed the family-gap
closure (-0.02 -> -0.45). That implied the closure was marginal detectors getting a free
giveaway from impossible values, not measured skill.

Raw-space displacement has the same exposure. This script measures, per delta:
  in_range  fraction of displaced points with EVERY feature inside the observed range
  int_ok    fraction of integer columns still integral
and then re-runs the pseudo-task with a CLIPPED (validity-projected) displacement arm.

If clipped displacement KEEPS the win, the effect is real.
If clipping destroys it, the win was the artifact and must be retracted.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
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
DELTAS = [0.0, 1.0, 4.0]
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\displace_validity.csv"

import glob
SETS = []
for _sub in ("dami", "adbench"):
    for _p in sorted(glob.glob(os.path.join(ROOT, "data", _sub, "*.npz"))):
        SETS.append((_sub, os.path.splitext(os.path.basename(_p))[0]))


def col_types(X):
    is_int = np.array([np.allclose(X[:, k], np.round(X[:, k])) for k in range(X.shape[1])])
    card = np.array([len(np.unique(X[:, k])) for k in range(X.shape[1])])
    return is_int, card


def project_valid(V, Xref, is_int, card):
    V = np.clip(V, Xref.min(0), Xref.max(0))
    for k in range(V.shape[1]):
        if card[k] <= 10:
            lv = np.unique(Xref[:, k])
            V[:, k] = lv[np.abs(V[:, k][:, None] - lv[None, :]).argmin(1)]
        elif is_int[k]:
            V[:, k] = np.round(V[:, k])
    return V


rows, t0 = [], time.time()
for sub, name in SETS:
    p = os.path.join(ROOT, "data", sub, name + ".npz")
    a = np.load(p)
    X, y = np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()
    rng = np.random.RandomState(SEED)
    if len(X) > MAX_N:
        keep = rng.choice(len(X), MAX_N, replace=False)
        X, y = X[keep], y[keep]
    if len(X) < 200 or y.sum() < 5:
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
    mu = Xn.mean(0)
    Z = embed(Xn, dim=16)
    lab = cluster(Z, K, seed=SEED)

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
        C = Xn[m]
        yt = np.r_[np.ones(nj), np.zeros(len(hold))]
        v = (C.mean(0) - mu) / sd
        nv = np.linalg.norm(v)
        v = v / nv if nv > 1e-9 else rng.normal(size=len(sd)) / np.sqrt(len(sd))
        lo, hi = Xtr.min(0), Xtr.max(0)

        for delta in DELTAS:
            Cd = C + delta * v * sd
            for arm, V in (("raw", Cd), ("clipped", project_valid(Cd.copy(), Xtr, is_int, card))):
                ir = float(np.mean(np.all((V >= lo - 1e-9) & (V <= hi + 1e-9), axis=1)))
                io_ = float(np.mean(np.isclose(V[:, is_int], np.round(V[:, is_int])))) if is_int.any() else np.nan
                Xsc = np.vstack([V, Xn[hold]])
                for d in DETS:
                    s = fit_and_score(d, Xtr, Xsc)
                    if s is None or d not in true:
                        continue
                    rows.append(dict(dataset=name, cluster=int(j), delta=delta, arm=arm,
                                     detector=d, fam=fam(d), in_range=ir, int_ok=io_,
                                     pseudo_auc=float(roc_auc_score(yt, s)), true_auc=true[d]))
        used += 1
    print(f"{name}: clusters={used} | {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)} elapsed={time.time()-t0:.0f}s")
