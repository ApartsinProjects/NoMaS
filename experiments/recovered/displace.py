# -*- coding: utf-8 -*-
"""DISPLACED-CLUSTER pseudo-anomalies: sweep how far the held-out cluster is moved.

IDEA: the held-out cluster sits INSIDE the global envelope, so global detectors correctly
score it low and are structurally handicapped. Translating the cluster OUTWARD turns it
into an exterior anomaly that global detectors can win legitimately. Sweeping the
displacement delta gives each detector a competence CURVE across anomaly types
(interior/local -> far exterior/global).

KEY VALIDITY PROPERTY: only the SCORED cluster is displaced. The training set is untouched,
so the detector is never fit on distorted data.

Displacement is applied in the RAW feature space (detectors are fit there), along the
direction from the global mean to the cluster mean, in units of per-feature std.

Pre-registered kill criterion: there must exist a delta range where (a) the global-minus-
local pseudo gap closes materially AND (b) the task has NOT saturated (mean pseudo-AUC
below ~0.95, so detectors still separate).

Labels are used only for the true-AUC evaluation column.
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

SEED, K, MAX_CLUSTERS, MAX_N = 0, 20, 10, 5000
DELTAS = [0.0, 0.5, 1.0, 2.0, 4.0]
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\displace.csv"

# balanced by true-best family (measured earlier today)
SETS = [("dami", "Annthyroid"), ("dami", "Cardiotocography"), ("dami", "PageBlocks"),
        ("dami", "SpamBase"), ("dami", "Stamps"), ("dami", "WDBC"),
        ("dami", "Waveform"), ("dami", "Wilt"),
        ("adbench", "20_letter"), ("adbench", "22_magic.gamma")]

rows, t0 = [], time.time()
for sub, name in SETS:
    p = os.path.join(ROOT, "data", sub, name + ".npz")
    if not os.path.exists(p):
        print(f"missing {name}", flush=True); continue
    a = np.load(p)
    X, y = np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()
    rng = np.random.RandomState(SEED)
    if len(X) > MAX_N:
        keep = rng.choice(len(X), MAX_N, replace=False)
        X, y = X[keep], y[keep]
    if y.sum() < 5:
        print(f"skip {name}", flush=True); continue

    true = {}
    for d in DETS:
        s = fit_and_score(d, X, X)
        if s is not None:
            true[d] = float(roc_auc_score(y, s))
    if len(true) < 6:
        print(f"skip {name} detectors-failed", flush=True); continue

    Xn = X[y == 0]
    sd = Xn.std(axis=0) + 1e-9
    mu = Xn.mean(axis=0)
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
        rest = np.where(~m)[0]
        rng.shuffle(rest)
        nh = max(20, int(0.2 * len(rest)))
        hold, tr = rest[:nh], rest[nh:]
        Xtr = Xn[tr]                                  # TRAINING SET NEVER DISPLACED
        C = Xn[m]
        # outward direction in RAW space, standardized units
        v = (C.mean(axis=0) - mu) / sd
        nv = np.linalg.norm(v)
        v = v / nv if nv > 1e-9 else rng.normal(size=len(sd)) / np.sqrt(len(sd))

        for delta in DELTAS:
            Cd = C + delta * v * sd                   # shift by delta * per-feature std
            Xsc = np.vstack([Cd, Xn[hold]])
            yt = np.r_[np.ones(nj), np.zeros(len(hold))]
            for d in DETS:
                s = fit_and_score(d, Xtr, Xsc)
                if s is None or d not in true:
                    continue
                rows.append(dict(dataset=name, cluster=int(j), delta=delta, detector=d,
                                 fam=fam(d), pseudo_auc=float(roc_auc_score(yt, s)),
                                 true_auc=true[d]))
        used += 1
    print(f"{name}: clusters={used} best={max(true,key=true.get)} | {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)} elapsed={time.time()-t0:.0f}s")
