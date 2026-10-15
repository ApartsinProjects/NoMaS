# -*- coding: utf-8 -*-
"""SelBench: a benchmark curated for MODEL-SELECTION difficulty, not detection difficulty.

RATIONALE. A model-selection method can only be judged where the CHOICE MATTERS. Measured
on the paper's own correct protocol, 43% of the 69-dataset benchmark is dead for this
purpose: 16/69 are CEILING (best >= 0.95, spread < 0.03 - every detector already wins) and
14/69 are FLOOR (best < 0.70 - nothing worth choosing). Averaging selection quality over
those is measuring noise.

This differs from all scouted prior art (Emmott 2015 point difficulty, Roechner 2025,
Wu & Keogh triviality, HADB): those curate by ANOMALY difficulty. This curates by SELECTION
difficulty - the spread of achievable performance across the candidate pool.

DESIGN:
  1. Score a LARGE pool (many families x hyperparameters), not the 9 defaults.
  2. Use the PAPER'S TARGET PROTOCOL (train on 80% of NORMALS, score held-out 20% normals +
     all anomalies) - never the in-sample contaminated target that broke today's harnesses.
  3. Keep a dataset if the pool's spread is large enough that the choice matters.
  4. FLOOR RESCUE: a dataset where the 9 classical detectors are all bad is NOT dropped
     until the expanded pool has had its chance. Report how many floor datasets the larger
     pool rescues into usable spread.

Outputs a manifest with, per dataset: pool-best/worst/spread under the classical-9 vs the
full pool, the dead-zone classification under each, and whether expansion rescued it.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as P  # noqa: E402

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.pca import PCA as PPCA
from pyod.models.cblof import CBLOF
from pyod.models.loda import LODA

SEEDS = [0, 1, 2]
MAX_N = 5000
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\selbench.csv"

# CLASSICAL-9 = the paper's current panel (defaults). OCSVM deliberately excluded: it has a
# documented repeatable libsvm C-level crash on this machine.
CLASSIC = [
    ("IForest", lambda: IForest(n_estimators=100, random_state=0)),
    ("LOF",     lambda: LOF(n_neighbors=20)),
    ("KNN",     lambda: KNN(n_neighbors=5)),
    ("ECOD",    lambda: ECOD()),
    ("COPOD",   lambda: COPOD()),
    ("HBOS",    lambda: HBOS(n_bins=10)),
    ("PCA",     lambda: PPCA(n_components=0.5, random_state=0)),
    ("CBLOF",   lambda: CBLOF(n_clusters=8, random_state=0)),
    ("LODA",    lambda: LODA(n_bins=10)),
]

def full_pool():
    P_ = list(CLASSIC)
    for k in [3, 10, 35, 50, 100, 200]:
        P_.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k)))
        P_.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50, 100]:
        P_.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9, 0.99]:
        P_.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    for ncl in [4, 16, 32]:
        P_.append((f"CBLOF_c{ncl}", lambda ncl=ncl: CBLOF(n_clusters=ncl, random_state=0)))
    for ne in [50, 300]:
        P_.append((f"IF_n{ne}", lambda ne=ne: IForest(n_estimators=ne, random_state=0)))
    for nb in [20, 50]:
        P_.append((f"LODA_b{nb}", lambda nb=nb: LODA(n_bins=nb)))
    return P_

POOL = full_pool()
CLASSIC_NAMES = {n for n, _ in CLASSIC}
print(f"pool: {len(POOL)} variants ({len(CLASSIC)} classical + {len(POOL)-len(CLASSIC)} extra)",
      flush=True)


def score(ctor, Xtr, Xte):
    import contextlib, io as _io
    try:
        m = ctor()
        with contextlib.redirect_stdout(_io.StringIO()):
            m.fit(Xtr)
        s = np.asarray(m.decision_function(Xte), float)
        return s if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12 else None
    except Exception:
        return None


datasets = []
for sub in ("adbench", "dami"):
    d = os.path.join(ROOT, "data", sub)
    if os.path.isdir(d):
        datasets += P.load_npz_dir(d)
print(f"loaded {len(datasets)} datasets", flush=True)

rows, t0 = [], time.time()
for ds in datasets:
    X, y = ds.X, ds.y
    if len(X) > MAX_N:
        r = np.random.RandomState(0)
        keep = r.choice(len(X), MAX_N, replace=False)
        X, y = X[keep], y[keep]
    if y.sum() < 10 or y.sum() == len(y):
        continue
    for seed in SEEDS:
        # PAPER'S TARGET PROTOCOL: train on 80% of NORMALS, score held-out normals+anomalies
        g = np.random.default_rng(seed)
        nm = y == 0
        Xn = X[nm]
        idx = np.arange(len(Xn)); g.shuffle(idx)
        cut = int(0.8 * len(idx))
        Xtr = Xn[idx[:cut]]
        Xte = np.vstack([Xn[idx[cut:]], X[~nm]])
        yte = np.r_[np.zeros(len(idx) - cut), np.ones(int((~nm).sum()))]
        if len(Xtr) < 50 or yte.sum() < 5:
            continue
        for name, ctor in POOL:
            s = score(ctor, Xtr, Xte)
            if s is None:
                continue
            rows.append(dict(dataset=ds.name, seed=seed, variant=name,
                             is_classic=name in CLASSIC_NAMES,
                             auc=float(roc_auc_score(yte, s)),
                             ap=float(average_precision_score(yte, s))))
    print(f"{ds.name}: done | {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

D = pd.DataFrame(rows)
D.to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(D)} datasets={D.dataset.nunique()}")

if len(D):
    def summarize(sub, tag):
        g = sub.groupby(["dataset", "seed"]).auc.agg(["max", "min", "mean"])
        return g.groupby("dataset").mean().rename(
            columns={"max": f"best_{tag}", "min": f"worst_{tag}", "mean": f"mean_{tag}"})

    c = summarize(D[D.is_classic], "cls")
    f = summarize(D, "all")
    M = c.join(f)
    M["spread_cls"] = M.best_cls - M.mean_cls
    M["spread_all"] = M.best_all - M.mean_all

    def zone(best, spread):
        if best >= 0.95 and spread < 0.03:
            return "ceiling"
        if best < 0.70:
            return "floor"
        return "live"
    M["zone_cls"] = [zone(b, s) for b, s in zip(M.best_cls, M.spread_cls)]
    M["zone_all"] = [zone(b, s) for b, s in zip(M.best_all, M.spread_all)]

    print("\n=== dead zones: CLASSICAL-9 vs FULL POOL ===")
    print(pd.crosstab(M.zone_cls, M.zone_all, rownames=["classical-9"],
                      colnames=["full pool"]).to_string())
    resc = M[(M.zone_cls == "floor") & (M.zone_all == "live")]
    print(f"\n  FLOOR datasets RESCUED by the larger pool: {len(resc)}/"
          f"{int((M.zone_cls=='floor').sum())}")
    if len(resc):
        print(resc[["best_cls", "best_all", "spread_all"]].round(3).to_string())
    print(f"\n  best-AUC gain from expanding the pool: mean "
          f"{(M.best_all-M.best_cls).mean():+.4f}, max {(M.best_all-M.best_cls).max():+.4f}")

    print("\n=== SelBench inclusion (choice actually matters) ===")
    for thr in [0.05, 0.10, 0.15]:
        keep = M[(M.zone_all == "live") & (M.spread_all >= thr)]
        print(f"  spread_all >= {thr:.2f} and live: {len(keep):3d}/{len(M)} datasets")
    M.reset_index().to_csv(OUT.replace(".csv", "_manifest.csv"), index=False)
    print(f"\nwrote {OUT.replace('.csv','_manifest.csv')}")
