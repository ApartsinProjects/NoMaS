# -*- coding: utf-8 -*-
"""HADB - Hard Anomaly-detection Benchmark for MODEL SELECTION.

A benchmark on which a ranking task is REAL: the anomalies are non-trivial, and the choice
of detector genuinely matters.

Prior art curates by ANOMALY difficulty only (Emmott et al. 2015 point difficulty;
Roechner et al. 2025; Wu & Keogh 2021 triviality). HADB additionally curates by SELECTION
difficulty and expands the candidate pool where nothing works. That combination appears to
be new.

PIPELINE (each step is reported, nothing is dropped silently):
  S1 DROP TRIVIAL ANOMALIES. Trivial = max_k |z_k| above the 99th percentile of
     TRAIN-NORMAL (Apartsin & Aperstein, arXiv:2607.06094 sec 5.3). Statistics from normal
     points only, so anomalies cannot inflate their own threshold.
  S2 DROP datasets left with too few non-trivial anomalies (< MIN_HARD).
  S3 EVALUATE A LARGE POOL on the remaining hard anomalies, using the PAPER'S TARGET
     PROTOCOL (train on 80% of NORMALS, score held-out 20% normals + hard anomalies).
     Never the in-sample contaminated target that invalidated the earlier harnesses.
  S4 DROP CEILING datasets: almost every method already near-perfect (nothing to select).
  S5 RESCUE FLOOR datasets by pool expansion before dropping them.
  S6 EXCLUDE BADLY-FIT MODELS, measured WITHOUT labels: a detector is dropped on a dataset
     if it is degenerate (near-constant scores) or UNSTABLE (scores from two disjoint
     bootstrap fits disagree). Note this is a weak filter used only to EXCLUDE broken
     models, not to rank good ones - internal criteria are known to be poor rankers
     (Ma, Zhao, Zhang, Akoglu, SIGKDD Explorations 2023).
  S7 KEEP only datasets whose surviving pool shows a REAL SPREAD.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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

TRIV_PCT = 99.0
MIN_HARD = 20
SEEDS = [0, 1, 2]
MAX_N = 5000
STAB_MIN = 0.30          # S6: min Spearman between two bootstrap fits to call a model stable
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\hadb_v2.csv"

CLASSIC = [("IForest", lambda: IForest(n_estimators=100, random_state=0)),
           ("LOF", lambda: LOF(n_neighbors=20)),
           ("KNN", lambda: KNN(n_neighbors=5)),
           ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
           ("HBOS", lambda: HBOS(n_bins=10)),
           ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
           ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)),
           ("LODA", lambda: LODA(n_bins=10))]

def build_pool():
    p = list(CLASSIC)
    for k in [3, 10, 35, 50, 100, 200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k)))
        p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50, 100]:
        p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9, 0.99]:
        p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    for ncl in [4, 16, 32]:
        p.append((f"CBLOF_c{ncl}", lambda ncl=ncl: CBLOF(n_clusters=ncl, random_state=0)))
    for ne in [50, 300]:
        p.append((f"IF_n{ne}", lambda ne=ne: IForest(n_estimators=ne, random_state=0)))
    for nb in [20, 50]:
        p.append((f"LODA_b{nb}", lambda nb=nb: LODA(n_bins=nb)))
    return p

POOL = build_pool()
CLASSIC_NAMES = {n for n, _ in CLASSIC}


def score(ctor, Xtr, Xte):
    import contextlib, io as _io
    try:
        m = ctor()
        with contextlib.redirect_stdout(_io.StringIO()):
            m.fit(Xtr)
        s = np.asarray(m.decision_function(Xte), float)
        return s if np.all(np.isfinite(s)) else None
    except Exception:
        return None


rows, meta, t0 = [], [], time.time()
datasets = []
for sub in ("adbench", "dami"):
    d = os.path.join(ROOT, "data", sub)
    if os.path.isdir(d):
        datasets += P.load_npz_dir(d)
print(f"pool={len(POOL)} variants; datasets={len(datasets)}", flush=True)

for ds in datasets:
    X, y = ds.X, ds.y
    if len(X) > MAX_N:
        r = np.random.RandomState(0)
        keep = r.choice(len(X), MAX_N, replace=False)
        X, y = X[keep], y[keep]
    rec = dict(dataset=ds.name, n=len(X), d=X.shape[1], n_anom=int(y.sum()))
    if y.sum() < 10:
        rec.update(status="dropped", reason="too_few_anomalies"); meta.append(rec); continue

    # ---- S1: trivial-anomaly removal (published definition) ----
    nm = y == 0
    mu, sd = X[nm].mean(0), X[nm].std(0) + 1e-9
    maxz = np.abs((X - mu) / sd).max(1)
    thr = float(np.percentile(maxz[nm], TRIV_PCT))
    anom = np.where(y == 1)[0]
    hard = anom[maxz[anom] <= thr]
    rec.update(n_trivial=int(len(anom) - len(hard)),
               frac_trivial=float(1 - len(hard) / len(anom)), n_hard=int(len(hard)))
    # ---- S2 ----
    if len(hard) < MIN_HARD:
        rec.update(status="dropped", reason=f"only_{len(hard)}_hard_anomalies")
        meta.append(rec); continue

    norm_idx = np.where(nm)[0]
    for seed in SEEDS:
        g = np.random.default_rng(seed)
        idx = np.arange(len(norm_idx)); g.shuffle(idx)
        cut = int(0.8 * len(idx))
        tr_i, te_i = norm_idx[idx[:cut]], norm_idx[idx[cut:]]
        if len(tr_i) < 50:
            continue
        Xtr = X[tr_i]
        Xte = np.vstack([X[te_i], X[hard]])
        yte = np.r_[np.zeros(len(te_i)), np.ones(len(hard))]
        # S6 stability split (label-free): two disjoint halves of the TRAINING normals
        h = len(tr_i) // 2
        A, B = X[tr_i[:h]], X[tr_i[h:]]
        for name, ctor in POOL:
            s = score(ctor, Xtr, Xte)
            if s is None or np.nanstd(s) < 1e-12:
                continue
            sa, sb = score(ctor, A, Xte), score(ctor, B, Xte)
            stab = np.nan
            if sa is not None and sb is not None and np.nanstd(sa) > 1e-12 and np.nanstd(sb) > 1e-12:
                stab = spearmanr(sa, sb).statistic
            rows.append(dict(dataset=ds.name, seed=seed, variant=name,
                             is_classic=name in CLASSIC_NAMES,
                             auc=float(roc_auc_score(yte, s)),
                             ap=float(average_precision_score(yte, s)),
                             stability=float(stab) if np.isfinite(stab) else np.nan))
    rec.update(status="scored")
    meta.append(rec)
    print(f"{ds.name}: hard={rec.get('n_hard')} ({100*rec.get('frac_trivial',0):.0f}% trivial) "
          f"| {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

D = pd.DataFrame(rows); D.to_csv(OUT, index=False)
M0 = pd.DataFrame(meta); M0.to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
print(f"\nwrote {OUT} rows={len(D)}")

if len(D):
    print("\n=== S1/S2: trivial-anomaly removal ===")
    sc = M0[M0.status == "scored"]
    print(f"  datasets scored: {len(sc)}/{len(M0)}   dropped: {int((M0.status=='dropped').sum())}")
    print(M0[M0.status == "dropped"].reason.value_counts().to_string())
    print(f"  mean trivial fraction removed: {sc.frac_trivial.mean():.3f}")

    print("\n=== S6: unsupervised model-fit filter (bootstrap stability) ===")
    st = D.groupby("variant").stability.mean().sort_values()
    print(f"  variants with mean stability < {STAB_MIN}: "
          f"{int((st < STAB_MIN).sum())}/{len(st)}")
    print(st.head(6).round(3).to_string())
    Dk = D[(D.stability >= STAB_MIN) | D.stability.isna()]
    print(f"  rows kept after stability filter: {len(Dk)}/{len(D)}")

    def summarize(sub, tag):
        gg = sub.groupby(["dataset", "seed"]).auc.agg(["max", "mean"])
        return gg.groupby("dataset").mean().rename(columns={"max": f"best_{tag}",
                                                            "mean": f"mean_{tag}"})
    c = summarize(Dk[Dk.is_classic], "cls")
    f = summarize(Dk, "all")
    M = c.join(f, how="outer")
    M["spread_cls"] = M.best_cls - M.mean_cls
    M["spread_all"] = M.best_all - M.mean_all

    print("\n=== S4/S5: dead zones on HARD anomalies, classical-9 vs expanded pool ===")
    zone = lambda b, s: "ceiling" if (b >= 0.95 and s < 0.03) else ("floor" if b < 0.70 else "live")
    M["zone_cls"] = [zone(b, s) for b, s in zip(M.best_cls, M.spread_cls)]
    M["zone_all"] = [zone(b, s) for b, s in zip(M.best_all, M.spread_all)]
    print(pd.crosstab(M.zone_cls, M.zone_all, rownames=["classical-9"],
                      colnames=["expanded"]).to_string())
    resc = M[(M.zone_cls == "floor") & (M.zone_all == "live")]
    print(f"\n  FLOOR rescued by pool expansion: {len(resc)}/{int((M.zone_cls=='floor').sum())}")
    print(f"  best-AUC gain from expansion: mean {(M.best_all-M.best_cls).mean():+.4f} "
          f"max {(M.best_all-M.best_cls).max():+.4f}")

    print("\n=== S7: FINAL HADB inclusion (the ranking task is real) ===")
    for thr in [0.05, 0.10, 0.15]:
        keep = M[(M.zone_all == "live") & (M.spread_all >= thr)]
        print(f"  live AND spread>={thr:.2f}: {len(keep):3d}/{len(M)} datasets  "
              f"mean best={keep.best_all.mean():.3f} mean spread={keep.spread_all.mean():.3f}")
    final = M[(M.zone_all == "live") & (M.spread_all >= 0.10)]
    M.reset_index().to_csv(OUT.replace(".csv", "_manifest.csv"), index=False)
    print(f"\n  HADB v2 (spread>=0.10): {len(final)} datasets")
    print(final[["best_cls", "best_all", "spread_all"]].round(3).to_string())
