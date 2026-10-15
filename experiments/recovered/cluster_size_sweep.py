# -*- coding: utf-8 -*-
"""Does SHRINKING the held-out cluster fix the hyperparameter anti-correlation?

MECHANISM (measured today): the pseudo-task prefers LOF/KNN k=5 while the truth prefers
k=100, because holding out a whole cluster creates a large CONTIGUOUS VOID that small-k
methods detect sharply, whereas real anomalies are ISOLATED POINTS that large-k smoothing
serves better. The failure is an EXTENT MISMATCH between pseudo-anomaly and real anomaly.

PREDICTION (stated in advance): as K increases the excised clusters get smaller and closer
in extent to an isolated anomaly, so
  (1) the pseudo-task's preferred k should RISE toward the true optimum,
  (2) Spearman(pseudo_auc, true_auc) across the k-grid should become LESS NEGATIVE and
      possibly positive,
  (3) hyperparameter-selection regret should FALL.
If none of these move with K, the extent explanation is wrong and cluster size is not the
lever.

KILL: if Spearman stays at about -0.4 across the whole K range, drop this line.

Supporting note: the paper's own ablation (Table A.1) already found `smallest`-cluster
selection beats random and the hand-designed composite, which is consistent with smaller
being better; nobody pushed on how far.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import embed  # noqa: E402
from sklearn.cluster import MiniBatchKMeans

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS

SEED = 0
MAX_N = 4000
KS = [30, 60, 120, 250, 500]          # more clusters => smaller held-out clusters
M = 12                                 # regimes per K
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\cluster_size_sweep.csv"

POOL = []
for k in [5, 10, 20, 35, 50, 100]:
    POOL.append(("LOF", f"LOF_k{k}", k, lambda k=k: LOF(n_neighbors=k)))
    POOL.append(("KNN", f"KNN_k{k}", k, lambda k=k: KNN(n_neighbors=k)))
for nb in [5, 10, 20, 50]:
    POOL.append(("HBOS", f"HBOS_b{nb}", nb, lambda nb=nb: HBOS(n_bins=nb)))
for ne in [50, 100, 200]:
    POOL.append(("IForest", f"IF_n{ne}", ne, lambda ne=ne: IForest(n_estimators=ne, random_state=SEED)))

SETS = [("dami", n) for n in ["Annthyroid", "Cardiotocography", "PageBlocks", "SpamBase",
                              "Stamps", "Waveform", "Wilt", "Pima"]] + \
       [("adbench", n) for n in ["20_letter", "22_magic.gamma", "19_landsat", "6_cardio"]]


def sc(ctor, Xtr, Xsc):
    import contextlib, io as _io
    try:
        m = ctor()
        with contextlib.redirect_stdout(_io.StringIO()):
            m.fit(Xtr)
        s = np.asarray(m.decision_function(Xsc), float)
        return s if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12 else None
    except Exception:
        return None


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
    if y.sum() < 10:
        continue

    true = {}
    for fml, vn, hp, ctor in POOL:
        s = sc(ctor, X, X)
        if s is not None:
            true[vn] = float(roc_auc_score(y, s))
    if len(true) < 10:
        continue

    Xn = X[y == 0]
    Z = embed(Xn, dim=16)
    for K in KS:
        Keff = min(K, max(2, len(Z) // 8))
        labc = MiniBatchKMeans(n_clusters=Keff, random_state=SEED, n_init=5,
                               batch_size=256).fit_predict(Z)
        acc = {v: [] for v in true}
        sizes, used = [], 0
        rng2 = np.random.RandomState(SEED)
        for j in np.unique(labc):
            if used >= M:
                break
            m = labc == j
            nj = int(m.sum())
            if nj < 5 or nj > 0.25 * len(Xn):        # floor lowered to 5 so small clusters qualify
                continue
            rest = np.where(~m)[0]; rng2.shuffle(rest)
            nh = max(20, int(0.2 * len(rest)))
            hold, tr = rest[:nh], rest[nh:]
            Xsc = np.vstack([Xn[m], Xn[hold]])
            yt = np.r_[np.ones(nj), np.zeros(len(hold))]
            for fml, vn, hp, ctor in POOL:
                if vn not in true:
                    continue
                s = sc(ctor, Xn[tr], Xsc)
                if s is not None:
                    acc[vn].append(roc_auc_score(yt, s))
            sizes.append(nj); used += 1
        pseudo = {v: float(np.mean(w)) for v, w in acc.items() if w}
        if used < 3 or len(pseudo) < 10:
            continue
        meta = {vn: (fml, hp) for fml, vn, hp, _ in POOL}
        for vn in pseudo:
            fml, hp = meta[vn]
            rows.append(dict(dataset=name, K=K, Keff=Keff, mean_cluster_size=float(np.mean(sizes)),
                             n_regimes=used, detector=fml, variant=vn, hp=hp,
                             true_auc=true[vn], pseudo_auc=pseudo[vn]))
        print(f"{name} K={K}: Keff={Keff} regimes={used} mean_size={np.mean(sizes):.1f} "
              f"| {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

D = pd.DataFrame(rows)
D.to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(D)}")

if len(D):
    print("\n=== PREDICTION TEST: does shrinking clusters fix the anti-correlation? ===")
    print(f"{'K':>5s} {'mean size':>10s} {'LOF rho':>9s} {'KNN rho':>9s} "
          f"{'LOF k_pick':>11s} {'LOF k_true':>11s} {'regret':>8s}")
    for K, g in D.groupby("K"):
        out = {}
        for det in ["LOF", "KNN"]:
            rs = []
            for ds, gg in g[g.detector == det].groupby("dataset"):
                if gg.true_auc.nunique() > 2:
                    r = spearmanr(gg.pseudo_auc, gg.true_auc).statistic
                    if np.isfinite(r):
                        rs.append(r)
            out[det] = np.mean(rs) if rs else np.nan
        lof = g[g.detector == "LOF"]
        kp = lof.groupby("dataset").apply(lambda s: s.loc[s.pseudo_auc.idxmax(), "hp"], include_groups=False)
        kt = lof.groupby("dataset").apply(lambda s: s.loc[s.true_auc.idxmax(), "hp"], include_groups=False)
        reg = []
        for (ds, det), gg in g.groupby(["dataset", "detector"]):
            reg.append(gg.true_auc.max() - gg.loc[gg.pseudo_auc.idxmax(), "true_auc"])
        print(f"{K:5d} {g.mean_cluster_size.mean():10.1f} {out['LOF']:+9.3f} {out['KNN']:+9.3f} "
              f"{kp.median():11.1f} {kt.median():11.1f} {np.mean(reg):8.4f}")
