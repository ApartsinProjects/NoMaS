# -*- coding: utf-8 -*-
"""IS THE TARGET RANKING FAIR? Two protocols compared.

BUG FOUND: every experiment today computed the ground-truth ranking as

    fit_and_score(d, X, X)       # fit on ALL data (contaminated), score the SAME data

i.e. IN-SAMPLE and CONTAMINATED. But the paper's own `true_rank_from_labels`
(src/adrank/pipeline.py:477) trains on 80% of the NORMALS ONLY and scores a HELD-OUT test
set of 20% normals + all anomalies. Clean and held-out.

WHY IT MATTERS MOST FOR HYPERPARAMETERS: in-sample bias is k-DEPENDENT. A training point
lies inside its own neighbourhood, and how badly that distorts LOF/KNN depends on k. The
NoMaS pseudo-task, by contrast, is properly HELD OUT. So the reported anti-correlation
(LOF -0.387, KNN -0.417) may be nothing but held-out-vs-in-sample, an artifact of the
harness rather than a property of the method.

This script recomputes the SAME hyperparameter pool under BOTH targets and reports:
  - does the truly-best k change between protocols?
  - does the anti-correlation with the pseudo-task survive the CLEAN target?

If the anti-correlation vanishes under the paper's protocol, the hyperparameter conclusion
must be retracted and the extent-mismatch mechanism with it.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import embed, cluster  # noqa: E402

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA as PPCA
from pyod.models.cblof import CBLOF

SEED, K, M, MAX_N = 0, 30, 12, 4000
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\target_protocol.csv"

POOL = []
for k in [5, 10, 20, 35, 50, 100]:
    POOL.append(("LOF", f"LOF_k{k}", k, lambda k=k: LOF(n_neighbors=k)))
    POOL.append(("KNN", f"KNN_k{k}", k, lambda k=k: KNN(n_neighbors=k)))
for nb in [5, 10, 20, 50]:
    POOL.append(("HBOS", f"HBOS_b{nb}", nb, lambda nb=nb: HBOS(n_bins=nb)))
for ne in [50, 100, 200]:
    POOL.append(("IForest", f"IF_n{ne}", ne, lambda ne=ne: IForest(n_estimators=ne, random_state=SEED)))
for nc in [0.3, 0.5, 0.9]:
    POOL.append(("PCA", f"PCA_c{nc}", nc, lambda nc=nc: PPCA(n_components=nc, random_state=SEED)))
for ncl in [4, 8, 16]:
    POOL.append(("CBLOF", f"CBLOF_c{ncl}", ncl, lambda ncl=ncl: CBLOF(n_clusters=ncl, random_state=SEED)))

SETS = [("dami", n) for n in ["Annthyroid", "Cardiotocography", "PageBlocks", "SpamBase",
                              "Stamps", "WDBC", "Waveform", "Wilt", "Pima"]] + \
       [("adbench", n) for n in ["20_letter", "22_magic.gamma", "19_landsat",
                                 "23_mammography", "6_cardio", "31_satimage-2"]]


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

    # ---- TARGET A: what I did today - in-sample, contaminated ----
    tA = {}
    for fml, vn, hp, ctor in POOL:
        s = sc(ctor, X, X)
        if s is not None:
            tA[vn] = float(roc_auc_score(y, s))

    # ---- TARGET B: the PAPER's protocol - train on 80% normals, score held-out ----
    g = np.random.default_rng(SEED)
    nmask = y == 0
    Xnorm = X[nmask]
    idx = np.arange(len(Xnorm)); g.shuffle(idx)
    cut = int(0.8 * len(idx))
    Xtr_t = Xnorm[idx[:cut]]
    Xte = np.vstack([Xnorm[idx[cut:]], X[~nmask]])
    yte = np.r_[np.zeros(len(idx) - cut), np.ones(int((~nmask).sum()))]
    tB = {}
    for fml, vn, hp, ctor in POOL:
        s = sc(ctor, Xtr_t, Xte)
        if s is not None:
            tB[vn] = float(roc_auc_score(yte, s))

    # ---- NoMaS pseudo-AUC (unchanged, properly held out, label-free) ----
    Xn = X[y == 0]
    Z = embed(Xn, dim=16)
    labc = cluster(Z, K, seed=SEED)
    acc = {vn: [] for _, vn, _, _ in POOL}
    used = 0
    r2 = np.random.RandomState(SEED)
    for j in np.unique(labc):
        if used >= M:
            break
        m = labc == j
        nj = int(m.sum())
        if nj < 15 or nj > 0.25 * len(Xn):
            continue
        rest = np.where(~m)[0]; r2.shuffle(rest)
        nh = max(20, int(0.2 * len(rest)))
        hold, tr = rest[:nh], rest[nh:]
        Xsc = np.vstack([Xn[m], Xn[hold]])
        yt = np.r_[np.ones(nj), np.zeros(len(hold))]
        for fml, vn, hp, ctor in POOL:
            s = sc(ctor, Xn[tr], Xsc)
            if s is not None:
                acc[vn].append(roc_auc_score(yt, s))
        used += 1
    pse = {v: float(np.mean(w)) for v, w in acc.items() if w}
    if used < 3:
        continue
    meta = {vn: (fml, hp) for fml, vn, hp, _ in POOL}
    for vn in pse:
        if vn in tA and vn in tB:
            fml, hp = meta[vn]
            rows.append(dict(dataset=name, detector=fml, variant=vn, hp=hp,
                             true_insample=tA[vn], true_heldout=tB[vn],
                             pseudo_auc=pse[vn], n_regimes=used))
    print(f"{name}: variants={len(pse)} regimes={used} | {time.time()-t0:.0f}s", flush=True)

D = pd.DataFrame(rows)
D.to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(D)} datasets={D.dataset.nunique()}")

if len(D):
    print("\n=== DO THE TWO TARGETS EVEN AGREE? ===")
    for det, g in D.groupby("detector"):
        rs = []
        for ds, gg in g.groupby("dataset"):
            if gg.true_insample.nunique() > 2:
                r = spearmanr(gg.true_insample, gg.true_heldout).statistic
                if np.isfinite(r):
                    rs.append(r)
        print(f"  {det:8s} Spearman(in-sample target, held-out target) = {np.mean(rs):+.3f}")

    print("\n=== THE DECISIVE TEST: pseudo vs each target ===")
    print(f"{'detector':>8s} {'rho vs IN-SAMPLE':>18s} {'rho vs HELD-OUT':>17s}")
    for det, g in D.groupby("detector"):
        ra, rb = [], []
        for ds, gg in g.groupby("dataset"):
            if gg.true_insample.nunique() > 2:
                x = spearmanr(gg.pseudo_auc, gg.true_insample).statistic
                if np.isfinite(x):
                    ra.append(x)
            if gg.true_heldout.nunique() > 2:
                x = spearmanr(gg.pseudo_auc, gg.true_heldout).statistic
                if np.isfinite(x):
                    rb.append(x)
        print(f"  {det:>8s} {np.mean(ra):+18.3f} {np.mean(rb):+17.3f}")

    print("\n=== which k is truly best under each protocol? (LOF/KNN) ===")
    for det in ["LOF", "KNN"]:
        g = D[D.detector == det]
        a = g.groupby("dataset").apply(lambda s: s.loc[s.true_insample.idxmax(), "hp"], include_groups=False)
        b = g.groupby("dataset").apply(lambda s: s.loc[s.true_heldout.idxmax(), "hp"], include_groups=False)
        c = g.groupby("dataset").apply(lambda s: s.loc[s.pseudo_auc.idxmax(), "hp"], include_groups=False)
        print(f"  {det}: median k  in-sample={a.median():6.1f}  held-out={b.median():6.1f}  "
              f"pseudo-pick={c.median():6.1f}")
