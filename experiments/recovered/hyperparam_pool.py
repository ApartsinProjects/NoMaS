# -*- coding: utf-8 -*-
"""Can NoMaS select HYPERPARAMETERS, not just detector families?

Motivation: today's regret decomposition showed 60% of regret is WITHIN-family, which
routing/debiasing/edge-stratification structurally cannot reach. Hyperparameter selection
is within-family by construction. And the marginal-extremity problem that killed every
cross-family fix does NOT apply here: LOF(k=5) vs LOF(k=50) are both local detectors judged
on a local pseudo-task, so the task is FAIR for this comparison in a way it is not for
global-vs-local.

The paper's own Limitations: "combining NoMaS with per-detector hyperparameter search is
left to future work."

PRECONDITION (checked first, and the whole idea dies if it fails): does true AUC actually
VARY across hyperparameters? If every k gives the same AUC there is nothing to select.

Measured:
  spread_within   per detector, max-minus-min true AUC across its hyperparameter grid
  regret_within   NoMaS picking the hyperparameter WITHIN one detector, vs that detector's best
  regret_default  using the library default hyperparameter instead
  regret_pool     NoMaS over the FULL pool (all detectors x all hyperparameters)
  regret_pool_dflt  NoMaS over defaults only (the paper's current setting)
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
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

SEED, K, M = 0, 30, 12
MAX_N = 4000
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\hyperparam_pool.csv"

# (family, name, ctor, is_default)
def build_pool():
    P = []
    for k in [5, 10, 20, 35, 50, 100]:
        P.append(("LOF", f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k), k == 20))
        P.append(("KNN", f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k), k == 5))
    for ne in [50, 100, 200]:
        P.append(("IForest", f"IForest_n{ne}", lambda ne=ne: IForest(n_estimators=ne, random_state=SEED), ne == 100))
    for nb in [5, 10, 20, 50]:
        P.append(("HBOS", f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb), nb == 10))
    for nc in [0.3, 0.5, 0.9]:
        P.append(("PCA", f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=SEED), nc == 0.5))
    for ncl in [4, 8, 16]:
        P.append(("CBLOF", f"CBLOF_c{ncl}", lambda ncl=ncl: CBLOF(n_clusters=ncl, random_state=SEED), ncl == 8))
    return P


def score(ctor, Xtr, Xsc):
    import contextlib, io as _io
    try:
        m = ctor()
        with contextlib.redirect_stdout(_io.StringIO()):
            m.fit(Xtr)
        s = np.asarray(m.decision_function(Xsc), dtype=float)
        if not np.all(np.isfinite(s)) or np.nanstd(s) < 1e-12:
            return None
        return s
    except Exception:
        return None


SETS = [("dami", n) for n in ["Annthyroid", "Cardiotocography", "PageBlocks", "SpamBase",
                              "Stamps", "WDBC", "Waveform", "Wilt", "Pima"]] + \
       [("adbench", n) for n in ["20_letter", "22_magic.gamma", "19_landsat",
                                 "23_mammography", "6_cardio", "31_satimage-2"]]

POOL = build_pool()
print(f"candidate pool: {len(POOL)} variants across "
      f"{len(set(p[0] for p in POOL))} detectors", flush=True)

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
    if y.sum() < 10:
        continue

    # ---- TRUE auc per variant (labels used only here) ----
    true = {}
    for fml, vname, ctor, isdef in POOL:
        s = score(ctor, X, X)
        if s is not None:
            true[vname] = float(roc_auc_score(y, s))
    if len(true) < 10:
        print(f"{name}: too few variants", flush=True); continue

    # ---- NoMaS pseudo-AUC per variant (no labels) ----
    Xn = X[y == 0]
    Z = embed(Xn, dim=16)
    labc = cluster(Z, K, seed=SEED)
    acc = {v: [] for v in true}
    used = 0
    rng2 = np.random.RandomState(SEED)
    for j in np.unique(labc):
        if used >= M:
            break
        m = labc == j
        nj = int(m.sum())
        if nj < 15 or nj > 0.25 * len(Xn):
            continue
        rest = np.where(~m)[0]; rng2.shuffle(rest)
        nh = max(20, int(0.2 * len(rest)))
        hold, tr = rest[:nh], rest[nh:]
        Xsc = np.vstack([Xn[m], Xn[hold]])
        yt = np.r_[np.ones(nj), np.zeros(len(hold))]
        for fml, vname, ctor, isdef in POOL:
            if vname not in true:
                continue
            s = score(ctor, Xn[tr], Xsc)
            if s is not None:
                acc[vname].append(roc_auc_score(yt, s))
        used += 1
    pseudo = {v: float(np.mean(w)) for v, w in acc.items() if w}
    if used < 3 or len(pseudo) < 10:
        print(f"{name}: too few regimes ({used})", flush=True); continue

    meta = {v: (f, d) for f, v, c, d in POOL}
    for fml, vname, ctor, isdef in POOL:
        if vname in true and vname in pseudo:
            rows.append(dict(dataset=name, detector=fml, variant=vname, is_default=isdef,
                             true_auc=true[vname], pseudo_auc=pseudo[vname], n_regimes=used))
    print(f"{name}: variants={len(true)} regimes={used} | {time.time()-t0:.0f}s", flush=True)

D = pd.DataFrame(rows)
D.to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(D)} datasets={D.dataset.nunique()}")

# ---------------- PRECONDITION + RESULTS ----------------
if len(D):
    print("\n=== PRECONDITION: does true AUC vary across hyperparameters? ===")
    sw = D.groupby(["dataset", "detector"]).true_auc.agg(lambda s: s.max() - s.min())
    print(sw.groupby("detector").agg(["mean", "max"]).round(4).to_string())
    print(f"  mean within-detector AUC spread across hyperparameters: {sw.mean():.4f}")

    print("\n=== WITHIN-DETECTOR selection: can NoMaS pick the hyperparameter? ===")
    res = []
    for (ds, det), g in D.groupby(["dataset", "detector"]):
        best = g.true_auc.max()
        pick = g.loc[g.pseudo_auc.idxmax()]
        dflt = g[g.is_default]
        res.append(dict(dataset=ds, detector=det, spread=best - g.true_auc.min(),
                        regret_nomas=best - pick.true_auc,
                        regret_default=best - (dflt.true_auc.iloc[0] if len(dflt) else np.nan),
                        regret_random=best - g.true_auc.mean()))
    R = pd.DataFrame(res)
    print(R.groupby("detector")[["spread", "regret_nomas", "regret_default", "regret_random"]]
          .mean().round(4).to_string())
    print(f"\n  OVERALL  NoMaS={R.regret_nomas.mean():.4f}  "
          f"default={R.regret_default.mean():.4f}  random={R.regret_random.mean():.4f}")
    from scipy.stats import wilcoxon
    v = R.dropna(subset=["regret_default"])
    if len(v) and np.any(v.regret_nomas != v.regret_default):
        print(f"  NoMaS vs DEFAULT: p={wilcoxon(v.regret_nomas, v.regret_default).pvalue:.4f} "
              f"(wins={int((v.regret_default>v.regret_nomas).sum())} "
              f"losses={int((v.regret_default<v.regret_nomas).sum())})")

    print("\n=== FULL POOL vs DEFAULTS-ONLY (the paper's current setting) ===")
    out = []
    for ds, g in D.groupby("dataset"):
        best_all = g.true_auc.max()
        pool_pick = g.loc[g.pseudo_auc.idxmax()].true_auc
        gd = g[g.is_default]
        dflt_pick = gd.loc[gd.pseudo_auc.idxmax()].true_auc if len(gd) else np.nan
        out.append(dict(dataset=ds, regret_pool=best_all - pool_pick,
                        regret_defaults_only=best_all - dflt_pick,
                        best_is_default=bool(g.loc[g.true_auc.idxmax()].is_default)))
    O = pd.DataFrame(out)
    print(O.round(4).to_string(index=False))
    print(f"\n  regret with FULL POOL      : {O.regret_pool.mean():.4f}")
    print(f"  regret with DEFAULTS ONLY  : {O.regret_defaults_only.mean():.4f}")
    print(f"  datasets where the best variant IS a default: "
          f"{int(O.best_is_default.sum())}/{len(O)}")
