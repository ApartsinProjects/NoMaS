# -*- coding: utf-8 -*-
"""WHICH RANKING METRIC? Does the choice of target metric change which detector wins?

ROC-AUC is the paper's target, but under 1-5% anomaly rates it is dominated by the
high-FPR region nobody deploys in. Candidate replacements, and what each buys:

  roc_auc    threshold-free, base-rate independent, but NOT top-heavy      (current)
  pauc10     standardized partial AUC over FPR <= 0.10 (McClish)  <- top-heavy AND
  pauc01     standardized partial AUC over FPR <= 0.01                threshold-free AND
                                                                      comparable across sets
  ap         average precision: top-heavy, but scales with base rate, so it is NOT
             comparable across datasets without normalisation
  ap_norm    (AP - base_rate) / (1 - base_rate): AP made comparable
  p_at_n     precision within a budget of n = number of true anomalies (AD-literature
             standard; the most interpretable, but discretized)
  lift_n     p_at_n / base_rate: practical AND comparable across datasets

THE DECISIVE QUESTION is empirical, not philosophical: if every metric crowns the SAME
detector, the choice is moot and ROC-AUC can stay for continuity. If they disagree, the
metric choice silently changes the answer key for every experiment.

Reported: pairwise Spearman between metrics over the candidate pool, and how often the
ARGMAX detector differs.
"""
import os, sys, time, warnings, itertools
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

SEEDS = [0, 1, 2]
MAX_N = 5000
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\metric_choice.csv"

def build_pool():
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)),
         ("LOF", lambda: LOF(n_neighbors=20)), ("KNN", lambda: KNN(n_neighbors=5)),
         ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
         ("HBOS", lambda: HBOS(n_bins=10)),
         ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)),
         ("LODA", lambda: LODA(n_bins=10))]
    for k in [3, 10, 35, 50, 100]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k)))
        p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50]:
        p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.9]:
        p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    for ncl in [4, 16]:
        p.append((f"CBLOF_c{ncl}", lambda ncl=ncl: CBLOF(n_clusters=ncl, random_state=0)))
    return p

POOL = build_pool()
METRICS = ["roc_auc", "pauc10", "pauc01", "ap", "ap_norm", "p_at_n", "lift_n"]


def all_metrics(y, s):
    """Every candidate target metric from one score vector."""
    br = float(y.mean())
    n_a = int(y.sum())
    out = {"roc_auc": roc_auc_score(y, s), "ap": average_precision_score(y, s),
           "base_rate": br}
    # McClish-standardized partial AUC (sklearn's max_fpr does this standardization)
    for tag, f in (("pauc10", 0.10), ("pauc01", 0.01)):
        try:
            out[tag] = roc_auc_score(y, s, max_fpr=f)
        except Exception:
            out[tag] = np.nan
    out["ap_norm"] = (out["ap"] - br) / (1 - br) if br < 1 else np.nan
    # precision within a budget of n = number of true anomalies
    order = np.argsort(-s)
    out["p_at_n"] = float(y[order[:n_a]].mean()) if n_a else np.nan
    out["lift_n"] = out["p_at_n"] / br if br > 0 else np.nan
    return out


datasets = []
for sub in ("adbench", "dami"):
    d = os.path.join(ROOT, "data", sub)
    if os.path.isdir(d):
        datasets += P.load_npz_dir(d)
print(f"pool={len(POOL)} variants; datasets={len(datasets)}", flush=True)

rows, t0 = [], time.time()
for ds in datasets:
    X, y = np.asarray(ds.X, float), np.asarray(ds.y, int).ravel()
    _, uniq = np.unique(X, axis=0, return_index=True)
    uniq = np.sort(uniq); X, y = X[uniq], y[uniq]
    if len(X) > MAX_N:
        r = np.random.RandomState(0)
        keep = r.choice(len(X), MAX_N, replace=False); X, y = X[keep], y[keep]
    if y.sum() < 10:
        continue
    nm = y == 0
    norm_idx = np.where(nm)[0]
    for seed in SEEDS:
        g = np.random.default_rng(seed)
        idx = np.arange(len(norm_idx)); g.shuffle(idx)
        cut = int(0.8 * len(idx))
        tr_i, te_i = norm_idx[idx[:cut]], norm_idx[idx[cut:]]
        if len(tr_i) < 50:
            continue
        Xtr = X[tr_i]
        Xte = np.vstack([X[te_i], X[~nm]])
        yte = np.r_[np.zeros(len(te_i)), np.ones(int((~nm).sum()))]
        for name, ctor in POOL:
            import contextlib, io as _io
            try:
                m = ctor()
                with contextlib.redirect_stdout(_io.StringIO()):
                    m.fit(Xtr)
                s = np.asarray(m.decision_function(Xte), float)
                if not np.all(np.isfinite(s)) or np.nanstd(s) < 1e-12:
                    continue
            except Exception:
                continue
            rows.append(dict(dataset=ds.name, seed=seed, variant=name, **all_metrics(yte, s)))
    print(f"{ds.name}: done | {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

D = pd.DataFrame(rows); D.to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(D)} datasets={D.dataset.nunique()}")

if len(D):
    print(f"\nmedian base rate: {D.base_rate.median():.4f}")
    print("\n=== 1. Do the metrics RANK the pool the same? (mean per-dataset Spearman) ===")
    print(f"{'':>9s}" + "".join(f"{m:>9s}" for m in METRICS))
    for a in METRICS:
        line = f"{a:>9s}"
        for b in METRICS:
            if a == b:
                line += f"{'-':>9s}"; continue
            rs = []
            for (ds, sd), g in D.groupby(["dataset", "seed"]):
                if g[a].nunique() > 2 and g[b].nunique() > 2:
                    r = spearmanr(g[a], g[b]).statistic
                    if np.isfinite(r):
                        rs.append(r)
            line += f"{np.mean(rs):>9.3f}"
        print(line)

    print("\n=== 2. Do they crown the SAME detector? (% of dataset-seeds where argmax agrees) ===")
    print(f"{'':>9s}" + "".join(f"{m:>9s}" for m in METRICS))
    for a in METRICS:
        line = f"{a:>9s}"
        for b in METRICS:
            if a == b:
                line += f"{'-':>9s}"; continue
            agree = tot = 0
            for (ds, sd), g in D.groupby(["dataset", "seed"]):
                if g[a].notna().any() and g[b].notna().any():
                    tot += 1
                    agree += int(g.loc[g[a].idxmax(), "variant"] == g.loc[g[b].idxmax(), "variant"])
            line += f"{100*agree/max(tot,1):>8.0f}%"
        print(line)

    print("\n=== 3. What does choosing the WRONG metric cost? ===")
    print("   (pick the argmax under metric A, measure the regret it incurs under metric B)")
    for a in ["roc_auc", "pauc10", "ap"]:
        for b in ["pauc10", "p_at_n"]:
            if a == b:
                continue
            regs = []
            for (ds, sd), g in D.groupby(["dataset", "seed"]):
                if g[a].notna().any() and g[b].notna().any():
                    regs.append(g[b].max() - g.loc[g[a].idxmax(), b])
            print(f"   pick by {a:>8s} -> regret under {b:>8s}: {np.mean(regs):.4f}")
