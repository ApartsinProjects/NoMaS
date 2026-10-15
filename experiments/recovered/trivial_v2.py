# -*- coding: utf-8 -*-
"""Easy/Difficult anomaly stratification using the PUBLISHED definition.

Adopts the trivial-detector split from Apartsin & Aperstein, "Modeling Normal Is All You
Need", arXiv:2607.06094, section 5.3:

  "A trivial detector scores each window by the maximum absolute standardized per-channel
   window mean, max_ch |z| ... An anomaly window is labeled easy if this trivial score
   exceeds the 99th percentile of train-normal, and difficult otherwise."

This fixes a flaw in the first pass, which used a FIXED max|z| >= 3. The maximum over d
features exceeds 3 by chance in high dimensions, so the fixed cutoff inflates the trivial
fraction with dimensionality (measured: 0.32 at d<=10 rising to 0.65 at d>50). The
percentile-of-normal threshold adapts to d automatically.

Protocol follows the paper: STRATIFY, do not filter. For every subset the positive class is
that subset's anomalies and the negative class is the FULL set of normals.

Also adds the paper's second, stricter criterion in simplified form:
  LinRes - predict each feature linearly from the others; a point is "linearly separable"
  if its max standardized residual exceeds the 99th percentile of train-normal.
  DOUBLE-HARD = anomaly caught by NEITHER max|z| NOR LinRes.

Thresholds are swept (95/99/99.5 pct) so the conclusion's robustness is visible.
"""
import os, sys, glob, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import fit_and_score  # noqa: E402

DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
fam = lambda d: "local" if d in LOCAL else ("global" if d in GLOBAL else "other")
MAX_N = 6000
PCTS = [95.0, 99.0, 99.5]
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\trivial_v2.csv"

SETS = []
for sub in ("dami", "adbench"):
    for p in sorted(glob.glob(os.path.join(ROOT, "data", sub, "*.npz"))):
        SETS.append((sub, os.path.splitext(os.path.basename(p))[0]))


def linres_score(X, norm_mask):
    """Max standardized residual when each feature is predicted from the others.
    Fitted on NORMAL points only."""
    n, d = X.shape
    if d < 2 or d > 200:
        return None
    R = np.zeros((n, d))
    Xn = X[norm_mask]
    for k in range(d):
        oth = [i for i in range(d) if i != k]
        try:
            m = Ridge(alpha=1.0).fit(Xn[:, oth], Xn[:, k])
            pred = m.predict(X[:, oth])
            res = X[:, k] - pred
            s = res[norm_mask].std() + 1e-9
            R[:, k] = np.abs(res) / s
        except Exception:
            R[:, k] = 0.0
    return R.max(1)


rows, t0 = [], time.time()
for sub, name in SETS:
    a = np.load(os.path.join(ROOT, "data", sub, name + ".npz"))
    X, y = np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()
    rng = np.random.RandomState(0)
    if len(X) > MAX_N:
        keep = rng.choice(len(X), MAX_N, replace=False)
        X, y = X[keep], y[keep]
    if len(X) < 200 or y.sum() < 20:
        continue

    S = {}
    for d_ in DETS:
        s = fit_and_score(d_, X, X)
        if s is not None:
            S[d_] = s
    if len(S) < 6:
        continue

    nm = y == 0
    Xn = X[nm]
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-9
    maxz = np.abs((X - mu) / sd).max(1)
    lin = linres_score(X, nm)
    anom = np.where(y == 1)[0]
    norm_idx = np.where(nm)[0]

    def auc_on(idx):
        yy = np.r_[np.ones(len(idx)), np.zeros(len(norm_idx))]
        return {k: roc_auc_score(yy, np.r_[v[idx], v[norm_idx]]) for k, v in S.items()}

    all_auc = auc_on(anom)
    ba = max(all_auc, key=all_auc.get)

    for pct in PCTS:
        thr = np.percentile(maxz[nm], pct)
        easy = maxz[anom] > thr
        hard = anom[~easy]
        rec = dict(dataset=name, n=len(X), d=X.shape[1], pct=pct,
                   n_anom=len(anom), n_hard=int(len(hard)),
                   frac_trivial=float(easy.mean()),
                   best_all=ba, fam_all=fam(ba), auc_best_all=all_auc[ba],
                   mean_auc_all=float(np.mean(list(all_auc.values()))),
                   spread_all=all_auc[ba] - float(np.mean(list(all_auc.values()))))
        if len(hard) >= 10:
            h = auc_on(hard)
            bh = max(h, key=h.get)
            rec.update(best_hard=bh, fam_hard=fam(bh), auc_best_hard=h[bh],
                       mean_auc_hard=float(np.mean(list(h.values()))),
                       spread_hard=h[bh] - float(np.mean(list(h.values()))))
        # double-hard: caught by NEITHER criterion
        if lin is not None:
            lthr = np.percentile(lin[nm], pct)
            dh = anom[(~easy) & (lin[anom] <= lthr)]
            rec["n_doublehard"] = int(len(dh))
            if len(dh) >= 10:
                h2 = auc_on(dh)
                b2 = max(h2, key=h2.get)
                rec.update(best_dh=b2, fam_dh=fam(b2), auc_best_dh=h2[b2],
                           mean_auc_dh=float(np.mean(list(h2.values()))))
        rows.append(rec)
    print(f"{name}: d={X.shape[1]} anom={len(anom)} best_all={ba}({fam(ba)}) "
          f"| {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)} elapsed={time.time()-t0:.0f}s")
