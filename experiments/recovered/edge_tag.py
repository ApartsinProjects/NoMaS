# -*- coding: utf-8 -*-
"""EDGE-vs-INTERIOR cluster tagging (smoke test).

HYPOTHESIS (stated before running):
  A held-out cluster that sits at the EDGE of the normal distribution resembles a
  GLOBAL anomaly, so global detectors (HBOS/PCA/COPOD/ECOD) can win that pseudo-task
  legitimately. A held-out cluster in the INTERIOR is a purely LOCAL puzzle, so only
  local detectors (LOF/KNN/CBLOF) can win it. Therefore:
    H1 (mechanism) the global-minus-local pseudo-AUC gap RISES with the cluster's
       edge score.
    H2 (payoff)    on datasets whose TRUE best detector is global, regret computed
       from EDGE clusters is lower than from INTERIOR clusters.

KILL: if H1's correlation is ~0 (|rho| < 0.15), the tag carries no signal, stop.

Everything used to build the tag comes from X only. Labels are used solely to
compute true AUC for evaluation, never inside the pseudo-task.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import embed, cluster, fit_and_score  # noqa: E402

LOCAL = ["LOF", "KNN", "CBLOF"]
GLOBAL = ["HBOS", "COPOD", "ECOD", "PCA"]
DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
SEED = 0
K = 30
DATA = os.path.join(ROOT, "data", "dami")
MAX_DS = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def edge_scores(Z, labels, j):
    """LABEL-FREE measures of how peripheral cluster j is."""
    m = labels == j
    C, R = Z[m], Z[~m]
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    # 1. standardized distance of cluster centroid from the global centroid
    cen = np.linalg.norm((C.mean(0) - mu) / sd) / np.sqrt(Z.shape[1])
    # 2. fraction of cluster points beyond the 90th pct of global centroid-distance
    dist = np.linalg.norm((Z - mu) / sd, axis=1)
    thr = np.percentile(dist, 90)
    frac = float((np.linalg.norm((C - mu) / sd, axis=1) > thr).mean())
    # 3. mean marginal percentile displacement from the median (per-feature extremeness)
    pct = np.array([(Z[:, k] < C[:, k].mean()).mean() for k in range(Z.shape[1])])
    marg = float(np.mean(np.abs(pct - 0.5)) * 2)
    # 4. global-PCA reconstruction error of the cluster relative to the rest
    from sklearn.decomposition import PCA
    q = max(1, min(Z.shape[1] - 1, Z.shape[1] // 2))
    p = PCA(n_components=q, random_state=SEED).fit(R)
    err_c = np.mean(np.sum((C - p.inverse_transform(p.transform(C))) ** 2, axis=1))
    err_r = np.mean(np.sum((R - p.inverse_transform(p.transform(R))) ** 2, axis=1))
    rec = float(err_c / (err_r + 1e-12))
    return dict(edge_centroid=cen, edge_frac_outer=frac, edge_marginal=marg, edge_recon=rec)


rows = []
files = sorted(f for f in os.listdir(DATA) if f.endswith(".npz"))[:MAX_DS]
for fn in files:
    ds = fn[:-4]
    a = np.load(os.path.join(DATA, fn))
    X, y = np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()
    if X.shape[0] < 200 or X.shape[0] > 20000:
        print(f"skip {ds} (n={X.shape[0]})"); continue
    # ---- TRUE auc per detector (labels used ONLY here) ----
    true = {}
    for d in DETS:
        s = fit_and_score(d, X, X)
        if s is not None:
            true[d] = roc_auc_score(y, s)
    if len(true) < 6:
        print(f"skip {ds} (detectors failed)"); continue
    best_det = max(true, key=true.get)
    best_fam = "local" if best_det in LOCAL else ("global" if best_det in GLOBAL else "other")

    # ---- pseudo-task per held-out cluster (labels NEVER used) ----
    Xn = X[y == 0]                      # normal-only pool, as the method assumes
    Z = embed(Xn, dim=16)
    lab = cluster(Z, K, seed=SEED)
    rng = np.random.RandomState(SEED)
    for j in np.unique(lab):
        m = lab == j
        nj = int(m.sum())
        if nj < 15 or nj > 0.25 * len(Xn):
            continue
        rest = np.where(~m)[0]
        rng.shuffle(rest)
        n_hold = max(20, int(0.2 * len(rest)))
        hold, tr = rest[:n_hold], rest[n_hold:]
        Xtr = Xn[tr]
        Xsc = np.vstack([Xn[m], Xn[hold]])
        ytask = np.r_[np.ones(nj), np.zeros(len(hold))]
        pa = {}
        for d in DETS:
            s = fit_and_score(d, Xtr, Xsc)
            if s is not None:
                pa[d] = roc_auc_score(ytask, s)
        if len(pa) < 6:
            continue
        e = edge_scores(Z, lab, j)
        z = np.array([pa[d] for d in pa]); z = (z - z.mean()) / (z.std() + 1e-9)
        zz = dict(zip(pa, z))
        gap = np.mean([zz[d] for d in GLOBAL if d in zz]) - np.mean([zz[d] for d in LOCAL if d in zz])
        pick = max(pa, key=pa.get)
        rows.append(dict(dataset=ds, cluster=int(j), size=nj, best_fam=best_fam,
                         best_det=best_det, gap_global_minus_local=gap,
                         regret=max(true.values()) - true.get(pick, np.nan), pick=pick,
                         **e))
    print(f"{ds}: best={best_det}({best_fam})  clusters used={sum(1 for r in rows if r['dataset']==ds)}")

R = pd.DataFrame(rows)
out = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\edge_rows.csv"
R.to_csv(out, index=False)
print(f"\nclusters total: {len(R)}  datasets: {R.dataset.nunique()}")

print("\n=== H1 MECHANISM: does the edge tag predict the global-vs-local pseudo gap? ===")
for c in ["edge_centroid", "edge_frac_outer", "edge_marginal", "edge_recon"]:
    rho, p = spearmanr(R[c], R.gap_global_minus_local)
    flag = "  <-- signal" if abs(rho) >= 0.15 else ""
    print(f"  {c:16s} Spearman rho = {rho:+.3f}  (p={p:.1e}){flag}")

print("\n=== H2 PAYOFF: regret from EDGE vs INTERIOR clusters ===")
for c in ["edge_recon", "edge_centroid"]:
    hi = R[R[c] >= R[c].quantile(0.67)]; lo = R[R[c] <= R[c].quantile(0.33)]
    print(f"  by {c}: edge regret={hi.regret.mean():.4f} (n={len(hi)})  "
          f"interior regret={lo.regret.mean():.4f} (n={len(lo)})  delta={lo.regret.mean()-hi.regret.mean():+.4f}")
    for f in ["global", "local"]:
        h2, l2 = hi[hi.best_fam == f], lo[lo.best_fam == f]
        if len(h2) and len(l2):
            print(f"      best={f:6s}: edge={h2.regret.mean():.4f} (n={len(h2)})  "
                  f"interior={l2.regret.mean():.4f} (n={len(l2)})  delta={l2.regret.mean()-h2.regret.mean():+.4f}")
print(f"\nwrote {out}")
