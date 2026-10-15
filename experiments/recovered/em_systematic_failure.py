# -*- coding: utf-8 -*-
"""Does EM SYSTEMATICALLY mis-rank a detector CLASS on a dataset CLASS, where the dataset class
is measured INDEPENDENT of any detector (normals-only)?

Per (dataset, detector):  em_error = pct_rank(EM) - pct_rank(true ap_norm)  among the dataset's pool.
  em_error > 0  => EM OVER-rates this detector (thinks it better than it is)
Aggregate into cells: detector family (local/global/other) x dataset class (by an independent
normals-only metric: multimodality m_sil, dimensionality d). One-sample t-test per cell vs 0.
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata, ttest_1samp, spearmanr
warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import true_apnorm
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}
mp = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv"))  # has independent metrics m_sil, m_gmm, d
rows = []
per_ds = []
for _, r in mp.iterrows():
    try: ap, emv = true_apnorm(CSVMAP[r.corpus], r.dataset)
    except Exception: continue
    common = [v for v in ap.index if v in emv.index and ap[v] == ap[v] and emv[v] == emv[v]]
    if len(common) < 6: continue
    a = np.array([ap[v] for v in common]); e = np.array([emv[v] for v in common])
    tp = rankdata(a) / len(a); ep = rankdata(e) / len(e)     # percentile ranks
    err = ep - tp
    # within-dataset EM-vs-truth ordering quality
    rho = spearmanr(e, a).statistic
    per_ds.append({"dataset": r.dataset, "m_sil": r.m_sil, "m_gmm": r.m_gmm, "d": r.d, "rho_em": rho})
    for v, x in zip(common, err):
        rows.append({"dataset": r.dataset, "det": v, "fam": fam(v), "em_err": x,
                     "m_sil": r.m_sil, "m_gmm": r.m_gmm, "d": r.d})
df = pd.DataFrame(rows); pds = pd.DataFrame(per_ds)
print(f"=== EM systematic mis-ranking  ({df.dataset.nunique()} datasets, {len(df)} detector-observations) ===")
print("em_err = pct_rank(EM) - pct_rank(true ap_norm);  >0 = EM OVER-rates the detector\n")
print("(1) By detector family, marginal (all datasets):")
for f, g in df.groupby("fam"):
    t = ttest_1samp(g.em_err, 0); print(f"    {f:7s} mean_em_err={g.em_err.mean():+.3f}  n={len(g):4d}  t-test vs0 p={t.pvalue:.1e}")

for metric, lo_name, hi_name in [("m_sil", "unimodal", "multimodal"), ("d", "low-dim", "high-dim")]:
    thr = df[metric].median(); df["cls"] = np.where(df[metric] >= thr, hi_name, lo_name)
    print(f"\n(2) detector-family x dataset-class by INDEPENDENT metric '{metric}' (split at median {thr:.3g}):")
    print(f"    {'family':7s} {lo_name:>12s}            {hi_name:>12s}")
    for f in ["local", "global", "other"]:
        cells = []
        for cls in [lo_name, hi_name]:
            g = df[(df.fam == f) & (df.cls == cls)]
            if len(g) >= 8:
                t = ttest_1samp(g.em_err, 0); cells.append(f"{g.em_err.mean():+.3f} (n={len(g):3d},p={t.pvalue:.0e})")
            else: cells.append(f"(n={len(g)})")
        print(f"    {f:7s} {cells[0]:>22s}  {cells[1]:>22s}")

print(f"\n(3) does EM's within-dataset ordering quality (rho_em) depend on independent metrics?")
for m in ["m_sil", "m_gmm", "d"]:
    ok = pds[m].notna() & pds.rho_em.notna()
    print(f"    Spearman(rho_em, {m}) = {spearmanr(pds[m][ok], pds.rho_em[ok]).statistic:+.3f}   (mean rho_em={pds.rho_em.mean():+.3f})")
df.to_csv(os.path.join(S, "EM_SYSTEMATIC.csv"), index=False)
print("\nsaved EM_SYSTEMATIC.csv")
