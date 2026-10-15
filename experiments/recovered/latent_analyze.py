# -*- coding: utf-8 -*-
"""Analyze: does cluster separation predict regret, within dataset, across latents?

KILL CRITERION (pre-registered): mean within-dataset Spearman(separation, regret) must be
NEGATIVE and sign-consistent on >= 2/3 of datasets.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

CSV = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\latent_sep.csv"
D = pd.read_csv(CSV)
print(f"rows={len(D)} datasets={D.dataset.nunique()} latents={sorted(D.latent.unique())}")
print()
print(D.pivot_table(index="dataset", columns="latent", values="regret").round(4).to_string())
print()
print("silhouette by latent:")
print(D.pivot_table(index="dataset", columns="latent", values="silhouette").round(3).to_string())

for metric in ["silhouette", "calinski", "davies"]:
    print(f"\n=== {metric} vs regret, WITHIN dataset (across latents) ===")
    rr = []
    for ds, g in D.groupby("dataset"):
        if g[metric].nunique() >= 3 and g.regret.nunique() >= 2:
            rho, _ = spearmanr(g[metric], g.regret)
            if np.isfinite(rho):
                rr.append((ds, rho, len(g)))
    for ds, r, n in rr:
        print(f"  {ds:18s} rho={r:+.3f}  (n_latents={n})")
    if rr:
        v = np.array([r for _, r, _ in rr])
        neg = int((v < 0).sum())
        print(f"  MEAN rho = {v.mean():+.3f}   negative on {neg}/{len(v)} datasets")
        verdict = (v.mean() < 0) and (neg >= 2 * len(v) / 3)
        print(f"  KILL CRITERION: {'PASS - separation predicts regret' if verdict else 'FAIL - separation does NOT predict regret'}")

# Practical check: would picking the best-separated latent beat the paper's default pca16?
print("\n=== would choosing the best-SEPARATED latent (label-free) beat pca16? ===")
recs = []
for ds, g in D.groupby("dataset"):
    if "pca16" not in set(g.latent):
        continue
    dflt = g[g.latent == "pca16"].regret.iloc[0]
    bysil = g.loc[g.silhouette.idxmax()]
    oracle = g.regret.min()
    recs.append(dict(dataset=ds, pca16=dflt, best_sil=bysil.regret,
                     best_sil_latent=bysil.latent, oracle_latent=oracle))
R = pd.DataFrame(recs)
if len(R):
    print(R.round(4).to_string(index=False))
    print(f"\n  pca16 default        : {R.pca16.mean():.4f}")
    print(f"  pick-best-separation : {R.best_sil.mean():.4f}  "
          f"({100*(1-R.best_sil.mean()/max(R.pca16.mean(),1e-9)):+.1f}%)")
    print(f"  ORACLE best latent   : {R.oracle_latent.mean():.4f}  (upper bound on this whole line)")
    d = R.pca16 - R.best_sil
    if np.any(d != 0):
        print(f"  wins={int((d>0).sum())} losses={int((d<0).sum())} ties={int((d==0).sum())} "
              f"p={wilcoxon(R.pca16, R.best_sil).pvalue:.3f}")
