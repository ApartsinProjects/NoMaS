# -*- coding: utf-8 -*-
"""Replicate the within-family finding on the INDEPENDENT dev benchmark (69 datasets).

If the pseudo-score is informative inside the local family but not inside the
global family on OddBench, the same pattern should appear here. Independent data,
independent pipeline run.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = r"E:\Projects\Submitted\ADRank"
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}


def fam(d):
    return "local" if d in LOCAL else ("global" if d in GLOBAL else "other")


ps = pd.read_parquet(ROOT + r"\results\raw\modal_pseudo_allseeds.parquet")
tr = pd.read_parquet(ROOT + r"\results\raw\modal_true_allseeds.parquet")

# aggregate pseudo_auc the way the method does: mean over subsets, default K=30/smallest
ps = ps[(ps.K == 30) & (ps.selection == "smallest")]
agg = ps.groupby(["dataset", "seed", "detector"], as_index=False).pseudo_auc.mean()

m = agg.merge(tr[["dataset", "seed", "detector", "true_auc", "modality"]],
              on=["dataset", "seed", "detector"], how="inner").dropna(subset=["true_auc"])
print(f"merged rows={len(m)}  datasets={m.dataset.nunique()}  seeds={m.seed.nunique()}")

rows = []
for (ds, sd), g in m.groupby(["dataset", "seed"]):
    if len(g) < 5:
        continue
    g = g.copy(); g["fam"] = g.detector.map(fam)
    bd = g.loc[g.true_auc.idxmax(), "detector"]
    F = fam(bd)
    best = g.true_auc.max()
    pick_all = g.loc[g.pseudo_auc.idxmax(), "detector"]
    inf = g[g.fam == F]
    pick_inf = inf.loc[inf.pseudo_auc.idxmax(), "detector"]
    rho = spearmanr(inf.pseudo_auc, inf.true_auc).statistic if len(inf) >= 3 else np.nan
    rows.append(dict(dataset=ds, seed=sd, fam=F, modality=g.modality.iloc[0],
                     reg_now=best - g.set_index("detector").true_auc[pick_all],
                     reg_route=best - g.set_index("detector").true_auc[pick_inf],
                     rho_in_fam=rho))
R = pd.DataFrame(rows)
print(f"rows={len(R)}")

print("\n=== DEV BENCHMARK (69 datasets) ===")
print(f"current regret        : {R.reg_now.mean():.4f}")
print(f"oracle family routing : {R.reg_route.mean():.4f}")
tot = R.reg_now.mean()
print(f"  removable by routing: {tot - R.reg_route.mean():.4f} ({100*(tot-R.reg_route.mean())/tot:.0f}%)")
print(f"  within-family resid : {R.reg_route.mean():.4f} ({100*R.reg_route.mean()/tot:.0f}%)")

print("\ntrue-best family distribution:")
print(R.fam.value_counts(normalize=True).round(3).to_string())

print("\n=== REPLICATION: pseudo-score informativeness inside each family ===")
for f in ["local", "global", "other"]:
    s = R[(R.fam == f) & R.rho_in_fam.notna()]
    if len(s):
        print(f"  {f:6s} n={len(s):4d}  mean Spearman(pseudo_auc, true_auc) within family = {s.rho_in_fam.mean():+.3f}")
