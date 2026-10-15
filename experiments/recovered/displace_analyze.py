# -*- coding: utf-8 -*-
"""Analyze the displaced-cluster delta sweep.

Reports, per delta:
  (a) SATURATION   mean pseudo-AUC across detectors (near 1.0 = task too easy, no signal)
  (b) GAP CLOSURE  global-minus-local mean z-scored pseudo-AUC (0 = no family handicap)
  (c) PAYOFF       regret when ranking detectors by mean pseudo-AUC at that delta
And the SPECTRUM variant: average pseudo-AUC over ALL deltas, so no anomaly type is
structurally favoured (plain averaging, NOT variance weighting - variance weighting is
what broke the earlier global-tail regime).

Pre-registered kill criterion: some delta must close the gap materially WITHOUT saturating.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CSV = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\displace.csv"
D = pd.read_csv(CSV)
print(f"rows={len(D)} datasets={D.dataset.nunique()} deltas={sorted(D.delta.unique())}")

# z-score within (dataset, cluster, delta) so families are comparable
D["z"] = D.groupby(["dataset", "cluster", "delta"]).pseudo_auc.transform(
    lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))

print("\n=== (a) SATURATION and (b) GAP CLOSURE by delta ===")
print(f"{'delta':>6s} {'mean pAUC':>10s} {'frac>0.95':>10s} {'global-local z gap':>20s}")
for dl, g in D.groupby("delta"):
    gap = g[g.fam == "global"].z.mean() - g[g.fam == "local"].z.mean()
    print(f"{dl:6.1f} {g.pseudo_auc.mean():10.3f} {(g.pseudo_auc > 0.95).mean():10.3f} {gap:20.3f}")

# (c) payoff: regret per delta, and the spectrum aggregate
def regret_from(scores, true):
    pick = max(scores, key=scores.get)
    return max(true.values()) - true[pick]

recs = []
for ds, g in D.groupby("dataset"):
    true = g.groupby("detector").true_auc.first().to_dict()
    row = {"dataset": ds}
    for dl, gd in g.groupby("delta"):
        sc = gd.groupby("detector").pseudo_auc.mean().to_dict()
        row[f"d{dl}"] = regret_from(sc, true)
    row["spectrum"] = regret_from(g.groupby("detector").pseudo_auc.mean().to_dict(), true)
    bd = max(true, key=true.get)
    row["best_fam"] = ("local" if bd in {"LOF", "KNN", "CBLOF"}
                       else ("global" if bd in {"HBOS", "COPOD", "ECOD", "PCA"} else "other"))
    recs.append(row)
R = pd.DataFrame(recs)
cols = [c for c in R.columns if c.startswith("d") and c != "dataset"] + ["spectrum"]
print("\n=== (c) PAYOFF: regret per dataset ===")
print(R[["dataset", "best_fam"] + cols].round(4).to_string(index=False))

base = "d0.0"
print(f"\n{'variant':>10s} {'regret':>8s} {'vs d0':>9s} {'W/L/T':>10s} {'p':>7s}")
for c in cols:
    d = R[base] - R[c]
    p = wilcoxon(R[base], R[c]).pvalue if np.any(d != 0) else np.nan
    print(f"{c:>10s} {R[c].mean():8.4f} {100*d.mean()/max(R[base].mean(),1e-9):+8.1f}% "
          f"{int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum()):>2d} {p:7.3f}")

print("\n=== by true-best family ===")
for f, g in R.groupby("best_fam"):
    print(f"  best={f:6s} n={len(g)}  " + "  ".join(f"{c}={g[c].mean():.4f}" for c in cols))
