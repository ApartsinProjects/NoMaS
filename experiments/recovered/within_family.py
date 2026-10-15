# -*- coding: utf-8 -*-
"""Why does perfect routing leave so much regret? Decompose the regret into
(a) the between-family error routing could fix, and (b) the within-family
error routing cannot touch.

For each row, with F = the TRUE best detector's family:
  reg_now      : argmax(score) over all detectors        (current)
  reg_route    : argmax(score) restricted to F           (oracle routing)
  reg_infam_or : best detector inside F                  (= 0 by construction)
  spearman of score vs true_auc INSIDE F                 (is the score informative there?)
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CSV = r"E:\Projects\Submitted\ADRank\results\oddbench.csv"
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}


def fam(d):
    return "local" if d in LOCAL else ("global" if d in GLOBAL else "other")


df = pd.read_csv(CSV)
df = df[df.status == "ok"]

out = []
for _, r in df.iterrows():
    ta = json.loads(r.true_auc_json); sc = json.loads(r.score_json)
    dets = [d for d in ta if d in sc and not pd.isna(ta[d])]
    if len(dets) < 3:
        continue
    best = max(ta[d] for d in dets)
    bd = max(dets, key=lambda d: ta[d]); F = fam(bd)
    inf = [d for d in dets if fam(d) == F]
    pick_all = max(dets, key=lambda d: sc[d])
    pick_inf = max(inf, key=lambda d: sc[d])
    rho = np.nan
    if len(inf) >= 3:
        rho = spearmanr([sc[d] for d in inf], [ta[d] for d in inf]).statistic
    out.append(dict(fam=F, n_in_fam=len(inf),
                    reg_now=best - ta[pick_all],
                    reg_route=best - ta[pick_inf],
                    rho_in_fam=rho,
                    spread_in_fam=max(ta[d] for d in inf) - min(ta[d] for d in inf),
                    spread_all=best - min(ta[d] for d in dets)))
R = pd.DataFrame(out)

print("=== regret decomposition (OddBench, 561 rows) ===")
print(f"current                 : {R.reg_now.mean():.4f}")
print(f"oracle family routing   : {R.reg_route.mean():.4f}")
print(f"oracle detector (=0)    : 0.0000")
print(f"\nof the {R.reg_now.mean():.4f} current regret:")
print(f"  removable by routing  : {R.reg_now.mean()-R.reg_route.mean():.4f} "
      f"({100*(R.reg_now.mean()-R.reg_route.mean())/R.reg_now.mean():.0f}%)")
print(f"  WITHIN-family residual: {R.reg_route.mean():.4f} "
      f"({100*R.reg_route.mean()/R.reg_now.mean():.0f}%)  <- routing cannot touch this")

print("\n=== is the pseudo-score informative INSIDE a family? ===")
for f in ["local", "global", "other"]:
    s = R[(R.fam == f) & R.rho_in_fam.notna()]
    if len(s):
        print(f"  {f:6s} n={len(s):4d}  mean Spearman(score, true_auc) within family = "
              f"{s.rho_in_fam.mean():+.3f}   (0 = uninformative)")

print("\n=== how much true-AUC spread sits inside vs across families ===")
print(f"  mean spread within the true-best family : {R.spread_in_fam.mean():.4f}")
print(f"  mean spread across all detectors        : {R.spread_all.mean():.4f}")
