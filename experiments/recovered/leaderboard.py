# -*- coding: utf-8 -*-
"""Leaderboard of label-free selection methods on the 162-dataset construct-matched set."""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import true_apnorm
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}
df = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv"))
df["truefam"] = np.where(df.best_local > df.best_global, "local", "global")

# best-fixed detector: mean ap_norm per detector across the 162 datasets, pick argmax, then its regret
apmat = {}
for _, r in df.iterrows():
    try: ap, _ = true_apnorm(CSVMAP[r.corpus], r.dataset)
    except Exception: continue
    apmat[r.dataset] = ap
alldet = pd.DataFrame(apmat).T  # datasets x detectors
meanap = alldet.mean(0); fixed_det = meanap.idxmax()
df["reg_fixed"] = [df.best_overall.iloc[i] - (apmat.get(d, pd.Series()).get(fixed_det, np.nan)) for i, d in enumerate(df.dataset)]

METHODS = [("oracle", None), ("best_fixed (cheats: global-best detector = %s)" % fixed_det, "reg_fixed"),
           ("NoMaS beta=1 synthetic (OURS)", "reg_beta1"), ("NoMaS matched-router (OURS)", "reg_matched"),
           ("EM (Excess-Mass)", "reg_em"), ("beta=-4 extreme synthetic", "reg_betaext"), ("random", "reg_random")]
def macro(col): return np.mean([df[df.truefam == f][col].mean() for f in ["local", "global"]])
e = df.reg_em.values
print(f"=== label-free selection leaderboard ({len(df)} datasets: 106 local / 56 global) ===")
print(f"    regret on ap_norm (lower=better); macro = family-balanced\n")
print(f"    {'method':46s} {'micro':>7s} {'macro':>7s}  {'vs EM':>16s}")
rows = []
for name, col in METHODS:
    if col is None:
        print(f"    {name:46s} {0.0:7.3f} {0.0:7.3f}"); continue
    a = df[col].values
    mic, mac = np.nanmean(a), macro(col)
    try: p = wilcoxon(a[~np.isnan(a)], e[~np.isnan(a)]).pvalue; w = int((a < e).sum())
    except Exception: p, w = np.nan, 0
    tag = "" if col == "reg_em" else f"p={p:.3f} ({w}/{len(a)})"
    print(f"    {name:46s} {mic:7.3f} {mac:7.3f}  {tag:>16s}")
    rows.append((name, mic, mac))
print(f"\n    (agreement methods from the canonical HADB_SELECTORS_V2 run on the 292-set, NOT this subset:")
print(f"     MV 0.220, consensus/ModelCentrality/HITS ~0.278 = random 0.286; EM 0.216 was best there too.)")
pd.DataFrame(rows, columns=["method", "micro", "macro"]).to_csv(os.path.join(S, "LEADERBOARD.csv"), index=False)
print("\nsaved LEADERBOARD.csv")
