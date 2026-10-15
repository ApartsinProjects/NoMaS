# -*- coding: utf-8 -*-
"""Do the two real audit findings actually BIAS the results?

F1  unequal panels: ~9% of cells have 8 detectors not 9. If the panel differs BETWEEN ARMS
    for the same cluster, the arms are not compared on the same detector set - an unfair
    comparison. If it differs only ACROSS clusters, every arm sees the same panel and the
    comparison is fair.

F2  best-detector flips across experiments: 5/38 datasets. If the flipped detectors are
    NEAR-TIES (tiny true-AUC gap), the flip is harmless for regret. If the gap is large,
    the family labels (and therefore the global/local splits) are unreliable.
"""
import os
import numpy as np
import pandas as pd

S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"

print("=== F1: do detector FAILURES differ BY ARM? ===")
D = pd.read_csv(os.path.join(S, "displace_full.csv"))
n = D.groupby(["dataset", "cluster", "delta"]).detector.nunique().rename("ndet").reset_index()
print(n.groupby("delta").ndet.agg(["mean", "min"]).round(3).to_string())
w = n.pivot_table(index=["dataset", "cluster"], columns="delta", values="ndet")
uneq = int((w.nunique(axis=1) > 1).sum())
print(f"  clusters whose panel SIZE differs across deltas: {uneq} of {len(w)}")
if uneq == 0:
    print("  -> every delta sees the SAME detector panel per cluster: comparison is FAIR")
else:
    print("  -> UNFAIR: some arms are scored on a different panel than others")

# which detectors fail, and does failure correlate with delta?
fails = []
allde = sorted(D.detector.unique())
for (ds, cl, dl), g in D.groupby(["dataset", "cluster", "delta"]):
    miss = set(allde) - set(g.detector)
    for m in miss:
        fails.append((dl, m))
if fails:
    F = pd.DataFrame(fails, columns=["delta", "detector"])
    print("\n  which detector fails, by delta:")
    print(F.pivot_table(index="detector", columns="delta", aggfunc=len, fill_value=0).to_string())

print("\n=== F2: are the flipped best-detectors NEAR-TIES? ===")
targets = ["28_pendigits", "26_optdigits", "Cardiotocography", "43_WDBC", "23_mammography"]
for f in ["displace_full.csv", "gen_anom.csv", "gen_valid.csv", "perm_band.csv"]:
    p = os.path.join(S, f)
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    if "true_auc" not in d:
        continue
    for ds in targets:
        g = d[d.dataset == ds]
        if not len(g):
            continue
        t = g.groupby("detector").true_auc.first().sort_values(ascending=False)
        if len(t) < 2:
            continue
        print(f"  {f:18s} {ds:17s} {t.index[0]:8s}={t.iloc[0]:.4f}  "
              f"{t.index[1]:8s}={t.iloc[1]:.4f}  gap={t.iloc[0]-t.iloc[1]:.4f}")
