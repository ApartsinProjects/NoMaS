# -*- coding: utf-8 -*-
"""Systematic audit of every intermediate result produced today.

Checks, per file where applicable:
  C1 metric range      : all AUCs in [0,1]
  C2 regret sign       : regret >= 0 always (regret = best - picked, cannot be negative)
  C3 NaN / inf         : none in the load-bearing columns
  C4 duplicate rows    : same (dataset, cluster, arm, detector) appearing twice
  C5 panel completeness: how many cells have fewer than 9 detectors (unequal panels bias
                         regret, because if the TRUE-BEST detector failed on a cell, the
                         pick cannot be right)
  C6 CROSS-EXPERIMENT CONSISTENCY: the same (dataset, detector) true_auc computed in
                         different experiments should agree. They subsample differently
                         (MAX_N 4000 vs 5000, different RNG draws), so disagreement is
                         EXPECTED - the question is whether it is large enough to flip
                         the BEST detector, which would make the global/local family
                         labels inconsistent between experiments.
"""
import os
import numpy as np
import pandas as pd

S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
fam = lambda d: "local" if d in LOCAL else ("global" if d in GLOBAL else "other")

FILES = ["displace_full.csv", "displace.csv", "gen_anom.csv", "gen_valid.csv",
         "perm_band.csv", "perm_filter2.csv", "edge_full_all.csv", "latent_sep.csv",
         "latent_umap.csv", "edge_rows.csv"]

print("=" * 78)
for f in FILES:
    p = os.path.join(S, f)
    if not os.path.exists(p):
        print(f"{f:24s} MISSING")
        continue
    D = pd.read_csv(p)
    issues = []
    # C1 metric range
    for c in ["pseudo_auc", "true_auc", "silhouette"]:
        if c in D:
            v = D[c].dropna()
            if c != "silhouette" and len(v) and (v.min() < -1e-9 or v.max() > 1 + 1e-9):
                issues.append(f"C1 {c} out of [0,1]: [{v.min():.3f},{v.max():.3f}]")
    # C2 regret sign
    if "regret" in D:
        neg = (D.regret < -1e-9).sum()
        if neg:
            issues.append(f"C2 NEGATIVE regret in {neg} rows (min {D.regret.min():.4f})")
    # C3 NaN
    for c in ["pseudo_auc", "true_auc", "regret"]:
        if c in D and D[c].isna().any():
            issues.append(f"C3 NaN in {c}: {int(D[c].isna().sum())}")
    # C4 duplicates
    keys = [k for k in ["dataset", "cluster", "detector", "delta", "gen", "filt", "q",
                        "frac", "band", "arm", "latent"] if k in D]
    if "detector" in D and len(keys) > 1:
        dup = int(D.duplicated(subset=keys).sum())
        if dup:
            issues.append(f"C4 {dup} duplicate rows on {keys}")
    # C5 panel completeness
    if "detector" in D:
        gk = [k for k in keys if k != "detector"]
        if gk:
            n = D.groupby(gk).detector.nunique()
            short = int((n < 9).sum())
            if short:
                issues.append(f"C5 {short}/{len(n)} cells have <9 detectors (min {n.min()})")
    print(f"{f:24s} rows={len(D):6d} " + ("OK" if not issues else ""))
    for i in issues:
        print(f"{'':24s}   !! {i}")

# ---- C6 cross-experiment consistency of true_auc and of the BEST detector ----
print("=" * 78)
print("C6 CROSS-EXPERIMENT CONSISTENCY of true_auc / best detector")
best, aucs = {}, {}
for f in ["displace_full.csv", "gen_anom.csv", "gen_valid.csv", "perm_band.csv",
          "perm_filter2.csv", "edge_full_all.csv"]:
    p = os.path.join(S, f)
    if not os.path.exists(p):
        continue
    D = pd.read_csv(p)
    if "true_auc" not in D:
        continue
    for ds, g in D.groupby("dataset"):
        t = g.groupby("detector").true_auc.first()
        best.setdefault(ds, {})[f] = t.idxmax()
        aucs.setdefault(ds, {})[f] = t

rows = []
for ds, d in best.items():
    vals = list(d.values())
    if len(vals) < 2:
        continue
    fams = {fam(v) for v in vals}
    rows.append(dict(dataset=ds, n_exp=len(vals), distinct_best=len(set(vals)),
                     distinct_fam=len(fams), bests="/".join(sorted(set(vals)))))
R = pd.DataFrame(rows).sort_values("distinct_fam", ascending=False)
print(R.to_string(index=False))
if len(R):
    print(f"\n  datasets where the BEST DETECTOR differs across experiments: "
          f"{int((R.distinct_best > 1).sum())}/{len(R)}")
    print(f"  datasets where the FAMILY LABEL differs across experiments : "
          f"{int((R.distinct_fam > 1).sum())}/{len(R)}   <-- these make the "
          f"global/local splits inconsistent between experiments")

# magnitude of true_auc disagreement
diffs = []
for ds, d in aucs.items():
    if len(d) < 2:
        continue
    M = pd.DataFrame(d)
    diffs.append((M.max(axis=1) - M.min(axis=1)).max())
if diffs:
    print(f"\n  max |true_auc| spread for a detector across experiments: "
          f"median {np.median(diffs):.4f}, worst {np.max(diffs):.4f}")
