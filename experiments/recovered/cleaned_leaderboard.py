# -*- coding: utf-8 -*-
"""Re-run the leaderboard on the OR-triviality-filtered benchmark (drop datasets a per-feature
calibrated OR rule already solves: mv_gain < eps). Compares full vs cleaned; drops beta=-4."""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
lb = pd.read_csv(os.path.join(S, "FULL_LEADERBOARD.csv"))
co = pd.read_csv(os.path.join(S, "CALIBRATED_OR.csv"))[["dataset", "mv_gain"]]
df0 = lb.merge(co, on="dataset", how="inner")
METH = [("NoMaS matched (ours)", "reg_matched"), ("NoMaS beta=1 (ours)", "reg_beta1"),
        ("EM", "reg_em"), ("MV", "reg_mv"), ("consensus/UDR", "reg_consensus"),
        ("ModelCentrality", "reg_model_centrality"), ("HITS", "reg_hits"),
        ("iforest_random", "reg_iforest_random"), ("random", "reg_random")]

def report(df, title):
    lo = (df.truefam == "local").sum(); gl = (df.truefam == "global").sum()
    def macro(c): return np.mean([df[df.truefam == f][c].mean() for f in ["local", "global"] if (df.truefam == f).any()])
    e, mv = df.reg_em.values, df.reg_mv.values
    print(f"\n=== {title}  (n={len(df)}: {lo} local / {gl} global) ===")
    print(f"    {'method':24s} {'micro':>7s} {'macro':>7s} {'vsEM p':>8s} {'vsMV p':>8s}")
    print(f"    {'oracle':24s} {0.0:7.3f} {0.0:7.3f}")
    for name, c in METH:
        a = df[c].values; ok = ~np.isnan(a)
        pe = wilcoxon(a[ok], e[ok]).pvalue if c != "reg_em" and (a[ok] != e[ok]).any() else np.nan
        pm = wilcoxon(a[ok], mv[ok]).pvalue if c != "reg_mv" and (a[ok] != mv[ok]).any() else np.nan
        print(f"    {name:24s} {np.nanmean(a):7.3f} {macro(c):7.3f} {pe:8.3f} {pm:8.3f}")

report(df0, "FULL benchmark")
for eps in [0.05]:
    report(df0[df0.mv_gain >= eps], f"CLEANED (drop trivial OR, eps={eps})")
