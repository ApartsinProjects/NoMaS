# -*- coding: utf-8 -*-
"""Richer selection-method comparison: mean regret is outlier-skewed, so also report median, tails
(p75/p90/max), frac picking the oracle (regret~0), frac badly wrong (>0.3), and the full regret
HISTOGRAM. Reads a STREAM_RANK csv (arg or default)."""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"; D = os.path.join(S, "scratchpad", "streamline")
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "STREAM_RANK.csv")
df = pd.read_csv(SRC)
METH = [("NoMaS matched", "reg_matched"), ("NoMaS beta=1", "reg_beta1"), ("EM", "reg_em"), ("MV", "reg_mv"),
        ("consensus/UDR", "reg_consensus"), ("ModelCentrality", "reg_model_centrality"), ("HITS", "reg_hits"), ("random", "reg_random")]
def macro(c): return np.mean([df[df.truefam == fa][c].mean() for fa in ["local", "global"] if (df.truefam == fa).any()])
mv, e = df.reg_mv.values, df.reg_em.values
print(f"=== regret distribution across {len(df)} datasets  (src={os.path.basename(SRC)}) ===")
print(f"  {'method':17s} {'mean':>6s} {'macro':>6s} {'median':>7s} {'p75':>6s} {'p90':>6s} {'max':>6s} {'%oracle':>8s} {'%>.3':>6s} {'vsMV':>6s}")
rows = []
for nm, c in METH:
    a = df[c].values
    pm = wilcoxon(a, mv).pvalue if c != "reg_mv" and (a != mv).any() else np.nan
    r = dict(method=nm, mean=a.mean(), macro=macro(c), median=np.median(a), p75=np.percentile(a, 75), p90=np.percentile(a, 90),
             max=a.max(), pct_oracle=(a < 0.01).mean(), pct_bad=(a > 0.3).mean(), vsMV=pm)
    rows.append(r)
    print(f"  {nm:17s} {r['mean']:6.3f} {r['macro']:6.3f} {r['median']:7.3f} {r['p75']:6.3f} {r['p90']:6.3f} {r['max']:6.3f} {r['pct_oracle']:7.0%} {r['pct_bad']:6.0%} {pm:6.3f}")
pd.DataFrame(rows).to_csv(os.path.join(D, "STREAM_REGRET_DIST.csv"), index=False)
# histograms: matched vs MV vs EM vs random
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6)); bins = np.linspace(0, max(0.6, df[["reg_matched", "reg_mv", "reg_em"]].values.max()), 31)
for nm, c, col in [("NoMaS matched", "reg_matched", "#2a9d8f"), ("MV", "reg_mv", "#e76f51"), ("EM", "reg_em", "#6a4c93"), ("random", "reg_random", "#bbb")]:
    ax[0].hist(df[c], bins=bins, histtype="step", lw=2, label=f"{nm} (med {np.median(df[c]):.3f})", color=col)
ax[0].set_xlabel("per-dataset regret"); ax[0].set_ylabel("# datasets"); ax[0].set_title("regret histogram (mean is right-skewed by the tail)"); ax[0].legend()
# ECDF (clearer for skew): fraction of datasets with regret <= x
for nm, c, col in [("NoMaS matched", "reg_matched", "#2a9d8f"), ("MV", "reg_mv", "#e76f51"), ("EM", "reg_em", "#6a4c93")]:
    x = np.sort(df[c].values); y = np.arange(1, len(x) + 1) / len(x); ax[1].step(x, y, lw=2, label=nm, color=col)
ax[1].set_xlabel("regret"); ax[1].set_ylabel("fraction of datasets <= regret"); ax[1].set_title("regret ECDF (higher curve = better)"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); fig.savefig(os.path.join(D, "STREAM_REGRET_DIST.png"), dpi=110)
print("\n  note: mean vs median gap shows skew; %oracle = picked the best detector; %>.3 = badly wrong")
print("saved streamline/STREAM_REGRET_DIST.csv + .png")
