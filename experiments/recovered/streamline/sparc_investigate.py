# -*- coding: utf-8 -*-
"""Diagnose why SPARC's routing is weak and search probe-based selection rules that beat EM/MV.
Per dataset we have ap[det] (test ground truth), a1[det] (dependency-probe AUC), a4[det] (marginal-probe
AUC). regret(rule) = max ap - ap[rule's pick]. Report pooled regret + paired Wilcoxon vs EM/MV/beta1,
plus probe-quality diagnostics and the routing ceiling."""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); SRC = os.path.join(D, "deepres3")
lb = pd.read_csv(os.path.join(D, "STREAM_FINAL_LEADERBOARD.csv")).set_index("key")
recs = {}
for fn in glob.glob(os.path.join(SRC, "shard_*.jsonl")):
    for ln in open(fn):
        ln = ln.strip()
        if not ln: continue
        r = json.loads(ln)
        if "ap" not in r or len(r.get("ap", {})) < 6: continue
        names = [n for n in r["ap"] if n in r.get("a1", {}) and n in r.get("a4", {})]
        if len(names) < 6 or r["key"] not in lb.index: continue
        recs[r["key"]] = (names, r["ap"], r["a1"], r["a4"])
keys = list(recs)
def reg_of(crit):  # crit(names,ap,a1,a4)->picked detector; returns per-dataset regret array
    out = []
    for k in keys:
        names, ap, a1, a4 = recs[k]; best = max(ap[n] for n in names); pick = crit(names, ap, a1, a4); out.append(best - ap[pick])
    return np.array(out)
rules = {
    "beta1 (argmax a1)": lambda nm, ap, a1, a4: max(nm, key=lambda v: a1[v]),
    "pe (argmax a4)": lambda nm, ap, a1, a4: max(nm, key=lambda v: a4[v]),
    "sum a1+a4": lambda nm, ap, a1, a4: max(nm, key=lambda v: a1[v] + a4[v]),
    "min(a1,a4)": lambda nm, ap, a1, a4: max(nm, key=lambda v: min(a1[v], a4[v])),
    "max(a1,a4)": lambda nm, ap, a1, a4: max(nm, key=lambda v: max(a1[v], a4[v])),
    "prod a1*a4": lambda nm, ap, a1, a4: max(nm, key=lambda v: a1[v] * a4[v]),
    "harmonic": lambda nm, ap, a1, a4: max(nm, key=lambda v: (2 * a1[v] * a4[v]) / (a1[v] + a4[v] + 1e-9)),
}
def borda(nm, ap, a1, a4):
    r1 = {v: i for i, v in enumerate(sorted(nm, key=lambda v: a1[v]))}
    r4 = {v: i for i, v in enumerate(sorted(nm, key=lambda v: a4[v]))}
    return max(nm, key=lambda v: r1[v] + r4[v])
rules["borda(a1,a4)"] = borda
em = lb.loc[keys, "reg_em"].values; mv = lb.loc[keys, "reg_mv"].values; comb = lb.loc[keys, "reg_combined"].values
rand = lb.loc[keys, "reg_random"].values
print(f"reference: EM {em.mean():.3f}  MV {mv.mean():.3f}  SPARC-combined {comb.mean():.3f}  random {rand.mean():.3f}  (n={len(keys)})")
print(f"\n{'rule':22s}{'regret':>8s}{'p_vs_EM':>9s}{'p_vs_MV':>9s}{'W/L vs MV':>12s}")
res = {}
for nm, cr in rules.items():
    rg = reg_of(cr); res[nm] = rg
    pE = wilcoxon(rg, em).pvalue; pM = wilcoxon(rg, mv).pvalue; w = (rg < mv).sum(); l = (rg > mv).sum()
    print(f"{nm:22s}{rg.mean():>8.3f}{pE:>9.4f}{pM:>9.4f}{f'{w}/{l}':>12s}")
# routing ceiling: perfect pick between p1 and pe
p1r = reg_of(rules["beta1 (argmax a1)"]); per = reg_of(rules["pe (argmax a4)"])
ceil = np.minimum(p1r, per)
print(f"\nrouting CEILING (oracle pick of p1 vs pe): {ceil.mean():.3f}  (beta1 {p1r.mean():.3f}, pe {per.mean():.3f})")
allpick = reg_of(lambda nm, ap, a1, a4: max(nm, key=lambda v: max(a1[v], a4[v])))
# probe quality: per-dataset Spearman between probe grade and true ap ranking
s1 = np.mean([spearmanr([recs[k][2][n] for n in recs[k][0]], [recs[k][1][n] for n in recs[k][0]]).correlation for k in keys])
s4 = np.mean([spearmanr([recs[k][3][n] for n in recs[k][0]], [recs[k][1][n] for n in recs[k][0]]).correlation for k in keys])
sbest = np.mean([spearmanr([max(recs[k][2][n], recs[k][3][n]) for n in recs[k][0]], [recs[k][1][n] for n in recs[k][0]]).correlation for k in keys])
print(f"\nprobe->truth rank corr (mean per-dataset Spearman): a1 {s1:.3f}  a4 {s4:.3f}  max(a1,a4) {sbest:.3f}")
# how often is beta1's pick the oracle-best?
hit = np.mean([max(recs[k][0], key=lambda v: recs[k][2][v]) == max(recs[k][0], key=lambda v: recs[k][1][v]) for k in keys])
print(f"beta1 top-1 hit rate (probe-best == oracle-best): {hit:.2f}")
# best new rule vs EM/MV significance summary
best_rule = min(res, key=lambda r: res[r].mean())
print(f"\nbest simple rule: {best_rule} = {res[best_rule].mean():.3f}")
