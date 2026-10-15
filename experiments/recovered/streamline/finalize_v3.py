# -*- coding: utf-8 -*-
"""FINAL leaderboard (v3): consistent standardized preprocessing for the WHOLE pool. Classical detectors
re-scored standardized (deepres_v2_clstd) + deep detectors from deepres_v2 (already standardized). Fixes
the raw-vs-standardized harness bug. Merges the two, computes UOMS baselines + SPARC variants, applies
post-filters, reports overall + per-modality. Writes STREAM_V3_LEADERBOARD.csv."""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_round2_common import consensus_scores, model_centrality_scores, hits_authority_scores, _em_auc, _mv_auc
CLS = os.path.join(D, "deepres_v2_clstd"); DEEP = os.path.join(D, "deepres_v2")
mine = pd.read_csv(os.path.join(D, "STREAM_MINE_FIXED.csv")); corr = dict(zip(mine.key, mine.corr_str))
idx = pd.read_csv(os.path.join(D, "bundle_v2", "_index.csv")); nh = dict(zip(idx.key, idx.n_h))
NGEN = 1000; TAU = -0.14; t = np.linspace(0, 100, 1000); alpha = np.linspace(0.9, 0.999, 1000)
def load(dirn):
    J = {}
    for fn in glob.glob(os.path.join(dirn, "shard_*.jsonl")):
        for ln in open(fn):
            try:
                r = json.loads(ln)
                if "ap" in r: J[r["key"]] = r
            except Exception: pass
    return J
Jc = load(CLS); Jd = load(DEEP)
rows = []
for key in Jc:
    if key not in Jd: continue
    scc = os.path.join(CLS, f"sc_{key}.npz"); scd = os.path.join(DEEP, f"sc_{key}.npz")
    if not (os.path.exists(scc) and os.path.exists(scd)): continue
    zc = np.load(scc); zd = np.load(scd)
    cls = [n for n in Jc[key]["ap"] if not n.startswith("D_") and f"val__{n}" in zc.files]  # std classical
    deep = [n for n in Jd[key]["ap"] if n.startswith("D_") and f"val__{n}" in zd.files]      # deep (std already)
    names = cls + deep
    if len(names) < 6: continue
    ap = {**{n: Jc[key]["ap"][n] for n in cls}, **{n: Jd[key]["ap"][n] for n in deep}}
    a1 = {**{n: Jc[key]["a1"][n] for n in cls if n in Jc[key]["a1"]}, **{n: Jd[key]["a1"][n] for n in deep if n in Jd[key]["a1"]}}
    a4 = {**{n: Jc[key]["a4"][n] for n in cls if n in Jc[key]["a4"]}, **{n: Jd[key]["a4"][n] for n in deep if n in Jd[key]["a4"]}}
    names = [n for n in names if n in a1 and n in a4]
    if len(names) < 6: continue
    val = {n: (zc if n in cls else zd)[f"val__{n}"].astype(float) for n in names}
    uni = {n: (zc if n in cls else zd)[f"uni__{n}"].astype(float) for n in names}
    Vm = np.column_stack([val[n] for n in names]); emv, mvv = {}, {}
    for n in names:
        sv = val[n]; su = np.nan_to_num(uni[n], nan=float(np.nanmedian(uni[n][np.isfinite(uni[n])])) if np.isfinite(uni[n]).any() else 0.0)
        emv[n] = _em_auc(t, 1.0, -su, -sv, NGEN); mvv[n] = _mv_auc(alpha, 1.0, -su, -sv, NGEN)
    cons = dict(zip(names, consensus_scores(Vm))); mc = dict(zip(names, model_centrality_scores(Vm))); ht = dict(zip(names, hits_authority_scores(Vm)))
    av = pd.Series(ap); best = av.max(); clsn = [n for n in names if not n.startswith("D_")]
    p1 = max(names, key=lambda v: a1[v]); pe = max(names, key=lambda v: a4[v]); lev = max(a1[v] for v in names) - max(a4[v] for v in names)
    p1c = max(clsn, key=lambda v: a1[v]); pec = max(clsn, key=lambda v: a4[v]); cs = corr.get(key, np.nan)
    rows.append(dict(key=key, corpus=key.split("__")[0], cs=cs, lev=lev, oracle_ap=best, n_h=nh.get(key, 999),
        oracle_cls=max(av[n] for n in clsn), oracle_deep=max([av[n] for n in names if n.startswith("D_")] or [np.nan]),
        reg_em=best - av[max(names, key=lambda v: emv[v])], reg_mv=best - av[min(names, key=lambda v: mvv[v])],
        reg_consensus=best - av[max(names, key=lambda v: cons[v])], reg_mc=best - av[max(names, key=lambda v: mc[v])],
        reg_hits=best - av[max(names, key=lambda v: ht[v])], reg_random=best - av.mean(),
        reg_beta1=best - av[p1], reg_pe=best - av[pe], reg_p1c=best - av[p1c], reg_pec=best - av[pec]))
df = pd.DataFrame(rows); med = df.cs.median()
df["reg_combined"] = np.where((df.cs > med) | (df.lev > TAU), df.reg_beta1, df.reg_pe)
df["reg_combined_cls"] = np.where((df.cs > med) | (df.lev > TAU), df.reg_p1c, df.reg_pec)
df["reg_hybrid"] = np.where(df.cs > med, df.reg_combined_cls, df.reg_em)
df.to_csv(os.path.join(D, "STREAM_V3_LEADERBOARD.csv"), index=False)
keep = (df.oracle_ap > 0.05) & (df.reg_random > 0.02) & (df.n_h >= 100)
print(f"v3 (standardized pool): {len(df)} scored -> {int(keep.sum())} in benchmark; deep-oracle-gap {(df.oracle_ap-df.oracle_cls).mean():.3f} (was 0.034 raw)")
def rep(g, lab):
    a = g.reg_combined_cls.values
    print(f"[{lab} n={len(g)}] SPARC-cls {a.mean():.3f}", end="")
    for nm, c in [("EM", "reg_em"), ("MV", "reg_mv"), ("consensus", "reg_consensus"), ("random", "reg_random"), ("hybrid", "reg_hybrid")]:
        p = wilcoxon(a, g[c].values).pvalue if not np.allclose(a, g[c].values) else np.nan
        print(f"  {nm} {g[c].mean():.3f}(p={p:.4f})", end="")
    print()
rep(df[keep], "ALL")
for cp in ["tsbad_m", "ovrbench", "oddbench"]:
    g = df[keep][df[keep].corpus == cp]
    if len(g) >= 5: rep(g, cp)
print("saved STREAM_V3_LEADERBOARD.csv")
