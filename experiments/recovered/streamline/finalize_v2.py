# -*- coding: utf-8 -*-
"""Corrected-benchmark leaderboard (v2): full pool on the grouped-split + base-rate-capped bundle.
Applies post-filters (separability floor, selection-trivial floor, min anomalies), computes UOMS
baselines + all SPARC variants (full-pool and classical-only probe candidates), and reports overall
+ per-modality with paired Wilcoxon. Writes STREAM_V2_LEADERBOARD.csv."""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_round2_common import consensus_scores, model_centrality_scores, hits_authority_scores, _em_auc, _mv_auc
SRC = os.path.join(D, "deepres_v2")
mine = pd.read_csv(os.path.join(D, "STREAM_MINE_FIXED.csv")); corr = dict(zip(mine.key, mine.corr_str))
idx = pd.read_csv(os.path.join(D, "bundle_v2", "_index.csv")); nh = dict(zip(idx.key, idx.n_h))
NGEN = 1000; TAU = -0.14; t = np.linspace(0, 100, 1000); alpha = np.linspace(0.9, 0.999, 1000)
J = {}
for fn in glob.glob(os.path.join(SRC, "shard_*.jsonl")):
    for ln in open(fn):
        try:
            r = json.loads(ln)
            if "ap" in r and len(r["ap"]) >= 6: J[r["key"]] = r
        except Exception: pass
rows = []
for key, r in J.items():
    scf = os.path.join(SRC, f"sc_{key}.npz")
    if not os.path.exists(scf): continue
    z = np.load(scf); names = [n for n in r["ap"] if f"val__{n}" in z.files and f"uni__{n}" in z.files and n in r["a1"] and n in r["a4"]]
    if len(names) < 6: continue
    ap = {n: r["ap"][n] for n in names}; a1 = r["a1"]; a4 = r["a4"]; Vm = np.column_stack([z[f"val__{n}"].astype(float) for n in names])
    emv, mvv = {}, {}
    for n in names:
        sv = z[f"val__{n}"].astype(float); su = z[f"uni__{n}"].astype(float)
        su = np.nan_to_num(su, nan=float(np.nanmedian(su[np.isfinite(su)])) if np.isfinite(su).any() else 0.0)
        emv[n] = _em_auc(t, 1.0, -su, -sv, NGEN); mvv[n] = _mv_auc(alpha, 1.0, -su, -sv, NGEN)
    cons = dict(zip(names, consensus_scores(Vm))); mc = dict(zip(names, model_centrality_scores(Vm))); ht = dict(zip(names, hits_authority_scores(Vm)))
    av = pd.Series(ap); best = av.max(); cls = [n for n in names if not n.startswith("D_")]
    p1 = max(names, key=lambda v: a1[v]); pe = max(names, key=lambda v: a4[v]); lev = max(a1[v] for v in names) - max(a4[v] for v in names)
    p1c = max(cls, key=lambda v: a1[v]); pec = max(cls, key=lambda v: a4[v])
    cs = corr.get(key, np.nan)
    rows.append(dict(key=key, corpus=key.split("__")[0], cs=cs, lev=lev, oracle_ap=best, n_h=nh.get(key, 999),
        reg_em=best - av[max(names, key=lambda v: emv[v])], reg_mv=best - av[min(names, key=lambda v: mvv[v])],
        reg_consensus=best - av[max(names, key=lambda v: cons[v])], reg_mc=best - av[max(names, key=lambda v: mc[v])],
        reg_hits=best - av[max(names, key=lambda v: ht[v])], reg_random=best - av.mean(),
        reg_beta1=best - av[p1], reg_pe=best - av[pe], reg_p1c=best - av[p1c], reg_pec=best - av[pec]))
df = pd.DataFrame(rows); med = df.cs.median()
# routing variants (full-pool candidates)
df["reg_corr"] = np.where(df.cs > med, df.reg_beta1, df.reg_pe)
df["reg_combined"] = np.where((df.cs > med) | (df.lev > TAU), df.reg_beta1, df.reg_pe)
# classical-candidate variants
df["reg_corr_cls"] = np.where(df.cs > med, df.reg_p1c, df.reg_pec)
df["reg_combined_cls"] = np.where((df.cs > med) | (df.lev > TAU), df.reg_p1c, df.reg_pec)
df.to_csv(os.path.join(D, "STREAM_V2_LEADERBOARD.csv"), index=False)
# post-filters
FLOOR, TRIVSEL = 0.05, 0.02
keep = (df.oracle_ap > FLOOR) & (df.reg_random > TRIVSEL) & (df.n_h >= 100)
print(f"post-filter: {len(df)} -> {int(keep.sum())} datasets  (dropped: separability<={FLOOR}: {int((df.oracle_ap<=FLOOR).sum())}, selection-trivial reg_random<={TRIVSEL}: {int((df.reg_random<=TRIVSEL).sum())}, n_h<100: {int((df.n_h<100).sum())})")
dff = df[keep]
def report(g, label):
    a = g.reg_combined_cls.values  # headline = classical-candidate combined router
    def st(c):
        p = wilcoxon(a, g[c].values).pvalue if not np.allclose(a, g[c].values) else np.nan
        return f"{g[c].mean():.3f} (p={p:.4f})"
    print(f"\n[{label} n={len(g)}]  SPARC-combined(cls) {a.mean():.3f}")
    for nm, c in [("SPARC-combined(full)", "reg_combined"), ("SPARC-corr(cls)", "reg_corr_cls"), ("SPARC-beta1", "reg_beta1"),
                  ("EM", "reg_em"), ("MV", "reg_mv"), ("consensus", "reg_consensus"), ("random", "reg_random")]:
        print(f"    {nm:20s} {st(c)}")
report(dff, "ALL (filtered)")
for cp in ["tsbad_m", "ovrbench", "oddbench"]:
    g = dff[dff.corpus == cp]
    if len(g) >= 5: report(g, cp)
print(f"\nsaved STREAM_V2_LEADERBOARD.csv ({len(df)} scored)")
