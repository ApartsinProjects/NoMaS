# -*- coding: utf-8 -*-
"""FINAL unified leaderboard: full pool (classical + deep) scored on all datasets; ALL selectors ranked
over that same pool, construct-matched. Reads deepres3/ (jsonl ap/a1/a4 + sc_<key>.npz val/uni scores)
+ STREAM_MINE.csv (corr_str). Selectors: EM, MV, consensus, ModelCentrality, HITS (over full pool),
NoMaS matched (local_ev route), NoMaS corr_str route, beta1, random. Reports micro/macro/median +
Wilcoxon vs MV/EM with Holm correction."""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_round2_common import consensus_scores, model_centrality_scores, hits_authority_scores, _em_auc, _mv_auc
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "deepres3")
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
NGEN = 1000; TAU = -0.14; t = np.linspace(0, 100, 1000); alpha = np.linspace(0.9, 0.999, 1000)
# jsonl: ap/a1/a4 per detector
J = {}
for fn in glob.glob(os.path.join(SRC, "shard_*.jsonl")):
    for ln in open(fn):
        ln = ln.strip()
        if not ln: continue
        r = json.loads(ln)
        if "ap" in r and len(r.get("ap", {})) >= 6: J[r["key"]] = r
mine = pd.read_csv(os.path.join(D, "STREAM_MINE.csv")); mine["key"] = mine.corpus + "__" + mine.dataset
corr = dict(zip(mine.key, mine.corr_str))
rows = []
for key, r in J.items():
    scf = os.path.join(SRC, f"sc_{key}.npz")
    if not os.path.exists(scf): continue
    z = np.load(scf); yte = z["yte"]
    names = [n for n in r["ap"] if f"val__{n}" in z.files and f"uni__{n}" in z.files and n in r["a1"] and n in r["a4"]]
    if len(names) < 6: continue
    ap = {n: r["ap"][n] for n in names}; a1 = r["a1"]; a4 = r["a4"]
    Vm = np.column_stack([z[f"val__{n}"].astype(float) for n in names])       # n_val x n_det
    emv, mvv = {}, {}
    for i, n in enumerate(names):
        sv = z[f"val__{n}"].astype(float); su = z[f"uni__{n}"].astype(float)
        su = np.nan_to_num(su, nan=float(np.nanmedian(su[np.isfinite(su)])) if np.isfinite(su).any() else 0.0)
        emv[n] = _em_auc(t, 1.0, -su, -sv, NGEN); mvv[n] = _mv_auc(alpha, 1.0, -su, -sv, NGEN)
    cons = dict(zip(names, consensus_scores(Vm))); mc = dict(zip(names, model_centrality_scores(Vm))); ht = dict(zip(names, hits_authority_scores(Vm)))
    av = pd.Series(ap); best = av.max()
    loc = [av[v] for v in names if fam(v) == "local"]; glo = [av[v] for v in names if fam(v) == "global"]
    tf = "local" if (loc and (not glo or max(loc) > max(glo))) else "global"
    p1 = max(names, key=lambda v: a1[v]); pe = max(names, key=lambda v: a4[v]); lev = max(a1[v] for v in names) - max(a4[v] for v in names)
    rec = {"key": key, "truefam": tf, "cs": corr.get(key, np.nan), "reg_p1": best - av[p1], "reg_pe": best - av[pe], "lev": lev,
           "reg_em": best - av[max(names, key=lambda v: emv[v])], "reg_mv": best - av[min(names, key=lambda v: mvv[v])],
           "reg_consensus": best - av[max(names, key=lambda v: cons[v])], "reg_mc": best - av[max(names, key=lambda v: mc[v])],
           "reg_hits": best - av[max(names, key=lambda v: ht[v])], "reg_beta1": best - av[p1], "reg_random": best - av.mean(),
           "reg_matched": best - av[p1 if lev > TAU else pe]}
    rows.append(rec)
df = pd.DataFrame(rows); med = df.cs.median()
df["reg_corr_str"] = np.where(df.cs > med, df.reg_p1, df.reg_pe)
df["reg_combined"] = np.where((df.cs > med) | (df.lev > TAU), df.reg_p1, df.reg_pe)   # best router (union of corr_str & local_ev)
df.to_csv(os.path.join(D, "STREAM_FINAL_LEADERBOARD.csv"), index=False)
def macro(c): return np.mean([df[df.truefam == fa][c].mean() for fa in ["local", "global"] if (df.truefam == fa).any()])
mv, e = df.reg_mv.values, df.reg_em.values
print(f"=== FINAL leaderboard: full pool (classical+deep), all selectors ({len(df)} datasets: {int((df.truefam=='local').sum())}L/{int((df.truefam=='global').sum())}G) ===")
print(f"  {'selector':24s} {'micro':>6s} {'macro':>6s} {'median':>7s} {'vsMV':>7s} {'vsEM':>7s}")
board = [("NoMaS combined (ours)", "reg_combined"), ("NoMaS corr_str (ours)", "reg_corr_str"), ("NoMaS matched (ours)", "reg_matched"), ("NoMaS beta=1 (ours)", "reg_beta1"),
         ("MV", "reg_mv"), ("EM", "reg_em"), ("consensus", "reg_consensus"), ("ModelCentrality", "reg_mc"), ("HITS", "reg_hits"), ("random", "reg_random")]
pv = {}
for nm, c in board:
    a = df[c].values
    pm = wilcoxon(a, mv).pvalue if c != "reg_mv" and (a != mv).any() else np.nan
    pe_ = wilcoxon(a, e).pvalue if c != "reg_em" and (a != e).any() else np.nan
    pv[nm] = (pm, pe_); print(f"  {nm:24s} {a.mean():6.3f} {macro(c):6.3f} {np.median(a):7.3f} {pm:7.3f} {pe_:7.3f}")
print("\nsaved streamline/STREAM_FINAL_LEADERBOARD.csv")
