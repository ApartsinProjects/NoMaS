# -*- coding: utf-8 -*-
"""Operational (probe-based) detector family, replacing the hand-picked LOCAL/GLOBAL name list.
Each detector's family = local if it separates the dependency probe (a1) better than the marginal
probe (a4), averaged over datasets; global otherwise. Covers the WHOLE pool (iForest, LODA, deep, ...).
Per-dataset truefam = family of the oracle-best (max ap) detector. Reuses the existing leaderboard CSV's
per-dataset reg_* (unchanged by family); only recomputes truefam + macro. Updates STREAM_FINAL_LEADERBOARD
+ STREAM_MINE fam, writes the detector->family table, and prints all new numbers."""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); SRC = os.path.join(D, "deepres3")
# 1) collect a1/a4/ap per detector per dataset
A1, A4 = {}, {}; best_det = {}
for fn in glob.glob(os.path.join(SRC, "shard_*.jsonl")):
    for ln in open(fn):
        ln = ln.strip()
        if not ln: continue
        r = json.loads(ln)
        if "ap" not in r or len(r.get("ap", {})) < 6: continue
        key = r["key"]; a1 = r.get("a1", {}); a4 = r.get("a4", {}); ap = r["ap"]
        names = [n for n in ap if n in a1 and n in a4]
        if len(names) < 6: continue
        for n in names:
            A1.setdefault(n, []).append(float(a1[n])); A4.setdefault(n, []).append(float(a4[n]))
        best_det[key] = max(names, key=lambda v: ap[v])
# 2) operational per-detector family: localness score = mean(a1[dependency] - a4[marginal]); the marginal
# probe is uniformly easier (all deltas < 0), so split on delta RELATIVE to the pool median (a detector is
# "local" if it responds to dependency structure MORE than the typical pool member does).
dets = sorted(A1)
delta = {n: np.mean(A1[n]) - np.mean(A4[n]) for n in dets}
THR = float(np.median(list(delta.values())))
detfam = {n: ("local" if delta[n] > THR else "global") for n in dets}
print(f"[localness split at pool-median delta = {THR:+.3f}]")
ftab = pd.DataFrame([{"detector": n, "mean_a1_dep": np.mean(A1[n]), "mean_a4_marg": np.mean(A4[n]),
                      "delta": np.mean(A1[n]) - np.mean(A4[n]), "family": detfam[n], "n": len(A1[n])} for n in dets]).sort_values("delta", ascending=False)
ftab.to_csv(os.path.join(D, "STREAM_DETFAM.csv"), index=False)
print("=== operational detector->family (by mean a1[dependency] - a4[marginal]) ===")
for _, r in ftab.iterrows(): print(f"  {r.detector:22s} a1={r.mean_a1_dep:.3f} a4={r.mean_a4_marg:.3f} delta={r.delta:+.3f} -> {r.family}")
# validate vs old hand list
OLD_L = ("LOF", "KNN", "CBLOF"); OLD_G = ("HBOS", "COPOD", "ECOD", "PCA")
def oldfam(n):
    return "local" if n.startswith(OLD_L) else ("global" if n.startswith(OLD_G) else "other")
agree = [(n, detfam[n], oldfam(n)) for n in dets if oldfam(n) != "other"]
match = sum(1 for _, a, b in agree if a == b)
print(f"\nagreement with old hand-list on its 7 canonical detectors: {match}/{len(agree)}")
for n, a, b in agree:
    if a != b: print(f"  FLIP {n}: operational={a} handlist={b}")
# 3) recompute truefam per dataset from existing leaderboard CSV
lb = pd.read_csv(os.path.join(D, "STREAM_FINAL_LEADERBOARD.csv"))
lb["truefam_old"] = lb["truefam"]
lb["truefam"] = lb["key"].map(lambda k: detfam.get(best_det.get(k, ""), np.nan))
miss = lb.truefam.isna().sum()
if miss: print(f"WARN {miss} datasets without operational truefam"); lb = lb.dropna(subset=["truefam"])
lb.to_csv(os.path.join(D, "STREAM_FINAL_LEADERBOARD.csv"), index=False)
nL = int((lb.truefam == "local").sum()); nG = int((lb.truefam == "global").sum())
print(f"\n=== new family counts: {nL} local / {nG} global (was 104/69); flipped {(lb.truefam!=lb.truefam_old).sum()} datasets ===")
# 4) macro leaderboard with new truefam
def macro(c): return np.mean([lb[lb.truefam == fa][c].mean() for fa in ["local", "global"]])
board = [("SPARC-combined", "reg_combined"), ("SPARC-corr", "reg_corr_str"), ("SPARC-beta1", "reg_beta1"), ("SPARC-matched", "reg_matched"),
         ("EM", "reg_em"), ("MV", "reg_mv"), ("consensus", "reg_consensus"), ("ModelCentrality", "reg_mc"), ("HITS", "reg_hits"), ("random", "reg_random")]
a_comb = lb["reg_combined"].values
print(f"\n{'selector':16s}{'micro':>8s}{'macro':>8s}{'p_vs_comb':>11s}")
for nm, c in board:
    p = wilcoxon(a_comb, lb[c].values).pvalue if c != "reg_combined" else float("nan")
    print(f"{nm:16s}{lb[c].mean():8.3f}{macro(c):8.3f}{p:11.4f}")
# 5) update STREAM_MINE fam to operational truefam (for the figures)
mine = pd.read_csv(os.path.join(D, "STREAM_MINE.csv")); mine["key"] = mine.corpus + "__" + mine.dataset
tfmap = dict(zip(lb.key, lb.truefam)); mine["fam_old"] = mine["fam"]
mine["fam"] = mine["key"].map(tfmap).fillna(mine["fam"])
mine.drop(columns=["key"]).to_csv(os.path.join(D, "STREAM_MINE.csv"), index=False)
# 6) dose-response with new truefam: anisotropy index -> local-win fraction
d3 = mine.dropna(subset=["top1_var", "corr_str", "eigengap"]).copy()
z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
d3["idx"] = z(d3.top1_var) + z(d3.corr_str) + z(d3.eigengap)
yb = (d3.fam == "local").astype(int)
AUC = max(roc_auc_score(yb, d3.idx), 1 - roc_auc_score(yb, d3.idx))
d3["q"] = pd.qcut(d3.idx, 5, labels=False)
qs = [(q, (d3[d3.q == q].fam == "local").mean(), int((d3.q == q).sum())) for q in range(5)]
print(f"\n=== dose-response (new fam): anisotropy index AUC={AUC:.3f}, base={ (mine.fam=='local').mean():.3f} ===")
print("  quintile local-win:", [(q, round(p, 2), n) for q, p, n in qs])
print("\nsaved STREAM_DETFAM.csv, updated STREAM_FINAL_LEADERBOARD.csv + STREAM_MINE.csv")
