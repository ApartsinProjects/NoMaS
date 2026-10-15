# -*- coding: utf-8 -*-
"""Corrected leaderboard (v4): SPARC = always-dependency probe (no routing), argmax over classical
detectors of the FIXED a1 grade (deepres_v2_clfix), with ap for regret taken from the same classical
table the oracle uses (deepres_v2_clstd) for consistency. EM/MV/consensus/MC/HITS/random reused from
STREAM_V3_LEADERBOARD (probe-independent). Goswami from GOSWAMI_INJ. Adds the best-single-detector
constant baseline. Reports overall + per modality with paired Wilcoxon vs constant and vs entropy."""
import os, json, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
D = os.path.dirname(os.path.abspath(__file__))


def load(dirn, want=("ap",)):
    J = {}
    for fn in glob.glob(os.path.join(dirn, "shard_*.jsonl")):
        for ln in open(fn):
            try:
                r = json.loads(ln)
                if "ap" in r: J[r["key"]] = r
            except Exception: pass
    return J


clstd = load(os.path.join(D, "deepres_v2_clstd"))
clfix = load(os.path.join(D, "deepres_v2_clfix"))
lb = pd.read_csv(os.path.join(D, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].copy().set_index("key")
oracle = keep.oracle_ap.to_dict()
gos = {}
for ln in open(os.path.join(D, "GOSWAMI_INJ.jsonl")):
    r = json.loads(ln)
    if r.get("sel"): gos[r["key"]] = r["sel"]

keys = [k for k in keep.index if k in clstd and k in clfix and k in gos]
# sanity: fixed-probe re-fit reproduces ap (probe change must NOT affect ap)
apdiff = [abs(clstd[k]["ap"][d] - clfix[k]["ap"][d]) for k in keys for d in clstd[k]["ap"] if d in clfix[k].get("ap", {})]
print(f"ap reproduction clstd vs clfix: max|diff|={max(apdiff):.2e} (should be ~0)")

apc = {k: clstd[k]["ap"] for k in keys}              # ap for regret (same source as oracle)
a1f = {k: clfix[k]["a1"] for k in keys}              # FIXED dependency-probe grade for SELECTION
# SPARC (fixed, no routing) = classical argmax fixed-a1
sparc = {}
for k in keys:
    cls = [d for d in apc[k] if d in a1f[k]]
    sel = max(cls, key=lambda d: a1f[k][d]); sparc[k] = oracle[k] - apc[k][sel]
# best constant detector (classical, min mean regret)
allc = set.intersection(*[set(apc[k]) for k in keys])
const_det = min(allc, key=lambda d: np.mean([oracle[k] - apc[k][d] for k in keys]))
def regvec(f): return np.array([f(k) for k in keys])
M = {
    "SPARC (fixed, always-dep)": regvec(lambda k: sparc[k]),
    "Goswami injection": regvec(lambda k: oracle[k] - apc[k][gos[k]]),
    f"Best single detector ({const_det})": regvec(lambda k: oracle[k] - apc[k][const_det]),
    "Entropy": keep.loc[keys, "reg_em"].values,
    "Mass-volume": keep.loc[keys, "reg_mv"].values,
    "Consensus": keep.loc[keys, "reg_consensus"].values,
    "Random": keep.loc[keys, "reg_random"].values,
}
CONST = M[f"Best single detector ({const_det})"]; EM = M["Entropy"]
corp = keep.loc[keys, "corpus"].values
out = pd.DataFrame({k: v for k, v in M.items()}); out.insert(0, "corpus", corp); out.insert(0, "key", keys)
out.to_csv(os.path.join(D, "STREAM_V4_LEADERBOARD.csv"), index=False)


def board(mask, lab):
    print(f"\n=== {lab}: n={mask.sum()} ===")
    for nm, v in sorted(M.items(), key=lambda x: x[1][mask].mean()):
        pc = wilcoxon(v[mask], CONST[mask]).pvalue if not np.allclose(v[mask], CONST[mask]) else np.nan
        pe = wilcoxon(v[mask], EM[mask]).pvalue if not np.allclose(v[mask], EM[mask]) else np.nan
        print(f"  {nm:32s} {v[mask].mean():.3f}   vs const p={pc:.3f}  vs EM p={pe:.3f}")


allm = np.ones(len(keys), bool)
board(allm, "ALL")
for cp in ["tsbad_m", "oddbench", "ovrbench"]:
    board(corp == cp, cp)
print("\nsaved STREAM_V4_LEADERBOARD.csv")
