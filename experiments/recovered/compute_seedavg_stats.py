# -*- coding: utf-8 -*-
"""From SEED_AVG_CLASSICAL.pkl, produce every classical-pool number the paper needs at the TEN-SEED MEAN:
Table 6 classical SPARC, Table 6b per-source (observed), Table 7 rows 1-2 (observed & bininterp All),
Figure 7 per-source SPARC + per-source paired p-values, and the cluster bootstrap over source collections.
All paired tests are SPARC seed-mean per-task regret vs the fixed detector (LOF_k10)."""
import json, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
ST = os.path.join(S, "streamline")
D = pd.read_pickle(os.path.join(S, "SEED_AVG_CLASSICAL.pkl")); reg = D["reg"]; src3 = D["src"]
clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(S, "clsstd_valuni", "ap.jsonl"))}
lb = pd.read_csv(os.path.join(ST, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict(); corpus = keep.corpus.to_dict()

def src5(c):
    c = str(c).lower()
    if "ovr" in c: return "OvrBench"
    if "odd" in c: return "OddBench"
    if "tsb" in c or c == "ts" or "series" in c: return "TS"
    if "adbench" in c or "adb" in c: return "ADBench"
    return "DAMI"

keys = [k for k in reg["observed"] if "LOF_k10" in clsap.get(k, {})]
sparc_sm = {k: float(np.mean(reg["observed"][k])) for k in keys}     # seed-mean per task
binint_sm = {k: float(np.mean(reg["bininterp"][k])) for k in keys}
fixed = {k: oracle[k] - clsap[k]["LOF_k10"] for k in keys}
s5 = {k: src5(corpus.get(k, "")) for k in keys}

def blk(sub, label):
    sp = np.array([sparc_sm[k] for k in sub]); fx = np.array([fixed[k] for k in sub])
    p = wilcoxon(sp, fx).pvalue if len(sub) >= 6 and np.any(sp != fx) else float("nan")
    print(f"  {label:10s} n={len(sub):3d}  SPARC {sp.mean():.4f}  fixed {fx.mean():.4f}  adv {fx.mean()-sp.mean():+.4f}  p={p:.4f}")

print("=== observed-value SPARC (ten-seed mean), classical pool ===")
blk(keys, "All")
for src in ["OvrBench", "OddBench", "TS", "ADBench", "DAMI"]:
    blk([k for k in keys if s5[k] == src], src)
print(f"\n  Table 7 row 1 (observed, All)     = {np.mean([sparc_sm[k] for k in keys]):.4f}")
print(f"  Table 7 row 2 (bininterp, All)    = {np.mean([binint_sm[k] for k in keys]):.4f}")
print(f"  fixed detector (All)              = {np.mean([fixed[k] for k in keys]):.4f}")

# seed spread of the All mean (observed)
per_seed = [np.mean([reg['observed'][k][s] for k in keys]) for s in range(10)]
print(f"  observed All per-seed: min {min(per_seed):.4f} max {max(per_seed):.4f} mean {np.mean(per_seed):.4f} sd {np.std(per_seed):.4f}")

# cluster bootstrap over the 5 source collections, seed-mean basis
srcs = ["OvrBench", "OddBench", "TS", "ADBench", "DAMI"]
by = {s: [k for k in keys if s5[k] == s] for s in srcs}
rng = np.random.default_rng(0); B = 5000; advs = []
for _ in range(B):
    chosen = rng.choice(len(srcs), len(srcs), replace=True)
    pool = [k for ci in chosen for k in by[srcs[ci]]]
    if not pool: continue
    a = np.mean([fixed[k] - sparc_sm[k] for k in pool]); advs.append(a)
advs = np.array(advs)
print(f"\n  cluster bootstrap (seed-mean): adv {advs.mean():+.4f}  95% CI [{np.percentile(advs,2.5):.4f}, {np.percentile(advs,97.5):.4f}]  P(>0)={np.mean(advs>0):.3f}")
