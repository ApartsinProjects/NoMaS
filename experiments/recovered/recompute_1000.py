# -*- coding: utf-8 -*-
"""Recompute all SPARC numbers at the 1000-point probe. Isolates the probe-size effect: keeps the
PUBLISHED ap (classical=clstd, deep=probefix) and STREAM_V3 oracle FIXED, swaps ONLY the probe grade
a1 -> a1@1000 (from the modal_probe1000 run). Outputs Table 6 (SPARC 3 pools), Table 6b (by source),
Table 8 (family), and the abstract/section-6 headline, with invariant checks.
INVARIANT: SPARC classical@1000 ~ 0.109 (matches local sparc_a1_1000.json) and <= the 200-point 0.113."""
import json, glob, os, numpy as np, pandas as pd
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
ST = os.path.join(S, "streamline"); PF = os.path.join(S, "probefix")
DEEP = lambda d: d.startswith("D_")
def loaddir(d):
    J = {}
    for fn in glob.glob(os.path.join(d, "shard_*.jsonl")):
        for ln in open(fn):
            try: r = json.loads(ln)
            except Exception: continue
            if "key" in r: J[r["key"]] = r
    return J
# published ap basis
clstd = loaddir(os.path.join(ST, "deepres_v2_clstd"))
pf = {}
for f in glob.glob(os.path.join(PF, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("ap"): pf[r["key"]] = r
# 1000-point grades
p1000 = loaddir(os.path.join(ST, "probe1000_results"))
lb = pd.read_csv(os.path.join(ST, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict(); corpus = keep.corpus.to_dict()
keys = [k for k in keep.index if k in clstd and k in pf and k in p1000]
print(f"tasks with 1000-point grades: {len(keys)} / {len(keep)}")

def ap_of(k, d): return (pf[k]["ap"].get(d) if DEEP(d) else clstd[k]["ap"].get(d))
def a1(k): return p1000[k]["a1"]
def fam(d):
    if DEEP(d): return "deep"
    if d.startswith(("LOF", "KNN", "CBLOF")): return "neighbor"
    if d.startswith(("HBOS", "ECOD", "COPOD")): return "distribution"
    if d.startswith(("PCA", "LODA")): return "projection"
    if d.startswith("IForest"): return "isolation"
    return "other"

def sparc_regret(pool_filter):
    R = []
    for k in keys:
        g = {d: v for d, v in a1(k).items() if pool_filter(d) and ap_of(k, d) is not None}
        if not g: continue
        sel = max(g, key=g.get); R.append(oracle[k] - ap_of(k, sel))
    return np.mean(R), len(R)

cls_only = lambda d: not DEEP(d)
deep_only = lambda d: DEEP(d)
allp = lambda d: True
r_cls, n = sparc_regret(cls_only); r_deep, _ = sparc_regret(deep_only); r_all, _ = sparc_regret(allp)
print("\n=== Table 6 SPARC row @1000 (published ap/oracle fixed) ===")
print(f"  classical(25): {r_cls:.4f}   deep(7): {r_deep:.4f}   full(32): {r_all:.4f}")
print(f"  INVARIANT classical@1000 ~0.109 and <= 0.113 : {'OK' if r_cls <= 0.115 else 'CHECK'}")
# cross-check vs local sparc_a1_1000.json
try:
    loc = json.load(open(os.path.join(S, "sparc_a1_1000.json")))
    R = []
    for k in keys:
        if k not in loc: continue
        g = {d: v for d, v in loc[k].items() if not DEEP(d) and clstd[k]["ap"].get(d) is not None}
        if g: sel = max(g, key=g.get); R.append(oracle[k] - clstd[k]["ap"][sel])
    print(f"  cross-check local sparc_a1_1000.json classical = {np.mean(R):.4f} (n={len(R)})")
except Exception as e: print("  (local cross-check skipped:", e, ")")

print("\n=== Table 6b SPARC by source @1000 (classical pool) ===")
disp = {"ovrbench": "OvrBench", "oddbench": "OddBench", "tsbad_m": "TSB-AD/TS"}
by = {}
for k in keys:
    g = {d: v for d, v in a1(k).items() if not DEEP(d) and ap_of(k, d) is not None}
    if not g: continue
    reg = oracle[k] - ap_of(k, max(g, key=g.get)); by.setdefault(corpus.get(k, "other"), []).append(reg)
for c, name in disp.items():
    if c in by: print(f"  {name:10s}({len(by[c])}): {np.mean(by[c]):.4f}")
allreg = [x for v in by.values() for x in v]; print(f"  All({len(allreg)}): {np.mean(allreg):.4f}")

print("\n=== Table 8 family @1000 (probe #1 vs oracle over all 32) ===")
FAMS = ["neighbor", "distribution", "projection", "isolation", "deep"]
probe_top = {f: 0 for f in FAMS}; real_win = {f: 0 for f in FAMS}
for k in keys:
    g = {d: v for d, v in a1(k).items() if ap_of(k, d) is not None}
    aps = {d: ap_of(k, d) for d in g}
    if not g: continue
    probe_top[fam(max(g, key=g.get))] += 1; real_win[fam(max(aps, key=aps.get))] += 1
for f in FAMS: print(f"  {f:13s} probe#1={probe_top[f]:3d}  oracle={real_win[f]:3d}")
