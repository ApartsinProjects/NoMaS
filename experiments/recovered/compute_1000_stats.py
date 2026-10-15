# -*- coding: utf-8 -*-
"""p-values and invariants at the 1000-point probe, all on ONE construct-matched basis:
classical ap=deepres_v2_clstd, deep ap=probefix, oracle=STREAM_V3, SPARC grade=probe1000 a1@1000,
Goswami=Borda over 5 injection types (goswami_all_post logic). Paired Wilcoxon over the shared key set.
INVARIANTS: SPARC-cls@1000~0.109, SPARC-full@1000~0.138, Goswami-full~0.151, fixed(LOF_k10)~0.130."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
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
clstd = loaddir(os.path.join(ST, "deepres_v2_clstd")); p1000 = loaddir(os.path.join(ST, "probe1000_results"))
pf = {}
for f in glob.glob(os.path.join(PF, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("ap"): pf[r["key"]] = r
aucs = {}
for f in glob.glob(os.path.join(S, "goswami2", "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("aucs"): aucs[r["key"]] = r["aucs"]
lb = pd.read_csv(os.path.join(ST, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict(); corpus = keep.corpus.to_dict()
TYPES = ["spike", "noise", "scale", "contextual", "cutoff"]
def ap_of(k, d): return (pf[k]["ap"].get(d) if DEEP(d) else clstd[k]["ap"].get(d))
def borda_pick(k, dets):
    names = [d for d in dets if d in aucs[k] and all(t in aucs[k][d] for t in TYPES)]
    if len(names) < 6: return None
    b = {d: 0.0 for d in names}
    for t in TYPES:
        for rk, d in enumerate(sorted(names, key=lambda x: aucs[k][x][t])): b[d] += rk
    return max(names, key=lambda d: b[d])
keys = [k for k in keep.index if k in clstd and k in pf and k in p1000 and k in aucs]
print(f"matched keys: {len(keys)}")
rows = []
for k in keys:
    o = oracle[k]; a1 = p1000[k]["a1"]
    cls_g = {d: v for d, v in a1.items() if not DEEP(d) and ap_of(k, d) is not None}
    all_g = {d: v for d, v in a1.items() if ap_of(k, d) is not None}
    if not cls_g or not all_g: continue
    if "LOF_k10" not in clstd[k]["ap"]: continue
    alld = [d for d in aucs[k] if ap_of(k, d) is not None]
    pa = borda_pick(k, alld)
    if pa is None: continue
    rows.append(dict(key=k, corpus=corpus.get(k, "?"),
                     sparc_cls=o - ap_of(k, max(cls_g, key=cls_g.get)),
                     sparc_full=o - ap_of(k, max(all_g, key=all_g.get)),
                     fixed=o - clstd[k]["ap"]["LOF_k10"],
                     gos_full=o - ap_of(k, pa)))
df = pd.DataFrame(rows)
print(f"paired tasks: {len(df)}\n")
print("=== invariants (means) ===")
print(f"  SPARC classical@1000 = {df.sparc_cls.mean():.4f}  (~0.109)")
print(f"  SPARC full@1000      = {df.sparc_full.mean():.4f}  (~0.138)")
print(f"  Goswami full         = {df.gos_full.mean():.4f}  (~0.151)")
print(f"  best fixed (LOF_k10) = {df.fixed.mean():.4f}  (~0.130)")
print("\n=== p-values (paired Wilcoxon) ===")
print(f"  SPARC-cls@1000 vs fixed      : delta={df.sparc_cls.mean()-df.fixed.mean():+.4f}  p={wilcoxon(df.sparc_cls, df.fixed).pvalue:.4f}")
print(f"  SPARC-full@1000 vs Goswami-full: delta={df.sparc_full.mean()-df.gos_full.mean():+.4f}  p={wilcoxon(df.sparc_full, df.gos_full).pvalue:.4f}")
od = df[df.corpus == "oddbench"]
print(f"  OddBench SPARC-cls@1000 vs fixed (n={len(od)}): {od.sparc_cls.mean():.4f} vs {od.fixed.mean():.4f}  p={wilcoxon(od.sparc_cls, od.fixed).pvalue:.4f}")
