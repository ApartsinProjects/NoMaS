"""Combine SPARC's dependency probe with Goswami injection types (Borda over per-detector grades, no
re-fit). SPARC grade = clfix a1; Goswami grades = goswami2 per-type AUC. Classical pool; regret vs
STREAM_V3 oracle, ap=clstd. Does SPARC + a marginal type (spike) beat SPARC alone (0.113)?"""
import json, glob, os, itertools, numpy as np, pandas as pd
from scipy.stats import rankdata, wilcoxon
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
DEEP = lambda d: d.startswith("D_")
def ld(d):
    J={}
    for fn in glob.glob(os.path.join(d,"shard_*.jsonl")):
        for ln in open(fn):
            try: r=json.loads(ln)
            except: continue
            if "key" in r: J[r["key"]]=r
    return J
clstd = ld(os.path.join(S,"streamline","deepres_v2_clstd")); clfix = ld(os.path.join(S,"streamline","deepres_v2_clfix"))
aucs = {}
for f in glob.glob(os.path.join(S,"goswami2","*.jsonl")):
    for ln in open(f):
        try: r=json.loads(ln)
        except: continue
        if r.get("aucs"): aucs[r["key"]]=r["aucs"]
lb = pd.read_csv(os.path.join(S,"streamline","STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap>0.05)&(lb.reg_random>0.02)&(lb.n_h>=100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
TYPES = ["spike","noise","scale","contextual","cutoff"]
keys = [k for k in keep.index if k in clfix and k in clstd and k in aucs]
def dets(k):
    ap = clstd[k]["ap"]; a1 = clfix[k]["a1"]
    return [d for d in ap if not DEEP(d) and d in a1 and d in aucs[k] and all(t in aucs[k][d] for t in TYPES)]
def regret(grades):
    """grades: list of per-detector dict-getters; Borda over them."""
    R=[]
    for k in keys:
        ds = dets(k)
        if len(ds) < 5: continue
        tot = np.zeros(len(ds))
        for g in grades: tot = tot + rankdata([g(k, d) for d in ds])
        R.append(oracle[k] - clstd[k]["ap"][ds[int(np.argmax(tot))]])
    return np.array(R)
SPARC = lambda k,d: clfix[k]["a1"][d]
GT = {t: (lambda k,d,t=t: aucs[k][d][t]) for t in TYPES}
combos = {
 "SPARC (dependency) alone": [SPARC],
 "SPARC + spike":            [SPARC, GT["spike"]],
 "SPARC + scale":            [SPARC, GT["scale"]],
 "SPARC + contextual":       [SPARC, GT["contextual"]],
 "SPARC + cutoff":           [SPARC, GT["cutoff"]],
 "SPARC + spike + scale":    [SPARC, GT["spike"], GT["scale"]],
 "SPARC + spike + contextual":[SPARC, GT["spike"], GT["contextual"]],
 "SPARC + all-5 Goswami":    [SPARC]+[GT[t] for t in TYPES],
 "Goswami full-5 (no SPARC)":[GT[t] for t in TYPES],
}
base = regret([SPARC])
print(f"tasks: {len(base)}   (SPARC dependency alone = published 0.113; Goswami = 0.122)")
for name, gr in combos.items():
    r = regret(gr); n = min(len(r), len(base))
    p = wilcoxon(r[:n], base[:n]).pvalue if name!="SPARC (dependency) alone" and (r[:n]!=base[:n]).any() else float("nan")
    print(f"  {name:28s}: {r.mean():.4f}   delta vs SPARC = {r[:n].mean()-base[:n].mean():+.4f}  p={p:.3f}")
