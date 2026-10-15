"""Test a STRUCTURE-GATED router: use dependency-probe separability (SPARC's own max a1, a label-free
normal-structure signal) to decide when to trust the dependency probe vs a marginal probe (Goswami spike).
Report: SPARC alone (0.113), the ORACLE router upper bound (per task take the better of the two probes'
picks -- the max any router could achieve), and the structure-gated router at a principled threshold."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
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
keys = [k for k in keep.index if k in clfix and k in clstd and k in aucs]
def dets(k):
    ap=clstd[k]["ap"]; a1=clfix[k]["a1"]
    return [d for d in ap if not DEEP(d) and d in a1 and d in aucs[k] and "spike" in aucs[k][d]]
rec = []
for k in keys:
    ds=dets(k)
    if len(ds)<5: continue
    ap=clstd[k]["ap"]; o=oracle[k]
    dep_pick = max(ds, key=lambda d: clfix[k]["a1"][d]); r_dep = o-ap[dep_pick]
    sp_pick  = max(ds, key=lambda d: aucs[k][d]["spike"]); r_sp = o-ap[sp_pick]
    dep_sep  = max(clfix[k]["a1"][d] for d in ds)   # SPARC's own probe separability
    rec.append(dict(key=k, r_dep=r_dep, r_sp=r_sp, dep_sep=dep_sep))
df = pd.DataFrame(rec)
print(f"tasks: {len(df)}")
print(f"SPARC dependency alone : {df.r_dep.mean():.4f}  (published 0.113)")
print(f"spike (marginal) alone : {df.r_sp.mean():.4f}")
print(f"ORACLE router (upper bound, per-task min of the two): {np.minimum(df.r_dep,df.r_sp).mean():.4f}")
print(f"  -> headroom of PERFECT dep-vs-spike routing = {df.r_dep.mean()-np.minimum(df.r_dep,df.r_sp).mean():.4f}")
print("\nstructure-gated router: use spike when dep_sep < threshold (weak joint structure), else dependency")
for thr in [0.60, 0.70, 0.75, 0.80, df.dep_sep.median()]:
    use_sp = df.dep_sep < thr
    r = np.where(use_sp, df.r_sp, df.r_dep)
    p = wilcoxon(r, df.r_dep).pvalue if (r!=df.r_dep.values).any() else float("nan")
    print(f"  thr={thr:.3f}: regret={r.mean():.4f}  delta vs SPARC={r.mean()-df.r_dep.mean():+.4f}  p={p:.3f}  (routed to spike on {use_sp.sum()} tasks)")
