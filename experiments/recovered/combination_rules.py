"""Evaluate alternative label-free combination rules over probes (SPARC dependency + Goswami types),
to see if any beats the single dependency probe (0.113) despite the routing problem. Rules use ONLY the
per-detector probe grades (no real labels): Borda sum, worst-case/min-rank, rank-product, discrimination-
weighted Borda, per-task confidence routing (use the probe with the strongest/most-peaked signal).
Classical pool; regret vs STREAM_V3 oracle, ap=clstd."""
import json, glob, os, numpy as np, pandas as pd
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
    ap=clstd[k]["ap"]; a1=clfix[k]["a1"]
    return [d for d in ap if not DEEP(d) and d in a1 and d in aucs[k] and all(t in aucs[k][d] for t in TYPES)]
def grades(k, ds, probeset):
    """return dict probe-> np.array of that probe's grade over ds."""
    G = {}
    if "dep" in probeset: G["dep"] = np.array([clfix[k]["a1"][d] for d in ds])
    for t in TYPES:
        if t in probeset: G[t] = np.array([aucs[k][d][t] for d in ds])
    return G
def eval_rule(rule, probeset):
    R=[]
    for k in keys:
        ds=dets(k)
        if len(ds)<5: continue
        G=grades(k,ds,probeset); pick = rule(G, ds)
        R.append(oracle[k]-clstd[k]["ap"][ds[pick]])
    return np.array(R)
# --- rules (return index into ds) ---
def borda(G,ds):   return int(np.argmax(sum(rankdata(v) for v in G.values())))
def minrank(G,ds): return int(np.argmax(np.min([rankdata(v) for v in G.values()],axis=0)))   # worst-case best
def rankprod(G,ds):return int(np.argmax(np.prod([rankdata(v) for v in G.values()],axis=0)))
def discw(G,ds):   # weight each probe by its grade spread (discrimination) on this task
    w={p:np.std(v) for p,v in G.items()}; tot=sum(w.values())+1e-9
    return int(np.argmax(sum((w[p]/tot)*rankdata(v) for p,v in G.items())))
def conf_route(G,ds):  # use only the probe whose top grade is highest (most confident)
    p=max(G,key=lambda p:np.max(G[p])); return int(np.argmax(G[p]))
def margin_route(G,ds):  # use the probe with the largest top-minus-median margin (most peaked)
    def marg(v): s=np.sort(v); return s[-1]-np.median(v)
    p=max(G,key=lambda p:marg(G[p])); return int(np.argmax(G[p]))
RULES = {"Borda sum":borda, "min-rank (worst-case)":minrank, "rank-product":rankprod,
         "discrimination-weighted":discw, "confidence-route":conf_route, "margin-route":margin_route}
base = eval_rule(lambda G,ds: int(np.argmax(G["dep"])), {"dep"})
print(f"tasks: {len(base)}   SPARC dependency alone = {base.mean():.4f} (published 0.113)\n")
for setname, ps in [("{dep, spike}", {"dep","spike"}), ("{dep, spike, contextual}", {"dep","spike","contextual"}),
                    ("{dep + all 5}", {"dep"}|set(TYPES))]:
    print(f"probe set {setname}:")
    for rn, rule in RULES.items():
        r=eval_rule(rule, ps); n=min(len(r),len(base))
        p=wilcoxon(r[:n],base[:n]).pvalue if (r[:n]!=base[:n]).any() else float("nan")
        star = " *BEATS SPARC*" if r.mean()<base.mean()-1e-9 else ""
        print(f"    {rn:24s}: {r.mean():.4f}  delta={r[:n].mean()-base[:n].mean():+.4f} p={p:.3f}{star}")
    print()
