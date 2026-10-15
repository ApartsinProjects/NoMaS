# -*- coding: utf-8 -*-
"""Tier-2 statistics for the review response: per-task classical-pool regret for every method, then
medians + upper-tail, win/tie/loss vs fixed, SPARC-vs-Goswami paired test + CI, EM/MV-vs-fixed and
agreement-vs-random tests, deep-pool expansion tests per method, and the Table 8 confusion matrix.
Sources: probe1000 a1 (SPARC), goswami2 borda (Goswami), clstd/probefix ap, RECONCILE_UOMS_V2.csv
(EM/MV/consensus/centrality/hits per task, classical `_cls` and full `_all`), STREAM_V3 oracle."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
ST = os.path.join(S, "streamline"); PF = os.path.join(S, "probefix"); DEEP = lambda d: d.startswith("D_")
def ld(d):
    J={}
    for fn in glob.glob(os.path.join(d,"shard_*.jsonl")):
        for ln in open(fn):
            try: r=json.loads(ln)
            except: continue
            if "key" in r: J[r["key"]]=r
    return J
clstd = ld(os.path.join(ST,"deepres_v2_clstd")); p1000 = ld(os.path.join(ST,"probe1000_results"))
pf={}
for f in glob.glob(os.path.join(PF,"*.jsonl")):
    for ln in open(f):
        try: r=json.loads(ln)
        except: continue
        if r.get("ap"): pf[r["key"]]=r
aucs={}
for f in glob.glob(os.path.join(S,"goswami2","*.jsonl")):
    for ln in open(f):
        try: r=json.loads(ln)
        except: continue
        if r.get("aucs"): aucs[r["key"]]=r["aucs"]
rec = pd.read_csv(os.path.join(S,"RECONCILE_UOMS_V2.csv")).set_index("key")
lb = pd.read_csv(os.path.join(ST,"STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap>0.05)&(lb.reg_random>0.02)&(lb.n_h>=100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
TYPES=["spike","noise","scale","contextual","cutoff"]
def ap_of(k,d): return pf[k]["ap"].get(d) if DEEP(d) else clstd[k]["ap"].get(d)
def borda(k,dets):
    names=[d for d in dets if d in aucs.get(k,{}) and all(t in aucs[k][d] for t in TYPES)]
    if len(names)<6: return None
    b={d:0.0 for d in names}
    for t in TYPES:
        for rk,d in enumerate(sorted(names,key=lambda x:aucs[k][x][t])): b[d]+=rk
    return max(names,key=lambda d:b[d])
def fam(d):
    if DEEP(d): return "deep"
    if d.startswith(("LOF","KNN","CBLOF")): return "neighbor"
    if d.startswith(("HBOS","ECOD","COPOD")): return "distribution"
    if d.startswith(("PCA","LODA")): return "projection"
    return "isolation"
keys=[k for k in keep.index if k in clstd and k in pf and k in p1000 and k in aucs and k in rec.index and "LOF_k10" in clstd[k]["ap"]]
print(f"paired tasks: {len(keys)}")
R={}  # per-task classical regret vectors
def sparc(k,pool):
    g={d:v for d,v in p1000[k]["a1"].items() if pool(d) and ap_of(k,d) is not None}
    return oracle[k]-ap_of(k,max(g,key=g.get)) if g else np.nan
cls=lambda d: not DEEP(d); allp=lambda d: True
R["SPARC"]=np.array([sparc(k,cls) for k in keys]); R["SPARC_full"]=np.array([sparc(k,allp) for k in keys])
R["Goswami"]=np.array([oracle[k]-ap_of(k,borda(k,[d for d in aucs[k] if not DEEP(d) and ap_of(k,d) is not None])) for k in keys])
R["Goswami_full"]=np.array([oracle[k]-ap_of(k,borda(k,[d for d in aucs[k] if ap_of(k,d) is not None])) for k in keys])
R["fixed"]=np.array([oracle[k]-clstd[k]["ap"]["LOF_k10"] for k in keys])
R["random"]=np.array([oracle[k]-np.mean([v for d,v in clstd[k]["ap"].items() if not DEEP(d)]) for k in keys])
for m,c in [("EM","EM"),("MV","MV"),("consensus","consensus"),("HITS","hits"),("MC","centrality")]:
    R[m]=rec.loc[keys,f"{c}_cls"].values; R[m+"_full"]=rec.loc[keys,f"{c}_all"].values

print("\n=== classical-pool regret: mean / median / 90th-pct ===")
for m in ["SPARC","Goswami","fixed","EM","MV","consensus","HITS","MC","random"]:
    v=R[m]; print(f"  {m:9s} mean {np.mean(v):.3f}  median {np.median(v):.3f}  p90 {np.percentile(v,90):.3f}")
print("\n=== SPARC vs fixed: win/tie/loss (classical) ===")
d=R["SPARC"]-R["fixed"]; print(f"  win {int((d<-1e-9).sum())}  tie {int((abs(d)<=1e-9).sum())}  loss {int((d>1e-9).sum())}  (win=SPARC lower regret)")
def boot_ci(x,n=10000,seed=0):
    rng=np.random.default_rng(seed); bs=[np.mean(rng.choice(x,len(x),replace=True)) for _ in range(n)]
    return np.percentile(bs,2.5),np.percentile(bs,97.5)
print("\n=== paired tests (classical) ===")
lo,hi=boot_ci(R["SPARC"]-R["Goswami"]); print(f"  SPARC vs Goswami: delta={np.mean(R['SPARC']-R['Goswami']):+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] p={wilcoxon(R['SPARC'],R['Goswami']).pvalue:.3f}")
print(f"  EM vs fixed:      delta={np.mean(R['EM']-R['fixed']):+.4f} p={wilcoxon(R['EM'],R['fixed']).pvalue:.4f}")
print(f"  MV vs fixed:      delta={np.mean(R['MV']-R['fixed']):+.4f} p={wilcoxon(R['MV'],R['fixed']).pvalue:.4f}")
for m in ["consensus","HITS","MC"]:
    print(f"  {m} vs random:  delta={np.mean(R[m]-R['random']):+.4f} p={wilcoxon(R[m],R['random']).pvalue:.4f}")
lo,hi=boot_ci(R["fixed"]-R["SPARC"]); print(f"  SPARC vs fixed:   delta={np.mean(R['SPARC']-R['fixed']):+.4f} 95%CI[{-hi:+.4f},{-lo:+.4f}] p={wilcoxon(R['SPARC'],R['fixed']).pvalue:.4f}")
print("\n=== deep-pool expansion (classical -> full), paired p per method ===")
for m in ["SPARC","Goswami","EM","MV","consensus","HITS","MC"]:
    a,b=R[m],R[m+"_full"]; p=wilcoxon(b,a).pvalue if (a!=b).any() else float("nan")
    print(f"  {m:9s} {np.mean(a):.3f} -> {np.mean(b):.3f}  delta={np.mean(b-a):+.4f} p={p:.4f}")
print("\n=== Table 8 confusion matrix: probe-family (rows) x oracle-family (cols), all 32 ===")
FAMS=["neighbor","distribution","projection","isolation","deep"]
M={pf_:{of_:0 for of_ in FAMS} for pf_ in FAMS}
for k in keys:
    a1={d:v for d,v in p1000[k]["a1"].items() if ap_of(k,d) is not None}
    ap={d:ap_of(k,d) for d in a1}
    pfam=fam(max(a1,key=a1.get)); ofam=fam(max(ap,key=ap.get)); M[pfam][ofam]+=1
print("  probe\\oracle  "+" ".join(f"{o[:5]:>6}" for o in FAMS))
for p_ in FAMS: print(f"  {p_:11s} "+" ".join(f"{M[p_][o]:>6d}" for o in FAMS))
# neighbor-when-neighbor-oracle accuracy + cost of wrong neighbor picks
nn=M["neighbor"]["neighbor"]; probe_neighbor=sum(M["neighbor"].values()); oracle_neighbor=sum(M[f]["neighbor"] for f in FAMS)
print(f"\n  probe picks neighbor on {probe_neighbor} tasks; of those {nn} are neighbor-oracle (accuracy {nn/probe_neighbor:.0%})")
print(f"  neighbor is oracle on {oracle_neighbor} tasks; probe catches {nn} of them (recall {nn/oracle_neighbor:.0%})")
